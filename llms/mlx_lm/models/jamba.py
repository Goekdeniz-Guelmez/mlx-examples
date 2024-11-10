from operator import call
from typing import Deque, Union, Optional, Any
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers import Dropout
from transformers.models.hiera.modeling_hiera import Dict

from .base import BaseModelArgs, create_attention_mask

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int

    intermediate_size: int

    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5

    # mamba related
    d_state: int = 16 # N in paper
    expand_factor: int = 2 # N in paper
    d_conv: int = 4
    dt_rank: Union[int, str] = 'auto'

    mamba_d_conv: int = 4
    mamba_d_state: int = 16
    mamba_expand: int = 2
    mamba_dt_rank: int = 256
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False
    inner_layernorms: bool = True

    # attention related
    num_attention_heads: int = 32
    num_key_value_heads: int = 8 # GQA
    attention_dropout: float = 0.0

    # MoE related
    num_experts: int = 16
    num_experts_per_tok: int = 2

    # structure
    attn_layer_offset: int = 4
    attn_layer_period: int = 8
    expert_layer_offset: int = 1
    expert_layer_period: int = 2

    # language modeling
    vocab_size: int = 65536
    tie_word_embeddings: bool = False


class DepthWiseConv1d(nn.Module):
    def __init__(self, channels, kernel_size, bias=True, padding=0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight = mx.random.normal((self.channels, kernel_size, 1))
        self.bias = mx.zeros((channels,)) if bias else None

    def __call__(self, x, cache=None):
        B, L, C = x.shape
        groups, K, _ = self.weight.shape

        if cache is not None:
            x = mx.concatenate([cache, x], axis=1)
        else:
            x = mx.pad(x, [(0, 0), (K - 1, 0), (0, 0)])

        y = mx.conv_general(x, self.weight, groups=groups)

        if self.bias is not None:
            y = y + self.bias

        return y, x[:, -K + 1 :, :]


def repeat_kv(hidden_states: mx.array, n_rep: int) -> mx.array:
    """
    This is the equivalent of repeat_interleave. The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    # In MLX, we use broadcast_to instead of expand
    hidden_states = mx.broadcast_to(
        hidden_states[:, :, None, :, :],
        (batch, num_key_value_heads, n_rep, slen, head_dim)
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.attention_dropout = args.attention_dropout
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.head_dim = args.hidden_size // self.num_heads

        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None,
    ):
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            past_keys, past_values = cache

            if past_keys is not None:
                keys = mx.concat([past_keys, keys], axis=2)
                values = mx.concat([past_values, values], axis=2)

            cache = (keys, values)

        keys = repeat_kv(keys, self.num_key_value_groups)
        values = repeat_kv(values, self.num_key_value_groups)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return self.o_proj(output), cache


class AttentionLayer(nn.Module):
    def __init__(self, args: ModelArgs, num_experts: int):
        super().__init__()

        self.self_attn = Attention(args)

        num_experts_per_tok = args.num_experts_per_tok if num_experts > 1 else 1
        self.moe = SparseMoEBlock(args, num_experts, num_experts_per_tok)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_moe_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None
    ):
        r, cache = self.self_attn(self.input_layernorm(x), cache, mask)
        h = x + r
        x, router_logits = self.moe(self.pre_moe_layernorm(h))
        r = h + x
        return r, cache


class MambaLayer(nn.Module):
    def __init__(self, args: ModelArgs, num_experts: int):
        super().__init__()
        self.args = args

        self.mamba = MambaBlock(args)

        num_experts_per_tok = args.num_experts_per_tok if num_experts > 1 else 1
        self.moe = SparseMoEBlock(args, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_moe_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None
    ):
        # mamba
        x = self.input_layernorm(input)
        if cache is None:
            x = self.mamba(x)
        else:
            x, cache = self.mamba(x, cache)
            x = mx.expand_dims(x, axis=1)
        x = input + x

        # FFN
        residual = x
        x = self.pre_moe_layernorm(x)
        x, router_logits = self.moe(x)
        x = residual + x

        outputs = (x, router_logits)

        return outputs, cache

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.state_size
        self.conv_kernel_size = args.conv_kernel
        self.intermediate_size = args.intermediate_size
        self.time_step_rank = int(args.time_step_rank)
        self.use_conv_bias = args.use_conv_bias

        self.in_proj = nn.Linear(
            self.hidden_size, self.intermediate_size * 2, bias=args.use_bias
        )

        self.conv1d = DepthWiseConv1d(
            channels=self.intermediate_size,
            kernel_size=self.conv_kernel_size,
            bias=self.use_conv_bias,
            padding=self.conv_kernel_size - 1,
        )

        self.x_proj = nn.Linear(
            self.intermediate_size,
            self.time_step_rank + 2 * self.ssm_state_size,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        A = mx.repeat(
            mx.arange(1.0, self.ssm_state_size + 1.0).reshape([1, self.ssm_state_size]),
            repeats=self.intermediate_size,
            axis=0,
        )
        self.A_log = mx.log(A)
        self.D = mx.ones([self.intermediate_size])

        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=args.use_bias
        )

    def ssm_step(self, x, state=None):
        A = -mx.exp(self.A_log)
        D = self.D
        deltaBC = self.x_proj(x)
        delta, B, C = mx.split(
            deltaBC,
            indices_or_sections=[
                self.time_step_rank,
                self.time_step_rank + self.ssm_state_size,
            ],
            axis=-1,
        )
        delta = nn.softplus(self.dt_proj(delta))
        new_state = mx.expand_dims(delta * x, -1) * mx.expand_dims(B, 1)
        if state is not None:
            new_state += state * mx.exp(mx.expand_dims(delta, -1) * A)
        y = (new_state @ mx.expand_dims(C, -1)).squeeze(2)
        y = y + D * x
        return y, new_state

    def __call__(self, x, cache):
        B, T, D = x.shape
        if cache is None:
            cache = [None, None]

        outputs = []
        for t in range(T):
            xt = x[:, t, :]
            xz = self.in_proj(xt)
            x_t, z_t = xz.split(indices_or_sections=2, axis=1)
            conv_out, cache[0] = self.conv1d(mx.expand_dims(x_t, 1), cache[0])
            x_t = conv_out.squeeze(1)
            x_t = nn.silu(x_t)
            y_t, cache[1] = self.ssm_step(x_t, cache[1])
            z_t = nn.silu(z_t)
            output_t = y_t * z_t
            output_t = self.out_proj(output_t)
            outputs.append(output_t)
        output = mx.stack(outputs, axis=1)
        return output


class SparseMoEBlock(nn.Module):
    def __init__(self, args: ModelArgs, num_experts: int, num_experts_per_tok: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok

        self.router = None
        if num_experts > 1:
            self.router = nn.Linear(args.hidden_size, args.num_experts, bias=False)

        self.experts = [MLP(args) for _ in range(args.num_experts)]

    def one_hot(self, indices, num_classes):
        shape = indices.shape + (num_classes,)
        x = mx.zeros(shape)
        x = x.at[..., indices] = 1
        return x

    def __call__(
        self,
        x: mx.array
    ):
        B, L, D = x.shape

        # no routing
        if self.num_experts == 1:
            final_hidden_states = self.experts[0](x)
            router_logits = mx.ones((B * L, 1))
            return final_hidden_states, router_logits

        # routing
        x = x.reshape(-1, D)
        router_logits = self.router(x)

        routing_weights = mx.softmax(router_logits, axis=1)

        # MLX equivalent of topk
        routing_weights, selected_experts = mx.sort(routing_weights, axis=-1)
        routing_weights = routing_weights[:, -self.top_k:]
        selected_experts = selected_experts[:, -self.top_k:]

        final_hidden_states = mx.zeros((B * L, D))

        # One hot encode the selected experts
        expert_mask = self.one_hot(selected_experts, self.num_experts).transpose(2, 1, 0)

        # loop over experts
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            # Create indices using comparison
            mask = expert_mask[expert_idx] > 0
            indices = mx.arange(mask.size)[mask]
            top_x = indices % B * L
            idx = (indices // (B * L)).astype(mx.int32)

            if top_x.size == 0:
                continue

            current_state = x[top_x].reshape(-1, D)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            updates = mx.zeros_like(final_hidden_states)
            updates = updates.at[top_x] = current_hidden_states
            final_hidden_states = final_hidden_states + updates

        final_hidden_states = final_hidden_states.reshape(B, L, D)
        return final_hidden_states, router_logits


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_dim = args.hidden_size
        self.intermediate_size = args.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_size, bias=False)

    def __call__(
        self,
        x: mx.array
    ):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Jamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        decoder_layers = []
        for i in range(args.num_hidden_layers):
            is_attn = True if (i - self.args.attn_layer_offset) % self.args.attn_layer_period == 0 else False
            is_expert = True if (i - self.args.expert_layer_offset) % self.args.expert_layer_period == 0 else False

            num_experts = self.args.num_experts if is_expert else 1

            if is_attn:
                decoder_layers.append(AttentionLayer(args, num_experts))
            else:
                decoder_layers.append(MambaLayer(args, num_experts))

        self.layers = decoder_layers

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None
    ):
        for i, decoder_layer in enumerate(self.layers):
            layer_output, cache[i] = decoder_layer(x, cache[i], mask)
            x = layer_output[0]

        return x, cache


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.embedding = nn.Embedding(args.vocab_size, args.hidden_size)
        self.jamba = Jamba(args)

        self.final_layernorm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None
    )
        B, L = inputs.shape
        x = self.embedding(inputs)

        mask = create_attention_mask(x, cache)

        x, cache = self.jamba(x, cache, mask)
        out = self.final_layernorm(x)

        if self.args.tie_word_embeddings:
            out = self.embedding.as_linear(out)
        else:
            out = self.lm_head(out)
        return out, cache
