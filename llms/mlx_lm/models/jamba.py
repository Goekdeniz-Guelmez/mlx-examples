from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import BaseModelArgs

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    attention_dropout: float
    mamba_inner_layernorms: float
    attn_layer_offset: int
    attn_layer_period: int
    expert_layer_offset: int
    expert_layer_period: int
    hidden_size: int
    intermediate_size: int
    initializer_range: float
    mamba_conv_bias: bool
    mamba_d_conv: int
    mamba_d_state: int
    mamba_dt_rank: int
    mamba_expand: int
    mamba_proj_bias: bool
    max_position_embeddings: int
    num_attention_heads: int
    num_experts: int
    num_experts_per_tok: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    router_aux_loss_coef: int
    tie_word_embeddings: bool
    vocab_size: int


@dataclass
class MambaCacheParams:
    seqlen_offset: int = 0
    conv_states: Dict[int, mx.array] = field(default_factory=dict)
    ssm_states: Dict[int, mx.array] = field(default_factory=dict)


class JambaAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # self.attention_dropout = args.attention_dropout
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.attention_dropout = nn.Dropout(args.attention_dropout)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ):
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values)


class JambaMambaMixer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.mamba_d_state
        self.conv_kernel_size = args.mamba_d_conv
        self.intermediate_size = args.mamba_expand * args.hidden_size
        self.time_step_rank = args.mamba_dt_rank
        self.use_conv_bias = args.mamba_conv_bias
        self.use_bias = args.mamba_proj_bias

        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            # groups=self.intermediate_size,
            padding=self.conv_kernel_size - 1,
        )

        self.act = nn.SiLU
        self.apply_inner_layernorms = args.mamba_inner_layernorms

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=self.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = mx.arange(1, self.ssm_state_size + 1)[None, :]
        A = mx.expand_dims(A, axis=-1)
        self.A_log = mx.log(A)
        self.D = mx.ones(self.intermediate_size)

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.use_bias)

        if self.apply_inner_layernorms:
            self.dt_layernorm = nn.RMSNorm(self.time_step_rank, eps=args.rms_norm_eps)
            self.B_layernorm = nn.RMSNorm(self.ssm_state_size, eps=args.rms_norm_eps)
            self.C_layernorm = nn.RMSNorm(self.ssm_state_size, eps=args.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def mixer_forward(
        self,
        x: mx.array,
        cache_params: MambaCacheParams = None,
    ):
        B, L, D = x.shape

        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(x).transpose(0, 2, 1, 3) # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, axis=1)

        # 2. Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states

            if cache_params.seqlen_offset > 0:
                conv_state = cache_params.conv_states # [batch, intermediate_size, conv_kernel_size]
                conv_state = mx.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache_params.conv_states.copy_(conv_state)
                hidden_states = mx.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).unsqueeze(-1) # [batch, intermediate_size, 1] : decoding
            else:
                conv_state = mx.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states.copy_(conv_state)
                hidden_states = self.act(self.conv1d(hidden_states)[..., :L]) # [batch, intermediate_size, seq_len]
        else:
            ssm_state = mx.zeros(
                (B, self.intermediate_size, self.ssm_state_size)
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :L]) # [batch, intermediate_size, seq_len]

            # 3. State Space Model sequence transformation
            # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
            ssm_parameters = self.x_proj(hidden_states.transpose(0, 2, 1, 3))
            time_step, B, C = mx.split(
                ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], axis=-1
            )
            time_step, B, C = self._apply_layernorms(time_step, B, C)
            discrete_time_step = self.dt_proj(time_step) # [batch, seq_len, intermediate_size]
            discrete_time_step = mx.softplus(discrete_time_step).transpose(0, 2, 1, 3) # [batch, intermediate_size, seq_len]

            # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
            A = -mx.exp(self.A_log.float()) # [intermediate_size, ssm_state_size]
            discrete_A = mx.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
            discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float() # [batch, intermediade_size, seq_len, ssm_state_size]
            deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

            # 3.c perform the recurrence y ← SSM(A, B, C)(x)
            scan_outputs = []
            for i in range(L):
                ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
                scan_output = mx.matmul(ssm_state, C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
                scan_outputs.append(scan_output[:, :, 0])
            scan_output = mx.stack(scan_outputs, axis=-1)                                # [batch, seq_len, intermediade_size]
            scan_output = scan_output + (hidden_states * self.D[None, :, None])
            scan_output = (scan_output * self.act(gate))

            if cache_params is not None:
                cache_params.ssm_states.copy_(ssm_state)

            # 4. Final linear projection
            contextualized_states = self.out_proj(scan_output.transpose(1, 2))             # [batch, seq_len, hidden_size]
            return contextualized_states

    def __call__(
        self,
        hidden_states,
        cache,
    ):
        B, L, _ = hidden_states.shape
        if cache is not None:
            cache_params = MambaCacheParams(
                seqlen_offset=0 if hidden_states.shape[1] > 1 else cache.seen_tokens,
            )

            if len(cache.key_cache) > L:
                # we already have cache for this layer, use it
                # remove the dummy seqlen dim (dim=2)
                cache_params.conv_states = cache.key_cache.squeeze(2)
                cache_params.ssm_states = cache.value_cache.squeeze(2)
            else:
                # we don't have cache for this layer, initialize it with zeros
                batch_size = hidden_states.shape[0]
                cache_params.conv_states = mx.zeros(
                    batch_size,
                    self.intermediate_size,
                    self.conv_kernel_size,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                cache_params.ssm_states = mx.zeros(
                    batch_size,
                    self.intermediate_size,
                    self.ssm_state_size,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
        else:
            cache_params = None

        res = self.mixer_forward(hidden_states, cache_params)

        if cache is not None:
            cache.concatinate(
                # add dummy seqlen dim (dim=2) to match the number of dimensions of the attention cache
                cache_params.conv_states.unsqueeze(2),
                cache_params.ssm_states.unsqueeze(2),
            )

        return res, cache


class JambaMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ffn_dim = args.intermediate_size
        self.hidden_dim = args.hidden_size

        self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.down_proj = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.SiLU(self.gate_proj(x)) * self.up_proj(x))


class JambaSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs, num_experts_per_tok):
        super().__init__()
        self.hidden_dim = args.hidden_size
        self.ffn_dim = args.intermediate_size
        self.num_experts = args.num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # gating
        self.router = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = [
            JambaMLP(args) for _ in range(self.num_experts)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        ne = self.num_experts_per_tok
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        gates = self.router(x)

        inds = mx.stop_gradient(mx.argpartition(-gates, kth=ne - 1, axis=-1)[:, :ne])

        scores = mx.softmax(
            mx.take_along_axis(gates, inds, axis=-1).astype(mx.float32),
            axis=-1,
        ).astype(gates.dtype)

        if self.training:
            inds = np.array(inds)
            y = mx.zeros((x.shape[0], ne, x.shape[-1]), x.dtype)
            for e, expert in enumerate(self.experts):
                idx1, idx2 = map(mx.array, np.where(inds == e))
                if idx1.size == 0:
                    continue
                y[idx1, idx2] = expert(x[idx1])

            y = (y * scores[:, :, None]).sum(axis=1)
        else:
            y = []
            for xt, st, it in zip(x, scores, inds.tolist()):
                yt = mx.stack([self.experts[e](xt) for e in it], axis=-1)
                yt = (yt * st).sum(axis=-1)
                y.append(yt[None, :])
            y = mx.concatenate(y)

        return y.reshape(orig_shape)


class JambaAttentionDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, num_experts: int):
        super().__init__()
        self.hidden_size = args.hidden_size

        self.self_attn = JambaAttention(args)

        num_experts_per_tok = args.num_experts_per_tok if args.num_experts > 1 else 1
        self.moe = JambaSparseMoeBlock(args, num_experts_per_tok)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_moe_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ):
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.moe(self.pre_moe_layernorm(h))
        out = h + r
        return out, cache


class JambaMambaDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, num_experts: int):
        super().__init__()
        self.mamba = JambaMambaMixer(args)

        num_experts_per_tok = args.num_experts_per_tok if args.num_experts > 1 else 1

        self.moe = JambaSparseMoeBlock(args, num_experts_per_tok)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_moe_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        def __call__(
            self,
            x: mx.array,
            mask: Optional[mx.array] = None,
            cache: Optional[Tuple[mx.array, mx.array]] = None,
        ):
            r, cache = self.mamba(self.input_layernorm(x), mask, cache)
            B, L, _ = x.shape

            past_seqlen = self._get_past_seqlen(cache, L)

            num_attention_heads = self.mamba.config.num_attention_heads
            self_attn_weights = mx.zeros(B, num_attention_heads, L, past_seqlen)

            h = x + r

            r, router_logits = self.moe(self.pre_moe_layernorm(h))
            out = h + r
            return out, cache

        def _get_past_seqlen(self, past_key_value, seqlen):
            if past_key_value is None:
                return seqlen
            past_seqlen = past_key_value.get_seq_length()
            if past_seqlen == 0:
                return seqlen
            if past_key_value.attention_layer_idx is None:
                return seqlen
            if self.mamba.layer_idx < past_key_value.attention_layer_idx:
                return past_seqlen + 1
            return past_seqlen


class JambaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        # init each model layer, decide if it's mamba/attention and has experts or not
        decoder_layers = []
        for i in range(args.num_hidden_layers):
            is_attn = True if (i - self.args.attn_layer_offset) % self.args.attn_layer_period == 0 else False
            is_expert = True if (i - self.args.expert_layer_offset) % self.args.expert_layer_period == 0 else False

            num_experts = self.args.num_experts if is_expert else 1
            if is_attn:
                decoder_layers.append(JambaAttentionDecoderLayer(args, num_experts))
            else:
                decoder_layers.append(JambaMambaDecoderLayer(args, num_experts))

        self.layers = decoder_layers

        self.final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)
        return self.final_layernorm(h), cache


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.model = JambaModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out, cache = self.model(inputs, cache)
        return self.lm_head(out), cache

    @property
    def layers(self):
        return self.model.layers
