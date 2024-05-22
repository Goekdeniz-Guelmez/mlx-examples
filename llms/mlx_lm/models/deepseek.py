from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.transformer import Dropout
import numpy as np

from .base import BaseModelArgs

@dataclass
class ModelArgs(BaseModelArgs):
    max_positions: int
    rope_theta: float
    attention_dropout: float
    attention_bias: bool
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    rope_traditional: bool
    rope_scaling: Optional[Dict[str, Union[str, float]]]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.attention_dropout = args.attention_dropout
        self.hidden_size = args.hidden_size
        self.num_heads = n_heads = args.num_attention_heads
        self.rope_theta = args.rope_theta

        self.head_dim = head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.num_key_value_heads = args.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=args.attention_bias)

        self.rope = nn.RoPE(
            dims=self.head_dim,
            traditional=args.rope_traditional,
            base=self.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ):
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        attn_output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(attn_output), (keys, values)


class DeepseekDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size

        self.self_attn = Attention(args)

        self.mlp = DeepseekMoE(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        def __call__(
            self,
            x: mx.array,
            mask: Optional[mx.array] = None,
            cache: Optional[Tuple[mx.array, mx.array]] = None,
        ):
            r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
            h = x + r
            r = self.mlp(self.post_attention_layernorm(h))
            out = h + r
            return out, cache


class DeepseekMLP(nn.Module):
    def __init__(self, args, hidden_size = None, intermediate_size = None):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = args.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.top_k = args.num_experts_per_tok
        self.n_routed_experts = args.n_routed_experts

        self.scoring_func = args.scoring_func
        self.alpha = args.aux_loss_alpha
        self.seq_aux = args.seq_aux

        # topk selection algorithm
        self.norm_topk_prob = args.norm_topk_prob
        self.gating_dim = args.hidden_size
        self.weight = mx.zeros((self.n_routed_experts, self.gating_dim))

    def __call__(self, x):
        bsz, seq_len, h = x.shape

        ### compute gating score
        x = x.view(-1, h)
        logits = nn.linear(x, self.weight, None)
        scores = logits.softmax(dim=-1)

        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        ### select top-k experts
        topk_weight, topk_idx = mx.topk(scores, k=self.top_k, axis=-1, sorted=False)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = mx.zeros(bsz, self.n_routed_experts)
                ce.scatter_add_(1, topk_idx_for_aux_loss, mx.ones(bsz, seq_len * aux_topk)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class DeepseekMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_experts_per_tok = args.num_experts_per_tok
        self.experts = [DeepseekMLP(args, intermediate_size = args.moe_intermediate_size) for i in range(args.n_routed_experts)]
        self.gate = MoEGate(args)
        if args.n_shared_experts is not None:
            intermediate_size = args.moe_intermediate_size * args.n_shared_experts
            self.shared_experts = DeepseekMLP(args=args, intermediate_size = intermediate_size)

    def __call__(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.args.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache
