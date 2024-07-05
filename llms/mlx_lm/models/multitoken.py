from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

import math

from base import BaseModelArgs, KVCache, create_additive_causal_mask


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "multi-token"
    dim: int = 4096
    n_layers: int = 4
    n_heads: int = 2
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    n_future_tokens: int = 4
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    rope_traditional: bool = False

    max_batch_size: int = 32
    max_seq_len: int = 2048
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        if self.rope_scaling:
            required_keys = {"factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] != "linear":
                raise ValueError("rope_scaling 'type' currently only supports 'linear'")


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = n_heads = args.n_heads

        self.head_dim = args.dim // args.n_heads
        self.scale = self.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        rope_scale = (
            1 / args.rope_scaling["factor"]
            if args.rope_scaling is not None and args.rope_scaling["type"] == "linear"
            else 1
        )
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
            scale=rope_scale,
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[KVCache] = None) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array], cache: Optional[KVCache] = None):
        h = x + self.attention(self.attention_norm(x), mask, cache)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class MultiTokenModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.n_future_tokens = args.n_future_tokens

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = [
            TransformerBlock(args=args) for _ in range(args.n_layers - self.n_future_tokens + 1)
        ]

        self.extra_heads = [
            TransformerBlock(args=args) for _ in range(self.n_layers - self.n_future_tokens + 1, self.n_layers)
        ]

        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, return_all_heads: bool = False):
        h = self.tok_embeddings(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = create_additive_causal_mask(
                h.shape[1], cache[0].offset if cache is not None else 0
            )
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)
        h_trunk = h

        # Prediction heads.
        latents = []
        n_heads_to_use = self.n_future_tokens if return_all_heads else 1
        prediction_heads = [self.layers[-1]] + list(self.extra_heads)
        for layer, c in zip(prediction_heads[:n_heads_to_use], cache):
            h = layer(h_trunk, mask, cache=c)
            latents.append(h)

        h = mx.stack(latents, axis=-2)  # (_bsz, seqlen, n_heads_to_use, dim)
        h = self.norm(h)
        return self.output(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MultiTokenModel(args)

    def __call__(self, inputs: mx.array, cache=None, return_all_heads: bool = False):
        out = self.model(inputs=inputs, cache=cache, return_all_heads=return_all_heads)
        return out

    def sanitize(self, weights):
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.dim // self.args.n_heads

    @property
    def n_kv_heads(self):
        return self.args.n_kv_heads


model = Model(ModelArgs())

print(model)

tokens = mx.array([1])
tokens = mx.expand_dims(tokens, axis=0)

output = model(tokens, return_all_heads=True)
print(output)

output = mx.softmax(output, axis=-1)
print(output)

output = mx.argmax(output)
print(output)