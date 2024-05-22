from dataclasses import dataclass
from typing import Optional, Tuple
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from numpy.core.multiarray import array

from .base import BaseModelArgs

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    attn_pdrop: float
    embd_pdrop: float
    resid_pdrop: float
    initializer_range: float
    layer_norm_epsilon: float
    n_ctx: int
    n_embd: int
    n_head: int
    n_layer: int
    vocab_size: int
    dropout: float = 0.0
    bias: bool = False
    n_kv_heads: Optional[int] = 2 # TODO GPT2 does NOT need that but the code from base.py requires it. Investigate


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.n_embd % args.n_head == 0
        self.n_embd = args.n_embd
        self.n_head = args.n_head

        self.c_attn = nn.Linear(args.n_embd, 3 * args.n_embd, bias=args.bias)
        self.c_proj = nn.Linear(args.n_embd, args.n_embd, bias=args.bias)

        self.attn_dropout = nn.Dropout(args.attn_pdrop)
        self.resid_dropout = nn.Dropout(args.resid_pdrop)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, L, D = x.shape

        queries, keys, values = self.c_attn(x).split(3, axis=2)

        queries = queries.reshape(B, L, self.n_head, D // self.n_head).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_head, D // self.n_head).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_head, D // self.n_head).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)

        # ATTENTION BEGIN
        # Manual Attention implemantation because of attn_dropout
        att = (queries @ keys.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(keys.shape[-1]))

        if mask is not None:
            att = att + mask

        att = mx.softmax(att, axis=-1)
        att = self.attn_dropout(att) # <----- Here
        out = att @ values
        # ATTENTION END

        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.resid_dropout(self.c_proj(out)), (keys, values)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.c_fc = nn.Linear(args.n_embd, 4 * args.n_embd, bias=args.bias)
        self.c_proj = nn.Linear(4 * args.n_embd, args.n_embd, bias=args.bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(args.resid_pdrop)

    def __call__(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class GPT2Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ln_1 = nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon, bias=args.bias)
        self.attn = Attention(args)
        self.ln_2 = nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon, bias=args.bias)
        self.mlp = MLP(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        r, cache = self.attn(self.ln_1(x), mask, cache)
        h = x + r
        r = self.mlp(self.ln_2(h))
        out = h + r
        return out, cache


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.bias = args.bias
        self.model_type = args.model_type

        self.wte = nn.Embedding(args.vocab_size, args.n_embd)
        self.wpe = nn.Embedding(args.n_ctx, args.n_embd)

        self.drop = nn.Dropout(args.embd_pdrop)
        self.h = [
            GPT2Block(args=args) for _ in range(args.n_layer)
        ]
        self.ln_f = nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon, bias=args.bias)

    # def __call__(
    #     self,
    #     inputs: mx.array,
    #     cache=None,
    # ):
    #     B, L = inputs.shape
    #     pos = mx.arange(0, L, 1, dtype=inputs.dtype)

    #     tok_emb = self.wte(inputs)
    #     pos_emb = self.wpe(pos)
    #     x = self.drop(tok_emb + pos_emb)

    #     mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
    #     mask = mask.astype(x.dtype)

    #     if cache is None:
    #         cache = [None] * len(self.h)

    #     for e, layer in enumerate(self.h):
    #         x, cache[e] = layer(x, mask, cache[e])

    #     return self.ln_f(x) @ self.wte.weight.T, cache

    def _forward_transformer_blocks(
        self, x: mx.array, pos: mx.array, mask=None, cache=None, build_cache=False
    ):
        tok_emb = self.wte(x)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        kv_cache = []

        if cache is not None:
            for i in range(len(cache)):
                x, cache[i] = self.h[i](x, mask=None, cache=cache[i])
        else:
            for block in self.h:
                x, curr_cache = block(x, mask=mask)
                if build_cache:
                    kv_cache.append(curr_cache)

        x = self.ln_f(x)
        return x, kv_cache if build_cache else cache


    def __call__(self, x: mx.array, cache: mx.array = None): # cache is target
        b, t = x.shape
        pos = mx.arange(0, t, 1, dtype=x.dtype)

        mask = nn.MultiHeadAttention.create_additive_causal_mask(t)
        mask = mask.astype(self.wte.weight.dtype)

        x, _ = self._forward_transformer_blocks(x, pos, mask=mask, build_cache=True)

        return x @ self.wte.weight.T

    def sanitize(self, weights):
        transpose_suffixes = (
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        )

        for key in list(weights.keys()):
            if key.endswith(transpose_suffixes):
                weights[key] = weights[key].T

        if not self.bias:
            weights = {k: v for k, v in weights.items() if 'bias' not in k}
        return weights

    @property
    def layers(self):
        return self.h

    @property
    def head_dim(self):
        return self.args.n_embd // self.args.n_head

    @property
    def n_kv_heads(self):
        return self.args.n_kv_heads


# class Model(nn.Module):
#     def __init__(self, args: ModelArgs):
#         super().__init__()
#         self.model_type = args.model_type

#         self.n_kv_heads = args.n_kv_heads # because base.py
#         self.head_dim = args.n_embd // args.n_head # because base.py

#         self.bias = args.bias
#         self.transformer = GPTModel(args)

#     def __call__(self, inputs: mx.array, cache=None):
#         out = self.transformer(inputs, cache)
#         return out

#     def sanitize(self, weights):
#         transpose_suffixes = (
#             "attn.c_attn.weight",
#             "attn.c_proj.weight",
#             "mlp.c_fc.weight",
#             "mlp.c_proj.weight",
#         )

#         for key in list(weights.keys()):
#             if key.endswith(transpose_suffixes):
#                 weights[key] = weights[key].T

#         if not self.bias:
#             weights = {k: v for k, v in weights.items() if 'bias' not in k}
#         return weights

#     @property
#     def layers(self):
#         return self.transformer.h
