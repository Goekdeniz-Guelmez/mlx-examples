from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import math

from llms.mlx_lm.models.mamba import MambaModel

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    num_heads: int
    num_hidden_layers: int
    hidden_size: int
    attention_hidden_size: int
    layer_norm_epsilon: float
    vocab_size: int
    intermediate_size: int
    rescale_every: int


def rwkv_linear_attention(
    time_decay,
    time_first,
    keys,
    values,
    cache=None
):
    _, L, _ = keys.shape
    output = mx.zeros_like(keys)

    if cache is None:
        num_state = mx.zeros_like(keys[:, 0]) # dtype should be float32 on all 3
        den_state = mx.zeros_like(keys[:, 0])
        max_state = mx.zeros_like(keys[:, 0]) - 1e38
    else:
        num_state, den_state, max_state = cache

    time_decay = -mx.exp(time_decay)

    for i in range(L):
        current_keys = keys[:, i].float()
        current_values = values[:, i]

        # wkv computation at time t
        max_for_output = mx.maximum(max_state, current_keys + time_first)
        e1 = mx.exp(max_state - max_for_output)
        e2 = mx.exp(current_keys + time_first - max_for_output)
        numerator = e1 * num_state + e2 * current_values
        denominator = e1 * den_state + e2
        output[:, i] = (numerator / denominator).to(output.dtype)

        # Update state for next iteration
        max_for_state = mx.maximum(max_state + time_decay, current_keys)
        e1 = mx.exp(max_state + time_decay - max_for_state)
        e2 = mx.exp(current_keys - max_for_state)
        num_state = e1 * num_state + e2 * current_values
        den_state = e1 * den_state + e2
        max_state = max_for_state

        if cache is not None:
            state = [num_state, den_state, max_state]

        return output, cache

class RwkvSelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        hidden_size = args.hidden_size

        attention_hidden_size = (
            args.attention_hidden_size if args.attention_hidden_size is not None else hidden_size
        )
        self.attention_hidden_size = attention_hidden_size

        self.time_decay = np.empty(attention_hidden_size)
        self.time_first = np.empty(attention_hidden_size)

        self.time_mix_key = np.empty(1, 1, hidden_size)
        self.time_mix_value = np.empty(1, 1, hidden_size)
        self.time_mix_receptance = np.empty(1, 1, hidden_size)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.output = nn.Linear(attention_hidden_size, hidden_size, bias=False)

        def __call__(x: mx.array, cache=None):
            receptance, kesy, values, cache = self.extract_key_value(x, cache=cache) # TODO implement

            rwkv, caches = rwkv_linear_attention(
                self.time_decay,
                self.time_first,
                keys,
                values,
                cache
            )

            if caches is not None:
                cache[2][:, :, self.layer_id] = caches[0]
                cache[3][:, :, self.layer_id] = caches[1]
                cache[4][:, :, self.layer_id] = caches[2]

            return self.output(receptance * rwkv), cache


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_size = args.hidden_size
        intermediate_size = (
            args.intermediate_size if args.intermediate_size is not None else 4 * args.hidden_size
        )

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.time_mix_key = np.empty(1, 1, hidden_size)
        self.time_mix_receptance = np.empty(1, 1, hidden_size)

        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array):
        shifted = self.time_shift(x)

        key = x * self.time_mix_key + shifted * (1 - self.time_mix_key)
        receptance = x * self.time_mix_receptance + shifted * (1 - self.time_mix_receptance)
        key = mx.square(nn.ReLU(self.key(key)))
        value = self.value(key)
        receptance = mx.sigmoid(self.receptance(receptance))

        return receptance * value


class RwkvBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # self.pre_ln = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_epsilon)

        self.ln1 = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_epsilon)

        self.attention = RwkvSelfAttention(args)
        self.feed_forward = MLP(args)

    def __call__(self, x: mx.array, cache=None):
        a, cache = self.attention(self.ln1(x), cache)
        h = x + a
        r = self.feed_forward(self.ln2(h))
        out = h + r
        return out, cache


class RwkvModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.blocks = [RwkvBlock(args) for idx in range(args.num_hidden_layers)]
        self.ln_out = nn.LayerNorm(args.hidden_size)

    def __call__(self, x: mx.array, cache: Optional[List[mx.array]] = None):
        x = self.embeddings(x)

        if cache is None:
            shape = (x.size(0), self.args.hidden_size)
            cache = [
                mx.zeros(*shape)
                for i in range(5)
            ]
            cache[4] -= 1e30

        for idx, block in enumerate(self.blocks):
            x, cache = block(x, cache)
            if self.layers_are_rescaled and self.args.rescale_every > 0 and (idx + 1) % self.args.rescale_every == 0:
                x = x / 2

        return self.ln_out(x), cache


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.rwkv = RwkvModel(args)
        self.head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        rwkv_outputs, state = self.rwkv(
            inputs,
            cache=cache
        )

        return self.head(rwkv_outputs), state
