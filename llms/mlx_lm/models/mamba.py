from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "mamba"
    d_model: int = 1024
    hidden_size: int = 1024
    intermediate_size: int = 2048
    n_layer: int = 48
    vocab_size: int = 50280
    state_size: int = 16
    expand: int = 2
    conv_kernel: int = 4
    layer_norm_epsilon: float = 1e-05
    residual_in_fp32: bool = True

    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_init_scheme: str = "random" # "random" or "constant"
    time_step_scale: float = 1.0
    time_step_floor: float = 1e-4
    time_step_rank: int = 64

    use_bias: bool = False
    use_conv_bias: bool = True

    pscan: bool = False # use parallel scan mode or sequential mode when training. on MLX, the pscan isn't performant.


class MambaCache:

    def __init__(
        self, args: ModelArgs, batch_size: int
    ):
        self.seqlen_offset = 0
        intermediate_size = args.intermediate_size
        ssm_state_size = args.state_size
        conv_kernel_size = args.conv_kernel

        self.conv_states = {
            i: mx.zeros(batch_size, intermediate_size)
            for i in range(args.n_layer)
        }
        self.ssm_states = {
            i: mx.zeros(batch_size, intermediate_size)
            for i in range(args.n_layer)
        }


def pscan_f(A, X):
    Aa = A
    Xa = X

    B, D, L, _ = A.shape

    num_steps = int(math.log2(L))

    # up sweep
    for k in range(num_steps):
        T = 2 * (Xa.shape[2] // 2)

        Aa = Aa[:, :, :T].reshape(B, D, T//2, 2, -1)
        Xa = Xa[:, :, :T].reshape(B, D, T//2, 2, -1)

        Xa[:, :, :, 1] += Aa[:, :, :, 1] * Xa[:, :, :, 0]
        Aa[:, :, :, 1] *= Aa[:, :, :, 0]

        A[:, :, 2**(k+1)-1::2**(k+1)] = Aa[:, :, :, 1]
        X[:, :, 2**(k+1)-1::2**(k+1)] = Xa[:, :, :, 1]

        Aa = Aa[:, :, :, 1]
        Xa = Xa[:, :, :, 1]

    # down sweep
    for k in range(num_steps-1, -1, -1):
        Aa = A[:, :, 2**k-1::2**k]
        Xa = X[:, :, 2**k-1::2**k]

        step_len = Xa.shape[2]
        T = 2 * (step_len // 2)

        if T < step_len:
            last_val_aa = Aa[:, :, -1] * Aa[:, :, -2]
            last_val_xa = Xa[:, :, -1] + Aa[:, :, -1] * Xa[:, :, -2]

        Aa = Aa[:, :, :T].reshape(B, D, T//2, 2, -1)
        Xa = Xa[:, :, :T].reshape(B, D, T//2, 2, -1)

        Xa[:, :, 1:, 0] += Aa[:, :, 1:, 0] * Xa[:, :, :-1, 1]
        Aa[:, :, 1:, 0] *= Aa[:, :, :-1, 1]

        if T == step_len:
            A[:, :, 2**k-1::2**(k+1)] = Aa[:, :, :, 0]
            X[:, :, 2**k-1::2**(k+1)] = Xa[:, :, :, 0]
        else:
            A[:, :, 2**k-1::2**(k+1)] = mx.concatenate([Aa[:, :, :, 0], mx.array([last_val_aa]).reshape(B, D, 1, -1)], axis=2)
            X[:, :, 2**k-1::2**(k+1)] = mx.concatenate([Xa[:, :, :, 0], mx.array([last_val_xa]).reshape(B, D, 1, -1)], axis=2)


def pscan(A_in, X_in):
    """
    Applies the parallel scan operation, as defined above. Returns a new tensor.

    Args:
        A_in : (B, L, ED, N)
        X_in : (B, L, ED, N)

    Returns:
        H : (B, L, ED, N)
    """

    A = A_in[:].transpose(0, 2, 1, 3)
    X = X_in[:].transpose(0, 2, 1, 3)

    pscan_f(A, X)

    return X.transpose(0, 2, 1, 3)


def softplus(x, beta=1, threshold=20):
    scaled_x = beta * x
    mask = scaled_x > threshold
    return mx.where(mask, x, 1/beta * mx.logaddexp(0, x))


def unsqueeze(x, axis):
    assert axis <= len(x.shape)
    if axis >= 0:
        new_shape = x.shape[:axis] + tuple([1]) + x.shape[axis:]
    else:
        new_shape = x.shape + tuple([1])
    return x.reshape(new_shape)


def clamp(x, min=None, max=None):
    if min is not None:
        mask_lower = x < min
    if max is not None:
        mask_upper = x > max

    if min is not None:
        if max is not None:
            return mx.where(mask_upper, max, mx.where(mask_lower, min, x))
        return mx.where(mask_lower, min, x)

    return mx.where(mask_upper, max, x)


class MambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.state_size
        self.conv_kernel_size = args.conv_kernel
        self.intermediate_size = args.intermediate_size
        self.time_step_rank = int(args.time_step_rank)
        self.use_conv_bias = args.use_conv_bias

        self.act = nn.SiLU

        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=args.use_bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=args.use_conv_bias,
            kernel_size=args.conv_kernel,
            # groups=self.intermediate_size,
            padding=args.conv_kernel - 1,
        )
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        dt_init_std = args.time_step_rank**-0.5 * args.time_step_scale

        if args.time_step_init_scheme == "constant":
            self.dt_proj.weight = dt_init_std * mx.ones_like(self.dt_proj.weight)
        elif args.time_step_init_scheme == "random":
            self.dt_proj.weight = mx.random.uniform(-dt_init_std, dt_init_std, self.dt_proj.weight.shape)
        else:
            raise NotImplementedError

        dt = clamp(mx.exp(
            mx.random.uniform(shape=[args.hidden_size]) * (math.log(args.time_step_max) - math.log(args.time_step_min)) + math.log(args.time_step_min)
        ), min=args.time_step_floor)
        inv_dt = dt + mx.log1p(-mx.exp(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        self.dt_proj.bias = inv_dt

        A = mx.repeat(mx.arange(1., 16 + 1.).reshape([1, 16]), repeats=args.hidden_size, axis=0)
        self.A_log = mx.log(A)
        self.D = mx.ones([args.hidden_size])

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=args.use_bias)


    def __call__(self, x: mx.array, cache: Optional[MambaCache]=None):
        _, L, _ = x.shape
        xz = self.in_proj(x)
        x, z = x.split(2, axis=2)

        # depthwise convolution over time, with a short filter
        x = self.conv1d(x)[:, :L, :]
        x = self.ssm(self.act(x))

        # z branch
        z = self.act(z)
        return self.out_proj(x * z)

    def ssm(self, x: mx.array):
        A = -mx.exp(self.A_log)
        D = self.D

        deltaBC = self.x_proj(x)

        delta, B, C = mx.split(deltaBC, indices_or_sections=[self.args.time_step_rank, self.args.time_step_rank+self.args.hidden_size], axis=-1)
        delta = softplus(self.dt_proj(delta))

        if self.args.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):
        deltaA = mx.exp(unsqueeze(delta, -1) * A) # (B, L, ED, N)
        deltaB = unsqueeze(delta, -1) * unsqueeze(B, 2) # (B, L, ED, N)

        BX = deltaB * unsqueeze(x, -1) # (B, L, ED, N)

        hs = pscan(deltaA, BX)

        y = (hs @ unsqueeze(C, -1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x
        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        _, L, _ = x.shape

        deltaA = mx.exp(unsqueeze(delta, -1) * A)
        deltaB = unsqueeze(delta, -1) * unsqueeze(B, 2)

        BX = deltaB * unsqueeze(x, -1)

        h = mx.zeros([x.shape[0], self.args.hidden_size, self.args.state_size])
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = mx.stack(hs, axis=1)

        y = (hs @ unsqueeze(C, -1)).squeeze(3)

        y = y + D * x

        return y


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.norm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.mixer = MambaMixer(args)

    def forward(self, x: mx.array, cache: Optional[MambaCache] = None):
        r = self.mixer(self.norm(x), cache)
        h = r + x
        return h


class MambaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args

        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)

        self.layers = [MambaBlock(args) for _ in range(args.n_layer)]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def __call__(self, x, cache):
        x = self.embeddings(x)

        if cache is None:
            cache_params = MambaCache(
                self.config, x.size(0)
            )

        for i, layer in enumerate(self.layers):
            x = layer(x)

        return self.norm_f(x)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type

        self.backbone = MambaModel(args)

        self.lm_head = nn.Linear(self.args.hidden_size, self.args.vocab_size, bias=False)

    def __call__(self, x, cache):

        x, caches = self.backbone(x, cache)

        return self.lm_head(x)

    # def sanitize(self, weights):
    #     for k in weights:
    #         if "embedding." in k:
    #             weights[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
    #             break
    #     return weights



    @property
    def layers(self):
        return self.backbone.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // 32

    @property
    def n_kv_heads(self):
        return 8
