from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
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

    def __post_init__(self):
        self.hidden_size = self.expand * self.d_model # E*D = ED in comments

def softplus(x, beta=1, threshold=20):
    scaled_x = beta * x
    mask = scaled_x > threshold
    return mx.where(mask, x, 1/beta * mx.logaddexp(0, x))

def unsqueeze(x, axis):
    """
    Same API as PyTorch.
    """

    assert axis <= len(x.shape)
    if axis >= 0:
        new_shape = x.shape[:axis] + tuple([1]) + x.shape[axis:]
    else:
        new_shape = x.shape + tuple([1])
    return x.reshape(new_shape)

def clamp(x, min=None, max=None):
    mask_lower = None
    mask_upper = None
    if min is not None:
        mask_lower = x < min
    if max is not None:
        mask_upper = x > max

    if min is not None:
        if max is not None:
            return mx.where(mask_upper, max, mx.where(mask_lower, min, x))
        return mx.where(mask_lower, min, x)

    return mx.where(mask_upper, max, x)

def topk(x, k):
    """
    Returns the top k biggest values of x along the 2nd dim.

    Args:
        x : (B, vocab_size). can be probs or logits

    Returns:
        values : (B, k). ordered from lowest to biggest val
    """

    return mx.sort(x)[:, -k:]

class DepthWiseConv1d(nn.Module):
    def __init__(self, channels, kernel_size, bias, padding):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = padding

        self.conv1d = nn.Conv1d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, bias=True, padding=padding)

        # see comment below
        indices = mx.arange(channels)
        mask = mx.zeros_like(self.conv1d.weight)
        mask[indices, :, indices] = 1
        self.conv1d.weight *= mask

    def __call__(self, x):
        return self.conv1d(x)


def pscan_f(A, X):
    # A : (B, D, L, N)
    # X : (B, D, L, N)

    # modifies X in place by doing a parallel scan.
    # more formally, X will be populated by these values :
    # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
    # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

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

        last_val_aa = None
        last_val_xa = None
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

# main function, used in the Mamba model (mamba_mlx.py)
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


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.residual_in_fp32 = args.residual_in_fp32

        self.norm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.mixer = MambaMixer(args)

    def __call__(self, x): #, cache_params: Optional[MambaCache] = None):
        h = self.norm(x.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = x.to(torch.float32)
        h = self.mixer(h ) #), cache_params=cache_params)
        r = x + h
        return r


class MambaMixer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.state_size
        self.conv_kernel_size = args.conv_kernel
        self.intermediate_size = args.intermediate_size
        self.time_step_rank = int(args.time_step_rank)
        self.use_conv_bias = args.use_conv_bias
        self.use_bias = args.use_bias

        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=args.use_conv_bias,
            kernel_size=args.conv_kernel,
            # groups=self.intermediate_size,
            padding=args.conv_kernel - 1,
        )

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=args.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = mx.repeat(mx.arange(1., 16 + 1.).reshape([1, 16]), repeats=args.hidden_size, axis=0)
        self.A_log = mx.log(A)
        self.D = torch.ones(self.intermediate_size)
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=args.use_bias)

    # def slow_forward(self, input_states, cache_params: Optional[MambaCache]=None):
    #     batch_size, seq_len, _ = input_states.shape
    #     dtype = input_states.dtype
    #     # 1. Gated MLP's linear projection
    #     projected_states = self.in_proj(input_states).transpose(1, 2)                   # [batch, 2 * intermediate_size, seq_len]
    #     hidden_states, gate = projected_states.chunk(2, dim=1)

    #     # 2. Convolution sequence transformation
    #     if cache_params is not None:
    #         ssm_state = cache_params.ssm_states[self.layer_idx].clone()
    #         if cache_params.seqlen_offset > 0:
    #             conv_state = cache_params.conv_states[self.layer_idx]                   # [batch, intermediate_size, conv_kernel_size]
    #             conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
    #             conv_state[:, :, -1] = hidden_states[:, :, 0]
    #             cache_params.conv_states[self.layer_idx].copy_(conv_state)
    #             hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
    #             if self.use_conv_bias:
    #                 hidden_states += self.conv1d.bias
    #             hidden_states = nn.sliu(hidden_states).to(dtype).unsqueeze(-1)         # [batch, intermediate_size, 1] : decoding
    #         else:
    #             conv_state = mx.pad(
    #                 hidden_states,
    #                 (self.conv_kernel_size - hidden_states.shape[-1], 0)
    #             )
    #             cache_params.conv_states[self.layer_idx].copy_(conv_state)
    #             hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])     # [batch, intermediate_size, seq_len]
    #     else:
    #         ssm_state = torch.zeros(
    #             (batch_size, self.intermediate_size, self.ssm_state_size),
    #             device=hidden_states.device, dtype=dtype
    #         )
    #         hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])         # [batch, intermediate_size, seq_len]

    #     # 3. State Space Model sequence transformation
    #     # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
    #     ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    #     time_step, B, C = torch.split(
    #         ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
    #     )
    #     discrete_time_step = self.dt_proj(time_step)                                    # [batch, seq_len, intermediate_size]
    #     discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]

    #     # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
    #     A = -torch.exp(self.A_log.float())                                              # [intermediate_size, ssm_state_size]
    #     discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
    #     discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()       # [batch, intermediate_size, seq_len, ssm_state_size]
    #     deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

    #     # 3.c perform the recurrence y ← SSM(A, B, C)(x)
    #     scan_outputs = []
    #     for i in range(seq_len):
    #         ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediate_size, ssm_state]
    #         scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediate_size, 1]
    #         scan_outputs.append(scan_output[:, :, 0])
    #     scan_output = torch.stack(scan_outputs, dim=-1)                                # [batch, intermediate_size, seq_len]
    #     scan_output = scan_output + (hidden_states * self.D[None, :, None])
    #     scan_output = (scan_output * self.act(gate))

    #     if cache_params is not None:
    #         cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

    #     # 4. Final linear projection
    #     contextualized_states = self.out_proj(scan_output.transpose(1, 2))  # [batch, seq_len, hidden_size]
    #     return contextualized_states

    def __call__(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        _, L, _ = x.shape

        xz = self.in_proj(x) # (B, L, 2*ED)
        x, z = xz.split(indices_or_sections=2, axis=2) # (B, L, ED), (B, L, ED)

        # x branch
        x = self.conv1d(x)[:, :L, :] # depthwise convolution over time, with a short filter

        x = nn.silu(x)
        y = self.ssm(x)

        # z branch
        z = nn.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, L, D)

        return output

    def ssm(self, x):
        # x : (B, L, ED)

        # y : (B, L, ED)

        A = -mx.exp(self.A_log) # (ED, N)
        D = self.D

        deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)

        delta, B, C = mx.split(deltaBC, indices_or_sections=[self.args.time_step_rank, self.args.time_step_rank+self.args.state_size], axis=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta = softplus(self.dt_proj(delta)) # (B, L, ED)

        if self.args.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = mx.exp(unsqueeze(delta, -1) * A) # (B, L, ED, N)
        deltaB = unsqueeze(delta, -1) * unsqueeze(B, 2) # (B, L, ED, N)

        BX = deltaB * unsqueeze(x, -1) # (B, L, ED, N)

        hs = pscan(deltaA, BX)

        y = (hs @ unsqueeze(C, -1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = mx.exp(unsqueeze(delta, -1) * A) # (B, L, ED, N)
        deltaB = unsqueeze(delta, -1) * unsqueeze(B, 2) # (B, L, ED, N)

        BX = deltaB * unsqueeze(x, -1) # (B, L, ED, N)

        h = mx.zeros([x.shape[0], self.args.hidden_size, self.args.state_size]) # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = mx.stack(hs, axis=1)

        y = (hs @ unsqueeze(C, -1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y


import torch

def torch_to_mlx_depthwise_weights(torch_weights):
    torch_weights = torch_weights.moveaxis(2, 1)  # (channels, kernel_size, 1) = (ED, conv_kernel, 1)
    channels, kernel_size, _ = torch_weights.shape

    mlx_weights = torch.zeros(channels, kernel_size, channels)

    indices = torch.arange(channels)
    if torch_weights[:, :, 0].dtype == torch.bfloat16:
        mlx_weights[indices, :, indices] = torch_weights[:, :, 0].float()
    else:
        mlx_weights[indices, :, indices] = torch_weights[:, :, 0]
    return mlx_weights

def map_mambapy_torch_to_mlx(weights):
    new_state_dict = {}
    for key, value in weights.items():
        # from torch to mlx, we need to convert the conv weights (see misc.py for explanations)
        if 'conv1d.weight' in key:
            value = torch_to_mlx_depthwise_weights(value)

        if 'conv1d' in key:
            key = key.replace('conv1d', 'conv1d.conv1d')

        new_state_dict[key] = value
    return new_state_dict


class MambaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args

        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)

        self.layers = [MambaBlock(args) for _ in range(args.n_layer)]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def __call__(self, x):
        x = self.embeddings(x)

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

        x, caches = self.backbone(x)

        return self.lm_head(x), cache

    def sanitize(self, weights):
        for k in weights:
            if "embedding." in k:
                weights[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break
        return weights



    @property
    def layers(self):
        return self.backbone.layers
