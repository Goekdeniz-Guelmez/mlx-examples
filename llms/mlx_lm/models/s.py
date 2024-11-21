import math
from dataclasses import dataclass, field
from typing import Tuple, Union
import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .cache import Mamba2Cache

@dataclass
class ModelArgs(BaseModelArgs):
    num_heads: int
    head_dim: int
    vocab_size: int
    hidden_size: int
    state_size: int
    num_hidden_layers: int
    layer_norm_epsilon: float
    expand: int
    conv_kernel: int
    n_groups: int
    use_bias: bool
    use_conv_bias: bool
    initializer_range: float
    residual_in_fp32: bool
    time_step_min: float
    time_step_max: float
    time_step_floor: float
    rescale_prenorm_residual: bool
    rms_norm: bool
    chunk_size: int
    tie_word_embeddings: bool
    use_cache: bool = True
    time_step_limit: Tuple[float, float] = field(default_factory=lambda: (0.0, float("inf")))
    time_step_rank: Union[int, str] = "auto"
    model_type: str = "mamba2"

    def __post_init__(self):
        if not hasattr(self, "intermediate_size"):
            self.intermediate_size = int(self.expand * self.hidden_size)
        if not hasattr(self, "head_dim"):
            self.head_dim = self.hidden_size // self.num_heads
        if self.time_step_rank == "auto":
            self.time_step_rank = math.ceil(self.hidden_size / 16)


class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps

    def __call__(self, hidden_states, gate=None):
        if gate is not None:
            hidden_states = hidden_states * nn.silu(gate)
        variance = mx.mean(hidden_states ** 2, axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


def silu(x):
    return x * mx.sigmoid(x)

def ssd(x, A, B, C, chunk_size):
    # Replace einsum operations with explicit reshape and matrix multiply
    batch, seqlen, nheads, dim = x.shape
    B = mx.expand_dims(B, axis=2)
    C = mx.expand_dims(C, axis=2)

    state = mx.zeros((batch, nheads, dim, B.shape[-1]))
    outputs = []

    for i in range(0, seqlen, chunk_size):
        chunk = slice(i, min(i + chunk_size, seqlen))
        dA = mx.exp(mx.expand_dims(A[chunk], axis=0))

        # Replace einsum with explicit operations
        x_chunk = x[:, chunk]  # [batch, chunk_size, nheads, dim]
        x_chunk = mx.transpose(x_chunk, [0, 2, 3, 1])  # [batch, nheads, dim, chunk_size]
        B_chunk = B[:, chunk]  # [batch, chunk_size, state_size]
        dBx = mx.matmul(x_chunk, B_chunk)  # [batch, nheads, dim, state_size]

        state = state * mx.expand_dims(dA, axis=-1) + dBx

        # Replace einsum with explicit operations
        C_chunk = C[:, chunk]  # [batch, chunk_size, state_size]
        y = mx.matmul(state, mx.transpose(C_chunk, [0, 2, 1]))  # [batch, nheads, dim, chunk_size]
        y = mx.transpose(y, [0, 3, 1, 2])  # [batch, chunk_size, nheads, dim]
        outputs.append(y)

    return mx.concatenate(outputs, axis=1), state


class DepthWiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, groups=None, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups if groups is not None else in_channels

        assert in_channels == out_channels, "In and out channels must be same for depthwise convolution"
        assert self.groups == in_channels, "Groups must be equal to in_channels for depthwise convolution"

        # Initialize weight with correct shape [C_out, 1, kernel_size]
        self.weight = mx.random.normal((out_channels, 1, kernel_size))
        self.bias = mx.zeros((out_channels,)) if bias else None

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        B, L, C = x.shape
        K = self.kernel_size

        assert C == self.in_channels, f"Input channels {C} doesn't match expected {self.in_channels}"

        # Handle caching for sequential processing
        if cache is not None and cache.conv_states[0] is not None:
            if isinstance(cache.conv_states[0], type(None)):
                cache.conv_states[0] = mx.zeros((B, K-1, C))
            x = mx.concatenate([cache.conv_states[0], x], axis=1)

        # Process each channel independently
        outputs = []
        for c in range(C):
            # Extract and reshape the channel
            x_c = x[:, :, c]  # [B, L]
            x_c = mx.expand_dims(x_c, axis=1)  # [B, 1, L]

            # Get weight for this channel - already in correct shape [1, 1, K]
            w_c = mx.expand_dims(self.weight[c], axis=0)  # Ensure [1, 1, K]

            # Apply convolution
            y_c = mx.conv_general(
                x_c,
                w_c,
                stride=1,
                padding=self.padding
            )

            if self.bias is not None:
                y_c = y_c + self.bias[c]

            outputs.append(mx.squeeze(y_c, axis=1))

        y = mx.stack(outputs, axis=-1)

        # Update cache
        if cache is not None:
            cache.conv_states[0] = x[:, -K+1:, :] if x.shape[1] >= K else x

        return y


class Mamba2Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.chunk_size = args.chunk_size

        d_in_proj = 2 * args.intermediate_size + 2 * args.state_size + args.num_heads
        self.in_proj = nn.Linear(args.hidden_size, d_in_proj, bias=args.use_bias)

        self.conv_dim = args.intermediate_size + 2 * args.state_size
        self.conv1d = DepthWiseConv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=args.conv_kernel,
            groups=self.conv_dim,
            bias=args.use_conv_bias,
            padding=args.conv_kernel - 1
        )

        self.dt_bias = mx.random.normal((args.num_heads,)) * args.initializer_range
        self.A_log = mx.random.normal((args.num_heads,)) * args.initializer_range
        self.D = mx.random.normal((args.num_heads,)) * args.initializer_range

        self.norm = MambaRMSNormGated(args.intermediate_size, eps=args.layer_norm_epsilon)
        self.out_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.use_bias)

        if args.rescale_prenorm_residual:
            layer_scale = math.sqrt(1.0 / args.num_hidden_layers)
            self.out_proj.weight = self.out_proj.weight * layer_scale

    def __call__(self, u: mx.array, cache=None):
        # Expect input shape: [batch_size, 1, hidden_size]
        batch_size, seq_len, _ = u.shape
        pad_size = self.chunk_size - (seq_len % self.chunk_size)

        # Initialize states if needed
        if cache.conv_states[0] is None:
            cache.conv_states[0] = mx.zeros((
                batch_size,
                self.args.conv_kernel - 1,
                self.conv_dim
            ))

        if cache.ssm_states[0] is None:
            cache.ssm_states[0] = mx.zeros((
                batch_size,
                self.args.num_heads,
                self.args.head_dim,
                self.args.state_size
            ))

        # Project input
        zxbcdt = self.in_proj(u)

        # Split projections
        z = zxbcdt[:, :, :self.args.intermediate_size]
        xBC = zxbcdt[:, :, self.args.intermediate_size:self.args.intermediate_size + 2*self.args.state_size + self.args.intermediate_size]
        dt = zxbcdt[:, :, -(self.args.num_heads):]

        # Process delta time
        dt = mx.reshape(dt, (batch_size, seq_len, self.args.num_heads))
        dt = mx.squeeze(dt, axis=0)  # Remove sequence dimension for single token
        dt = mx.clip(
            nn.softplus(dt + self.dt_bias),
            self.args.time_step_min,
            self.args.time_step_max
        )
        dt = mx.maximum(dt, self.args.time_step_floor)

        # Convolution step
        xBC = self.conv1d(xBC, cache=cache)
        xBC = silu(xBC)

        # Split conv output
        x = xBC[:, :, :self.args.intermediate_size]
        B = xBC[:, :, self.args.intermediate_size:self.args.intermediate_size + self.args.state_size]
        C = xBC[:, :, -self.args.state_size:]

        # Reshape for SSM
        x = mx.reshape(x, (batch_size, 1, self.args.num_heads, self.args.head_dim))
        x = mx.squeeze(x, axis=1)

        B = mx.reshape(B, (batch_size, 1, self.args.state_size))
        B = mx.broadcast_to(B, (batch_size, self.args.num_heads, self.args.state_size))
        B = mx.expand_dims(B, axis=2)

        C = mx.reshape(C, (batch_size, 1, self.args.state_size))
        C = mx.broadcast_to(C, (batch_size, self.args.num_heads, self.args.state_size))
        C = mx.expand_dims(C, axis=3)

        # SSM state update
        A = -mx.exp(self.A_log)
        dA = mx.exp(dt * mx.expand_dims(A, 0))
        dA = mx.expand_dims(mx.expand_dims(dA, -1), -1)

        x = mx.expand_dims(x, axis=3)
        dBx = mx.matmul(x, B)

        cache.ssm_states[0] = cache.ssm_states[0] * dA + dBx

        # Output computation
        y = mx.matmul(cache.ssm_states[0], C)
        y = mx.squeeze(y, axis=-1)

        # y = y + x[:, :, :, 0] * mx.expand_dims(self.D, -1)
        if pad_size > 0:
            y = y[:, :seq_len, :, :]

        # Final reshape and projections
        y = mx.reshape(y, (batch_size, 1, self.args.num_heads * self.args.head_dim))
        y = self.norm(y + z)

        return self.out_proj(y)


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.residual_in_fp32 = args.residual_in_fp32

        self.mixer = Mamba2Block(args)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(self, x: mx.array, cache):
        if self.residual_in_fp32:
            x = x.astype(mx.float32)
        return self.mixer(self.norm(x), cache) + x


class Mamba2(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [ResidualBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def __call__(self, x: mx.array, cache):
        x = self.embeddings(x)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            x = layer(x, c)
        return self.norm_f(x)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type

        self.backbone = Mamba2(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        B, T = inputs.shape

        x = self.backbone(inputs, cache)

        if self.args.tie_word_embeddings:
            logits = self.backbone.embeddings.as_linear(x)
        else:
            logits = self.lm_head(x)

        return logits

    def make_cache(self, batch_size=1):
        return [Mamba2Cache(batch_size, self.args.conv_kernel) for _ in range(len(self.layers))]

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "conv1d.weight" in k:
                # Ensure weights are in correct shape (channels, 1, kernel_size)
                if v.ndim == 2:
                    v = mx.expand_dims(v, axis=1)
                elif v.ndim == 1:
                    v = mx.expand_dims(mx.expand_dims(v, axis=0), axis=0)
                sanitized[k] = v
            else:
                sanitized[k] = v
        return sanitized

    @property
    def layers(self):
        return self.backbone.layers