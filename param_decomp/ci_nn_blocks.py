"""Generic transformer building blocks used by the CI functions."""

from typing import override

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from param_decomp.components import _NonlinearityType, init_param_


class ParallelLinear(nn.Module):
    """`C` independent linear layers applied in parallel along an extra axis.

    Weights `[C, d_in, d_out]`, biases `[C, d_out]`; each slice is initialised
    independently via `init_param_`.
    """

    def __init__(self, C: int, input_dim: int, output_dim: int, nonlinearity: _NonlinearityType):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.empty(C, input_dim, output_dim))
        self.b = nn.Parameter(torch.zeros(C, output_dim))
        init_param_(self.W, fan_val=input_dim, nonlinearity=nonlinearity)

    @override
    def forward(self, x: Float[Tensor, "... C d_in"]) -> Float[Tensor, "... C d_out"]:
        return einops.einsum(x, self.W, "... C d_in, C d_in d_out -> ... C d_out") + self.b


class Linear(nn.Module):
    """Linear layer with zero-initialised bias and `init_param_` weight init.

    Fan value passed to `init_param_` is `input_dim`.
    """

    def __init__(self, input_dim: int, output_dim: int, nonlinearity: _NonlinearityType):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.empty(input_dim, output_dim))
        self.b = nn.Parameter(torch.zeros(output_dim))
        init_param_(self.W, fan_val=input_dim, nonlinearity=nonlinearity)

    @override
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einops.einsum(x, self.W, "... d_in, d_in d_out -> ... d_out") + self.b


class RoPEEmbedding(nn.Module):
    """Rotary Position Embedding applied to query/key tensors.

    Requires even `d_head`; supports sequence lengths up to `max_len`.
    """

    def __init__(self, d_head: int, max_len: int = 2048, base: float = 10000.0):
        super().__init__()
        assert d_head % 2 == 0, f"RoPE requires even d_head, got {d_head}"
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len
        self.d_head = d_head

    @override
    def forward(
        self,
        q: Float[Tensor, "... n_heads seq d_head"],
        k: Float[Tensor, "... n_heads seq d_head"],
    ) -> tuple[Float[Tensor, "... n_heads seq d_head"], Float[Tensor, "... n_heads seq d_head"]]:
        seq_len = q.shape[-2]
        assert seq_len <= self.max_len, f"seq_len {seq_len} exceeds max_len {self.max_len}"

        assert isinstance(self.inv_freq, Tensor)
        positions = torch.arange(seq_len, device=q.device, dtype=self.inv_freq.dtype)
        angles = einops.einsum(positions, self.inv_freq, "seq, d -> seq d")
        cos_emb = torch.cat([angles.cos(), angles.cos()], dim=-1)
        sin_emb = torch.cat([angles.sin(), angles.sin()], dim=-1)

        q_rot = self._apply_rotation(q, cos_emb, sin_emb)
        k_rot = self._apply_rotation(k, cos_emb, sin_emb)
        return q_rot, k_rot

    def _apply_rotation(
        self,
        x: Float[Tensor, "... n_heads seq d_head"],
        cos: Float[Tensor, "seq d_head"],
        sin: Float[Tensor, "seq d_head"],
    ) -> Float[Tensor, "... n_heads seq d_head"]:
        """Apply rotation: x' = x * cos + rotate_half(x) * sin."""
        x1 = x[..., : self.d_head // 2]
        x2 = x[..., self.d_head // 2 :]
        x_rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + x_rotated * sin


class SelfAttention(nn.Module):
    """Multi-head bidirectional self-attention with RoPE (`is_causal=False`).

    `d_model` must be divisible by `n_heads`.
    """

    def __init__(self, d_model: int, n_heads: int, max_len: int = 2048, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RoPEEmbedding(self.d_head, max_len, rope_base)

    @override
    def forward(self, x: Float[Tensor, "... seq d_model"]) -> Float[Tensor, "... seq d_model"]:
        *batch_dims, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(*batch_dims, seq_len, self.n_heads, self.d_head).transpose(-3, -2)
        k = k.view(*batch_dims, seq_len, self.n_heads, self.d_head).transpose(-3, -2)
        v = v.view(*batch_dims, seq_len, self.n_heads, self.d_head).transpose(-3, -2)

        q, k = self.rope(q, k)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )

        attn_out = attn_out.transpose(-3, -2).contiguous().view(*batch_dims, seq_len, self.d_model)
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm-attn-residual then RMSNorm-MLP-residual.

    The MLP is `Linear` layers with GELU between hidden layers and a linear projection
    back to `d_model`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_hidden_dims: list[int],
        max_len: int = 2048,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.attn = SelfAttention(
            d_model=d_model, n_heads=n_heads, max_len=max_len, rope_base=rope_base
        )
        self.d_model = d_model

        mlp_layers = nn.Sequential()
        in_dim = d_model
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(Linear(in_dim, hidden_dim, nonlinearity="relu"))
            mlp_layers.append(nn.GELU())
            in_dim = hidden_dim
        mlp_layers.append(Linear(in_dim, d_model, nonlinearity="linear"))
        self.mlp = mlp_layers

    @override
    def forward(self, x: Float[Tensor, "... seq d_model"]) -> Float[Tensor, "... seq d_model"]:
        x = x + self.attn(F.rms_norm(x, (self.d_model,)))
        x = x + self.mlp(F.rms_norm(x, (self.d_model,)))
        return x
