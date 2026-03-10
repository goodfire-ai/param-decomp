"""RoPE-aware Q-K dot product helpers for SPD component analysis scripts.

Decomposes the RoPE-modulated dot product into content-aligned and cross-half
coefficients, allowing evaluation at arbitrary relative position offsets:

    W(Δ) = Σ_d [A_d · cos(Δ·θ_d) + B_d · sin(Δ·θ_d)]

Assumes non-adjacent-pairs RoPE layout (first-half/second-half dimension split).
"""

from collections.abc import Sequence

import torch
from einops import einsum
from torch import Tensor


def compute_qk_rope_coefficients(
    U_q: Tensor,
    U_k: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute A_d and B_d RoPE coefficients for all (q, k) pairs.

    Args:
        U_q: (n_q, head_dim) query weight vectors for a single head
        U_k: (n_k, head_dim) key weight vectors for a single head

    Returns:
        A: (n_q, n_k, half_dim) content-aligned coefficients
        B: (n_q, n_k, half_dim) cross-half coefficients (zero contribution at Δ=0)
    """
    half = U_q.shape[-1] // 2
    q1, q2 = U_q[..., :half], U_q[..., half:]
    k1, k2 = U_k[..., :half], U_k[..., half:]
    A = einsum(q1, k1, "q d, k d -> q k d") + einsum(q2, k2, "q d, k d -> q k d")
    B = einsum(q1, k2, "q d, k d -> q k d") - einsum(q2, k1, "q d, k d -> q k d")
    return A, B


def evaluate_qk_at_offsets(
    A: Tensor,
    B: Tensor,
    rotary_cos: Tensor,
    rotary_sin: Tensor,
    offsets: Sequence[int],
) -> Tensor:
    """Evaluate W(Δ) at specified relative position offsets.

    Args:
        A: (n_q, n_k, half_dim) content-aligned coefficients
        B: (n_q, n_k, half_dim) cross-half coefficients
        rotary_cos: (n_ctx, head_dim) precomputed cos buffer from model
        rotary_sin: (n_ctx, head_dim) precomputed sin buffer from model
        offsets: relative position offsets Δ to evaluate

    Returns:
        (n_offsets, n_q, n_k) dot product values at each offset
    """
    half = A.shape[-1]
    results = []
    for delta in offsets:
        cos_d = rotary_cos[delta, :half].float()
        sin_d = rotary_sin[delta, :half].float()
        W = (A * cos_d + B * sin_d).sum(dim=-1)
        results.append(W)
    return torch.stack(results)
