"""Verify that flash-attention can dispatch on representative SDPA inputs.

Why not the strict per-call or global toggle?
  * Per-call ``sdpa_kernel(FLASH_ATTENTION)`` doesn't compose with
    ``torch.compile`` — Dynamo runs the FA dispatch check against
    FakeTensors at trace time and fails with ``No available kernel``.
  * The global backend toggle (``torch.backends.cuda.enable_*_sdp``) has
    the same effect during Dynamo's trace.

What we do instead:
  * Call ``verify_flash_attention_available(...)`` once at trainer init
    with the shapes our production SDPA calls use (head_dim, dtype,
    is_causal mode). If FA can't dispatch on those shapes, raise loudly.
  * After that, leave SDPA's runtime backend selection alone. PyTorch will
    pick FA when inputs allow it (and our inputs always will, since they
    came from the same config that passed verification).

The cheat is that PD's SDPA inputs are config-determined and stable across
the run — startup pass therefore implies runtime pass.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def verify_flash_attention_available(
    *,
    head_dim: int,
    n_heads: int = 8,
    seq_len: int = 64,
    is_causal: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Confirm FA can dispatch on a representative SDPA call. Raises if not.

    Args:
        head_dim: per-head dimension. FA caps at 128 (older builds) or 256
            (newer) — pass our largest production head_dim so a too-large
            config errors early.
        n_heads / seq_len: shape of the test SDPA. Don't matter much for
            FA's dispatch decision; defaults keep the test cheap.
        is_causal: match the mask mode of the production call.
        device: defaults to current CUDA device.
        dtype: bf16 by default (FA requires bf16/fp16).
    """
    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    q = torch.randn(1, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal)
    except RuntimeError as e:
        raise RuntimeError(
            f"flash-attention dispatch failed for shape "
            f"(n_heads={n_heads}, seq_len={seq_len}, head_dim={head_dim}, "
            f"dtype={dtype}, is_causal={is_causal}) — production SDPA calls "
            f"will silently fall back to a slower kernel. Underlying error: {e}"
        ) from e
