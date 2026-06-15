"""Startup check that flash-attention can dispatch on our production SDPA shapes.

Why a startup probe rather than a per-call or global toggle:
  - Per-call `sdpa_kernel(FLASH_ATTENTION)` does not compose with `torch.compile`:
    Dynamo runs the FA dispatch check against FakeTensors at trace time and fails
    with "No available kernel".
  - The global backend toggle (`torch.backends.cuda.enable_*_sdp`) has the same
    effect during Dynamo's trace.

Instead we probe once at trainer init with the shapes the production SDPA calls
use (head_dim, dtype, is_causal). If FA cannot dispatch on those, we raise loudly;
otherwise we leave SDPA's runtime backend selection alone. PD's SDPA inputs are
config-determined and stable across the run, so a passing startup probe implies a
passing runtime dispatch.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def verify_flash_attention_available(
    *,
    head_dim: int,
    n_heads: int,
    seq_len: int,
    is_causal: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Confirm flash-attention can dispatch on a representative SDPA call; raise if not.

    Args:
        head_dim: per-head dimension; pass the largest production head_dim so a
            too-large config (FA caps head_dim at 128 on older builds, 256 newer)
            errors early.
        n_heads: number of attention heads in the probe.
        seq_len: query/key/value sequence length in the probe.
        is_causal: match the mask mode of the production call.
        device: CUDA device to run the probe on.
        dtype: must be bf16 or fp16 (FA does not support fp32).
    """
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
            f"dtype={dtype}, is_causal={is_causal}) — production SDPA calls would "
            f"silently fall back to a slower kernel. Underlying error: {e}"
        ) from e
