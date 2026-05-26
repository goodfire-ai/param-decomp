"""Process-wide enforcement that every ``scaled_dot_product_attention`` call
dispatches to flash-attention. If FA can't service the inputs (head_dim > 128,
fp32 acts, unsupported mask shape, etc.) the SDPA call errors instead of
silently falling back to math / memory-efficient.

Why a global toggle instead of per-call ``sdpa_kernel(FLASH_ATTENTION)``?
The context-manager form doesn't compose with ``torch.compile`` — Dynamo's
FX tracing runs against FakeTensors and can't satisfy FA's per-input
dispatch check, so the trace fails with ``No available kernel``. Toggling
the global backend selection sidesteps that: the dispatch decision happens
at kernel call time against the real tensors, not at trace time.

Call ``enforce_flash_attention_only()`` once per process before any model
forward.
"""

import torch


def enforce_flash_attention_only() -> None:
    """Disable non-flash-attention SDPA backends process-wide.

    After this call, ``F.scaled_dot_product_attention(...)`` either runs on
    the flash kernel or raises ``RuntimeError('No available kernel...')`` —
    no silent fallback.

    Idempotent (safe to call multiple times).
    """
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    # cuDNN's SDPA is also a fallback path — keep it off so FA stays the
    # only legal kernel.
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
