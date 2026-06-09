"""Straight-through fake-quantization for probing low-precision PPGD warmup (Track-2).

A context manager sets a thread-local bit-width; while active, `maybe_fake_quant`
quantizes the operands of the `Components` matmuls. The forward sees low-precision
(quant→dequant) values, while the backward is the identity (straight-through estimator),
so gradients to the PPGD sources still flow. This simulates the numerics of a
low-precision warmup forward without needing real fp8 kernels.
"""

import contextvars
from collections.abc import Generator
from contextlib import contextmanager

import torch
from torch import Tensor

_fake_quant_bits: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "fake_quant_bits", default=None
)


@contextmanager
def fake_quant(bits: int | None) -> Generator[None]:
    """Activate per-tensor STE fake-quantization to `bits` for the block; no-op if None."""
    token = _fake_quant_bits.set(bits)
    try:
        yield
    finally:
        _fake_quant_bits.reset(token)


def maybe_fake_quant(t: Tensor) -> Tensor:
    bits = _fake_quant_bits.get()
    if bits is None:
        return t
    return _fake_quant_per_tensor_ste(t, bits)


def _fake_quant_per_tensor_ste(t: Tensor, bits: int) -> Tensor:
    """Symmetric per-tensor quant→dequant with a straight-through (identity) backward."""
    assert bits >= 2, f"fake-quant needs >= 2 bits, got {bits}"
    qmax = 2 ** (bits - 1) - 1
    amax = t.detach().abs().amax()
    if amax == 0:
        return t
    scale = amax / qmax
    q = torch.round(t / scale).clamp(-(qmax + 1), qmax) * scale
    return t + (q - t).detach()
