"""Real fp8 (e4m3) matmul for the PPGD warmup on Hopper+ (H100), via `torch._scaled_mm`.

`fp8_warmup(enabled)` activates fp8 for the duration of the warmup loop; while active, the
`LinearComponents` matmuls route through `maybe_fp8_linear`, a per-tensor-scaled e4m3
`_scaled_mm` with an **fp8 forward and a bf16 backward** (the source gradient is computed in
bf16). Falls back to a plain `x @ w` when disabled, off-CUDA, or when a matmul dim is not a
multiple of 16 (`_scaled_mm` requires the contraction dim, at least, to be 16-aligned).
"""

import contextvars
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0

_warmup_fp8: contextvars.ContextVar[bool] = contextvars.ContextVar("warmup_fp8", default=False)


@contextmanager
def fp8_warmup(enabled: bool) -> Generator[None]:
    token = _warmup_fp8.set(enabled)
    try:
        yield
    finally:
        _warmup_fp8.reset(token)


def _to_e4m3(t: Tensor) -> tuple[Tensor, Tensor]:
    """Per-tensor symmetric cast to e4m3; returns `(fp8_tensor, dequant_scale)`."""
    scale = (t.detach().abs().amax().float() / _FP8_MAX).clamp(min=1e-12)
    return (t.float() / scale).to(_FP8), scale


class _Fp8Linear(Function):
    """`x @ w` with an fp8-e4m3 `_scaled_mm` forward and a bf16 backward."""

    @override
    @staticmethod
    def forward(ctx: Any, x: Tensor, w: Tensor) -> Tensor:  # x: [M, K], w: [K, N]
        ctx.save_for_backward(x, w)
        x8, sx = _to_e4m3(x)
        w8, sw = _to_e4m3(w)
        return torch._scaled_mm(
            x8.contiguous(),
            w8.t().contiguous().t(),  # _scaled_mm wants mat2 column-major
            scale_a=sx,
            scale_b=sw,
            out_dtype=x.dtype,
        )

    @override
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor | None, Tensor | None]:
        grad = grad_outputs[0]
        x, w = ctx.saved_tensors
        gx = grad @ w.t() if ctx.needs_input_grad[0] else None
        gw = x.t() @ grad if ctx.needs_input_grad[1] else None
        return gx, gw


def maybe_fp8_linear(x: Float[Tensor, "... k"], w: Float[Tensor, "k n"]) -> Float[Tensor, "... n"]:
    """`x @ w`; under an active `fp8_warmup` (and 16-aligned shapes on CUDA) the forward is fp8."""
    k, n = w.shape
    m = x.numel() // k if k else 0
    if not _warmup_fp8.get() or not x.is_cuda or k % 16 or n % 16 or m % 16:
        return x @ w
    out: Tensor = _Fp8Linear.apply(x.reshape(m, k), w)  # pyright: ignore[reportAssignmentType]
    return out.reshape(*x.shape[:-1], n)
