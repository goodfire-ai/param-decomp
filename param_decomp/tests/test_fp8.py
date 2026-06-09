import pytest
import torch

from param_decomp.fp8 import fp8_warmup, maybe_fp8_linear

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="fp8 _scaled_mm is CUDA-only")


def test_noop_outside_context() -> None:
    x = torch.randn(16, 32)
    w = torch.randn(32, 48)
    # Outside the context it must equal a plain matmul exactly.
    assert torch.equal(maybe_fp8_linear(x, w), x @ w)


@cuda
def test_fp8_forward_close_to_bf16() -> None:
    x = torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(2048, 3072, device="cuda", dtype=torch.bfloat16)
    ref = x.float() @ w.float()
    with fp8_warmup(True):
        out = maybe_fp8_linear(x, w)
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert out.dtype == torch.bfloat16
    assert rel < 0.1, f"fp8 rel err too high: {rel}"


@cuda
def test_fp8_backward_flows() -> None:
    x = torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(2048, 3072, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with fp8_warmup(True):
        maybe_fp8_linear(x, w).sum().backward()
    assert x.grad is not None and w.grad is not None
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()


@cuda
def test_falls_back_when_not_16_aligned() -> None:
    # Contraction dim not %16 -> must fall back to bf16 matmul (no _scaled_mm error).
    x = torch.randn(64, 2050, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(2050, 3072, device="cuda", dtype=torch.bfloat16)
    with fp8_warmup(True):
        out = maybe_fp8_linear(x, w)
    assert torch.equal(out, x @ w)
