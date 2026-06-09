import torch

from param_decomp.fake_quant import fake_quant, maybe_fake_quant


def test_noop_outside_context() -> None:
    x = torch.randn(64)
    assert maybe_fake_quant(x) is x


def test_quantizes_inside_context() -> None:
    x = torch.randn(1000)
    with fake_quant(2):
        q = maybe_fake_quant(x)
    # 2-bit symmetric => at most 4 distinct levels; forward values are coarsened.
    assert not torch.equal(q, x)
    assert q.unique().numel() <= 4


def test_straight_through_gradient_is_identity() -> None:
    x = torch.randn(128, requires_grad=True)
    with fake_quant(4):
        maybe_fake_quant(x).sum().backward()
    assert x.grad is not None
    assert torch.equal(x.grad, torch.ones_like(x))


def test_context_restores_on_exit() -> None:
    x = torch.randn(8)
    with fake_quant(8):
        assert not torch.equal(maybe_fake_quant(x), x)
    assert maybe_fake_quant(x) is x
