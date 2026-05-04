from typing import override

import pytest
import torch
from torch import Tensor, nn

from param_decomp.models.batch_and_loss_fns import make_run_batch, recon_loss_kl, recon_loss_mse


class _TensorModel(nn.Module):
    @override
    def forward(self, x: Tensor) -> Tensor:
        return x * 2


class _TupleModel(nn.Module):
    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return x * 2, x * 3


class _LogitsModel(nn.Module):
    """Model returning a HuggingFace-style object with a `.logits` attribute."""

    class _Output:
        def __init__(self, logits: Tensor) -> None:
            self.logits = logits

    @override
    def forward(self, x: Tensor) -> "_LogitsModel._Output":
        return _LogitsModel._Output(logits=x * 5)


def test_make_run_batch_none_passthrough() -> None:
    run_batch = make_run_batch(output_extract=None)
    model = _TensorModel()
    batch = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    out = run_batch(model, batch)
    assert torch.equal(out, batch * 2)


@pytest.mark.parametrize("idx", [0, 1])
def test_make_run_batch_int_indexes_tuple(idx: int) -> None:
    run_batch = make_run_batch(output_extract=idx)
    model = _TupleModel()
    batch = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    out = run_batch(model, batch)
    expected = batch * (2 if idx == 0 else 3)
    assert torch.equal(out, expected)


def test_make_run_batch_str_gets_attr() -> None:
    run_batch = make_run_batch(output_extract="logits")
    model = _LogitsModel()
    batch = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    out = run_batch(model, batch)
    assert torch.equal(out, batch * 5)


def test_recon_loss_mse_shape_mismatch_asserts() -> None:
    pred = torch.zeros(2, 3)
    target = torch.zeros(2, 4)
    with pytest.raises(AssertionError):
        recon_loss_mse(pred=pred, target=target)


def test_recon_loss_kl_identical_logits_is_zero() -> None:
    logits = torch.randn(2, 4, 7)
    sum_kl, n_positions = recon_loss_kl(pred=logits, target=logits)
    assert n_positions == 2 * 4
    assert torch.isclose(sum_kl, torch.tensor(0.0), atol=1e-6)


def test_recon_loss_kl_matches_manual_computation() -> None:
    torch.manual_seed(0)
    pred = torch.randn(3, 5, 8)
    target = torch.randn(3, 5, 8)

    log_q = torch.log_softmax(pred, dim=-1)
    log_p = torch.log_softmax(target, dim=-1)
    p = torch.softmax(target, dim=-1)
    expected_per_position = (p * (log_p - log_q)).sum(dim=-1)
    expected_sum = expected_per_position.sum()

    sum_kl, n_positions = recon_loss_kl(pred=pred, target=target)

    assert n_positions == 3 * 5
    assert torch.isclose(sum_kl, expected_sum, atol=1e-5)


def test_recon_loss_kl_n_positions_counts_all_leading_dims() -> None:
    for shape in [(10, 7), (3, 4, 7), (2, 3, 5, 7)]:
        pred = torch.randn(*shape)
        target = torch.randn(*shape)
        _, n_positions = recon_loss_kl(pred=pred, target=target)
        expected = 1
        for d in shape[:-1]:
            expected *= d
        assert n_positions == expected
