from typing import Any
from unittest.mock import Mock

import pytest
import torch

from param_decomp.component_model import CIOutputs
from param_decomp.metrics.bottleneck_sparsity import (
    BottleneckSparsityLoss,
    BottleneckSparsityLossConfig,
    _bottleneck_sparsity_loss,
)


def _make_ctx(bottleneck_codes: torch.Tensor | None) -> Any:
    ctx = Mock()
    ctx.ci = CIOutputs(
        lower_leaky={},
        upper_leaky={},
        pre_sigmoid={},
        bottleneck_codes=bottleneck_codes,
    )
    return ctx


def _make_metric(pnorm: float = 1.0, eps: float = 0.0) -> BottleneckSparsityLoss:
    metric = BottleneckSparsityLoss(BottleneckSparsityLossConfig(coeff=1.0, pnorm=pnorm, eps=eps))
    metric.bind(model=Mock(), device="cpu")
    return metric


class TestBottleneckSparsityLoss:
    def test_exact_value_p1(self) -> None:
        # p=1, eps=0: loss = mean over batch of sum_d |z_d|
        codes = torch.tensor([[1.0, -2.0, 0.0], [0.0, 0.0, 3.0]])
        loss = _bottleneck_sparsity_loss(codes, pnorm=1.0, eps=0.0)
        assert torch.allclose(loss, torch.tensor(3.0))  # (3 + 3) / 2

    def test_exact_value_sub_one_p(self) -> None:
        codes = torch.tensor([[4.0, 0.0]])
        loss = _bottleneck_sparsity_loss(codes, pnorm=0.5, eps=0.0)
        assert torch.allclose(loss, torch.tensor(2.0))  # 4^0.5 + 0^0.5

    def test_update_returns_live_loss(self) -> None:
        metric = _make_metric()
        codes = torch.tensor([[1.0, -1.0]], requires_grad=True)
        loss = metric.update(_make_ctx(codes))
        assert torch.allclose(loss, torch.tensor(2.0))
        assert loss.requires_grad

    def test_update_asserts_without_codes(self) -> None:
        metric = _make_metric()
        with pytest.raises(AssertionError, match="requires a CI fn with a bottleneck"):
            metric.update(_make_ctx(None))

    def test_compute_keys_and_l0(self) -> None:
        metric = _make_metric()
        # 3 examples (2 + 1 across two batches), 2 dims; dim 1 never fires
        metric.update(_make_ctx(torch.tensor([[1.0, 0.0], [0.0, 0.0]])))
        metric.update(_make_ctx(torch.tensor([[3.0, 0.0]])))
        out = metric.compute()
        assert isinstance(out, dict)
        name = "BottleneckSparsityLoss"
        assert set(out) == {name, f"{name}_code_l0", f"{name}_frac_dims_never_fired"}
        # losses per example: 1, 0, 3 -> mean 4/3
        assert torch.allclose(out[name], torch.tensor(4.0 / 3.0))
        # L0 per example: 1, 0, 1 -> mean 2/3
        assert torch.allclose(out[f"{name}_code_l0"], torch.tensor(2.0 / 3.0))
        assert torch.allclose(out[f"{name}_frac_dims_never_fired"], torch.tensor(0.5))

    def test_warmup_scales_live_loss_not_accumulators(self) -> None:
        cfg = BottleneckSparsityLossConfig(coeff=1.0, pnorm=1.0, eps=0.0, warmup_end_frac=0.5)
        metric = BottleneckSparsityLoss(cfg)
        metric.bind(model=Mock(), device="cpu")
        codes = torch.tensor([[1.0, -1.0]])

        ctx = _make_ctx(codes)
        ctx.current_frac_of_training = 0.25  # halfway through warmup -> scale 0.5
        assert torch.allclose(metric.update(ctx), torch.tensor(1.0))

        ctx.current_frac_of_training = 0.75  # past warmup -> full strength
        assert torch.allclose(metric.update(ctx), torch.tensor(2.0))

        # Accumulated eval loss is unscaled: mean over both updates is 2.0
        out = metric.compute()
        assert isinstance(out, dict)
        assert torch.allclose(out["BottleneckSparsityLoss"], torch.tensor(2.0))

    def test_seq_dim_treated_as_examples(self) -> None:
        metric = _make_metric()
        codes = torch.zeros(2, 3, 4)
        codes[0, 0, 0] = 5.0
        loss = metric.update(_make_ctx(codes))
        # 6 (batch*seq) examples, total |z| = 5 -> mean 5/6
        assert torch.allclose(loss, torch.tensor(5.0 / 6.0))
