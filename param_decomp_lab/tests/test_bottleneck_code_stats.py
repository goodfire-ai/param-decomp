from typing import Any
from unittest.mock import Mock

import torch

from param_decomp.ci_fns import (
    CiBottleneckConfig,
    GlobalCiFnWrapper,
    GlobalSharedTransformerCiFn,
    TargetLayerConfig,
)
from param_decomp.component_model import CIOutputs
from param_decomp_lab.eval_metrics.bottleneck_code_stats import (
    BottleneckCodeHistograms,
    BottleneckCodeHistogramsConfig,
    BottleneckCodeStats,
    BottleneckCodeStatsConfig,
)

THETA_INIT = 0.05
BANDWIDTH = 0.05
D = 4


def _mock_model_with_bottleneck() -> Mock:
    ci_fn = GlobalSharedTransformerCiFn(
        target_model_layer_configs={"layer_a": TargetLayerConfig(input_dim=8, C=5)},
        d_model=16,
        n_layers=1,
        n_heads=2,
        max_len=8,
        bottleneck_config=CiBottleneckConfig(
            bottleneck_dim=D, decoder_hidden_dims=[], theta_init=THETA_INIT, bandwidth=BANDWIDTH
        ),
    )
    model = Mock()
    model.ci_fn = GlobalCiFnWrapper(global_ci_fn=ci_fn, components={})
    return model


def _make_ctx(codes: torch.Tensor) -> Any:
    ctx = Mock()
    ctx.ci = CIOutputs(lower_leaky={}, upper_leaky={}, pre_sigmoid={}, bottleneck_codes=codes)
    return ctx


class TestBottleneckCodeStats:
    def test_scalar_stats(self) -> None:
        metric = BottleneckCodeStats(BottleneckCodeStatsConfig())
        metric.bind(model=_mock_model_with_bottleneck(), device="cpu")

        # 2 examples, 4 dims; dim 3 never fires. One active value (0.06) is within
        # theta + bandwidth/2 = 0.075, the others (1.0, 2.0) are not.
        codes = torch.tensor([[1.0, 0.0, 0.06, 0.0], [-2.0, 0.0, 0.0, 0.0]])
        metric.update(_make_ctx(codes))
        out = metric.compute()
        assert isinstance(out, dict)

        assert torch.allclose(out["code_l0"], torch.tensor(1.5))
        assert torch.allclose(out["frac_dims_dead"], torch.tensor(0.5))
        assert torch.allclose(out["mean_active_magnitude"], torch.tensor((1.0 + 0.06 + 2.0) / 3))
        assert torch.allclose(out["frac_active_in_theta_grad_window"], torch.tensor(1 / 3))
        assert torch.allclose(out["theta_mean"], torch.tensor(THETA_INIT))


class TestBottleneckCodeHistograms:
    def test_returns_figure(self) -> None:
        metric = BottleneckCodeHistograms(BottleneckCodeHistogramsConfig(n_batches_accum=None))
        metric.bind(model=_mock_model_with_bottleneck(), device="cpu")
        codes = torch.randn(8, D) * (torch.rand(8, D) > 0.5)
        metric.update(_make_ctx(codes))
        out = metric.compute()
        assert isinstance(out, dict)
        assert "bottleneck_code" in out
