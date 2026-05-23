"""Tests for evaluation metrics and figures, particularly CIHistograms."""

from typing import cast
from unittest.mock import Mock

import pytest
import torch

from param_decomp.ci_sigmoids import lower_leaky_hard_sigmoid, upper_leaky_hard_sigmoid
from param_decomp.component_model import CIOutputs, ComponentModel
from param_decomp.metrics.context import MetricContext
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse
from param_decomp_lab.eval_metrics.ci_histograms import CIHistograms, CIHistogramsConfig


def _make_ctx(batch: torch.Tensor, target_out: torch.Tensor, ci: CIOutputs) -> MetricContext:
    return MetricContext(
        model=cast(ComponentModel, Mock(spec=ComponentModel)),
        batch=batch,
        target_out=target_out,
        pre_weight_acts={},
        ci=ci,
        weight_deltas={},
        step=0,
        total_steps=1,
        use_delta_component=False,
        sampling="continuous",
        n_mask_samples=1,
        reconstruction_loss=recon_loss_mse,
        is_eval=True,
    )


class TestCIHistograms:
    """Test suite for CIHistograms class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock ComponentModel."""
        model = Mock(spec=ComponentModel)
        model.module_to_c = {"layer1": 8, "layer2": 8}
        model.components = {"layer1": Mock(), "layer2": Mock()}
        return model

    @pytest.fixture
    def sample_ci(self):
        pre_sigmoid = {
            "layer1": torch.randn(4, 8),
            "layer2": torch.randn(4, 8),
        }
        return CIOutputs(
            lower_leaky={
                "layer1": lower_leaky_hard_sigmoid(pre_sigmoid["layer1"]),
                "layer2": lower_leaky_hard_sigmoid(pre_sigmoid["layer2"]),
            },
            upper_leaky={
                "layer1": upper_leaky_hard_sigmoid(pre_sigmoid["layer1"]),
                "layer2": upper_leaky_hard_sigmoid(pre_sigmoid["layer2"]),
            },
            pre_sigmoid=pre_sigmoid,
        )

    def test_n_batches_accum_enforcement(self, mock_model: Mock, sample_ci: CIOutputs):
        n_batches_accum = 3
        ci_hist = CIHistograms(CIHistogramsConfig(n_batches_accum=n_batches_accum))
        ci_hist.bind(model=mock_model, device="cpu")
        batch = torch.randn(4, 8)
        target_out = torch.randn(4, 8, 100)
        for _ in range(n_batches_accum + 2):
            ci_hist.update(_make_ctx(batch, target_out, sample_ci))
        assert ci_hist.batches_seen == n_batches_accum
        assert len(ci_hist.lower_leaky_causal_importances["layer1"]) == n_batches_accum
        assert len(ci_hist.lower_leaky_causal_importances["layer2"]) == n_batches_accum
        assert len(ci_hist.pre_sigmoid_causal_importances["layer1"]) == n_batches_accum
        assert len(ci_hist.pre_sigmoid_causal_importances["layer2"]) == n_batches_accum

    def test_none_n_batches_accum(self, mock_model: Mock, sample_ci: CIOutputs):
        ci_hist = CIHistograms(CIHistogramsConfig(n_batches_accum=None))
        ci_hist.bind(model=mock_model, device="cpu")
        batch = torch.randn(4, 8)
        target_out = torch.randn(4, 8, 100)
        num_batches = 10
        for _ in range(num_batches):
            ci_hist.update(_make_ctx(batch, target_out, sample_ci))
        assert ci_hist.batches_seen == num_batches
        assert len(ci_hist.lower_leaky_causal_importances["layer1"]) == num_batches
        assert len(ci_hist.lower_leaky_causal_importances["layer2"]) == num_batches
        assert len(ci_hist.pre_sigmoid_causal_importances["layer1"]) == num_batches
        assert len(ci_hist.pre_sigmoid_causal_importances["layer2"]) == num_batches

    def test_empty_compute(self, mock_model: Mock):
        ci_hist = CIHistograms(CIHistogramsConfig(n_batches_accum=None))
        ci_hist.bind(model=mock_model, device="cpu")
        with pytest.raises(RuntimeError, match="No batches seen yet"):
            ci_hist.compute()
