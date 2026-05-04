"""Tests for evaluation metrics and figures, particularly CIHistograms."""

from unittest.mock import Mock

import pytest
import torch

from param_decomp.configs import Config
from param_decomp.metrics import CIHistograms
from param_decomp.models.component_model import CIOutputs, ComponentModel
from param_decomp.models.sigmoids import lower_leaky_hard_sigmoid, upper_leaky_hard_sigmoid


class TestCIHistograms:
    """Test suite for CIHistograms class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        config = Mock(spec=Config)
        config.ci_alive_threshold = 0.5
        config.sigmoid_type = "straight_through"
        return config

    @pytest.fixture
    def mock_model(self):
        """Create a mock ComponentModel."""
        model = Mock(spec=ComponentModel)
        model.module_to_c = {"layer1": 8, "layer2": 8}
        model.components = {"layer1": Mock(), "layer2": Mock()}
        return model

    @pytest.fixture
    def sample_ci(self):
        """Create sample causal importance tensors."""

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
        """Test that CIHistograms stops accumulating after n_batches_accum."""
        n_batches_accum = 3
        ci_hist = CIHistograms(mock_model, n_batches_accum=n_batches_accum)

        # Create dummy batch and target_out
        batch = torch.randn(4, 8)
        target_out = torch.randn(4, 8, 100)

        # Watch more batches than n_batches_accum
        for _ in range(n_batches_accum + 2):
            ci_hist.update(
                batch=batch,
                target_out=target_out,
                ci=sample_ci,
            )

        # Check that only n_batches_accum were accumulated
        assert ci_hist.batches_seen == n_batches_accum
        assert len(ci_hist.lower_leaky_causal_importances["layer1"]) == n_batches_accum
        assert len(ci_hist.lower_leaky_causal_importances["layer2"]) == n_batches_accum
        assert len(ci_hist.pre_sigmoid_causal_importances["layer1"]) == n_batches_accum
        assert len(ci_hist.pre_sigmoid_causal_importances["layer2"]) == n_batches_accum

    def test_none_n_batches_accum(self, mock_model: Mock, sample_ci: CIOutputs):
        """Test unlimited batch accumulation when n_batches_accum is None."""
        ci_hist = CIHistograms(mock_model, n_batches_accum=None)

        batch = torch.randn(4, 8)
        target_out = torch.randn(4, 8, 100)

        # Watch many batches
        num_batches = 10
        for _ in range(num_batches):
            ci_hist.update(
                batch=batch,
                target_out=target_out,
                ci=sample_ci,
            )

        # All batches should be accumulated
        assert ci_hist.batches_seen == num_batches
        assert len(ci_hist.lower_leaky_causal_importances["layer1"]) == num_batches
        assert len(ci_hist.lower_leaky_causal_importances["layer2"]) == num_batches
        assert len(ci_hist.pre_sigmoid_causal_importances["layer1"]) == num_batches
        assert len(ci_hist.pre_sigmoid_causal_importances["layer2"]) == num_batches

    def test_empty_compute(self, mock_model: Mock):
        """Test compute() when no batches have been updated."""

        ci_hist = CIHistograms(mock_model)

        # When no batches watched, compute will raise a RuntimeError
        with pytest.raises(RuntimeError, match="No batches seen yet"):
            ci_hist.compute()
