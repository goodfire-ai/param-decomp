"""Tests for linear annealing of loss coefficients."""

import pytest

from spd.configs import FaithfulnessLossConfig
from spd.utils.annealing import linearly_anneal_value


class TestLinearlyAnnealValue:
    def test_no_annealing_when_final_is_none(self):
        assert linearly_anneal_value(0.5, 1.0, 0.0, None, 1.0) == 1.0

    def test_no_annealing_when_start_frac_is_one(self):
        assert linearly_anneal_value(0.5, 1.0, 1.0, 0.0, 1.0) == 1.0

    def test_returns_initial_before_start(self):
        assert linearly_anneal_value(0.1, 1.0, 0.25, 0.0, 0.75) == 1.0

    def test_returns_final_at_end(self):
        assert linearly_anneal_value(0.75, 1.0, 0.25, 0.0, 0.75) == 0.0

    def test_returns_final_after_end(self):
        assert linearly_anneal_value(0.9, 1.0, 0.25, 0.0, 0.75) == 0.0

    def test_midpoint_interpolation(self):
        assert linearly_anneal_value(0.5, 1.0, 0.0, 0.0, 1.0) == pytest.approx(0.5)

    def test_interpolation_within_subrange(self):
        # start=0.25, end=0.75, halfway = 0.5 -> midway between initial and final
        assert linearly_anneal_value(0.5, 2.0, 0.25, 0.0, 0.75) == pytest.approx(1.0)

    def test_supports_increasing_values(self):
        assert linearly_anneal_value(0.5, 0.0, 0.0, 1.0, 1.0) == pytest.approx(0.5)

    def test_end_before_start_raises(self):
        with pytest.raises(AssertionError):
            linearly_anneal_value(0.5, 1.0, 0.5, 0.0, 0.25)


class TestLossMetricConfigGetCoeff:
    def test_default_no_annealing(self):
        cfg = FaithfulnessLossConfig(coeff=10.0)
        assert cfg.get_coeff(0.0) == 10.0
        assert cfg.get_coeff(0.5) == 10.0
        assert cfg.get_coeff(1.0) == 10.0

    def test_linear_anneal_to_zero_over_first_half(self):
        cfg = FaithfulnessLossConfig(
            coeff=1.0,
            coeff_anneal_start_frac=0.0,
            coeff_anneal_final_coeff=0.0,
            coeff_anneal_end_frac=0.5,
        )
        assert cfg.get_coeff(0.0) == pytest.approx(1.0)
        assert cfg.get_coeff(0.25) == pytest.approx(0.5)
        assert cfg.get_coeff(0.5) == pytest.approx(0.0)
        assert cfg.get_coeff(0.75) == pytest.approx(0.0)
        assert cfg.get_coeff(1.0) == pytest.approx(0.0)

    def test_get_coeff_requires_coeff(self):
        cfg = FaithfulnessLossConfig()
        with pytest.raises(AssertionError):
            cfg.get_coeff(0.5)
