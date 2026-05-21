"""Tests for ScheduleConfig and get_scheduled_value."""

from typing import Literal

import pytest

from param_decomp.configs import ScheduleConfig
from param_decomp.utils.general_utils import get_scheduled_value


class TestConstantSchedule:
    """Tests for constant schedule."""

    @pytest.mark.parametrize("step", [0, 50, 99])
    def test_constant_returns_start_val(self, step: int):
        """Constant schedule always returns start_val."""
        config = ScheduleConfig(start_val=0.001, fn_type="constant")
        assert get_scheduled_value(step, 100, config) == 0.001


class TestLinearSchedule:
    """Tests for linear schedule."""

    @pytest.mark.parametrize(
        "final_val_frac,expected_start,expected_end",
        [
            (0.0, 1.0, 0.0),  # Decay to 0
            (0.1, 1.0, 0.1),  # Decay to 10%
            (0.5, 1.0, 0.5),  # Decay to 50%
            (1.0, 1.0, 1.0),  # No change
            (2.0, 1.0, 2.0),  # Increase to 200%
        ],
    )
    def test_linear_endpoints(
        self, final_val_frac: float, expected_start: float, expected_end: float
    ):
        """Linear schedule has correct start and end values (as multipliers of start_val)."""
        config = ScheduleConfig(start_val=1.0, fn_type="linear", final_val_frac=final_val_frac)
        total_steps = 100

        start_val = get_scheduled_value(0, total_steps, config)
        end_val = get_scheduled_value(total_steps - 1, total_steps, config)

        assert start_val == pytest.approx(expected_start)
        assert end_val == pytest.approx(expected_end)

    def test_linear_is_monotonic_decay(self):
        """Linear decay should be monotonically decreasing."""
        config = ScheduleConfig(start_val=0.001, fn_type="linear", final_val_frac=0.1)
        total_steps = 100

        values = [get_scheduled_value(step, total_steps, config) for step in range(total_steps)]

        for i in range(1, len(values)):
            assert values[i] <= values[i - 1], f"Not monotonic at step {i}"

    def test_linear_is_monotonic_increase(self):
        """Linear increase should be monotonically increasing."""
        config = ScheduleConfig(start_val=0.001, fn_type="linear", final_val_frac=2.0)
        total_steps = 100

        values = [get_scheduled_value(step, total_steps, config) for step in range(total_steps)]

        for i in range(1, len(values)):
            assert values[i] >= values[i - 1], f"Not monotonic at step {i}"


class TestCosineSchedule:
    """Tests for cosine schedule."""

    @pytest.mark.parametrize(
        "final_val_frac,expected_start,expected_end",
        [
            (0.0, 1.0, 0.0),  # Decay to 0
            (0.1, 1.0, 0.1),  # Decay to 10%
            (0.5, 1.0, 0.5),  # Decay to 50%
            (1.0, 1.0, 1.0),  # No change
            (2.0, 1.0, 2.0),  # Increase to 200%
        ],
    )
    def test_cosine_endpoints(
        self, final_val_frac: float, expected_start: float, expected_end: float
    ):
        """Cosine schedule has correct start and end values (as multipliers of start_val)."""
        config = ScheduleConfig(start_val=1.0, fn_type="cosine", final_val_frac=final_val_frac)
        total_steps = 100

        start_val = get_scheduled_value(0, total_steps, config)
        end_val = get_scheduled_value(total_steps - 1, total_steps, config)

        assert start_val == pytest.approx(expected_start)
        assert end_val == pytest.approx(expected_end)

    def test_cosine_is_monotonic_decay(self):
        """Cosine decay should be monotonically decreasing."""
        config = ScheduleConfig(start_val=0.001, fn_type="cosine", final_val_frac=0.1)
        total_steps = 100

        values = [get_scheduled_value(step, total_steps, config) for step in range(total_steps)]

        for i in range(1, len(values)):
            assert values[i] <= values[i - 1], f"Not monotonic at step {i}"

    def test_cosine_is_half_period(self):
        """Verify cosine uses half-period (smooth start and end)."""
        config = ScheduleConfig(start_val=1.0, fn_type="cosine", final_val_frac=0.0)
        total_steps = 101  # Odd number for clean midpoint

        # At midpoint, half-period cosine should be at 0.5 (halfway between start and end)
        midpoint_val = get_scheduled_value(50, total_steps, config)
        assert midpoint_val == pytest.approx(0.5, rel=0.01)


class TestWarmup:
    """Tests for warmup behavior."""

    @pytest.mark.parametrize("fn_type", ["constant", "linear", "cosine"])
    def test_warmup_starts_at_zero(self, fn_type: Literal["constant", "linear", "cosine"]):
        """With warmup, schedule should start at 0."""
        final_val_frac = 1.0 if fn_type == "constant" else 0.1
        config = ScheduleConfig(
            start_val=0.001, fn_type=fn_type, warmup_pct=0.1, final_val_frac=final_val_frac
        )
        assert get_scheduled_value(0, 100, config) == 0.0

    @pytest.mark.parametrize("fn_type", ["constant", "linear", "cosine"])
    def test_warmup_reaches_start_val(self, fn_type: Literal["constant", "linear", "cosine"]):
        """At end of warmup, schedule should reach start_val."""
        final_val_frac = 1.0 if fn_type == "constant" else 0.1
        config = ScheduleConfig(
            start_val=0.001, fn_type=fn_type, warmup_pct=0.1, final_val_frac=final_val_frac
        )
        # warmup_pct=0.1 with 100 steps means warmup ends at step 10
        warmup_end_val = get_scheduled_value(10, 100, config)
        assert warmup_end_val == pytest.approx(0.001)

    def test_warmup_is_linear(self):
        """Warmup should be linear from 0 to start_val."""
        config = ScheduleConfig(start_val=0.001, fn_type="constant", warmup_pct=0.1)
        total_steps = 100
        warmup_steps = 10

        for step in range(warmup_steps):
            expected = 0.001 * (step / warmup_steps)
            actual = get_scheduled_value(step, total_steps, config)
            assert actual == pytest.approx(expected)

    def test_warmup_then_decay(self):
        """Warmup followed by decay should work correctly."""
        config = ScheduleConfig(
            start_val=0.001, fn_type="cosine", warmup_pct=0.1, final_val_frac=0.1
        )
        total_steps = 100

        # During warmup (steps 0-9): increasing
        warmup_vals = [get_scheduled_value(step, total_steps, config) for step in range(10)]
        for i in range(1, len(warmup_vals)):
            assert warmup_vals[i] > warmup_vals[i - 1]

        # After warmup (steps 10-99): decreasing
        decay_vals = [get_scheduled_value(step, total_steps, config) for step in range(10, 100)]
        for i in range(1, len(decay_vals)):
            assert decay_vals[i] <= decay_vals[i - 1]


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_step(self):
        """Single step should return start_val."""
        config = ScheduleConfig(start_val=0.001, fn_type="cosine", final_val_frac=0.1)
        assert get_scheduled_value(0, 1, config) == 0.001

    def test_two_steps(self):
        """Two steps should have correct start and end."""
        config = ScheduleConfig(start_val=1.0, fn_type="linear", final_val_frac=0.0)
        assert get_scheduled_value(0, 2, config) == pytest.approx(1.0)
        assert get_scheduled_value(1, 2, config) == pytest.approx(0.0)

    def test_full_warmup(self):
        """100% warmup should keep increasing throughout."""
        config = ScheduleConfig(start_val=0.001, fn_type="cosine", warmup_pct=1.0)
        total_steps = 100

        # Last warmup step should be close to start_val
        # (step 99 out of 100 warmup steps = 99% of start_val)
        last_val = get_scheduled_value(99, total_steps, config)
        assert last_val == pytest.approx(0.001 * 0.99)

    def test_no_warmup_no_decay(self):
        """No warmup + constant = always start_val."""
        config = ScheduleConfig(start_val=0.001, fn_type="constant", warmup_pct=0.0)
        for step in [0, 25, 50, 75, 99]:
            assert get_scheduled_value(step, 100, config) == 0.001

    @pytest.mark.parametrize("start_val", [1e-6, 1e-3, 1.0, 100.0])
    def test_various_start_vals(self, start_val: float):
        """Schedule should scale correctly with different start_val."""
        config = ScheduleConfig(start_val=start_val, fn_type="linear", final_val_frac=0.5)
        assert get_scheduled_value(0, 100, config) == pytest.approx(start_val)
        assert get_scheduled_value(99, 100, config) == pytest.approx(start_val * 0.5)
