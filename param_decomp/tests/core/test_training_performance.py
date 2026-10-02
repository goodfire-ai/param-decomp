"""Utilization windows retain matching work and time across training interruptions."""

import math

import pytest

from param_decomp.core.flops.types import UsefulFlops
from param_decomp.core.training_performance import MfuAccounting, PerformanceTracker


def test_evaluation_and_checkpoints_count_only_against_overall_mfu() -> None:
    work = (UsefulFlops(100.0, 20.0), UsefulFlops(200.0, 40.0))
    tracker = PerformanceTracker(10.0, MfuAccounting(lambda step: work[step], 100.0))
    tracker.training_started(10.0)
    tracker.record_step(0)
    tracker.training_paused(12.0)
    tracker.training_started(17.0)
    tracker.record_step(1)
    tracker.training_paused(20.0)

    metrics = tracker.metrics(25.0)
    assert metrics["train/perf/runtime_s"] == 15.0
    assert metrics["train/perf/training_time_s"] == 5.0
    assert metrics["train/perf/step_time_s"] == 2.5
    assert metrics["train/perf/overall_mfu_without_optimizer"] == pytest.approx(0.2)
    assert metrics["train/perf/overall_mfu_with_optimizer"] == pytest.approx(0.24)
    assert metrics["train/perf/step_mfu_without_optimizer"] == pytest.approx(0.6)
    assert metrics["train/perf/step_mfu_with_optimizer"] == pytest.approx(0.72)


def test_log_window_reset_preserves_attempt_totals() -> None:
    work = (UsefulFlops(100.0, 50.0), UsefulFlops(300.0, 75.0))
    tracker = PerformanceTracker(0.0, MfuAccounting(lambda step: work[step], 100.0))
    tracker.training_started(0.0)
    tracker.record_step(0)
    tracker.training_paused(2.0)
    tracker.metrics(3.0)
    tracker.reset_window(4.0)
    tracker.training_started(4.0)
    tracker.record_step(1)
    tracker.training_paused(5.0)

    metrics = tracker.metrics(5.0)
    assert metrics["train/perf/useful_model_flops"] == 400.0
    assert metrics["train/perf/useful_optimizer_flops"] == 125.0
    assert metrics["train/perf/useful_flops"] == 525.0
    assert metrics["train/perf/n_completed_steps"] == 2.0
    assert metrics["train/perf/n_window_steps"] == 1.0
    assert metrics["train/perf/training_time_s"] == 3.0
    assert metrics["train/perf/window_training_time_s"] == 1.0
    assert metrics["train/perf/overall_mfu_without_optimizer"] == 0.8
    assert metrics["train/perf/step_mfu_without_optimizer"] == 3.0


def test_new_tracker_starts_with_empty_work_and_time() -> None:
    tracker = PerformanceTracker(
        500.0, MfuAccounting(lambda _step: UsefulFlops(100.0, 10.0), 100.0)
    )
    tracker.training_started(500.0)
    tracker.record_step(0)
    tracker.training_paused(501.0)
    metrics = tracker.metrics(502.0)
    assert metrics["train/perf/n_completed_steps"] == 1.0
    assert metrics["train/perf/useful_model_flops"] == 100.0
    assert metrics["train/perf/runtime_s"] == 2.0
    assert metrics["train/perf/overall_mfu_without_optimizer"] == 0.5


def test_final_checkpoint_refreshes_overall_without_changing_window_mfu() -> None:
    tracker = PerformanceTracker(0.0, MfuAccounting(lambda _step: UsefulFlops(100.0, 10.0), 100.0))
    tracker.training_started(0.0)
    tracker.record_step(0)
    tracker.training_paused(1.0)
    before_save = tracker.metrics(1.0)
    after_save = tracker.metrics(5.0)
    assert (
        before_save["train/perf/step_mfu_without_optimizer"]
        == after_save["train/perf/step_mfu_without_optimizer"]
    )
    assert after_save["train/perf/overall_mfu_without_optimizer"] == 0.2


def test_disabled_accounting_reports_only_timing() -> None:
    tracker = PerformanceTracker(0.0, None)
    tracker.training_started(0.0)
    tracker.record_step(0)
    tracker.training_paused(1.0)
    metrics = tracker.metrics(1.0)
    assert metrics["train/perf/n_completed_steps"] == 1
    assert metrics["train/perf/step_time_s"] == 1
    assert not any("mfu" in name or "flops" in name for name in metrics)


def test_empty_window_has_no_step_average() -> None:
    tracker = PerformanceTracker(0.0, MfuAccounting(lambda _step: UsefulFlops(100.0, 10.0), 100.0))
    metrics = tracker.metrics(0.0)
    assert metrics["train/perf/n_completed_steps"] == 0.0
    assert "train/perf/step_time_s" not in metrics
    assert "train/perf/step_mfu_without_optimizer" not in metrics
    assert all(math.isfinite(value) for value in metrics.values())


def test_training_intervals_must_alternate_and_measurement_requires_a_pause() -> None:
    tracker = PerformanceTracker(0.0, MfuAccounting(lambda _step: UsefulFlops(100.0, 10.0), 100.0))
    with pytest.raises(ValueError, match="already paused"):
        tracker.training_paused(1.0)
    with pytest.raises(ValueError, match="during training"):
        tracker.record_step(0)
    tracker.training_started(1.0)
    with pytest.raises(ValueError, match="already running"):
        tracker.training_started(2.0)
    with pytest.raises(ValueError, match="Pause training"):
        tracker.metrics(2.0)
    with pytest.raises(ValueError, match="Pause training"):
        tracker.reset_window(2.0)
    tracker.training_paused(2.0)


@pytest.mark.parametrize("invalid", [-1.0, math.nan, math.inf])
def test_invalid_timestamps_are_rejected(invalid: float) -> None:
    tracker = PerformanceTracker(0.0, MfuAccounting(lambda _step: UsefulFlops(100.0, 10.0), 100.0))
    with pytest.raises(ValueError, match="monotonic"):
        tracker.training_started(invalid)


@pytest.mark.parametrize("invalid", [0.0, -1.0, math.nan, math.inf])
def test_invalid_peak_capacity_is_rejected(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        MfuAccounting(lambda _step: UsefulFlops(100.0, 10.0), invalid)
