"""Loop-wide useful work and training windows on an explicit monotonic clock.

The caller synchronizes queued training before pausing. Evaluation, checkpointing and
logging keep the overall clock running while training is paused. The loop starts after
initialization and compilation.
A new tracker starts a new attempt, including when it restores a checkpoint.
"""

from dataclasses import dataclass
from math import isfinite

from param_decomp.core.dict_utils import dict_safe_update_
from param_decomp.core.flops.types import StepFlops


@dataclass(frozen=True)
class MfuAccounting:
    step_flops: StepFlops
    peak_flops_per_second: float

    def __post_init__(self) -> None:
        if not isfinite(self.peak_flops_per_second) or self.peak_flops_per_second <= 0:
            raise ValueError("Fleet peak FLOPs per second must be finite and positive")


@dataclass(frozen=True)
class _Paused:
    pass


@dataclass(frozen=True)
class _Training:
    started_at: float


class PerformanceTracker:
    """Count only this attempt's work, keeping wall time and training time separate."""

    def __init__(
        self,
        loop_started_at: float,
        accounting: MfuAccounting | None,
    ) -> None:
        if not isfinite(loop_started_at):
            raise ValueError("Loop start must be a finite monotonic timestamp")
        self._loop_started_at = loop_started_at
        self._accounting = accounting
        self._n_completed_steps = 0
        self._last_timestamp = loop_started_at
        self._phase: _Paused | _Training = _Paused()
        self._training_time = 0.0
        self._window_training_time = 0.0
        self._n_window_steps = 0
        self._model_flops = 0.0
        self._optimizer_flops = 0.0
        self._window_model_flops = 0.0
        self._window_optimizer_flops = 0.0

    def _check_timestamp(self, at: float) -> None:
        if not isfinite(at) or at < self._last_timestamp:
            raise ValueError("Performance timestamps must be finite and monotonic")

    def training_started(self, at: float) -> None:
        self._check_timestamp(at)
        match self._phase:
            case _Paused():
                self._phase = _Training(at)
                self._last_timestamp = at
            case _Training():
                raise ValueError("Training is already running")

    def training_paused(self, at: float) -> None:
        self._check_timestamp(at)
        match self._phase:
            case _Paused():
                raise ValueError("Training is already paused")
            case _Training(started_at=started_at):
                duration = at - started_at
                self._training_time += duration
                self._window_training_time += duration
                self._phase = _Paused()
                self._last_timestamp = at

    def record_step(self, step: int) -> None:
        """Record a completed step, including useful work when accounting is enabled."""
        match self._phase:
            case _Paused():
                raise ValueError("A training step must be recorded during training")
            case _Training():
                pass
        self._n_completed_steps += 1
        self._n_window_steps += 1
        if self._accounting is not None:
            work = self._accounting.step_flops(step)
            self._model_flops += work.model
            self._optimizer_flops += work.optimizer
            self._window_model_flops += work.model
            self._window_optimizer_flops += work.optimizer

    def reset_window(self, now: float) -> None:
        """Start a fresh main-step window after a training log."""
        self._check_timestamp(now)
        match self._phase:
            case _Training():
                raise ValueError("Pause training before resetting its window")
            case _Paused():
                pass
        self._window_training_time = 0.0
        self._n_window_steps = 0
        self._window_model_flops = 0.0
        self._window_optimizer_flops = 0.0
        self._last_timestamp = now

    def metrics(self, now: float) -> dict[str, float]:
        """Snapshot synchronized work; paused overhead changes only overall utilization."""
        self._check_timestamp(now)
        match self._phase:
            case _Training():
                raise ValueError("Pause training before measuring its completed work")
            case _Paused():
                pass
        runtime = now - self._loop_started_at
        if runtime == 0 and self._model_flops + self._optimizer_flops > 0:
            raise ValueError("Completed work requires positive elapsed time")
        if self._n_window_steps and self._window_training_time <= 0:
            raise ValueError("Completed steps require positive training time")
        self._last_timestamp = now
        record = {
            "runtime_s": runtime,
            "training_time_s": self._training_time,
            "n_completed_steps": float(self._n_completed_steps),
            "window_training_time_s": self._window_training_time,
            "n_window_steps": float(self._n_window_steps),
        }
        if self._n_window_steps:
            record["step_time_s"] = self._window_training_time / self._n_window_steps
        if self._accounting is None:
            return {f"train/perf/{name}": value for name, value in record.items()}
        dict_safe_update_(
            record,
            {
                "useful_model_flops": self._model_flops,
                "useful_optimizer_flops": self._optimizer_flops,
                "useful_flops": self._model_flops + self._optimizer_flops,
            },
        )
        peak = self._accounting.peak_flops_per_second
        if runtime > 0:
            record["overall_mfu_without_optimizer"] = self._model_flops / runtime / peak
            record["overall_mfu_with_optimizer"] = (
                (self._model_flops + self._optimizer_flops) / runtime / peak
            )
        if self._n_window_steps:
            record["step_mfu_without_optimizer"] = (
                self._window_model_flops / self._window_training_time / peak
            )
            record["step_mfu_with_optimizer"] = (
                (self._window_model_flops + self._window_optimizer_flops)
                / self._window_training_time
                / peak
            )
        return {f"train/perf/{name}": value for name, value in record.items()}
