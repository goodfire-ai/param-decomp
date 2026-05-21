"""`RunSink`: the contract `optimize()` uses for cadence + side effects.

Defined here as a runtime-checkable `Protocol` so the core trainer can document
exactly what it requires of the caller without committing to an output format,
a cadence policy, or any specific dependency (wandb, tqdm, filesystem).

Concrete implementations live with the caller. The in-repo experiments use
``param_decomp_lab.run_sink.RunSink`` (local files + wandb + `is_main_process`
no-op fan-out), but external callers are free to bring their own.

Methods the trainer calls:

* Cadence gates: ``should_log_train``, ``should_eval``, ``should_run_slow_eval``,
  ``should_save``.
* Side effects: ``log`` (flat metrics dict), ``console`` (free-form lines),
  ``checkpoint`` (state dict at step).
* Eval shape: ``n_eval_steps`` (how many eval batches per eval call).
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RunSink(Protocol):
    """Side-effect sink + cadence contract for a PD training run.

    The trainer treats this object as opaque: it asks when to act and how to
    record results, never *where* the results go. Callers implement the methods
    to point at whatever output channels they want (local files, wandb, S3, a
    no-op handle on non-main DP ranks, …).
    """

    @property
    def n_eval_steps(self) -> int:
        """Number of eval batches to run per `should_eval(step)` tick."""
        ...

    def should_log_train(self, step: int) -> bool: ...

    def should_eval(self, step: int) -> bool: ...

    def should_run_slow_eval(self, step: int) -> bool: ...

    def should_save(self, step: int, *, total_steps: int) -> bool: ...

    def log(
        self,
        metrics: dict[str, Any],
        *,
        step: int,
        section: str | None = None,
    ) -> None:
        """Record a flat metrics dict. `section` is an optional W&B-style prefix."""
        ...

    def console(self, *lines: str) -> None:
        """Emit free-form lines (e.g. tqdm-friendly progress)."""
        ...

    def checkpoint(self, state_dict: dict[str, Any], *, step: int) -> None:
        """Persist a model state dict at the given step."""
        ...
