"""`RunSink`: where `optimize()` sends its output.

Defined here as a runtime-checkable `Protocol` so the core trainer can document
exactly what it requires of the caller without committing to an output format
or any specific dependency (wandb, tqdm, filesystem).

Cadence — when to log/eval/save — lives separately in `param_decomp.configs.Cadence`.
This Protocol describes side effects only: where structured metrics, free-form
console output, and checkpoint state dicts go.

Concrete implementations live with the caller. The in-repo experiments use
``param_decomp_lab.run_sink.RunSink`` (local files + wandb + `is_main_process`
no-op fan-out), but external callers are free to bring their own.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RunSink(Protocol):
    """Side-effect sink for a PD training run.

    The trainer treats this object as opaque: it tells it what happened, never
    *where* the record should go. Callers implement the methods to point at
    whatever output channels they want (local files, wandb, S3, a no-op handle
    on non-main DP ranks, …).
    """

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Record a flat metrics dict at `step`. Keys are already namespaced
        (e.g. `train/loss/total`, `eval/ci_l0/L0`) by the caller."""
        ...

    def console(self, *lines: str) -> None:
        """Emit free-form lines (e.g. tqdm-friendly progress)."""
        ...

    def checkpoint(self, state_dict: dict[str, Any], step: int) -> None:
        """Persist a model state dict at the given step."""
        ...
