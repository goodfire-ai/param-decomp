"""`RunSink` Protocol: where `optimize()` sends its output (metrics, console lines, checkpoints)."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RunSink(Protocol):
    """Side-effect sink for a PD training run.

    The trainer treats this object as opaque: it reports what happened, never *where*
    the record should go. Callers point the methods at whatever output channels they
    want (local files, wandb, S3, a no-op handle on non-main DP ranks, ...).
    """

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Record a flat metrics dict at `step`.

        Args:
            metrics: Flat dict whose keys are already namespaced (e.g.
                `"train/loss/total"`, `"eval/ci_l0/L0"`) by the trainer. Values may be
                scalars, PIL images, or other artefact types the concrete sink supports.
            step: Training step at which the values were measured.
        """
        ...

    def console(self, *lines: str) -> None:
        """Emit free-form lines (e.g. tqdm-friendly progress)."""
        ...

    def checkpoint(self, state_dict: dict[str, Any], step: int) -> None:
        """Persist a model state dict at `step`.

        Args:
            state_dict: Tensor state dict to serialise.
            step: Training step used in the checkpoint identifier.
        """
        ...

    def finish(self) -> None:
        """End-of-run cleanup (close handles, finish wandb run, etc.)."""
        ...
