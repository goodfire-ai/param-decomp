"""The `OnePoolRunSink` Protocol — the trainer's side-effect boundary.

The concrete implementation lives in `param_decomp_lab.run_sink`
(`OnePoolSink`): local-files / wandb / console plumbing plus a typed
`checkpoint` that delegates to `_persist`. (The pool-suffixed names leave
room for the n-pool subsystems' sibling sinks, which live on their own
branch.)

Timing — when the trainer emits — lives separately: `param_decomp_config.pd.Cadence`
owns train-log + checkpoint periods, and `param_decomp.train_step.EvalLoop` owns
the eval period.
"""

from typing import Any, Protocol, runtime_checkable

from param_decomp.training_state import TrainingState


@runtime_checkable
class OnePoolRunSink(Protocol):
    """Side-effect sink for a 1-pool training run."""

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Record a flat metrics dict at ``step``."""
        ...

    def console(self, *lines: str) -> None:
        """Emit free-form lines (e.g. tqdm-friendly progress)."""
        ...

    def checkpoint(self, snapshot: TrainingState, *, final: bool) -> None:
        """Persist a 1-pool training state.

        The lab sink writes ``snapshot.component_model`` to
        ``model_<step>.pth`` (for downstream tools) and the whole ``snapshot``
        to ``training_<step>.pth`` (for resumption). ``final=True`` marks the
        end-of-training save — the lab sink uploads ``model_<step>.pth`` to
        wandb at that point (but never the multi-GB ``training_<step>.pth``).
        """
        ...

    def finish(self) -> None:
        """End-of-run cleanup (close handles, finish wandb run, etc.)."""
        ...


RunSink = OnePoolRunSink
