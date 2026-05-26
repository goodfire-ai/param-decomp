"""Per-pool `RunSink` Protocols.

Each pool's trainer accepts a typed sink: 1-pool's `Trainer` takes
`OnePoolRunSink`, 3-pool's `ThreePoolTrainer` takes `ThreePoolRunSink`. The
only difference between the two protocols is the `checkpoint` parameter's
state type. `log`, `console`, and `finish` are identical.

Concrete implementations live in `param_decomp_lab.run_sink` (`OnePoolSink`,
`ThreePoolSink`) — they share a private base for the local-files / wandb /
console plumbing, then add typed `checkpoint` methods that delegate to a
shared `_persist`.

Timing — when the trainer emits — lives separately: `param_decomp.configs.Cadence`
owns train-log + checkpoint periods, and `param_decomp.optimize.EvalLoop` owns
the eval period.
"""

from typing import Any, Protocol, runtime_checkable

from param_decomp.training_state import ThreePoolTrainingState, TrainingState


@runtime_checkable
class OnePoolRunSink(Protocol):
    """Side-effect sink for a 1-pool training run."""

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Record a flat metrics dict at ``step``."""
        ...

    def console(self, *lines: str) -> None:
        """Emit free-form lines (e.g. tqdm-friendly progress)."""
        ...

    def checkpoint(self, snapshot: TrainingState) -> None:
        """Persist a 1-pool training state.

        The lab sink writes ``snapshot.component_model`` to
        ``model_<step>.pth`` (for downstream tools) and the whole ``snapshot``
        to ``training_<step>.pth`` (for resumption).
        """
        ...

    def finish(self) -> None:
        """End-of-run cleanup (close handles, finish wandb run, etc.)."""
        ...


@runtime_checkable
class ThreePoolRunSink(Protocol):
    """Side-effect sink for a 3-pool training run.

    Identical to `OnePoolRunSink` apart from the checkpoint parameter's state
    type. Two separate protocols rather than a union so each trainer's
    ``run()`` signature can only accept a sink wired to its own pool's state.
    """

    def log(self, metrics: dict[str, Any], step: int) -> None: ...
    def console(self, *lines: str) -> None: ...
    def checkpoint(self, snapshot: ThreePoolTrainingState) -> None: ...
    def finish(self) -> None: ...


# Convenience alias for the few call sites that don't care which pool the sink
# serves (e.g. 2-pool's `run()` while resumption stays unported there).
RunSink = OnePoolRunSink | ThreePoolRunSink
