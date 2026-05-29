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


@runtime_checkable
class ThreePoolRunSink(Protocol):
    """Side-effect sink for a 3-pool training run.

    Unlike `OnePoolRunSink`, the 3-pool sink does NOT persist a training state
    from the train loop. The trainer writes self-contained per-rank partials to
    a shared-FS scratch dir (cheap, no rank-0 read), then calls
    `checkpoint_written` so the sink can fire the async consolidation+eval job
    that reads those partials, assembles ``model_<step>.pth`` +
    ``training_<step>.pth`` off the critical path, and runs the slow eval.
    """

    def log(self, metrics: dict[str, Any], step: int) -> None: ...
    def console(self, *lines: str) -> None: ...
    def checkpoint_written(self, step: int, *, final: bool) -> None: ...
    def finish(self) -> None: ...


# Convenience alias for the few call sites that don't care which pool the sink serves.
RunSink = OnePoolRunSink | ThreePoolRunSink
