"""`RunSink`: where `Trainer.run()` sends its output.

Defined here as a runtime-checkable `Protocol` so the core trainer can document
exactly what it requires of the caller without committing to an output format
or any specific dependency (wandb, tqdm, filesystem).

Timing — when the loop emits — lives separately: `param_decomp.configs.Cadence`
owns train-log + checkpoint periods, and `param_decomp.optimize.EvalLoop` owns
the eval period (alongside the runtime eval objects). This Protocol describes
side effects only: where structured metrics, free-form console output, and
trainer checkpoints go.

Concrete implementations live with the caller. The in-repo experiments use
``param_decomp_lab.run_sink.RunSink`` (local files + wandb + `is_main_process`
no-op fan-out), but external callers are free to bring their own.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TrainerLike(Protocol):
    """The subset of trainer behaviour `RunSink.checkpoint` may reach for.

    `Trainer` and the pool-specific trainers (`TwoPoolTrainer`,
    `ThreePoolTrainer`) all satisfy this Protocol structurally — there is no
    inheritance relationship between them. Sink implementations choose which
    methods to call: the default lab sink writes only the consumable model
    state dict; a resume-aware sink additionally writes the full state blob.
    """

    def consumable_model_state_dict(self) -> dict[str, Any]:
        """Return the model state dict in the form downstream tools expect
        (gathered to a single full state dict on rank 0 for sharded pools)."""
        ...

    def state_blob(self) -> dict[str, Any]:
        """Return the atomic cfg + state blob used for resumption."""
        ...


@runtime_checkable
class RunSink(Protocol):
    """Side-effect sink for a PD training run.

    The trainer treats this object as opaque: it reports what happened, never *where*
    the record should go. Callers implement the methods to point at whatever output
    channels they want (local files, wandb, S3, a no-op handle on non-main DP ranks,
    ...).
    """

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Record a flat metrics dict at ``step``.

        Args:
            metrics: Flat dict whose keys are already namespaced (e.g. ``train/loss/total``,
                ``eval/ci_l0/L0``) by the trainer.
            step: Train step at which the metrics were recorded.
        """
        ...

    def console(self, *lines: str) -> None:
        """Emit free-form lines (e.g. tqdm-friendly progress)."""
        ...

    def checkpoint(self, trainer: TrainerLike, step: int) -> None:
        """Persist trainer state at the given step.

        Implementations choose what to save — typically the consumable model
        state dict at minimum; resume-aware sinks additionally persist
        ``trainer.state_blob()`` as a per-rank shard alongside.
        """
        ...
