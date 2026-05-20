"""Open-world experiment driver interface.

The core PD optimizer only needs a target model bundle plus train/eval dataloaders.
An experiment driver is the boundary layer that turns a serializable `RunConfig`
into those runtime objects. The set of drivers is open-world: custom users can register
their own driver class without editing core code.

Drivers own the typing of `run_cfg.target` and `run_cfg.data` — core stores both as
raw dicts. Each driver validates them via its own pydantic models in `validate_config`
and re-parses at the top of each `build_*` method.

Drivers don't own reload-time state: reload calls `build_target`,
`build_train_loader`, and `build_eval_loader` exactly like a fresh run,
re-fetching whatever upstream the config points at (wandb pretrain run, HF model,
etc.). Saved PD runs depend on their upstream continuing to exist.
"""

from typing import Any, ClassVar, Protocol

from torch.utils.data import DataLoader

from param_decomp.driver_path import load_driver
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.run import RunConfig
from param_decomp.utils.distributed_utils import DistributedState


class ExperimentDriver(Protocol):
    """Converts a serializable `RunConfig` into runtime PD objects."""

    name: ClassVar[str]

    def validate_config(self, run_cfg: RunConfig) -> None:
        """Parse driver-private fields (`target`, `data`) and raise on bad shape.

        Implementations call e.g. ``MyTargetConfig.model_validate(run_cfg.target)``
        and ``MyDataConfig.model_validate(run_cfg.data)``, letting pydantic raise
        on bad shape. Called pre-flight by ``pd-run`` (single run) and the sweep
        launcher (so a 200-config sweep dies before SLURM submit, not in task 17).
        Idempotent and cheap (no I/O).
        """
        ...

    def build_target(self, run_cfg: RunConfig) -> PDTarget:
        """Build the target model bundle from upstream."""
        ...

    def build_train_loader(
        self,
        run_cfg: RunConfig,
        *,
        device: str,
        batch_size_override: int | None = None,
        dist_state: DistributedState | None = None,
    ) -> DataLoader[Any]:
        """Build the train dataloader.

        Defaults to ``run_cfg.pd.batch_size``; pass ``batch_size_override`` to use a
        different batch size (e.g. for offline analysis scripts that want a custom
        batch size without rewriting the saved ``run_cfg``).

        ``device`` is the distributed-aware target device (``cuda:<local_rank>`` for
        DDP, ``"cpu"`` otherwise). Synthetic-data drivers (TMS, ResidMLP) generate
        batches on this device to avoid per-step CPU→GPU copies; LM-style drivers
        that hand off raw tensors can ignore it. No default — silently falling back
        to ``"cpu"`` would mis-route the synthetic drivers on a GPU run.
        """
        ...

    def build_eval_loader(
        self,
        run_cfg: RunConfig,
        *,
        device: str,
        batch_size_override: int | None = None,
        dist_state: DistributedState | None = None,
    ) -> DataLoader[Any]:
        """Build the eval dataloader.

        Defaults to ``run_cfg.logging.eval_batch_size``; pass ``batch_size_override``
        to use a different batch size. See ``build_train_loader`` for ``device``.
        """
        ...


__all__ = [
    "ExperimentDriver",
    "load_driver",
]
