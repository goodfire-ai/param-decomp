"""Open-world experiment driver interface.

The core PD optimizer only needs a target model bundle plus train/eval dataloaders.
An experiment driver is the boundary layer that turns a serializable `RunConfig`
into those runtime objects. The set of drivers is open-world: custom users can register
their own driver class without editing core code.

Drivers don't own reload-time state: reload calls `build_target`,
`build_train_loader`, and `build_eval_loader` exactly like a fresh run,
re-fetching whatever upstream the config points at (wandb pretrain run, HF model,
etc.). Saved PD runs depend on their upstream continuing to exist.
"""

from pathlib import Path
from typing import Any, ClassVar, Protocol

from torch.utils.data import DataLoader

from param_decomp.driver_path import load_driver
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.models.component_model import ComponentModel
from param_decomp.run import RunConfig
from param_decomp.run_sink import RunSink
from param_decomp.utils.distributed_utils import DistributedState


class ExperimentDriver[RunConfigT: RunConfig](Protocol):
    """Converts a serializable `RunConfig` into runtime PD objects."""

    name: ClassVar[str]

    @property
    def config_type(self) -> type[RunConfigT]:
        """Pydantic model type used to validate serialized `RunConfig` data."""
        ...

    def build_target(self, run_cfg: RunConfigT) -> PDTarget:
        """Build the target model bundle from upstream."""
        ...

    def build_train_loader(
        self,
        run_cfg: RunConfigT,
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
        run_cfg: RunConfigT,
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

    def optimize(
        self,
        run_cfg: RunConfigT,
        target: PDTarget,
        train_loader: DataLoader[Any],
        eval_loader: DataLoader[Any],
        *,
        device: str,
        dist_state: DistributedState | None,
        sink: RunSink,
    ) -> None:
        """Run the PD training loop for this experiment.

        Most drivers delegate to ``param_decomp.run_pd.optimize`` (the generic
        DDP-aware trainer). Experiments with per-step idiosyncrasies (e.g. TMS
        weight tying) can dispatch to their own loop here.
        """
        ...

    def load_model(self, run_cfg: RunConfigT, checkpoint_path: Path) -> ComponentModel:
        """Reload a trained ``ComponentModel`` from a checkpoint.

        Most drivers call ``ComponentModel.from_checkpoint`` with the target this
        driver builds. Experiments with reload-time setup (e.g. TMS weight tying)
        apply it here.
        """
        ...


__all__ = [
    "ExperimentDriver",
    "load_driver",
]
