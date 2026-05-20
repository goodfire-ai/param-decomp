"""Open-world experiment driver interface.

The core PD optimizer only needs a target model bundle plus train/eval dataloaders.
An experiment driver is the boundary layer that turns a serializable `Run` config
into those runtime objects. The set of drivers is open-world: custom users can register
their own driver class without editing core code.

Drivers don't own reload-time state: reload calls `build_target` and `build_dataloaders`
exactly like a fresh run, re-fetching whatever upstream the config points at (wandb
pretrain run, HF model, etc.). Saved PD runs depend on their upstream continuing to exist.
"""

from typing import Any, ClassVar, Protocol

from torch.utils.data import DataLoader

from param_decomp.driver_path import load_driver
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.run import Run
from param_decomp.utils.distributed_utils import DistributedState


class ExperimentDriver[RunT: Run](Protocol):
    """Converts a serializable `Run` config into runtime PD objects."""

    name: ClassVar[str]

    @property
    def config_type(self) -> type[RunT]:
        """Pydantic model type used to validate serialized `Run` configs."""
        ...

    def build_target(self, run: RunT) -> PDTarget:
        """Build the target model bundle from upstream."""
        ...

    def build_dataloaders(
        self,
        run: RunT,
        *,
        train_batch_size: int,
        eval_batch_size: int,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
    ) -> tuple[DataLoader[Any], DataLoader[Any]]:
        """Build train/eval dataloaders."""
        ...


__all__ = [
    "ExperimentDriver",
    "load_driver",
]
