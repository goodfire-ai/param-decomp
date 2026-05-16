"""Open-world experiment driver interface.

The core PD optimizer only needs a target model bundle plus train/eval dataloaders.
An experiment driver is the boundary layer that turns a serializable experiment config
into those runtime objects. The set of drivers is open-world: custom users can register
their own driver class without editing core code.
"""

from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar, Protocol

from torch.utils.data import DataLoader

from param_decomp.base_config import BaseConfig
from param_decomp.configs import PDConfig
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.utils.distributed_utils import DistributedState


class ExperimentConfig(BaseConfig):
    """Pure-data config shared by all experiment configs. Drivers subclass this."""

    pd: PDConfig


class ExperimentDriver[ConfigT: ExperimentConfig](Protocol):
    """Converts a serializable experiment config into runtime PD objects.

    A driver is a stateless object that owns its config schema (`config_type`) and knows
    how to build a `PDTarget`, train/eval dataloaders, and (optionally) extra files to
    persist beside the checkpoint so that the run is self-contained on reload.
    """

    name: ClassVar[str]

    @property
    def config_type(self) -> type[ConfigT]:
        """Pydantic model type used to validate serialized experiment configs."""
        ...

    def build_target(self, config: ConfigT, *, run_dir: Path | None = None) -> PDTarget:
        """Build the target model bundle. If `run_dir` is given, prefer locally bundled files."""
        ...

    def build_dataloaders(
        self,
        config: ConfigT,
        *,
        train_batch_size: int,
        eval_batch_size: int,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
        run_dir: Path | None = None,
    ) -> tuple[DataLoader[Any], DataLoader[Any]]:
        """Build train/eval dataloaders. If `run_dir` is given, prefer locally bundled files."""
        ...

    def artifacts(self, config: ConfigT, target: PDTarget) -> dict[str, Any]:
        """Filename → data for extra files to persist beside the checkpoint. Default: {}."""
        ...


def load_driver(driver_path: str) -> ExperimentDriver[Any]:
    """Load a driver object or no-arg driver class from a `module:attr` import path."""
    module_path, sep, attr = driver_path.partition(":")
    if sep == "":
        raise ValueError(f"Driver path must be of the form 'module:attr', got {driver_path!r}")
    module = import_module(module_path)
    driver = getattr(module, attr)
    if isinstance(driver, type):
        driver = driver()
    return driver


__all__ = [
    "ExperimentConfig",
    "ExperimentDriver",
    "load_driver",
]
