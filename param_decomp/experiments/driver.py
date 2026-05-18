"""Open-world experiment driver interface.

The core PD optimizer only needs a target model bundle plus train/eval dataloaders.
An experiment driver is the boundary layer that turns a serializable experiment config
into those runtime objects. The set of drivers is open-world: custom users can register
their own driver class without editing core code.

Drivers don't own reload-time state: reload calls `build_target` and `build_dataloaders`
exactly like a fresh run, re-fetching whatever upstream the config points at (wandb
pretrain run, HF model, etc.). Saved PD runs depend on their upstream continuing to exist.
"""

from importlib import import_module
from typing import Any, ClassVar, Protocol, Self

from pydantic import Field, model_validator
from torch.utils.data import DataLoader

from param_decomp.base_config import BaseConfig
from param_decomp.configs import LoggingConfig, PDConfig, RuntimeConfig
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.utils.distributed_utils import DistributedState


class ExperimentConfig(BaseConfig):
    """Pure-data config shared by all experiment configs. Drivers subclass this."""

    pd: PDConfig
    logging: LoggingConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def validate_metric_overlap(self) -> Self:
        loss_names = {name for name, val in self.pd.loss_metrics if val is not None}
        eval_names = {name for name, val in self.logging.eval_metrics if val is not None}
        overlap = loss_names & eval_names
        assert not overlap, (
            f"The same metric was set under both pd.loss_metrics and logging.eval_metrics: "
            f"{sorted(overlap)}. Loss metrics are automatically evaluated; remove the "
            "logging.eval_metrics entry, or move it out of pd.loss_metrics if you want eval-only."
        )
        return self


class ExperimentDriver[ConfigT: ExperimentConfig](Protocol):
    """Converts a serializable experiment config into runtime PD objects."""

    name: ClassVar[str]

    @property
    def config_type(self) -> type[ConfigT]:
        """Pydantic model type used to validate serialized experiment configs."""
        ...

    def build_target(self, config: ConfigT) -> PDTarget:
        """Build the target model bundle from upstream."""
        ...

    def build_dataloaders(
        self,
        config: ConfigT,
        *,
        train_batch_size: int,
        eval_batch_size: int,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
    ) -> tuple[DataLoader[Any], DataLoader[Any]]:
        """Build train/eval dataloaders."""
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
