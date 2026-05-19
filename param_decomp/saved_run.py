"""Handle to a saved PD run on disk or wandb."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from param_decomp.configs import PDConfig
from param_decomp.driver_path import load_driver
from param_decomp.experiments.driver import ExperimentDriver
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.models.component_model import ComponentModel
from param_decomp.run import RUN_METADATA_FILENAME, Run
from param_decomp.types import ModelPath
from param_decomp.utils.distributed_utils import DistributedState
from param_decomp.utils.run_files import resolve_run_files


@dataclass
class PDRun:
    """A saved PD run, resolved to local paths and parsed `Run` config."""

    path: Path
    run: Run
    checkpoint_path: Path

    @classmethod
    def from_path(cls, path: ModelPath) -> "PDRun":
        files = resolve_run_files(
            path,
            config_filename=RUN_METADATA_FILENAME,
            checkpoint_prefix="model",
        )
        return cls(
            path=files.config_path.parent,
            run=Run.from_file(files.config_path),
            checkpoint_path=files.checkpoint_path,
        )

    @cached_property
    def driver(self) -> ExperimentDriver[Any] | None:
        return load_driver(self.run.driver_path) if self.run.driver_path else None

    @property
    def pd_config(self) -> PDConfig:
        return self.run.pd

    @property
    def name(self) -> str:
        if self.driver is not None:
            return self.driver.name
        return "custom"

    def load_target(self) -> PDTarget:
        assert self.driver is not None, (
            "Run has no driver. Use `load_component_model(path, target=...)` with an "
            "explicit target."
        )
        return self.driver.build_target(self.run)

    def load_dataloaders(
        self,
        *,
        train_batch_size: int,
        eval_batch_size: int,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
    ) -> tuple[DataLoader[Any], DataLoader[Any]]:
        assert self.driver is not None, (
            "Run has no driver. Build dataloaders explicitly for custom runs."
        )
        return self.driver.build_dataloaders(
            self.run,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            dist_state=dist_state,
            device=device,
        )

    def load_model(self, target: PDTarget | None = None) -> ComponentModel:
        target = target if target is not None else self.load_target()
        return ComponentModel.from_checkpoint(
            config=self.pd_config,
            checkpoint_path=self.checkpoint_path,
            target_model=target.model,
            run_batch=target.run_batch,
            tied_weights=target.tied_weights,
        )


def load_component_model(
    path: ModelPath,
    *,
    target: PDTarget | None = None,
) -> ComponentModel:
    """Load a `ComponentModel` from a saved PD run.

    Args:
        path: Run directory, wandb path (`wandb:entity/project/runs/id`), or checkpoint file.
        target: Optional override. When ``None``, the run's driver reconstructs the target
            from the saved `Run` config. For manual/notebook runs (no driver), ``target`` is
            required.
    """
    return PDRun.from_path(path).load_model(target=target)
