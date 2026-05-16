"""Handle to a saved PD run on disk or wandb."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from param_decomp.configs import PDConfig
from param_decomp.experiments.driver import (
    ExperimentConfig,
    ExperimentDriver,
    load_driver,
)
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.models.component_model import ComponentModel
from param_decomp.run_metadata import RUN_METADATA_FILENAME, RunMetadata
from param_decomp.types import ModelPath
from param_decomp.utils.distributed_utils import DistributedState
from param_decomp.utils.run_files import resolve_config_path, resolve_run_files


@dataclass
class PDRun:
    """A saved PD run, resolved to local paths and parsed metadata."""

    path: Path
    metadata: RunMetadata
    checkpoint_path: Path

    @classmethod
    def from_path(cls, path: ModelPath) -> "PDRun":
        files = resolve_run_files(
            path,
            config_filename=RUN_METADATA_FILENAME,
            checkpoint_prefix="model",
            extras_from_config_path=_artifact_filenames_from,
        )
        return cls(
            path=files.config_path.parent,
            metadata=RunMetadata.from_file(files.config_path),
            checkpoint_path=files.checkpoint_path,
        )

    @classmethod
    def metadata_from_path(cls, path: ModelPath) -> RunMetadata:
        """Load just the run metadata, without resolving or downloading checkpoints."""
        return RunMetadata.from_file(
            resolve_config_path(path, config_filename=RUN_METADATA_FILENAME)
        )

    @cached_property
    def driver(self) -> ExperimentDriver[Any] | None:
        return load_driver(self.metadata.driver) if self.metadata.driver else None

    @cached_property
    def experiment_config(self) -> ExperimentConfig | None:
        if self.driver is None:
            return None
        return self.driver.config_type.model_validate(self.metadata.config)

    @property
    def pd_config(self) -> PDConfig:
        return PDConfig.model_validate(self.metadata.config["pd"])

    @property
    def name(self) -> str:
        if self.driver is not None:
            return self.driver.name
        return "custom"

    def load_target(self) -> PDTarget:
        assert self.driver is not None and self.experiment_config is not None, (
            "Run has no driver. Use `load_pd(path, target=...)` with an explicit target."
        )
        return self.driver.load_target(self.experiment_config, self.path)

    def load_dataloaders(
        self,
        *,
        train_batch_size: int,
        eval_batch_size: int,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
    ) -> tuple[DataLoader[Any], DataLoader[Any]]:
        assert self.driver is not None and self.experiment_config is not None, (
            "Run has no driver. Build dataloaders explicitly for custom runs."
        )
        return self.driver.build_dataloaders(
            self.experiment_config,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            dist_state=dist_state,
            device=device,
            run_dir=self.path,
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


def _artifact_filenames_from(config_path: Path) -> list[str]:
    return RunMetadata.from_file(config_path).artifact_filenames
