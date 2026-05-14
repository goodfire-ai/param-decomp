"""Handle to a saved PD run on disk or wandb."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from param_decomp.configs import PDConfig
from param_decomp.experiments.driver import (
    EXPERIMENT_MANIFEST_FILENAME,
    ExperimentConfig,
    ExperimentDriver,
    load_driver,
)
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.param_decomp_types import ModelPath
from param_decomp.utils.distributed_utils import DistributedState
from param_decomp.utils.run_files import resolve_config_path, resolve_run_files

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from param_decomp.models.component_model import ComponentModel


@dataclass
class PDRun:
    """A saved PD run, resolved to local paths and a parsed manifest dict.

    Manifest schema:
        driver: import path to the driver class, or null for notebook/custom runs
        name: human label (also used as the wandb tag)
        config: full ExperimentConfig dump (or {"pd": ...} for custom runs)
        artifact_filenames: list of extra files saved beside the checkpoint
    """

    path: Path
    manifest: dict[str, Any]
    checkpoint_path: Path

    @classmethod
    def from_path(cls, path: ModelPath) -> "PDRun":
        files = resolve_run_files(
            path,
            config_filename=EXPERIMENT_MANIFEST_FILENAME,
            checkpoint_prefix="model",
            extras_from_config_path=_artifact_filenames_from,
        )
        return cls(
            path=files.config_path.parent,
            manifest=_load_manifest(files.config_path),
            checkpoint_path=files.checkpoint_path,
        )

    @classmethod
    def manifest_from_path(cls, path: ModelPath) -> dict[str, Any]:
        """Load just the manifest dict, without resolving or downloading checkpoints."""
        return _load_manifest(
            resolve_config_path(path, config_filename=EXPERIMENT_MANIFEST_FILENAME)
        )

    @cached_property
    def driver(self) -> ExperimentDriver[Any] | None:
        driver_path = self.manifest.get("driver")
        return load_driver(driver_path) if driver_path else None

    @cached_property
    def experiment_config(self) -> ExperimentConfig | None:
        if self.driver is None:
            return None
        return self.driver.config_type.model_validate(self.manifest["config"])

    @property
    def pd_config(self) -> PDConfig:
        return PDConfig.model_validate(self.manifest["config"]["pd"])

    @property
    def name(self) -> str:
        return self.manifest.get("name", "custom")

    def load_target(self) -> PDTarget:
        assert self.driver is not None and self.experiment_config is not None, (
            "Run manifest has no driver. Use `load_pd(path, target=...)` with an explicit target."
        )
        return self.driver.build_target(self.experiment_config, run_dir=self.path)

    def load_dataloaders(
        self,
        *,
        train_batch_size: int,
        eval_batch_size: int,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
    ) -> "tuple[DataLoader[Any], DataLoader[Any]]":
        assert self.driver is not None and self.experiment_config is not None, (
            "Run manifest has no driver. Build dataloaders explicitly for custom runs."
        )
        return self.driver.build_dataloaders(
            self.experiment_config,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            dist_state=dist_state,
            device=device,
            run_dir=self.path,
        )

    def load_model(self, target: PDTarget | None = None) -> "ComponentModel":
        from param_decomp.models.component_model import ComponentModel

        target = target if target is not None else self.load_target()
        return ComponentModel.from_checkpoint(
            config=self.pd_config,
            checkpoint_path=self.checkpoint_path,
            target_model=target.model,
            run_batch=target.run_batch,
            tied_weights=target.tied_weights,
        )


def _load_manifest(path: Path) -> dict[str, Any]:
    assert path.exists(), f"{EXPERIMENT_MANIFEST_FILENAME} not found at {path}"
    with open(path) as f:
        return yaml.safe_load(f)


def _artifact_filenames_from(config_path: Path) -> list[str]:
    return list(_load_manifest(config_path).get("artifact_filenames", []))
