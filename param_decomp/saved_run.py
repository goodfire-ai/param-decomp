"""`SavedRun`: handle to a completed PD run on disk or in W&B.

A `SavedRun` is the **reader** end of a run's three-phase lifecycle:
`RunConfig` (recipe) → `RunSink` (writer during training) → `SavedRun`
(reader after).

Construct via `SavedRun.from_path(path)` and use it to reconstruct the
target/dataloaders and load the model checkpoint. Methods are driver-mediated
only — notebook callers who trained via `optimize(...)` without a `RunConfig`
should reload their checkpoint with `ComponentModel.from_checkpoint(...)`
directly.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from param_decomp.configs import PDConfig
from param_decomp.driver_path import load_driver
from param_decomp.experiments.driver import ExperimentDriver
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.models.component_model import ComponentModel
from param_decomp.run import RUN_CONFIG_FILENAME, RunConfig
from param_decomp.types import ModelPath
from param_decomp.utils.distributed_utils import DistributedState
from param_decomp.utils.run_files import resolve_config_path, resolve_run_files


@dataclass(frozen=True)
class SavedRun:
    """A completed PD run, resolved to local paths and parsed `RunConfig`.

    `driver` is resolved from `run_cfg.driver_path` at construction time and
    paired with `run_cfg` (whose concrete subtype is checked against
    `driver.config_type`). Both fields are required — there is no
    notebook-only `SavedRun`; notebook callers reload checkpoints via
    `ComponentModel.from_checkpoint(...)` directly.
    """

    path: Path
    run_cfg: RunConfig
    checkpoint_path: Path
    driver: ExperimentDriver[Any]

    @classmethod
    def from_path(cls, path: ModelPath) -> "SavedRun":
        """Reload from disk or W&B. Resolves spec + checkpoint, instantiates the driver."""
        files = resolve_run_files(
            path, config_filename=RUN_CONFIG_FILENAME, checkpoint_prefix="model"
        )
        run_cfg = RunConfig.from_file(files.config_path)
        driver = load_driver(run_cfg.driver_path)
        assert isinstance(run_cfg, driver.config_type), (
            f"RunConfig has type {type(run_cfg).__name__}, expected {driver.config_type.__name__}"
        )
        return cls(
            path=files.config_path.parent,
            run_cfg=run_cfg,
            checkpoint_path=files.checkpoint_path,
            driver=driver,
        )

    @classmethod
    def run_cfg_from_path(cls, path: ModelPath) -> RunConfig:
        """Load just the `RunConfig` without resolving the checkpoint."""
        return RunConfig.from_file(resolve_config_path(path, config_filename=RUN_CONFIG_FILENAME))

    @property
    def pd_config(self) -> PDConfig:
        return self.run_cfg.pd

    @property
    def name(self) -> str:
        return self.driver.name

    def load_target(self) -> PDTarget:
        return self.driver.build_target(self.run_cfg)

    def build_train_loader(
        self,
        *,
        device: str,
        batch_size_override: int | None = None,
        dist_state: DistributedState | None = None,
    ) -> DataLoader[Any]:
        return self.driver.build_train_loader(
            self.run_cfg,
            device=device,
            batch_size_override=batch_size_override,
            dist_state=dist_state,
        )

    def build_eval_loader(
        self,
        *,
        device: str,
        batch_size_override: int | None = None,
        dist_state: DistributedState | None = None,
    ) -> DataLoader[Any]:
        return self.driver.build_eval_loader(
            self.run_cfg,
            device=device,
            batch_size_override=batch_size_override,
            dist_state=dist_state,
        )

    def load_model(self) -> ComponentModel:
        target = self.load_target()
        return ComponentModel.from_checkpoint(
            config=self.pd_config,
            checkpoint_path=self.checkpoint_path,
            target_model=target.model,
            run_batch=target.run_batch,
            tied_weights=target.tied_weights,
        )


def load_component_model(path: ModelPath) -> ComponentModel:
    """Load a `ComponentModel` from a saved driver-mediated PD run.

    Thin convenience over `SavedRun.from_path(path).load_model()`.
    """
    return SavedRun.from_path(path).load_model()
