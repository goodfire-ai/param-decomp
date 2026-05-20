"""Handle to a saved PD run on disk or wandb.

Driver-mediated only. Notebook callers that called ``optimize(...)`` directly
have no ``RunConfig`` on disk; if they want to reload a checkpoint they manage
that themselves (e.g. via ``ComponentModel.from_checkpoint(...)``).
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


@dataclass
class PDRun:
    """A saved PD run, resolved to local paths and parsed `RunConfig`.

    `driver` is resolved from `run_cfg.driver_path` at construction time and
    type-checked against `driver.config_type`. Always present — there is no
    no-driver flavour of `PDRun`.
    """

    path: Path
    run_cfg: RunConfig
    checkpoint_path: Path
    driver: ExperimentDriver[Any]

    @classmethod
    def from_path(cls, path: ModelPath) -> "PDRun":
        files = resolve_run_files(
            path,
            config_filename=RUN_CONFIG_FILENAME,
            checkpoint_prefix="model",
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
    def run_from_path(cls, path: ModelPath) -> RunConfig:
        """Load just the `RunConfig`, without resolving or downloading checkpoints."""
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

    The run's driver reconstructs the target from the saved `RunConfig`.
    Notebook callers without a `RunConfig` on disk should use
    ``ComponentModel.from_checkpoint(...)`` directly.
    """
    return PDRun.from_path(path).load_model()
