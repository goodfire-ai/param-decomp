"""Handle to a saved PD run on disk or wandb."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from torch.utils.data import DataLoader

from param_decomp.compose import resolve_run
from param_decomp.configs import PDConfig
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

    The driver is always present — it's resolved from ``run_cfg.driver_path`` at
    construction time via the composition root.
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
        with open(files.config_path) as f:
            run_cfg, driver = resolve_run(yaml.safe_load(f))
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
    """Load a `ComponentModel` from a saved PD run.

    Args:
        path: Run directory, wandb path (`wandb:entity/project/runs/id`), or checkpoint file.
    """
    return PDRun.from_path(path).load_model()
