"""Handle to a saved PD run on disk or wandb."""

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

    `driver` is resolved from `run_cfg.driver_path` at construction time and paired
    with `run_cfg` (whose concrete subtype is checked against `driver.config_type`).
    `None` for runs produced via direct `run_pd` without a driver — those reload with
    `load_component_model(path, target=...)`.
    """

    path: Path
    run_cfg: RunConfig
    checkpoint_path: Path
    driver: ExperimentDriver[Any] | None

    @classmethod
    def from_path(cls, path: ModelPath) -> "PDRun":
        files = resolve_run_files(
            path,
            config_filename=RUN_CONFIG_FILENAME,
            checkpoint_prefix="model",
        )
        run_cfg = RunConfig.from_file(files.config_path)
        driver = load_driver(run_cfg.driver_path) if run_cfg.driver_path else None
        if driver is not None:
            assert isinstance(run_cfg, driver.config_type), (
                f"Run has type {type(run_cfg).__name__}, expected {driver.config_type.__name__}"
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
        if self.driver is not None:
            return self.driver.name
        return "custom"

    def load_target(self) -> PDTarget:
        assert self.driver is not None, (
            "Run has no driver. Use `load_component_model(path, target=...)` with an "
            "explicit target."
        )
        return self.driver.build_target(self.run_cfg)

    def build_train_loader(
        self,
        *,
        batch_size_override: int | None = None,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
    ) -> DataLoader[Any]:
        assert self.driver is not None, (
            "Run has no driver. Build dataloaders explicitly for custom runs."
        )
        return self.driver.build_train_loader(
            self.run_cfg,
            batch_size_override=batch_size_override,
            dist_state=dist_state,
            device=device,
        )

    def build_eval_loader(
        self,
        *,
        batch_size_override: int | None = None,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
    ) -> DataLoader[Any]:
        assert self.driver is not None, (
            "Run has no driver. Build dataloaders explicitly for custom runs."
        )
        return self.driver.build_eval_loader(
            self.run_cfg,
            batch_size_override=batch_size_override,
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
            from the saved `RunConfig`. For manual/notebook runs (no driver), ``target`` is
            required.
    """
    return PDRun.from_path(path).load_model(target=target)
