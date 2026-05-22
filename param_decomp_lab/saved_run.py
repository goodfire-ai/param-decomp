"""`SavedRun`: lab-side handle to a completed PD run on disk or in W&B.

Reads the `run_meta.yaml` produced by an experiment's `run.py`, dispatches to the registered
experiment module to rebuild the target model and dataloaders, and exposes a `load_model()`
that returns a fully-loaded `ComponentModel`.

Notebook-only runs (trained via `optimize(...)` without saving a `run_meta.yaml`) reload
their checkpoint with `load_component_model_from_checkpoint(...)` directly.
"""

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml
from torch.utils.data import DataLoader

from param_decomp.component_model import ComponentModel
from param_decomp.configs import PDConfig, RuntimeConfig
from param_decomp.distributed import DistributedState
from param_decomp_lab.component_model_io import load_component_model_from_checkpoint
from param_decomp_lab.experiments import EXPERIMENTS
from param_decomp_lab.experiments.utils import RUN_META_FILENAME
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import resolve_config_path, resolve_run_files


@dataclass(frozen=True)
class RunMeta:
    """The deserialized shape of ``run_meta.yaml``."""

    experiment_name: str
    pd_config: PDConfig
    runtime_config: RuntimeConfig
    target_dict: dict[str, Any]
    data_dict: dict[str, Any]

    @classmethod
    def from_path(cls, path: Path) -> "RunMeta":
        with open(path) as f:
            payload = yaml.safe_load(f)
        return cls(
            experiment_name=payload["experiment"],
            pd_config=PDConfig.model_validate(payload["pd"]),
            runtime_config=RuntimeConfig.model_validate(payload["runtime"]),
            target_dict=payload["target"],
            data_dict=payload["data"],
        )


@dataclass(frozen=True)
class SavedRun:
    """A completed PD run resolved to local paths + parsed meta + the matching experiment module.

    Always constructed via :meth:`from_path`. Notebook-only runs (no `run_meta.yaml`) reload
    checkpoints with `load_component_model_from_checkpoint(...)` directly.
    """

    path: Path
    meta: RunMeta
    checkpoint_path: Path
    experiment: ModuleType

    @classmethod
    def from_path(cls, path: ModelPath) -> "SavedRun":
        files = resolve_run_files(
            path, config_filename=RUN_META_FILENAME, checkpoint_prefix="model"
        )
        meta = RunMeta.from_path(files.config_path)
        assert meta.experiment_name in EXPERIMENTS, (
            f"unknown experiment {meta.experiment_name!r} (registered: {sorted(EXPERIMENTS)})"
        )
        return cls(
            path=files.config_path.parent,
            meta=meta,
            checkpoint_path=files.checkpoint_path,
            experiment=EXPERIMENTS[meta.experiment_name],
        )

    @classmethod
    def meta_from_path(cls, path: ModelPath) -> RunMeta:
        """Load just ``run_meta.yaml`` without resolving the checkpoint."""
        return RunMeta.from_path(resolve_config_path(path, config_filename=RUN_META_FILENAME))

    # ---------- Config accessors ----------

    @property
    def pd_config(self) -> PDConfig:
        return self.meta.pd_config

    @property
    def runtime_config(self) -> RuntimeConfig:
        return self.meta.runtime_config

    @property
    def experiment_name(self) -> str:
        return self.meta.experiment_name

    @property
    def target_cfg(self) -> Any:
        return self.experiment.TargetConfig.model_validate(self.meta.target_dict)

    @property
    def data_cfg(self) -> Any:
        return self.experiment.DataConfig.model_validate(self.meta.data_dict)

    # ---------- Rebuild ----------

    def load_target(self) -> Any:
        return self.experiment.build_target(self.target_cfg)

    def build_train_loader(
        self,
        *,
        device: str,
        batch_size: int | None = None,
        dist_state: DistributedState | None = None,
        seed: int | None = None,
    ) -> DataLoader[Any]:
        return self.experiment.build_train_loader(
            self.target_cfg,
            self.data_cfg,
            batch_size=batch_size if batch_size is not None else self.pd_config.batch_size,
            device=device,
            dist_state=dist_state,
            seed=seed if seed is not None else self.pd_config.seed,
        )

    def build_eval_loader(
        self,
        *,
        device: str,
        batch_size: int,
        dist_state: DistributedState | None = None,
        seed: int | None = None,
    ) -> DataLoader[Any]:
        return self.experiment.build_eval_loader(
            self.target_cfg,
            self.data_cfg,
            batch_size=batch_size,
            device=device,
            dist_state=dist_state,
            seed=seed if seed is not None else self.pd_config.seed,
        )

    def load_model(self) -> ComponentModel:
        target_model = self.load_target()
        return load_component_model_from_checkpoint(
            ci_config=self.pd_config.ci_config,
            sigmoid_type=self.pd_config.sigmoid_type,
            decomposition_targets=self.pd_config.decomposition_targets,
            identity_decomposition_targets=self.pd_config.identity_decomposition_targets,
            checkpoint_path=self.checkpoint_path,
            target_model=target_model,
            run_batch=self.experiment.make_run_batch(self.target_cfg),
            tied_weights=self.pd_config.tied_weights,
        )
