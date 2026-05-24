"""`SavedRun`: lab-side handle to a completed PD run on disk or in W&B.

Reads the `run_meta.yaml` produced by an experiment's `run.py`, dispatches on
`experiment_kind` to the matching `experiments/<kind>/run.py` module, validates the
`target` / `data` blocks against that module's `TARGET_CONFIG_TYPE` /
`DATA_CONFIG_TYPE`, and exposes `build_target` / `build_loader` / `make_run_batch` /
`load_model` so callers can rebuild the run for postprocessing.

Notebook-only runs (trained via `optimize(...)` without saving a `run_meta.yaml`) reload
their checkpoint with `load_component_model_from_checkpoint(...)` directly.
"""

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import yaml
from torch import nn
from torch.utils.data import DataLoader

from param_decomp.base_config import BaseConfig
from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.component_model import ComponentModel
from param_decomp.configs import Cadence, PDConfig, RuntimeConfig
from param_decomp.distributed import DistributedState
from param_decomp_lab.component_model_io import load_component_model_from_checkpoint
from param_decomp_lab.experiments.utils import RUN_META_FILENAME, EvalConfig, RunKind
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import resolve_config_path, resolve_run_files

_RUN_MODULE_PATHS: dict[RunKind, str] = {
    "lm": "param_decomp_lab.experiments.lm.run",
    "tms": "param_decomp_lab.experiments.tms.run",
    "resid_mlp": "param_decomp_lab.experiments.resid_mlp.run",
}


def _run_module(kind: RunKind) -> ModuleType:
    """Import the `experiments/<kind>/run.py` module on demand.

    Each module exposes `TARGET_CONFIG_TYPE`, `DATA_CONFIG_TYPE`, `build_target`,
    `build_loader`, and `make_run_batch`.
    """
    return importlib.import_module(_RUN_MODULE_PATHS[kind])


@dataclass(frozen=True)
class RunMeta:
    """Deserialized shape of ``run_meta.yaml``.

    The shared `ExperimentConfig` blocks are validated up front; `target` and `data`
    remain raw dicts because their concrete schemas depend on `kind` — the matching
    experiment module's `TARGET_CONFIG_TYPE` / `DATA_CONFIG_TYPE` validate them in
    `SavedRun.from_path`.

    Attributes:
        kind: Experiment kind literal (e.g. `"tms"`, `"resid_mlp"`, `"lm"`).
        pd_config: Resolved PD algorithm config.
        runtime_config: Resolved runtime substrate config.
        cadence: Resolved train-log + checkpoint cadence.
        eval_cfg: Resolved eval config, or `None` if eval was disabled.
        target_dict: Raw `target:` YAML block, pending experiment-specific validation.
        data_dict: Raw `data:` YAML block, pending experiment-specific validation.
    """

    kind: RunKind
    pd_config: PDConfig
    runtime_config: RuntimeConfig
    cadence: Cadence
    eval_cfg: EvalConfig | None
    target_dict: dict[str, Any]
    data_dict: dict[str, Any]

    @classmethod
    def from_path(cls, path: Path) -> "RunMeta":
        """Load and validate a `run_meta.yaml` file at `path`.

        Returns:
            The parsed `RunMeta`. The `target` / `data` blocks remain as raw dicts.
        """
        with open(path) as f:
            payload = yaml.safe_load(f)
        eval_payload = payload.get("eval")
        kind = payload["experiment_kind"]
        assert kind in _RUN_MODULE_PATHS, f"Unknown experiment_kind={kind!r}"
        return cls(
            kind=kind,
            pd_config=PDConfig.model_validate(payload["pd"]),
            runtime_config=RuntimeConfig.model_validate(payload["runtime"]),
            cadence=Cadence.model_validate(payload["cadence"]),
            eval_cfg=EvalConfig.model_validate(eval_payload) if eval_payload is not None else None,
            target_dict=payload["target"],
            data_dict=payload["data"],
        )


@dataclass(frozen=True)
class SavedRun:
    """Handle to a completed PD run resolved to local paths + parsed meta + validated configs.

    Always constructed via :meth:`from_path`. Notebook-only runs (no `run_meta.yaml`)
    should reload checkpoints with `load_component_model_from_checkpoint(...)` directly.

    Attributes:
        path: Directory containing `run_meta.yaml` and the checkpoint.
        meta: Parsed `RunMeta` (shared blocks + raw target/data dicts).
        checkpoint_path: Resolved path to the chosen `model_<step>.pth` checkpoint.
        target_cfg: `meta.target_dict` validated against the experiment module's
            `TARGET_CONFIG_TYPE`.
        data_cfg: `meta.data_dict` validated against the experiment module's
            `DATA_CONFIG_TYPE`.
    """

    path: Path
    meta: RunMeta
    checkpoint_path: Path
    target_cfg: BaseConfig
    data_cfg: BaseConfig

    @classmethod
    def from_path(cls, path: ModelPath) -> "SavedRun":
        """Resolve a run directory or W&B path into a fully-validated `SavedRun`.

        Reads `run_meta.yaml`, imports the matching experiment module on demand, and
        validates the raw `target` / `data` blocks against its config types.

        Args:
            path: Local directory or W&B path containing the run files.
        """
        files = resolve_run_files(
            path, config_filename=RUN_META_FILENAME, checkpoint_prefix="model"
        )
        meta = RunMeta.from_path(files.config_path)
        mod = _run_module(meta.kind)
        return cls(
            path=files.config_path.parent,
            meta=meta,
            checkpoint_path=files.checkpoint_path,
            target_cfg=mod.TARGET_CONFIG_TYPE.model_validate(meta.target_dict),
            data_cfg=mod.DATA_CONFIG_TYPE.model_validate(meta.data_dict),
        )

    @classmethod
    def meta_from_path(cls, path: ModelPath) -> RunMeta:
        """Load just ``run_meta.yaml`` without resolving the checkpoint."""
        return RunMeta.from_path(resolve_config_path(path, config_filename=RUN_META_FILENAME))

    # ---------- Config accessors ----------

    @property
    def kind(self) -> RunKind:
        return self.meta.kind

    @property
    def pd_config(self) -> PDConfig:
        return self.meta.pd_config

    @property
    def runtime_config(self) -> RuntimeConfig:
        return self.meta.runtime_config

    # ---------- Rebuild ----------

    def build_target(self) -> nn.Module:
        """Rebuild the target `nn.Module` via the experiment module's `build_target`."""
        return _run_module(self.kind).build_target(self.target_cfg)

    def build_loader(
        self,
        *,
        split: Literal["train", "eval"],
        device: str,
        batch_size: int,
        dist_state: DistributedState | None = None,
        seed: int | None = None,
    ) -> DataLoader[Any]:
        """Rebuild a `DataLoader` for the requested split via the experiment module."""
        return _run_module(self.kind).build_loader(
            self.target_cfg,
            self.data_cfg,
            split=split,
            device=device,
            batch_size=batch_size,
            dist_state=dist_state,
            seed=seed,
        )

    def make_run_batch(self) -> RunBatch:
        """Rebuild the `RunBatch` callable via the experiment module's `make_run_batch`."""
        return _run_module(self.kind).make_run_batch(self.target_cfg)

    def load_model(self) -> ComponentModel:
        """Materialize the `ComponentModel` from the saved checkpoint and configs."""
        return load_component_model_from_checkpoint(
            ci_config=self.pd_config.ci_config,
            sigmoid_type=self.pd_config.sigmoid_type,
            decomposition_targets=self.pd_config.decomposition_targets,
            identity_decomposition_targets=self.pd_config.identity_decomposition_targets,
            checkpoint_path=self.checkpoint_path,
            target_model=self.build_target(),
            run_batch=self.make_run_batch(),
            tied_weights=self.pd_config.tied_weights,
        )
