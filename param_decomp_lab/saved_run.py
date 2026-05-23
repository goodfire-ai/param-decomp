"""`SavedRun`: lab-side handle to a completed PD run on disk or in W&B.

Reads the `run_meta.yaml` produced by an experiment's `run.py`, resolves the experiment's
`Reloader` class by its fully-qualified name, validates the `target` / `data` blocks
against the reloader's bound config types, and uses the reloader to rebuild the target
model, dataloaders, and `run_batch` callable for postprocessing.

Notebook-only runs (trained via `optimize(...)` without saving a `run_meta.yaml`) reload
their checkpoint with `load_component_model_from_checkpoint(...)` directly.
"""

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Self, runtime_checkable

import yaml
from torch import nn
from torch.utils.data import DataLoader

from param_decomp.base_config import BaseConfig
from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.component_model import ComponentModel
from param_decomp.configs import Cadence, PDConfig, RuntimeConfig
from param_decomp.distributed import DistributedState
from param_decomp_lab.component_model_io import load_component_model_from_checkpoint
from param_decomp_lab.experiments.utils import RUN_META_FILENAME, EvalConfig
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import resolve_config_path, resolve_run_files


@runtime_checkable
class ExperimentReloader(Protocol):
    """The per-experiment object that owns target/loader/run_batch construction.

    Every experiment's `run.py` defines a concrete class implementing this Protocol.
    The class FQN is written into `run_meta.yaml::reloader_class` so `SavedRun` can
    rebuild a run without a central registry.
    """

    target_cfg: BaseConfig
    data_cfg: BaseConfig

    @classmethod
    def from_meta(cls, meta: "RunMeta") -> Self: ...

    def build_target(self) -> nn.Module: ...

    def build_loader(
        self,
        *,
        split: Literal["train", "eval"],
        device: str,
        batch_size: int,
        dist_state: DistributedState | None = None,
        seed: int | None = None,
    ) -> DataLoader[Any]: ...

    def make_run_batch(self) -> RunBatch: ...


@dataclass(frozen=True)
class RunMeta:
    """The deserialized shape of ``run_meta.yaml`` — the resolved ExperimentConfig dump
    plus the `reloader_class` FQN. `target` and `data` remain raw dicts; the reloader
    validates them against its own bound config types."""

    reloader_class_fqn: str
    pd_config: PDConfig
    runtime_config: RuntimeConfig
    cadence: Cadence
    eval_cfg: EvalConfig | None
    target_dict: dict[str, Any]
    data_dict: dict[str, Any]

    @classmethod
    def from_path(cls, path: Path) -> "RunMeta":
        with open(path) as f:
            payload = yaml.safe_load(f)
        eval_payload = payload.get("eval")
        return cls(
            reloader_class_fqn=payload["reloader_class"],
            pd_config=PDConfig.model_validate(payload["pd"]),
            runtime_config=RuntimeConfig.model_validate(payload["runtime"]),
            cadence=Cadence.model_validate(payload["cadence"]),
            eval_cfg=EvalConfig.model_validate(eval_payload) if eval_payload is not None else None,
            target_dict=payload["target"],
            data_dict=payload["data"],
        )


def _resolve_reloader_class(fqn: str) -> type[ExperimentReloader]:
    """Import a `<module>:<class>` FQN and return the class object."""
    module_path, _, class_name = fqn.partition(":")
    assert class_name, f"reloader_class FQN must be 'module:Class', got {fqn!r}"
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


@dataclass(frozen=True)
class SavedRun:
    """A completed PD run resolved to local paths + parsed meta + the matching reloader.

    Always constructed via :meth:`from_path`. Notebook-only runs (no `run_meta.yaml`) reload
    checkpoints with `load_component_model_from_checkpoint(...)` directly.
    """

    path: Path
    meta: RunMeta
    checkpoint_path: Path
    reloader: ExperimentReloader

    @classmethod
    def from_path(cls, path: ModelPath) -> "SavedRun":
        files = resolve_run_files(
            path, config_filename=RUN_META_FILENAME, checkpoint_prefix="model"
        )
        meta = RunMeta.from_path(files.config_path)
        reloader_cls = _resolve_reloader_class(meta.reloader_class_fqn)
        reloader = reloader_cls.from_meta(meta)
        return cls(
            path=files.config_path.parent,
            meta=meta,
            checkpoint_path=files.checkpoint_path,
            reloader=reloader,
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
    def target_cfg(self) -> Any:
        return self.reloader.target_cfg

    @property
    def data_cfg(self) -> Any:
        return self.reloader.data_cfg

    # ---------- Rebuild ----------

    def load_model(self) -> ComponentModel:
        target_model = self.reloader.build_target()
        return load_component_model_from_checkpoint(
            ci_config=self.pd_config.ci_config,
            sigmoid_type=self.pd_config.sigmoid_type,
            decomposition_targets=self.pd_config.decomposition_targets,
            identity_decomposition_targets=self.pd_config.identity_decomposition_targets,
            checkpoint_path=self.checkpoint_path,
            target_model=target_model,
            run_batch=self.reloader.make_run_batch(),
            tied_weights=self.pd_config.tied_weights,
        )
