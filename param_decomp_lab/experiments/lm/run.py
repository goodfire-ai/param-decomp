"""LM PD experiment: pool-agnostic builders + `SavedLMRun` reload.

The `LMExperimentConfig` schema (target spec, data config) lives in
`param_decomp_config.lm`.

The torch training driver has been retired (the JAX single-pool trainer is
production; the torch oracle lives at git tag `torch-oracle`). What remains is the
consumer bridge that post-processing and offline-eval reach for: the pure builders
(`build_target`, `build_lm_loader`, `make_run_batch`) and `SavedLMRun`, which load a
saved LM decomposition off disk.
"""

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import DistributedState
from param_decomp_config.lm import (
    HFTarget,
    HFWeightsInVendored,
    LMDataConfig,
    LMExperimentConfig,
    LMTargetConfig,
    PretrainedTarget,
)
from param_decomp_lab.batch_and_loss_fns import make_run_batch as _make_run_batch
from param_decomp_lab.component_model_io import load_component_model
from param_decomp_lab.distributed import ensure_cached_and_call
from param_decomp_lab.experiments.lm.data import (
    collate_fn_for,
    create_lm_data_loader,
    rank_batch_size,
)
from param_decomp_lab.experiments.utils import EXPERIMENT_CONFIG_FILENAME
from param_decomp_lab.infra.paths import ModelPath, validate_path
from param_decomp_lab.infra.run_files import resolve_run_files


def _resolve_class(fqn: str) -> type:
    """Load a class from a fully-qualified name, e.g. 'transformers.LlamaForCausalLM'."""
    module_path, _, class_name = fqn.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_target(target_cfg: LMTargetConfig) -> nn.Module:
    """Load the LM target model in eval mode, dispatching on `target_cfg.spec.kind`."""
    spec = target_cfg.spec
    cls = _resolve_class(spec.model_class)
    match spec:
        case HFTarget():
            target_model = ensure_cached_and_call(cls.from_pretrained, spec.model_name)
        case PretrainedTarget():
            from param_decomp_lab.experiments.lm.pretrain.run_info import PretrainRunInfo

            run_info = ensure_cached_and_call(
                PretrainRunInfo.from_path, validate_path(spec.run_path)
            )
            # Older PretrainRunInfo objects predate model_type; default it from the model class.
            if "model_type" not in run_info.model_config_dict:
                run_info.model_config_dict["model_type"] = spec.model_class.rsplit(".", 1)[-1]
            target_model = cls.from_run_info(run_info)
        case HFWeightsInVendored():
            assert hasattr(cls, "from_hf_pretrained"), (
                f"HFWeightsInVendored target requires {spec.model_class!r} to expose a "
                "`from_hf_pretrained` classmethod"
            )
            target_model = ensure_cached_and_call(cls.from_hf_pretrained, spec.model_name)
    if target_cfg.activation_checkpointing:
        assert hasattr(target_model, "enable_activation_checkpointing"), (
            f"activation_checkpointing=True but {type(target_model).__name__} has no "
            "`enable_activation_checkpointing()` method"
        )
        target_model.enable_activation_checkpointing()
    if target_cfg.weights_dtype == "bfloat16":
        # Frozen target only — make_components creates V/U as fp32 nn.Parameters regardless,
        # so componentizing after this keeps the trained components in fp32.
        target_model = target_model.to(torch.bfloat16)
    target_model.eval()
    return target_model


def build_lm_loader(
    target_cfg: LMTargetConfig,
    data_cfg: LMDataConfig,
    *,
    split: Literal["train", "eval"],
    device: str,
    batch_size: int,
    dist_state: DistributedState | None = None,
    seed: int | None = None,
) -> DataLoader[Any]:
    """LM `DataLoader` for the requested split.

    The eval seed is offset by 1 so eval shuffles differently from train when both come
    from the same `pd_config.seed`.
    """
    del target_cfg, device
    effective_seed = (seed or 0) + (1 if split == "eval" else 0)
    split_name = data_cfg.eval_split if split == "eval" else data_cfg.train_split
    loader, _ = create_lm_data_loader(
        data_cfg,
        split=split_name,
        batch_size=rank_batch_size(batch_size, dist_state, label=f"{split}_batch_size"),
        seed=effective_seed,
        dist_state=dist_state,
        collate_fn=collate_fn_for(data_cfg),
    )
    return loader


def make_run_batch(target_cfg: LMTargetConfig) -> RunBatch:
    return _make_run_batch(target_cfg.output_extract)


@dataclass(frozen=True)
class SavedLMRun:
    """Handle to a completed LM PD run on disk or in W&B."""

    cfg: LMExperimentConfig
    checkpoint_path: Path

    @classmethod
    def from_path(cls, path: ModelPath) -> "SavedLMRun":
        """Resolve a run directory or W&B path into a fully-validated `SavedLMRun`."""
        files = resolve_run_files(
            path, config_filename=EXPERIMENT_CONFIG_FILENAME, checkpoint_prefix="model"
        )
        return cls(
            cfg=LMExperimentConfig.from_file(files.config_path),
            checkpoint_path=files.checkpoint_path,
        )

    def load_model(self) -> ComponentModel:
        return load_component_model(
            pd_config=self.cfg.pd,
            checkpoint_path=self.checkpoint_path,
            target_model=build_target(self.cfg.target),
            run_batch=make_run_batch(self.cfg.target),
        )
