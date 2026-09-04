"""Read the stable training facts needed by offline consumers of an LM run.

`load_deliverable` resolves target structure, CI definition, datasets, and seed from the
finished run without restoring its checkpoint. It accepts the normalized deliverable file
and the current launch config used by older runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ConfigDict

from param_decomp.core.base_config import BaseConfig
from param_decomp.core.built_run import LAUNCH_CONFIG_FILENAME
from param_decomp.experiments.lm.config import (
    LMCIFnArch,
    LMDataConfig,
    LMDecompositionConfig,
    LMTargetConfig,
    resolve_decomposition,
    resolve_lm_ci_arch,
)
from param_decomp.experiments.lm.resolved import AnyLMTargetConfig, ResolvedLMData
from param_decomp.infra.dataset_store import resolve_dataset_ref

DELIVERABLE_FILENAME = "deliverable.yaml"


class _ScheduleSeed(BaseConfig):
    model_config = ConfigDict(extra="ignore")

    seed: int


@dataclass(frozen=True)
class ResolvedDeliverable:
    """Target, CI definition, datasets, and seed fixed by a finished training run."""

    target: AnyLMTargetConfig
    ci_fn: LMCIFnArch
    data: ResolvedLMData
    seed: int


def _mapping(raw: object, field: str) -> dict[str, Any]:
    assert isinstance(raw, dict), f"stored run {field} must be a mapping"
    return raw


def product_document(run_dir: Path) -> Path:
    normalized = run_dir / DELIVERABLE_FILENAME
    return normalized if normalized.is_file() else run_dir / LAUNCH_CONFIG_FILENAME


def load_deliverable(run_dir: Path, data_root: Path) -> ResolvedDeliverable:
    """Resolve the current product schema from a normalized product or current run pin."""
    raw = _mapping(yaml.safe_load(product_document(run_dir).read_text()), "config")
    target_raw = _mapping(raw.get("target"), "target")
    # Runs authored before attention routing became explicit used the same adaptive
    # cuDNN/XLA choice now named ``auto``. Normalize that historical product fact at the
    # storage boundary while keeping new authored configs strict.
    if "attention_implementation" not in target_raw:
        target_raw = {**target_raw, "attention_implementation": "auto"}
    target_config = LMTargetConfig.model_validate(target_raw)
    decomposition = LMDecompositionConfig.model_validate(
        _mapping(raw.get("decomposition"), "decomposition")
    )
    data = LMDataConfig.model_validate(_mapping(raw.get("data"), "data"))
    schedule = _ScheduleSeed.model_validate(_mapping(raw.get("pd"), "pd"))

    resolved = resolve_decomposition(target_config, decomposition, data_root)
    ci_fn = resolve_lm_ci_arch(resolved.tree, decomposition.ci, resolved.grammar)
    return ResolvedDeliverable(
        target=resolved.target,
        ci_fn=ci_fn,
        data=ResolvedLMData(
            dir=resolve_dataset_ref(data.train, data_root),
            eval_dir=resolve_dataset_ref(data.eval, data_root),
        ),
        seed=schedule.seed,
    )
