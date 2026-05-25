"""Pretrained model architecture info endpoint.

Fetches target model architecture from pretrain runs, without loading checkpoints.
Used by the run picker to show architecture summaries and by the data sources tab
to show topology and raw pretrain config.
"""

from typing import Any

import wandb
from fastapi import APIRouter
from pydantic import BaseModel

from param_decomp.log import logger
from param_decomp_lab.app.backend.dependencies import DepLoadedRun
from param_decomp_lab.app.backend.utils import log_errors
from param_decomp_lab.experiments.lm.run import LMExperimentConfig, LMTargetConfig
from param_decomp_lab.experiments.utils import RUN_META_FILENAME
from param_decomp_lab.infra.run_files import resolve_config_path
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.infra.wandb import parse_wandb_run_path

router = APIRouter(prefix="/api/pretrain_info", tags=["pretrain_info"])


class BlockStructure(BaseModel):
    index: int
    attn_type: str  # "separate" or "fused"
    attn_projections: list[str]  # e.g. ["q","k","v","o"] or ["qkv","o"]
    ffn_type: str  # "glu" or "mlp"
    ffn_projections: list[str]  # e.g. ["gate","up","down"] or ["up","down"]


class TopologyInfo(BaseModel):
    n_blocks: int
    block_structure: list[BlockStructure]


class PretrainInfoResponse(BaseModel):
    model_type: str
    summary: str
    dataset_short: str | None
    target_model_config: dict[str, Any] | None
    pretrain_config: dict[str, Any] | None
    pretrain_wandb_path: str | None
    topology: TopologyInfo | None


def _load_pretrain_configs(pretrain_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load model config and training config from a pretrain run, config files only."""
    import yaml

    entity, project, run_id = parse_wandb_run_path(pretrain_path)

    cache_dir = PARAM_DECOMP_OUT_DIR / "pretrain_cache" / f"{project}-{run_id}"
    model_config_path = cache_dir / "model_config.yaml"
    config_path = cache_dir / "final_config.yaml"

    if not model_config_path.exists() or not config_path.exists():
        logger.info(f"[pretrain_info] Downloading pretrain configs for {pretrain_path}")
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        for f in run.files():
            if f.name in ("model_config.yaml", "final_config.yaml"):
                f.download(root=str(cache_dir), exist_ok=True)

    assert model_config_path.exists(), f"model_config.yaml not found at {model_config_path}"
    assert config_path.exists(), f"final_config.yaml not found at {config_path}"

    with open(model_config_path) as f:
        target_model_config = yaml.safe_load(f)
    with open(config_path) as f:
        pretrain_config = yaml.safe_load(f)

    return target_model_config, pretrain_config


_MODEL_TYPE_TOPOLOGY: dict[str, tuple[str, list[str], str, list[str]]] = {
    # model_type -> (attn_type, attn_projs, ffn_type, ffn_projs)
    "LlamaSimple": ("separate", ["q", "k", "v", "o"], "glu", ["gate", "up", "down"]),
    "LlamaSimpleMLP": ("separate", ["q", "k", "v", "o"], "mlp", ["up", "down"]),
    "GPT2Simple": ("separate", ["q", "k", "v", "o"], "mlp", ["up", "down"]),
    "GPT2": ("fused", ["qkv", "o"], "mlp", ["up", "down"]),
    "Llama": ("separate", ["q", "k", "v", "o"], "glu", ["gate", "up", "down"]),
}


def _build_topology(model_type: str, n_blocks: int) -> TopologyInfo | None:
    topo = _MODEL_TYPE_TOPOLOGY.get(model_type)
    if topo is None:
        return None
    attn_type, attn_projs, ffn_type, ffn_projs = topo
    blocks = [
        BlockStructure(
            index=i,
            attn_type=attn_type,
            attn_projections=attn_projs,
            ffn_type=ffn_type,
            ffn_projections=ffn_projs,
        )
        for i in range(n_blocks)
    ]
    return TopologyInfo(n_blocks=n_blocks, block_structure=blocks)


def _build_summary(model_type: str, target_model_config: dict[str, Any] | None) -> str:
    """One-line architecture summary for the run picker."""
    if target_model_config is None:
        return model_type

    parts = [model_type]

    n_layer = target_model_config.get("n_layer")
    n_embd = target_model_config.get("n_embd")
    n_intermediate = target_model_config.get("n_intermediate")
    n_head = target_model_config.get("n_head")
    n_kv = target_model_config.get("n_key_value_heads")
    vocab = target_model_config.get("vocab_size")
    ctx = target_model_config.get("n_ctx")

    if n_layer is not None:
        parts.append(f"{n_layer}L")
    dims = []
    if n_embd is not None:
        dims.append(f"d={n_embd}")
    if n_intermediate is not None:
        dims.append(f"ff={n_intermediate}")
    if dims:
        parts.append(" ".join(dims))
    heads = []
    if n_head is not None:
        heads.append(f"{n_head}h")
    if n_kv is not None and n_kv != n_head:
        heads.append(f"{n_kv}kv")
    if heads:
        parts.append("/".join(heads))
    meta = []
    if vocab is not None:
        meta.append(f"vocab={vocab}")
    if ctx is not None:
        meta.append(f"ctx={ctx}")
    if meta:
        parts.append(" ".join(meta))

    return " · ".join(parts)


_DATASET_SHORT_NAMES: dict[str, str] = {
    "simplestories": "SS",
    "pile": "Pile",
    "tinystories": "TS",
}


def _get_dataset_short(pretrain_config: dict[str, Any] | None) -> str | None:
    """Extract a short dataset label from the pretrain config."""
    if pretrain_config is None:
        return None
    dataset_name: str = (
        pretrain_config.get("train_dataset_config", {}).get("name", "")
        or pretrain_config.get("data", {}).get("dataset_name", "")
        or pretrain_config.get("dataset", "")
    ).lower()
    for key, short in _DATASET_SHORT_NAMES.items():
        if key in dataset_name:
            return short
    return None


def _get_pretrain_info(lm_target: LMTargetConfig) -> PretrainInfoResponse:
    """Extract pretrain info from an LM target config."""
    from param_decomp_lab.experiments.lm.run import PretrainedTarget

    spec = lm_target.spec
    model_class_name = spec.model_class
    model_type = model_class_name.rsplit(".", 1)[-1]

    target_model_config: dict[str, Any] | None = None
    pretrain_config: dict[str, Any] | None = None
    pretrain_wandb_path: str | None = None

    if isinstance(spec, PretrainedTarget):
        pretrain_path = str(spec.run_path)
        try:
            pretrain_wandb_path = pretrain_path
            target_model_config, pretrain_config = _load_pretrain_configs(pretrain_path)
            if "model_type" in target_model_config:
                model_type = target_model_config["model_type"]
        except Exception:
            logger.exception(
                f"[pretrain_info] Failed to load pretrain configs from {pretrain_path}"
            )

    n_blocks = target_model_config.get("n_layer", 0) if target_model_config else 0
    topology = _build_topology(model_type, n_blocks)
    summary = _build_summary(model_type, target_model_config)
    dataset_short = _get_dataset_short(pretrain_config)

    return PretrainInfoResponse(
        model_type=model_type,
        summary=summary,
        dataset_short=dataset_short,
        target_model_config=target_model_config,
        pretrain_config=pretrain_config,
        pretrain_wandb_path=pretrain_wandb_path,
        topology=topology,
    )


@router.get("")
@log_errors
def get_pretrain_info_for_run(wandb_path: str) -> PretrainInfoResponse:
    """Get pretrained model architecture info for a PD run.

    Fetches only config files (no checkpoints) for efficiency.
    """
    cfg = LMExperimentConfig.from_file(
        resolve_config_path(wandb_path, config_filename=RUN_META_FILENAME)
    )
    return _get_pretrain_info(cfg.target)


@router.get("/loaded")
@log_errors
def get_pretrain_info_for_loaded_run(loaded: DepLoadedRun) -> PretrainInfoResponse:
    """Get pretrained model architecture info for the currently loaded run.

    Uses the already-loaded LM config (no additional wandb downloads).
    """
    return _get_pretrain_info(loaded.lm_target)
