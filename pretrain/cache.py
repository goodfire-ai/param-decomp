"""Write a pretrained target to the decomposition trainer's pretrain-cache layout.

`param_decomp.llama_simple_mlp.load_target_from_pretrain_cache` reads a cache dir
`pretrain_cache/<project>-<run_id>/` holding exactly one `model_step_<N>.safetensors`
plus a `model_config.yaml` (the torch `LlamaSimpleMLPConfig` dump). This module emits
that layout from a freshly-pretrained model so a target is decomposable with no
conversion. The run dir's own `ckpts/` (orbax) is the resume substrate; the cache is the
hand-off artifact.
"""

from pathlib import Path

import numpy as np
import yaml
from safetensors.numpy import save_file

from pretrain.config import PretrainConfig
from pretrain.models import PretrainModel


def cache_dir_for(out_root: Path, project: str, run_id: str) -> Path:
    return out_root / "pretrain_cache" / f"{project}-{run_id}"


def write_pretrain_cache(
    cache_dir: Path, model: PretrainModel, model_config_dict: dict[str, object], step: int
) -> Path:
    """Write `model_step_<step>.safetensors` + `model_config.yaml` into `cache_dir`,
    removing any stale `model_step_*.safetensors` first (the loader asserts exactly one)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for stale in cache_dir.glob("model_step_*.safetensors"):
        stale.unlink()
    tensors = {k: np.asarray(v, dtype=np.float32) for k, v in model.state_dict().items()}
    ckpt = cache_dir / f"model_step_{step}.safetensors"
    save_file(tensors, str(ckpt))
    (cache_dir / "model_config.yaml").write_text(yaml.safe_dump(model_config_dict, sort_keys=True))
    return ckpt


def torch_model_config_dict(cfg: PretrainConfig) -> dict[str, object]:
    """The `model_config.yaml` shape `param_decomp.llama_simple_mlp` parses — the torch
    `LlamaSimpleMLPConfig` field names, with the rotary/GQA fields the loader asserts on."""
    model = cfg.model
    base: dict[str, object] = {
        "model_type": model.model_type,
        "block_size": model.block_size,
        "vocab_size": model.vocab_size,
        "n_layer": model.n_layer,
        "n_head": model.n_head,
        "n_embd": model.n_embd,
    }
    match model.model_type:
        case "GPT2Simple":
            return base | {
                "n_intermediate": model.n_intermediate,
                "flash_attention": model.flash_attention,
            }
        case "LlamaSimple" | "LlamaSimpleMLP":
            return base | {
                "n_intermediate": model.n_intermediate,
                "mlp_bias": model.mlp_bias,
                "attn_bias": model.attn_bias,
                "rotary_adjacent_pairs": model.rotary_adjacent_pairs,
                "rotary_dim": model.rotary_dim,
                "rotary_base": model.rotary_base,
                "n_ctx": model.n_ctx,
                "n_key_value_heads": model.n_key_value_heads,
                "use_grouped_query_attention": model.use_grouped_query_attention,
                "flash_attention": model.flash_attention,
                "rms_norm_eps": model.rms_norm_eps,
            }
