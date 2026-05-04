"""Precompute QK contribution data for explicit (q, k) component pairs.

Reads a YAML/JSON config specifying wandb_path, layer, prompts, and pairs.
Outputs a JSON file with the W tensor (per head, per offset), component activations,
and alive-only attention for each prompt.

Usage:
    python -m param_decomp.scripts.interactive_qk_contributions.compute_pair_data path/to/config.yaml
"""

import json
from pathlib import Path

import fire
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from param_decomp.app.backend.app_tokenizer import AppTokenizer
from param_decomp.base_config import BaseConfig
from param_decomp.log import logger
from param_decomp.models.component_model import ComponentModel, ParamDecompRunInfo
from param_decomp.param_decomp_types import ModelPath
from param_decomp.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from param_decomp.scripts.interactive_qk_contributions.compute_data import (
    _compute_qk_tensors,
    _make_label,
)
from param_decomp.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent


class PairDataConfig(BaseConfig):
    wandb_path: ModelPath
    layer: int
    prompts: list[str]
    pairs: list[tuple[int, int]]


def compute_pair_data(config_path: str, output: str | None = None) -> None:
    config = PairDataConfig.from_file(config_path)

    _entity, _project, run_id = parse_wandb_run_path(str(config.wandb_path))
    out_path = Path(output) if output is not None else SCRIPT_DIR / "out" / run_id / "pairs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_info = ParamDecompRunInfo.from_path(config.wandb_path)
    model_config = run_info.config

    model = ComponentModel.from_run_info(run_info)
    model.eval()

    target_model = model.target_model
    assert isinstance(target_model, LlamaSimpleMLP)
    for blk in target_model._h:
        blk.attn.flash_attention = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    assert model_config.tokenizer_name is not None
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    app_tokenizer = AppTokenizer(tokenizer)

    assert 0 <= config.layer < len(target_model._h), f"layer {config.layer} out of range"

    alive_q = sorted({q for q, _ in config.pairs})
    alive_k = sorted({k for _, k in config.pairs})

    logger.info(
        f"Layer {config.layer}: {len(alive_q)} q components, {len(alive_k)} k components, "
        f"{len(config.pairs)} pairs, {len(config.prompts)} prompts"
    )

    prompts_out: list[dict[str, object]] = []
    for p_idx, prompt in enumerate(config.prompts):
        encoded: list[int] = tokenizer.encode(prompt, add_special_tokens=False)  # pyright: ignore[reportAssignmentType]
        input_ids = torch.tensor([encoded], device=device)
        tokens = app_tokenizer.get_spans(encoded)
        label = _make_label(prompt)

        tensors = _compute_qk_tensors(
            model, target_model, input_ids, config.layer, alive_q, alive_k
        )
        W = tensors["W"]
        q_acts = tensors["q_acts"]
        k_acts = tensors["k_acts"]
        component_attn = tensors["component_attn"]
        assert isinstance(W, np.ndarray)
        assert isinstance(q_acts, np.ndarray)
        assert isinstance(k_acts, np.ndarray)
        assert isinstance(component_attn, np.ndarray)

        prompts_out.append(
            {
                "tokens": tokens,
                "label": label,
                "layer_idx": config.layer,
                "n_heads": tensors["n_heads"],
                "head_dim": tensors["head_dim"],
                "scale": tensors["scale"],
                "alive_q": alive_q,
                "alive_k": alive_k,
                "pairs": [list(p) for p in config.pairs],
                "W": W.tolist(),
                "q_acts": q_acts.tolist(),
                "k_acts": k_acts.tolist(),
                "component_model_attn": component_attn.tolist(),
            }
        )
        logger.info(f"[{p_idx + 1}/{len(config.prompts)}] {len(tokens)} tokens: {label}")

    with open(out_path, "w") as f:
        json.dump({"prompts": prompts_out}, f)
    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.info(f"Saved {out_path} ({len(prompts_out)} prompts, {size_mb:.1f} MB)")


if __name__ == "__main__":
    fire.Fire(compute_pair_data)
