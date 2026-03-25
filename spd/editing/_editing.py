"""Utilities for SPD component editing."""

import torch
from jaxtyping import Float, Int
from torch import Tensor

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.models.component_model import ComponentModel, SPDRunInfo


def parse_component_key(key: str) -> tuple[str, int]:
    """'h.1.mlp.c_fc:802' -> ('h.1.mlp.c_fc', 802)."""
    layer, idx_str = key.rsplit(":", 1)
    return layer, int(idx_str)


def load_model(wandb_path: str, device: str = "cuda") -> tuple[ComponentModel, AppTokenizer]:
    """Load a ComponentModel + tokenizer from a wandb path."""
    run_info = SPDRunInfo.from_path(wandb_path)
    model = ComponentModel.from_run_info(run_info).to(device).eval()
    assert run_info.config.tokenizer_name is not None
    tokenizer = AppTokenizer.from_pretrained(run_info.config.tokenizer_name)
    return model, tokenizer


def get_ci(
    model: ComponentModel,
    tokens: Int[Tensor, " seq"],
) -> dict[str, Float[Tensor, " seq C"]]:
    """Get CI values for all components at all positions. Returns {layer: [seq, C]}."""
    with torch.no_grad():
        out = model(tokens.unsqueeze(0), cache_type="input")
        ci = model.calc_causal_importances(
            pre_weight_acts=out.cache,
            sampling="continuous",
            detach_inputs=False,
        )
    return {layer: vals.squeeze(0) for layer, vals in ci.lower_leaky.items()}


def get_component_activations(
    model: ComponentModel,
    tokens: Int[Tensor, " seq"],
    key: str,
) -> Float[Tensor, " seq"]:
    """Component activation (v_c^T @ x) at each sequence position."""
    layer, idx = parse_component_key(key)
    with torch.no_grad():
        out = model(tokens.unsqueeze(0), cache_type="input")
    pre_weight_acts = out.cache[layer]  # [1, seq, d_in]
    comp = model.components[layer]
    return (pre_weight_acts @ comp.V[:, idx]).squeeze(0)  # [seq]
