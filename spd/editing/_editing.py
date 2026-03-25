"""Utilities for SPD component editing."""

import re
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.repo import InterpRepo
from spd.harvest.repo import HarvestRepo
from spd.models.component_model import ComponentModel, SPDRunInfo


def parse_component_key(key: str) -> tuple[str, int]:
    """'h.1.mlp.c_fc:802' -> ('h.1.mlp.c_fc', 802)."""
    layer, idx_str = key.rsplit(":", 1)
    return layer, int(idx_str)


@dataclass
class ComponentMatch:
    key: str
    label: str
    firing_density: float
    mean_activations: dict[str, float]


def search_interpretations(
    harvest: HarvestRepo,
    interp: InterpRepo,
    pattern: str,
    min_firing_density: float = 0.0,
) -> list[ComponentMatch]:
    """Search component interpretations by regex on label. Sorted by firing density desc."""
    all_interps = interp.get_all_interpretations()
    summary = harvest.get_summary()

    matches = []
    for key, result in all_interps.items():
        if key not in summary:
            continue
        if not re.search(pattern, result.label, re.IGNORECASE):
            continue
        s = summary[key]
        if s.firing_density < min_firing_density:
            continue
        matches.append(
            ComponentMatch(
                key=key,
                label=result.label,
                firing_density=s.firing_density,
                mean_activations=s.mean_activations,
            )
        )

    matches.sort(key=lambda m: -m.firing_density)
    return matches


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
