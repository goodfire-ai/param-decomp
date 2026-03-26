"""Utilities for SPD component editing."""

import re
from dataclasses import dataclass

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.repo import InterpRepo
from spd.configs import Config
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


def load_model(
    wandb_path: str, device: str = "cuda"
) -> tuple[ComponentModel, AppTokenizer, Config]:
    """Load a ComponentModel + tokenizer + config from a wandb path."""
    run_info = SPDRunInfo.from_path(wandb_path)
    model = ComponentModel.from_run_info(run_info).to(device).eval()
    assert run_info.config.tokenizer_name is not None
    tokenizer = AppTokenizer.from_pretrained(run_info.config.tokenizer_name)
    return model, tokenizer, run_info.config
