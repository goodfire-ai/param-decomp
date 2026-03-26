"""Component-level model editing for VPD decompositions."""

from spd.editing.compare import ExampleDiff, TokenDiff, compute_diffs
from spd.editing.component_trainer import train_write_vector, write_edit
from spd.editing.lora_baseline import LoRATrainer
from spd.editing.utils import (
    ComponentMatch,
    load_model,
    parse_component_key,
    search_interpretations,
)
from spd.editing.viz import render_edit_comparison

__all__ = [
    "ComponentMatch",
    "ExampleDiff",
    "LoRATrainer",
    "TokenDiff",
    "compute_diffs",
    "load_model",
    "parse_component_key",
    "render_edit_comparison",
    "search_interpretations",
    "train_write_vector",
    "write_edit",
]
