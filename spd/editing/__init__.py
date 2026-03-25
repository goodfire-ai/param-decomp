"""Component-level model editing for VPD decompositions."""

from spd.editing._editing import (
    get_ci,
    get_component_activations,
    load_model,
    parse_component_key,
)
from spd.editing.compare import ExampleDiff, TokenDiff, TrainResult, train_and_compare
from spd.editing.component_trainer import train_write_delta, write_edit
from spd.editing.lora_baseline import LoRATrainer
from spd.editing.viz import render_edit_comparison

__all__ = [
    "ExampleDiff",
    "LoRATrainer",
    "TokenDiff",
    "TrainResult",
    "get_ci",
    "get_component_activations",
    "load_model",
    "parse_component_key",
    "render_edit_comparison",
    "train_and_compare",
    "train_write_delta",
    "write_edit",
]
