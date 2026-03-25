"""Component-level model editing for VPD decompositions."""

from spd.editing._editing import (
    AblationEffect,
    AlignmentResult,
    ComponentMatch,
    ComponentVectors,
    EditableModel,
    ForwardFn,
    TokenGroupShift,
    TokenPMIMatch,
    UnembedMatch,
    generate,
    get_ci,
    get_component_activations,
    inspect_component,
    load_model,
    measure_kl,
    measure_token_probs,
    parse_component_key,
    search_by_token_pmi,
    search_interpretations,
)
from spd.editing.compare import ExampleDiff, TokenDiff, TrainResult, train_and_compare
from spd.editing.component_trainer import train_write_delta, write_edit
from spd.editing.lora_baseline import LoRATrainer
from spd.editing.viz import render_edit_comparison

__all__ = [
    "AblationEffect",
    "AlignmentResult",
    "ComponentMatch",
    "ComponentVectors",
    "EditableModel",
    "ExampleDiff",
    "ForwardFn",
    "LoRATrainer",
    "TokenDiff",
    "TokenGroupShift",
    "TokenPMIMatch",
    "TrainResult",
    "UnembedMatch",
    "generate",
    "get_ci",
    "get_component_activations",
    "inspect_component",
    "load_model",
    "measure_kl",
    "measure_token_probs",
    "parse_component_key",
    "render_edit_comparison",
    "search_by_token_pmi",
    "search_interpretations",
    "train_and_compare",
    "train_write_delta",
    "write_edit",
]
