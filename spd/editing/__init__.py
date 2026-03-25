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
    inspect_component,
    measure_kl,
    measure_token_probs,
    parse_component_key,
    search_by_token_pmi,
    search_interpretations,
)
from spd.editing.compare import ExampleDiff, TokenDiff, TrainResult, train_and_compare
from spd.editing.component_trainer import ComponentTrainer
from spd.editing.viz import render_edit_comparison

__all__ = [
    "AblationEffect",
    "AlignmentResult",
    "ComponentMatch",
    "ComponentTrainer",
    "ExampleDiff",
    "TokenDiff",
    "TrainResult",
    "train_and_compare",
    "render_edit_comparison",
    "ComponentVectors",
    "EditableModel",
    "ForwardFn",
    "TokenGroupShift",
    "TokenPMIMatch",
    "UnembedMatch",
    "generate",
    "inspect_component",
    "measure_kl",
    "measure_token_probs",
    "parse_component_key",
    "search_by_token_pmi",
    "search_interpretations",
]
