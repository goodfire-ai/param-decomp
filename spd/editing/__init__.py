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
from spd.editing.component_trainer import ComponentTrainer

__all__ = [
    "AblationEffect",
    "AlignmentResult",
    "ComponentMatch",
    "ComponentTrainer",
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
