"""Component-level model editing for VPD decompositions."""

# Re-export everything from the main module so `from spd.editing import ...` still works
from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.compute import OptimizedPromptAttributionResult
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

__all__ = [
    "AblationEffect",
    "AlignmentResult",
    "AppTokenizer",
    "OptimizedPromptAttributionResult",
    "ComponentMatch",
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
