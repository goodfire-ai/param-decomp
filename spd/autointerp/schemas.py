"""Data types for autointerp pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from spd.settings import SPD_OUT_DIR

# Base directory for autointerp data
AUTOINTERP_DATA_DIR = SPD_OUT_DIR / "autointerp"


def get_autointerp_dir(decomposition_id: str) -> Path:
    """Get the top-level autointerp directory for an SPD run."""
    return AUTOINTERP_DATA_DIR / decomposition_id


def get_autointerp_subrun_dir(decomposition_id: str, autointerp_run_id: str) -> Path:
    """Get the directory for a specific autointerp run (timestamped subdirectory)."""
    return get_autointerp_dir(decomposition_id) / autointerp_run_id


DecompositionMethod = Literal["spd", "clt", "transcoder"]

DECOMPOSITION_DESCRIPTIONS: dict[DecompositionMethod, str] = {
    "spd": (
        "Each component is a rank-1 parameter vector learned by Stochastic Parameter "
        "Decomposition (SPD). A weight matrix W is decomposed as a sum of outer products "
        "W ≈ Σ u_i v_i^T. Each component has a causal importance (CI) value predicted per "
        "token position: CI near 1 means the component is essential at that position, CI near "
        "0 means it can be ablated without affecting output. A component 'fires' when its CI "
        "is high."
    ),
    "clt": (
        "Each component is a feature from a Cross-Layer Transcoder (CLT). CLTs learn sparse, "
        "interpretable features that map activations at one layer to contributions at another. "
        "A component 'fires' when its activation magnitude is high."
    ),
    "transcoder": (
        "Each component is a feature from a Transcoder, which learns a sparse dictionary of "
        "linear transformations mapping MLP inputs to MLP outputs. A component 'fires' when "
        "its encoder activation is above threshold."
    ),
}


@dataclass
class ModelMetadata:
    n_blocks: int
    model_class: str
    dataset_name: str
    layer_descriptions: dict[str, str]
    seq_len: int
    decomposition_method: DecompositionMethod


@dataclass
class InterpretationResult:
    component_key: str
    label: str
    reasoning: str
    raw_response: str
    prompt: str
