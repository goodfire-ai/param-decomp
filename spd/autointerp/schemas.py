"""Data types for autointerp pipeline."""

from dataclasses import dataclass
from pathlib import Path

from spd.settings import SPD_OUT_DIR

# Base directory for autointerp data
AUTOINTERP_DATA_DIR = SPD_OUT_DIR / "autointerp"


def get_autointerp_dir(wandb_run_id: str) -> Path:
    """Get the autointerp (interpretations) directory for a run."""
    return AUTOINTERP_DATA_DIR / wandb_run_id


@dataclass
class ArchitectureInfo:
    n_blocks: int
    c_per_layer: dict[str, int]  # Maps layer name -> number of components
    model_class: str
    dataset_name: str
    tokenizer_name: str
    seq_len: int
    context_tokens_per_side: int


@dataclass
class InterpretationResult:
    component_key: str
    label: str
    confidence: str
    reasoning: str
    raw_response: str
    prompt: str
