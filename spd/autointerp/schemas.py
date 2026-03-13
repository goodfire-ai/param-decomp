"""Data types for autointerp pipeline."""

from dataclasses import dataclass
from pathlib import Path

from spd.settings import SPD_OUT_DIR

# Base directory for autointerp data
AUTOINTERP_DATA_DIR = SPD_OUT_DIR / "autointerp"


def get_autointerp_dir(decomposition_id: str) -> Path:
    """Get the top-level autointerp directory for an SPD run."""
    return AUTOINTERP_DATA_DIR / decomposition_id


def get_autointerp_subrun_dir(decomposition_id: str, autointerp_run_id: str) -> Path:
    """Get the directory for a specific autointerp run (timestamped subdirectory)."""
    return get_autointerp_dir(decomposition_id) / autointerp_run_id


@dataclass
class ModelMetadata:
    n_blocks: int
    model_class: str
    dataset_name: str
    layer_descriptions: dict[str, str]
    seq_len: int


@dataclass
class InterpretationResult:
    component_key: str
    label: str
    confidence: str
    reasoning: str
    raw_response: str
    prompt: str
