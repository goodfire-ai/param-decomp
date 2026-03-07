"""Data types and path helpers for graph interpretation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from spd.settings import SPD_OUT_DIR

GRAPH_INTERP_DIR = SPD_OUT_DIR / "graph_interp"


def get_graph_interp_dir(decomposition_id: str) -> Path:
    return GRAPH_INTERP_DIR / decomposition_id


def get_graph_interp_subrun_dir(decomposition_id: str, subrun_id: str) -> Path:
    return get_graph_interp_dir(decomposition_id) / subrun_id


@dataclass
class LabelResult:
    component_key: str
    label: str
    confidence: str
    reasoning: str
    raw_response: str
    prompt: str


@dataclass
class PromptEdge:
    component_key: str
    related_key: str
    pass_name: Literal["output", "input"]
    attribution: float
    related_label: str | None
    related_confidence: str | None
