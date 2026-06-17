"""Data types for harvest pipeline."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel

from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR


def get_harvest_dir(decomposition_id: str) -> Path:
    """Base harvest dir for a decomposition."""
    return PARAM_DECOMP_OUT_DIR / "runs" / decomposition_id / "harvest"


def get_harvest_subrun_dir(decomposition_id: str, subrun_id: str) -> Path:
    """Subrun dir for a specific harvest invocation."""
    return get_harvest_dir(decomposition_id) / subrun_id


@dataclass
class HarvestBatch:
    """Per-batch component statistics fed to the `Harvester`.

    The JAX worker converts a frozen forward pass into one of these, then feeds it to
    the Harvester.

    firings/activations are keyed by layer name. activations values are keyed by
    activation type ("causal_importance", "component_activation").
    """

    tokens: Int[np.ndarray, "batch seq"]
    firings: dict[str, Bool[np.ndarray, "batch seq c"]]
    activations: dict[str, dict[str, Float[np.ndarray, "batch seq c"]]]
    output_probs: Float[np.ndarray, "batch seq vocab"]


class ActivationExample(BaseModel):
    """Activation example for a single component. no padding"""

    token_ids: list[int]
    firings: list[bool]
    activations: dict[str, list[float]]


class ComponentTokenPMI(BaseModel):
    top: list[tuple[int, float]]
    bottom: list[tuple[int, float]]


class ComponentSummary(BaseModel):
    """Lightweight summary of a component (for /summary endpoint)."""

    layer: str
    component_idx: int
    firing_density: float
    mean_activations: dict[str, float]
    """Key is activation type, (e.g. "causal_importance", "component_activation", etc.)"""


class ComponentData(BaseModel):
    component_key: str
    layer: str
    component_idx: int
    mean_activations: dict[str, float]
    firing_density: float
    activation_examples: list[ActivationExample]
    input_token_pmi: ComponentTokenPMI
    output_token_pmi: ComponentTokenPMI
