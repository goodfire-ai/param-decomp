"""Data types for paper visualisation dashboards.

JSON-serializable types that bridge the harvest/autointerp pipeline outputs
into static dashboard data. Each DecompositionData bundles everything needed
to render a component-level comparison dashboard for one decomposition method.
"""

from pydantic import BaseModel


class TokenSpan(BaseModel):
    """A token with its firing/activation state in context."""

    token: str
    is_firing: bool
    activation: float


class ActivationExampleData(BaseModel):
    """One activation example: a window of tokens around a firing."""

    tokens: list[TokenSpan]
    center_idx: int


class TokenPMIData(BaseModel):
    """Top tokens by PMI for a component."""

    top: list[tuple[str, float]]
    bottom: list[tuple[str, float]]


class ScoreData(BaseModel):
    """Autointerp eval score for a component."""

    score: float
    n_trials: int


class ComponentDashboardData(BaseModel):
    """Everything we know about a single component, ready for the dashboard."""

    component_key: str
    layer: str
    component_idx: int

    # Harvest data
    firing_density: float
    mean_activation: float
    activation_examples: list[ActivationExampleData]
    input_token_pmi: TokenPMIData
    output_token_pmi: TokenPMIData

    # Autointerp data (None if not yet interpreted)
    label: str | None
    confidence: str | None
    reasoning: str | None

    # Scoring data (None if not yet scored)
    detection_score: ScoreData | None
    fuzzing_score: ScoreData | None


class DecompositionData(BaseModel):
    """All dashboard data for one decomposition method."""

    decomposition_id: str
    method: str  # "vpd" or "transcoder"
    base_model: str
    n_components: int
    n_layers: int
    components: list[ComponentDashboardData]


class ComparisonDashboardData(BaseModel):
    """Top-level data for a VPD-vs-transcoder comparison dashboard."""

    vpd: DecompositionData
    transcoder: DecompositionData
