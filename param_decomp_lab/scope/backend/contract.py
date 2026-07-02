"""Frozen scope API contract: the response models the backend serves and the frontend mirrors."""

from typing import Literal

from pydantic import BaseModel

SortKey = Literal["mean_ci", "density", "max_act", "unlabeled_first"]
SubrunStatus = Literal["present", "in_flight"]


class ScopeNotFoundError(Exception):
    """Unknown run/site/component, or a site with no present subrun yet."""


class SubrunEntry(BaseModel):
    subrun_id: str
    status: SubrunStatus
    n_batches: int
    progress: float


class SiteEntry(BaseModel):
    site: str
    n_components: int
    n_labeled: int
    subruns: list[SubrunEntry]


class RunEntry(BaseModel):
    run_id: str
    sites: list[SiteEntry]


class CatalogResponse(BaseModel):
    runs: list[RunEntry]


class ComponentRow(BaseModel):
    idx: int
    mean_ci: float
    density: float
    max_act: float
    label: str | None


class ComponentListResponse(BaseModel):
    total: int
    page: int
    items: list[ComponentRow]


class ComponentLabel(BaseModel):
    text: str
    model: str
    cost_usd: float
    created_at: str


class ActivationExample(BaseModel):
    tokens: list[str]
    acts: list[float]
    cis: list[float]
    max_act: float


class ComponentDetail(BaseModel):
    idx: int
    rank: int
    """Position in the site's mean-CI ordering (0 = highest mean CI)."""
    prev_idx: int | None
    next_idx: int | None
    density: float
    max_act: float
    mean_ci: float
    label: ComponentLabel | None
    input_pmi: list[tuple[str, float]]
    output_pmi: list[tuple[str, float]]
    n_examples: int
    """Total activating examples stored for this component (the full reservoir pool)."""
    example_page: int
    examples: list[ActivationExample]
    """One page of examples, ranked by peak causal importance (highest first)."""
