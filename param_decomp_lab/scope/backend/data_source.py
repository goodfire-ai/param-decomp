"""Frozen scope API contract: response models + the ScopeDataSource protocol."""

from typing import Literal, Protocol

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
    examples: list[ActivationExample]


class ScopeDataSource(Protocol):
    """Paged access to per-component postprocess data.

    Implementations must never materialize O(n_components) detail objects per
    request; listing sort/filter operates on compact per-site columns and
    component detail is produced for one idx at a time.
    """

    def catalog(self) -> CatalogResponse: ...

    def list_components(
        self, run_id: str, site: str, sort: SortKey, page: int, page_size: int, q: str
    ) -> ComponentListResponse: ...

    def component_detail(self, run_id: str, site: str, idx: int) -> ComponentDetail: ...

    def create_label(self, run_id: str, site: str, idx: int) -> ComponentLabel: ...
