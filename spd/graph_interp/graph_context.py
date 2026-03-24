"""Gather related components from attribution graph."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from spd.dataset_attributions.storage import DatasetAttributionEntry
from spd.graph_interp.ordering import parse_component_key
from spd.graph_interp.schemas import LabelResult
from spd.harvest.storage import CorrelationStorage


@dataclass
class RelatedComponent:
    component_key: str
    attribution: float
    pmi: float | None
    label: str | None
    summary: str | None


GetAttributed = Callable[[str, int, Literal["positive", "negative"]], list[DatasetAttributionEntry]]


def get_related_components(
    component_key: str,
    get_attributed: GetAttributed,
    correlation_storage: CorrelationStorage,
    labels_so_far: dict[str, LabelResult],
    k: int,
) -> list[RelatedComponent]:
    """Top-K components connected via attribution, enriched with PMI and labels."""
    my_layer, _ = parse_component_key(component_key)

    pos = get_attributed(component_key, k * 2, "positive")
    neg = get_attributed(component_key, k * 2, "negative")

    candidates = pos + neg
    candidates.sort(key=lambda e: abs(e.value), reverse=True)
    candidates = candidates[:k]

    result = []
    for e in candidates:
        r_layer, _ = parse_component_key(e.component_key)
        assert r_layer != my_layer, (
            f"Same-layer component {e.component_key} in related list for {component_key}"
        )

        label_result = labels_so_far.get(e.component_key)
        result.append(
            RelatedComponent(
                component_key=e.component_key,
                attribution=e.value,
                pmi=correlation_storage.pmi(component_key, e.component_key),
                label=label_result.label if label_result else None,
                summary=label_result.summary_for_neighbors if label_result else None,
            )
        )

    return result
