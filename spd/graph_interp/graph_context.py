"""Gather related components from attribution graph."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from spd.dataset_attributions.storage import DatasetAttributionEntry
from spd.graph_interp.ordering import parse_component_key
from spd.graph_interp.schemas import LabelResult
from spd.harvest.analysis import get_correlated_components
from spd.harvest.storage import CorrelationStorage


@dataclass
class RelatedComponent:
    component_key: str
    attribution: float
    label: str | None
    confidence: str | None
    jaccard: float | None
    pmi: float | None


GetAttributed = Callable[[str, int, Literal["positive", "negative"]], list[DatasetAttributionEntry]]


def get_related_components(
    component_key: str,
    get_attributed: GetAttributed,
    correlation_storage: CorrelationStorage,
    labels_so_far: dict[str, LabelResult],
    k: int,
) -> list[RelatedComponent]:
    """Top-K components connected via attribution, enriched with co-firing stats and labels."""
    my_layer, _ = parse_component_key(component_key)

    pos = get_attributed(component_key, k * 2, "positive")
    neg = get_attributed(component_key, k * 2, "negative")

    candidates = pos + neg
    candidates.sort(key=lambda e: abs(e.value), reverse=True)
    candidates = candidates[:k]

    cofiring = _build_cofiring_lookup(component_key, correlation_storage, k * 3)
    result = [_build_related(e.component_key, e.value, cofiring, labels_so_far) for e in candidates]

    for r in result:
        r_layer, _ = parse_component_key(r.component_key)
        assert r_layer != my_layer, (
            f"Same-layer component {r.component_key} in related list for {component_key}"
        )

    return result


def _build_cofiring_lookup(
    component_key: str,
    correlation_storage: CorrelationStorage,
    k: int,
) -> dict[str, tuple[float, float | None]]:
    lookup: dict[str, tuple[float, float | None]] = {}

    jaccard_results = get_correlated_components(
        correlation_storage, component_key, metric="jaccard", top_k=k
    )
    for c in jaccard_results:
        lookup[c.component_key] = (c.score, None)

    pmi_results = get_correlated_components(
        correlation_storage, component_key, metric="pmi", top_k=k
    )
    for c in pmi_results:
        if c.component_key in lookup:
            jaccard_val = lookup[c.component_key][0]
            lookup[c.component_key] = (jaccard_val, c.score)
        else:
            lookup[c.component_key] = (0.0, c.score)

    return lookup


def _build_related(
    related_key: str,
    attribution: float,
    cofiring: dict[str, tuple[float, float | None]],
    labels_so_far: dict[str, LabelResult],
) -> RelatedComponent:
    label = labels_so_far.get(related_key)
    jaccard, pmi = cofiring.get(related_key, (None, None))

    return RelatedComponent(
        component_key=related_key,
        attribution=attribution,
        label=label.label if label else None,
        confidence=label.confidence if label else None,
        jaccard=jaccard,
        pmi=pmi,
    )
