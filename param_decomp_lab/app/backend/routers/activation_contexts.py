"""Activation contexts endpoints.

These endpoints serve activation context data from the harvest pipeline output.
"""

from collections import defaultdict
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from param_decomp_lab.app.backend.app_tokenizer import AppTokenizer
from param_decomp_lab.app.backend.dependencies import DepLoadedRun
from param_decomp_lab.app.backend.schemas import (
    SubcomponentActivationContexts,
    SubcomponentMetadata,
)
from param_decomp_lab.app.backend.utils import log_errors
from param_decomp_lab.harvest.schemas import ComponentData

router = APIRouter(prefix="/api/activation_contexts", tags=["activation_contexts"])


def example_to_activation_contexts(
    comp: ComponentData, tokenizer: AppTokenizer, limit: int | None = None
) -> SubcomponentActivationContexts:
    examples = comp.activation_examples
    if limit is not None:
        examples = examples[:limit]

    mean_ci = comp.mean_activations["causal_importance"]
    example_tokens = [tokenizer.get_spans(ex.token_ids) for ex in examples]
    example_ci = [ex.activations["causal_importance"] for ex in examples]
    example_component_acts = [ex.activations["component_activation"] for ex in examples]

    return SubcomponentActivationContexts(
        subcomponent_idx=comp.component_idx,
        # We might consider replacing mean_ci here with firing density
        mean_ci=mean_ci,
        example_tokens=example_tokens,
        example_ci=example_ci,
        example_component_acts=example_component_acts,
    )


@router.get("/summary")
@log_errors
def get_activation_contexts_summary(
    loaded: DepLoadedRun,
) -> dict[str, list[SubcomponentMetadata]]:
    """Return lightweight summary of activation contexts (just idx + mean_ci per component)."""
    if loaded.harvest is None:
        raise HTTPException(status_code=404, detail="No harvest data available")
    summary_data = loaded.harvest.get_summary()

    summary: dict[str, list[SubcomponentMetadata]] = defaultdict(list)
    for comp in summary_data.values():
        canonical_layer = loaded.topology.target_to_canon(comp.layer)
        summary[canonical_layer].append(
            SubcomponentMetadata(
                subcomponent_idx=comp.component_idx,
                mean_ci=comp.mean_activations["causal_importance"],
            )
        )

    # Sort by mean CI descending within each layer
    for layer in summary:
        summary[layer].sort(key=lambda x: x.mean_ci, reverse=True)

    return dict(summary)


@router.get("/{layer}/{component_idx}")
@log_errors
def get_activation_context_detail(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
    limit: Annotated[int | None, Query(ge=1, description="Max examples to return")] = None,
) -> SubcomponentActivationContexts:
    """Return full activation context data for a single component.

    Args:
        limit: Maximum number of activation examples to return. If None, returns all.
               Use limit=30 for initial load, then fetch more via pagination if needed.

    TODO: Add offset parameter for pagination to allow fetching remaining examples
          after initial view is loaded.
    """
    assert loaded.harvest is not None, "No harvest data available"
    concrete_layer = loaded.topology.canon_to_target(layer)
    component_key = f"{concrete_layer}:{component_idx}"
    comp = loaded.harvest.get_component(component_key)
    if comp is None:
        raise HTTPException(status_code=404, detail=f"Component {component_key} not found")

    return example_to_activation_contexts(comp, loaded.tokenizer, limit)


class BulkActivationContextsRequest(BaseModel):
    """Request for bulk activation contexts."""

    component_keys: list[str]  # canonical keys, e.g. ["0.mlp.up:5", "1.attn.q:12"]
    limit: int = 30


@router.post("/bulk")
@log_errors
def get_activation_contexts_bulk(
    request: BulkActivationContextsRequest,
    loaded: DepLoadedRun,
) -> dict[str, SubcomponentActivationContexts]:
    """Bulk fetch activation contexts for multiple components.

    Returns a dict keyed by component_key. Components not found are omitted.
    Uses optimized bulk loader with single file handle and sorted seeks.
    """

    # Translate canonical component keys to concrete paths for harvest lookup
    def _to_concrete_key(canonical_key: str) -> str:
        layer, idx = canonical_key.rsplit(":", 1)
        concrete = loaded.topology.canon_to_target(layer)
        return f"{concrete}:{idx}"

    assert loaded.harvest is not None, "No harvest data available"
    concrete_to_canonical = {_to_concrete_key(k): k for k in request.component_keys}
    concrete_keys = list(concrete_to_canonical.keys())
    components = loaded.harvest.get_components_bulk(concrete_keys)

    # Convert to response format with limit applied, keyed by canonical keys
    result: dict[str, SubcomponentActivationContexts] = {}
    for concrete_key, comp in components.items():
        canonical_key = concrete_to_canonical[concrete_key]
        result[canonical_key] = example_to_activation_contexts(
            comp, loaded.tokenizer, request.limit
        )

    return result
