"""Activation contexts endpoints.

These endpoints serve activation context data from the harvest pipeline output.
"""

from collections import defaultdict
from typing import Annotated

import torch
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.compute import compute_ci_only
from spd.app.backend.dependencies import DepLoadedRun
from spd.app.backend.schemas import SubcomponentActivationContexts, SubcomponentMetadata
from spd.app.backend.utils import log_errors
from spd.harvest.schemas import ComponentData
from spd.utils.distributed_utils import get_device


class ComponentProbeRequest(BaseModel):
    """Request to probe a component's CI on custom text."""

    text: str
    layer: str
    component_idx: int


class ComponentProbeResponse(BaseModel):
    """Response with CI and subcomponent activation values for a component on custom text."""

    tokens: list[str]
    ci_values: list[float]
    subcomp_acts: list[float]
    next_token_probs: list[float | None]  # Probability of next token (last is None)


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


@router.post("/probe")
@log_errors
def probe_component(
    request: ComponentProbeRequest,
    loaded: DepLoadedRun,
) -> ComponentProbeResponse:
    """Probe a component's CI and subcomponent activation values on custom text.

    Fast endpoint for testing hypotheses about component activation.
    Only requires a single forward pass.
    """
    device = get_device()

    token_ids = loaded.tokenizer.encode(request.text)
    assert len(token_ids) > 0, "Text produced no tokens"

    tokens_tensor = torch.tensor([token_ids], device=device)

    result = compute_ci_only(
        model=loaded.model,
        tokens=tokens_tensor,
        sampling=loaded.config.sampling,
    )

    concrete_layer = loaded.topology.canon_to_target(request.layer)
    assert concrete_layer in loaded.model.components, f"Layer {request.layer} not in model"

    ci_tensor = result.ci_lower_leaky[concrete_layer]
    ci_values = ci_tensor[0, :, request.component_idx].tolist()
    spans = loaded.tokenizer.get_spans(token_ids)

    subcomp_acts_tensor = result.component_acts[concrete_layer]
    subcomp_acts = subcomp_acts_tensor[0, :, request.component_idx].tolist()

    # Get probability of next token at each position
    probs = result.target_out_probs[0]  # [seq, vocab]
    next_token_probs: list[float | None] = []
    for i in range(len(token_ids) - 1):
        next_token_id = token_ids[i + 1]
        prob = probs[i, next_token_id].item()
        next_token_probs.append(prob)
    next_token_probs.append(None)  # No next token for last position

    return ComponentProbeResponse(
        tokens=spans,
        ci_values=ci_values,
        subcomp_acts=subcomp_acts,
        next_token_probs=next_token_probs,
    )
