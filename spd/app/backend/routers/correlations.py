"""Component correlation and interpretation endpoints.

These endpoints serve data produced by the harvest pipeline (spd.harvest),
which computes component co-occurrence statistics, token associations, and interpretations.
"""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from spd.app.backend.dependencies import DepLoadedRun
from spd.app.backend.utils import log_errors
from spd.harvest import analysis
from spd.harvest.loaders import load_component_activation_contexts
from spd.log import logger


class CorrelatedComponent(BaseModel):
    """A component correlated with a query component."""

    component_key: str
    score: float
    count_i: int  # Subject (query component) firing count
    count_j: int  # Object (this component) firing count
    count_ij: int  # Co-occurrence count
    n_tokens: int  # Total tokens


class ComponentCorrelationsResponse(BaseModel):
    """Correlation data for a component across different metrics."""

    precision: list[CorrelatedComponent]
    recall: list[CorrelatedComponent]
    jaccard: list[CorrelatedComponent]
    pmi: list[CorrelatedComponent]
    bottom_pmi: list[CorrelatedComponent]


class TokenPRLiftPMI(BaseModel):
    """Token precision, recall, lift, and PMI lists."""

    top_recall: list[tuple[str, float]]  # [(token, value), ...] sorted desc
    top_precision: list[tuple[str, float]]  # [(token, value), ...] sorted desc
    top_lift: list[tuple[str, float]]  # [(token, lift), ...] sorted desc
    top_pmi: list[tuple[str, float]]  # [(token, pmi), ...] highest positive association
    bottom_pmi: list[tuple[str, float]] | None  # [(token, pmi), ...] highest negative association


class TokenStatsResponse(BaseModel):
    """Token stats for a component (from batch job).

    Contains both input token stats (what tokens activate this component)
    and output token stats (what tokens this component predicts).
    """

    input: TokenPRLiftPMI  # Stats for input tokens
    output: TokenPRLiftPMI  # Stats for output (predicted) tokens


router = APIRouter(prefix="/api/correlations", tags=["correlations"])


# =============================================================================
# Interpretation Endpoint
# =============================================================================


class InterpretationHeadline(BaseModel):
    """Lightweight interpretation headline for bulk fetching."""

    label: str
    confidence: str


class InterpretationDetail(BaseModel):
    """Full interpretation detail fetched on-demand."""

    reasoning: str
    prompt: str


@router.get("/interpretations")
@log_errors
def get_all_interpretations(
    loaded: DepLoadedRun,
) -> dict[str, InterpretationHeadline]:
    """Get all interpretation headlines (label + confidence only).

    Returns a dict keyed by component_key (layer:cIdx).
    Reasoning and prompt are excluded - fetch individually via
    GET /interpretations/{layer}/{component_idx} when needed.
    """
    return {
        key: InterpretationHeadline(
            label=result.label,
            confidence=result.confidence,
        )
        for key, result in loaded.harvest.interpretations.items()
    }


@router.get("/interpretations/{layer}/{component_idx}")
@log_errors
def get_interpretation_detail(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
) -> InterpretationDetail:
    """Get the full interpretation detail (reasoning + prompt).

    Returns reasoning and prompt for the specified component.
    """
    component_key = f"{layer}:{component_idx}"
    interpretations = loaded.harvest.interpretations

    if component_key not in interpretations:
        raise HTTPException(
            status_code=404,
            detail=f"No interpretation found for component {component_key}",
        )

    result = interpretations[component_key]
    return InterpretationDetail(reasoning=result.reasoning, prompt=result.prompt)


@router.post("/interpretations/{layer}/{component_idx}")
@log_errors
async def request_component_interpretation(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
) -> InterpretationHeadline:
    """Generate an interpretation for a component on-demand.

    Requires OPENROUTER_API_KEY environment variable.
    Returns the headline (label + confidence). Full detail available via GET endpoint.
    """
    import json
    import os
    from dataclasses import asdict

    from openrouter import OpenRouter

    from spd.autointerp.interpret import (
        OpenRouterModelName,
        get_architecture_info,
        interpret_component,
    )
    from spd.autointerp.schemas import get_autointerp_dir

    component_key = f"{layer}:{component_idx}"

    interpretations = loaded.harvest.interpretations

    if component_key in interpretations:
        result = interpretations[component_key]
        return InterpretationHeadline(
            label=result.label,
            confidence=result.confidence,
        )

    component_data = load_component_activation_contexts(loaded.harvest.run_id, component_key)

    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY environment variable not set",
        )

    from spd.harvest.schemas import get_activation_contexts_dir

    # Get architecture info and tokenizer
    activation_contexts_dir = get_activation_contexts_dir(loaded.harvest.run_id)
    arch = get_architecture_info(loaded.run.wandb_path, activation_contexts_dir)

    # Get token stats
    token_stats = loaded.harvest.token_stats

    input_token_stats = analysis.get_input_token_stats(
        token_stats, component_key, loaded.tokenizer, top_k=20
    )
    output_token_stats = analysis.get_output_token_stats(
        token_stats, component_key, loaded.tokenizer, top_k=50
    )
    if input_token_stats is None or output_token_stats is None:
        raise HTTPException(
            status_code=400,
            detail=f"Token stats not available for component {component_key}",
        )

    # Interpret the component
    model_name = OpenRouterModelName.GEMINI_3_FLASH_PREVIEW

    async with OpenRouter(api_key=api_key) as client:
        res = await interpret_component(
            client=client,
            model=model_name,
            component=component_data,
            arch=arch,
            tokenizer=loaded.tokenizer,
            input_token_stats=input_token_stats,
            output_token_stats=output_token_stats,
        )

    if res is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate interpretation",
        )

    result, _, _ = res

    # Save to file
    autointerp_dir = get_autointerp_dir(loaded.harvest.run_id)
    autointerp_dir.mkdir(parents=True, exist_ok=True)
    output_path = autointerp_dir / "results.jsonl"
    with open(output_path, "a") as f:
        f.write(json.dumps(asdict(result)) + "\n")

    # Update the cache
    if loaded.harvest._interpretations is None:
        loaded.harvest._interpretations = {}
    assert isinstance(loaded.harvest._interpretations, dict)
    loaded.harvest._interpretations[component_key] = result

    logger.info(f"Generated interpretation for {component_key}: {result.label}")

    return InterpretationHeadline(
        label=result.label,
        confidence=result.confidence,
    )


# =============================================================================
# Component Correlation Data Endpoints
# =============================================================================


@router.get("/token_stats/{layer}/{component_idx}")
@log_errors
def get_component_token_stats(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
    top_k: Annotated[int, Query(ge=1)],
) -> TokenStatsResponse | None:
    """Get token precision/recall/lift/PMI for a component.

    Returns stats for both input tokens (what activates this component)
    and output tokens (what this component predicts).
    Returns None if token stats haven't been harvested for this run.
    """
    token_stats = loaded.harvest.token_stats
    component_key = f"{layer}:{component_idx}"

    input_stats = analysis.get_input_token_stats(
        token_stats, component_key, loaded.tokenizer, top_k
    )
    output_stats = analysis.get_output_token_stats(
        token_stats, component_key, loaded.tokenizer, top_k
    )

    if input_stats is None or output_stats is None:
        return None

    assert input_stats.bottom_pmi is None, "Input stats should not have bottom PMI"
    assert output_stats.bottom_pmi is not None, "Output stats should have bottom PMI"

    return TokenStatsResponse(
        input=TokenPRLiftPMI(
            top_recall=input_stats.top_recall,
            top_precision=input_stats.top_precision,
            top_lift=input_stats.top_lift,
            top_pmi=input_stats.top_pmi,
            bottom_pmi=input_stats.bottom_pmi,
        ),
        output=TokenPRLiftPMI(
            top_recall=output_stats.top_recall,
            top_precision=output_stats.top_precision,
            top_lift=output_stats.top_lift,
            top_pmi=output_stats.top_pmi,
            bottom_pmi=output_stats.bottom_pmi,
        ),
    )


@router.get("/components/{layer}/{component_idx}")
@log_errors
def get_component_correlations(
    layer: str,
    component_idx: int,
    loaded: DepLoadedRun,
    top_k: Annotated[int, Query(ge=1)],
) -> ComponentCorrelationsResponse:
    """Get correlated components for a specific component.

    Returns top-k correlations across different metrics (precision, recall, Jaccard, PMI).
    Returns None if correlations haven't been harvested for this run.
    """
    correlations = loaded.harvest.correlations
    component_key = f"{layer}:{component_idx}"

    if not analysis.has_component(correlations, component_key):
        raise HTTPException(
            status_code=404, detail=f"Component {component_key} not found in correlations"
        )

    def to_schema(c: analysis.CorrelatedComponent) -> CorrelatedComponent:
        return CorrelatedComponent(
            component_key=c.component_key,
            score=c.score,
            count_i=c.count_i,
            count_j=c.count_j,
            count_ij=c.count_ij,
            n_tokens=c.count_total,
        )

    return ComponentCorrelationsResponse(
        precision=[
            to_schema(c)
            for c in analysis.get_correlated_components(
                correlations, component_key, "precision", top_k
            )
        ],
        recall=[
            to_schema(c)
            for c in analysis.get_correlated_components(
                correlations, component_key, "recall", top_k
            )
        ],
        jaccard=[
            to_schema(c)
            for c in analysis.get_correlated_components(
                correlations, component_key, "jaccard", top_k
            )
        ],
        pmi=[
            to_schema(c)
            for c in analysis.get_correlated_components(correlations, component_key, "pmi", top_k)
        ],
        bottom_pmi=[
            to_schema(c)
            for c in analysis.get_correlated_components(
                correlations, component_key, "pmi", top_k, largest=False
            )
        ],
    )
