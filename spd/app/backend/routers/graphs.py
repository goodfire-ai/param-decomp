"""Graph computation endpoints for tokenization and attribution graphs."""

import json
import math
import queue
import sys
import threading
import time
import traceback
from collections.abc import Callable, Generator
from dataclasses import dataclass
from itertools import groupby
from typing import Annotated, Any, Literal

import torch
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.compute import (
    MAX_OUTPUT_NODES_PER_POS,
    Edge,
    compute_prompt_attributions,
    compute_prompt_attributions_optimized,
)
from spd.app.backend.database import GraphType, OptimizationParams, StoredGraph
from spd.app.backend.dependencies import DepLoadedRun, DepStateManager
from spd.app.backend.optim_cis import (
    AdvPGDConfig,
    CELossConfig,
    KLLossConfig,
    LossConfig,
    MaskType,
    OptimCIConfig,
)
from spd.app.backend.schemas import OutputProbability
from spd.app.backend.utils import log_errors
from spd.configs import ImportanceMinimalityLossConfig
from spd.log import logger
from spd.utils.distributed_utils import get_device


class EdgeData(BaseModel):
    """Edge in the attribution graph."""

    src: str  # "layer:seq:cIdx"
    tgt: str  # "layer:seq:cIdx"
    val: float
    is_cross_seq: bool = False


class PromptPreview(BaseModel):
    """Preview of a stored prompt for listing."""

    id: int
    token_ids: list[int]
    tokens: list[str]
    preview: str


class GraphData(BaseModel):
    """Full attribution graph data."""

    id: int
    graphType: GraphType
    tokens: list[str]
    edges: list[EdgeData]
    outputProbs: dict[str, OutputProbability]
    nodeCiVals: dict[
        str, float
    ]  # node key -> CI value (or output prob for output nodes or 1 for embed node)
    nodeSubcompActs: dict[str, float]  # node key -> subcomponent activation (v_i^T @ a)
    maxAbsAttr: float  # max absolute edge value
    maxAbsSubcompAct: float  # max absolute subcomponent activation for normalization
    l0_total: int  # total active components at current CI threshold


class CELossResult(BaseModel):
    """CE loss result (specific token target)."""

    type: Literal["ce"] = "ce"
    coeff: float
    position: int
    label_token: int
    label_str: str


class KLLossResult(BaseModel):
    """KL loss result (distribution matching)."""

    type: Literal["kl"] = "kl"
    coeff: float
    position: int


class OptimizationMetricsResult(BaseModel):
    """Final loss metrics from CI optimization."""

    ci_masked_label_prob: float | None = None  # Probability of label under CI mask (CE loss only)
    stoch_masked_label_prob: float | None = (
        None  # Probability of label under stochastic mask (CE loss only)
    )
    adv_pgd_label_prob: float | None = None  # Probability of label under adversarial mask (CE only)
    l0_total: float  # Total L0 (active components)


class OptimizationResult(BaseModel):
    """Results from optimized CI computation."""

    imp_min_coeff: float
    steps: int
    pnorm: float
    beta: float
    mask_type: MaskType
    loss: CELossResult | KLLossResult
    metrics: OptimizationMetricsResult
    adv_pgd_n_steps: int | None = None
    adv_pgd_step_size: float | None = None


class GraphDataWithOptimization(GraphData):
    """Attribution graph data with optimization results."""

    optimization: OptimizationResult


class ComponentStats(BaseModel):
    """Statistics for a component across prompts."""

    prompt_count: int
    avg_max_ci: float
    prompt_ids: list[int]


class PromptSearchQuery(BaseModel):
    """Query parameters for prompt search."""

    components: list[str]
    mode: str


class PromptSearchResponse(BaseModel):
    """Response from prompt search endpoint."""

    query: PromptSearchQuery
    count: int
    results: list[PromptPreview]


class TokenizeResponse(BaseModel):
    """Response from tokenize endpoint."""

    token_ids: list[int]
    tokens: list[str]
    text: str
    next_token_probs: list[float | None]  # Probability of next token (last token is None)


class TokenInfo(BaseModel):
    """A single token from the tokenizer vocabulary."""

    id: int
    string: str


class TokensResponse(BaseModel):
    """Response containing all tokens in the vocabulary."""

    tokens: list[TokenInfo]


# SSE streaming message types
class ProgressMessage(BaseModel):
    """Progress update during streaming computation."""

    type: Literal["progress"]
    current: int
    total: int
    stage: str


class ErrorMessage(BaseModel):
    """Error message during streaming computation."""

    type: Literal["error"]
    error: str


class CompleteMessage(BaseModel):
    """Completion message with result data."""

    type: Literal["complete"]
    data: GraphData


class CompleteMessageWithOptimization(BaseModel):
    """Completion message with optimization result data."""

    type: Literal["complete"]
    data: GraphDataWithOptimization


router = APIRouter(prefix="/api/graphs", tags=["graphs"])

DEVICE = get_device()

# This is a bit of a hack. We want to limit the number of edges returned to avoid overwhelming the frontend.
GLOBAL_EDGE_LIMIT = 50_000


ProgressCallback = Callable[[int, int, str], None]


def _build_out_probs(
    ci_masked_out_logits: torch.Tensor,
    target_out_logits: torch.Tensor,
    tok_display: Callable[[int], str],
    adv_pgd_out_logits: torch.Tensor | None = None,
) -> dict[str, OutputProbability]:
    """Build output probs dict from logit tensors.

    Takes top MAX_OUTPUT_NODES_PER_POS per position (CI slider handles threshold filtering).
    """
    ci_masked_out_probs = torch.softmax(ci_masked_out_logits, dim=-1)
    target_out_probs = torch.softmax(target_out_logits, dim=-1)
    adv_pgd_out_probs = (
        torch.softmax(adv_pgd_out_logits, dim=-1) if adv_pgd_out_logits is not None else None
    )

    out_probs: dict[str, OutputProbability] = {}
    for s in range(ci_masked_out_probs.shape[0]):
        pos_probs = ci_masked_out_probs[s]
        top_vals, top_idxs = torch.topk(
            pos_probs, min(MAX_OUTPUT_NODES_PER_POS, pos_probs.shape[0])
        )
        for prob_t, c_idx_t in zip(top_vals, top_idxs, strict=True):
            prob = float(prob_t.item())
            c_idx = int(c_idx_t.item())
            logit = float(ci_masked_out_logits[s, c_idx].item())
            target_prob = float(target_out_probs[s, c_idx].item())
            target_logit = float(target_out_logits[s, c_idx].item())

            adv_pgd_prob: float | None = None
            adv_pgd_logit: float | None = None
            if adv_pgd_out_probs is not None and adv_pgd_out_logits is not None:
                adv_pgd_prob = round(float(adv_pgd_out_probs[s, c_idx].item()), 6)
                adv_pgd_logit = round(float(adv_pgd_out_logits[s, c_idx].item()), 4)

            key = f"{s}:{c_idx}"
            out_probs[key] = OutputProbability(
                prob=round(prob, 6),
                logit=round(logit, 4),
                target_prob=round(target_prob, 6),
                target_logit=round(target_logit, 4),
                adv_pgd_prob=adv_pgd_prob,
                adv_pgd_logit=adv_pgd_logit,
                token=tok_display(c_idx),
            )
    return out_probs


def stream_computation(
    work: Callable[[ProgressCallback], GraphData | GraphDataWithOptimization],
) -> StreamingResponse:
    """Run graph computation in a thread with SSE streaming for progress updates."""
    progress_queue: queue.Queue[dict[str, Any]] = queue.Queue()

    def on_progress(current: int, total: int, stage: str) -> None:
        progress_queue.put({"type": "progress", "current": current, "total": total, "stage": stage})

    def compute_thread() -> None:
        try:
            result = work(on_progress)
            progress_queue.put({"type": "result", "result": result})
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            progress_queue.put({"type": "error", "error": str(e)})

    def generate() -> Generator[str]:
        thread = threading.Thread(target=compute_thread)
        thread.start()

        while True:
            try:
                msg = progress_queue.get(timeout=0.1)
            except queue.Empty:
                if not thread.is_alive():
                    break
                continue

            if msg["type"] == "progress":
                yield f"data: {json.dumps(msg)}\n\n"
            elif msg["type"] == "error":
                yield f"data: {json.dumps(msg)}\n\n"
                break
            elif msg["type"] == "result":
                complete_data = {"type": "complete", "data": msg["result"].model_dump()}
                yield f"data: {json.dumps(complete_data)}\n\n"
                break

        thread.join()

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/tokenize")
@log_errors
def tokenize_text(text: str, loaded: DepLoadedRun) -> TokenizeResponse:
    """Tokenize text and return tokens with probability of next token."""
    device = get_device()
    token_ids = loaded.tokenizer.encode(text)

    if len(token_ids) == 0:
        return TokenizeResponse(
            text=text,
            token_ids=[],
            tokens=[],
            next_token_probs=[],
        )

    tokens_tensor = torch.tensor([token_ids], device=device)

    with torch.no_grad():
        logits = loaded.model(tokens_tensor)
        probs = torch.softmax(logits, dim=-1)

    # Get probability of next token at each position
    next_token_probs: list[float | None] = []
    for i in range(len(token_ids) - 1):
        next_token_id = token_ids[i + 1]
        prob = probs[0, i, next_token_id].item()
        next_token_probs.append(prob)
    next_token_probs.append(None)  # No next token for last position

    return TokenizeResponse(
        text=text,
        token_ids=token_ids,
        tokens=loaded.tokenizer.get_spans(token_ids),
        next_token_probs=next_token_probs,
    )


@router.get("/tokens")
@log_errors
def get_all_tokens(loaded: DepLoadedRun) -> TokensResponse:
    """Get all tokens in the tokenizer vocabulary for client-side search."""
    tokens = [
        TokenInfo(id=tid, string=loaded.tokenizer.get_tok_display(tid))
        for tid in range(loaded.tokenizer.vocab_size)
    ]
    return TokensResponse(tokens=tokens)


class TokenSearchResponse(BaseModel):
    """Response from token search endpoint."""

    tokens: list[TokenInfo]


@router.get("/tokens/search")
@log_errors
def search_tokens(
    q: Annotated[str, Query(min_length=1)],
    loaded: DepLoadedRun,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> TokenSearchResponse:
    """Search tokens by substring match. Returns up to `limit` results."""
    query = q.lower()
    matches: list[TokenInfo] = []
    for tid in range(loaded.tokenizer.vocab_size):
        string = loaded.tokenizer.get_tok_display(tid)
        if query in string.lower():
            matches.append(TokenInfo(id=tid, string=string))
            if len(matches) >= limit:
                break
    return TokenSearchResponse(tokens=matches)


NormalizeType = Literal["none", "target", "layer"]


def compute_max_abs_attr(edges: list[Edge]) -> float:
    """Compute max absolute edge strength for normalization."""
    if not edges:
        return 0.0
    return max(abs(edge.strength) for edge in edges)


def compute_max_abs_subcomp_act(node_subcomp_acts: dict[str, float]) -> float:
    """Compute max absolute subcomponent activation for normalization."""
    if not node_subcomp_acts:
        return 0.0
    return max(abs(v) for v in node_subcomp_acts.values())


@router.post("")
@log_errors
def compute_graph_stream(
    prompt_id: Annotated[int, Query()],
    normalize: Annotated[NormalizeType, Query()],
    loaded: DepLoadedRun,
    manager: DepStateManager,
    ci_threshold: Annotated[float, Query()],
    included_nodes: Annotated[str | None, Query()] = None,
):
    """Compute attribution graph for a prompt with streaming progress.

    If included_nodes is provided (JSON array of node keys), creates a "manual" graph
    with only those nodes. Otherwise creates a "standard" graph.

    Args:
        included_nodes: JSON array of node keys to include (creates manual graph if provided)
    """
    # Parse and validate included_nodes if provided
    included_nodes_set: set[str] | None = None
    included_nodes_list: list[str] | None = None
    if included_nodes is not None:
        try:
            parsed_nodes = json.loads(included_nodes)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail="Invalid included_nodes JSON") from e

        if not isinstance(parsed_nodes, list):
            raise HTTPException(status_code=400, detail="included_nodes must be a JSON array")

        if len(parsed_nodes) > 10000:
            raise HTTPException(status_code=400, detail="Too many nodes (max 10000)")

        for node in parsed_nodes:
            if not isinstance(node, str):
                raise HTTPException(status_code=400, detail="All node keys must be strings")
            if len(node) > 100:  # Node keys follow format "layer:seq:cIdx", 100 chars is generous
                raise HTTPException(status_code=400, detail=f"Node key too long: {node[:50]}...")

        included_nodes_list = parsed_nodes
        included_nodes_set = set(parsed_nodes)

    is_manual = included_nodes_set is not None
    graph_type: GraphType = "manual" if is_manual else "standard"

    db = manager.db
    prompt = db.get_prompt(prompt_id)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")

    token_ids = prompt.token_ids
    spans = loaded.tokenizer.get_spans(token_ids)
    tokens_tensor = torch.tensor([token_ids], device=DEVICE)

    def work(on_progress: ProgressCallback) -> GraphData:
        t_total = time.perf_counter()

        result = compute_prompt_attributions(
            model=loaded.model,
            topology=loaded.topology,
            tokens=tokens_tensor,
            sources_by_target=loaded.sources_by_target,
            output_prob_threshold=0.01,
            sampling=loaded.config.sampling,
            device=DEVICE,
            on_progress=on_progress,
            included_nodes=included_nodes_set,
        )

        ci_masked_out_logits = result.ci_masked_out_logits.cpu()
        target_out_logits = result.target_out_logits.cpu()

        t0 = time.perf_counter()
        graph_id = db.save_graph(
            prompt_id=prompt_id,
            graph=StoredGraph(
                graph_type=graph_type,
                edges=result.edges,
                ci_masked_out_logits=ci_masked_out_logits,
                target_out_logits=target_out_logits,
                node_ci_vals=result.node_ci_vals,
                node_subcomp_acts=result.node_subcomp_acts,
                included_nodes=included_nodes_list,
            ),
        )
        logger.info(f"[perf] save_graph: {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        fg = filter_graph_for_display(
            raw_edges=result.edges,
            node_ci_vals=result.node_ci_vals,
            node_subcomp_acts=result.node_subcomp_acts,
            ci_masked_out_logits=ci_masked_out_logits,
            target_out_logits=target_out_logits,
            tok_display=loaded.tokenizer.get_tok_display,
            num_tokens=len(token_ids),
            ci_threshold=ci_threshold,
            normalize=normalize,
        )
        logger.info(
            f"[perf] filter_graph: {time.perf_counter() - t0:.2f}s ({len(fg.edges)} edges after filter)"
        )
        logger.info(f"[perf] Total graph computation: {time.perf_counter() - t_total:.2f}s")

        return GraphData(
            id=graph_id,
            graphType=graph_type,
            tokens=spans,
            edges=fg.edges,
            outputProbs=fg.out_probs,
            nodeCiVals=fg.node_ci_vals,
            nodeSubcompActs=result.node_subcomp_acts,
            maxAbsAttr=fg.max_abs_attr,
            maxAbsSubcompAct=fg.max_abs_subcomp_act,
            l0_total=fg.l0_total,
        )

    return stream_computation(work)


def _edge_to_edge_data(edge: Edge) -> EdgeData:
    """Convert Edge (internal format) to EdgeData (API format)."""
    return EdgeData(
        src=str(edge.source),
        tgt=str(edge.target),
        val=edge.strength,
        is_cross_seq=edge.is_cross_seq,
    )


def _normalize_edges(edges: list[Edge], normalize: NormalizeType) -> list[Edge]:
    """Normalize edges by target node or target layer."""
    if normalize == "none":
        return edges

    def get_group_key(edge: Edge) -> str:
        if normalize == "target":
            return str(edge.target)
        return edge.target.layer

    sorted_edges = sorted(edges, key=get_group_key)
    groups = groupby(sorted_edges, key=get_group_key)

    out_edges = []
    for _, group_edges in groups:
        group_edges = list(group_edges)
        group_strength = math.sqrt(sum(edge.strength**2 for edge in group_edges))
        if group_strength == 0:
            continue
        for edge in group_edges:
            out_edges.append(
                Edge(
                    source=edge.source,
                    target=edge.target,
                    is_cross_seq=edge.is_cross_seq,
                    strength=edge.strength / group_strength,
                )
            )
    return out_edges


LossType = Literal["ce", "kl"]


@router.post("/optimized/stream")
@log_errors
def compute_graph_optimized_stream(
    prompt_id: Annotated[int, Query()],
    imp_min_coeff: Annotated[float, Query(gte=0)],
    steps: Annotated[int, Query(gt=0)],
    pnorm: Annotated[float, Query(gt=0)],
    beta: Annotated[float, Query(ge=0)],
    normalize: Annotated[NormalizeType, Query()],
    loaded: DepLoadedRun,
    manager: DepStateManager,
    ci_threshold: Annotated[float, Query()],
    mask_type: Annotated[MaskType, Query()],
    loss_type: Annotated[LossType, Query()],
    loss_coeff: Annotated[float, Query(gt=0)],
    loss_position: Annotated[int, Query(ge=0)],
    label_token: Annotated[int | None, Query()] = None,
    adv_pgd_n_steps: Annotated[int | None, Query(gt=0)] = None,
    adv_pgd_step_size: Annotated[float | None, Query(gt=0)] = None,
):
    """Compute optimized attribution graph for a prompt with streaming progress.

    loss_type determines whether to use CE (cross-entropy for specific token) or KL (distribution matching).
    label_token is required when loss_type is "ce".
    adv_pgd_n_steps and adv_pgd_step_size enable adversarial PGD when both are provided.
    """
    # Build loss config based on type
    loss_config: LossConfig
    match loss_type:
        case "ce":
            if label_token is None:
                raise HTTPException(status_code=400, detail="label_token is required for CE loss")
            loss_config = CELossConfig(
                coeff=loss_coeff, position=loss_position, label_token=label_token
            )
        case "kl":
            loss_config = KLLossConfig(coeff=loss_coeff, position=loss_position)

    lr = 1e-2

    db = manager.db
    prompt = db.get_prompt(prompt_id)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")

    token_ids = prompt.token_ids
    if loss_position >= len(token_ids):
        raise HTTPException(
            status_code=400,
            detail=f"loss_position {loss_position} out of bounds for prompt with {len(token_ids)} tokens",
        )

    label_str = loaded.tokenizer.get_tok_display(label_token) if label_token is not None else None
    spans = loaded.tokenizer.get_spans(token_ids)
    tokens_tensor = torch.tensor([token_ids], device=DEVICE)

    # Slice tokens to only include positions <= loss_position
    num_tokens = loss_position + 1
    spans_sliced = spans[:num_tokens]

    opt_params = OptimizationParams(
        imp_min_coeff=imp_min_coeff,
        steps=steps,
        pnorm=pnorm,
        beta=beta,
        mask_type=mask_type,
        loss=loss_config,
        adv_pgd_n_steps=adv_pgd_n_steps,
        adv_pgd_step_size=adv_pgd_step_size,
    )

    optim_config = OptimCIConfig(
        seed=0,
        lr=lr,
        steps=steps,
        weight_decay=0.0,
        lr_schedule="cosine",
        lr_exponential_halflife=None,
        lr_warmup_pct=0.01,
        log_freq=max(1, steps // 4),
        imp_min_config=ImportanceMinimalityLossConfig(coeff=imp_min_coeff, pnorm=pnorm, beta=beta),
        loss_config=loss_config,
        sampling=loaded.config.sampling,
        ce_kl_rounding_threshold=0.5,
        mask_type=mask_type,
        adv_pgd=AdvPGDConfig(n_steps=adv_pgd_n_steps, step_size=adv_pgd_step_size, init="random")
        if adv_pgd_n_steps is not None and adv_pgd_step_size is not None
        else None,
    )

    def work(on_progress: ProgressCallback) -> GraphDataWithOptimization:
        result = compute_prompt_attributions_optimized(
            model=loaded.model,
            topology=loaded.topology,
            tokens=tokens_tensor,
            sources_by_target=loaded.sources_by_target,
            optim_config=optim_config,
            output_prob_threshold=0.01,
            device=DEVICE,
            on_progress=on_progress,
        )

        ci_masked_out_logits = result.ci_masked_out_logits.cpu()
        target_out_logits = result.target_out_logits.cpu()
        adv_pgd_out_logits = (
            result.adv_pgd_out_logits.cpu() if result.adv_pgd_out_logits is not None else None
        )

        graph_id = db.save_graph(
            prompt_id=prompt_id,
            graph=StoredGraph(
                graph_type="optimized",
                edges=result.edges,
                ci_masked_out_logits=ci_masked_out_logits,
                target_out_logits=target_out_logits,
                adv_pgd_out_logits=adv_pgd_out_logits,
                node_ci_vals=result.node_ci_vals,
                node_subcomp_acts=result.node_subcomp_acts,
                optimization_params=opt_params,
            ),
        )

        fg = filter_graph_for_display(
            raw_edges=result.edges,
            node_ci_vals=result.node_ci_vals,
            node_subcomp_acts=result.node_subcomp_acts,
            ci_masked_out_logits=ci_masked_out_logits,
            target_out_logits=target_out_logits,
            tok_display=loaded.tokenizer.get_tok_display,
            num_tokens=num_tokens,
            ci_threshold=ci_threshold,
            normalize=normalize,
            adv_pgd_out_logits=adv_pgd_out_logits,
        )

        # Build loss result based on config type
        loss_result: CELossResult | KLLossResult
        match loss_config:
            case CELossConfig(coeff=coeff, position=pos, label_token=label_tok):
                assert label_str is not None
                loss_result = CELossResult(
                    coeff=coeff,
                    position=pos,
                    label_token=label_tok,
                    label_str=label_str,
                )
            case KLLossConfig(coeff=coeff, position=pos):
                loss_result = KLLossResult(coeff=coeff, position=pos)

        return GraphDataWithOptimization(
            id=graph_id,
            graphType="optimized",
            tokens=spans_sliced,
            edges=fg.edges,
            outputProbs=fg.out_probs,
            nodeCiVals=fg.node_ci_vals,
            nodeSubcompActs=result.node_subcomp_acts,
            maxAbsAttr=fg.max_abs_attr,
            maxAbsSubcompAct=fg.max_abs_subcomp_act,
            l0_total=fg.l0_total,
            optimization=OptimizationResult(
                imp_min_coeff=imp_min_coeff,
                steps=steps,
                pnorm=pnorm,
                beta=beta,
                mask_type=mask_type,
                loss=loss_result,
                metrics=OptimizationMetricsResult(
                    ci_masked_label_prob=result.metrics.ci_masked_label_prob,
                    stoch_masked_label_prob=result.metrics.stoch_masked_label_prob,
                    adv_pgd_label_prob=result.metrics.adv_pgd_label_prob,
                    l0_total=result.metrics.l0_total,
                ),
                adv_pgd_n_steps=adv_pgd_n_steps,
                adv_pgd_step_size=adv_pgd_step_size,
            ),
        )

    return stream_computation(work)


@dataclass
class FilteredGraph:
    """Result of filtering a raw graph for display."""

    edges: list[EdgeData]
    node_ci_vals: dict[str, float]  # with pseudo nodes
    out_probs: dict[str, OutputProbability]
    max_abs_attr: float
    max_abs_subcomp_act: float
    l0_total: int


def filter_graph_for_display(
    raw_edges: list[Edge],
    node_ci_vals: dict[str, float],
    node_subcomp_acts: dict[str, float],
    ci_masked_out_logits: torch.Tensor,
    target_out_logits: torch.Tensor,
    tok_display: Callable[[int], str],
    num_tokens: int,
    ci_threshold: float,
    normalize: NormalizeType,
    edge_limit: int = GLOBAL_EDGE_LIMIT,
    adv_pgd_out_logits: torch.Tensor | None = None,
) -> FilteredGraph:
    """Filter and transform a raw attribution graph for display.

    1. Build out_probs from logit tensors (top MAX_OUTPUT_NODES_PER_POS per position)
    2. Filter component nodes by CI threshold
    3. Add embed (CI=1.0) and output (CI=prob) pseudo-nodes
    4. Drop edges not connecting surviving nodes
    5. Normalize edge strengths (if requested)
    6. Cap edges at edge_limit
    """
    out_probs = _build_out_probs(
        ci_masked_out_logits, target_out_logits, tok_display, adv_pgd_out_logits
    )

    filtered_node_ci_vals = {k: v for k, v in node_ci_vals.items() if v > ci_threshold}

    # Add pseudo-nodes: embed always visible, output nodes use their probability
    node_ci_vals_with_pseudo = dict(filtered_node_ci_vals)
    for seq_pos in range(num_tokens):
        node_ci_vals_with_pseudo[f"embed:{seq_pos}:0"] = 1.0
    for key, out_prob in out_probs.items():
        seq_pos, token_id = key.split(":")
        node_ci_vals_with_pseudo[f"output:{seq_pos}:{token_id}"] = out_prob.prob

    # Filter edges to only those connecting surviving nodes
    node_keys = set(node_ci_vals_with_pseudo.keys())
    edges = [e for e in raw_edges if str(e.source) in node_keys and str(e.target) in node_keys]

    edges = _normalize_edges(edges=edges, normalize=normalize)
    max_abs_attr = compute_max_abs_attr(edges=edges)

    # Always sort by abs(strength) desc so frontend can just slice(0, topK) without re-sorting
    edges = sorted(edges, key=lambda e: abs(e.strength), reverse=True)

    if len(edges) > edge_limit:
        logger.warning(f"Edge limit {edge_limit} exceeded ({len(edges)} edges), truncating")
        edges = edges[:edge_limit]

    return FilteredGraph(
        edges=[_edge_to_edge_data(e) for e in edges],
        node_ci_vals=node_ci_vals_with_pseudo,
        out_probs=out_probs,
        max_abs_attr=max_abs_attr,
        max_abs_subcomp_act=compute_max_abs_subcomp_act(node_subcomp_acts),
        l0_total=len(filtered_node_ci_vals),
    )


def stored_graph_to_response(
    graph: StoredGraph,
    token_ids: list[int],
    tokenizer: AppTokenizer,
    normalize: NormalizeType,
    ci_threshold: float,
) -> GraphData | GraphDataWithOptimization:
    """Convert a StoredGraph to API response format."""
    spans = tokenizer.get_spans(token_ids)
    num_tokens = len(token_ids)
    is_optimized = graph.optimization_params is not None

    if is_optimized:
        assert graph.optimization_params is not None
        num_tokens = graph.optimization_params.loss.position + 1
        spans = spans[:num_tokens]

    fg = filter_graph_for_display(
        raw_edges=graph.edges,
        node_ci_vals=graph.node_ci_vals,
        node_subcomp_acts=graph.node_subcomp_acts,
        ci_masked_out_logits=graph.ci_masked_out_logits,
        target_out_logits=graph.target_out_logits,
        tok_display=tokenizer.get_tok_display,
        num_tokens=num_tokens,
        ci_threshold=ci_threshold,
        normalize=normalize,
        adv_pgd_out_logits=graph.adv_pgd_out_logits,
    )

    if not is_optimized:
        return GraphData(
            id=graph.id,
            graphType=graph.graph_type,
            tokens=spans,
            edges=fg.edges,
            outputProbs=fg.out_probs,
            nodeCiVals=fg.node_ci_vals,
            nodeSubcompActs=graph.node_subcomp_acts,
            maxAbsAttr=fg.max_abs_attr,
            maxAbsSubcompAct=fg.max_abs_subcomp_act,
            l0_total=fg.l0_total,
        )

    assert graph.optimization_params is not None
    opt = graph.optimization_params

    # Build loss result based on stored config type
    loss_result: CELossResult | KLLossResult
    match opt.loss:
        case CELossConfig(coeff=coeff, position=pos, label_token=label_tok):
            label_str = tokenizer.get_tok_display(label_tok)
            loss_result = CELossResult(
                coeff=coeff,
                position=pos,
                label_token=label_tok,
                label_str=label_str,
            )
        case KLLossConfig(coeff=coeff, position=pos):
            loss_result = KLLossResult(coeff=coeff, position=pos)

    return GraphDataWithOptimization(
        id=graph.id,
        graphType=graph.graph_type,
        tokens=spans,
        edges=fg.edges,
        outputProbs=fg.out_probs,
        nodeCiVals=fg.node_ci_vals,
        nodeSubcompActs=graph.node_subcomp_acts,
        maxAbsAttr=fg.max_abs_attr,
        maxAbsSubcompAct=fg.max_abs_subcomp_act,
        l0_total=fg.l0_total,
        optimization=OptimizationResult(
            imp_min_coeff=opt.imp_min_coeff,
            steps=opt.steps,
            pnorm=opt.pnorm,
            beta=opt.beta,
            mask_type=opt.mask_type,
            loss=loss_result,
            # Metrics not stored in DB for cached graphs - use l0_total from graph
            metrics=OptimizationMetricsResult(l0_total=float(fg.l0_total)),
            adv_pgd_n_steps=opt.adv_pgd_n_steps,
            adv_pgd_step_size=opt.adv_pgd_step_size,
        ),
    )


@router.get("/{prompt_id}")
@log_errors
def get_graphs(
    prompt_id: int,
    normalize: Annotated[NormalizeType, Query()],
    ci_threshold: Annotated[float, Query(ge=0)],
    loaded: DepLoadedRun,
    manager: DepStateManager,
) -> list[GraphData | GraphDataWithOptimization]:
    """Get all stored graphs for a prompt.

    Returns list of graphs (both standard and optimized) for the given prompt.
    Returns empty list if no graphs exist.
    """
    db = manager.db
    prompt = db.get_prompt(prompt_id)
    if prompt is None:
        return []

    stored_graphs = db.get_graphs(prompt_id)
    return [
        stored_graph_to_response(
            graph=graph,
            token_ids=prompt.token_ids,
            tokenizer=loaded.tokenizer,
            normalize=normalize,
            ci_threshold=ci_threshold,
        )
        for graph in stored_graphs
    ]
