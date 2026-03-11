"""MCP (Model Context Protocol) endpoint for Claude Code integration.

This router implements the MCP JSON-RPC protocol over HTTP, allowing Claude Code
to use SPD tools directly with proper schemas and streaming progress.

MCP Spec: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports
"""

import inspect
import json
import queue
import threading
import traceback
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from spd.app.backend.compute import (
    compute_ci_only,
    compute_prompt_attributions_optimized,
)
from spd.app.backend.database import StoredGraph
from spd.app.backend.optim_cis import CELossConfig, OptimCIConfig
from spd.app.backend.routers.graphs import _build_out_probs
from spd.app.backend.routers.pretrain_info import _get_pretrain_info
from spd.app.backend.state import StateManager
from spd.configs import ImportanceMinimalityLossConfig
from spd.harvest import analysis
from spd.log import logger
from spd.utils.distributed_utils import get_device

router = APIRouter(tags=["mcp"])

DEVICE = get_device()

# MCP protocol version
MCP_PROTOCOL_VERSION = "2024-11-05"


@dataclass
class InvestigationConfig:
    """Configuration for investigation mode. All paths are required when in investigation mode."""

    events_log_path: Path
    investigation_dir: Path


_investigation_config: InvestigationConfig | None = None


def set_investigation_config(config: InvestigationConfig) -> None:
    """Configure MCP for investigation mode."""
    global _investigation_config
    _investigation_config = config


def _log_event(event_type: str, message: str, details: dict[str, Any] | None = None) -> None:
    """Log an event to the events file if in investigation mode."""
    if _investigation_config is None:
        return
    event = {
        "event_type": event_type,
        "timestamp": datetime.now(UTC).isoformat(),
        "message": message,
        "details": details or {},
    }
    with open(_investigation_config.events_log_path, "a") as f:
        f.write(json.dumps(event) + "\n")


# =============================================================================
# MCP Protocol Types
# =============================================================================


class MCPRequest(BaseModel):
    """JSON-RPC 2.0 request."""

    jsonrpc: Literal["2.0"]
    id: int | str | None = None
    method: str
    params: dict[str, Any] | None = None


class MCPResponse(BaseModel):
    """JSON-RPC 2.0 response.

    Per JSON-RPC 2.0 spec, exactly one of result/error must be present (not both, not neither).
    Use model_dump(exclude_none=True) when serializing to avoid including null fields.
    """

    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str | None
    result: Any | None = None
    error: dict[str, Any] | None = None


class ToolDefinition(BaseModel):
    """MCP tool definition."""

    name: str
    description: str
    inputSchema: dict[str, Any]


# =============================================================================
# Tool Definitions
# =============================================================================

TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="optimize_graph",
        description="""Optimize a sparse circuit for a specific behavior.

Given a prompt and target token, finds the minimal set of components that produce the target prediction.
Returns the optimized graph with component CI values and edges showing information flow.

This is the primary tool for understanding how the model produces a specific output.""",
        inputSchema={
            "type": "object",
            "properties": {
                "prompt_text": {
                    "type": "string",
                    "description": "The input text to analyze (e.g., 'The boy said that')",
                },
                "target_token": {
                    "type": "string",
                    "description": "The token to predict (e.g., ' he'). Include leading space if needed.",
                },
                "loss_position": {
                    "type": "integer",
                    "description": "Position to optimize prediction at (0-indexed, usually last position). If not specified, uses the last position.",
                },
                "steps": {
                    "type": "integer",
                    "description": "Optimization steps (default: 100, more = sparser but slower)",
                    "default": 100,
                },
                "ci_threshold": {
                    "type": "number",
                    "description": "CI threshold for including components (default: 0.5, lower = more components)",
                    "default": 0.5,
                },
            },
            "required": ["prompt_text", "target_token"],
        },
    ),
    ToolDefinition(
        name="get_component_info",
        description="""Get detailed information about a component.

Returns the component's interpretation (what it does), token statistics (what tokens
activate it and what it predicts), and correlated components.

Use this to understand what role a component plays in a circuit.""",
        inputSchema={
            "type": "object",
            "properties": {
                "layer": {
                    "type": "string",
                    "description": "Canonical layer name (e.g., '0.mlp.up', '2.attn.o')",
                },
                "component_idx": {
                    "type": "integer",
                    "description": "Component index within the layer",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top tokens/correlations to return (default: 20)",
                    "default": 20,
                },
            },
            "required": ["layer", "component_idx"],
        },
    ),
    ToolDefinition(
        name="run_ablation",
        description="""Run an ablation experiment with only selected components active.

Tests a hypothesis by running the model with a sparse set of components.
Returns predictions showing what the circuit produces vs the full model.

Use this to verify that identified components are necessary and sufficient.""",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Input text for the ablation",
                },
                "selected_nodes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Node keys to keep active (format: 'layer:seq_pos:component_idx')",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top predictions to return per position (default: 10)",
                    "default": 10,
                },
            },
            "required": ["text", "selected_nodes"],
        },
    ),
    ToolDefinition(
        name="search_dataset",
        description="""Search the SimpleStories training dataset for patterns.

Finds stories containing the query string. Use this to find examples of
specific linguistic patterns (pronouns, verb forms, etc.) for investigation.""",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to search for (case-insensitive)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 20)",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
    ),
    ToolDefinition(
        name="create_prompt",
        description="""Create a prompt for analysis.

Tokenizes the text and returns token IDs and next-token probabilities.
The returned prompt_id can be used with other tools.""",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to create a prompt from",
                },
            },
            "required": ["text"],
        },
    ),
    ToolDefinition(
        name="update_research_log",
        description="""Append content to your research log.

Use this to document your investigation progress, findings, and next steps.
The research log is your primary output for humans to follow your work.

Call this frequently (every few minutes) with updates on what you're doing.""",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Markdown content to append to the research log",
                },
            },
            "required": ["content"],
        },
    ),
    ToolDefinition(
        name="save_explanation",
        description="""Save a complete behavior explanation.

Use this when you have finished investigating a behavior and want to document
your findings. This creates a structured record of the behavior, the components
involved, and your explanation of how they work together.

Only call this for complete, validated explanations - not preliminary hypotheses.""",
        inputSchema={
            "type": "object",
            "properties": {
                "subject_prompt": {
                    "type": "string",
                    "description": "A prompt that demonstrates the behavior",
                },
                "behavior_description": {
                    "type": "string",
                    "description": "Clear description of the behavior",
                },
                "components_involved": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "component_key": {
                                "type": "string",
                                "description": "Component key (e.g., '0.mlp.up:5')",
                            },
                            "role": {
                                "type": "string",
                                "description": "The role this component plays",
                            },
                            "interpretation": {
                                "type": "string",
                                "description": "Auto-interp label if available",
                            },
                        },
                        "required": ["component_key", "role"],
                    },
                    "description": "List of components and their roles",
                },
                "explanation": {
                    "type": "string",
                    "description": "How the components work together",
                },
                "supporting_evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "evidence_type": {
                                "type": "string",
                                "enum": [
                                    "ablation",
                                    "attribution",
                                    "activation_pattern",
                                    "correlation",
                                    "other",
                                ],
                            },
                            "description": {"type": "string"},
                            "details": {"type": "object"},
                        },
                        "required": ["evidence_type", "description"],
                    },
                    "description": "Evidence supporting this explanation",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Your confidence level",
                },
                "alternative_hypotheses": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Other hypotheses you considered",
                },
                "limitations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Known limitations of this explanation",
                },
            },
            "required": [
                "subject_prompt",
                "behavior_description",
                "components_involved",
                "explanation",
                "confidence",
            ],
        },
    ),
    ToolDefinition(
        name="set_investigation_summary",
        description="""Set a title and summary for your investigation.

Call this when you've completed your investigation (or periodically as you make progress)
to provide a human-readable title and summary that will be shown in the investigations UI.

The title should be short and descriptive. The summary should be 1-3 sentences
explaining what you investigated and what you found.""",
        inputSchema={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short title for the investigation (e.g., 'Gendered Pronoun Circuit')",
                },
                "summary": {
                    "type": "string",
                    "description": "Brief summary of findings (1-3 sentences)",
                },
                "status": {
                    "type": "string",
                    "enum": ["in_progress", "completed", "inconclusive"],
                    "description": "Current status of the investigation",
                    "default": "in_progress",
                },
            },
            "required": ["title", "summary"],
        },
    ),
    ToolDefinition(
        name="save_graph_artifact",
        description="""Save a graph as an artifact for inclusion in your research report.

After calling optimize_graph and getting a graph_id, call this to save the graph
as an artifact. Then reference it in your research log using the spd:graph syntax:

```spd:graph
artifact: graph_001
```

This allows humans reviewing your investigation to see interactive circuit visualizations
inline with your research notes.""",
        inputSchema={
            "type": "object",
            "properties": {
                "graph_id": {
                    "type": "integer",
                    "description": "The graph ID returned by optimize_graph",
                },
                "caption": {
                    "type": "string",
                    "description": "Optional caption describing what this graph shows",
                },
            },
            "required": ["graph_id"],
        },
    ),
    ToolDefinition(
        name="probe_component",
        description="""Fast CI probing on custom text.

Computes causal importance values and subcomponent activations for a specific component
across all positions in the input text. Also returns next-token probabilities.

Use this for quick, targeted analysis of how a component responds to specific inputs.""",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The input text to probe",
                },
                "layer": {
                    "type": "string",
                    "description": "Canonical layer name (e.g., '0.mlp.up')",
                },
                "component_idx": {
                    "type": "integer",
                    "description": "Component index within the layer",
                },
            },
            "required": ["text", "layer", "component_idx"],
        },
    ),
    ToolDefinition(
        name="get_component_activation_examples",
        description="""Get activation examples from harvest data for a component.

Returns examples showing token windows where the component fires, along with
CI values and activation strengths at each position.

Use this to understand what inputs activate a component.""",
        inputSchema={
            "type": "object",
            "properties": {
                "layer": {
                    "type": "string",
                    "description": "Canonical layer name (e.g., '0.mlp.up')",
                },
                "component_idx": {
                    "type": "integer",
                    "description": "Component index within the layer",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of examples to return (default: 10)",
                    "default": 10,
                },
            },
            "required": ["layer", "component_idx"],
        },
    ),
    ToolDefinition(
        name="get_component_attributions",
        description="""Get dataset-level component dependencies from pre-computed attributions.

Returns the top source and target components that this component attributes to/from,
aggregated over the training dataset. Both positive and negative attributions are returned.

Use this to understand a component's role in the broader network.""",
        inputSchema={
            "type": "object",
            "properties": {
                "layer": {
                    "type": "string",
                    "description": "Canonical layer name (e.g., '0.mlp.up') or 'output'",
                },
                "component_idx": {
                    "type": "integer",
                    "description": "Component index within the layer",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of top attributions to return per direction (default: 10)",
                    "default": 10,
                },
            },
            "required": ["layer", "component_idx"],
        },
    ),
    ToolDefinition(
        name="get_model_info",
        description="""Get architecture details about the pretrained model.

Returns model type, summary, target model config, topology, and pretrain info.
No parameters required.""",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]


# =============================================================================
# Tool Implementations
# =============================================================================


def _get_state():
    """Get state manager and loaded run, raising clear errors if not available."""
    manager = StateManager.get()
    if manager.run_state is None:
        raise ValueError("No run loaded. The backend must load a run first.")
    return manager, manager.run_state


def _canonicalize_layer(layer: str, loaded: Any) -> str:
    """Translate concrete layer name to canonical, passing through 'output'."""
    if layer == "output":
        return layer
    return loaded.topology.target_to_canon(layer)


def _canonicalize_key(concrete_key: str, loaded: Any) -> str:
    """Translate concrete component key (e.g. 'h.0.mlp.c_fc:444') to canonical ('0.mlp.up:444')."""
    layer, idx = concrete_key.rsplit(":", 1)
    return f"{_canonicalize_layer(layer, loaded)}:{idx}"


def _tool_optimize_graph(params: dict[str, Any]) -> Generator[dict[str, Any]]:
    """Optimize a sparse circuit for a behavior. Yields progress events."""
    manager, loaded = _get_state()

    prompt_text = params["prompt_text"]
    target_token = params["target_token"]
    steps = params.get("steps", 100)
    ci_threshold = params.get("ci_threshold", 0.5)

    # Tokenize prompt
    token_ids = loaded.tokenizer.encode(prompt_text)
    if not token_ids:
        raise ValueError("Prompt text produced no tokens")

    # Find target token ID
    target_token_ids = loaded.tokenizer.encode(target_token)
    if len(target_token_ids) != 1:
        raise ValueError(
            f"Target token '{target_token}' tokenizes to {len(target_token_ids)} tokens, expected 1. "
            f"Token IDs: {target_token_ids}"
        )
    label_token = target_token_ids[0]

    # Determine loss position
    loss_position = params.get("loss_position")
    if loss_position is None:
        loss_position = len(token_ids) - 1

    if loss_position >= len(token_ids):
        raise ValueError(
            f"loss_position {loss_position} out of bounds for prompt with {len(token_ids)} tokens"
        )

    _log_event(
        "tool_start",
        f"optimize_graph: '{prompt_text}' → '{target_token}'",
        {"steps": steps, "loss_position": loss_position},
    )

    yield {"type": "progress", "current": 0, "total": steps, "stage": "starting optimization"}

    # Create prompt in DB
    prompt_id = manager.db.add_custom_prompt(
        run_id=loaded.run.id,
        token_ids=token_ids,
        context_length=loaded.context_length,
    )

    # Build optimization config
    loss_config = CELossConfig(coeff=1.0, position=loss_position, label_token=label_token)

    optim_config = OptimCIConfig(
        adv_pgd=None,  # AdvPGDConfig(n_steps=10, step_size=0.01, init="random"),
        seed=0,
        lr=1e-2,
        steps=steps,
        weight_decay=0.0,
        lr_schedule="cosine",
        lr_exponential_halflife=None,
        lr_warmup_pct=0.01,
        log_freq=max(1, steps // 10),
        imp_min_config=ImportanceMinimalityLossConfig(coeff=0.1, pnorm=0.5, beta=0.0),
        loss_config=loss_config,
        sampling=loaded.config.sampling,
        ce_kl_rounding_threshold=0.5,
        mask_type="ci",
    )

    tokens_tensor = torch.tensor([token_ids], device=DEVICE)
    progress_queue: queue.Queue[dict[str, Any]] = queue.Queue()

    def on_progress(current: int, total: int, stage: str) -> None:
        progress_queue.put({"current": current, "total": total, "stage": stage})

    # Run optimization in thread
    result_holder: list[Any] = []
    error_holder: list[Exception] = []

    def compute():
        try:
            with manager.gpu_lock():
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
                result_holder.append(result)
        except Exception as e:
            error_holder.append(e)

    thread = threading.Thread(target=compute)
    thread.start()

    # Yield progress events (throttle logging to every 10% or 10 steps)
    last_logged_step = -1
    log_interval = max(1, steps // 10)

    while thread.is_alive() or not progress_queue.empty():
        try:
            progress = progress_queue.get(timeout=0.1)
            current = progress["current"]
            # Log to events.jsonl at intervals (for human monitoring)
            if current - last_logged_step >= log_interval or current == progress["total"]:
                _log_event(
                    "optimization_progress",
                    f"optimize_graph: step {current}/{progress['total']} ({progress['stage']})",
                    {"prompt": prompt_text, "target": target_token, **progress},
                )
                last_logged_step = current
            # Always yield to SSE stream (for Claude)
            yield {"type": "progress", **progress}
        except queue.Empty:
            continue

    thread.join()

    if error_holder:
        raise error_holder[0]

    if not result_holder:
        raise RuntimeError("Optimization completed but no result was produced")

    result = result_holder[0]

    ci_masked_out_logits = result.ci_masked_out_logits.cpu()
    target_out_logits = result.target_out_logits.cpu()

    # Build output probs for response
    out_probs = _build_out_probs(
        ci_masked_out_logits,
        target_out_logits,
        loaded.tokenizer.get_tok_display,
    )

    # Save graph to DB
    from spd.app.backend.database import OptimizationParams

    opt_params = OptimizationParams(
        imp_min_coeff=0.1,
        steps=steps,
        pnorm=0.5,
        beta=0.0,
        mask_type="ci",
        loss=loss_config,
        ci_masked_label_prob=result.metrics.ci_masked_label_prob,
        stoch_masked_label_prob=result.metrics.stoch_masked_label_prob,
        adv_pgd_label_prob=result.metrics.adv_pgd_label_prob,
    )
    graph_id = manager.db.save_graph(
        prompt_id=prompt_id,
        graph=StoredGraph(
            graph_type="optimized",
            edges=result.edges,
            edges_abs=result.edges_abs,
            ci_masked_out_logits=ci_masked_out_logits,
            target_out_logits=target_out_logits,
            node_ci_vals=result.node_ci_vals,
            node_subcomp_acts=result.node_subcomp_acts,
            optimization_params=opt_params,
        ),
    )

    # Filter nodes by CI threshold
    active_components = {k: v for k, v in result.node_ci_vals.items() if v >= ci_threshold}

    # Get target token probability
    target_key = f"{loss_position}:{label_token}"
    target_prob = out_probs.get(target_key)

    token_strings = [loaded.tokenizer.get_tok_display(t) for t in token_ids]

    final_result = {
        "graph_id": graph_id,
        "prompt_id": prompt_id,
        "tokens": token_strings,
        "target_token": target_token,
        "target_token_id": label_token,
        "target_position": loss_position,
        "target_probability": target_prob.prob if target_prob else None,
        "target_probability_baseline": target_prob.target_prob if target_prob else None,
        "active_components": active_components,
        "total_active": len(active_components),
        "output_probs": {k: {"prob": v.prob, "token": v.token} for k, v in out_probs.items()},
    }

    _log_event(
        "tool_complete",
        f"optimize_graph complete: {len(active_components)} active components",
        {"graph_id": graph_id, "target_prob": target_prob.prob if target_prob else None},
    )

    yield {"type": "result", "data": final_result}


def _tool_get_component_info(params: dict[str, Any]) -> dict[str, Any]:
    """Get detailed information about a component."""
    _, loaded = _get_state()

    layer = params["layer"]
    component_idx = params["component_idx"]
    top_k = params.get("top_k", 20)
    canonical_key = f"{layer}:{component_idx}"

    # Harvest/interp repos store concrete keys (e.g. "h.0.mlp.c_fc:444")
    concrete_layer = loaded.topology.canon_to_target(layer)
    concrete_key = f"{concrete_layer}:{component_idx}"

    _log_event(
        "tool_call",
        f"get_component_info: {canonical_key}",
        {"layer": layer, "idx": component_idx},
    )

    result: dict[str, Any] = {"component_key": canonical_key}

    # Get interpretation
    if loaded.interp is not None:
        interp = loaded.interp.get_interpretation(concrete_key)
        if interp is not None:
            result["interpretation"] = {
                "label": interp.label,
                "confidence": interp.confidence,
                "reasoning": interp.reasoning,
            }
        else:
            result["interpretation"] = None
    else:
        result["interpretation"] = None

    # Get token stats
    assert loaded.harvest is not None, "harvest data not loaded"
    token_stats = loaded.harvest.get_token_stats()
    if token_stats is not None:
        input_stats = analysis.get_input_token_stats(
            token_stats, concrete_key, loaded.tokenizer, top_k
        )
        output_stats = analysis.get_output_token_stats(
            token_stats, concrete_key, loaded.tokenizer, top_k
        )
        if input_stats and output_stats:
            result["token_stats"] = {
                "input": {
                    "top_recall": input_stats.top_recall,
                    "top_precision": input_stats.top_precision,
                    "top_pmi": input_stats.top_pmi,
                },
                "output": {
                    "top_recall": output_stats.top_recall,
                    "top_precision": output_stats.top_precision,
                    "top_pmi": output_stats.top_pmi,
                    "bottom_pmi": output_stats.bottom_pmi,
                },
            }
        else:
            result["token_stats"] = None
    else:
        result["token_stats"] = None

    # Get correlations (return canonical keys)
    correlations = loaded.harvest.get_correlations()
    if correlations is not None and analysis.has_component(correlations, concrete_key):
        result["correlated_components"] = {
            "precision": [
                {"key": _canonicalize_key(c.component_key, loaded), "score": c.score}
                for c in analysis.get_correlated_components(
                    correlations, concrete_key, "precision", top_k
                )
            ],
            "pmi": [
                {"key": _canonicalize_key(c.component_key, loaded), "score": c.score}
                for c in analysis.get_correlated_components(
                    correlations, concrete_key, "pmi", top_k
                )
            ],
        }
    else:
        result["correlated_components"] = None

    return result


def _tool_run_ablation(params: dict[str, Any]) -> dict[str, Any]:
    """Run ablation with selected components."""
    from spd.app.backend.compute import (
        DEFAULT_EVAL_PGD_CONFIG,
        compute_intervention,
    )
    from spd.app.backend.optim_cis import MeanKLLossConfig

    manager, loaded = _get_state()

    text = params["text"]
    selected_nodes = params["selected_nodes"]
    top_k = params.get("top_k", 10)

    _log_event(
        "tool_call",
        f"run_ablation: '{text[:50]}...' with {len(selected_nodes)} nodes",
        {"text": text, "n_nodes": len(selected_nodes)},
    )

    token_ids = loaded.tokenizer.encode(text)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)

    active_nodes = []
    for key in selected_nodes:
        parts = key.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid node key format: {key!r} (expected 'layer:seq:cIdx')")
        layer, seq_str, cidx_str = parts
        if layer in ("wte", "embed", "output"):
            raise ValueError(f"Cannot intervene on {layer!r} nodes - only internal layers allowed")
        active_nodes.append((layer, int(seq_str), int(cidx_str)))

    with manager.gpu_lock():
        result = compute_intervention(
            model=loaded.model,
            tokens=tokens,
            active_nodes=active_nodes,
            nodes_to_ablate=None,
            tokenizer=loaded.tokenizer,
            adv_pgd_config=DEFAULT_EVAL_PGD_CONFIG,
            loss_config=MeanKLLossConfig(),
            sampling=loaded.config.sampling,
            top_k=top_k,
        )

    predictions = []
    for pos_predictions in result.ci:
        pos_result = []
        for pred in pos_predictions:
            pos_result.append(
                {
                    "token": pred.token,
                    "token_id": pred.token_id,
                    "circuit_prob": round(pred.prob, 6),
                    "full_model_prob": round(pred.target_prob, 6),
                }
            )
        predictions.append(pos_result)

    return {
        "input_tokens": result.input_tokens,
        "predictions_per_position": predictions,
        "selected_nodes": selected_nodes,
    }


def _tool_search_dataset(params: dict[str, Any]) -> dict[str, Any]:
    """Search the SimpleStories dataset."""
    import time

    from datasets import Dataset, load_dataset

    query = params["query"]
    limit = params.get("limit", 20)
    search_query = query.lower()

    _log_event("tool_call", f"search_dataset: '{query}'", {"query": query, "limit": limit})

    start_time = time.time()
    dataset = load_dataset("lennart-finke/SimpleStories", split="train")
    assert isinstance(dataset, Dataset)

    filtered = dataset.filter(
        lambda x: search_query in x["story"].lower(),
        num_proc=4,
    )

    results = []
    for i, item in enumerate(filtered):
        if i >= limit:
            break
        item_dict: dict[str, Any] = dict(item)
        story: str = item_dict["story"]
        results.append(
            {
                "story": story[:500] + "..." if len(story) > 500 else story,
                "occurrence_count": story.lower().count(search_query),
            }
        )

    return {
        "query": query,
        "total_matches": len(filtered),
        "returned": len(results),
        "search_time_seconds": round(time.time() - start_time, 2),
        "results": results,
    }


def _tool_create_prompt(params: dict[str, Any]) -> dict[str, Any]:
    """Create a prompt from text."""
    manager, loaded = _get_state()

    text = params["text"]

    _log_event("tool_call", f"create_prompt: '{text[:50]}...'", {"text": text})

    token_ids = loaded.tokenizer.encode(text)
    if not token_ids:
        raise ValueError("Text produced no tokens")

    prompt_id = manager.db.add_custom_prompt(
        run_id=loaded.run.id,
        token_ids=token_ids,
        context_length=loaded.context_length,
    )

    # Compute next token probs
    tokens_tensor = torch.tensor([token_ids], device=DEVICE)
    with torch.no_grad():
        logits = loaded.model(tokens_tensor)
        probs = torch.softmax(logits, dim=-1)

    next_token_probs = []
    for i in range(len(token_ids) - 1):
        next_token_id = token_ids[i + 1]
        prob = probs[0, i, next_token_id].item()
        next_token_probs.append(round(prob, 6))
    next_token_probs.append(None)

    token_strings = [loaded.tokenizer.get_tok_display(t) for t in token_ids]

    return {
        "prompt_id": prompt_id,
        "text": text,
        "tokens": token_strings,
        "token_ids": token_ids,
        "next_token_probs": next_token_probs,
    }


def _require_investigation_config() -> InvestigationConfig:
    """Get investigation config, raising if not in investigation mode."""
    assert _investigation_config is not None, "Not running in investigation mode"
    return _investigation_config


def _tool_update_research_log(params: dict[str, Any]) -> dict[str, Any]:
    """Append content to the research log."""
    config = _require_investigation_config()
    content = params["content"]
    research_log_path = config.investigation_dir / "research_log.md"

    _log_event(
        "tool_call", f"update_research_log: {len(content)} chars", {"preview": content[:100]}
    )

    with open(research_log_path, "a") as f:
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")

    return {"status": "ok", "path": str(research_log_path)}


def _tool_save_explanation(params: dict[str, Any]) -> dict[str, Any]:
    """Save a behavior explanation to explanations.jsonl."""
    from spd.investigate.schemas import BehaviorExplanation, ComponentInfo, Evidence

    config = _require_investigation_config()

    _log_event(
        "tool_call",
        f"save_explanation: '{params['behavior_description'][:50]}...'",
        {"prompt": params["subject_prompt"]},
    )

    components = [
        ComponentInfo(
            component_key=c["component_key"],
            role=c["role"],
            interpretation=c.get("interpretation"),
        )
        for c in params["components_involved"]
    ]

    evidence = [
        Evidence(
            evidence_type=e["evidence_type"],
            description=e["description"],
            details=e.get("details", {}),
        )
        for e in params.get("supporting_evidence", [])
    ]

    explanation = BehaviorExplanation(
        subject_prompt=params["subject_prompt"],
        behavior_description=params["behavior_description"],
        components_involved=components,
        explanation=params["explanation"],
        supporting_evidence=evidence,
        confidence=params["confidence"],
        alternative_hypotheses=params.get("alternative_hypotheses", []),
        limitations=params.get("limitations", []),
    )

    explanations_path = config.investigation_dir / "explanations.jsonl"
    with open(explanations_path, "a") as f:
        f.write(explanation.model_dump_json() + "\n")

    _log_event(
        "explanation",
        f"Saved explanation: {params['behavior_description']}",
        {"confidence": params["confidence"], "n_components": len(components)},
    )

    return {"status": "ok", "path": str(explanations_path)}


def _tool_set_investigation_summary(params: dict[str, Any]) -> dict[str, Any]:
    """Set the investigation title and summary."""
    config = _require_investigation_config()

    summary = {
        "title": params["title"],
        "summary": params["summary"],
        "status": params.get("status", "in_progress"),
        "updated_at": datetime.now(UTC).isoformat(),
    }

    _log_event(
        "tool_call",
        f"set_investigation_summary: {params['title']}",
        summary,
    )

    summary_path = config.investigation_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return {"status": "ok", "path": str(summary_path)}


def _tool_save_graph_artifact(params: dict[str, Any]) -> dict[str, Any]:
    """Save a graph as an artifact for the research report.

    Uses the same filtering logic as the main graph API:
    1. Filter nodes by CI threshold
    2. Add pseudo nodes (wte, output)
    3. Filter edges to only active nodes
    4. Apply edge limit
    """
    config = _require_investigation_config()
    manager, loaded = _get_state()

    graph_id = params["graph_id"]
    caption = params.get("caption")
    ci_threshold = params.get("ci_threshold", 0.5)
    edge_limit = params.get("edge_limit", 5000)

    _log_event(
        "tool_call",
        f"save_graph_artifact: graph_id={graph_id}",
        {"graph_id": graph_id, "caption": caption},
    )

    # Fetch graph from DB
    result = manager.db.get_graph(graph_id)
    if result is None:
        raise ValueError(f"Graph with id={graph_id} not found")

    graph, prompt_id = result

    # Get tokens from prompt
    prompt_record = manager.db.get_prompt(prompt_id)
    if prompt_record is None:
        raise ValueError(f"Prompt with id={prompt_id} not found")

    tokens = [loaded.tokenizer.get_tok_display(tid) for tid in prompt_record.token_ids]
    num_tokens = len(tokens)

    # Create artifacts directory
    artifacts_dir = config.investigation_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Generate artifact ID (find max existing number to avoid collisions)
    existing_nums = []
    for f in artifacts_dir.glob("graph_*.json"):
        try:
            num = int(f.stem.split("_")[1])
            existing_nums.append(num)
        except (IndexError, ValueError):
            continue
    artifact_num = max(existing_nums, default=0) + 1
    artifact_id = f"graph_{artifact_num:03d}"

    # Compute out_probs from stored logits
    out_probs = _build_out_probs(
        graph.ci_masked_out_logits,
        graph.target_out_logits,
        loaded.tokenizer.get_tok_display,
    )

    # Step 1: Filter nodes by CI threshold (same as main graph API)
    filtered_ci_vals = {k: v for k, v in graph.node_ci_vals.items() if v > ci_threshold}
    l0_total = len(filtered_ci_vals)

    # Step 2: Add pseudo nodes (embed and output) - same as _add_pseudo_layer_nodes
    node_ci_vals_with_pseudo = dict(filtered_ci_vals)
    for seq_pos in range(num_tokens):
        node_ci_vals_with_pseudo[f"embed:{seq_pos}:0"] = 1.0
    for key, out_prob in out_probs.items():
        seq_pos, token_id = key.split(":")
        node_ci_vals_with_pseudo[f"output:{seq_pos}:{token_id}"] = out_prob.prob

    # Step 3: Filter edges to only active nodes
    active_node_keys = set(node_ci_vals_with_pseudo.keys())
    filtered_edges = [
        e
        for e in graph.edges
        if str(e.source) in active_node_keys and str(e.target) in active_node_keys
    ]

    # Step 4: Sort by strength and apply edge limit
    filtered_edges.sort(key=lambda e: abs(e.strength), reverse=True)
    filtered_edges = filtered_edges[:edge_limit]

    # Build edges data
    edges_data = [
        {
            "src": str(e.source),
            "tgt": str(e.target),
            "val": e.strength,
        }
        for e in filtered_edges
    ]

    # Compute max abs attr from filtered edges
    max_abs_attr = max((abs(e.strength) for e in filtered_edges), default=0.0)

    # Filter nodeSubcompActs to match nodeCiVals
    filtered_subcomp_acts = {
        k: v for k, v in graph.node_subcomp_acts.items() if k in node_ci_vals_with_pseudo
    }

    # Build artifact data (self-contained GraphData, same structure as API response)
    artifact = {
        "type": "graph",
        "id": artifact_id,
        "caption": caption,
        "graph_id": graph_id,
        "data": {
            "tokens": tokens,
            "edges": edges_data,
            "outputProbs": {
                k: {
                    "prob": v.prob,
                    "logit": v.logit,
                    "target_prob": v.target_prob,
                    "target_logit": v.target_logit,
                    "token": v.token,
                }
                for k, v in out_probs.items()
            },
            "nodeCiVals": node_ci_vals_with_pseudo,
            "nodeSubcompActs": filtered_subcomp_acts,
            "maxAbsAttr": max_abs_attr,
            "l0_total": l0_total,
        },
    }

    # Save artifact
    artifact_path = artifacts_dir / f"{artifact_id}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2))

    _log_event(
        "artifact_saved",
        f"Saved graph artifact: {artifact_id}",
        {"artifact_id": artifact_id, "graph_id": graph_id, "path": str(artifact_path)},
    )

    return {"artifact_id": artifact_id, "path": str(artifact_path)}


def _tool_probe_component(params: dict[str, Any]) -> dict[str, Any]:
    """Fast CI probing on custom text for a specific component."""
    manager, loaded = _get_state()

    text = params["text"]
    layer = params["layer"]
    component_idx = params["component_idx"]

    _log_event(
        "tool_call",
        f"probe_component: '{text[:50]}...' layer={layer} idx={component_idx}",
        {"text": text, "layer": layer, "component_idx": component_idx},
    )

    token_ids = loaded.tokenizer.encode(text)
    assert token_ids, "Text produced no tokens"
    tokens_tensor = torch.tensor([token_ids], device=DEVICE)

    concrete_layer = loaded.topology.canon_to_target(layer)

    with manager.gpu_lock():
        result = compute_ci_only(
            model=loaded.model, tokens=tokens_tensor, sampling=loaded.config.sampling
        )

    ci_values = result.ci_lower_leaky[concrete_layer][0, :, component_idx].tolist()
    subcomp_acts = result.component_acts[concrete_layer][0, :, component_idx].tolist()

    # Get next token probs from target model output
    next_token_probs = []
    for i in range(len(token_ids) - 1):
        next_token_id = token_ids[i + 1]
        prob = result.target_out_probs[0, i, next_token_id].item()
        next_token_probs.append(round(prob, 6))
    next_token_probs.append(None)

    token_strings = [loaded.tokenizer.get_tok_display(t) for t in token_ids]

    return {
        "tokens": token_strings,
        "ci_values": ci_values,
        "subcomp_acts": subcomp_acts,
        "next_token_probs": next_token_probs,
    }


def _tool_get_component_activation_examples(params: dict[str, Any]) -> dict[str, Any]:
    """Get activation examples from harvest data."""
    _, loaded = _get_state()

    layer = params["layer"]
    component_idx = params["component_idx"]
    limit = params.get("limit", 10)

    concrete_layer = loaded.topology.canon_to_target(layer)
    component_key = f"{concrete_layer}:{component_idx}"

    _log_event(
        "tool_call",
        f"get_component_activation_examples: {component_key}",
        {"layer": layer, "component_idx": component_idx, "limit": limit},
    )

    assert loaded.harvest is not None, "harvest data not loaded"
    canonical_key = f"{layer}:{component_idx}"
    comp = loaded.harvest.get_component(component_key)
    if comp is None:
        return {"component_key": canonical_key, "examples": [], "total": 0}

    examples = []
    for ex in comp.activation_examples[:limit]:
        token_strings = [loaded.tokenizer.get_tok_display(t) for t in ex.token_ids]
        examples.append(
            {
                "tokens": token_strings,
                "ci_values": ex.activations["causal_importance"],
                "component_acts": ex.activations["component_activation"],
            }
        )

    return {
        "component_key": canonical_key,
        "examples": examples,
        "total": len(comp.activation_examples),
        "mean_ci": comp.mean_activations["causal_importance"],
    }


def _tool_get_model_info(_params: dict[str, Any]) -> dict[str, Any]:
    """Get architecture details about the pretrained model."""
    _, loaded = _get_state()

    _log_event("tool_call", "get_model_info", {})

    info = _get_pretrain_info(loaded.config)
    return info.model_dump()


# =============================================================================
# MCP Protocol Handler
# =============================================================================


_STREAMING_TOOLS: dict[str, Callable[..., Generator[dict[str, Any]]]] = {
    "optimize_graph": _tool_optimize_graph,
}

_SIMPLE_TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_component_info": _tool_get_component_info,
    "run_ablation": _tool_run_ablation,
    "search_dataset": _tool_search_dataset,
    "create_prompt": _tool_create_prompt,
    "update_research_log": _tool_update_research_log,
    "save_explanation": _tool_save_explanation,
    "set_investigation_summary": _tool_set_investigation_summary,
    "save_graph_artifact": _tool_save_graph_artifact,
    "probe_component": _tool_probe_component,
    "get_component_activation_examples": _tool_get_component_activation_examples,
    # "get_component_attributions": _tool_get_component_attributions,
    "get_model_info": _tool_get_model_info,
}


def _handle_initialize(_params: dict[str, Any] | None) -> dict[str, Any]:
    """Handle initialize request."""
    return {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "spd-app", "version": "1.0.0"},
    }


def _handle_tools_list() -> dict[str, Any]:
    """Handle tools/list request."""
    return {"tools": [t.model_dump() for t in TOOLS]}


def _handle_tools_call(
    params: dict[str, Any],
) -> Generator[dict[str, Any]] | dict[str, Any]:
    """Handle tools/call request. May return generator for streaming tools."""
    name = params.get("name")
    arguments = params.get("arguments", {})

    if name in _STREAMING_TOOLS:
        return _STREAMING_TOOLS[name](arguments)

    if name in _SIMPLE_TOOLS:
        result = _SIMPLE_TOOLS[name](arguments)
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

    raise ValueError(f"Unknown tool: {name}")


@router.post("/mcp")
async def mcp_endpoint(request: Request):
    """MCP JSON-RPC endpoint.

    Handles initialize, tools/list, and tools/call methods.
    Returns SSE stream for streaming tools, JSON for others.
    """
    try:
        body = await request.json()
        mcp_request = MCPRequest(**body)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content=MCPResponse(
                id=None, error={"code": -32700, "message": f"Parse error: {e}"}
            ).model_dump(exclude_none=True),
        )

    logger.info(f"[MCP] {mcp_request.method} (id={mcp_request.id})")

    try:
        if mcp_request.method == "initialize":
            result = _handle_initialize(mcp_request.params)
            return JSONResponse(
                content=MCPResponse(id=mcp_request.id, result=result).model_dump(exclude_none=True),
                headers={"Mcp-Session-Id": "spd-session"},
            )

        elif mcp_request.method == "notifications/initialized":
            # Client confirms initialization
            return JSONResponse(status_code=202, content={})

        elif mcp_request.method == "tools/list":
            result = _handle_tools_list()
            return JSONResponse(
                content=MCPResponse(id=mcp_request.id, result=result).model_dump(exclude_none=True)
            )

        elif mcp_request.method == "tools/call":
            if mcp_request.params is None:
                raise ValueError("tools/call requires params")

            result = _handle_tools_call(mcp_request.params)

            # Check if result is a generator (streaming)
            if inspect.isgenerator(result):
                # Streaming response via SSE
                gen = result  # Capture for closure

                def generate_sse() -> Generator[str]:
                    try:
                        final_result = None
                        for event in gen:
                            if event.get("type") == "progress":
                                # Send progress notification
                                progress_msg = {
                                    "jsonrpc": "2.0",
                                    "method": "notifications/progress",
                                    "params": event,
                                }
                                yield f"data: {json.dumps(progress_msg)}\n\n"
                            elif event.get("type") == "result":
                                final_result = event["data"]

                        # Send final response
                        response = MCPResponse(
                            id=mcp_request.id,
                            result={
                                "content": [
                                    {"type": "text", "text": json.dumps(final_result, indent=2)}
                                ]
                            },
                        )
                        yield f"data: {json.dumps(response.model_dump(exclude_none=True))}\n\n"
                    except Exception as e:
                        tb = traceback.format_exc()
                        logger.error(f"[MCP] Tool error: {e}\n{tb}")
                        error_response = MCPResponse(
                            id=mcp_request.id,
                            error={"code": -32000, "message": str(e)},
                        )
                        yield f"data: {json.dumps(error_response.model_dump(exclude_none=True))}\n\n"

                return StreamingResponse(generate_sse(), media_type="text/event-stream")

            else:
                # Non-streaming response
                return JSONResponse(
                    content=MCPResponse(id=mcp_request.id, result=result).model_dump(
                        exclude_none=True
                    )
                )

        else:
            return JSONResponse(
                content=MCPResponse(
                    id=mcp_request.id,
                    error={"code": -32601, "message": f"Method not found: {mcp_request.method}"},
                ).model_dump(exclude_none=True)
            )

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[MCP] Error handling {mcp_request.method}: {e}\n{tb}")
        return JSONResponse(
            content=MCPResponse(
                id=mcp_request.id,
                error={"code": -32000, "message": str(e)},
            ).model_dump(exclude_none=True)
        )
