"""MCP (Model Context Protocol) endpoint for Claude Code integration.

This router implements the MCP JSON-RPC protocol over HTTP, allowing Claude Code
to use PD tools directly with proper schemas and streaming progress.

MCP Spec: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports
"""

import inspect
import json
import traceback
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from param_decomp.log import logger
from param_decomp_lab.app.backend.inference import next_token_probs
from param_decomp_lab.app.backend.routers.pretrain_info import _get_pretrain_info
from param_decomp_lab.app.backend.state import StateManager
from param_decomp_lab.harvest import analysis

router = APIRouter(tags=["mcp"])

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


def _tool_search_dataset(params: dict[str, Any]) -> dict[str, Any]:
    """Search the loaded run's training dataset for rows containing a query string."""
    import time

    from datasets import Dataset, load_dataset

    from param_decomp_lab.app.backend.routers.dataset_search import _assert_simplestories

    _, loaded = _get_state()
    dataset_name = loaded.lm_data.dataset_name
    text_column = loaded.lm_data.column_name
    _assert_simplestories(dataset_name)

    query = params["query"]
    limit = params.get("limit", 20)
    search_query = query.lower()

    _log_event(
        "tool_call",
        f"search_dataset: '{query}' on {dataset_name}",
        {"query": query, "limit": limit, "dataset": dataset_name},
    )

    start_time = time.time()
    dataset = load_dataset(dataset_name, split="train")
    assert isinstance(dataset, Dataset)

    filtered = dataset.filter(
        lambda x: search_query in x[text_column].lower(),
        num_proc=4,
    )

    results = []
    for i, item in enumerate(filtered):
        if i >= limit:
            break
        item_dict: dict[str, Any] = dict(item)
        text: str = item_dict[text_column]
        results.append(
            {
                "text": text[:500] + "..." if len(text) > 500 else text,
                "occurrence_count": text.lower().count(search_query),
            }
        )

    return {
        "query": query,
        "dataset_name": dataset_name,
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

    probs = next_token_probs(loaded.jax_run, token_ids)
    rounded_probs = [round(p, 6) if p is not None else None for p in probs]

    token_strings = [loaded.tokenizer.get_tok_display(t) for t in token_ids]

    return {
        "prompt_id": prompt_id,
        "text": text,
        "tokens": token_strings,
        "token_ids": token_ids,
        "next_token_probs": rounded_probs,
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
    from param_decomp_lab.investigate.schemas import (
        BehaviorExplanation,
        ComponentInfo,
        Evidence,
    )

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

    info = _get_pretrain_info(loaded.lm_target)
    return info.model_dump()


# =============================================================================
# MCP Protocol Handler
# =============================================================================


_STREAMING_TOOLS: dict[str, Callable[..., Generator[dict[str, Any]]]] = {}

_SIMPLE_TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_component_info": _tool_get_component_info,
    "search_dataset": _tool_search_dataset,
    "create_prompt": _tool_create_prompt,
    "update_research_log": _tool_update_research_log,
    "save_explanation": _tool_save_explanation,
    "set_investigation_summary": _tool_set_investigation_summary,
    "get_component_activation_examples": _tool_get_component_activation_examples,
    "get_model_info": _tool_get_model_info,
}


def _handle_initialize(_params: dict[str, Any] | None) -> dict[str, Any]:
    """Handle initialize request."""
    return {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "pd-app", "version": "1.0.0"},
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
                headers={"Mcp-Session-Id": "pd-session"},
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
