"""Scribe Notebook MCP Server — agent-facing tool definitions."""

import asyncio
import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.utilities.types import Image

from scribe.notebook._file_parser import parse_named_cells
from scribe.notebook._image_result_utils import cleanup_all_session_images, cleanup_session_images
from scribe.notebook._output_formatting import format_raw_outputs
from scribe.notebook._server_connection import cleanup as cleanup_server
from scribe.notebook._server_connection import get_status, get_token, get_url, post
from scribe.notebook._types import AsyncStarted, ExecuteResponse, ReadResponse


_MCP_INSTRUCTIONS = """\
Scribe is a managed Jupyter notebook server. You write and execute code in notebook cells.

## Code execution — file-based workflow

1. `start_new_session` returns a `script_path` — a .py file for your code.
2. Write code using Write/Edit tools with named `# %% tag` cell markers.
   Use `# %% md:tag` for markdown cells.
3. Call `run_cell(session_id, "tag")` to execute, or `run_cells(session_id)` for all.

For long-running cells, use `start_cell`/`start_cells` (non-blocking) + `get_output` to poll.
Use `cancel` to interrupt. Use `restart_kernel` for a fresh kernel.
"""

mcp = FastMCP("scribe", instructions=_MCP_INSTRUCTIONS)

# Config from environment
_save_images_locally = os.environ.get("SCRIBE_SAVE_NB_IMAGES", "").lower() == "true"
_compact_output = os.environ.get("SCRIBE_COMPACT_OUTPUT", "true").lower() == "true"

# Session state
_active_sessions: set[str] = set()
_session_script_paths: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _sync_session(session_id: str) -> None:
    """Read the session's script file, parse cells, sync to notebook."""
    script_path = _session_script_paths.get(session_id)
    assert script_path, f"No script_path cached for session {session_id}"

    path = Path(script_path)
    if not path.exists():
        return

    content = path.read_text()
    cells = parse_named_cells(content)
    if not cells:
        return

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, post, "sync",
        {"session_id": session_id, "cells": [[c.tag, c.cell_type, c.source] for c in cells]},
    )


def _format_cell_result(tag: str, raw_outputs: list, status: str, session_id: str) -> list[str | Image]:
    """Format a single cell's outputs into text + images for the agent."""
    formatted = format_raw_outputs(raw_outputs, session_id, _save_images_locally, _compact_output)
    output_text = next((s for s in formatted if isinstance(s, str)), "")
    images = [s for s in formatted if not isinstance(s, str)]

    match status:
        case "running":
            prefix = f"[{tag}] {output_text}\n[still running...]" if output_text and output_text != "(no output)" else f"[{tag}] [still running...]"
        case "timeout":
            prefix = f"[{tag}] ERROR — timed out"
        case _ if output_text and output_text != "(no output)":
            prefix = f"[{tag}] {output_text}"
        case _:
            prefix = f"[{tag}] ok"

    return [prefix, *images]


# ---------------------------------------------------------------------------
# Session tools
# ---------------------------------------------------------------------------

@mcp.tool
async def start_new_session(
    experiment_name: str | None = None,
    notebook_dir: str | None = None,
) -> dict[str, Any]:
    """
    Start a new Jupyter kernel session with an empty notebook.

    Args:
        experiment_name: Custom name for the notebook.
        notebook_dir: Custom directory. Only use if the user explicitly requested it.

    Returns:
        session_id, notebook_path, script_path, vscode_url, kernel_id, status, kernel_name.
    """
    body: dict[str, Any] = {}
    if experiment_name:
        body["experiment_name"] = experiment_name
    if notebook_dir:
        body["notebook_dir"] = notebook_dir

    data = post("start", body)
    session_id = data["session_id"]
    script_path = data["script_path"]

    _session_script_paths[session_id] = script_path
    _active_sessions.add(session_id)

    url = get_url()
    token = get_token()

    return {
        "session_id": session_id,
        "notebook_path": data["notebook_path"],
        "script_path": script_path,
        "vscode_url": f"{data.get('server_url', url)}/?token={data.get('token', token)}",
        "kernel_id": data.get("kernel_id"),
        "status": "started",
        "kernel_name": data.get("kernel_name", data.get("kernel_display_name", "Scribe Kernel")),
    }


@mcp.tool
async def restart_kernel(session_id: str) -> str:
    """
    Restart the kernel, clearing all state. Notebook and script file are kept.
    Follow with `run_cells` to selectively restore state.
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, post, "restart", {"session_id": session_id})
    return f"Kernel restarted. State cleared."


@mcp.tool
async def shutdown_session(session_id: str) -> str:
    """Shutdown a kernel session. Only use if the user instructs you to."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, post, "shutdown", {"session_id": session_id})
    _session_script_paths.pop(session_id, None)
    if _save_images_locally and session_id in _active_sessions:
        cleanup_session_images(session_id)
    _active_sessions.discard(session_id)
    return f"Session shut down."


# ---------------------------------------------------------------------------
# Sync execution
# ---------------------------------------------------------------------------

@mcp.tool
async def run_cell(session_id: str, tag: str, timeout_seconds: int = 300) -> list[str | Image]:
    """
    Sync script file and execute a single cell (blocking).

    Args:
        session_id: The session ID.
        tag: Cell tag to execute.
        timeout_seconds: Max execution time. Default 300.
    """
    await _sync_session(session_id)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, post, "execute",
        {"session_id": session_id, "tags": [tag], "timeout_per_cell": timeout_seconds},
    )

    read = ReadResponse(**await loop.run_in_executor(None, post, "read", {"session_id": session_id, "cell_tag": tag}))
    assert read.cells, f"No output for tag '{tag}'"
    return format_raw_outputs(read.cells[0].outputs, session_id, _save_images_locally, _compact_output)


@mcp.tool
async def run_cells(
    session_id: str, tags: list[str] | None = None, timeout_seconds: int = 300,
) -> list[str | Image]:
    """
    Sync script file and execute cells (blocking). Runs all if tags is None. Stops on first error.

    Args:
        session_id: The session ID.
        tags: Cell tags to execute, or None for all.
        timeout_seconds: Per-cell timeout. Default 300.
    """
    await _sync_session(session_id)

    loop = asyncio.get_event_loop()
    exec_raw = await loop.run_in_executor(
        None, post, "execute",
        {"session_id": session_id, "tags": tags, "timeout_per_cell": timeout_seconds},
    )
    execute = ExecuteResponse(**exec_raw)

    read = ReadResponse(**await loop.run_in_executor(None, post, "read", {"session_id": session_id}))
    outputs_by_tag = {c.tag: c for c in read.cells}

    results: list[str | Image] = []
    for cell_result in execute.cells:
        cell_data = outputs_by_tag.get(cell_result.tag)
        outputs = cell_data.outputs if cell_data else []
        results.extend(_format_cell_result(cell_result.tag, outputs, cell_result.status, session_id))
    return results


# ---------------------------------------------------------------------------
# Async execution
# ---------------------------------------------------------------------------

@mcp.tool
async def start_cell(session_id: str, tag: str) -> str:
    """
    Start executing a cell in the background (non-blocking).
    Poll `get_output` for progress. Use `cancel` to stop.
    """
    await _sync_session(session_id)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, post, "execute-async",
        {"session_id": session_id, "tags": [tag]},
    )
    return f"Cell '{tag}' started. Use get_output to poll."


@mcp.tool
async def start_cells(session_id: str, tags: list[str] | None = None) -> str:
    """
    Start executing cells in the background (non-blocking). All if tags is None.
    Poll `get_output` for progress.
    """
    await _sync_session(session_id)
    loop = asyncio.get_event_loop()
    started = AsyncStarted(**await loop.run_in_executor(
        None, post, "execute-async",
        {"session_id": session_id, "tags": tags},
    ))
    return f"Executing {started.n_cells} cells. Use get_output to poll."


# ---------------------------------------------------------------------------
# Inspect + control
# ---------------------------------------------------------------------------

@mcp.tool
async def get_output(session_id: str, tag: str | None = None) -> list[str | Image]:
    """
    Read cell outputs from the notebook. Works for completed and in-progress executions.

    Args:
        session_id: The session ID.
        tag: Specific cell tag, or None for all.
    """
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, post, "read", {"session_id": session_id, "cell_tag": tag})

    cells = data.get("cells", [])
    if not cells:
        return ["No cells found."]

    results: list[str | Image] = []
    for cell_data in cells:
        cell_tag = cell_data["tag"]
        if cell_data.get("cell_type") == "markdown":
            results.append(f"[{cell_tag}] (markdown)")
            continue
        results.extend(_format_cell_result(
            cell_tag, cell_data.get("outputs", []), cell_data.get("status", ""), session_id
        ))

    if data.get("async_status") == "running":
        results.append("[execution in progress...]")

    return results


@mcp.tool
async def cancel(session_id: str) -> str:
    """Cancel a running background execution. Interrupts the kernel."""
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, post, "cancel", {"session_id": session_id})
    if data.get("status") == "idle":
        return "No running execution to cancel."
    return "Execution cancelled."


# ---------------------------------------------------------------------------
# Resource
# ---------------------------------------------------------------------------

@mcp.resource(
    uri="scribe://server/status",
    name="ScribeNotebookServerStatus",
    description="Current Scribe server status and connection info.",
)
async def server_status() -> str:
    status = get_status()
    lines = [
        "# Scribe Server Status", "",
        f"**Status:** {status['status']}",
        f"**URL:** {status['url'] or 'N/A'}",
        f"**Port:** {status['port'] or 'N/A'}",
        f"**VSCode URL:** {status.get('vscode_url') or 'N/A'}",
        f"**Health:** {status['health']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
