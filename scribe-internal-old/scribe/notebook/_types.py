"""Shared types for the scribe notebook system."""

from typing import Any, Literal

from pydantic import BaseModel

CellType = Literal["code", "markdown"]
CellStatus = Literal["ok", "error", "timeout", "running", "stale", "complete", ""]
AsyncStatus = Literal["idle", "running", "complete", "error", "cancelled"]


class CellResult(BaseModel):
    """Per-cell result from execute."""
    tag: str
    status: CellStatus


class CellOutput(BaseModel):
    """Per-cell data from read — tag, type, status, and raw jupyter outputs."""
    tag: str
    cell_type: CellType
    status: CellStatus
    outputs: list[dict[str, Any]] = []


class ExecuteResponse(BaseModel):
    """Response from blocking execute — per-cell results."""
    cells: list[CellResult]


class AsyncStarted(BaseModel):
    """Response from non-blocking execute — confirmation."""
    n_cells: int


class ReadResponse(BaseModel):
    """Response from the server's read_cells method."""
    async_status: AsyncStatus
    cells: list[CellOutput] = []
