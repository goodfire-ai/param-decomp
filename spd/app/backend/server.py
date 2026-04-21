"""Unified FastAPI server for the SPD app.

Merges the main app backend with the prompt attributions server.
Supports multiple runs, on-demand attribution graph computation,
and activation contexts generation.

Usage:
    python -m spd.app.backend.server --port 8000
"""

import os
import time
import traceback
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import fire
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from spd.app.backend.database import PromptAttrDB
from spd.app.backend.routers import (
    activation_contexts_router,
    agents_router,
    autointerp_compare_router,
    clusters_router,
    correlations_router,
    data_sources_router,
    dataset_attributions_router,
    dataset_search_router,
    graph_interp_router,
    graphs_router,
    intervention_router,
    investigations_router,
    mcp_router,
    pretrain_info_router,
    prompts_router,
    run_registry_router,
    runs_router,
)
from spd.app.backend.state import StateManager
from spd.log import logger
from spd.settings import SPD_APP_DEFAULT_RUN
from spd.utils.distributed_utils import get_device

DEVICE = get_device()


@asynccontextmanager
async def lifespan(app: FastAPI):  # pyright: ignore[reportUnusedParameter]
    """Initialize DB connection at startup. Model loaded on-demand via /api/runs/load."""
    from spd.app.backend.routers.mcp import InvestigationConfig, set_investigation_config

    manager = StateManager.get()

    db = PromptAttrDB(check_same_thread=False)
    db.init_schema()
    manager.initialize(db)

    logger.info(f"[STARTUP] DB initialized: {db.db_path}")
    logger.info(f"[STARTUP] Device: {DEVICE}")
    logger.info(f"[STARTUP] CUDA available: {torch.cuda.is_available()}")

    # Configure MCP for investigation mode (derives paths from investigation dir)
    investigation_dir = os.environ.get("SPD_INVESTIGATION_DIR")
    if investigation_dir:
        inv_dir = Path(investigation_dir)
        set_investigation_config(
            InvestigationConfig(
                events_log_path=inv_dir / "events.jsonl",
                investigation_dir=inv_dir,
            )
        )
        logger.info(f"[STARTUP] Investigation mode enabled: dir={investigation_dir}")

    if SPD_APP_DEFAULT_RUN is not None:
        from spd.app.backend.routers.runs import load_run

        logger.info(f"[STARTUP] Auto-loading default run: {SPD_APP_DEFAULT_RUN}")
        load_run(SPD_APP_DEFAULT_RUN, context_length=512, manager=manager)

    yield

    manager.close()


app = FastAPI(title="SPD App API", lifespan=lifespan, debug=True)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def log_request_timing(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Log timing for slow requests (>1s)."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    if duration_ms > 1000:
        logger.warning(f"[SLOW] {request.method} {request.url.path} -> {duration_ms:.1f}ms")
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Log validation errors (400s) with full details."""
    logger.error(f"[VALIDATION ERROR] {request.method} {request.url.path}")
    logger.error(f"[VALIDATION ERROR] Errors: {exc.errors()}")
    if exc.body is not None:
        logger.error(f"[VALIDATION ERROR] Request body: {exc.body}")

    return JSONResponse(
        status_code=400,
        content={
            "detail": exc.errors(),
            "type": "RequestValidationError",
            "path": request.url.path,
            "method": request.method,
            "body": str(exc.body) if exc.body is not None else None,
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Log HTTP exceptions with context."""
    logger.error(f"[HTTP {exc.status_code}] {request.method} {request.url.path}")
    logger.error(f"[HTTP {exc.status_code}] Detail: {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "type": "HTTPException",
            "path": request.url.path,
            "method": request.method,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log full exception details for debugging."""
    tb = traceback.format_exc()
    logger.error(f"[ERROR] {request.method} {request.url.path}")
    logger.error(f"[ERROR] Exception: {type(exc).__name__}: {exc}")
    logger.error(f"[ERROR] Traceback:\n{tb}")

    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": type(exc).__name__,
            "path": request.url.path,
            "method": request.method,
        },
    )


# Routers
app.include_router(runs_router)
app.include_router(autointerp_compare_router)
app.include_router(prompts_router)
app.include_router(graphs_router)
app.include_router(activation_contexts_router)
app.include_router(correlations_router)
app.include_router(clusters_router)
app.include_router(intervention_router)
app.include_router(dataset_search_router)
app.include_router(dataset_attributions_router)
app.include_router(agents_router)
app.include_router(investigations_router)
app.include_router(mcp_router)
app.include_router(data_sources_router)
app.include_router(graph_interp_router)
app.include_router(pretrain_info_router)
app.include_router(run_registry_router)


def cli(port: int = 8000) -> None:
    """Run the server.

    Args:
        port: Port to serve on (default 8000)
    """
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        "spd.app.backend.server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    fire.Fire(cli)
