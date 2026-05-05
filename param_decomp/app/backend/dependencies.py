"""FastAPI dependency injection helpers.

Provides Depends() callables for injecting state into route handlers.
This enables clean separation of concerns and easier testing.

Usage in routers:
    from param_decomp.app.backend.dependencies import DepStateManager, DepLoadedRun

    @router.get("/endpoint")
    def my_endpoint(manager: DepStateManager, loaded: DepLoadedRun):
        ...
"""

from typing import Annotated

from fastapi import Depends, HTTPException

from param_decomp.app.backend.database import PromptAttrDB
from param_decomp.app.backend.state import RunState, StateManager
from param_decomp.log import logger


def get_state_manager() -> StateManager:
    """Get the StateManager singleton."""
    try:
        return StateManager.get()
    except Exception as e:
        logger.error(f"[DEPENDENCY] Failed to get StateManager: {e}")
        raise


def get_db() -> PromptAttrDB:
    """Get database connection."""
    return StateManager.get().db


def get_loaded_run() -> RunState:
    """Get loaded run state. Raises HTTPException if no run is loaded."""
    manager = StateManager.get()
    if manager.run_state is None:
        raise HTTPException(
            status_code=400, detail="No run loaded. Call POST /api/runs/load first."
        )
    return manager.run_state


# Type aliases for dependency injection (avoids B008 linter warnings)
DepStateManager = Annotated[StateManager, Depends(get_state_manager)]
DepLoadedRun = Annotated[RunState, Depends(get_loaded_run)]
DepDB = Annotated[PromptAttrDB, Depends(get_db)]
