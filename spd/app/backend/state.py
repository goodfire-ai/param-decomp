"""Application state management for the SPD backend.

Contains:
- RunState: Runtime state for a loaded run (model, tokenizer, repos)
- StateManager: Singleton managing app-wide state with proper lifecycle
"""

import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.database import PromptAttrDB, Run
from spd.autointerp.repo import InterpRepo
from spd.configs import Config
from spd.dataset_attributions.repo import AttributionRepo
from spd.graph_interp.repo import GraphInterpRepo
from spd.harvest.repo import HarvestRepo
from spd.models.component_model import ComponentModel
from spd.topology import TransformerTopology


@dataclass
class RunState:
    """Runtime state for a loaded run (model, tokenizer, etc.)"""

    run: Run
    model: ComponentModel
    topology: TransformerTopology
    tokenizer: AppTokenizer
    sources_by_target: dict[str, list[str]]
    config: Config
    context_length: int
    harvest: HarvestRepo | None
    interp: InterpRepo | None
    attributions: AttributionRepo | None
    graph_interp: GraphInterpRepo | None


@dataclass
class DatasetSearchState:
    """State for dataset search results (memory-only, no persistence)."""

    results: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass
class AppState:
    """Server state. DB is always available; run_state is set after /api/runs/load."""

    db: PromptAttrDB
    run_state: RunState | None = field(default=None)
    dataset_search_state: DatasetSearchState | None = field(default=None)


class StateManager:
    """Singleton managing app state with proper lifecycle.

    Use StateManager.get() to access the singleton instance.
    The instance is initialized during FastAPI lifespan startup.
    """

    _instance: "StateManager | None" = None

    def __init__(self) -> None:
        self._state: AppState | None = None
        self._gpu_lock = threading.Lock()

    @classmethod
    def get(cls) -> "StateManager":
        """Get the singleton instance, creating if needed."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    def initialize(self, db: PromptAttrDB) -> None:
        """Initialize state with database connection."""
        self._state = AppState(db=db)

    @property
    def state(self) -> AppState:
        """Get app state. Fails fast if not initialized."""
        assert self._state is not None, "App state not initialized - lifespan not started"
        return self._state

    @property
    def db(self) -> PromptAttrDB:
        """Get database connection."""
        return self.state.db

    @property
    def run_state(self) -> RunState | None:
        """Get loaded run state (may be None)."""
        return self.state.run_state

    @run_state.setter
    def run_state(self, value: RunState | None) -> None:
        """Set loaded run state."""
        self.state.run_state = value

    def close(self) -> None:
        """Clean up resources."""
        if self._state is not None:
            self._state.db.close()

    @contextmanager
    def gpu_lock(self) -> Generator[None]:
        """Acquire GPU lock or fail with 503 if another GPU operation is in progress.

        Use this for GPU-intensive endpoints to prevent concurrent operations
        that would cause the server to hang.
        """
        acquired = self._gpu_lock.acquire(blocking=False)
        if not acquired:
            raise HTTPException(
                status_code=503,
                detail="GPU operation already in progress. Please wait and retry.",
            )
        try:
            yield
        finally:
            self._gpu_lock.release()
