"""Application state management for the PD backend.

The app is a read-only viewer over a saved JAX run: it opens the orbax checkpoint via
`open_jax_run` (the model forward, torch-free) and reads the run's pinned torch-free
`LMExperimentConfig` for target/data/algorithm metadata, plus the pre-computed
harvest/autointerp/cluster repos. No torch.
"""

from dataclasses import dataclass, field
from typing import Any

from jax_single_pool.load_run import LoadedJaxRun

from param_decomp_config.lm import LMDataConfig, LMTargetConfig
from param_decomp_config.pd import PDConfig
from param_decomp_lab.app.backend.app_tokenizer import AppTokenizer
from param_decomp_lab.app.backend.database import PromptAttrDB, Run
from param_decomp_lab.app.backend.topology import AppTopology
from param_decomp_lab.autointerp.repo import InterpRepo
from param_decomp_lab.harvest.repo import HarvestRepo


@dataclass
class RunState:
    """Runtime state for a loaded JAX run (model forward + tokenizer + repos)."""

    run: Run
    jax_run: LoadedJaxRun
    topology: AppTopology
    tokenizer: AppTokenizer
    config: PDConfig
    # The token-based app only loads LM runs.
    lm_target: LMTargetConfig
    lm_data: LMDataConfig
    context_length: int
    harvest: HarvestRepo | None
    interp: InterpRepo | None


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
