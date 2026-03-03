"""Shared schema types used across multiple backend modules.

These types are kept here to avoid circular imports between routers,
database, state, and lib modules. Router-specific schemas should be
defined in their respective router files.
"""

from pydantic import BaseModel

# =============================================================================
# Shared Types (used by database.py, state.py, lib/activation_contexts.py)
# =============================================================================


class OutputProbability(BaseModel):
    """Output probability for a specific token at a specific position."""

    prob: float  # CI-masked (SPD model) probability
    logit: float  # CI-masked (SPD model) raw logit
    target_prob: float  # Target model probability
    target_logit: float  # Target model raw logit
    token: str


# =============================================================================
# Configuration Models
# =============================================================================


class ActivationContextsGenerationConfig(BaseModel):
    """Configuration for generating activation contexts."""

    importance_threshold: float = 0.01
    n_batches: int = 100
    batch_size: int = 32
    n_tokens_either_side: int = 5
    topk_examples: int = 20
    separation_tokens: int = 0


class SubcomponentMetadata(BaseModel):
    """Lightweight metadata for a subcomponent (without examples/token_prs)"""

    subcomponent_idx: int
    mean_ci: float


class SubcomponentActivationContexts(BaseModel):
    """Activation context data for a single subcomponent, using columnar layout for efficiency.

    Note: Token P/R/lift stats are now computed by the batch job and served via the
    /token_stats endpoint, not stored here.
    """

    subcomponent_idx: int
    mean_ci: float

    # Examples - columnar arrays (n_examples ~ topk, window_size ~ 2*n_tokens_either_side+1)
    example_tokens: list[list[str]]  # [n_examples][window_size]
    example_ci: list[list[float]]  # [n_examples][window_size]
    example_component_acts: list[
        list[float]
    ]  # [n_examples][window_size] - normalized component activations
