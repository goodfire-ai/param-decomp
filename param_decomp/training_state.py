"""Canonical training-state dataclasses persisted to `training_<step>.pth`.

`TrainingState` is the 1-pool shape; `ThreePoolTrainingState` is the 3-pool shape.
The two are deliberately separate (no shared base class) — common fields are
duplicated. Pool-specific fields (e.g. `three_pool_config`, `layout_fingerprint`)
live on the concrete dataclass that needs them.

Lives in its own module so both `param_decomp.optimize` /
`param_decomp_lab.three_pool.optimize` (where trainers produce these) and
`param_decomp.run_sink` (where they're consumed by the sink protocol) can
import without a cycle.
"""

from dataclasses import dataclass
from typing import Any

from torch import Tensor


@dataclass(frozen=True)
class TrainingState:
    """Canonical 1-pool training state, persisted to `training_<step>.pth`.

    Produced by `Trainer.snapshot()` and consumed by `Trainer.from_snapshot()`
    to reconstruct the trainer. For DDP, every rank produces an identical
    instance (model and optimizers are replicated); rank 0's write is the
    canonical artifact.

    Optimizer states are keyed by parameter name (e.g.
    `components.h.0.attn.q_proj.V`, `ci_fn.embed.weight`) rather than the
    optimizer's integer indices, so they survive a topology change on resume.
    """

    step: int
    pd_config: dict[str, Any]
    runtime_config: dict[str, Any]
    component_model: dict[str, Tensor]
    components_optimizer: dict[str, dict[str, Any]]
    ci_fn_optimizer: dict[str, dict[str, Any]]
    loss_metrics: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ThreePoolTrainingState:
    """Canonical 3-pool training state, persisted to `training_<step>.pth`.

    Produced by `ThreePoolTrainer.snapshot()` on rank 0 only — non-rank-0 ranks
    return `None` instead (the lab sink is silent on those ranks anyway).
    Rank 0 assembles the canonical state from cross-rank gathers:

    * `component_model`: full gathered model state (every site's V/U + the CI fn).
    * `components_optimizer` / `ci_fn_optimizer`: per-parameter optimizer state
      keyed by parameter name, merged across the chunkwise + CI pools. Same
      shape as 1-pool's `TrainingState` — topology-independent.
    * `layout_fingerprint`: 3-pool world layout summary. Checked on resume to
      flag incompatible topologies.

    The PPGD adversarial sources are NOT in here. They're `bsc`-scoped
    (sized by `batch x seq x n_components`) — the one piece of persisted state that's
    data-shaped rather than parameter-shaped, so aggregating it onto one rank doesn't
    scale (~2.3 TB at batch 1280). It stays sharded per-rank in `ppgd_<step>/rank_<r>.pth`
    next to this file; each adversary rank reads its own shard on resume. That ties a
    PPGD resume to the same pool layout (a missing shard ⇒ the adversary re-warms).
    """

    step: int
    pd_config: dict[str, Any]
    runtime_config: dict[str, Any]
    three_pool_config: dict[str, Any]
    layout_fingerprint: dict[str, Any]
    component_model: dict[str, Tensor]
    components_optimizer: dict[str, dict[str, Any]]
    ci_fn_optimizer: dict[str, dict[str, Any]]
