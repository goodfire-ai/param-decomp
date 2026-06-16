"""Canonical training-state dataclass persisted to `training_<step>.pth`.

Lives in its own module so `param_decomp.run_sink` (where the sink protocol
references it) can import it without pulling in the rest of the train-loop. The
torch trainer that produced it has been retired (oracle at git tag `torch-oracle`);
this dataclass remains as the on-disk schema the sink protocol is typed against.
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
