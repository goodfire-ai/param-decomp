"""Typed per-pool training state — replaces the optional-attr bag + string dispatch.

Each rank plays exactly one of three pool roles, and each role holds a
genuinely different set of mutable training objects:

  * ``CIState``   — the CI fn's optimizer + its parameter list. No V/U, no PPGD.
  * ``LWState``   — the components' optimizer + the owned-site parameter list.
  * ``PPGDState`` — neither optimizer nor V/U params of its own; it carries the
    persistent adversarial-source state (built lazily on the first batch).

Modelling these as a discriminated union (rather than a single object with
``optimizer: Optimizer | None``, ``ppgd_state: ... | None``, ``ci_fn_params``,
``component_params`` all hanging off it) makes the per-pool variation explicit:
``match pool_state`` is exhaustive, and a phase can't reach for an attribute the
current pool doesn't have. "CI pool with no ci_fn", "PPGD pool with an
optimizer", etc. become unrepresentable.
"""

from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn
from torch.optim import Optimizer

from param_decomp.metrics.persistent_pgd_state import PersistentPGDState


@dataclass
class CIState:
    optimizer: Optimizer
    ci_fn_params: list[nn.Parameter]


@dataclass
class LWState:
    optimizer: Optimizer
    component_params: list[nn.Parameter]


@dataclass
class PPGDState:
    """PPGD has no optimizer and no owned V/U params. ``ppgd_state`` is built
    lazily on the first batch (its source shapes depend on the data's seq dims);
    ``pending_resume_state`` carries a resumed source state dict until then.
    """

    ppgd_state: PersistentPGDState | None = None
    pending_resume_state: dict[str, Any] | None = field(default=None)


PoolState = CIState | LWState | PPGDState
