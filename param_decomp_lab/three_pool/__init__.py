"""3-pool training strategy for SPD on large frozen target models.

Splits GPUs into three heterogeneous pools that run in wall-clock parallel:

  CI pool         owns the CI fn + AdamW state (replicated across ranks, DP
                  across batch). Runs target+CI forward, importance-minimality
                  loss, fused backward through CI fn graph seeded by downstream
                  pools' CI gradients. Enables global shared CI fns by giving
                  the CI fn a dedicated, unsharded pool.
  Layerwise pool  owns V/U + AdamW state (sharded by site, block-DDP within
                  group). Runs target forward, faithfulness loss, per-site
                  streaming layerwise stoch recon.
  PPGD pool       stateless full V/U replica + persistent PPGD sources. Runs
                  target forward, PPGD warmup (inner loop owns source updates),
                  final recon backward seeding V/U + CI grads only.

``optimize_three_pool`` mirrors :func:`param_decomp.optimize.optimize`'s call
shape. ``ThreePoolConfig`` declares the topology.

See ``DESIGN.md`` for the per-step dependency graph + the pipelining tricks.
"""

from typing import TYPE_CHECKING, Any

from param_decomp_lab.three_pool.config import LayerwiseBlockGroupSpec, ThreePoolConfig

if TYPE_CHECKING:
    from param_decomp_lab.three_pool.optimize import ThreePoolTrainer, optimize_three_pool

__all__ = [
    "LayerwiseBlockGroupSpec",
    "ThreePoolConfig",
    "ThreePoolTrainer",
    "optimize_three_pool",
]


def __getattr__(name: str) -> Any:
    # Lazy export of the heavy `optimize` symbols (PEP 562) so importing leaf
    # submodules (e.g. `three_pool.routing_plan`, pulled by `three_pool_pd`) does
    # not eagerly import `optimize` — which imports `three_pool_pd` back, a cycle.
    if name in ("ThreePoolTrainer", "optimize_three_pool"):
        from param_decomp_lab.three_pool import optimize

        return getattr(optimize, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
