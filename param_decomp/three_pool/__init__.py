"""3-pool training strategy for SPD on large frozen target models.

Splits GPUs into three heterogeneous pools that run in wall-clock parallel:

  CI pool         owns the CI fn + AdamW state (replicated across ranks, DP
                  across batch). Runs target+CI forward, importance-minimality
                  loss, fused backward through CI fn graph seeded by downstream
                  pools' CI gradients. Enables global shared CI fns (which
                  2-pool's sharded-by-site Pool A cannot).
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

from param_decomp.three_pool.config import LayerwiseBlockGroupSpec, ThreePoolConfig
from param_decomp.three_pool.optimize import optimize_three_pool
from param_decomp.three_pool.profiler import PhaseProfiler

__all__ = [
    "LayerwiseBlockGroupSpec",
    "PhaseProfiler",
    "ThreePoolConfig",
    "optimize_three_pool",
]
