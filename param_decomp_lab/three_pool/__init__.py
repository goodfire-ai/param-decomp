"""3-pool training strategy for VPD on large frozen target models.

Splits GPUs into three heterogeneous pools that run in wall-clock parallel:

  CI pool         owns the CI fn + AdamW state (replicated across ranks, DP
                  across batch). Runs target+CI forward, importance-minimality
                  loss, fused backward through CI fn graph seeded by downstream
                  pools' CI gradients. Enables global shared CI fns by giving
                  the CI fn a dedicated, unsharded pool.
  Chunkwise pool  owns V/U + AdamW state (sharded into chunks of sites, DDP
                  within chunk). Runs target forward, faithfulness loss,
                  per-site streaming chunkwise stoch recon.
  PPGD pool       stateless full V/U replica + persistent PPGD sources. Runs
                  target forward, PPGD warmup (inner loop owns source updates),
                  final recon backward seeding V/U + CI grads only.

``optimize_three_pool`` mirrors :func:`param_decomp.optimize.optimize`'s call
shape. ``ThreePoolTopology`` declares the topology.

See ``DESIGN.md`` for the per-step dependency graph + the pipelining tricks.
"""

from param_decomp_lab.three_pool.config import (
    ChunkwiseSpec,
    PoolSpec,
    ResolvedLayout,
    ThreePoolTopology,
)
from param_decomp_lab.three_pool.optimize import ThreePoolTrainer, optimize_three_pool
from param_decomp_lab.three_pool.two_pool_config import TwoPoolResolvedLayout, TwoPoolTopology
from param_decomp_lab.three_pool.two_pool_optimize import TwoPoolTrainer, optimize_two_pool

__all__ = [
    "ChunkwiseSpec",
    "PoolSpec",
    "ResolvedLayout",
    "ThreePoolTopology",
    "ThreePoolTrainer",
    "TwoPoolResolvedLayout",
    "TwoPoolTopology",
    "TwoPoolTrainer",
    "optimize_three_pool",
    "optimize_two_pool",
]
