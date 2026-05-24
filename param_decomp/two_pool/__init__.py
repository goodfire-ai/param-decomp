"""2-pool training strategy for SPD on large frozen target models.

Splits GPUs into two heterogeneous pools that run in wall-clock parallel:

  Pool A (home)        owns canonical state — components V/U + per-module CI fn +
                       optimizer state, sharded by site. Runs target+CI forward,
                       per-site layerwise loss, faithfulness, importance-minimality.
  Pool B (scratchpad)  stateless — holds a transient replica of all V/U for the
                       full-model PPGD forward. Returns per-site V/U grads and
                       gradients w.r.t. the CI values it received.

``optimize_two_pool`` mirrors :func:`param_decomp.optimize.optimize`'s call
shape. ``TwoPoolConfig`` declares the topology (block groups + pool-B ranks).
"""

from param_decomp.two_pool.config import BlockGroupSpec, TwoPoolConfig
from param_decomp.two_pool.optimize import optimize_two_pool
from param_decomp.two_pool.profiler import PhaseProfiler

__all__ = [
    "BlockGroupSpec",
    "PhaseProfiler",
    "TwoPoolConfig",
    "optimize_two_pool",
]
