"""2-pool training architecture for SPD on large frozen target models.

Splits GPUs into two heterogeneous pools that run in wall-clock parallel:

  Pool A (home)        owns canonical state — components V/U + per-module CI fn +
                       optimizer state, sharded by site. Runs target+CI forward,
                       per-site layerwise loss, faithfulness, importance-minimality.
  Pool B (scratchpad)  stateless — holds a transient replica of all V/U for the
                       full-model PPGD forward. Returns per-site V/U grads and
                       gradients w.r.t. the CI values it received.

See `param_decomp/two_pool/layout.py` for the topology data model and the
cross-pool comm primitives, and `param_decomp/scripts/two_pool_benchmark/`
for runnable training scripts.

Origin: prototyped in `nano_param_decomp/two_pool/` and verified across single-node
and multi-node profiles at scales from 880K to 1B target / 10B CI fn before being
brought into the core codebase.
"""

from param_decomp.two_pool.install import (
    build_pool_a_module_path_info,
    build_pool_b_module_path_info,
)
from param_decomp.two_pool.layout import (
    BlockDDPLayout,
    BlockDDPWorld,
    BlockGroup,
    TwoPoolLayout,
    World,
    build_block_ddp_world,
    build_world,
)
from param_decomp.two_pool.run import (
    PhaseProfiler,
    TwoPoolConfig,
    optimize_two_pool,
    step_pool_a,
    step_pool_b,
)

__all__ = [
    "BlockDDPLayout",
    "BlockDDPWorld",
    "BlockGroup",
    "PhaseProfiler",
    "TwoPoolConfig",
    "TwoPoolLayout",
    "World",
    "build_block_ddp_world",
    "build_pool_a_module_path_info",
    "build_pool_b_module_path_info",
    "build_world",
    "optimize_two_pool",
    "step_pool_a",
    "step_pool_b",
]
