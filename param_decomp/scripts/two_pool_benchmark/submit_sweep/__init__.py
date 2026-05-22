"""YAML-driven 2-pool sweep submitter.

A sweep YAML describes a wave of sweep points (a cartesian grid over named
axes on top of a defaults block). The CLI:

  1. Loads the YAML into :class:`SweepConfig` (pydantic-validated).
  2. Expands ``grid`` into a flat ``list[SweepPoint]``.
  3. For each point, emits ``run.yaml + topology.yaml + job.sbatch`` under
     ``{GEN_ROOT}/<point_name>/``.
  4. Submits each via ``sbatch`` (skip with ``--dry-run``).

Run::

    python -m param_decomp.scripts.two_pool_benchmark.submit_sweep \\
        --config path/to/sweep.yaml [--dry-run]

Topology axes (per :class:`TopologySpec`):

  - ``grouping``: how to partition each transformer block's 7 sites.
      "fused"     : 1 group of 7 sites (all of one layer in one rank-group)
      "attn_mlp"  : 2 groups (q/k/v/o + gate/up/down)
      "per_site"  : 7 groups, 1 site each (one rank-group per matrix)
  - ``ddp``: ranks per group (within-group DDP factor).
  - ``blocks_per_group``: fuse N consecutive layers per group (fused only).
  - ``pool_b``: pool B size (None → auto-pad to align world to 8).
  - ``use_fused_kl``: bypass LM head + use the fused KL kernel.
"""

from param_decomp.scripts.two_pool_benchmark.submit_sweep.schema import (
    CiSpec,
    ModelSpec,
    RuntimeSpec,
    SweepConfig,
    SweepPoint,
    TopologySpec,
)
from param_decomp.scripts.two_pool_benchmark.submit_sweep.submit import (
    point_name,
    submit_point,
)
from param_decomp.scripts.two_pool_benchmark.submit_sweep.topology import (
    render_topology,
    topology_label,
)

__all__ = [
    "CiSpec",
    "ModelSpec",
    "RuntimeSpec",
    "SweepConfig",
    "SweepPoint",
    "TopologySpec",
    "point_name",
    "render_topology",
    "submit_point",
    "topology_label",
]
