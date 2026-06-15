"""Per-pool context — ``world`` + this rank's ``role`` + this pool's portals.

``PoolContext = CIContext | ChunkContext | PPGDContext``. The trainer builds one
on construction; the per-step loop ``match``es it and dispatches to the pool's
step function, which receives the matching context type. Because each context
only carries the portals for that pool's own DAG endpoints, a step function can
reach for exactly the cross-pool exchanges it's allowed to perform — a CI step
has no handle to ``UpdatedVuToPPGD``, a chunkwise step has no handle to the
CI-pool's ``GradCiFromPPGD``, and so on.
"""

from dataclasses import dataclass

from param_decomp_lab.three_pool.layout import World
from param_decomp_lab.three_pool.portals import (
    ChunkPortals,
    CIPortals,
    PPGDPortals,
    build_chunk_portals,
    build_ci_portals,
    build_ppgd_portals,
)
from param_decomp_lab.three_pool.role import ChunkRole, CIRole, PPGDRole, role_for_rank


@dataclass(frozen=True)
class CIContext:
    kind = "ci"
    world: World
    role: CIRole
    portals: CIPortals


@dataclass(frozen=True)
class ChunkContext:
    kind = "chunkwise"
    world: World
    role: ChunkRole
    portals: ChunkPortals


@dataclass(frozen=True)
class PPGDContext:
    kind = "ppgd"
    world: World
    role: PPGDRole
    portals: PPGDPortals


PoolContext = CIContext | ChunkContext | PPGDContext


def build_pool_context(world: World, rank: int) -> PoolContext:
    role = role_for_rank(world, rank)
    match role:
        case CIRole():
            return CIContext(world=world, role=role, portals=build_ci_portals(world, role))
        case ChunkRole():
            return ChunkContext(world=world, role=role, portals=build_chunk_portals(world, role))
        case PPGDRole():
            return PPGDContext(world=world, role=role, portals=build_ppgd_portals(world, role))
