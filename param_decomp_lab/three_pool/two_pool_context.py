"""Per-pool context under 2-pool — ``TwoPoolContext = PoolAContext | ChunkContext``.

Pool A merges the 3-pool CI and PPGD pools, so its portal bundle is the union of
the edges those two pools had at the Pool A ↔ chunk boundary — and ONLY those: the
CI↔PPGD edges of the 3-pool are gone (the adversary's g_CI is the local ``.grad`` of
the CI forward's ``lower_leaky``). Pool A reaches for:

  * ``ci_to_chunk`` (``CiValuesToChunkwise``) — masks A → chunk.
  * ``g_ci_from_chunk`` (``GradCiFromChunkwise``) — chunkwise's g_CI contribution back.
  * ``g_vu_to_chunk`` (``GradVuFromPPGD``) — adversary V/U grads A → chunk leaders.
  * ``updated_vu_from_chunk`` (``UpdatedVuToPPGD``) — fresh V/U chunk → A's replica.

The chunkwise pool reuses the 3-pool ``ChunkContext`` / ``ChunkPortals`` verbatim:
with ``ci_ranks == ppgd_ranks == pool_a_ranks`` (see ``two_pool_layout.py``) all four
of its cross-pool edges land on Pool A.
"""

from dataclasses import dataclass

from param_decomp_lab.three_pool.context import ChunkContext
from param_decomp_lab.three_pool.layout import World
from param_decomp_lab.three_pool.portals import (
    CiValuesToChunkwise,
    GradCiFromChunkwise,
    GradVuFromPPGD,
    UpdatedVuToPPGD,
    build_chunk_portals,
)
from param_decomp_lab.three_pool.role import ChunkRole
from param_decomp_lab.three_pool.two_pool_role import PoolARole, two_pool_role_for_rank


@dataclass(frozen=True)
class PoolAPortals:
    role: PoolARole
    ci_to_chunk: CiValuesToChunkwise
    g_ci_from_chunk: GradCiFromChunkwise
    g_vu_to_chunk: GradVuFromPPGD
    updated_vu_from_chunk: UpdatedVuToPPGD


def build_pool_a_portals(world: World, role: PoolARole) -> PoolAPortals:
    return PoolAPortals(
        role=role,
        ci_to_chunk=CiValuesToChunkwise(world),
        g_ci_from_chunk=GradCiFromChunkwise(world),
        g_vu_to_chunk=GradVuFromPPGD(world),
        updated_vu_from_chunk=UpdatedVuToPPGD(world),
    )


@dataclass(frozen=True)
class PoolAContext:
    kind = "pool_a"
    world: World
    role: PoolARole
    portals: PoolAPortals


TwoPoolContext = PoolAContext | ChunkContext


def build_two_pool_context(world: World, rank: int) -> TwoPoolContext:
    role = two_pool_role_for_rank(world, rank)
    match role:
        case PoolARole():
            return PoolAContext(world=world, role=role, portals=build_pool_a_portals(world, role))
        case ChunkRole():
            return ChunkContext(world=world, role=role, portals=build_chunk_portals(world, role))
