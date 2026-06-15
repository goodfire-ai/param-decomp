"""This rank's pool role under 2-pool — ``PoolARole | ChunkRole``.

Pool A merges the 3-pool CI and PPGD pools onto one rank set: each Pool A rank
holds the replicated CI fn AND a full V/U replica + adversary, on a single DP
shard of the batch. For the surviving cross-pool edges (which reuse the 3-pool
portal classes) a Pool A rank presents two views of itself: a ``CIRole`` (for the
mask send + g_CI recv on the chunk edge) and a ``PPGDRole`` (for the V/U grad
send + updated-V/U recv on the V/U-sync edge). Both views carry the same
``rank`` / ``slice_idx`` / ``is_pool_leader`` — Pool A's single batch arity plays
the role of both ``n_ci`` and ``n_ppgd``.

``ChunkRole`` is reused unchanged from the 3-pool ``role.py``.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from param_decomp_lab.three_pool.role import ChunkRole, CIRole, PPGDRole

if TYPE_CHECKING:
    from param_decomp_lab.three_pool.layout import World


@dataclass(frozen=True)
class PoolARole:
    """A Pool-A rank: replicated CI fn + full V/U replica + adversary, one DP shard."""

    kind: Literal["pool_a"]
    rank: int
    slice_idx: int
    is_pool_leader: bool

    def batch_slice(self, batch_local_a: int) -> slice:
        return slice(self.slice_idx * batch_local_a, (self.slice_idx + 1) * batch_local_a)

    def as_ci(self) -> CIRole:
        """The CI-side view used by the chunk-edge portals (masks out, g_CI back)."""
        return CIRole(
            kind="ci", rank=self.rank, slice_idx=self.slice_idx, is_pool_leader=self.is_pool_leader
        )

    def as_ppgd(self) -> PPGDRole:
        """The PPGD-side view used by the V/U-sync portals (V/U grads out, V/U in)."""
        return PPGDRole(
            kind="ppgd",
            rank=self.rank,
            slice_idx=self.slice_idx,
            is_pool_leader=self.is_pool_leader,
        )


TwoPoolRole = PoolARole | ChunkRole


def two_pool_role_for_rank(world: "World", rank: int) -> TwoPoolRole:
    # The 2-pool ``World`` stores Pool A as ``ci_ranks`` (== ``ppgd_ranks``); see
    # ``two_pool_layout.build_two_world``.
    pool_a_ranks = world.ci_ranks
    if rank in pool_a_ranks:
        slice_idx = pool_a_ranks.index(rank)
        return PoolARole(
            kind="pool_a",
            rank=rank,
            slice_idx=slice_idx,
            is_pool_leader=(rank == pool_a_ranks[0]),
        )
    for chunk_idx, chunk in enumerate(world.chunks):
        if rank in chunk.ranks:
            within = chunk.ranks.index(rank)
            return ChunkRole(
                kind="chunkwise",
                rank=rank,
                chunk_idx=chunk_idx,
                within_chunk_idx=within,
                is_chunk_leader=(within == 0),
                sites=chunk.sites,
            )
    raise ValueError(f"rank {rank} not in any pool")
