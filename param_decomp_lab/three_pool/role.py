"""This rank's pool role — a discriminated union over the three pools.

``PoolRole = CIRole | LWRole | PPGDRole``. Each variant carries ONLY the
per-rank fields that are meaningful for that pool, so accessing a field that
doesn't exist for this rank's pool is a type error, not a runtime ``None``.
This replaces the old ``ThreePoolLayout`` optional-attr bag (``my_ci_slice_idx:
int | None`` etc.) + ``assert self.my_pool == "..."`` guards: the union makes
"this field is only valid on the CI pool" a property of the type system.

``World.role_for_rank`` (in ``layout.py``) is the single constructor. Callers
``match`` on the role to get exhaustive, pool-specific code paths.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from param_decomp_lab.three_pool.layout import World


@dataclass(frozen=True)
class CIRole:
    """A CI-pool rank: replicates the CI fn, owns one DP shard of the batch."""

    kind: Literal["ci"]
    rank: int
    slice_idx: int
    is_pool_leader: bool

    def batch_slice(self, batch_local_ci: int) -> slice:
        return slice(self.slice_idx * batch_local_ci, (self.slice_idx + 1) * batch_local_ci)


@dataclass(frozen=True)
class LWRole:
    """A Layerwise-pool rank: owns V/U for ``owned_sites``, in a block-DDP group."""

    kind: Literal["layerwise"]
    rank: int
    block_idx: int
    within_block_idx: int
    is_block_leader: bool
    owned_sites: tuple[str, ...]

    def batch_slice(self, batch_local_lw: int) -> slice:
        return slice(
            self.within_block_idx * batch_local_lw,
            (self.within_block_idx + 1) * batch_local_lw,
        )


@dataclass(frozen=True)
class PPGDRole:
    """A PPGD-pool rank: full V/U replica + persistent sources, one DP shard."""

    kind: Literal["ppgd"]
    rank: int
    slice_idx: int
    is_pool_leader: bool

    def batch_slice(self, batch_local_ppgd: int) -> slice:
        return slice(self.slice_idx * batch_local_ppgd, (self.slice_idx + 1) * batch_local_ppgd)


PoolRole = CIRole | LWRole | PPGDRole


def role_for_rank(world: "World", rank: int) -> PoolRole:
    if rank in world.ci_ranks:
        slice_idx = world.ci_ranks.index(rank)
        return CIRole(
            kind="ci",
            rank=rank,
            slice_idx=slice_idx,
            is_pool_leader=(rank == world.ci_ranks[0]),
        )
    for block_idx, bg in enumerate(world.layerwise_block_groups):
        if rank in bg.ranks:
            within = bg.ranks.index(rank)
            return LWRole(
                kind="layerwise",
                rank=rank,
                block_idx=block_idx,
                within_block_idx=within,
                is_block_leader=(within == 0),
                owned_sites=bg.owned_sites,
            )
    if rank in world.ppgd_ranks:
        slice_idx = world.ppgd_ranks.index(rank)
        return PPGDRole(
            kind="ppgd",
            rank=rank,
            slice_idx=slice_idx,
            is_pool_leader=(rank == world.ppgd_ranks[0]),
        )
    raise ValueError(f"rank {rank} not in any pool")
