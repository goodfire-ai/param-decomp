"""``build_two_world`` — the 2-pool topology + cross-pool process groups.

The 2-pool world reuses the 3-pool :class:`~param_decomp_lab.three_pool.layout.World`
data model directly, with ONE structural identity: Pool A is both the "CI" pool and
the "PPGD" pool, so ``ci_ranks == ppgd_ranks == pool_a_ranks`` and the CI-pool group
IS the PPGD-pool group (one Pool A all-reduce group). With that identity:

  * the chunkwise pool (Pool B) is byte-for-byte the 3-pool chunkwise pool — same
    ``ChunkContext``, same ``step_chunkwise``, same portals (it sends g_CI to / recvs
    masks from "ci_ranks" = Pool A, and recvs V/U grads from / ships V/U to
    "ppgd_ranks" = Pool A);
  * the surviving cross-pool portal classes (``CiValuesToChunkwise`` /
    ``GradCiFromChunkwise`` / ``GradVuFromPPGD`` / ``UpdatedVuToPPGD``) work unchanged,
    keyed on ``ci_chunk_edge`` (now Pool A ↔ chunk) and the V/U bcast groups
    ``{chunk_leader} ∪ pool_a_ranks``;
  * the deleted CI↔PPGD edges (``CiValuesToPPGD`` / ``GradCiFromPPGD``) simply are
    never constructed by the Pool A context — the adversary's g_CI is the LOCAL
    ``.grad`` of the CI forward's own ``lower_leaky`` (same rank, same batch slice).

We construct the ``World`` directly rather than via ``build_world`` because the
latter asserts ``ci_ranks`` and ``ppgd_ranks`` are disjoint — true for 3-pool,
deliberately false here.
"""

from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from param_decomp_lab.three_pool.layout import Chunk, World, _prewarm_cross_pool_bcast_groups


def build_two_world(
    pool_a_ranks: list[int],
    chunks: list[Chunk],
    batch_global: int,
    pg_timeout: timedelta,
    device: torch.device | None = None,
) -> World:
    """Construct the 2-pool ``World`` + all process groups on every rank.

    Returns a ``World`` whose ``ci_ranks`` and ``ppgd_ranks`` are BOTH
    ``pool_a_ranks`` and whose ``ci_pool_group`` and ``ppgd_pool_group`` are the
    single Pool A all-reduce group. See module docstring for why this reuse is
    correct. ``pg_timeout`` is threaded into every subgroup (``new_group`` does
    not inherit ``init_process_group``'s timeout — see ``build_world``).
    """
    world_size = dist.get_world_size()
    chunkwise_ranks = [r for c in chunks for r in c.ranks]
    assert len(pool_a_ranks) + len(chunkwise_ranks) == world_size, (
        f"rank count mismatch: pool_a={len(pool_a_ranks)} + chunkwise={len(chunkwise_ranks)} "
        f"!= world_size={world_size}"
    )
    assert set(pool_a_ranks).isdisjoint(set(chunkwise_ranks))
    assert len(set(chunkwise_ranks)) == len(chunkwise_ranks)

    all_sites = tuple(s for c in chunks for s in c.sites)
    assert len(set(all_sites)) == len(all_sites), "a site is owned by more than one chunk"

    my_rank = dist.get_rank()

    def _make_group(ranks: list[int]) -> Any:
        return dist.new_group(ranks=ranks, timeout=pg_timeout)

    # One Pool A all-reduce group, used as BOTH the CI-pool group (CI-fn grad
    # reduce) and the PPGD-pool group (V/U grad sum-reduce). Same membership →
    # same group object.
    pool_a_group = _make_group(pool_a_ranks)
    chunkwise_pool_group = _make_group(chunkwise_ranks)
    chunk_groups = tuple(_make_group(list(c.ranks)) for c in chunks)
    cross_pool_bcast_groups = tuple(_make_group([c.leader, *pool_a_ranks]) for c in chunks)
    cross_pool_p2p_group = _make_group(list(range(world_size)))

    if device is not None:
        _prewarm_cross_pool_bcast_groups(
            cross_pool_bcast_groups=cross_pool_bcast_groups,
            chunks=chunks,
            ppgd_ranks=pool_a_ranks,
            my_rank=my_rank,
            device=device,
        )
        # Eager-init the whole-world cross-pool P2P communicator in lockstep at setup. The
        # CI-mask + grad exchanges run `batch_isend_irecv` on this group; NCCL's lazy
        # communicator init on first p2p is a BLOCKING whole-world collective, and Pool A
        # reaches its first p2p (the CI-mask send) only AFTER a heavy `_ci_fn_forward`, far
        # later than the chunk ranks reach their irecv — so the lazy init deadlocks at the
        # first step (chunk ranks blocked in the irecv comm-init; Pool A not yet there). A
        # dummy all-reduce on the group forces the init here, while all ranks are in lockstep.
        dist.all_reduce(torch.zeros(1, device=device), group=cross_pool_p2p_group)

    return World(
        world_size=world_size,
        ci_ranks=tuple(pool_a_ranks),
        chunks=tuple(chunks),
        ppgd_ranks=tuple(pool_a_ranks),
        all_sites=all_sites,
        batch_global=batch_global,
        ci_pool_group=pool_a_group,
        chunkwise_pool_group=chunkwise_pool_group,
        ppgd_pool_group=pool_a_group,
        chunk_groups=chunk_groups,
        cross_pool_bcast_groups=cross_pool_bcast_groups,
        cross_pool_p2p_group=cross_pool_p2p_group,
    )
