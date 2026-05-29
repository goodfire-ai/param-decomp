"""World — the 3-pool topology data model + the cross-pool process groups.

`World` is purely declarative — identical content on every rank, no per-rank
fields. Built once at startup after `dist.init_process_group`. It owns all the
batch-split routing math (which CI slice owns which downstream shard, sub-slices
within a CI batch tensor) and every process group.

The per-rank view (which pool am I, what do I own) lives in ``role.py``
(``CIRole | LWRole | PPGDRole``). The cross-pool exchanges live in
``portals.py`` (one typed object per DAG edge). A pool's ``PoolContext``
(``context.py``) bundles ``world`` + ``role`` + that pool's portals.

The defining wrinkle is **3-way batch slicing**: CI/LW/PPGD each shard the
global batch on their own axis. The constraint (enforced in
``ThreePoolConfig.validate_topology``) is cross-divisibility per edge:

    N_ci | N_per_block_layerwise   OR   N_per_block_layerwise | N_ci
    N_ci | N_ppgd                  OR   N_ppgd | N_ci

so each CI↔LW / CI↔PPGD overlap is a whole, aligned sub-slice — a one-to-K
fan-out (and K-to-one reduction) in EITHER direction. The geometry for one edge
lives in ``BatchEdge``, which answers every routing question symmetrically for
both fan directions (CI coarse vs CI fine). See "Batch-split routing" below.
"""

# pyright: reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from param_decomp._trace import trace

# ──────────────────────────────────────────────────────────────────────────────
# NCCL-op event timing (PD_NCCL_EVENT_TIMING=1).
#
# Differentiate "peer not ready, CPU spinning in wait()" (large CPU wall, small
# GPU stream-time) from "actual wire transfer" (large GPU stream-time). NCCL ops
# enqueue on an internal NCCL stream and stream-sync with the current stream;
# events recorded on the *current* default stream around a NCCL op capture
# enqueue → matching downstream-sync, which is a lower bound for the transfer.
# Good enough to tell "wait for peer" apart from "wait for the wire".
#
# Records are buffered and flushed via ``flush_nccl_event_timings()`` once per
# step (called from optimize.py at the same point ``_maybe_emit_ci_fn_bwd_breakdown``
# fires) to avoid synchronizing on the hot path.
# ──────────────────────────────────────────────────────────────────────────────


def _nccl_event_timing_enabled() -> bool:
    return os.environ.get("PD_NCCL_EVENT_TIMING", "").strip() in ("1", "true", "yes")


# (op_name, pre_event, post_event, cpu_ms)
_NCCL_EVENT_BUFFER: list[tuple[str, torch.cuda.Event, torch.cuda.Event, float]] = []


@contextmanager
def time_nccl_op(op_name: str) -> Any:
    """Time one NCCL op: CPU wall + GPU stream-time via CUDA events on the current stream.

    No-op (zero overhead, no events created) when ``PD_NCCL_EVENT_TIMING`` is off.
    """
    if not _nccl_event_timing_enabled():
        yield
        return
    pre = torch.cuda.Event(enable_timing=True)
    post = torch.cuda.Event(enable_timing=True)
    pre.record()
    cpu_t0 = time.perf_counter()
    try:
        yield
    finally:
        cpu_ms = (time.perf_counter() - cpu_t0) * 1000.0
        post.record()
        _NCCL_EVENT_BUFFER.append((op_name, pre, post, cpu_ms))


def flush_nccl_event_timings() -> None:
    """Per-op GPU stream-time, emit one ``trace()`` line per op, clear buffer.

    Per-event ``post.synchronize()`` rather than ``torch.cuda.synchronize()``:
    a full device sync would drain the NCCL streams holding in-flight async
    collectives that other pools haven't yet matched, stalling the whole pool
    set. Sync only on the events we're timing.
    """
    if not _NCCL_EVENT_BUFFER:
        return
    for op_name, pre, post, cpu_ms in _NCCL_EVENT_BUFFER:
        post.synchronize()
        gpu_ms = pre.elapsed_time(post)
        trace(f"nccl-event: {op_name} cpu={cpu_ms:.1f}ms gpu={gpu_ms:.1f}ms")
    _NCCL_EVENT_BUFFER.clear()


@dataclass(frozen=True)
class LayerwiseBlockGroup:
    """One LW block-DDP group: ranks that replicate V/U for `owned_sites`.

    The first rank is the block leader — the canonical actor for cross-pool
    sends. Within a group, in-block all-reduce keeps the replicas in sync.
    Runtime mirror of ``LayerwiseBlockGroupSpec``.
    """

    ranks: tuple[int, ...]
    owned_sites: tuple[str, ...]

    @property
    def leader(self) -> int:
        return self.ranks[0]

    def __post_init__(self) -> None:
        assert len(self.ranks) > 0, "block group must have at least one rank"
        assert len(self.ranks) == len(set(self.ranks)), (
            f"duplicate ranks in block group: {self.ranks}"
        )


@dataclass(frozen=True)
class World:
    """Declarative 3-pool topology — identical content on every rank.

    Three rank-disjoint pools:

      * ``ci_ranks``: replicate the CI fn, DP across batch.
      * ``layerwise_block_groups``: per-block replication of V/U (DDP within
        block) + per-block batch sharding across ranks-within-block.
      * ``ppgd_ranks``: replicate the full V/U, DP across batch.

    Process groups are constructed at world-build time and stored here so the
    layout doesn't have to plumb them through call sites.
    """

    world_size: int
    ci_ranks: tuple[int, ...]
    layerwise_block_groups: tuple[LayerwiseBlockGroup, ...]
    ppgd_ranks: tuple[int, ...]
    all_sites: tuple[str, ...]
    batch_global: int

    ci_pool_group: dist.ProcessGroup
    layerwise_pool_group: dist.ProcessGroup
    ppgd_pool_group: dist.ProcessGroup
    block_group_groups: tuple[dist.ProcessGroup, ...]
    # One process group per LW block: {block_leader} ∪ {ppgd_ranks}. Used for
    # leader-rooted broadcasts when shipping updated V/U from LW → PPGD pool.
    cross_pool_bcast_groups: tuple[dist.ProcessGroup, ...]
    # Dedicated world-wide process group carrying every cross-pool point-to-point
    # send/recv. Structurally separate from default_pg so the default communicator
    # only carries barriers — needed because NCCL gets wedged when eval-time
    # barriers share a communicator with un-progressed cross-pool p2p work.
    cross_pool_p2p_group: dist.ProcessGroup

    # ── Sizes ──

    @property
    def n_ci(self) -> int:
        return len(self.ci_ranks)

    @property
    def n_ppgd(self) -> int:
        return len(self.ppgd_ranks)

    @property
    def n_blocks(self) -> int:
        return len(self.layerwise_block_groups)

    @property
    def n_per_block(self) -> int:
        size = len(self.layerwise_block_groups[0].ranks)
        assert all(len(bg.ranks) == size for bg in self.layerwise_block_groups)
        return size

    @property
    def n_layerwise(self) -> int:
        return sum(len(bg.ranks) for bg in self.layerwise_block_groups)

    @property
    def layerwise_ranks(self) -> tuple[int, ...]:
        return tuple(r for bg in self.layerwise_block_groups for r in bg.ranks)

    @property
    def batch_local_ci(self) -> int:
        assert self.batch_global % self.n_ci == 0
        return self.batch_global // self.n_ci

    @property
    def batch_local_lw(self) -> int:
        assert self.batch_global % self.n_per_block == 0
        return self.batch_global // self.n_per_block

    @property
    def batch_local_ppgd(self) -> int:
        assert self.batch_global % self.n_ppgd == 0
        return self.batch_global // self.n_ppgd

    # ── Site routing ──

    def block_idx_of_site(self, site: str) -> int:
        for i, bg in enumerate(self.layerwise_block_groups):
            if site in bg.owned_sites:
                return i
        raise KeyError(site)

    def block_leader_of_site(self, site: str) -> int:
        return self.layerwise_block_groups[self.block_idx_of_site(site)].leader

    # ── Batch-split routing ──
    #
    # Each cross-pool edge's slice geometry is delegated to a ``BatchEdge`` (one
    # per edge), which answers every routing question SYMMETRICALLY for both fan
    # directions (CI coarse / CI fine). The portals call these edges so the
    # exchange methods never branch on which arity is larger.

    @property
    def ci_lw_edge(self) -> "BatchEdge":
        return BatchEdge(n_ci=self.n_ci, n_down=self.n_per_block, batch_global=self.batch_global)

    @property
    def ci_ppgd_edge(self) -> "BatchEdge":
        return BatchEdge(n_ci=self.n_ci, n_down=self.n_ppgd, batch_global=self.batch_global)


@dataclass(frozen=True)
class BatchEdge:
    """Symmetric batch-slice geometry for one cross-pool edge (CI ↔ downstream).

    ``n_down`` is the downstream pool's batch arity (``n_per_block`` for LW,
    ``n_ppgd`` for PPGD). The validator guarantees ``n_ci`` and ``n_down`` are
    cross-divisible, so exactly one of two regimes holds:

      * ``ci_is_coarse`` (``n_ci <= n_down``, ``n_ci | n_down``): each CI slice
        contains ``fanout = n_down // n_ci`` whole downstream slices. CI fans a
        sub-slice out to each; grads come back to be stitched. One downstream
        rank pairs with exactly one CI rank.
      * not ``ci_is_coarse`` (``n_ci > n_down``, ``n_down | n_ci``): each
        downstream slice contains ``fanout = n_ci // n_down`` whole CI slices.
        One downstream rank gathers from ``fanout`` CI ranks (concat) and
        scatters grads back to those same ``fanout`` CI ranks. One CI rank pairs
        with exactly one downstream rank.

    All slice arithmetic is in units of the FINER pool's shard, so every overlap
    is a whole, aligned sub-slice.
    """

    n_ci: int
    n_down: int
    batch_global: int

    def __post_init__(self) -> None:
        assert self.n_down % self.n_ci == 0 or self.n_ci % self.n_down == 0, (
            f"n_ci ({self.n_ci}) and n_down ({self.n_down}) must be cross-divisible"
        )
        assert self.batch_global % self.n_ci == 0 and self.batch_global % self.n_down == 0

    @property
    def ci_is_coarse(self) -> bool:
        return self.n_ci <= self.n_down

    @property
    def fanout(self) -> int:
        """Number of finer-pool slices nested in one coarser-pool slice."""
        return self.n_down // self.n_ci if self.ci_is_coarse else self.n_ci // self.n_down

    @property
    def b_ci(self) -> int:
        return self.batch_global // self.n_ci

    @property
    def b_down(self) -> int:
        return self.batch_global // self.n_down

    # ── Pairing: which ranks talk to which ──

    def ci_slices_for_down_slice(self, down_slice_idx: int) -> tuple[int, ...]:
        """CI slice idxs whose batches overlap downstream slice `down_slice_idx`.

        Coarse-CI regime: exactly one CI slice (the one containing it).
        Fine-CI regime: the `fanout` CI slices nested in it.
        """
        if self.ci_is_coarse:
            return (down_slice_idx // self.fanout,)
        return tuple(range(down_slice_idx * self.fanout, (down_slice_idx + 1) * self.fanout))

    def down_slices_for_ci_slice(self, ci_slice_idx: int) -> tuple[int, ...]:
        """Downstream slice idxs whose batches overlap CI slice `ci_slice_idx`.

        Coarse-CI regime: the `fanout` downstream slices nested in it.
        Fine-CI regime: exactly one downstream slice (the one containing it).
        """
        if self.ci_is_coarse:
            return tuple(range(ci_slice_idx * self.fanout, (ci_slice_idx + 1) * self.fanout))
        return (ci_slice_idx // self.fanout,)

    # ── Sub-slices: the overlap, expressed in each rank's local batch tensor ──

    def overlap_within_ci(self, ci_slice_idx: int, down_slice_idx: int) -> slice:
        """The CI↔downstream overlap as a sub-slice of CI rank `ci_slice_idx`'s
        local ``[B_ci, ...]`` tensor. Spans the FINER pool's shard."""
        return self._overlap(ci_slice_idx, down_slice_idx, base_is_ci=True)

    def overlap_within_down(self, ci_slice_idx: int, down_slice_idx: int) -> slice:
        """The CI↔downstream overlap as a sub-slice of downstream rank
        `down_slice_idx`'s local ``[B_down, ...]`` tensor."""
        return self._overlap(ci_slice_idx, down_slice_idx, base_is_ci=False)

    def _overlap(self, ci_slice_idx: int, down_slice_idx: int, *, base_is_ci: bool) -> slice:
        ci_global_start = ci_slice_idx * self.b_ci
        down_global_start = down_slice_idx * self.b_down
        overlap_global_start = max(ci_global_start, down_global_start)
        overlap_len = min(self.b_ci, self.b_down)
        base_start = ci_global_start if base_is_ci else down_global_start
        local_start = overlap_global_start - base_start
        assert local_start >= 0, f"non-overlapping slices: ci={ci_slice_idx} down={down_slice_idx}"
        return slice(local_start, local_start + overlap_len)


def build_world(
    ci_ranks: list[int],
    layerwise_block_groups: list[LayerwiseBlockGroup],
    ppgd_ranks: list[int],
    batch_global: int,
    pg_timeout: timedelta,
    device: torch.device | None = None,
) -> World:
    """Construct the World + all process groups. Must be called on every rank
    after ``dist.init_process_group``.

    ``pg_timeout`` is the collective timeout for every subgroup created here.
    ``dist.new_group`` does NOT inherit the timeout passed to
    ``init_process_group`` — with ``timeout=None`` it falls back to the NCCL
    library default of 10 minutes regardless of how the default group was
    configured. The 3-pool runs all of its real collectives (the cross-pool
    p2p group, the per-pool all-reduces, the bcast groups) on these subgroups,
    not the default group, so the timeout MUST be threaded explicitly or a slow
    checkpoint save trips the 10-min watchdog and aborts the job.

    Pass ``device`` (this rank's GPU) so we can pre-warm the cross-pool NCCL
    broadcast groups before they're first used inside the training loop. See
    ``_prewarm_cross_pool_bcast_groups`` for why pre-warming is needed.
    """
    world_size = dist.get_world_size()
    layerwise_ranks = [r for bg in layerwise_block_groups for r in bg.ranks]
    assert len(ci_ranks) + len(layerwise_ranks) + len(ppgd_ranks) == world_size, (
        f"rank count mismatch: ci={len(ci_ranks)} + lw={len(layerwise_ranks)} + "
        f"ppgd={len(ppgd_ranks)} != world_size={world_size}"
    )
    assert set(ci_ranks).isdisjoint(set(layerwise_ranks))
    assert set(ci_ranks).isdisjoint(set(ppgd_ranks))
    assert set(layerwise_ranks).isdisjoint(set(ppgd_ranks))
    assert len(set(layerwise_ranks)) == len(layerwise_ranks)

    all_sites = tuple(s for bg in layerwise_block_groups for s in bg.owned_sites)
    assert len(set(all_sites)) == len(all_sites), "a site is owned by more than one block group"

    my_rank = dist.get_rank()

    def _make_group(name: str, ranks: list[int]) -> Any:
        # Per-rank trace before/after each collective new_group. Lets us
        # localize precisely where the world wedges if NCCL deadlocks.
        print(f"[build_world rank={my_rank}] before {name} ranks={ranks}", flush=True)
        g = dist.new_group(ranks=ranks, timeout=pg_timeout)
        print(f"[build_world rank={my_rank}] after  {name} ranks={ranks}", flush=True)
        return g

    ci_pool_group = _make_group("ci_pool_group", ci_ranks)
    layerwise_pool_group = _make_group("layerwise_pool_group", layerwise_ranks)
    ppgd_pool_group = _make_group("ppgd_pool_group", ppgd_ranks)
    block_group_groups = tuple(
        _make_group(f"block_group_groups[{i}]", list(bg.ranks))
        for i, bg in enumerate(layerwise_block_groups)
    )
    cross_pool_bcast_groups = tuple(
        _make_group(f"cross_pool_bcast_groups[{i}]", [bg.leader, *ppgd_ranks])
        for i, bg in enumerate(layerwise_block_groups)
    )
    cross_pool_p2p_group = _make_group("cross_pool_p2p_group", list(range(world_size)))

    if device is not None:
        _prewarm_cross_pool_bcast_groups(
            cross_pool_bcast_groups=cross_pool_bcast_groups,
            layerwise_block_groups=layerwise_block_groups,
            ppgd_ranks=ppgd_ranks,
            my_rank=my_rank,
            device=device,
        )

    return World(
        world_size=world_size,
        ci_ranks=tuple(ci_ranks),
        layerwise_block_groups=tuple(layerwise_block_groups),
        ppgd_ranks=tuple(ppgd_ranks),
        all_sites=all_sites,
        batch_global=batch_global,
        ci_pool_group=ci_pool_group,
        layerwise_pool_group=layerwise_pool_group,
        ppgd_pool_group=ppgd_pool_group,
        block_group_groups=block_group_groups,
        cross_pool_bcast_groups=cross_pool_bcast_groups,
        cross_pool_p2p_group=cross_pool_p2p_group,
    )


def _prewarm_cross_pool_bcast_groups(
    *,
    cross_pool_bcast_groups: tuple[Any, ...],
    layerwise_block_groups: list[LayerwiseBlockGroup],
    ppgd_ranks: list[int],
    my_rank: int,
    device: torch.device,
) -> None:
    """Trigger NCCL communicator init on each cross-pool bcast group.

    First use of a new NCCL process group blocks on a synchronous global
    communicator init across all participating ranks — even when the caller
    passes ``async_op=True``. In the training loop the V/U-from-LW broadcasts
    (PPGD recv + LW send) are the first users of these groups, and on log steps
    they can interleave with ``_log_train_metrics`` (a separate LW↔PPGD
    point-to-point) such that the blocking communicator init deadlocks: each
    pool waits for the other to call into NCCL.

    Pre-warming each group here with a 1-element dummy broadcast does the
    NCCL init once at setup time, while every participant is still in
    lockstep at ``build_world``. After this, the first real broadcast in the
    training loop is a normal communicator op that can interleave with other
    work.
    """
    dummy = torch.zeros(1, device=device)
    ppgd_set = set(ppgd_ranks)
    for bg, group in zip(layerwise_block_groups, cross_pool_bcast_groups, strict=True):
        if my_rank == bg.leader or my_rank in ppgd_set:
            dist.broadcast(dummy, src=bg.leader, group=group)
