"""World — the 3-pool topology data model + the cross-pool process groups.

`World` is purely declarative — identical content on every rank, no per-rank
fields. Built once at startup after `dist.init_process_group`. It owns all the
batch-split routing math (which CI slice owns which downstream shard, sub-slices
within a CI batch tensor) and every process group.

The per-rank view (which pool am I, what do I own) lives in ``role.py``
(``CIRole | ChunkRole | PPGDRole``). The cross-pool exchanges live in
``portals.py`` (one typed object per DAG edge). A pool's ``PoolContext``
(``context.py``) bundles ``world`` + ``role`` + that pool's portals.

The defining wrinkle is **3-way batch slicing**: CI/chunkwise/PPGD each shard the
global batch on their own axis. The constraint (enforced in
``ThreePoolTopology``) is cross-divisibility per edge:

    N_ci | chunk_dp   OR   chunk_dp | N_ci
    N_ci | N_ppgd     OR   N_ppgd | N_ci

so each CI↔chunk / CI↔PPGD overlap is a whole, aligned sub-slice — a one-to-K
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
class Chunk:
    """One chunkwise DDP group: ranks that replicate V/U for ``sites``.

    The first rank is the chunk leader — the canonical actor for cross-pool
    sends. Within a chunk, in-chunk all-reduce keeps the replicas in sync.
    """

    ranks: tuple[int, ...]
    sites: tuple[str, ...]

    @property
    def leader(self) -> int:
        return self.ranks[0]

    def __post_init__(self) -> None:
        assert len(self.ranks) > 0, "chunk must have at least one rank"
        assert len(self.ranks) == len(set(self.ranks)), f"duplicate ranks in chunk: {self.ranks}"


@dataclass(frozen=True)
class World:
    """Declarative 3-pool topology — identical content on every rank.

    Three rank-disjoint pools:

      * ``ci_ranks``: replicate the CI fn, DP across batch.
      * ``chunks``: per-chunk replication of V/U (DDP within chunk) + per-chunk
        batch sharding across ranks-within-chunk.
      * ``ppgd_ranks``: replicate the full V/U, DP across batch.

    Process groups are constructed at world-build time and stored here so the
    layout doesn't have to plumb them through call sites.
    """

    world_size: int
    ci_ranks: tuple[int, ...]
    chunks: tuple[Chunk, ...]
    ppgd_ranks: tuple[int, ...]
    all_sites: tuple[str, ...]
    batch_global: int

    ci_pool_group: dist.ProcessGroup
    chunkwise_pool_group: dist.ProcessGroup
    ppgd_pool_group: dist.ProcessGroup
    chunk_groups: tuple[dist.ProcessGroup, ...]
    # One process group per chunk: {chunk_leader} ∪ {ppgd_ranks}. Used for
    # leader-rooted broadcasts when shipping updated V/U from chunkwise → PPGD pool.
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
    def n_chunks(self) -> int:
        return len(self.chunks)

    @property
    def chunk_dp(self) -> int:
        size = len(self.chunks[0].ranks)
        assert all(len(c.ranks) == size for c in self.chunks)
        return size

    @property
    def n_chunkwise(self) -> int:
        return sum(len(c.ranks) for c in self.chunks)

    @property
    def chunkwise_ranks(self) -> tuple[int, ...]:
        return tuple(r for c in self.chunks for r in c.ranks)

    @property
    def batch_local_ci(self) -> int:
        assert self.batch_global % self.n_ci == 0
        return self.batch_global // self.n_ci

    @property
    def batch_local_chunk(self) -> int:
        assert self.batch_global % self.chunk_dp == 0
        return self.batch_global // self.chunk_dp

    @property
    def batch_local_ppgd(self) -> int:
        assert self.batch_global % self.n_ppgd == 0
        return self.batch_global // self.n_ppgd

    # ── Site routing ──

    def chunk_idx_of_site(self, site: str) -> int:
        for i, c in enumerate(self.chunks):
            if site in c.sites:
                return i
        raise KeyError(site)

    def chunk_leader_of_site(self, site: str) -> int:
        return self.chunks[self.chunk_idx_of_site(site)].leader

    # ── Batch-split routing ──
    #
    # Each cross-pool edge's slice geometry is delegated to a ``BatchEdge`` (one
    # per edge), which answers every routing question SYMMETRICALLY for both fan
    # directions (CI coarse / CI fine). The portals call these edges so the
    # exchange methods never branch on which arity is larger.

    @property
    def ci_chunk_edge(self) -> "BatchEdge":
        return BatchEdge(n_ci=self.n_ci, n_down=self.chunk_dp, batch_global=self.batch_global)

    @property
    def ci_ppgd_edge(self) -> "BatchEdge":
        return BatchEdge(n_ci=self.n_ci, n_down=self.n_ppgd, batch_global=self.batch_global)


@dataclass(frozen=True)
class BatchEdge:
    """Symmetric batch-slice geometry for one cross-pool edge (CI ↔ downstream).

    ``n_down`` is the downstream pool's batch arity (``chunk_dp`` for chunkwise,
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
    chunks: list[Chunk],
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
    chunkwise_ranks = [r for c in chunks for r in c.ranks]
    assert len(ci_ranks) + len(chunkwise_ranks) + len(ppgd_ranks) == world_size, (
        f"rank count mismatch: ci={len(ci_ranks)} + chunkwise={len(chunkwise_ranks)} + "
        f"ppgd={len(ppgd_ranks)} != world_size={world_size}"
    )
    assert set(ci_ranks).isdisjoint(set(chunkwise_ranks))
    assert set(ci_ranks).isdisjoint(set(ppgd_ranks))
    assert set(chunkwise_ranks).isdisjoint(set(ppgd_ranks))
    assert len(set(chunkwise_ranks)) == len(chunkwise_ranks)

    all_sites = tuple(s for c in chunks for s in c.sites)
    assert len(set(all_sites)) == len(all_sites), "a site is owned by more than one chunk"

    my_rank = dist.get_rank()

    def _make_group(ranks: list[int]) -> Any:
        return dist.new_group(ranks=ranks, timeout=pg_timeout)

    ci_pool_group = _make_group(ci_ranks)
    chunkwise_pool_group = _make_group(chunkwise_ranks)
    ppgd_pool_group = _make_group(ppgd_ranks)
    chunk_groups = tuple(_make_group(list(c.ranks)) for c in chunks)
    cross_pool_bcast_groups = tuple(_make_group([c.leader, *ppgd_ranks]) for c in chunks)
    cross_pool_p2p_group = _make_group(list(range(world_size)))

    if device is not None:
        _prewarm_cross_pool_bcast_groups(
            cross_pool_bcast_groups=cross_pool_bcast_groups,
            chunks=chunks,
            ppgd_ranks=ppgd_ranks,
            my_rank=my_rank,
            device=device,
        )

    return World(
        world_size=world_size,
        ci_ranks=tuple(ci_ranks),
        chunks=tuple(chunks),
        ppgd_ranks=tuple(ppgd_ranks),
        all_sites=all_sites,
        batch_global=batch_global,
        ci_pool_group=ci_pool_group,
        chunkwise_pool_group=chunkwise_pool_group,
        ppgd_pool_group=ppgd_pool_group,
        chunk_groups=chunk_groups,
        cross_pool_bcast_groups=cross_pool_bcast_groups,
        cross_pool_p2p_group=cross_pool_p2p_group,
    )


def _prewarm_cross_pool_bcast_groups(
    *,
    cross_pool_bcast_groups: tuple[Any, ...],
    chunks: list[Chunk],
    ppgd_ranks: list[int],
    my_rank: int,
    device: torch.device,
) -> None:
    """Trigger NCCL communicator init on each cross-pool bcast group.

    First use of a new NCCL process group blocks on a synchronous global
    communicator init across all participating ranks — even when the caller
    passes ``async_op=True``. In the training loop the V/U-from-chunkwise broadcasts
    (PPGD recv + chunkwise send) are the first users of these groups, and on log steps
    they can interleave with ``_log_train_metrics`` (a separate chunkwise↔PPGD
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
    for c, group in zip(chunks, cross_pool_bcast_groups, strict=True):
        if my_rank == c.leader or my_rank in ppgd_set:
            dist.broadcast(dummy, src=c.leader, group=group)
