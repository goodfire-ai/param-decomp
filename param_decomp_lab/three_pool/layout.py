"""World / ThreePoolLayout — the 3-pool topology data model and cross-pool comms.

`World` is purely declarative — identical content on every rank, no per-rank
fields. Built once at startup after `dist.init_process_group`.

`ThreePoolLayout` wraps a World, adds this rank's perspective (`my_pool`,
`my_owned_sites`, `my_within_block_idx` or `my_ci_slice_idx` or
`my_ppgd_slice_idx`), and hangs the cross-pool comm orchestration methods off
itself.

Cross-pool exchanges (six total — see ``DESIGN.md`` for the per-step graph):

  CI  → LW   : CI_T per-site (owned + LW-rank batch slice)
  CI  → PPGD : CI_T full-model (per-PPGD-rank batch slice)
  LW  → CI   : g_CI_LW per owned site (per-LW-rank batch slice)
  PPGD→ CI   : g_CI_PPGD full-model (per-PPGD-rank batch slice)
  PPGD→ LW   : g_VU_PPGD per-owned-site (after in-pool sum-reduce; PPGD-leader-driven)
  LW  → PPGD : updated V/U per-owned-site (LW-block-leader-driven, broadcast to PPGD pool)

Plus three collective reductions:

  LW  : in-block all-reduce on V/U + faithfulness grads (one per LW block group)
  CI  : in-pool all-reduce on CI fn grads (one collective over the CI pool)
  PPGD: in-pool sum-reduce on V/U grads (one per site, over the PPGD pool)

The defining wrinkle is **3-way batch slicing**: CI/LW/PPGD each shard
the global batch on their own axis. The constraint (enforced in
``ThreePoolConfig.validate_topology``) is that each cross-pool edge's two
arities are cross-divisible — one divides the other:

    N_ci | N_per_block_layerwise   OR   N_per_block_layerwise | N_ci
    N_ci | N_ppgd                  OR   N_ppgd | N_ci

so each CI↔LW / CI↔PPGD edge is a clean one-to-K fan-out in WHICHEVER
direction the smaller arity owns the coarser slice. When CI is coarser
(``N_ci`` smaller) one CI rank fans a sub-slice out to K downstream ranks and
stitches K grads back. When CI is finer (``N_ci`` larger) one downstream rank
gathers from K CI ranks (concat) and scatters grads back to those K. The
``BatchEdge`` geometry (see "Batch-split routing" below) answers both
directions uniformly so the exchange methods don't branch on regime.
"""

# pyright: reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false

import os
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from param_decomp._trace import trace
from param_decomp.component_model import CIOutputs

# All cross-pool tensors are cast to this dtype on the wire (halves bytes vs fp32).
# Downstream pools run inside bf16 autocast already; CI grads and V/U grads
# accumulating into fp32 .grad upcast back to fp32 on receive — standard bf16
# mixed-precision pattern.
_WIRE_DTYPE: torch.dtype = torch.bfloat16


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
def _time_nccl_op(op_name: str) -> Any:
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
class _CiRecvPacket:
    """One incoming CI-values irecv covering ``row_slice`` of the destination.

    The packed buffer holds, for each site in ``sites`` order, ``overlap_len *
    seq_len * c_s`` ``_WIRE_DTYPE`` elements, where ``overlap_len`` is the length
    of ``row_slice`` along the batch dim.
    """

    packed: torch.Tensor
    work: "dist.Work"
    row_slice: slice


@dataclass(frozen=True)
class PendingCiRecv:
    """One or more coalesced CI-values irecvs, held until ``wait_and_unpack()``.

    Covers the full ``[b_down]`` CI tensor for this downstream rank's owned
    sites. In the CI-coarse regime that's a single packet spanning the whole
    tensor; in the CI-fine regime it's ``fanout`` packets, each filling a
    disjoint ``row_slice`` from a different CI rank. ``wait_and_unpack`` blocks
    on all works, stitches each packet's overlap rows into per-site
    ``[b_down, seq_len, c_s]`` destinations, and returns them.
    """

    packets: tuple[_CiRecvPacket, ...]
    sites: tuple[str, ...]
    site_to_c: dict[str, int]
    b_down: int
    seq_len: int
    device: torch.device

    def wait_and_unpack(self) -> dict[str, torch.Tensor]:
        out = {
            s: torch.empty(
                self.b_down, self.seq_len, self.site_to_c[s], device=self.device, dtype=_WIRE_DTYPE
            )
            for s in self.sites
        }
        for packet in self.packets:
            packet.work.wait()
            overlap_len = packet.row_slice.stop - packet.row_slice.start
            offset = 0
            for s in self.sites:
                c_s = self.site_to_c[s]
                numel = overlap_len * self.seq_len * c_s
                view = packet.packed[offset : offset + numel].view(overlap_len, self.seq_len, c_s)
                out[s][packet.row_slice].copy_(view)
                offset += numel
            assert offset == packet.packed.numel(), (
                f"unpack size mismatch: consumed {offset} of {packet.packed.numel()}"
            )
        return out


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

    # ── Batch-split routing (the cross-pool wrinkle) ──
    #
    # CI/LW/PPGD each shard the global batch on their own axis. The validator
    # constrains each cross-pool edge (CI↔LW, CI↔PPGD) so the two arities are
    # cross-divisible: one slice nests an integer number of the other's slices.
    # ``BatchEdge`` captures one such edge symmetrically — it answers every
    # routing question for both the "CI coarser" and "CI finer" regimes — so the
    # exchange methods don't branch on direction.

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
        local ``[B_ci, ...]`` tensor.

        The overlap spans the FINER pool's shard (``min(b_ci, b_down)`` rows).
        """
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
    device: torch.device | None = None,
) -> World:
    """Construct the World + all process groups. Must be called on every rank
    after ``dist.init_process_group``.

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
        g = dist.new_group(ranks=ranks)
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


@dataclass(frozen=True)
class ThreePoolLayout:
    """This rank's view of the 3-pool world + cross-pool comm methods.

    Single class with ``my_pool`` switch. Each comm method asserts its caller
    pool up-front so misuse is loud.
    """

    world: World
    my_rank: int
    my_pool: Literal["ci", "layerwise", "ppgd"]

    # LW-only fields
    my_block_idx: int | None
    my_within_block_idx: int | None
    my_is_block_leader: bool
    my_owned_sites: tuple[str, ...]

    # CI-only / PPGD-only fields
    my_ci_slice_idx: int | None
    my_ppgd_slice_idx: int | None
    my_is_pool_leader: bool

    @classmethod
    def from_world(cls, world: World, my_rank: int) -> "ThreePoolLayout":
        if my_rank in world.ci_ranks:
            slice_idx = world.ci_ranks.index(my_rank)
            return cls(
                world=world,
                my_rank=my_rank,
                my_pool="ci",
                my_block_idx=None,
                my_within_block_idx=None,
                my_is_block_leader=False,
                my_owned_sites=(),
                my_ci_slice_idx=slice_idx,
                my_ppgd_slice_idx=None,
                my_is_pool_leader=(my_rank == world.ci_ranks[0]),
            )
        for bg_idx, bg in enumerate(world.layerwise_block_groups):
            if my_rank in bg.ranks:
                within = bg.ranks.index(my_rank)
                return cls(
                    world=world,
                    my_rank=my_rank,
                    my_pool="layerwise",
                    my_block_idx=bg_idx,
                    my_within_block_idx=within,
                    my_is_block_leader=(within == 0),
                    my_owned_sites=bg.owned_sites,
                    my_ci_slice_idx=None,
                    my_ppgd_slice_idx=None,
                    my_is_pool_leader=False,
                )
        if my_rank in world.ppgd_ranks:
            slice_idx = world.ppgd_ranks.index(my_rank)
            return cls(
                world=world,
                my_rank=my_rank,
                my_pool="ppgd",
                my_block_idx=None,
                my_within_block_idx=None,
                my_is_block_leader=False,
                my_owned_sites=(),
                my_ci_slice_idx=None,
                my_ppgd_slice_idx=slice_idx,
                my_is_pool_leader=(my_rank == world.ppgd_ranks[0]),
            )
        raise ValueError(f"rank {my_rank} not in any pool")

    # ── Per-rank slice helpers ──

    def my_batch_slice_ci(self) -> slice:
        assert self.my_pool == "ci" and self.my_ci_slice_idx is not None
        b = self.world.batch_local_ci
        return slice(self.my_ci_slice_idx * b, (self.my_ci_slice_idx + 1) * b)

    def my_batch_slice_lw(self) -> slice:
        assert self.my_pool == "layerwise" and self.my_within_block_idx is not None
        b = self.world.batch_local_lw
        return slice(self.my_within_block_idx * b, (self.my_within_block_idx + 1) * b)

    def my_batch_slice_ppgd(self) -> slice:
        assert self.my_pool == "ppgd" and self.my_ppgd_slice_idx is not None
        b = self.world.batch_local_ppgd
        return slice(self.my_ppgd_slice_idx * b, (self.my_ppgd_slice_idx + 1) * b)

    def is_my_site(self, site: str) -> bool:
        return self.my_pool == "layerwise" and site in self.my_owned_sites

    def i_lead_site(self, site: str) -> bool:
        return self.is_my_site(site) and self.my_is_block_leader

    # ──────────────────────────────────────────────────────────────────────
    # CI-pool comm methods
    # ──────────────────────────────────────────────────────────────────────

    def async_send_ci_to_layerwise(
        self, ci_full: dict[str, Tensor]
    ) -> tuple[list["dist.Work"], list[Tensor]]:
        """CI → LW: for each LW rank whose batch shard overlaps my CI slice,
        isend the overlapping sub-slice of each of that LW rank's owned sites.

        ``ci_full`` is keyed by site (CI fn produced CI for ALL sites since the
        CI fn is global). Values have shape ``[B_local_ci, S, C_s]``. Each send
        carries ``overlap_len = min(b_ci, b_lw)`` rows — the whole overlap of
        this CI rank with the target LW rank, in either fan direction.
        Returned buffers must be kept alive until ``work.wait()`` completes.
        """
        assert self.my_pool == "ci" and self.my_ci_slice_idx is not None
        edge = self.world.ci_lw_edge
        my_down_slices = edge.down_slices_for_ci_slice(self.my_ci_slice_idx)
        works: list[dist.Work] = []
        buffers: list[Tensor] = []

        with _time_nccl_op("async_send_ci_to_layerwise"):
            for bg in self.world.layerwise_block_groups:
                for block_rank_idx in my_down_slices:
                    target_lw_rank = bg.ranks[block_rank_idx]
                    sub = edge.overlap_within_ci(self.my_ci_slice_idx, block_rank_idx)
                    # Coalesce all of this block's owned-sites into one packed
                    # send per (block, block_rank). Layout (must match recv):
                    # for each site in bg.owned_sites order, overlap_len * seq_len
                    # * C_s contiguous _WIRE_DTYPE elements.
                    parts = [
                        ci_full[site][sub].detach().to(_WIRE_DTYPE).contiguous().flatten()
                        for site in bg.owned_sites
                    ]
                    packed = torch.cat(parts)
                    works.append(
                        dist.isend(
                            packed, dst=target_lw_rank, group=self.world.cross_pool_p2p_group
                        )
                    )
                    buffers.append(packed)
        return works, buffers

    def async_send_ci_to_ppgd(
        self, ci_full: dict[str, Tensor]
    ) -> tuple[list["dist.Work"], list[Tensor]]:
        """CI → PPGD: for each PPGD rank whose batch shard overlaps my CI slice,
        isend the full-model CI overlap sub-slice (all sites)."""
        assert self.my_pool == "ci" and self.my_ci_slice_idx is not None
        edge = self.world.ci_ppgd_edge
        my_down_slices = edge.down_slices_for_ci_slice(self.my_ci_slice_idx)
        works: list[dist.Work] = []
        buffers: list[Tensor] = []

        with _time_nccl_op("async_send_ci_to_ppgd"):
            for ppgd_slice_idx in my_down_slices:
                target_ppgd_rank = self.world.ppgd_ranks[ppgd_slice_idx]
                sub = edge.overlap_within_ci(self.my_ci_slice_idx, ppgd_slice_idx)
                # Coalesce all sites into one packed send per PPGD target.
                # Layout (must match recv): for each site in self.world.all_sites
                # order, overlap_len * seq_len * C_s contiguous _WIRE_DTYPE elements.
                parts = [
                    ci_full[site][sub].detach().to(_WIRE_DTYPE).contiguous().flatten()
                    for site in self.world.all_sites
                ]
                packed = torch.cat(parts)
                works.append(
                    dist.isend(packed, dst=target_ppgd_rank, group=self.world.cross_pool_p2p_group)
                )
                buffers.append(packed)
        return works, buffers

    def send_ci_eval_to_ppgd(self, ci: CIOutputs) -> None:
        """CI → PPGD eval: synchronous send of full CIOutputs (all three dicts —
        lower_leaky, upper_leaky, pre_sigmoid) sliced to each PPGD rank within
        my CI slice.

        Training-time only ships ``lower_leaky``; eval ships all three so any
        metric reading ``ctx.ci`` works without a per-metric audit. Synchronous
        because eval is rare and overlap has no value here.

        Pack layout per send (must match ``recv_ci_eval_from_ci_pool``): three
        contiguous blocks in order (lower_leaky, upper_leaky, pre_sigmoid). Each
        block has, for each site in ``self.world.all_sites`` order,
        ``overlap_len * seq_len * C_s`` contiguous ``_WIRE_DTYPE`` elements.
        """
        assert self.my_pool == "ci" and self.my_ci_slice_idx is not None
        edge = self.world.ci_ppgd_edge
        my_down_slices = edge.down_slices_for_ci_slice(self.my_ci_slice_idx)

        with _time_nccl_op("send_ci_eval_to_ppgd"):
            for ppgd_slice_idx in my_down_slices:
                target = self.world.ppgd_ranks[ppgd_slice_idx]
                sub = edge.overlap_within_ci(self.my_ci_slice_idx, ppgd_slice_idx)
                parts: list[Tensor] = []
                for d in (ci.lower_leaky, ci.upper_leaky, ci.pre_sigmoid):
                    parts.extend(
                        d[site][sub].detach().to(_WIRE_DTYPE).contiguous().flatten()
                        for site in self.world.all_sites
                    )
                packed = torch.cat(parts)
                dist.send(packed, dst=target, group=self.world.cross_pool_p2p_group)

    def recv_g_ci_from_layerwise(
        self,
        site_to_c: dict[str, int],
        seq_len: int,
        device: torch.device,
    ) -> dict[str, Tensor]:
        """CI ← LW: recv per-site CI grads, coalesced per overlapping LW rank.

        Each LW rank coalesces its owned sites into one packed buffer (see
        ``send_g_ci_to_ci_pool``); we receive one packed buf per overlapping LW
        rank. Each carries ``overlap_len = min(b_ci, b_lw)`` rows — the overlap
        of this CI slice with that LW rank. Pack layout (must match sender): for
        each site in the LW block's owned sites, ``overlap_len * seq_len * c_s``
        contiguous ``_WIRE_DTYPE`` elements.

        When CI is finer than LW (one LW rank spans several CI slices), each LW
        rank still sends this CI rank exactly its own overlap; the recv fills
        the whole ``[b_ci]`` dest. When CI is coarser, several LW ranks each fill
        a disjoint sub-slice of the dest.
        """
        assert self.my_pool == "ci" and self.my_ci_slice_idx is not None
        edge = self.world.ci_lw_edge
        my_down_slices = edge.down_slices_for_ci_slice(self.my_ci_slice_idx)
        overlap_len = min(edge.b_ci, edge.b_down)

        # Post all irecvs upfront so they pipeline on the NIC.
        # Per source: one packed buf containing all of that source's owned sites.
        pending: list[tuple[int, Tensor, dist.Work, tuple[str, ...]]] = []
        with _time_nccl_op("recv_g_ci_from_layerwise:post_irecvs"):
            for bg in self.world.layerwise_block_groups:
                owned = bg.owned_sites
                packed_numel = sum(overlap_len * seq_len * site_to_c[s] for s in owned)
                for block_rank_idx in my_down_slices:
                    src = bg.ranks[block_rank_idx]
                    buf = torch.empty(packed_numel, device=device, dtype=_WIRE_DTYPE)
                    w = dist.irecv(buf, src=src, group=self.world.cross_pool_p2p_group)
                    assert w is not None
                    pending.append((block_rank_idx, buf, w, owned))

        # Wait + stitch. Allocate one fp32 dest per site, copy each piece in place.
        out: dict[str, Tensor] = {}
        b_ci = self.world.batch_local_ci
        for site in self.world.all_sites:
            c_s = site_to_c[site]
            out[site] = torch.empty(b_ci, seq_len, c_s, device=device, dtype=torch.float32)
        with _time_nccl_op("recv_g_ci_from_layerwise:wait"):
            for block_rank_idx, buf, w, owned in pending:
                w.wait()
                sub = edge.overlap_within_ci(self.my_ci_slice_idx, block_rank_idx)
                offset = 0
                for site in owned:
                    c_s = site_to_c[site]
                    n = overlap_len * seq_len * c_s
                    site_view = buf[offset : offset + n].view(overlap_len, seq_len, c_s)
                    out[site][sub].copy_(site_view.to(torch.float32))
                    offset += n
        return out

    def recv_g_ci_from_ppgd(
        self,
        site_to_c: dict[str, int],
        seq_len: int,
        device: torch.device,
    ) -> dict[str, Tensor]:
        """CI ← PPGD: recv full-model CI grads, coalesced.

        One packed irecv per overlapping PPGD source (instead of one per (site,
        source)), matching the coalesced ``send_g_ci_to_ci_pool_ppgd``. Each
        carries ``overlap_len = min(b_ci, b_pp)`` rows — the overlap of this CI
        slice with that PPGD rank.

        Pack layout (must match the sender exactly): for each site in
        ``self.world.all_sites`` order, ``overlap_len * seq_len * c_s``
        contiguous ``_WIRE_DTYPE`` elements.
        """
        assert self.my_pool == "ci" and self.my_ci_slice_idx is not None
        edge = self.world.ci_ppgd_edge
        my_down_slices = edge.down_slices_for_ci_slice(self.my_ci_slice_idx)
        overlap_len = min(edge.b_ci, edge.b_down)

        # Same total numel for every PPGD source (every source sends all sites).
        site_numels = {s: overlap_len * seq_len * site_to_c[s] for s in self.world.all_sites}
        packed_numel = sum(site_numels.values())

        pending: list[tuple[int, Tensor, dist.Work]] = []
        with _time_nccl_op("recv_g_ci_from_ppgd:post_irecvs"):
            for ppgd_slice_idx in my_down_slices:
                src = self.world.ppgd_ranks[ppgd_slice_idx]
                packed = torch.empty(packed_numel, device=device, dtype=_WIRE_DTYPE)
                w = dist.irecv(packed, src=src, group=self.world.cross_pool_p2p_group)
                assert w is not None
                pending.append((ppgd_slice_idx, packed, w))

        b_ci = self.world.batch_local_ci
        out: dict[str, Tensor] = {
            s: torch.empty(b_ci, seq_len, site_to_c[s], device=device, dtype=torch.float32)
            for s in self.world.all_sites
        }
        with _time_nccl_op("recv_g_ci_from_ppgd:wait"):
            for ppgd_slice_idx, packed, w in pending:
                w.wait()
                sub = edge.overlap_within_ci(self.my_ci_slice_idx, ppgd_slice_idx)
                offset = 0
                for site in self.world.all_sites:
                    c_s = site_to_c[site]
                    n = site_numels[site]
                    buf = packed[offset : offset + n].view(overlap_len, seq_len, c_s)
                    out[site][sub].copy_(buf.to(torch.float32))
                    offset += n
        return out

    def all_reduce_ci_fn_grads(self, params: Iterable[nn.Parameter]) -> None:
        """In-pool all-reduce on CI fn grads. Coalesced bucketed reduce —
        same pattern as ``all_reduce_grads_in_block`` but over the CI pool.
        """
        assert self.my_pool == "ci"
        if dist.get_world_size(self.world.ci_pool_group) <= 1:
            return
        grads: list[Tensor] = [p.grad for p in params if p.grad is not None]
        if not grads:
            return
        buckets: dict[tuple[torch.dtype, torch.device], list[Tensor]] = {}
        for g in grads:
            buckets.setdefault((g.dtype, g.device), []).append(g)
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

        with _time_nccl_op("all_reduce_ci_fn_grads"):
            for bucket in buckets.values():
                flat = _flatten_dense_tensors(bucket)
                dist.all_reduce(flat, op=dist.ReduceOp.AVG, group=self.world.ci_pool_group)
                for orig, reduced in zip(
                    bucket, _unflatten_dense_tensors(flat, bucket), strict=True
                ):
                    orig.copy_(reduced)

    # ──────────────────────────────────────────────────────────────────────
    # Layerwise-pool comm methods
    # ──────────────────────────────────────────────────────────────────────

    def async_recv_ci_from_ci_pool(
        self,
        site_to_c: dict[str, int],
        seq_len: int,
        device: torch.device,
    ) -> PendingCiRecv:
        """LW ← CI: irecv CI values for all of this LW rank's owned sites, from
        every CI rank whose slice overlaps my LW batch shard.

        Coarse-CI: one packet from one CI rank, spanning my whole ``[b_lw]``
        slice. Fine-CI: ``fanout`` packets, each from a different CI rank filling
        a disjoint sub-slice. Layout per packet (must match
        ``async_send_ci_to_layerwise``): for each site in ``self.my_owned_sites``
        order, ``overlap_len * seq_len * C_s`` contiguous ``_WIRE_DTYPE``
        elements. Caller calls ``wait_and_unpack()`` to get per-site
        ``[b_lw, seq_len, C_s]`` tensors.
        """
        assert self.my_pool == "layerwise" and self.my_within_block_idx is not None
        edge = self.world.ci_lw_edge
        my_idx = self.my_within_block_idx
        b_lw = self.world.batch_local_lw

        packets: list[_CiRecvPacket] = []
        with _time_nccl_op("async_recv_ci_from_ci_pool"):
            for ci_slice_idx in edge.ci_slices_for_down_slice(my_idx):
                src = self.world.ci_ranks[ci_slice_idx]
                row_slice = edge.overlap_within_down(ci_slice_idx, my_idx)
                overlap_len = row_slice.stop - row_slice.start
                packed_numel = sum(
                    overlap_len * seq_len * site_to_c[s] for s in self.my_owned_sites
                )
                packed = torch.empty(packed_numel, device=device, dtype=_WIRE_DTYPE)
                work = dist.irecv(packed, src=src, group=self.world.cross_pool_p2p_group)
                assert work is not None
                packets.append(_CiRecvPacket(packed=packed, work=work, row_slice=row_slice))
        return PendingCiRecv(
            packets=tuple(packets),
            sites=self.my_owned_sites,
            site_to_c=site_to_c,
            b_down=b_lw,
            seq_len=seq_len,
            device=device,
        )

    def send_g_ci_to_ci_pool(self, g_ci_owned: dict[str, Tensor]) -> None:
        """LW → CI: send per-owned-site CI grads to every CI rank whose slice
        overlaps my LW batch shard.

        Coarse-CI: one packed send (my whole ``[b_lw]`` grad) to one CI rank.
        Fine-CI: one packed send per overlapping CI rank, each carrying that CI
        rank's overlap sub-slice. Coalesces this rank's owned sites into one
        packed send per destination (vs one isend per site).
        """
        assert self.my_pool == "layerwise" and self.my_within_block_idx is not None
        edge = self.world.ci_lw_edge
        my_idx = self.my_within_block_idx
        with _time_nccl_op("send_g_ci_to_ci_pool"):
            for ci_slice_idx in edge.ci_slices_for_down_slice(my_idx):
                dst = self.world.ci_ranks[ci_slice_idx]
                sub = edge.overlap_within_down(ci_slice_idx, my_idx)
                parts = [
                    g_ci_owned[s][sub].detach().to(_WIRE_DTYPE).contiguous().flatten()
                    for s in self.my_owned_sites
                ]
                packed = torch.cat(parts)
                dist.send(packed, dst=dst, group=self.world.cross_pool_p2p_group)

    def recv_g_vu_from_ppgd(
        self,
        v_templates: dict[str, Tensor],
        u_templates: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """LW ← PPGD: leader recvs g_VU for owned sites from PPGD leader, then
        in-block broadcast. PPGD has already sum-reduced within its pool so a
        single recv carries the full-batch grad for our owned sites.
        """
        assert self.my_pool == "layerwise" and self.my_block_idx is not None
        v_grads: dict[str, Tensor] = {}
        u_grads: dict[str, Tensor] = {}

        if self.my_is_block_leader:
            my_sites = self.my_owned_sites
            packed_numel = sum(v_templates[s].numel() + u_templates[s].numel() for s in my_sites)
            sample = v_templates[my_sites[0]]
            packed = torch.empty(packed_numel, dtype=_WIRE_DTYPE, device=sample.device)
            ppgd_leader = self.world.ppgd_ranks[0]
            with _time_nccl_op("recv_g_vu_from_ppgd:recv"):
                dist.recv(packed, src=ppgd_leader, group=self.world.cross_pool_p2p_group)
            offset = 0
            for s in my_sites:
                v_n = v_templates[s].numel()
                u_n = u_templates[s].numel()
                v_grads[s] = (
                    packed[offset : offset + v_n].view_as(v_templates[s]).to(v_templates[s].dtype)
                )
                offset += v_n
                u_grads[s] = (
                    packed[offset : offset + u_n].view_as(u_templates[s]).to(u_templates[s].dtype)
                )
                offset += u_n
        else:
            for s in self.my_owned_sites:
                v_grads[s] = torch.empty_like(v_templates[s])
                u_grads[s] = torch.empty_like(u_templates[s])

        # In-block broadcast leader → other ranks so all replicas see the same g_VU.
        block_group = self.world.block_group_groups[self.my_block_idx]
        block_leader_rank = self.world.layerwise_block_groups[self.my_block_idx].leader
        with _time_nccl_op("recv_g_vu_from_ppgd:in_block_bcast"):
            for s in self.my_owned_sites:
                v_grads[s] = v_grads[s].contiguous()
                u_grads[s] = u_grads[s].contiguous()
                dist.broadcast(v_grads[s], src=block_leader_rank, group=block_group)
                dist.broadcast(u_grads[s], src=block_leader_rank, group=block_group)

        return v_grads, u_grads

    def async_send_updated_vu_to_ppgd(
        self,
        v_owned: dict[str, Tensor],
        u_owned: dict[str, Tensor],
    ) -> tuple[list["dist.Work"], list[Tensor]]:
        """LW → PPGD: coalesced leader-rooted broadcast of updated V/U to all
        PPGD ranks. Caller must keep the buffer alive until the work handle
        completes.
        """
        assert self.my_pool == "layerwise"
        if not self.my_is_block_leader:
            return [], []
        assert self.my_block_idx is not None
        my_sites = self.my_owned_sites
        parts: list[Tensor] = []
        for s in my_sites:
            parts.append(v_owned[s].detach().to(_WIRE_DTYPE).contiguous().flatten())
            parts.append(u_owned[s].detach().to(_WIRE_DTYPE).contiguous().flatten())
        packed = torch.cat(parts)
        bcast_group = self.world.cross_pool_bcast_groups[self.my_block_idx]
        with _time_nccl_op("async_send_updated_vu_to_ppgd"):
            w = dist.broadcast(packed, src=self.my_rank, group=bcast_group, async_op=True)
        assert w is not None
        return [w], [packed]

    def all_reduce_grads_in_block(self, params: Iterable[nn.Parameter]) -> None:
        """Coalesced in-block DDP all-reduce over V/U + faithfulness grads.

        One async all-reduce per (dtype, device) bucket (so the buckets pipeline
        across the NCCL stream), then wait + copy the reduced flat tensors back
        into the original ``.grad`` buffers. No-op when the block group is
        1-rank or there are no grads.
        """
        assert self.my_pool == "layerwise" and self.my_block_idx is not None
        block_group = self.world.block_group_groups[self.my_block_idx]
        if dist.get_world_size(block_group) <= 1:
            return
        grads: list[Tensor] = [p.grad for p in params if p.grad is not None]
        if not grads:
            return
        buckets: dict[tuple[torch.dtype, torch.device], list[Tensor]] = {}
        for g in grads:
            buckets.setdefault((g.dtype, g.device), []).append(g)
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

        states: list[tuple[list[Tensor], Tensor, dist.Work]] = []
        with _time_nccl_op("all_reduce_grads_in_block"):
            for bucket in buckets.values():
                flat = _flatten_dense_tensors(bucket)
                w = dist.all_reduce(flat, op=dist.ReduceOp.AVG, group=block_group, async_op=True)
                assert w is not None
                states.append((bucket, flat, w))
        for bucket, flat, w in states:
            w.wait()
            for orig, reduced in zip(bucket, _unflatten_dense_tensors(flat, bucket), strict=True):
                orig.copy_(reduced)

    # ──────────────────────────────────────────────────────────────────────
    # PPGD-pool comm methods
    # ──────────────────────────────────────────────────────────────────────

    def async_recv_ci_from_ci_pool_ppgd(
        self,
        site_to_c: dict[str, int],
        seq_len: int,
        device: torch.device,
    ) -> PendingCiRecv:
        """PPGD ← CI: irecv full-model CI values from every CI rank whose slice
        overlaps my PPGD batch shard.

        Coarse-CI: one packet from one CI rank, spanning my whole ``[b_pp]``
        slice. Fine-CI: ``fanout`` packets, each from a different CI rank filling
        a disjoint sub-slice. Layout per packet (must match
        ``async_send_ci_to_ppgd``): for each site in ``self.world.all_sites``
        order, ``overlap_len * seq_len * C_s`` contiguous ``_WIRE_DTYPE``
        elements. Caller calls ``wait_and_unpack()`` to get per-site
        ``[b_pp, seq_len, C_s]`` tensors.
        """
        assert self.my_pool == "ppgd" and self.my_ppgd_slice_idx is not None
        edge = self.world.ci_ppgd_edge
        my_idx = self.my_ppgd_slice_idx
        b_pp = self.world.batch_local_ppgd

        packets: list[_CiRecvPacket] = []
        with _time_nccl_op("async_recv_ci_from_ci_pool_ppgd"):
            for ci_slice_idx in edge.ci_slices_for_down_slice(my_idx):
                src = self.world.ci_ranks[ci_slice_idx]
                row_slice = edge.overlap_within_down(ci_slice_idx, my_idx)
                overlap_len = row_slice.stop - row_slice.start
                packed_numel = sum(
                    overlap_len * seq_len * site_to_c[s] for s in self.world.all_sites
                )
                packed = torch.empty(packed_numel, device=device, dtype=_WIRE_DTYPE)
                work = dist.irecv(packed, src=src, group=self.world.cross_pool_p2p_group)
                assert work is not None
                packets.append(_CiRecvPacket(packed=packed, work=work, row_slice=row_slice))
        return PendingCiRecv(
            packets=tuple(packets),
            sites=self.world.all_sites,
            site_to_c=site_to_c,
            b_down=b_pp,
            seq_len=seq_len,
            device=device,
        )

    def recv_ci_eval_from_ci_pool(
        self,
        site_to_c: dict[str, int],
        seq_len: int,
        device: torch.device,
    ) -> CIOutputs:
        """PPGD ← CI eval: synchronous recv of full ``CIOutputs`` from every CI
        rank whose slice overlaps my PPGD batch shard.

        Coarse-CI: one recv from one CI rank, spanning my whole ``[b_pp]`` slice.
        Fine-CI: one recv per overlapping CI rank, each filling a disjoint
        sub-slice. Pack layout per recv (must match ``send_ci_eval_to_ppgd``):
        three contiguous blocks in order (lower_leaky, upper_leaky, pre_sigmoid).
        Each block has, for each site in ``self.world.all_sites`` order,
        ``overlap_len * seq_len * C_s`` contiguous ``_WIRE_DTYPE`` elements.
        """
        assert self.my_pool == "ppgd" and self.my_ppgd_slice_idx is not None
        edge = self.world.ci_ppgd_edge
        my_idx = self.my_ppgd_slice_idx
        b_pp = self.world.batch_local_ppgd

        out: list[dict[str, Tensor]] = [
            {
                s: torch.empty(b_pp, seq_len, site_to_c[s], device=device, dtype=_WIRE_DTYPE)
                for s in self.world.all_sites
            }
            for _ in range(3)
        ]
        with _time_nccl_op("recv_ci_eval_from_ci_pool"):
            for ci_slice_idx in edge.ci_slices_for_down_slice(my_idx):
                src = self.world.ci_ranks[ci_slice_idx]
                row_slice = edge.overlap_within_down(ci_slice_idx, my_idx)
                overlap_len = row_slice.stop - row_slice.start
                per_block_numel = sum(
                    overlap_len * seq_len * site_to_c[s] for s in self.world.all_sites
                )
                packed = torch.empty(3 * per_block_numel, device=device, dtype=_WIRE_DTYPE)
                dist.recv(packed, src=src, group=self.world.cross_pool_p2p_group)
                offset = 0
                for block_idx in range(3):
                    for site in self.world.all_sites:
                        c_s = site_to_c[site]
                        numel = overlap_len * seq_len * c_s
                        view = packed[offset : offset + numel].view(overlap_len, seq_len, c_s)
                        out[block_idx][site][row_slice].copy_(view)
                        offset += numel
                assert offset == packed.numel(), f"unpack mismatch: {offset} of {packed.numel()}"
        return CIOutputs(lower_leaky=out[0], upper_leaky=out[1], pre_sigmoid=out[2])

    def send_g_ci_to_ci_pool_ppgd(self, g_ci_full: dict[str, Tensor]) -> None:
        """PPGD → CI: send full-model CI grads to every CI rank whose slice
        overlaps my PPGD batch shard.

        Coarse-CI: one packed send (my whole ``[b_pp]`` grad) to one CI rank.
        Fine-CI: one packed send per overlapping CI rank, each carrying that CI
        rank's overlap sub-slice. Coalesces all sites into a single packed buffer
        per destination — per-site isends launch ~10ms of NCCL overhead each, so
        at scale this would be ~1 s of pure launch latency every step.
        """
        assert self.my_pool == "ppgd" and self.my_ppgd_slice_idx is not None
        edge = self.world.ci_ppgd_edge
        my_idx = self.my_ppgd_slice_idx
        with _time_nccl_op("send_g_ci_to_ci_pool_ppgd"):
            for ci_slice_idx in edge.ci_slices_for_down_slice(my_idx):
                dst = self.world.ci_ranks[ci_slice_idx]
                sub = edge.overlap_within_down(ci_slice_idx, my_idx)
                parts = [
                    g_ci_full[s][sub].detach().to(_WIRE_DTYPE).contiguous().flatten()
                    for s in self.world.all_sites
                ]
                packed = torch.cat(parts)
                dist.send(packed, dst=dst, group=self.world.cross_pool_p2p_group)

    def send_g_vu_to_layerwise(
        self,
        v_grads: dict[str, Tensor],
        u_grads: dict[str, Tensor],
    ) -> None:
        """PPGD-leader-only: send g_VU per-block (coalesced) to each LW block leader.

        Assumes V/U grads have already been sum-reduced within the PPGD pool —
        every PPGD rank holds the same values, so only the leader sends.
        """
        assert self.my_pool == "ppgd"
        if not self.my_is_pool_leader:
            return
        works: list[dist.Work] = []
        buffers: list[Tensor] = []
        with _time_nccl_op("send_g_vu_to_layerwise:isends"):
            for bg in self.world.layerwise_block_groups:
                parts: list[Tensor] = []
                for site in bg.owned_sites:
                    parts.append(v_grads[site].to(_WIRE_DTYPE).contiguous().flatten())
                    parts.append(u_grads[site].to(_WIRE_DTYPE).contiguous().flatten())
                packed = torch.cat(parts)
                w = dist.isend(packed, dst=bg.leader, group=self.world.cross_pool_p2p_group)
                assert w is not None
                works.append(w)
                buffers.append(packed)
        with _time_nccl_op("send_g_vu_to_layerwise:wait"):
            for w in works:
                w.wait()
        del buffers

    def sum_reduce_ppgd_grads(self, grads: Iterable[Tensor]) -> None:
        """In-pool sum-reduce on PPGD V/U grads. Caller passes a flat iterable
        of tensors; each is all-reduced in place over the PPGD pool group.

        Coalesced bucketing like the other in-pool reductions.
        """
        assert self.my_pool == "ppgd"
        if dist.get_world_size(self.world.ppgd_pool_group) <= 1:
            return
        grads_list = list(grads)
        if not grads_list:
            return
        buckets: dict[tuple[torch.dtype, torch.device], list[Tensor]] = {}
        for g in grads_list:
            buckets.setdefault((g.dtype, g.device), []).append(g)
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

        with _time_nccl_op("sum_reduce_ppgd_grads"):
            for bucket in buckets.values():
                flat = _flatten_dense_tensors(bucket)
                dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.world.ppgd_pool_group)
                for orig, reduced in zip(
                    bucket, _unflatten_dense_tensors(flat, bucket), strict=True
                ):
                    orig.copy_(reduced)

    def recv_updated_vu_from_layerwise(
        self,
        v_templates: dict[str, Tensor],
        u_templates: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """PPGD ← LW: coalesced + pipelined recv of updated V/U from each LW block leader.

        Kicks off one async broadcast per block group (they pipeline across the
        per-group NCCL streams), then waits + unpacks each contiguous packet
        back into per-site V/U dicts (upcasting to the templates' dtype).
        Returns ``(v_new, u_new)`` ready for ``components[s].V.copy_(...)``.
        """
        assert self.my_pool == "ppgd"
        bufs: list[tuple[LayerwiseBlockGroup, Tensor, dist.Work]] = []
        with _time_nccl_op("recv_updated_vu_from_layerwise"):
            for bg_idx, bg in enumerate(self.world.layerwise_block_groups):
                owned = bg.owned_sites
                packed_numel = sum(v_templates[s].numel() + u_templates[s].numel() for s in owned)
                sample = v_templates[owned[0]]
                packed = torch.empty(packed_numel, dtype=_WIRE_DTYPE, device=sample.device)
                bcast_group = self.world.cross_pool_bcast_groups[bg_idx]
                w = dist.broadcast(packed, src=bg.leader, group=bcast_group, async_op=True)
                assert w is not None
                bufs.append((bg, packed, w))

        v_new: dict[str, Tensor] = {}
        u_new: dict[str, Tensor] = {}
        for bg, packed, w in bufs:
            w.wait()
            offset = 0
            for s in bg.owned_sites:
                v_n = v_templates[s].numel()
                u_n = u_templates[s].numel()
                v_new[s] = (
                    packed[offset : offset + v_n].view_as(v_templates[s]).to(v_templates[s].dtype)
                )
                offset += v_n
                u_new[s] = (
                    packed[offset : offset + u_n].view_as(u_templates[s]).to(u_templates[s].dtype)
                )
                offset += u_n
        return v_new, u_new
