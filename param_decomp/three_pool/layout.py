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

The new wrinkle vs 2-pool is **3-way batch slicing**: CI/LW/PPGD each shard
the global batch on their own axis. The MVP constraint (enforced in
``ThreePoolConfig.validate_topology``) is:

    N_ci | N_per_block_layerwise
    N_ci | N_ppgd

which reduces the otherwise many-to-many CI↔LW and CI↔PPGD routing to a
one-to-many fan-out (CI rank → K downstream ranks) plus a many-to-one
reduction (downstream ranks → one CI rank). See "Batch-split routing" below.
"""

# pyright: reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

# All cross-pool tensors are cast to this dtype on the wire (halves bytes vs fp32).
# Downstream pools run inside bf16 autocast already; CI grads and V/U grads
# accumulating into fp32 .grad upcast back to fp32 on receive — standard bf16
# mixed-precision pattern.
_WIRE_DTYPE: torch.dtype = torch.bfloat16


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
    # Matches the cross_pool_bcast_groups pattern in two_pool.
    cross_pool_bcast_groups: tuple[dist.ProcessGroup, ...]

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

    # ── Batch-split routing (the new wrinkle) ──

    @property
    def k_lw_per_ci(self) -> int:
        """How many LW batch shards (per block) fit inside one CI batch shard.

        Validator guarantees N_ci | N_per_block_layerwise, so this is an integer.
        """
        assert self.n_per_block % self.n_ci == 0
        return self.n_per_block // self.n_ci

    @property
    def k_ppgd_per_ci(self) -> int:
        """How many PPGD batch shards fit inside one CI batch shard.

        Validator guarantees N_ci | N_ppgd, so this is an integer.
        """
        assert self.n_ppgd % self.n_ci == 0
        return self.n_ppgd // self.n_ci

    def ci_slice_of_lw_block_rank(self, block_rank_idx: int) -> int:
        """Which CI rank's slice contains LW rank `block_rank_idx`'s batch shard."""
        return block_rank_idx // self.k_lw_per_ci

    def ci_slice_of_ppgd_slice(self, ppgd_slice_idx: int) -> int:
        """Which CI rank's slice contains PPGD rank `ppgd_slice_idx`'s batch shard."""
        return ppgd_slice_idx // self.k_ppgd_per_ci

    def lw_sub_slice_within_ci(self, block_rank_idx: int) -> slice:
        """Within a CI rank's local batch tensor [B_ci, ...], the sub-slice
        that corresponds to LW rank `block_rank_idx`.

        Assumes the caller is the CI rank with
        `ci_slice_idx == ci_slice_of_lw_block_rank(block_rank_idx)`.
        """
        b_lw = self.batch_local_lw
        ci_slice_idx = self.ci_slice_of_lw_block_rank(block_rank_idx)
        global_start = block_rank_idx * b_lw
        ci_global_start = ci_slice_idx * self.batch_local_ci
        local_start = global_start - ci_global_start
        return slice(local_start, local_start + b_lw)

    def ppgd_sub_slice_within_ci(self, ppgd_slice_idx: int) -> slice:
        """Within a CI rank's local batch tensor [B_ci, ...], the sub-slice
        that corresponds to PPGD rank `ppgd_slice_idx`."""
        b_pp = self.batch_local_ppgd
        ci_slice_idx = self.ci_slice_of_ppgd_slice(ppgd_slice_idx)
        global_start = ppgd_slice_idx * b_pp
        ci_global_start = ci_slice_idx * self.batch_local_ci
        local_start = global_start - ci_global_start
        return slice(local_start, local_start + b_pp)

    def lw_block_ranks_for_ci_slice(self, ci_slice_idx: int) -> tuple[int, ...]:
        """LW block_rank_idxs whose batch shards sit inside CI rank `ci_slice_idx`'s slice."""
        k = self.k_lw_per_ci
        return tuple(range(ci_slice_idx * k, (ci_slice_idx + 1) * k))

    def ppgd_slice_idxs_for_ci_slice(self, ci_slice_idx: int) -> tuple[int, ...]:
        """PPGD slice idxs whose batch shards sit inside CI rank `ci_slice_idx`'s slice."""
        k = self.k_ppgd_per_ci
        return tuple(range(ci_slice_idx * k, (ci_slice_idx + 1) * k))


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
    communicator init across all participating ranks — even when the user
    passes ``async_op=True``. In the training loop, PPGD's
    ``E_kickoff_async_recv_vu`` is the first user of these groups (it does
    irecv-side broadcasts for the V/U-from-LW pipeline that defer_vu_opt
    introduces). The matching send-side broadcast doesn't fire until LW
    step N+1 phase B4 — which means on log steps (``train_log_every`` is up),
    we end up in a deadlock:

      * LW rank 0 stuck in ``dist.recv`` from PPGD leader inside
        ``_log_train_metrics`` (PPGD leader hasn't sent yet).
      * PPGD leader stuck inside the first ``dist.broadcast`` of E_kickoff
        (communicator init waiting for LW block leaders to call into NCCL).
      * LW step N+1 (and hence phase B4) can't start until
        ``_log_train_metrics`` returns.

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

    Single class with ``my_pool`` switch (mirrors ``two_pool.BlockDDPLayout``).
    Each comm method asserts its caller pool up-front so misuse is loud.
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
        """CI → LW: for each site and each LW rank whose batch shard sits in
        my CI slice, isend the corresponding sub-slice.

        ``ci_full`` is keyed by site (CI fn produced CI for ALL sites since the
        CI fn is global). Values have shape ``[B_local_ci, S, C_s]``.
        Returned buffers must be kept alive until ``work.wait()`` completes.
        """
        assert self.my_pool == "ci" and self.my_ci_slice_idx is not None
        works: list[dist.Work] = []
        buffers: list[Tensor] = []
        my_lw_block_ranks = self.world.lw_block_ranks_for_ci_slice(self.my_ci_slice_idx)

        for bg in self.world.layerwise_block_groups:
            for block_rank_idx in my_lw_block_ranks:
                target_lw_rank = bg.ranks[block_rank_idx]
                sub = self.world.lw_sub_slice_within_ci(block_rank_idx)
                for site in bg.owned_sites:
                    buf = ci_full[site][sub].detach().to(_WIRE_DTYPE).contiguous()
                    works.append(dist.isend(buf, dst=target_lw_rank))
                    buffers.append(buf)
        return works, buffers

    def async_send_ci_to_ppgd(
        self, ci_full: dict[str, Tensor]
    ) -> tuple[list["dist.Work"], list[Tensor]]:
        """CI → PPGD: for each PPGD rank whose batch shard sits in my CI slice,
        isend the full-model CI sub-slice (all sites)."""
        assert self.my_pool == "ci" and self.my_ci_slice_idx is not None
        works: list[dist.Work] = []
        buffers: list[Tensor] = []
        my_ppgd_slice_idxs = self.world.ppgd_slice_idxs_for_ci_slice(self.my_ci_slice_idx)

        for ppgd_slice_idx in my_ppgd_slice_idxs:
            target_ppgd_rank = self.world.ppgd_ranks[ppgd_slice_idx]
            sub = self.world.ppgd_sub_slice_within_ci(ppgd_slice_idx)
            for site in self.world.all_sites:
                buf = ci_full[site][sub].detach().to(_WIRE_DTYPE).contiguous()
                works.append(dist.isend(buf, dst=target_ppgd_rank))
                buffers.append(buf)
        return works, buffers

    def recv_g_ci_from_layerwise(
        self,
        site_to_c: dict[str, int],
        seq_len: int,
        device: torch.device,
    ) -> dict[str, Tensor]:
        """CI ← LW: recv per-site CI grads. Stitch K_lw sub-slices per site
        into one [B_local_ci, S, C_s] tensor (fp32 after upcast)."""
        assert self.my_pool == "ci" and self.my_ci_slice_idx is not None
        my_lw_block_ranks = self.world.lw_block_ranks_for_ci_slice(self.my_ci_slice_idx)
        b_lw = self.world.batch_local_lw

        # Post all irecvs upfront so they pipeline on the NIC.
        pending: list[tuple[str, int, Tensor, dist.Work]] = []
        for site in self.world.all_sites:
            bg = self.world.layerwise_block_groups[self.world.block_idx_of_site(site)]
            c_s = site_to_c[site]
            for block_rank_idx in my_lw_block_ranks:
                src = bg.ranks[block_rank_idx]
                buf = torch.empty(b_lw, seq_len, c_s, device=device, dtype=_WIRE_DTYPE)
                w = dist.irecv(buf, src=src)
                assert w is not None
                pending.append((site, block_rank_idx, buf, w))

        # Wait + stitch. Allocate one fp32 dest per site, copy each piece in place.
        out: dict[str, Tensor] = {}
        b_ci = self.world.batch_local_ci
        for site in self.world.all_sites:
            c_s = site_to_c[site]
            out[site] = torch.empty(b_ci, seq_len, c_s, device=device, dtype=torch.float32)
        for site, block_rank_idx, buf, w in pending:
            w.wait()
            sub = self.world.lw_sub_slice_within_ci(block_rank_idx)
            out[site][sub].copy_(buf.to(torch.float32))
        return out

    def recv_g_ci_from_ppgd(
        self,
        site_to_c: dict[str, int],
        seq_len: int,
        device: torch.device,
    ) -> dict[str, Tensor]:
        """CI ← PPGD: recv full-model CI grads. Stitch K_ppgd sub-slices per
        site into one [B_local_ci, S, C_s] tensor (fp32 after upcast)."""
        assert self.my_pool == "ci" and self.my_ci_slice_idx is not None
        my_ppgd_slice_idxs = self.world.ppgd_slice_idxs_for_ci_slice(self.my_ci_slice_idx)
        b_pp = self.world.batch_local_ppgd

        pending: list[tuple[str, int, Tensor, dist.Work]] = []
        for ppgd_slice_idx in my_ppgd_slice_idxs:
            src = self.world.ppgd_ranks[ppgd_slice_idx]
            for site in self.world.all_sites:
                c_s = site_to_c[site]
                buf = torch.empty(b_pp, seq_len, c_s, device=device, dtype=_WIRE_DTYPE)
                w = dist.irecv(buf, src=src)
                assert w is not None
                pending.append((site, ppgd_slice_idx, buf, w))

        out: dict[str, Tensor] = {}
        b_ci = self.world.batch_local_ci
        for site in self.world.all_sites:
            c_s = site_to_c[site]
            out[site] = torch.empty(b_ci, seq_len, c_s, device=device, dtype=torch.float32)
        for site, ppgd_slice_idx, buf, w in pending:
            w.wait()
            sub = self.world.ppgd_sub_slice_within_ci(ppgd_slice_idx)
            out[site][sub].copy_(buf.to(torch.float32))
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

        for bucket in buckets.values():
            flat = _flatten_dense_tensors(bucket)
            dist.all_reduce(flat, op=dist.ReduceOp.AVG, group=self.world.ci_pool_group)
            for orig, reduced in zip(bucket, _unflatten_dense_tensors(flat, bucket), strict=True):
                orig.copy_(reduced)

    # ──────────────────────────────────────────────────────────────────────
    # Layerwise-pool comm methods
    # ──────────────────────────────────────────────────────────────────────

    def async_recv_ci_from_ci_pool(
        self,
        site_to_c: dict[str, int],
        seq_len: int,
        device: torch.device,
    ) -> tuple[dict[str, Tensor], list["dist.Work"]]:
        """LW ← CI: irecv per-owned-site CI values from the CI rank whose
        slice contains my LW batch shard. Returns raw bf16 buffers + work
        handles; caller waits + casts to fp32."""
        assert self.my_pool == "layerwise" and self.my_within_block_idx is not None
        src_ci_slice = self.world.ci_slice_of_lw_block_rank(self.my_within_block_idx)
        src = self.world.ci_ranks[src_ci_slice]
        b_lw = self.world.batch_local_lw

        out: dict[str, Tensor] = {}
        works: list[dist.Work] = []
        for site in self.my_owned_sites:
            C = site_to_c[site]
            buf = torch.empty(b_lw, seq_len, C, device=device, dtype=_WIRE_DTYPE)
            w = dist.irecv(buf, src=src)
            assert w is not None
            out[site] = buf
            works.append(w)
        return out, works

    def send_g_ci_to_ci_pool(self, g_ci_owned: dict[str, Tensor]) -> None:
        """LW → CI: send per-owned-site CI grads (full LW batch slice) to the
        CI rank that owns my slice. Synchronous — runs after the LW backward
        when grads are ready.
        """
        assert self.my_pool == "layerwise" and self.my_within_block_idx is not None
        dst_ci_slice = self.world.ci_slice_of_lw_block_rank(self.my_within_block_idx)
        dst = self.world.ci_ranks[dst_ci_slice]
        # Per-site isend; wait at end so they pipeline.
        works: list[dist.Work] = []
        buffers: list[Tensor] = []
        for site in self.my_owned_sites:
            buf = g_ci_owned[site].detach().to(_WIRE_DTYPE).contiguous()
            w = dist.isend(buf, dst=dst)
            assert w is not None
            works.append(w)
            buffers.append(buf)
        for w in works:
            w.wait()
        del buffers

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
            dist.recv(packed, src=ppgd_leader)
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
        PPGD ranks. Mirrors two_pool's async_send_updated_weights_to_pool_b.
        Caller must keep the buffer alive until the work handle completes.
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
        w = dist.broadcast(packed, src=self.my_rank, group=bcast_group, async_op=True)
        assert w is not None
        return [w], [packed]

    def all_reduce_grads_in_block(self, params: Iterable[nn.Parameter]) -> None:
        """Coalesced in-block DDP all-reduce over V/U + faithfulness grads.

        Synchronous variant — Python blocks until the collective completes.
        Identical pattern to ``two_pool.BlockDDPLayout.all_reduce_grads_in_block``.
        Used by the sync path in ``step_layerwise_tail``.
        """
        states = self.async_all_reduce_grads_in_block_kickoff(params)
        self.wait_and_unflatten_all_reduce(states)

    def async_all_reduce_grads_in_block_kickoff(
        self, params: Iterable[nn.Parameter]
    ) -> list[tuple[list[Tensor], Tensor, "dist.Work"]]:
        """Kick off async coalesced in-block all-reduce. Returns one
        ``(bucket, flat, work)`` tuple per (dtype, device) bucket. Caller MUST
        keep these alive (storing them across iteration boundaries) until
        ``wait_and_unflatten_all_reduce`` runs — the ``flat`` tensors are the
        NCCL buffers, and freeing them while NCCL is still operating would be
        undefined.

        Empty return when the block group is 1-rank (no-op) or no grads.

        The async variant lets the caller overlap the all-reduce with other
        compute on the default CUDA stream (e.g. the next iteration's
        target_fwd, which is V/U-independent and runs on a different stream).
        """
        assert self.my_pool == "layerwise" and self.my_block_idx is not None
        block_group = self.world.block_group_groups[self.my_block_idx]
        if dist.get_world_size(block_group) <= 1:
            return []
        grads: list[Tensor] = [p.grad for p in params if p.grad is not None]
        if not grads:
            return []
        buckets: dict[tuple[torch.dtype, torch.device], list[Tensor]] = {}
        for g in grads:
            buckets.setdefault((g.dtype, g.device), []).append(g)
        from torch._utils import _flatten_dense_tensors

        states: list[tuple[list[Tensor], Tensor, dist.Work]] = []
        for bucket in buckets.values():
            flat = _flatten_dense_tensors(bucket)
            w = dist.all_reduce(flat, op=dist.ReduceOp.AVG, group=block_group, async_op=True)
            assert w is not None
            states.append((bucket, flat, w))
        return states

    def wait_and_unflatten_all_reduce(
        self,
        states: list[tuple[list[Tensor], Tensor, "dist.Work"]],
    ) -> None:
        """Wait on each async all-reduce work and copy the reduced flat tensor
        back to the original ``.grad`` buffers."""
        from torch._utils import _unflatten_dense_tensors

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
    ) -> tuple[dict[str, Tensor], list["dist.Work"]]:
        """PPGD ← CI: irecv full-model CI from the CI rank that owns my slice."""
        assert self.my_pool == "ppgd" and self.my_ppgd_slice_idx is not None
        src_ci_slice = self.world.ci_slice_of_ppgd_slice(self.my_ppgd_slice_idx)
        src = self.world.ci_ranks[src_ci_slice]
        b_pp = self.world.batch_local_ppgd

        out: dict[str, Tensor] = {}
        works: list[dist.Work] = []
        for site in self.world.all_sites:
            C = site_to_c[site]
            buf = torch.empty(b_pp, seq_len, C, device=device, dtype=_WIRE_DTYPE)
            w = dist.irecv(buf, src=src)
            assert w is not None
            out[site] = buf
            works.append(w)
        return out, works

    def send_g_ci_to_ci_pool_ppgd(self, g_ci_full: dict[str, Tensor]) -> None:
        """PPGD → CI: send full-model CI grads (PPGD batch slice) to the CI
        rank that owns my slice."""
        assert self.my_pool == "ppgd" and self.my_ppgd_slice_idx is not None
        dst_ci_slice = self.world.ci_slice_of_ppgd_slice(self.my_ppgd_slice_idx)
        dst = self.world.ci_ranks[dst_ci_slice]
        works: list[dist.Work] = []
        buffers: list[Tensor] = []
        for site in self.world.all_sites:
            buf = g_ci_full[site].detach().to(_WIRE_DTYPE).contiguous()
            w = dist.isend(buf, dst=dst)
            assert w is not None
            works.append(w)
            buffers.append(buf)
        for w in works:
            w.wait()
        del buffers

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
        for bg in self.world.layerwise_block_groups:
            parts: list[Tensor] = []
            for site in bg.owned_sites:
                parts.append(v_grads[site].to(_WIRE_DTYPE).contiguous().flatten())
                parts.append(u_grads[site].to(_WIRE_DTYPE).contiguous().flatten())
            packed = torch.cat(parts)
            w = dist.isend(packed, dst=bg.leader)
            assert w is not None
            works.append(w)
            buffers.append(packed)
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

        for bucket in buckets.values():
            flat = _flatten_dense_tensors(bucket)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.world.ppgd_pool_group)
            for orig, reduced in zip(bucket, _unflatten_dense_tensors(flat, bucket), strict=True):
                orig.copy_(reduced)

    def recv_updated_vu_from_layerwise(
        self,
        v_templates: dict[str, Tensor],
        u_templates: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """PPGD ← LW: coalesced + pipelined recv of updated V/U from each LW block leader.

        Synchronous variant — kicks off all broadcasts then waits + unpacks.
        Use ``async_recv_updated_vu_from_layerwise_kickoff`` +
        ``wait_and_unpack_updated_vu`` to overlap with PPGD's target_fwd.
        """
        states = self.async_recv_updated_vu_from_layerwise_kickoff(v_templates, u_templates)
        return self.wait_and_unpack_updated_vu(states, v_templates, u_templates)

    def async_recv_updated_vu_from_layerwise_kickoff(
        self,
        v_templates: dict[str, Tensor],
        u_templates: dict[str, Tensor],
    ) -> list[tuple["LayerwiseBlockGroup", Tensor, "dist.Work"]]:
        """Kick off async coalesced V/U recv from every LW block leader. Returns
        per-block ``(block_group, packed_buf, work)`` tuples. Caller holds
        these across iteration boundaries until
        ``wait_and_unpack_updated_vu`` runs.

        Overlap target: PPGD's next-iter target_fwd, which runs on the default
        CUDA stream and doesn't depend on V/U. The broadcasts run on NCCL
        streams (one per block group's bcast group), so they pipeline.
        """
        assert self.my_pool == "ppgd"
        bufs: list[tuple[LayerwiseBlockGroup, Tensor, dist.Work]] = []
        for bg_idx, bg in enumerate(self.world.layerwise_block_groups):
            owned = bg.owned_sites
            packed_numel = sum(v_templates[s].numel() + u_templates[s].numel() for s in owned)
            sample = v_templates[owned[0]]
            packed = torch.empty(packed_numel, dtype=_WIRE_DTYPE, device=sample.device)
            bcast_group = self.world.cross_pool_bcast_groups[bg_idx]
            w = dist.broadcast(packed, src=bg.leader, group=bcast_group, async_op=True)
            assert w is not None
            bufs.append((bg, packed, w))
        return bufs

    def wait_and_unpack_updated_vu(
        self,
        states: list[tuple["LayerwiseBlockGroup", Tensor, "dist.Work"]],
        v_templates: dict[str, Tensor],
        u_templates: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Wait on each block's broadcast and unpack the contiguous packed
        tensor back into per-site V/U dicts (upcasting to the templates' dtype).
        Returns ``(v_new, u_new)`` ready for ``components[s].V.copy_(...)``.
        """
        v_new: dict[str, Tensor] = {}
        u_new: dict[str, Tensor] = {}
        for bg, packed, w in states:
            w.wait()
            owned = bg.owned_sites
            offset = 0
            for s in owned:
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
