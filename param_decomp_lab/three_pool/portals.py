"""Cross-pool exchanges as first-class typed objects — one per DAG edge.

The 3-pool program has six cross-pool point-to-point exchanges per step (plus
one eval-only). Previously each edge was split across two ``ThreePoolLayout``
methods (a send on the source pool, a recv on the dest pool) with the routing,
pack layout, and process group duplicated on both sides — free to drift.

Here each edge is ONE class. Its pack layout, routing bijection, wire dtype, and
process group live in a single place; the sender and receiver are the two halves
of the same class, so they cannot disagree. A pool only constructs the portal
halves at its own endpoints (see ``CIPortals`` / ``LWPortals`` / ``PPGDPortals``
below), so an LW rank physically cannot invoke a CI-pool send.

The six edges (see ``DESIGN.md``):

  CI  → LW   : ``CiValuesToLayerwise``   (per-site, owned + LW-rank slice)
  CI  → PPGD : ``CiValuesToPPGD``        (full-model, per-PPGD-rank slice)
  LW  → CI   : ``GradCiFromLayerwise``   (per-owned-site, per-LW-rank slice)
  PPGD→ CI   : ``GradCiFromPPGD``        (full-model, per-PPGD-rank slice)
  PPGD→ LW   : ``GradVuFromPPGD``        (per-owned-site, after in-pool sum-reduce)
  LW  → PPGD : ``UpdatedVuToPPGD``       (per-owned-site, leader-rooted broadcast)

Eval-only:

  CI  → PPGD : ``CiOutputsEvalToPPGD``   (full CIOutputs, three dicts)

Each P2P portal carries the shared ``cross_pool_p2p_group``. The V/U broadcast
edge uses the per-block ``cross_pool_bcast_groups`` ({block leader} ∪ PPGD).

All routing math (which CI slice owns which LW/PPGD shard, sub-slices within a CI
batch tensor) stays on ``World``; the portals call those helpers so there's
still a single source of truth for the topology.
"""

# pyright: reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from param_decomp.component_model import CIOutputs
from param_decomp_lab.three_pool.layout import (
    LayerwiseBlockGroup,
    World,
    time_nccl_op,
)
from param_decomp_lab.three_pool.role import CIRole, LWRole, PPGDRole

# All cross-pool tensors are cast to this dtype on the wire (halves bytes vs
# fp32). Downstream pools run inside bf16 autocast already; CI grads and V/U
# grads accumulate into fp32 ``.grad`` and upcast back on receive — standard
# bf16 mixed-precision pattern.
WIRE_DTYPE: torch.dtype = torch.bfloat16


# ──────────────────────────────────────────────────────────────────────────────
# Pending receives — a posted irecv held until ``wait()`` materializes its payload.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PendingCiValues:
    """A coalesced CI-values irecv, held until ``wait()``.

    The packed buffer carries ``sites`` worth of CI values (in order) as
    ``b * seq_len * c_s`` ``WIRE_DTYPE`` elements each. ``wait`` blocks on the
    ``dist.Work`` then materializes per-site ``[b, seq_len, c_s]`` views into
    the packed buffer (no copy).
    """

    packed: Tensor
    work: "dist.Work"
    sites: tuple[str, ...]
    site_to_c: dict[str, int]
    b: int
    seq_len: int

    def wait(self) -> dict[str, Tensor]:
        self.work.wait()
        out: dict[str, Tensor] = {}
        offset = 0
        for s in self.sites:
            c_s = self.site_to_c[s]
            numel = self.b * self.seq_len * c_s
            out[s] = self.packed[offset : offset + numel].view(self.b, self.seq_len, c_s)
            offset += numel
        assert offset == self.packed.numel(), (
            f"unpack size mismatch: consumed {offset} of {self.packed.numel()}"
        )
        return out


@dataclass(frozen=True)
class PendingUpdatedVu:
    """Pipelined per-block broadcasts of updated V/U, held until ``wait()``."""

    bufs: tuple[tuple[LayerwiseBlockGroup, Tensor, "dist.Work"], ...]
    v_templates: dict[str, Tensor]
    u_templates: dict[str, Tensor]

    def wait(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        v_new: dict[str, Tensor] = {}
        u_new: dict[str, Tensor] = {}
        for bg, packed, w in self.bufs:
            w.wait()
            offset = 0
            for s in bg.owned_sites:
                v_t, u_t = self.v_templates[s], self.u_templates[s]
                v_n, u_n = v_t.numel(), u_t.numel()
                v_new[s] = packed[offset : offset + v_n].view_as(v_t).to(v_t.dtype)
                offset += v_n
                u_new[s] = packed[offset : offset + u_n].view_as(u_t).to(u_t.dtype)
                offset += u_n
        return v_new, u_new


@dataclass(frozen=True)
class InFlightSends:
    """Async isends/broadcasts whose source buffers must stay alive until ``wait()``."""

    works: tuple["dist.Work", ...]
    buffers: tuple[Tensor, ...]

    def wait(self) -> None:
        for w in self.works:
            w.wait()


def _pack_sites(d: dict[str, Tensor], sites: Iterable[str], sub: slice | None) -> Tensor:
    """Flatten ``d[site][sub]`` (or ``d[site]`` if ``sub`` is None) for each site,
    cast to the wire dtype, and concatenate into one contiguous buffer."""
    parts = [
        (d[s][sub] if sub is not None else d[s]).detach().to(WIRE_DTYPE).contiguous().flatten()
        for s in sites
    ]
    return torch.cat(parts)


# ──────────────────────────────────────────────────────────────────────────────
# CI → LW : per-site CI values (owned sites, LW-rank batch sub-slice)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CiValuesToLayerwise:
    world: World

    def send(self, role: CIRole, ci_full: dict[str, Tensor]) -> InFlightSends:
        """For each site and each LW rank whose batch shard sits in my CI slice,
        isend the corresponding sub-slice. ``ci_full`` is keyed by site (the CI
        fn is global) with values ``[B_local_ci, S, C_s]``."""
        works: list[dist.Work] = []
        buffers: list[Tensor] = []
        my_lw_block_ranks = self.world.lw_block_ranks_for_ci_slice(role.slice_idx)
        with time_nccl_op("CiValuesToLayerwise.send"):
            for bg in self.world.layerwise_block_groups:
                for block_rank_idx in my_lw_block_ranks:
                    target = bg.ranks[block_rank_idx]
                    sub = self.world.lw_sub_slice_within_ci(block_rank_idx)
                    packed = _pack_sites(ci_full, bg.owned_sites, sub)
                    works.append(
                        dist.isend(packed, dst=target, group=self.world.cross_pool_p2p_group)
                    )
                    buffers.append(packed)
        return InFlightSends(works=tuple(works), buffers=tuple(buffers))

    def post_recv(
        self, role: LWRole, site_to_c: dict[str, int], seq_len: int, device: torch.device
    ) -> PendingCiValues:
        """irecv one coalesced packet of CI values for all of this LW rank's
        owned sites, from the CI rank whose slice contains my LW batch shard."""
        src_ci_slice = self.world.ci_slice_of_lw_block_rank(role.within_block_idx)
        src = self.world.ci_ranks[src_ci_slice]
        b_lw = self.world.batch_local_lw
        packed_numel = sum(b_lw * seq_len * site_to_c[s] for s in role.owned_sites)
        packed = torch.empty(packed_numel, device=device, dtype=WIRE_DTYPE)
        with time_nccl_op("CiValuesToLayerwise.recv"):
            work = dist.irecv(packed, src=src, group=self.world.cross_pool_p2p_group)
            assert work is not None
        return PendingCiValues(
            packed=packed,
            work=work,
            sites=role.owned_sites,
            site_to_c=site_to_c,
            b=b_lw,
            seq_len=seq_len,
        )


# ──────────────────────────────────────────────────────────────────────────────
# CI → PPGD : full-model CI values (all sites, PPGD-rank batch sub-slice)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CiValuesToPPGD:
    world: World

    def send(self, role: CIRole, ci_full: dict[str, Tensor]) -> InFlightSends:
        works: list[dist.Work] = []
        buffers: list[Tensor] = []
        my_ppgd_slice_idxs = self.world.ppgd_slice_idxs_for_ci_slice(role.slice_idx)
        with time_nccl_op("CiValuesToPPGD.send"):
            for ppgd_slice_idx in my_ppgd_slice_idxs:
                target = self.world.ppgd_ranks[ppgd_slice_idx]
                sub = self.world.ppgd_sub_slice_within_ci(ppgd_slice_idx)
                packed = _pack_sites(ci_full, self.world.all_sites, sub)
                works.append(dist.isend(packed, dst=target, group=self.world.cross_pool_p2p_group))
                buffers.append(packed)
        return InFlightSends(works=tuple(works), buffers=tuple(buffers))

    def post_recv(
        self, role: PPGDRole, site_to_c: dict[str, int], seq_len: int, device: torch.device
    ) -> PendingCiValues:
        src_ci_slice = self.world.ci_slice_of_ppgd_slice(role.slice_idx)
        src = self.world.ci_ranks[src_ci_slice]
        b_pp = self.world.batch_local_ppgd
        packed_numel = sum(b_pp * seq_len * site_to_c[s] for s in self.world.all_sites)
        packed = torch.empty(packed_numel, device=device, dtype=WIRE_DTYPE)
        with time_nccl_op("CiValuesToPPGD.recv"):
            work = dist.irecv(packed, src=src, group=self.world.cross_pool_p2p_group)
            assert work is not None
        return PendingCiValues(
            packed=packed,
            work=work,
            sites=self.world.all_sites,
            site_to_c=site_to_c,
            b=b_pp,
            seq_len=seq_len,
        )


# ──────────────────────────────────────────────────────────────────────────────
# LW → CI : per-owned-site CI grads (per-LW-rank batch slice; stitched on CI side)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GradCiFromLayerwise:
    world: World

    def send(self, role: LWRole, g_ci_owned: dict[str, Tensor]) -> None:
        """Send per-owned-site CI grads (full LW batch slice) to the CI rank
        that owns my slice. Owned sites coalesced into one packed send."""
        dst_ci_slice = self.world.ci_slice_of_lw_block_rank(role.within_block_idx)
        dst = self.world.ci_ranks[dst_ci_slice]
        packed = _pack_sites(g_ci_owned, role.owned_sites, sub=None)
        with time_nccl_op("GradCiFromLayerwise.send"):
            dist.send(packed, dst=dst, group=self.world.cross_pool_p2p_group)

    def recv(
        self, role: CIRole, site_to_c: dict[str, int], seq_len: int, device: torch.device
    ) -> dict[str, Tensor]:
        """Recv per-site CI grads, one coalesced buffer per (LW block, LW rank
        index), stitched into per-site ``[B_local_ci, S, C_s]`` fp32 dests."""
        my_lw_block_ranks = self.world.lw_block_ranks_for_ci_slice(role.slice_idx)
        b_lw = self.world.batch_local_lw

        pending: list[tuple[int, Tensor, dist.Work, tuple[str, ...]]] = []
        with time_nccl_op("GradCiFromLayerwise.recv:post_irecvs"):
            for bg in self.world.layerwise_block_groups:
                owned = bg.owned_sites
                packed_numel = sum(b_lw * seq_len * site_to_c[s] for s in owned)
                for block_rank_idx in my_lw_block_ranks:
                    src = bg.ranks[block_rank_idx]
                    buf = torch.empty(packed_numel, device=device, dtype=WIRE_DTYPE)
                    w = dist.irecv(buf, src=src, group=self.world.cross_pool_p2p_group)
                    assert w is not None
                    pending.append((block_rank_idx, buf, w, owned))

        b_ci = self.world.batch_local_ci
        out: dict[str, Tensor] = {
            s: torch.empty(b_ci, seq_len, site_to_c[s], device=device, dtype=torch.float32)
            for s in self.world.all_sites
        }
        with time_nccl_op("GradCiFromLayerwise.recv:wait"):
            for block_rank_idx, buf, w, owned in pending:
                w.wait()
                sub = self.world.lw_sub_slice_within_ci(block_rank_idx)
                offset = 0
                for site in owned:
                    c_s = site_to_c[site]
                    n = b_lw * seq_len * c_s
                    site_view = buf[offset : offset + n].view(b_lw, seq_len, c_s)
                    out[site][sub].copy_(site_view.to(torch.float32))
                    offset += n
        return out


# ──────────────────────────────────────────────────────────────────────────────
# PPGD → CI : full-model CI grads (per-PPGD-rank batch slice; stitched on CI side)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GradCiFromPPGD:
    world: World

    def send(self, role: PPGDRole, g_ci_full: dict[str, Tensor]) -> None:
        """Send full-model CI grads (PPGD batch slice) to the CI rank that owns
        my slice. All sites coalesced into a single packed buffer."""
        dst_ci_slice = self.world.ci_slice_of_ppgd_slice(role.slice_idx)
        dst = self.world.ci_ranks[dst_ci_slice]
        packed = _pack_sites(g_ci_full, self.world.all_sites, sub=None)
        with time_nccl_op("GradCiFromPPGD.send"):
            dist.send(packed, dst=dst, group=self.world.cross_pool_p2p_group)

    def recv(
        self, role: CIRole, site_to_c: dict[str, int], seq_len: int, device: torch.device
    ) -> dict[str, Tensor]:
        my_ppgd_slice_idxs = self.world.ppgd_slice_idxs_for_ci_slice(role.slice_idx)
        b_pp = self.world.batch_local_ppgd
        site_numels = {s: b_pp * seq_len * site_to_c[s] for s in self.world.all_sites}
        packed_numel = sum(site_numels.values())

        pending: list[tuple[int, Tensor, dist.Work]] = []
        with time_nccl_op("GradCiFromPPGD.recv:post_irecvs"):
            for ppgd_slice_idx in my_ppgd_slice_idxs:
                src = self.world.ppgd_ranks[ppgd_slice_idx]
                packed = torch.empty(packed_numel, device=device, dtype=WIRE_DTYPE)
                w = dist.irecv(packed, src=src, group=self.world.cross_pool_p2p_group)
                assert w is not None
                pending.append((ppgd_slice_idx, packed, w))

        b_ci = self.world.batch_local_ci
        out: dict[str, Tensor] = {
            s: torch.empty(b_ci, seq_len, site_to_c[s], device=device, dtype=torch.float32)
            for s in self.world.all_sites
        }
        with time_nccl_op("GradCiFromPPGD.recv:wait"):
            for ppgd_slice_idx, packed, w in pending:
                w.wait()
                sub = self.world.ppgd_sub_slice_within_ci(ppgd_slice_idx)
                offset = 0
                for site in self.world.all_sites:
                    n = site_numels[site]
                    buf = packed[offset : offset + n].view(b_pp, seq_len, site_to_c[site])
                    out[site][sub].copy_(buf.to(torch.float32))
                    offset += n
        return out


# ──────────────────────────────────────────────────────────────────────────────
# PPGD → LW : per-owned-site V/U grads (after PPGD in-pool sum-reduce)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GradVuFromPPGD:
    world: World

    def send(self, role: PPGDRole, v_grads: dict[str, Tensor], u_grads: dict[str, Tensor]) -> None:
        """PPGD-leader-only: send g_VU per-block (coalesced) to each LW block
        leader. Assumes V/U grads were already sum-reduced within the PPGD pool,
        so every PPGD rank holds the same values and only the leader sends."""
        if not role.is_pool_leader:
            return
        works: list[dist.Work] = []
        buffers: list[Tensor] = []
        with time_nccl_op("GradVuFromPPGD.send:isends"):
            for bg in self.world.layerwise_block_groups:
                parts: list[Tensor] = []
                for site in bg.owned_sites:
                    parts.append(v_grads[site].to(WIRE_DTYPE).contiguous().flatten())
                    parts.append(u_grads[site].to(WIRE_DTYPE).contiguous().flatten())
                packed = torch.cat(parts)
                w = dist.isend(packed, dst=bg.leader, group=self.world.cross_pool_p2p_group)
                assert w is not None
                works.append(w)
                buffers.append(packed)
        with time_nccl_op("GradVuFromPPGD.send:wait"):
            for w in works:
                w.wait()
        del buffers

    def recv(
        self, role: LWRole, v_templates: dict[str, Tensor], u_templates: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Block leader recvs g_VU for owned sites from PPGD leader, then
        in-block broadcasts so all replicas see the same grad."""
        v_grads: dict[str, Tensor] = {}
        u_grads: dict[str, Tensor] = {}

        if role.is_block_leader:
            my_sites = role.owned_sites
            packed_numel = sum(v_templates[s].numel() + u_templates[s].numel() for s in my_sites)
            sample = v_templates[my_sites[0]]
            packed = torch.empty(packed_numel, dtype=WIRE_DTYPE, device=sample.device)
            ppgd_leader = self.world.ppgd_ranks[0]
            with time_nccl_op("GradVuFromPPGD.recv:recv"):
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
            for s in role.owned_sites:
                v_grads[s] = torch.empty_like(v_templates[s])
                u_grads[s] = torch.empty_like(u_templates[s])

        block_group = self.world.block_group_groups[role.block_idx]
        block_leader_rank = self.world.layerwise_block_groups[role.block_idx].leader
        with time_nccl_op("GradVuFromPPGD.recv:in_block_bcast"):
            for s in role.owned_sites:
                v_grads[s] = v_grads[s].contiguous()
                u_grads[s] = u_grads[s].contiguous()
                dist.broadcast(v_grads[s], src=block_leader_rank, group=block_group)
                dist.broadcast(u_grads[s], src=block_leader_rank, group=block_group)
        return v_grads, u_grads


# ──────────────────────────────────────────────────────────────────────────────
# LW → PPGD : updated V/U (leader-rooted broadcast over {leader} ∪ PPGD)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class UpdatedVuToPPGD:
    world: World

    def send(
        self, role: LWRole, v_owned: dict[str, Tensor], u_owned: dict[str, Tensor]
    ) -> InFlightSends:
        """Coalesced leader-rooted broadcast of updated V/U to all PPGD ranks.
        Only the block leader sends; others no-op. Caller keeps the buffer alive
        until the work completes."""
        if not role.is_block_leader:
            return InFlightSends(works=(), buffers=())
        parts: list[Tensor] = []
        for s in role.owned_sites:
            parts.append(v_owned[s].detach().to(WIRE_DTYPE).contiguous().flatten())
            parts.append(u_owned[s].detach().to(WIRE_DTYPE).contiguous().flatten())
        packed = torch.cat(parts)
        bcast_group = self.world.cross_pool_bcast_groups[role.block_idx]
        with time_nccl_op("UpdatedVuToPPGD.send"):
            w = dist.broadcast(packed, src=role.rank, group=bcast_group, async_op=True)
        assert w is not None
        return InFlightSends(works=(w,), buffers=(packed,))

    def post_recv(
        self, v_templates: dict[str, Tensor], u_templates: dict[str, Tensor]
    ) -> PendingUpdatedVu:
        """Kick off one async broadcast per block group (they pipeline across
        the per-group NCCL streams); ``wait`` unpacks each into per-site V/U."""
        bufs: list[tuple[LayerwiseBlockGroup, Tensor, dist.Work]] = []
        with time_nccl_op("UpdatedVuToPPGD.recv"):
            for bg_idx, bg in enumerate(self.world.layerwise_block_groups):
                owned = bg.owned_sites
                packed_numel = sum(v_templates[s].numel() + u_templates[s].numel() for s in owned)
                sample = v_templates[owned[0]]
                packed = torch.empty(packed_numel, dtype=WIRE_DTYPE, device=sample.device)
                bcast_group = self.world.cross_pool_bcast_groups[bg_idx]
                w = dist.broadcast(packed, src=bg.leader, group=bcast_group, async_op=True)
                assert w is not None
                bufs.append((bg, packed, w))
        return PendingUpdatedVu(bufs=tuple(bufs), v_templates=v_templates, u_templates=u_templates)


# ──────────────────────────────────────────────────────────────────────────────
# CI → PPGD eval : full CIOutputs (lower_leaky, upper_leaky, pre_sigmoid)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CiOutputsEvalToPPGD:
    world: World

    def send(self, role: CIRole, ci: CIOutputs) -> None:
        """Synchronous send of full CIOutputs (all three dicts) sliced to each
        PPGD rank within my CI slice. Eval is rare; overlap has no value here.

        Pack layout (must match ``recv``): three contiguous blocks in order
        (lower_leaky, upper_leaky, pre_sigmoid)."""
        my_ppgd_slice_idxs = self.world.ppgd_slice_idxs_for_ci_slice(role.slice_idx)
        with time_nccl_op("CiOutputsEvalToPPGD.send"):
            for ppgd_slice_idx in my_ppgd_slice_idxs:
                target = self.world.ppgd_ranks[ppgd_slice_idx]
                sub = self.world.ppgd_sub_slice_within_ci(ppgd_slice_idx)
                parts: list[Tensor] = []
                for d in (ci.lower_leaky, ci.upper_leaky, ci.pre_sigmoid):
                    parts.append(_pack_sites(d, self.world.all_sites, sub))
                packed = torch.cat(parts)
                dist.send(packed, dst=target, group=self.world.cross_pool_p2p_group)

    def recv(
        self, role: PPGDRole, site_to_c: dict[str, int], seq_len: int, device: torch.device
    ) -> CIOutputs:
        src_ci_slice = self.world.ci_slice_of_ppgd_slice(role.slice_idx)
        src = self.world.ci_ranks[src_ci_slice]
        b_pp = self.world.batch_local_ppgd
        per_block_numel = sum(b_pp * seq_len * site_to_c[s] for s in self.world.all_sites)
        packed = torch.empty(3 * per_block_numel, device=device, dtype=WIRE_DTYPE)
        with time_nccl_op("CiOutputsEvalToPPGD.recv"):
            dist.recv(packed, src=src, group=self.world.cross_pool_p2p_group)

        out: list[dict[str, Tensor]] = [{}, {}, {}]
        offset = 0
        for block_idx in range(3):
            for site in self.world.all_sites:
                c_s = site_to_c[site]
                numel = b_pp * seq_len * c_s
                out[block_idx][site] = packed[offset : offset + numel].view(b_pp, seq_len, c_s)
                offset += numel
        assert offset == packed.numel(), f"unpack mismatch: {offset} of {packed.numel()}"
        return CIOutputs(lower_leaky=out[0], upper_leaky=out[1], pre_sigmoid=out[2])


# ──────────────────────────────────────────────────────────────────────────────
# In-pool collectives (one per pool, kept here so each pool's comms live together)
# ──────────────────────────────────────────────────────────────────────────────


def _bucketed_all_reduce(
    grads: Iterable[Tensor], op: "dist.ReduceOp.RedOpType", group: dist.ProcessGroup, label: str
) -> None:
    """Coalesce grads into (dtype, device) buckets, all-reduce each, copy back."""
    grads_list = list(grads)
    if not grads_list:
        return
    buckets: dict[tuple[torch.dtype, torch.device], list[Tensor]] = {}
    for g in grads_list:
        buckets.setdefault((g.dtype, g.device), []).append(g)
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    with time_nccl_op(label):
        for bucket in buckets.values():
            flat = _flatten_dense_tensors(bucket)
            dist.all_reduce(flat, op=op, group=group)
            for orig, reduced in zip(bucket, _unflatten_dense_tensors(flat, bucket), strict=True):
                orig.copy_(reduced)


def all_reduce_ci_fn_grads(world: World, params: Iterable[nn.Parameter]) -> None:
    """CI in-pool AVG-reduce on CI fn grads (standard DDP). No-op for 1-rank pool."""
    if dist.get_world_size(world.ci_pool_group) <= 1:
        return
    _bucketed_all_reduce(
        (p.grad for p in params if p.grad is not None),
        dist.ReduceOp.AVG,
        world.ci_pool_group,
        "all_reduce_ci_fn_grads",
    )


def sum_reduce_ppgd_grads(world: World, grads: Iterable[Tensor]) -> None:
    """PPGD in-pool SUM-reduce on V/U grads. No-op for 1-rank pool."""
    if dist.get_world_size(world.ppgd_pool_group) <= 1:
        return
    _bucketed_all_reduce(grads, dist.ReduceOp.SUM, world.ppgd_pool_group, "sum_reduce_ppgd_grads")


def all_reduce_grads_in_block(world: World, role: LWRole, params: Iterable[nn.Parameter]) -> None:
    """LW in-block DDP AVG-reduce over V/U + faithfulness grads (async buckets,
    wait + copy back). No-op when the block group is 1-rank or there are no grads."""
    block_group = world.block_group_groups[role.block_idx]
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
    with time_nccl_op("all_reduce_grads_in_block"):
        for bucket in buckets.values():
            flat = _flatten_dense_tensors(bucket)
            w = dist.all_reduce(flat, op=dist.ReduceOp.AVG, group=block_group, async_op=True)
            assert w is not None
            states.append((bucket, flat, w))
    for bucket, flat, w in states:
        w.wait()
        for orig, reduced in zip(bucket, _unflatten_dense_tensors(flat, bucket), strict=True):
            orig.copy_(reduced)


# ──────────────────────────────────────────────────────────────────────────────
# Per-pool portal bundles. A pool constructs only the portal halves at its own
# endpoints, so a step function can reach for exactly the exchanges it's allowed
# to perform and nothing else.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CIPortals:
    role: CIRole
    ci_to_lw: CiValuesToLayerwise
    ci_to_ppgd: CiValuesToPPGD
    g_ci_from_lw: GradCiFromLayerwise
    g_ci_from_ppgd: GradCiFromPPGD
    ci_eval_to_ppgd: CiOutputsEvalToPPGD


@dataclass(frozen=True)
class LWPortals:
    role: LWRole
    ci_from_ci_pool: CiValuesToLayerwise
    g_ci_to_ci_pool: GradCiFromLayerwise
    g_vu_from_ppgd: GradVuFromPPGD
    updated_vu_to_ppgd: UpdatedVuToPPGD


@dataclass(frozen=True)
class PPGDPortals:
    role: PPGDRole
    ci_from_ci_pool: CiValuesToPPGD
    g_ci_to_ci_pool: GradCiFromPPGD
    g_vu_to_lw: GradVuFromPPGD
    updated_vu_from_lw: UpdatedVuToPPGD
    ci_eval_from_ci_pool: CiOutputsEvalToPPGD


def build_ci_portals(world: World, role: CIRole) -> CIPortals:
    return CIPortals(
        role=role,
        ci_to_lw=CiValuesToLayerwise(world),
        ci_to_ppgd=CiValuesToPPGD(world),
        g_ci_from_lw=GradCiFromLayerwise(world),
        g_ci_from_ppgd=GradCiFromPPGD(world),
        ci_eval_to_ppgd=CiOutputsEvalToPPGD(world),
    )


def build_lw_portals(world: World, role: LWRole) -> LWPortals:
    return LWPortals(
        role=role,
        ci_from_ci_pool=CiValuesToLayerwise(world),
        g_ci_to_ci_pool=GradCiFromLayerwise(world),
        g_vu_from_ppgd=GradVuFromPPGD(world),
        updated_vu_to_ppgd=UpdatedVuToPPGD(world),
    )


def build_ppgd_portals(world: World, role: PPGDRole) -> PPGDPortals:
    return PPGDPortals(
        role=role,
        ci_from_ci_pool=CiValuesToPPGD(world),
        g_ci_to_ci_pool=GradCiFromPPGD(world),
        g_vu_to_lw=GradVuFromPPGD(world),
        updated_vu_from_lw=UpdatedVuToPPGD(world),
        ci_eval_from_ci_pool=CiOutputsEvalToPPGD(world),
    )
