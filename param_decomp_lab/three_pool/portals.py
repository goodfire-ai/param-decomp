"""Cross-pool exchanges as first-class typed objects — one per DAG edge.

The 3-pool program has six cross-pool point-to-point exchanges per step (plus
one eval-only). Previously each edge was split across two ``ThreePoolLayout``
methods (a send on the source pool, a recv on the dest pool) with the routing,
pack layout, and process group duplicated on both sides — free to drift.

Here each edge is ONE class. Its pack layout, routing bijection, wire dtype, and
process group live in a single place; the sender and receiver are the two halves
of the same class, so they cannot disagree. A pool only constructs the portal
halves at its own endpoints (see ``CIPortals`` / ``ChunkPortals`` / ``PPGDPortals``
below), so a chunkwise rank physically cannot invoke a CI-pool send.

The six edges (see ``DESIGN.md``):

  CI  → chunk : ``CiValuesToChunkwise``  (per-site, owned + chunk-rank slice)
  CI  → PPGD  : ``CiValuesToPPGD``       (full-model, per-PPGD-rank slice)
  chunk→ CI   : ``GradCiFromChunkwise``  (per-site, per-chunk-rank slice)
  PPGD→ CI    : ``GradCiFromPPGD``        (full-model, per-PPGD-rank slice)
  PPGD→ chunk : ``GradVuFromPPGD``        (per-site, after in-pool sum-reduce)
  chunk→ PPGD : ``UpdatedVuToPPGD``       (per-site, leader-rooted broadcast)

Eval-only:

  CI  → PPGD : ``CiOutputsEvalToPPGD``   (full CIOutputs, three dicts)

Each P2P portal carries the shared ``cross_pool_p2p_group``. The V/U broadcast
edge uses the per-chunk ``cross_pool_bcast_groups`` ({chunk leader} ∪ PPGD).

All routing math (which CI slice owns which chunk/PPGD shard, sub-slices within a
CI batch tensor) stays on ``World``; the portals call those helpers so there's
still a single source of truth for the topology.
"""

# pyright: reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from param_decomp.component_model import CIOutputs
from param_decomp_lab.three_pool.layout import (
    Chunk,
    World,
    time_nccl_op,
)
from param_decomp_lab.three_pool.role import ChunkRole, CIRole, PPGDRole

# Cross-pool tensors are cast to a 2-byte dtype on the wire (halves bytes vs fp32).
# The wire dtype is split by payload:
#
#   * CI VALUES (lower_leaky / upper_leaky) are bounded in ≈[0, 1] (leaky-hard
#     sigmoid). fp16's 10 mantissa bits give ~8× finer resolution near 1.0 than
#     bf16's 7; the exponent range bf16 buys is wasted on bounded data. So CI
#     values ship as fp16.
#   * GRADIENTS (CI grads, V/U grads) and the raw ``pre_sigmoid`` logit are
#     unbounded and can have large dynamic range; they keep bf16's exponent range
#     to avoid fp16 overflow. They accumulate into fp32 ``.grad`` and upcast on
#     receive — the standard bf16 mixed-precision pattern.
#
# V/U weights themselves (UpdatedVuToPPGD) are also unbounded → bf16.
CI_VALUE_WIRE_DTYPE: torch.dtype = torch.float16
CI_GRAD_WIRE_DTYPE: torch.dtype = torch.bfloat16
# Back-compat alias for the unbounded/grad/weight payloads that always used bf16.
WIRE_DTYPE: torch.dtype = CI_GRAD_WIRE_DTYPE


# ──────────────────────────────────────────────────────────────────────────────
# Pending receives — a posted irecv held until ``wait()`` materializes its payload.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _CiRecvPacket:
    """One in-flight CI-values irecv from a single CI rank, plus the sub-slice
    of the downstream rank's local batch tensor it fills."""

    work: "dist.Work"
    packed: Tensor
    overlap: slice
    overlap_len: int


@dataclass(frozen=True)
class PendingCiValues:
    """One or more coalesced CI-values irecvs, held until ``wait()``.

    Coarse-CI regime: a single packet covering this downstream rank's whole
    local batch. Fine-CI regime: ``fanout`` packets, each from a different CI
    rank, covering disjoint ``overlap`` sub-slices that tile the local batch.
    ``wait`` blocks on every packet then stitches them into per-site
    ``[b_down, seq_len, c_s]`` tensors.
    """

    packets: tuple[_CiRecvPacket, ...]
    sites: tuple[str, ...]
    site_to_c: dict[str, int]
    b_down: int
    seq_len: int
    device: torch.device

    def wait(self) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {
            s: torch.empty(
                self.b_down,
                self.seq_len,
                self.site_to_c[s],
                device=self.device,
                dtype=CI_VALUE_WIRE_DTYPE,
            )
            for s in self.sites
        }
        for packet in self.packets:
            packet.work.wait()
            offset = 0
            for s in self.sites:
                c_s = self.site_to_c[s]
                numel = packet.overlap_len * self.seq_len * c_s
                view = packet.packed[offset : offset + numel].view(
                    packet.overlap_len, self.seq_len, c_s
                )
                out[s][packet.overlap].copy_(view)
                offset += numel
            assert offset == packet.packed.numel(), (
                f"unpack size mismatch: consumed {offset} of {packet.packed.numel()}"
            )
        return out


@dataclass(frozen=True)
class PendingUpdatedVu:
    """Pipelined per-chunk broadcasts of updated V/U, held until ``wait()``."""

    bufs: tuple[tuple[Chunk, Tensor, "dist.Work"], ...]
    v_templates: dict[str, Tensor]
    u_templates: dict[str, Tensor]

    def wait(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        v_new: dict[str, Tensor] = {}
        u_new: dict[str, Tensor] = {}
        for chunk, packed, w in self.bufs:
            w.wait()
            offset = 0
            for s in chunk.sites:
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


def _pack_sites(
    d: dict[str, Tensor], sites: Iterable[str], sub: slice | None, dtype: torch.dtype
) -> Tensor:
    """Flatten ``d[site][sub]`` (or ``d[site]`` if ``sub`` is None) for each site,
    cast to ``dtype``, and concatenate into one contiguous buffer."""
    parts = [
        (d[s][sub] if sub is not None else d[s]).detach().to(dtype).contiguous().flatten()
        for s in sites
    ]
    return torch.cat(parts)


def _batch_p2p(
    group: dist.ProcessGroup, ops: list[tuple[Callable[..., "dist.Work"], Tensor, int]]
) -> list["dist.Work"]:
    """Issue all ``(op, tensor, peer)`` point-to-point ops as ONE NCCL group.

    ``op`` is ``dist.isend`` or ``dist.irecv``. Wrapping a call's concurrent cross-pool p2p
    in a single ``batch_isend_irecv`` lets NCCL satisfy the sends/recvs in any order — which is
    what prevents the ordering deadlock that serial ``isend``/``irecv`` hit when several large
    fanout transfers (e.g. CI→PPGD at seq 2048) must rendezvous on the shared communicator. A
    count/peer mismatch between the two sides still hangs (NCCL p2p has no tags) — that's guarded
    by ``tests/test_three_pool_portals.py``, not by this helper.
    """
    if not ops:
        return []
    return dist.batch_isend_irecv([dist.P2POp(op, t, peer, group) for op, t, peer in ops])


# ──────────────────────────────────────────────────────────────────────────────
# CI → chunkwise : per-site CI values (chunk sites, chunk-rank batch sub-slice)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CiValuesToChunkwise:
    world: World

    def send(self, role: CIRole, ci_full: dict[str, Tensor]) -> InFlightSends:
        """For each site and each chunk rank my CI slice overlaps, isend the
        overlapping sub-slice. ``ci_full`` is keyed by site (the CI fn is global)
        with values ``[B_local_ci, S, C_s]``.

        Coarse-CI: I overlap ``fanout`` whole chunk slices, each a sub-slice of my
        CI tensor. Fine-CI: I overlap exactly one chunk slice, sending my whole
        (smaller) CI tensor into the right offset of that chunk rank."""
        edge = self.world.ci_chunk_edge
        buffers: list[Tensor] = []
        ops: list[tuple[Callable[..., dist.Work], Tensor, int]] = []
        with time_nccl_op("CiValuesToChunkwise.send"):
            for chunk in self.world.chunks:
                for down_slice_idx in edge.down_slices_for_ci_slice(role.slice_idx):
                    target = chunk.ranks[down_slice_idx]
                    sub = edge.overlap_within_ci(role.slice_idx, down_slice_idx)
                    packed = _pack_sites(ci_full, chunk.sites, sub, CI_VALUE_WIRE_DTYPE)
                    buffers.append(packed)
                    ops.append((dist.isend, packed, target))
            works = _batch_p2p(self.world.cross_pool_p2p_group, ops)
        return InFlightSends(works=tuple(works), buffers=tuple(buffers))

    def post_recv(
        self, role: ChunkRole, site_to_c: dict[str, int], seq_len: int, device: torch.device
    ) -> PendingCiValues:
        """irecv CI values for this chunk rank's sites. Coarse-CI: one packet
        from the single CI rank whose slice contains my chunk shard. Fine-CI:
        ``fanout`` packets from the CI ranks nested in my chunk shard, each filling
        a disjoint sub-slice of my local batch."""
        edge = self.world.ci_chunk_edge
        b_chunk = self.world.batch_local_chunk
        recvs: list[tuple[Tensor, slice, int]] = []
        ops: list[tuple[Callable[..., dist.Work], Tensor, int]] = []
        with time_nccl_op("CiValuesToChunkwise.recv"):
            for ci_slice_idx in edge.ci_slices_for_down_slice(role.within_chunk_idx):
                src = self.world.ci_ranks[ci_slice_idx]
                overlap = edge.overlap_within_down(ci_slice_idx, role.within_chunk_idx)
                overlap_len = overlap.stop - overlap.start
                packed_numel = sum(overlap_len * seq_len * site_to_c[s] for s in role.sites)
                packed = torch.empty(packed_numel, device=device, dtype=CI_VALUE_WIRE_DTYPE)
                recvs.append((packed, overlap, overlap_len))
                ops.append((dist.irecv, packed, src))
            works = _batch_p2p(self.world.cross_pool_p2p_group, ops)
        packets = tuple(
            _CiRecvPacket(work=w, packed=p, overlap=ov, overlap_len=ol)
            for w, (p, ov, ol) in zip(works, recvs, strict=True)
        )
        return PendingCiValues(
            packets=packets,
            sites=role.sites,
            site_to_c=site_to_c,
            b_down=b_chunk,
            seq_len=seq_len,
            device=device,
        )


# ──────────────────────────────────────────────────────────────────────────────
# CI → PPGD : full-model CI values (all sites, PPGD-rank batch sub-slice)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CiValuesToPPGD:
    world: World

    def send(self, role: CIRole, ci_full: dict[str, Tensor]) -> InFlightSends:
        edge = self.world.ci_ppgd_edge
        buffers: list[Tensor] = []
        ops: list[tuple[Callable[..., dist.Work], Tensor, int]] = []
        with time_nccl_op("CiValuesToPPGD.send"):
            for ppgd_slice_idx in edge.down_slices_for_ci_slice(role.slice_idx):
                target = self.world.ppgd_ranks[ppgd_slice_idx]
                sub = edge.overlap_within_ci(role.slice_idx, ppgd_slice_idx)
                packed = _pack_sites(ci_full, self.world.all_sites, sub, CI_VALUE_WIRE_DTYPE)
                buffers.append(packed)
                ops.append((dist.isend, packed, target))
            works = _batch_p2p(self.world.cross_pool_p2p_group, ops)
        return InFlightSends(works=tuple(works), buffers=tuple(buffers))

    def post_recv(
        self, role: PPGDRole, site_to_c: dict[str, int], seq_len: int, device: torch.device
    ) -> PendingCiValues:
        edge = self.world.ci_ppgd_edge
        b_pp = self.world.batch_local_ppgd
        recvs: list[tuple[Tensor, slice, int]] = []
        ops: list[tuple[Callable[..., dist.Work], Tensor, int]] = []
        with time_nccl_op("CiValuesToPPGD.recv"):
            for ci_slice_idx in edge.ci_slices_for_down_slice(role.slice_idx):
                src = self.world.ci_ranks[ci_slice_idx]
                overlap = edge.overlap_within_down(ci_slice_idx, role.slice_idx)
                overlap_len = overlap.stop - overlap.start
                packed_numel = sum(
                    overlap_len * seq_len * site_to_c[s] for s in self.world.all_sites
                )
                packed = torch.empty(packed_numel, device=device, dtype=CI_VALUE_WIRE_DTYPE)
                recvs.append((packed, overlap, overlap_len))
                ops.append((dist.irecv, packed, src))
            works = _batch_p2p(self.world.cross_pool_p2p_group, ops)
        packets = tuple(
            _CiRecvPacket(work=w, packed=p, overlap=ov, overlap_len=ol)
            for w, (p, ov, ol) in zip(works, recvs, strict=True)
        )
        return PendingCiValues(
            packets=packets,
            sites=self.world.all_sites,
            site_to_c=site_to_c,
            b_down=b_pp,
            seq_len=seq_len,
            device=device,
        )


# ──────────────────────────────────────────────────────────────────────────────
# chunkwise → CI : per-site CI grads (per-chunk-rank batch slice; stitched on CI side)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GradCiFromChunkwise:
    world: World

    def send(self, role: ChunkRole, g_ci_owned: dict[str, Tensor]) -> None:
        """Send per-site CI grads to the CI rank(s) my chunk slice overlaps.

        Coarse-CI: one CI rank contains my whole slice — send all of it. Fine-CI:
        my slice spans ``fanout`` CI ranks — send each the overlapping sub-slice
        of my chunk grad. One coalesced send per destination."""
        edge = self.world.ci_chunk_edge
        buffers: list[Tensor] = []
        ops: list[tuple[Callable[..., dist.Work], Tensor, int]] = []
        with time_nccl_op("GradCiFromChunkwise.send"):
            for ci_slice_idx in edge.ci_slices_for_down_slice(role.within_chunk_idx):
                dst = self.world.ci_ranks[ci_slice_idx]
                sub = edge.overlap_within_down(ci_slice_idx, role.within_chunk_idx)
                packed = _pack_sites(g_ci_owned, role.sites, sub, WIRE_DTYPE)
                buffers.append(packed)
                ops.append((dist.isend, packed, dst))
            for w in _batch_p2p(self.world.cross_pool_p2p_group, ops):
                w.wait()

    def recv(
        self, role: CIRole, site_to_c: dict[str, int], seq_len: int, device: torch.device
    ) -> dict[str, Tensor]:
        """Recv per-site CI grads from the chunk rank(s) my CI slice overlaps,
        stitched into per-site ``[B_local_ci, S, C_s]`` fp32 dests. Coarse-CI:
        ``fanout`` chunk ranks tile my slice. Fine-CI: one chunk rank, my slice a
        sub-slice of it (I fill only my overlap)."""
        edge = self.world.ci_chunk_edge

        pending: list[tuple[slice, int, Tensor, tuple[str, ...]]] = []
        ops: list[tuple[Callable[..., dist.Work], Tensor, int]] = []
        with time_nccl_op("GradCiFromChunkwise.recv:post_irecvs"):
            for chunk in self.world.chunks:
                sites = chunk.sites
                for down_slice_idx in edge.down_slices_for_ci_slice(role.slice_idx):
                    src = chunk.ranks[down_slice_idx]
                    overlap = edge.overlap_within_ci(role.slice_idx, down_slice_idx)
                    overlap_len = overlap.stop - overlap.start
                    packed_numel = sum(overlap_len * seq_len * site_to_c[s] for s in sites)
                    buf = torch.empty(packed_numel, device=device, dtype=WIRE_DTYPE)
                    pending.append((overlap, overlap_len, buf, sites))
                    ops.append((dist.irecv, buf, src))
            works = _batch_p2p(self.world.cross_pool_p2p_group, ops)

        b_ci = self.world.batch_local_ci
        out: dict[str, Tensor] = {
            s: torch.empty(b_ci, seq_len, site_to_c[s], device=device, dtype=torch.float32)
            for s in self.world.all_sites
        }
        with time_nccl_op("GradCiFromChunkwise.recv:wait"):
            for (overlap, overlap_len, buf, sites), w in zip(pending, works, strict=True):
                w.wait()
                offset = 0
                for site in sites:
                    c_s = site_to_c[site]
                    n = overlap_len * seq_len * c_s
                    site_view = buf[offset : offset + n].view(overlap_len, seq_len, c_s)
                    out[site][overlap].copy_(site_view.to(torch.float32))
                    offset += n
        return out


# ──────────────────────────────────────────────────────────────────────────────
# PPGD → CI : full-model CI grads (per-PPGD-rank batch slice; stitched on CI side)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GradCiFromPPGD:
    world: World

    def send(self, role: PPGDRole, g_ci_full: dict[str, Tensor]) -> None:
        """Send full-model CI grads to the CI rank(s) my PPGD slice overlaps.

        Coarse-CI: one CI rank — send my whole slice. Fine-CI: ``fanout`` CI
        ranks — send each its overlapping sub-slice."""
        edge = self.world.ci_ppgd_edge
        buffers: list[Tensor] = []
        ops: list[tuple[Callable[..., dist.Work], Tensor, int]] = []
        with time_nccl_op("GradCiFromPPGD.send"):
            for ci_slice_idx in edge.ci_slices_for_down_slice(role.slice_idx):
                dst = self.world.ci_ranks[ci_slice_idx]
                sub = edge.overlap_within_down(ci_slice_idx, role.slice_idx)
                packed = _pack_sites(g_ci_full, self.world.all_sites, sub, WIRE_DTYPE)
                buffers.append(packed)
                ops.append((dist.isend, packed, dst))
            for w in _batch_p2p(self.world.cross_pool_p2p_group, ops):
                w.wait()

    def recv(
        self, role: CIRole, site_to_c: dict[str, int], seq_len: int, device: torch.device
    ) -> dict[str, Tensor]:
        edge = self.world.ci_ppgd_edge

        pending: list[tuple[slice, int, Tensor]] = []
        ops: list[tuple[Callable[..., dist.Work], Tensor, int]] = []
        with time_nccl_op("GradCiFromPPGD.recv:post_irecvs"):
            for ppgd_slice_idx in edge.down_slices_for_ci_slice(role.slice_idx):
                src = self.world.ppgd_ranks[ppgd_slice_idx]
                overlap = edge.overlap_within_ci(role.slice_idx, ppgd_slice_idx)
                overlap_len = overlap.stop - overlap.start
                packed_numel = sum(
                    overlap_len * seq_len * site_to_c[s] for s in self.world.all_sites
                )
                packed = torch.empty(packed_numel, device=device, dtype=WIRE_DTYPE)
                pending.append((overlap, overlap_len, packed))
                ops.append((dist.irecv, packed, src))
            works = _batch_p2p(self.world.cross_pool_p2p_group, ops)

        b_ci = self.world.batch_local_ci
        out: dict[str, Tensor] = {
            s: torch.empty(b_ci, seq_len, site_to_c[s], device=device, dtype=torch.float32)
            for s in self.world.all_sites
        }
        with time_nccl_op("GradCiFromPPGD.recv:wait"):
            for (overlap, overlap_len, packed), w in zip(pending, works, strict=True):
                w.wait()
                offset = 0
                for site in self.world.all_sites:
                    n = overlap_len * seq_len * site_to_c[site]
                    buf = packed[offset : offset + n].view(overlap_len, seq_len, site_to_c[site])
                    out[site][overlap].copy_(buf.to(torch.float32))
                    offset += n
        return out


# ──────────────────────────────────────────────────────────────────────────────
# PPGD → chunkwise : per-site V/U grads (after PPGD in-pool sum-reduce)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GradVuFromPPGD:
    world: World

    def send(self, role: PPGDRole, v_grads: dict[str, Tensor], u_grads: dict[str, Tensor]) -> None:
        """PPGD-leader-only: send g_VU per-chunk (coalesced) to each chunk leader.
        Assumes V/U grads were already sum-reduced within the PPGD pool, so every
        PPGD rank holds the same values and only the leader sends."""
        if not role.is_pool_leader:
            return
        buffers: list[Tensor] = []
        ops: list[tuple[Callable[..., dist.Work], Tensor, int]] = []
        with time_nccl_op("GradVuFromPPGD.send:isends"):
            for chunk in self.world.chunks:
                parts: list[Tensor] = []
                for site in chunk.sites:
                    parts.append(v_grads[site].to(WIRE_DTYPE).contiguous().flatten())
                    parts.append(u_grads[site].to(WIRE_DTYPE).contiguous().flatten())
                packed = torch.cat(parts)
                buffers.append(packed)
                ops.append((dist.isend, packed, chunk.leader))
            works = _batch_p2p(self.world.cross_pool_p2p_group, ops)
        with time_nccl_op("GradVuFromPPGD.send:wait"):
            for w in works:
                w.wait()
        del buffers

    def recv(
        self, role: ChunkRole, v_templates: dict[str, Tensor], u_templates: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Chunk leader recvs g_VU for its sites from PPGD leader; non-leaders
        get nothing.

        Contribute-once (see ``SUM_GRAD_CONVENTION.md``): PPGD's grad is identical
        across chunk replicas, so under the chunk SUM-reduce it must land on
        exactly ONE rank. We add it to the leader's ``.grad`` only and skip the
        old in-chunk broadcast — the SUM then distributes it to every replica
        exactly once. Non-leaders return empty dicts and add nothing.
        """
        if not role.is_chunk_leader:
            return {}, {}

        my_sites = role.sites
        packed_numel = sum(v_templates[s].numel() + u_templates[s].numel() for s in my_sites)
        sample = v_templates[my_sites[0]]
        packed = torch.empty(packed_numel, dtype=WIRE_DTYPE, device=sample.device)
        ppgd_leader = self.world.ppgd_ranks[0]
        with time_nccl_op("GradVuFromPPGD.recv:recv"):
            dist.recv(packed, src=ppgd_leader, group=self.world.cross_pool_p2p_group)
        v_grads: dict[str, Tensor] = {}
        u_grads: dict[str, Tensor] = {}
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
        return v_grads, u_grads


# ──────────────────────────────────────────────────────────────────────────────
# chunkwise → PPGD : updated V/U (leader-rooted broadcast over {leader} ∪ PPGD)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class UpdatedVuToPPGD:
    world: World

    def send(
        self, role: ChunkRole, v_owned: dict[str, Tensor], u_owned: dict[str, Tensor]
    ) -> InFlightSends:
        """Coalesced leader-rooted broadcast of updated V/U to all PPGD ranks.
        Only the chunk leader sends; others no-op. Caller keeps the buffer alive
        until the work completes."""
        if not role.is_chunk_leader:
            return InFlightSends(works=(), buffers=())
        parts: list[Tensor] = []
        for s in role.sites:
            parts.append(v_owned[s].detach().to(WIRE_DTYPE).contiguous().flatten())
            parts.append(u_owned[s].detach().to(WIRE_DTYPE).contiguous().flatten())
        packed = torch.cat(parts)
        bcast_group = self.world.cross_pool_bcast_groups[role.chunk_idx]
        with time_nccl_op("UpdatedVuToPPGD.send"):
            w = dist.broadcast(packed, src=role.rank, group=bcast_group, async_op=True)
        assert w is not None
        return InFlightSends(works=(w,), buffers=(packed,))

    def post_recv(
        self, v_templates: dict[str, Tensor], u_templates: dict[str, Tensor]
    ) -> PendingUpdatedVu:
        """Kick off one async broadcast per chunk (they pipeline across the
        per-chunk NCCL streams); ``wait`` unpacks each into per-site V/U."""
        bufs: list[tuple[Chunk, Tensor, dist.Work]] = []
        with time_nccl_op("UpdatedVuToPPGD.recv"):
            for chunk_idx, chunk in enumerate(self.world.chunks):
                sites = chunk.sites
                packed_numel = sum(v_templates[s].numel() + u_templates[s].numel() for s in sites)
                sample = v_templates[sites[0]]
                packed = torch.empty(packed_numel, dtype=WIRE_DTYPE, device=sample.device)
                bcast_group = self.world.cross_pool_bcast_groups[chunk_idx]
                w = dist.broadcast(packed, src=chunk.leader, group=bcast_group, async_op=True)
                assert w is not None
                bufs.append((chunk, packed, w))
        return PendingUpdatedVu(bufs=tuple(bufs), v_templates=v_templates, u_templates=u_templates)


# ──────────────────────────────────────────────────────────────────────────────
# CI → PPGD eval : full CIOutputs (lower_leaky, upper_leaky, pre_sigmoid)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CiOutputsEvalToPPGD:
    world: World

    def send(self, role: CIRole, ci: CIOutputs) -> None:
        """Synchronous send of full CIOutputs (all three dicts) to the PPGD
        rank(s) my CI slice overlaps. Eval is rare; overlap has no value here.

        Pack layout (must match ``recv``): three contiguous blocks in order
        (lower_leaky, upper_leaky, pre_sigmoid), each carrying the overlapping
        sub-slice of my CI tensor for the destination PPGD rank."""
        edge = self.world.ci_ppgd_edge
        buffers: list[Tensor] = []
        ops: list[tuple[Callable[..., dist.Work], Tensor, int]] = []
        with time_nccl_op("CiOutputsEvalToPPGD.send"):
            for ppgd_slice_idx in edge.down_slices_for_ci_slice(role.slice_idx):
                target = self.world.ppgd_ranks[ppgd_slice_idx]
                sub = edge.overlap_within_ci(role.slice_idx, ppgd_slice_idx)
                # Eval-only (rare): pack all three dicts into one buffer. Because this
                # bundles the unbounded ``pre_sigmoid`` logit alongside the bounded
                # lower/upper masks, the whole packet stays bf16 (WIRE_DTYPE) rather
                # than splitting the buffer by dtype — not worth the pack/unpack tangle
                # for an off-critical-path send. The per-step value sends
                # (CiValuesToChunkwise/PPGD) DO use fp16.
                parts: list[Tensor] = []
                for d in (ci.lower_leaky, ci.upper_leaky, ci.pre_sigmoid):
                    parts.append(_pack_sites(d, self.world.all_sites, sub, WIRE_DTYPE))
                packed = torch.cat(parts)
                buffers.append(packed)
                ops.append((dist.isend, packed, target))
            for w in _batch_p2p(self.world.cross_pool_p2p_group, ops):
                w.wait()

    def recv(
        self, role: PPGDRole, site_to_c: dict[str, int], seq_len: int, device: torch.device
    ) -> CIOutputs:
        """Recv full CIOutputs from the CI rank(s) my PPGD slice overlaps,
        stitched into per-site ``[B_local_ppgd, S, C_s]`` tensors (three dicts)."""
        edge = self.world.ci_ppgd_edge
        b_pp = self.world.batch_local_ppgd
        out: list[dict[str, Tensor]] = [
            {
                s: torch.empty(b_pp, seq_len, site_to_c[s], device=device, dtype=WIRE_DTYPE)
                for s in self.world.all_sites
            }
            for _ in range(3)
        ]
        recvs: list[tuple[Tensor, slice, int]] = []
        ops: list[tuple[Callable[..., dist.Work], Tensor, int]] = []
        with time_nccl_op("CiOutputsEvalToPPGD.recv"):
            for ci_slice_idx in edge.ci_slices_for_down_slice(role.slice_idx):
                src = self.world.ci_ranks[ci_slice_idx]
                overlap = edge.overlap_within_down(ci_slice_idx, role.slice_idx)
                overlap_len = overlap.stop - overlap.start
                per_block_numel = sum(
                    overlap_len * seq_len * site_to_c[s] for s in self.world.all_sites
                )
                packed = torch.empty(3 * per_block_numel, device=device, dtype=WIRE_DTYPE)
                recvs.append((packed, overlap, overlap_len))
                ops.append((dist.irecv, packed, src))
            works = _batch_p2p(self.world.cross_pool_p2p_group, ops)
            for (packed, overlap, overlap_len), w in zip(recvs, works, strict=True):
                w.wait()
                offset = 0
                for block_idx in range(3):
                    for site in self.world.all_sites:
                        c_s = site_to_c[site]
                        numel = overlap_len * seq_len * c_s
                        view = packed[offset : offset + numel].view(overlap_len, seq_len, c_s)
                        out[block_idx][site][overlap].copy_(view)
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

    with time_nccl_op(label):
        for bucket in buckets.values():
            flat = _flatten_dense_tensors(bucket)
            dist.all_reduce(flat, op=op, group=group)
            for orig, reduced in zip(bucket, _unflatten_dense_tensors(flat, bucket), strict=True):
                orig.copy_(reduced)


@dataclass(frozen=True)
class InFlightCiGradReduce:
    """An async CI in-pool SUM-reduce on CI fn grads, held until ``wait()``.

    Mirrors ``all_reduce_grads_in_chunk``'s async bucketed pattern: the reduce
    is kicked off (``async_op=True``) right after the fused backward, then the
    flat buffers are reduced in-flight while the CI rank does non-dependent work
    (the dead-time H_{T+1} prefetch). ``wait()`` blocks on every bucket and
    copies the reduced result back into each grad — it MUST be called before the
    first consumer of the reduced grads (grad-clip / ``optimizer.step``).

    A no-op (empty ``buckets``) is returned when the CI pool is 1-rank or there
    are no grads, so ``wait()`` is always safe to call.
    """

    buckets: tuple[tuple[list[Tensor], Tensor, "dist.Work"], ...]

    def wait(self) -> None:
        for bucket, flat, w in self.buckets:
            w.wait()
            for orig, reduced in zip(bucket, _unflatten_dense_tensors(flat, bucket), strict=True):
                orig.copy_(reduced)


def all_reduce_ci_fn_grads_async(
    world: World, params: Iterable[nn.Parameter]
) -> InFlightCiGradReduce:
    """Kick off the CI in-pool SUM-reduce of the CI fn grads async, returning the
    in-flight handle; the caller ``wait()``s before the first grad consumer.

    SUM, not AVG (see ``SUM_GRAD_CONVENTION.md``): each producer's CI grad is a
    partial sum already normalized by the honest global count, so the cross-rank
    SUM reassembles the single-pool total directly. No producer pre-scales by
    ``n_ci`` to survive this reduce. Same reduction as a blocking all-reduce —
    bit-identical — just non-blocking.
    """
    if dist.get_world_size(world.ci_pool_group) <= 1:
        return InFlightCiGradReduce(buckets=())
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return InFlightCiGradReduce(buckets=())
    grad_buckets: dict[tuple[torch.dtype, torch.device], list[Tensor]] = {}
    for g in grads:
        grad_buckets.setdefault((g.dtype, g.device), []).append(g)

    states: list[tuple[list[Tensor], Tensor, dist.Work]] = []
    with time_nccl_op("all_reduce_ci_fn_grads"):
        for bucket in grad_buckets.values():
            flat = _flatten_dense_tensors(bucket)
            w = dist.all_reduce(
                flat, op=dist.ReduceOp.SUM, group=world.ci_pool_group, async_op=True
            )
            assert w is not None
            states.append((bucket, flat, w))
    return InFlightCiGradReduce(buckets=tuple(states))


def sum_reduce_ppgd_grads(world: World, grads: Iterable[Tensor]) -> None:
    """PPGD in-pool SUM-reduce on V/U grads. No-op for 1-rank pool."""
    if dist.get_world_size(world.ppgd_pool_group) <= 1:
        return
    _bucketed_all_reduce(grads, dist.ReduceOp.SUM, world.ppgd_pool_group, "sum_reduce_ppgd_grads")


def all_reduce_grads_in_chunk(
    world: World, role: ChunkRole, params: Iterable[nn.Parameter]
) -> None:
    """Chunkwise in-chunk SUM-reduce over V/U grads (async buckets, wait + copy back).
    No-op when the chunk is 1-rank or there are no grads.

    SUM, not AVG (see ``SUM_GRAD_CONVENTION.md``): the per-rank stoch grad is a
    partial sum over a disjoint position slice, normalized by the honest global
    count, so the cross-rank SUM reassembles the single-pool total. The
    REPLICATED contributions (faith, broadcast PPGD grad) are emitted on the
    chunk leader ONLY — contribute-once — so they survive the SUM exactly once
    without any ``chunk_dp`` pre-scaling.
    """
    chunk_group = world.chunk_groups[role.chunk_idx]
    if dist.get_world_size(chunk_group) <= 1:
        return
    grads: list[Tensor] = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    buckets: dict[tuple[torch.dtype, torch.device], list[Tensor]] = {}
    for g in grads:
        buckets.setdefault((g.dtype, g.device), []).append(g)

    states: list[tuple[list[Tensor], Tensor, dist.Work]] = []
    with time_nccl_op("all_reduce_grads_in_chunk"):
        for bucket in buckets.values():
            flat = _flatten_dense_tensors(bucket)
            w = dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=chunk_group, async_op=True)
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
    ci_to_chunk: CiValuesToChunkwise
    ci_to_ppgd: CiValuesToPPGD
    g_ci_from_chunk: GradCiFromChunkwise
    g_ci_from_ppgd: GradCiFromPPGD
    ci_eval_to_ppgd: CiOutputsEvalToPPGD


@dataclass(frozen=True)
class ChunkPortals:
    role: ChunkRole
    ci_from_ci_pool: CiValuesToChunkwise
    g_ci_to_ci_pool: GradCiFromChunkwise
    g_vu_from_ppgd: GradVuFromPPGD
    updated_vu_to_ppgd: UpdatedVuToPPGD


@dataclass(frozen=True)
class PPGDPortals:
    role: PPGDRole
    ci_from_ci_pool: CiValuesToPPGD
    g_ci_to_ci_pool: GradCiFromPPGD
    g_vu_to_chunk: GradVuFromPPGD
    updated_vu_from_chunk: UpdatedVuToPPGD
    ci_eval_from_ci_pool: CiOutputsEvalToPPGD


def build_ci_portals(world: World, role: CIRole) -> CIPortals:
    return CIPortals(
        role=role,
        ci_to_chunk=CiValuesToChunkwise(world),
        ci_to_ppgd=CiValuesToPPGD(world),
        g_ci_from_chunk=GradCiFromChunkwise(world),
        g_ci_from_ppgd=GradCiFromPPGD(world),
        ci_eval_to_ppgd=CiOutputsEvalToPPGD(world),
    )


def build_chunk_portals(world: World, role: ChunkRole) -> ChunkPortals:
    return ChunkPortals(
        role=role,
        ci_from_ci_pool=CiValuesToChunkwise(world),
        g_ci_to_ci_pool=GradCiFromChunkwise(world),
        g_vu_from_ppgd=GradVuFromPPGD(world),
        updated_vu_to_ppgd=UpdatedVuToPPGD(world),
    )


def build_ppgd_portals(world: World, role: PPGDRole) -> PPGDPortals:
    return PPGDPortals(
        role=role,
        ci_from_ci_pool=CiValuesToPPGD(world),
        g_ci_to_ci_pool=GradCiFromPPGD(world),
        g_vu_to_chunk=GradVuFromPPGD(world),
        updated_vu_from_chunk=UpdatedVuToPPGD(world),
        ci_eval_from_ci_pool=CiOutputsEvalToPPGD(world),
    )
