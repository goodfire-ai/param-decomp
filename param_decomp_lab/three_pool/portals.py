"""Cross-pool exchanges as first-class typed portal objects.

Each of the six per-step cross-pool edges in the 3-pool dependency graph
(see ``DESIGN.md``) is defined here exactly ONCE. A portal owns everything
the edge needs — its payload shape, source/dest pool, the batch-position
routing (the ``lw_sub_slice_within_ci`` / ``ci_slice_of_*`` bijection), its
process group, its pack/unpack, and its bf16 wire dtype. Both the sending
rank and the receiving rank construct the SAME portal object (from the
shared ``World``) and invoke it from their respective sides:

    handle = portal.send(payload)   # sender side; later handle.wait()
    pending = portal.post_recv(...) # receiver side; later pending.wait() -> T

Because send and recv live on one object, the two sides' pack/unpack layout
cannot drift — the previous design split each edge across ``layout.py``
(sender) and the receiving step file, with the pack format duplicated in a
docstring on each side.

The six edges (sender → receiver):

  ``CiValuesToLayerwise``     CI → LW   : CI_T per-site (owned + LW-rank slice)
  ``CiValuesToPpgd``          CI → PPGD : CI_T full-model (per-PPGD-rank slice)
  ``GradCiFromLayerwise``     LW → CI   : g_CI_LW per owned site (per-LW-rank slice)
  ``GradCiFromPpgd``          PPGD → CI : g_CI_PPGD full-model (per-PPGD-rank slice)
  ``GradVuFromPpgd``          PPGD → LW : g_VU_PPGD per owned site (post in-pool reduce)
  ``UpdatedVuToPpgd``         LW → PPGD : updated V/U per owned site (leader broadcast)

Plus the eval-only ``CiOutputsToPpgd`` (CI → PPGD; full ``CIOutputs``).

The three in-pool collective reductions (CI-fn-grad all-reduce, LW in-block
all-reduce, PPGD V/U sum-reduce) are NOT cross-pool edges; they stay as
methods on ``ThreePoolLayout``.

All cross-pool tensors are cast to ``_WIRE_DTYPE`` (bf16) on the wire — half
the bytes vs fp32. Downstream pools run inside bf16 autocast; received grads
upcast to fp32 on receive (standard mixed-precision pattern).
"""

# pyright: reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor

from param_decomp.component_model import CIOutputs
from param_decomp_lab.three_pool.layout import World, _time_nccl_op

# All cross-pool tensors are cast to this dtype on the wire (halves bytes vs fp32).
_WIRE_DTYPE: torch.dtype = torch.bfloat16


# ──────────────────────────────────────────────────────────────────────────────
# Handles — typed deferral wrappers returned by portal send/recv.
#
# A ``SendHandle`` keeps the packed send buffers alive until ``wait()``; a
# ``Pending[T]`` blocks on its work then unpacks the wire buffer into the
# portal's typed payload. Receiver code cannot reach the payload without
# calling ``wait()``, so "use before the transfer completes" is unrepresentable.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SendHandle:
    """In-flight cross-pool sends + the buffers backing them.

    The buffers must stay referenced until every send completes, so they ride
    along on the handle. ``wait()`` blocks on all of them; a handle with no
    works (e.g. a non-leader rank that sends nothing) is a no-op wait.
    """

    works: list["dist.Work"]
    buffers: list[Tensor]

    def wait(self) -> None:
        for w in self.works:
            w.wait()


@dataclass(frozen=True)
class PendingPerSiteCi:
    """One coalesced per-site CI-values irecv, held until ``wait()``.

    The packed buffer carries ``sites`` worth of CI values (in order) as
    ``b * seq_len * c_s`` ``_WIRE_DTYPE`` elements each. ``wait`` blocks on the
    underlying ``dist.Work`` then materializes per-site ``[b, seq_len, c_s]``
    views into the packed buffer (no copy).
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


# ──────────────────────────────────────────────────────────────────────────────
# Edge 1: CI → LW. Per-site CI values, sub-sliced to each LW rank's batch shard.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CiValuesToLayerwise:
    """CI → LW per-site CI values.

    Sender (CI rank): for each LW block + each LW rank whose batch shard sits
    in my CI slice, isend that block's owned-sites packet sub-sliced to the
    LW rank. Receiver (LW rank): irecv one coalesced packet for my owned sites
    from the CI rank that owns my batch shard.

    Pack layout (one packet per (block, block-rank)): for each site in the
    block's owned-sites order, ``b_lw * seq_len * C_s`` contiguous
    ``_WIRE_DTYPE`` elements.
    """

    world: World

    def send(self, ci_full: dict[str, Tensor], *, my_ci_slice_idx: int) -> SendHandle:
        """``ci_full`` keyed by site, shape ``[B_local_ci, S, C_s]`` (CI fn is
        global so it has every site). Returns a handle held alive until wait."""
        w = self.world
        works: list[dist.Work] = []
        buffers: list[Tensor] = []
        my_lw_block_ranks = w.lw_block_ranks_for_ci_slice(my_ci_slice_idx)
        with _time_nccl_op("CiValuesToLayerwise.send"):
            for bg in w.layerwise_block_groups:
                for block_rank_idx in my_lw_block_ranks:
                    target = bg.ranks[block_rank_idx]
                    sub = w.lw_sub_slice_within_ci(block_rank_idx)
                    parts = [
                        ci_full[site][sub].detach().to(_WIRE_DTYPE).contiguous().flatten()
                        for site in bg.owned_sites
                    ]
                    packed = torch.cat(parts)
                    works.append(dist.isend(packed, dst=target, group=w.cross_pool_p2p_group))
                    buffers.append(packed)
        return SendHandle(works=works, buffers=buffers)

    def post_recv(
        self,
        *,
        my_within_block_idx: int,
        my_owned_sites: tuple[str, ...],
        site_to_c: dict[str, int],
        seq_len: int,
        device: torch.device,
    ) -> PendingPerSiteCi:
        w = self.world
        src_ci_slice = w.ci_slice_of_lw_block_rank(my_within_block_idx)
        src = w.ci_ranks[src_ci_slice]
        b_lw = w.batch_local_lw
        packed_numel = sum(b_lw * seq_len * site_to_c[s] for s in my_owned_sites)
        packed = torch.empty(packed_numel, device=device, dtype=_WIRE_DTYPE)
        with _time_nccl_op("CiValuesToLayerwise.post_recv"):
            work = dist.irecv(packed, src=src, group=w.cross_pool_p2p_group)
            assert work is not None
        return PendingPerSiteCi(
            packed=packed,
            work=work,
            sites=my_owned_sites,
            site_to_c=site_to_c,
            b=b_lw,
            seq_len=seq_len,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Edge 2: CI → PPGD. Full-model CI values, sub-sliced to each PPGD rank.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CiValuesToPpgd:
    """CI → PPGD full-model CI values.

    Sender (CI rank): for each PPGD rank whose batch shard sits in my CI slice,
    isend one packet of all sites sub-sliced to that PPGD rank. Receiver (PPGD
    rank): irecv one coalesced full-model packet from the CI rank that owns my
    batch shard.

    Pack layout: for each site in ``world.all_sites`` order, ``b_pp * seq_len *
    C_s`` contiguous ``_WIRE_DTYPE`` elements.
    """

    world: World

    def send(self, ci_full: dict[str, Tensor], *, my_ci_slice_idx: int) -> SendHandle:
        w = self.world
        works: list[dist.Work] = []
        buffers: list[Tensor] = []
        my_ppgd_slice_idxs = w.ppgd_slice_idxs_for_ci_slice(my_ci_slice_idx)
        with _time_nccl_op("CiValuesToPpgd.send"):
            for ppgd_slice_idx in my_ppgd_slice_idxs:
                target = w.ppgd_ranks[ppgd_slice_idx]
                sub = w.ppgd_sub_slice_within_ci(ppgd_slice_idx)
                parts = [
                    ci_full[site][sub].detach().to(_WIRE_DTYPE).contiguous().flatten()
                    for site in w.all_sites
                ]
                packed = torch.cat(parts)
                works.append(dist.isend(packed, dst=target, group=w.cross_pool_p2p_group))
                buffers.append(packed)
        return SendHandle(works=works, buffers=buffers)

    def post_recv(
        self,
        *,
        my_ppgd_slice_idx: int,
        site_to_c: dict[str, int],
        seq_len: int,
        device: torch.device,
    ) -> PendingPerSiteCi:
        w = self.world
        src_ci_slice = w.ci_slice_of_ppgd_slice(my_ppgd_slice_idx)
        src = w.ci_ranks[src_ci_slice]
        b_pp = w.batch_local_ppgd
        packed_numel = sum(b_pp * seq_len * site_to_c[s] for s in w.all_sites)
        packed = torch.empty(packed_numel, device=device, dtype=_WIRE_DTYPE)
        with _time_nccl_op("CiValuesToPpgd.post_recv"):
            work = dist.irecv(packed, src=src, group=w.cross_pool_p2p_group)
            assert work is not None
        return PendingPerSiteCi(
            packed=packed,
            work=work,
            sites=w.all_sites,
            site_to_c=site_to_c,
            b=b_pp,
            seq_len=seq_len,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Edge 3: LW → CI. Per-owned-site CI grads, stitched into the CI rank's slice.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GradCiFromLayerwise:
    """LW → CI per-owned-site CI grads.

    Sender (LW rank): coalesce my owned sites' grads (full LW batch slice) into
    one packed send to the CI rank that owns my slice. Receiver (CI rank): recv
    one packet per LW source whose batch shard sits in my CI slice and stitch
    each into a per-site fp32 ``[B_local_ci, S, C_s]`` dest.

    Pack layout (one packet per LW source): for each site in the source block's
    owned-sites order, ``b_lw * seq_len * c_s`` contiguous ``_WIRE_DTYPE`` elements.
    """

    world: World

    def send(
        self,
        g_ci_owned: dict[str, Tensor],
        *,
        my_within_block_idx: int,
        my_owned_sites: tuple[str, ...],
    ) -> None:
        w = self.world
        dst_ci_slice = w.ci_slice_of_lw_block_rank(my_within_block_idx)
        dst = w.ci_ranks[dst_ci_slice]
        parts = [
            g_ci_owned[s].detach().to(_WIRE_DTYPE).contiguous().flatten() for s in my_owned_sites
        ]
        packed = torch.cat(parts)
        with _time_nccl_op("GradCiFromLayerwise.send"):
            dist.send(packed, dst=dst, group=w.cross_pool_p2p_group)

    def recv(
        self,
        *,
        my_ci_slice_idx: int,
        site_to_c: dict[str, int],
        seq_len: int,
        device: torch.device,
    ) -> dict[str, Tensor]:
        w = self.world
        my_lw_block_ranks = w.lw_block_ranks_for_ci_slice(my_ci_slice_idx)
        b_lw = w.batch_local_lw

        pending: list[tuple[int, Tensor, dist.Work, tuple[str, ...]]] = []
        with _time_nccl_op("GradCiFromLayerwise.recv:post_irecvs"):
            for bg in w.layerwise_block_groups:
                owned = bg.owned_sites
                packed_numel = sum(b_lw * seq_len * site_to_c[s] for s in owned)
                for block_rank_idx in my_lw_block_ranks:
                    src = bg.ranks[block_rank_idx]
                    buf = torch.empty(packed_numel, device=device, dtype=_WIRE_DTYPE)
                    work = dist.irecv(buf, src=src, group=w.cross_pool_p2p_group)
                    assert work is not None
                    pending.append((block_rank_idx, buf, work, owned))

        b_ci = w.batch_local_ci
        out: dict[str, Tensor] = {
            s: torch.empty(b_ci, seq_len, site_to_c[s], device=device, dtype=torch.float32)
            for s in w.all_sites
        }
        with _time_nccl_op("GradCiFromLayerwise.recv:wait"):
            for block_rank_idx, buf, work, owned in pending:
                work.wait()
                sub = w.lw_sub_slice_within_ci(block_rank_idx)
                offset = 0
                for site in owned:
                    c_s = site_to_c[site]
                    n = b_lw * seq_len * c_s
                    site_view = buf[offset : offset + n].view(b_lw, seq_len, c_s)
                    out[site][sub].copy_(site_view.to(torch.float32))
                    offset += n
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Edge 4: PPGD → CI. Full-model CI grads, stitched into the CI rank's slice.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GradCiFromPpgd:
    """PPGD → CI full-model CI grads.

    Sender (PPGD rank): coalesce all sites' grads (PPGD batch slice) into one
    packed send to the CI rank that owns my slice. Receiver (CI rank): recv one
    packet per PPGD source in my CI slice and stitch each into a per-site fp32
    ``[B_local_ci, S, C_s]`` dest.

    Pack layout: for each site in ``world.all_sites`` order, ``b_pp * seq_len *
    c_s`` contiguous ``_WIRE_DTYPE`` elements.
    """

    world: World

    def send(self, g_ci_full: dict[str, Tensor], *, my_ppgd_slice_idx: int) -> None:
        w = self.world
        dst_ci_slice = w.ci_slice_of_ppgd_slice(my_ppgd_slice_idx)
        dst = w.ci_ranks[dst_ci_slice]
        parts = [g_ci_full[s].detach().to(_WIRE_DTYPE).contiguous().flatten() for s in w.all_sites]
        packed = torch.cat(parts)
        with _time_nccl_op("GradCiFromPpgd.send"):
            dist.send(packed, dst=dst, group=w.cross_pool_p2p_group)

    def recv(
        self,
        *,
        my_ci_slice_idx: int,
        site_to_c: dict[str, int],
        seq_len: int,
        device: torch.device,
    ) -> dict[str, Tensor]:
        w = self.world
        my_ppgd_slice_idxs = w.ppgd_slice_idxs_for_ci_slice(my_ci_slice_idx)
        b_pp = w.batch_local_ppgd

        site_numels = {s: b_pp * seq_len * site_to_c[s] for s in w.all_sites}
        packed_numel = sum(site_numels.values())

        pending: list[tuple[int, Tensor, dist.Work]] = []
        with _time_nccl_op("GradCiFromPpgd.recv:post_irecvs"):
            for ppgd_slice_idx in my_ppgd_slice_idxs:
                src = w.ppgd_ranks[ppgd_slice_idx]
                packed = torch.empty(packed_numel, device=device, dtype=_WIRE_DTYPE)
                work = dist.irecv(packed, src=src, group=w.cross_pool_p2p_group)
                assert work is not None
                pending.append((ppgd_slice_idx, packed, work))

        b_ci = w.batch_local_ci
        out: dict[str, Tensor] = {
            s: torch.empty(b_ci, seq_len, site_to_c[s], device=device, dtype=torch.float32)
            for s in w.all_sites
        }
        with _time_nccl_op("GradCiFromPpgd.recv:wait"):
            for ppgd_slice_idx, packed, work in pending:
                work.wait()
                sub = w.ppgd_sub_slice_within_ci(ppgd_slice_idx)
                offset = 0
                for site in w.all_sites:
                    c_s = site_to_c[site]
                    n = site_numels[site]
                    buf = packed[offset : offset + n].view(b_pp, seq_len, c_s)
                    out[site][sub].copy_(buf.to(torch.float32))
                    offset += n
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Edge 5: PPGD → LW. Per-owned-site V/U grads (post in-pool sum-reduce).
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GradVuFromPpgd:
    """PPGD → LW per-owned-site V/U grads.

    Sender (PPGD leader): one coalesced isend per LW block to its leader (V/U
    grads already sum-reduced within the PPGD pool, so the leader's copy is the
    full-batch grad). Receiver (LW): block leader recvs its owned sites, then
    in-block broadcasts so every replica sees the same grad.

    Pack layout (per LW block): for each site in the block's owned-sites order,
    ``V.numel()`` then ``U.numel()`` contiguous ``_WIRE_DTYPE`` elements.
    """

    world: World

    def send(
        self, v_grads: dict[str, Tensor], u_grads: dict[str, Tensor], *, is_pool_leader: bool
    ) -> None:
        if not is_pool_leader:
            return
        w = self.world
        works: list[dist.Work] = []
        buffers: list[Tensor] = []
        with _time_nccl_op("GradVuFromPpgd.send:isends"):
            for bg in w.layerwise_block_groups:
                parts: list[Tensor] = []
                for site in bg.owned_sites:
                    parts.append(v_grads[site].to(_WIRE_DTYPE).contiguous().flatten())
                    parts.append(u_grads[site].to(_WIRE_DTYPE).contiguous().flatten())
                packed = torch.cat(parts)
                work = dist.isend(packed, dst=bg.leader, group=w.cross_pool_p2p_group)
                assert work is not None
                works.append(work)
                buffers.append(packed)
        with _time_nccl_op("GradVuFromPpgd.send:wait"):
            for work in works:
                work.wait()
        del buffers

    def recv(
        self,
        v_templates: dict[str, Tensor],
        u_templates: dict[str, Tensor],
        *,
        my_block_idx: int,
        my_owned_sites: tuple[str, ...],
        my_is_block_leader: bool,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        w = self.world
        v_grads: dict[str, Tensor] = {}
        u_grads: dict[str, Tensor] = {}

        if my_is_block_leader:
            packed_numel = sum(
                v_templates[s].numel() + u_templates[s].numel() for s in my_owned_sites
            )
            sample = v_templates[my_owned_sites[0]]
            packed = torch.empty(packed_numel, dtype=_WIRE_DTYPE, device=sample.device)
            ppgd_leader = w.ppgd_ranks[0]
            with _time_nccl_op("GradVuFromPpgd.recv:recv"):
                dist.recv(packed, src=ppgd_leader, group=w.cross_pool_p2p_group)
            offset = 0
            for s in my_owned_sites:
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
            for s in my_owned_sites:
                v_grads[s] = torch.empty_like(v_templates[s])
                u_grads[s] = torch.empty_like(u_templates[s])

        block_group = w.block_group_groups[my_block_idx]
        block_leader_rank = w.layerwise_block_groups[my_block_idx].leader
        with _time_nccl_op("GradVuFromPpgd.recv:in_block_bcast"):
            for s in my_owned_sites:
                v_grads[s] = v_grads[s].contiguous()
                u_grads[s] = u_grads[s].contiguous()
                dist.broadcast(v_grads[s], src=block_leader_rank, group=block_group)
                dist.broadcast(u_grads[s], src=block_leader_rank, group=block_group)

        return v_grads, u_grads


# ──────────────────────────────────────────────────────────────────────────────
# Edge 6: LW → PPGD. Updated V/U, leader-rooted broadcast to the PPGD pool.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class UpdatedVuToPpgd:
    """LW → PPGD updated V/U.

    Sender (LW block leader): one coalesced leader-rooted broadcast of updated
    V/U over the {block-leader} ∪ {ppgd_ranks} group. Receiver (PPGD): one async
    broadcast recv per LW block (they pipeline across the per-group NCCL
    streams), waited + unpacked into per-site V/U.

    Pack layout (per LW block): for each site in the block's owned-sites order,
    ``V.numel()`` then ``U.numel()`` contiguous ``_WIRE_DTYPE`` elements.
    """

    world: World

    def send(
        self,
        v_owned: dict[str, Tensor],
        u_owned: dict[str, Tensor],
        *,
        my_rank: int,
        my_block_idx: int,
        my_owned_sites: tuple[str, ...],
        my_is_block_leader: bool,
    ) -> SendHandle:
        if not my_is_block_leader:
            return SendHandle(works=[], buffers=[])
        w = self.world
        parts: list[Tensor] = []
        for s in my_owned_sites:
            parts.append(v_owned[s].detach().to(_WIRE_DTYPE).contiguous().flatten())
            parts.append(u_owned[s].detach().to(_WIRE_DTYPE).contiguous().flatten())
        packed = torch.cat(parts)
        bcast_group = w.cross_pool_bcast_groups[my_block_idx]
        with _time_nccl_op("UpdatedVuToPpgd.send"):
            work = dist.broadcast(packed, src=my_rank, group=bcast_group, async_op=True)
        assert work is not None
        return SendHandle(works=[work], buffers=[packed])

    def recv(
        self, v_templates: dict[str, Tensor], u_templates: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        w = self.world
        bufs: list[tuple[tuple[str, ...], Tensor, dist.Work]] = []
        with _time_nccl_op("UpdatedVuToPpgd.recv"):
            for bg_idx, bg in enumerate(w.layerwise_block_groups):
                owned = bg.owned_sites
                packed_numel = sum(v_templates[s].numel() + u_templates[s].numel() for s in owned)
                sample = v_templates[owned[0]]
                packed = torch.empty(packed_numel, dtype=_WIRE_DTYPE, device=sample.device)
                bcast_group = w.cross_pool_bcast_groups[bg_idx]
                work = dist.broadcast(packed, src=bg.leader, group=bcast_group, async_op=True)
                assert work is not None
                bufs.append((owned, packed, work))

        v_new: dict[str, Tensor] = {}
        u_new: dict[str, Tensor] = {}
        for owned, packed, work in bufs:
            work.wait()
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


# ──────────────────────────────────────────────────────────────────────────────
# Eval-only edge: CI → PPGD. Full CIOutputs (lower/upper/pre_sigmoid).
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CiOutputsToPpgd:
    """CI → PPGD full ``CIOutputs`` (eval only).

    Training ships only ``lower_leaky``; eval ships all three dicts so any metric
    reading ``ctx.ci`` works without a per-metric audit. Synchronous because eval
    is rare and overlap has no value.

    Pack layout per send: three contiguous blocks (lower_leaky, upper_leaky,
    pre_sigmoid). Each block has, for each site in ``world.all_sites`` order,
    ``b_pp * seq_len * C_s`` contiguous ``_WIRE_DTYPE`` elements.
    """

    world: World

    def send(self, ci: CIOutputs, *, my_ci_slice_idx: int) -> None:
        w = self.world
        my_ppgd_slice_idxs = w.ppgd_slice_idxs_for_ci_slice(my_ci_slice_idx)
        with _time_nccl_op("CiOutputsToPpgd.send"):
            for ppgd_slice_idx in my_ppgd_slice_idxs:
                target = w.ppgd_ranks[ppgd_slice_idx]
                sub = w.ppgd_sub_slice_within_ci(ppgd_slice_idx)
                parts: list[Tensor] = []
                for d in (ci.lower_leaky, ci.upper_leaky, ci.pre_sigmoid):
                    parts.extend(
                        d[site][sub].detach().to(_WIRE_DTYPE).contiguous().flatten()
                        for site in w.all_sites
                    )
                packed = torch.cat(parts)
                dist.send(packed, dst=target, group=w.cross_pool_p2p_group)

    def recv(
        self,
        *,
        my_ppgd_slice_idx: int,
        site_to_c: dict[str, int],
        seq_len: int,
        device: torch.device,
    ) -> CIOutputs:
        w = self.world
        src_ci_slice = w.ci_slice_of_ppgd_slice(my_ppgd_slice_idx)
        src = w.ci_ranks[src_ci_slice]
        b_pp = w.batch_local_ppgd

        per_block_numel = sum(b_pp * seq_len * site_to_c[s] for s in w.all_sites)
        packed = torch.empty(3 * per_block_numel, device=device, dtype=_WIRE_DTYPE)
        with _time_nccl_op("CiOutputsToPpgd.recv"):
            dist.recv(packed, src=src, group=w.cross_pool_p2p_group)

        out: list[dict[str, Tensor]] = [{}, {}, {}]
        offset = 0
        for block_idx in range(3):
            for site in w.all_sites:
                c_s = site_to_c[site]
                numel = b_pp * seq_len * c_s
                out[block_idx][site] = packed[offset : offset + numel].view(b_pp, seq_len, c_s)
                offset += numel
        assert offset == packed.numel(), f"unpack mismatch: {offset} of {packed.numel()}"
        return CIOutputs(lower_leaky=out[0], upper_leaky=out[1], pre_sigmoid=out[2])


@dataclass(frozen=True)
class Portals:
    """The full set of cross-pool exchange portals, built once per rank from
    the shared ``World``. Every rank holds the same set; each pool invokes only
    the sides its role plays. Threaded into the step functions so the per-step
    flow reads as portal invocations against the dependency DAG.
    """

    ci_values_to_lw: CiValuesToLayerwise
    ci_values_to_ppgd: CiValuesToPpgd
    grad_ci_from_lw: GradCiFromLayerwise
    grad_ci_from_ppgd: GradCiFromPpgd
    grad_vu_from_ppgd: GradVuFromPpgd
    updated_vu_to_ppgd: UpdatedVuToPpgd
    ci_outputs_to_ppgd: CiOutputsToPpgd

    @classmethod
    def from_world(cls, world: World) -> "Portals":
        return cls(
            ci_values_to_lw=CiValuesToLayerwise(world),
            ci_values_to_ppgd=CiValuesToPpgd(world),
            grad_ci_from_lw=GradCiFromLayerwise(world),
            grad_ci_from_ppgd=GradCiFromPpgd(world),
            grad_vu_from_ppgd=GradVuFromPpgd(world),
            updated_vu_to_ppgd=UpdatedVuToPpgd(world),
            ci_outputs_to_ppgd=CiOutputsToPpgd(world),
        )
