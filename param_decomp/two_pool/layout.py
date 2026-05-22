"""World / TwoPoolLayout / BlockDDPLayout — the topology data model.

`World` (and `BlockDDPWorld`) is purely declarative — identical content on every rank,
no per-rank fields. Built once at startup after `dist.init_process_group`.

`TwoPoolLayout` (and `BlockDDPLayout`) wraps a World, adds this rank's perspective
(my_pool, my_owned_sites, my_slice_idx), and hangs the cross-pool comm orchestration
methods off itself.

Six comm methods cover the four cross-pool exchanges:
  send_owned_ci_to_pool_b   ↔   recv_ci_from_owners        (CI values, A → B)
  send_pool_b_grads_to_owners ↔ recv_grads_from_pool_b     (grads, B → A)
  send_updated_weights_to_pool_b ↔ recv_updated_weights_from_owners (weights, A → B)

Block-DDP variant additionally exposes `all_reduce_grads_in_block` for the in-block
DDP sync over replicated V/U + CI fn params within a block group.

Origin: `nano_param_decomp.two_pool.layout` / `.block_ddp`. See those files for the
implementation arc; this module is the production version.
"""

# pyright: reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

# All cross-pool tensors are cast to this dtype on the wire. Halves the bytes
# transferred per step (~22 GB → 11 GB for LlamaSimpleMLP-12L at 24×2+16 topology).
# Pool B's forward + PPGD backward run inside bf16 autocast already, so bf16 V/U
# values are precision-neutral on use. V/U grads accumulating into pool A's fp32
# .grad upcast back to fp32 on receive — same pattern as standard bf16 mixed
# precision training.
_WIRE_DTYPE: torch.dtype = torch.bfloat16

# ───────────────────────── plain (1 rank per block group) ─────────────────────────


@dataclass(frozen=True)
class World:
    """Declarative 2-pool topology — identical content on every rank.

    Each site has exactly one owning pool-A rank. No in-block DDP here.
    """

    world_size: int
    pool_a_ranks: tuple[int, ...]
    pool_b_ranks: tuple[int, ...]
    all_sites: tuple[str, ...]
    site_owner: dict[str, int]
    batch_global: int
    pool_a_group: dist.ProcessGroup
    pool_b_group: dist.ProcessGroup

    @property
    def n_pool_a(self) -> int:
        return len(self.pool_a_ranks)

    @property
    def n_pool_b(self) -> int:
        return len(self.pool_b_ranks)

    @property
    def batch_local_b(self) -> int:
        assert self.batch_global % self.n_pool_b == 0
        return self.batch_global // self.n_pool_b


def build_world(
    pool_a_ranks: list[int],
    pool_b_ranks: list[int],
    all_sites: list[str],
    site_owner: dict[str, int],
    batch_global: int,
) -> World:
    """Construct a World after dist.init_process_group on every rank.

    Every rank must call `dist.new_group` collectively for both pools.
    """
    world_size = dist.get_world_size()
    assert len(pool_a_ranks) + len(pool_b_ranks) == world_size, (
        f"pool ranks ({len(pool_a_ranks)} + {len(pool_b_ranks)}) != world_size ({world_size})"
    )
    assert set(pool_a_ranks).isdisjoint(set(pool_b_ranks))
    for site, owner in site_owner.items():
        assert owner in pool_a_ranks
        assert site in all_sites
    pool_a_group = dist.new_group(ranks=pool_a_ranks)
    pool_b_group = dist.new_group(ranks=pool_b_ranks)
    return World(
        world_size=world_size,
        pool_a_ranks=tuple(pool_a_ranks),
        pool_b_ranks=tuple(pool_b_ranks),
        all_sites=tuple(all_sites),
        site_owner=dict(site_owner),
        batch_global=batch_global,
        pool_a_group=pool_a_group,
        pool_b_group=pool_b_group,
    )


@dataclass(frozen=True)
class TwoPoolLayout:
    """This rank's view of the world + cross-pool comm orchestration."""

    world: World
    my_rank: int
    my_pool: Literal["a", "b"]
    my_is_pool_leader: bool
    my_owned_sites: tuple[str, ...]
    my_slice_idx: int | None

    @classmethod
    def from_world(cls, world: World, my_rank: int) -> "TwoPoolLayout":
        if my_rank in world.pool_a_ranks:
            return cls(
                world=world,
                my_rank=my_rank,
                my_pool="a",
                my_is_pool_leader=(my_rank == world.pool_a_ranks[0]),
                my_owned_sites=tuple(s for s in world.all_sites if world.site_owner[s] == my_rank),
                my_slice_idx=None,
            )
        elif my_rank in world.pool_b_ranks:
            return cls(
                world=world,
                my_rank=my_rank,
                my_pool="b",
                my_is_pool_leader=(my_rank == world.pool_b_ranks[0]),
                my_owned_sites=(),
                my_slice_idx=world.pool_b_ranks.index(my_rank),
            )
        raise ValueError(f"rank {my_rank} not in any pool")

    def is_my_site(self, site: str) -> bool:
        return self.my_pool == "a" and self.world.site_owner[site] == self.my_rank

    def owner_of(self, site: str) -> int:
        return self.world.site_owner[site]

    def slice_for_b_idx(self, slice_idx: int) -> slice:
        b = self.world.batch_local_b
        return slice(slice_idx * b, (slice_idx + 1) * b)

    def my_batch_slice(self) -> slice:
        assert self.my_pool == "b" and self.my_slice_idx is not None
        b = self.world.batch_local_b
        return slice(self.my_slice_idx * b, (self.my_slice_idx + 1) * b)

    # ── Cross-pool comm ──

    def send_owned_ci_to_pool_b(self, ci_owned: dict[str, Tensor]) -> None:
        assert self.my_pool == "a"
        for site in self.world.all_sites:
            if not self.is_my_site(site):
                continue
            for slice_idx, b_rank in enumerate(self.world.pool_b_ranks):
                sl = self.slice_for_b_idx(slice_idx)
                dist.send(ci_owned[site][sl].detach().contiguous(), dst=b_rank)

    def recv_ci_from_owners(
        self,
        site_to_c: dict[str, int],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Tensor]:
        assert self.my_pool == "b"
        out: dict[str, Tensor] = {}
        b_local = self.world.batch_local_b
        for site in self.world.all_sites:
            owner = self.world.site_owner[site]
            buf = torch.empty(b_local, seq_len, site_to_c[site], device=device, dtype=dtype)
            dist.recv(buf, src=owner)
            out[site] = buf
        return out

    def send_pool_b_grads_to_owners(
        self,
        v_grads: dict[str, Tensor],
        u_grads: dict[str, Tensor],
        ci_grads: dict[str, Tensor],
    ) -> None:
        assert self.my_pool == "b"
        if self.my_is_pool_leader:
            for site in self.world.all_sites:
                owner = self.world.site_owner[site]
                dist.send(v_grads[site].contiguous(), dst=owner)
                dist.send(u_grads[site].contiguous(), dst=owner)
        for site in self.world.all_sites:
            owner = self.world.site_owner[site]
            dist.send(ci_grads[site].contiguous(), dst=owner)

    def recv_grads_from_pool_b(
        self,
        v_templates: dict[str, Tensor],
        u_templates: dict[str, Tensor],
        ci_lower_owned: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        assert self.my_pool == "a"
        v_grads: dict[str, Tensor] = {}
        u_grads: dict[str, Tensor] = {}
        ci_grads: dict[str, Tensor] = {}

        b_leader = self.world.pool_b_ranks[0]
        for site in self.world.all_sites:
            if not self.is_my_site(site):
                continue
            v_buf = torch.empty_like(v_templates[site])
            u_buf = torch.empty_like(u_templates[site])
            dist.recv(v_buf, src=b_leader)
            dist.recv(u_buf, src=b_leader)
            v_grads[site] = v_buf
            u_grads[site] = u_buf

        b_local = self.world.batch_local_b
        for site in self.world.all_sites:
            if not self.is_my_site(site):
                continue
            full = ci_lower_owned[site]
            _, S, C = full.shape
            slices: list[Tensor] = []
            for b_rank in self.world.pool_b_ranks:
                buf = torch.empty((b_local, S, C), dtype=full.dtype, device=full.device)
                dist.recv(buf, src=b_rank)
                slices.append(buf)
            ci_grads[site] = torch.cat(slices, dim=0)

        return v_grads, u_grads, ci_grads

    def send_updated_weights_to_pool_b(
        self,
        v_owned: dict[str, Tensor],
        u_owned: dict[str, Tensor],
    ) -> None:
        assert self.my_pool == "a"
        for site in self.world.all_sites:
            if not self.is_my_site(site):
                continue
            for b_rank in self.world.pool_b_ranks:
                dist.send(v_owned[site].detach().contiguous(), dst=b_rank)
                dist.send(u_owned[site].detach().contiguous(), dst=b_rank)

    def recv_updated_weights_from_owners(
        self,
        v_templates: dict[str, Tensor],
        u_templates: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        assert self.my_pool == "b"
        v_new: dict[str, Tensor] = {}
        u_new: dict[str, Tensor] = {}
        for site in self.world.all_sites:
            owner = self.world.site_owner[site]
            v_buf = torch.empty_like(v_templates[site])
            u_buf = torch.empty_like(u_templates[site])
            dist.recv(v_buf, src=owner)
            dist.recv(u_buf, src=owner)
            v_new[site] = v_buf
            u_new[site] = u_buf
        return v_new, u_new


# ───────────────────────── Block-DDP (N ranks per block group) ─────────────────────────


@dataclass(frozen=True)
class BlockGroup:
    """One block-DDP group: the ranks that replicate V/U + CI fn for a shared set of sites.

    The first rank is the block leader — the canonical actor for cross-pool sends.
    Within a group, in-block all-reduce keeps the replicas in sync after each
    optimizer step.
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
class BlockDDPWorld:
    """Pool A organized into block groups; each block group's ranks replicate V/U + CI fn."""

    world_size: int
    block_groups: tuple[BlockGroup, ...]
    pool_b_ranks: tuple[int, ...]
    all_sites: tuple[str, ...]
    batch_global: int
    pool_a_group: dist.ProcessGroup
    pool_b_group: dist.ProcessGroup
    block_group_groups: tuple[dist.ProcessGroup, ...]
    # One process group per block: {block_leader} ∪ {pool_b_ranks}. Used for
    # collective A-leader → pool-B broadcasts (and the symmetric receive).
    # NCCL uses a tree topology so leader egress drops from N_pool_b× to 1×;
    # multiple broadcasts in different groups run concurrently via async_op.
    cross_pool_bcast_groups: tuple[dist.ProcessGroup, ...]

    @property
    def n_blocks(self) -> int:
        return len(self.block_groups)

    @property
    def n_per_block(self) -> int:
        size = len(self.block_groups[0].ranks)
        assert all(len(bg.ranks) == size for bg in self.block_groups)
        return size

    @property
    def n_pool_a(self) -> int:
        return sum(len(bg.ranks) for bg in self.block_groups)

    @property
    def n_pool_b(self) -> int:
        return len(self.pool_b_ranks)

    @property
    def pool_a_ranks(self) -> tuple[int, ...]:
        return tuple(r for bg in self.block_groups for r in bg.ranks)

    @property
    def batch_local_a(self) -> int:
        assert self.batch_global % self.n_per_block == 0
        return self.batch_global // self.n_per_block

    @property
    def batch_local_b(self) -> int:
        assert self.batch_global % self.n_pool_b == 0
        return self.batch_global // self.n_pool_b

    def block_idx_of_site(self, site: str) -> int:
        for i, bg in enumerate(self.block_groups):
            if site in bg.owned_sites:
                return i
        raise KeyError(site)

    def block_leader_of_site(self, site: str) -> int:
        return self.block_groups[self.block_idx_of_site(site)].leader


def build_block_ddp_world(
    block_groups: list[BlockGroup],
    pool_b_ranks: list[int],
    batch_global: int,
) -> BlockDDPWorld:
    world_size = dist.get_world_size()
    pool_a_ranks = [r for bg in block_groups for r in bg.ranks]
    assert len(pool_a_ranks) + len(pool_b_ranks) == world_size
    assert set(pool_a_ranks).isdisjoint(set(pool_b_ranks))
    assert len(set(pool_a_ranks)) == len(pool_a_ranks)

    all_sites = tuple(s for bg in block_groups for s in bg.owned_sites)
    assert len(set(all_sites)) == len(all_sites), "a site is owned by more than one block group"

    pool_a_group = dist.new_group(ranks=pool_a_ranks)
    pool_b_group = dist.new_group(ranks=pool_b_ranks)
    block_group_groups = tuple(dist.new_group(ranks=list(bg.ranks)) for bg in block_groups)
    cross_pool_bcast_groups = tuple(
        dist.new_group(ranks=[bg.leader, *pool_b_ranks]) for bg in block_groups
    )

    return BlockDDPWorld(
        world_size=world_size,
        block_groups=tuple(block_groups),
        pool_b_ranks=tuple(pool_b_ranks),
        all_sites=all_sites,
        batch_global=batch_global,
        pool_a_group=pool_a_group,
        pool_b_group=pool_b_group,
        block_group_groups=block_group_groups,
        cross_pool_bcast_groups=cross_pool_bcast_groups,
    )


@dataclass(frozen=True)
class BlockDDPLayout:
    """This rank's view of the block-DDP world + comm + in-block all-reduce."""

    world: BlockDDPWorld
    my_rank: int
    my_pool: Literal["a", "b"]
    my_block_idx: int | None
    my_within_block_idx: int | None
    my_is_block_leader: bool
    my_owned_sites: tuple[str, ...]
    my_slice_idx: int | None
    my_is_pool_leader: bool

    @classmethod
    def from_world(cls, world: BlockDDPWorld, my_rank: int) -> "BlockDDPLayout":
        for bg_idx, bg in enumerate(world.block_groups):
            if my_rank in bg.ranks:
                within = bg.ranks.index(my_rank)
                return cls(
                    world=world,
                    my_rank=my_rank,
                    my_pool="a",
                    my_block_idx=bg_idx,
                    my_within_block_idx=within,
                    my_is_block_leader=(within == 0),
                    my_owned_sites=bg.owned_sites,
                    my_slice_idx=None,
                    my_is_pool_leader=False,
                )
        if my_rank in world.pool_b_ranks:
            return cls(
                world=world,
                my_rank=my_rank,
                my_pool="b",
                my_block_idx=None,
                my_within_block_idx=None,
                my_is_block_leader=False,
                my_owned_sites=(),
                my_slice_idx=world.pool_b_ranks.index(my_rank),
                my_is_pool_leader=(my_rank == world.pool_b_ranks[0]),
            )
        raise ValueError(f"rank {my_rank} not in any pool")

    def is_my_site(self, site: str) -> bool:
        return self.my_pool == "a" and site in self.my_owned_sites

    def i_lead_site(self, site: str) -> bool:
        return self.is_my_site(site) and self.my_is_block_leader

    def my_batch_slice_a(self) -> slice:
        assert self.my_pool == "a" and self.my_within_block_idx is not None
        b = self.world.batch_local_a
        return slice(self.my_within_block_idx * b, (self.my_within_block_idx + 1) * b)

    def my_batch_slice_b(self) -> slice:
        assert self.my_pool == "b" and self.my_slice_idx is not None
        b = self.world.batch_local_b
        return slice(self.my_slice_idx * b, (self.my_slice_idx + 1) * b)

    def slice_for_b_idx(self, slice_idx: int) -> slice:
        b = self.world.batch_local_b
        return slice(slice_idx * b, (slice_idx + 1) * b)

    # ── Cross-pool comm (block leader is the canonical actor) ──

    def send_owned_ci_to_pool_b(self, ci_owned: dict[str, Tensor]) -> None:
        assert self.my_pool == "a"
        for site in self.world.all_sites:
            if not self.i_lead_site(site):
                continue
            for slice_idx, b_rank in enumerate(self.world.pool_b_ranks):
                sl = self.slice_for_b_idx(slice_idx)
                dist.send(ci_owned[site][sl].detach().contiguous(), dst=b_rank)

    def async_send_owned_ci_to_pool_b(
        self,
        ci_owned: dict[str, Tensor],
    ) -> tuple[list["dist.Work"], list[Tensor]]:
        """Async variant — issue isends and return (work_handles, kept-alive buffers).
        Caller must wait on the handles AND keep the buffers alive until then.
        Cast to bf16 on the wire (halves bytes; pool B uses bf16 autocast).
        """
        assert self.my_pool == "a"
        works: list[dist.Work] = []
        buffers: list[Tensor] = []
        for site in self.world.all_sites:
            if not self.i_lead_site(site):
                continue
            for slice_idx, b_rank in enumerate(self.world.pool_b_ranks):
                sl = self.slice_for_b_idx(slice_idx)
                buf = ci_owned[site][sl].detach().to(_WIRE_DTYPE).contiguous()
                works.append(dist.isend(buf, dst=b_rank))
                buffers.append(buf)
        return works, buffers

    def recv_ci_from_owners(
        self,
        site_to_c: dict[str, int],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Tensor]:
        """Receives bf16-on-wire CI, upcasts to caller's requested ``dtype``."""
        assert self.my_pool == "b"
        out: dict[str, Tensor] = {}
        b_local = self.world.batch_local_b
        for site in self.world.all_sites:
            leader = self.world.block_leader_of_site(site)
            buf = torch.empty(b_local, seq_len, site_to_c[site], device=device, dtype=_WIRE_DTYPE)
            dist.recv(buf, src=leader)
            out[site] = buf.to(dtype)
        return out

    def async_recv_ci_from_owners(
        self,
        site_to_c: dict[str, int],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[dict[str, Tensor], list["dist.Work"]]:
        """Async bf16-on-wire CI recv. Caller waits on the works, then casts to ``dtype``.

        We return the raw bf16 buffers; caller-side cast (after wait) lets the
        pool-B step keep the work-handle / consume-buffer separation simple.
        """
        assert self.my_pool == "b"
        out: dict[str, Tensor] = {}
        works: list[dist.Work] = []
        del dtype  # unused; pool-B caller casts after wait
        b_local = self.world.batch_local_b
        for site in self.world.all_sites:
            leader = self.world.block_leader_of_site(site)
            buf = torch.empty(b_local, seq_len, site_to_c[site], device=device, dtype=_WIRE_DTYPE)
            works.append(dist.irecv(buf, src=leader))
            out[site] = buf
        return out, works

    def send_pool_b_grads_to_owners(
        self,
        v_grads: dict[str, Tensor],
        u_grads: dict[str, Tensor],
        ci_grads: dict[str, Tensor],
    ) -> None:
        """Coalesced + pipelined grad send pool B → pool A leaders.

        Per A-leader, pack all owned sites' grads into one flat tensor.
        Use isend so the 24 destinations overlap on the NIC instead of
        serializing — each individual send still costs the same wire time
        but the per-message setup latency pipelines.
        """
        assert self.my_pool == "b"
        works: list[dist.Work] = []
        buffers: list[Tensor] = []  # keep alive until waits complete
        if self.my_is_pool_leader:
            for bg in self.world.block_groups:
                parts: list[Tensor] = []
                for site in bg.owned_sites:
                    parts.append(v_grads[site].to(_WIRE_DTYPE).contiguous().flatten())
                    parts.append(u_grads[site].to(_WIRE_DTYPE).contiguous().flatten())
                packed = torch.cat(parts)
                w = dist.isend(packed, dst=bg.leader)
                assert w is not None
                works.append(w)
                buffers.append(packed)
        for bg in self.world.block_groups:
            parts = [
                ci_grads[site].to(_WIRE_DTYPE).contiguous().flatten() for site in bg.owned_sites
            ]
            packed = torch.cat(parts)
            w = dist.isend(packed, dst=bg.leader)
            assert w is not None
            works.append(w)
            buffers.append(packed)
        for w in works:
            w.wait()
        del buffers

    def recv_grads_from_pool_b(
        self,
        v_templates: dict[str, Tensor],
        u_templates: dict[str, Tensor],
        ci_lower_owned: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        """Coalesced grad recv pool A leader ← pool B. Mirrors send_pool_b_grads_to_owners."""
        assert self.my_pool == "a"
        v_grads: dict[str, Tensor] = {}
        u_grads: dict[str, Tensor] = {}
        ci_grads: dict[str, Tensor] = {}

        b_leader = self.world.pool_b_ranks[0]

        if self.my_is_block_leader:
            # ── Post all irecvs upfront so they pipeline. Receiver-side
            #    setup latency runs concurrently with sender-side egress.
            #    Recv into bf16 buffers, then upcast on unpack so grads
            #    accumulate into pool A's fp32 .grad.
            my_sites = self.my_owned_sites

            # V/U grads: one coalesced recv from pool B leader.
            vu_numel = sum(v_templates[s].numel() + u_templates[s].numel() for s in my_sites)
            sample_v = v_templates[my_sites[0]]
            vu_buf = torch.empty(vu_numel, dtype=_WIRE_DTYPE, device=sample_v.device)
            vu_work = dist.irecv(vu_buf, src=b_leader)
            assert vu_work is not None

            # ci_grads: one coalesced recv per pool B rank.
            b_local = self.world.batch_local_b
            per_b_numel = sum(ci_lower_owned[s][:b_local].numel() for s in my_sites)
            sample_ci = ci_lower_owned[my_sites[0]]
            ci_bufs: list[Tensor] = []
            ci_works: list[dist.Work] = []
            for b_rank in self.world.pool_b_ranks:
                ci_buf = torch.empty(per_b_numel, dtype=_WIRE_DTYPE, device=sample_ci.device)
                w = dist.irecv(ci_buf, src=b_rank)
                assert w is not None
                ci_bufs.append(ci_buf)
                ci_works.append(w)

            # Now wait + unpack each as it arrives. Upcast on view.
            vu_work.wait()
            offset = 0
            for s in my_sites:
                v_n = v_templates[s].numel()
                u_n = u_templates[s].numel()
                v_grads[s] = (
                    vu_buf[offset : offset + v_n].view_as(v_templates[s]).to(v_templates[s].dtype)
                )
                offset += v_n
                u_grads[s] = (
                    vu_buf[offset : offset + u_n].view_as(u_templates[s]).to(u_templates[s].dtype)
                )
                offset += u_n

            slices_per_site: dict[str, list[Tensor]] = {s: [] for s in my_sites}
            for ci_buf, w in zip(ci_bufs, ci_works, strict=True):
                w.wait()
                offset = 0
                for s in my_sites:
                    full = ci_lower_owned[s]
                    _, S, C = full.shape
                    slice_n = b_local * S * C
                    slices_per_site[s].append(
                        ci_buf[offset : offset + slice_n].view(b_local, S, C).to(full.dtype)
                    )
                    offset += slice_n
            for s in my_sites:
                ci_grads[s] = torch.cat(slices_per_site[s], dim=0)
        else:
            # Non-leader pool A ranks pre-allocate; populated via broadcast below.
            for s in self.my_owned_sites:
                v_grads[s] = torch.empty_like(v_templates[s])
                u_grads[s] = torch.empty_like(u_templates[s])
                ci_grads[s] = torch.empty_like(ci_lower_owned[s])

        # In-block broadcast leader → other ranks in the group (unchanged).
        assert self.my_block_idx is not None
        block_group = self.world.block_group_groups[self.my_block_idx]
        block_leader_rank = self.world.block_groups[self.my_block_idx].leader
        for s in self.my_owned_sites:
            v_grads[s] = v_grads[s].contiguous()
            u_grads[s] = u_grads[s].contiguous()
            ci_grads[s] = ci_grads[s].contiguous()
            dist.broadcast(v_grads[s], src=block_leader_rank, group=block_group)
            dist.broadcast(u_grads[s], src=block_leader_rank, group=block_group)
            dist.broadcast(ci_grads[s], src=block_leader_rank, group=block_group)

        return v_grads, u_grads, ci_grads

    def all_reduce_grads_in_block(self, params: Iterable[nn.Parameter]) -> None:
        """Coalesced in-block DDP all-reduce.

        Flatten all param grads of the same dtype + device into one big buffer,
        all-reduce once, scatter back. Cuts NCCL message count from `len(params)`
        (e.g. ~76 for the CI fn at 64-GPU 24×2+16) to 1 per (dtype, device)
        bucket. HTA showed the per-param pattern was costing ~250ms/step.
        """
        assert self.my_pool == "a" and self.my_block_idx is not None
        block_group = self.world.block_group_groups[self.my_block_idx]
        if dist.get_world_size(block_group) <= 1:
            return  # 1-rank block: in-block DDP is a no-op

        grads: list[Tensor] = [p.grad for p in params if p.grad is not None]
        if not grads:
            return
        # Bucket by (dtype, device) — typically just one bucket in practice.
        buckets: dict[tuple[torch.dtype, torch.device], list[Tensor]] = {}
        for g in grads:
            buckets.setdefault((g.dtype, g.device), []).append(g)
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

        for (dtype, device), bucket in buckets.items():
            del dtype, device  # used as key
            flat = _flatten_dense_tensors(bucket)
            dist.all_reduce(flat, op=dist.ReduceOp.AVG, group=block_group)
            for orig, reduced in zip(bucket, _unflatten_dense_tensors(flat, bucket), strict=True):
                orig.copy_(reduced)

    def send_updated_weights_to_pool_b(
        self,
        v_owned: dict[str, Tensor],
        u_owned: dict[str, Tensor],
    ) -> None:
        assert self.my_pool == "a"
        for site in self.world.all_sites:
            if not self.i_lead_site(site):
                continue
            for b_rank in self.world.pool_b_ranks:
                dist.send(v_owned[site].detach().contiguous(), dst=b_rank)
                dist.send(u_owned[site].detach().contiguous(), dst=b_rank)

    def async_send_updated_weights_to_pool_b(
        self,
        v_owned: dict[str, Tensor],
        u_owned: dict[str, Tensor],
    ) -> tuple[list["dist.Work"], list[Tensor]]:
        """Coalesced async weight send pool A leader → all pool B ranks.

        Pack all owned sites' V then U into one contiguous tensor; reuse the
        same buffer for the isend to every pool B rank (NCCL P2P sends are
        read-only, can share a source tensor). Cuts the per-leader send count
        from `N_owned × 2 × N_pool_b` (e.g. 4×2×16=128) to `N_pool_b` (16).

        Caller must keep the buffer alive (returned in `buffers`) until all
        work handles are done.
        """
        assert self.my_pool == "a"
        if not self.my_is_block_leader:
            return [], []
        my_sites = self.my_owned_sites
        assert self.my_block_idx is not None
        parts: list[Tensor] = []
        for s in my_sites:
            parts.append(v_owned[s].detach().to(_WIRE_DTYPE).contiguous().flatten())
            parts.append(u_owned[s].detach().to(_WIRE_DTYPE).contiguous().flatten())
        packed = torch.cat(parts)  # owns its own memory; safe across optimizer.step
        # One async broadcast to all pool-B ranks via the pre-built
        # {leader} ∪ {pool_b_ranks} group. NCCL uses a tree topology so
        # leader egress drops from N_pool_b× isends to 1×.
        bcast_group = self.world.cross_pool_bcast_groups[self.my_block_idx]
        w = dist.broadcast(packed, src=self.my_rank, group=bcast_group, async_op=True)
        assert w is not None
        return [w], [packed]

    def recv_updated_weights_from_owners(
        self,
        v_templates: dict[str, Tensor],
        u_templates: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Coalesced + pipelined weight recv pool B ← all pool A leaders.

        Post all 24 irecvs upfront so they pipeline, then wait+unpack in
        completion order (skip the wait order — wait() on each in order is
        fine because incomplete waits don't block already-arrived ones).
        """
        assert self.my_pool == "b"
        v_new: dict[str, Tensor] = {}
        u_new: dict[str, Tensor] = {}
        bufs: list[tuple[BlockGroup, Tensor, dist.Work]] = []
        for bg_idx, bg in enumerate(self.world.block_groups):
            owned = bg.owned_sites
            packed_numel = sum(v_templates[s].numel() + u_templates[s].numel() for s in owned)
            sample = v_templates[owned[0]]
            packed = torch.empty(packed_numel, dtype=_WIRE_DTYPE, device=sample.device)
            # Mirror the leader's broadcast: post into the leader-rooted group
            # for this block. async_op=True lets all N_blocks broadcasts run
            # concurrently on independent NCCL communicators.
            bcast_group = self.world.cross_pool_bcast_groups[bg_idx]
            w = dist.broadcast(packed, src=bg.leader, group=bcast_group, async_op=True)
            assert w is not None
            bufs.append((bg, packed, w))
        for bg, packed, w in bufs:
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
