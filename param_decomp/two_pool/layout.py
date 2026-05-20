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
                my_owned_sites=tuple(
                    s for s in world.all_sites if world.site_owner[s] == my_rank
                ),
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
        self, site_to_c: dict[str, int], seq_len: int, device: torch.device, dtype: torch.dtype,
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
        self, v_owned: dict[str, Tensor], u_owned: dict[str, Tensor],
    ) -> None:
        assert self.my_pool == "a"
        for site in self.world.all_sites:
            if not self.is_my_site(site):
                continue
            for b_rank in self.world.pool_b_ranks:
                dist.send(v_owned[site].detach().contiguous(), dst=b_rank)
                dist.send(u_owned[site].detach().contiguous(), dst=b_rank)

    def recv_updated_weights_from_owners(
        self, v_templates: dict[str, Tensor], u_templates: dict[str, Tensor],
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
class BlockDDPWorld:
    """Pool A organized into block groups; each block group's ranks replicate V/U + CI fn."""

    world_size: int
    block_groups: tuple[tuple[int, ...], ...]
    block_owned_sites: tuple[tuple[str, ...], ...]
    pool_b_ranks: tuple[int, ...]
    all_sites: tuple[str, ...]
    batch_global: int
    pool_b_group: dist.ProcessGroup
    block_group_groups: tuple[dist.ProcessGroup, ...]

    @property
    def n_blocks(self) -> int:
        return len(self.block_groups)

    @property
    def n_per_block(self) -> int:
        size = len(self.block_groups[0])
        assert all(len(bg) == size for bg in self.block_groups)
        return size

    @property
    def n_pool_a(self) -> int:
        return sum(len(bg) for bg in self.block_groups)

    @property
    def n_pool_b(self) -> int:
        return len(self.pool_b_ranks)

    @property
    def pool_a_ranks(self) -> tuple[int, ...]:
        return tuple(r for bg in self.block_groups for r in bg)

    @property
    def batch_local_a(self) -> int:
        assert self.batch_global % self.n_per_block == 0
        return self.batch_global // self.n_per_block

    @property
    def batch_local_b(self) -> int:
        assert self.batch_global % self.n_pool_b == 0
        return self.batch_global // self.n_pool_b

    def block_idx_of_site(self, site: str) -> int:
        for i, sites in enumerate(self.block_owned_sites):
            if site in sites:
                return i
        raise KeyError(site)

    def block_leader_of_site(self, site: str) -> int:
        return self.block_groups[self.block_idx_of_site(site)][0]


def build_block_ddp_world(
    block_groups: list[list[int]],
    block_owned_sites: list[list[str]],
    pool_b_ranks: list[int],
    batch_global: int,
) -> BlockDDPWorld:
    world_size = dist.get_world_size()
    assert len(block_groups) == len(block_owned_sites)
    pool_a_ranks = [r for bg in block_groups for r in bg]
    assert len(pool_a_ranks) + len(pool_b_ranks) == world_size
    assert set(pool_a_ranks).isdisjoint(set(pool_b_ranks))
    assert len(set(pool_a_ranks)) == len(pool_a_ranks)

    all_sites = tuple(s for sites in block_owned_sites for s in sites)
    pool_b_group = dist.new_group(ranks=pool_b_ranks)
    block_group_groups = tuple(dist.new_group(ranks=list(bg)) for bg in block_groups)

    return BlockDDPWorld(
        world_size=world_size,
        block_groups=tuple(tuple(bg) for bg in block_groups),
        block_owned_sites=tuple(tuple(s) for s in block_owned_sites),
        pool_b_ranks=tuple(pool_b_ranks),
        all_sites=all_sites,
        batch_global=batch_global,
        pool_b_group=pool_b_group,
        block_group_groups=block_group_groups,
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
            if my_rank in bg:
                within = bg.index(my_rank)
                return cls(
                    world=world, my_rank=my_rank, my_pool="a",
                    my_block_idx=bg_idx, my_within_block_idx=within,
                    my_is_block_leader=(within == 0),
                    my_owned_sites=world.block_owned_sites[bg_idx],
                    my_slice_idx=None, my_is_pool_leader=False,
                )
        if my_rank in world.pool_b_ranks:
            return cls(
                world=world, my_rank=my_rank, my_pool="b",
                my_block_idx=None, my_within_block_idx=None,
                my_is_block_leader=False, my_owned_sites=(),
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

    def recv_ci_from_owners(
        self, site_to_c: dict[str, int], seq_len: int, device: torch.device, dtype: torch.dtype,
    ) -> dict[str, Tensor]:
        assert self.my_pool == "b"
        out: dict[str, Tensor] = {}
        b_local = self.world.batch_local_b
        for site in self.world.all_sites:
            leader = self.world.block_leader_of_site(site)
            buf = torch.empty(b_local, seq_len, site_to_c[site], device=device, dtype=dtype)
            dist.recv(buf, src=leader)
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
                leader = self.world.block_leader_of_site(site)
                dist.send(v_grads[site].contiguous(), dst=leader)
                dist.send(u_grads[site].contiguous(), dst=leader)
        for site in self.world.all_sites:
            leader = self.world.block_leader_of_site(site)
            dist.send(ci_grads[site].contiguous(), dst=leader)

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
            if self.my_is_block_leader:
                dist.recv(v_buf, src=b_leader)
                dist.recv(u_buf, src=b_leader)
            v_grads[site] = v_buf
            u_grads[site] = u_buf

        for site in self.world.all_sites:
            if not self.is_my_site(site):
                continue
            full = ci_lower_owned[site]
            ci_grads[site] = torch.empty_like(full)
            if self.my_is_block_leader:
                _, S, C = full.shape
                slices: list[Tensor] = []
                for b_rank in self.world.pool_b_ranks:
                    buf = torch.empty(
                        (self.world.batch_local_b, S, C), dtype=full.dtype, device=full.device,
                    )
                    dist.recv(buf, src=b_rank)
                    slices.append(buf)
                ci_grads[site].copy_(torch.cat(slices, dim=0))

        assert self.my_block_idx is not None
        block_group = self.world.block_group_groups[self.my_block_idx]
        block_leader_rank = self.world.block_groups[self.my_block_idx][0]
        for site in self.world.all_sites:
            if not self.is_my_site(site):
                continue
            v_grads[site] = v_grads[site].contiguous()
            u_grads[site] = u_grads[site].contiguous()
            ci_grads[site] = ci_grads[site].contiguous()
            dist.broadcast(v_grads[site], src=block_leader_rank, group=block_group)
            dist.broadcast(u_grads[site], src=block_leader_rank, group=block_group)
            dist.broadcast(ci_grads[site], src=block_leader_rank, group=block_group)

        return v_grads, u_grads, ci_grads

    def all_reduce_grads_in_block(self, params: Iterable[nn.Parameter]) -> None:
        assert self.my_pool == "a" and self.my_block_idx is not None
        block_group = self.world.block_group_groups[self.my_block_idx]
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=block_group)

    def send_updated_weights_to_pool_b(
        self, v_owned: dict[str, Tensor], u_owned: dict[str, Tensor],
    ) -> None:
        assert self.my_pool == "a"
        for site in self.world.all_sites:
            if not self.i_lead_site(site):
                continue
            for b_rank in self.world.pool_b_ranks:
                dist.send(v_owned[site].detach().contiguous(), dst=b_rank)
                dist.send(u_owned[site].detach().contiguous(), dst=b_rank)

    def recv_updated_weights_from_owners(
        self, v_templates: dict[str, Tensor], u_templates: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        assert self.my_pool == "b"
        v_new: dict[str, Tensor] = {}
        u_new: dict[str, Tensor] = {}
        for site in self.world.all_sites:
            leader = self.world.block_leader_of_site(site)
            v_buf = torch.empty_like(v_templates[site])
            u_buf = torch.empty_like(u_templates[site])
            dist.recv(v_buf, src=leader)
            dist.recv(u_buf, src=leader)
            v_new[site] = v_buf
            u_new[site] = u_buf
        return v_new, u_new
