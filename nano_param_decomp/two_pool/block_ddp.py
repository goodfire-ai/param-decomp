"""Block-DDP variant of the 2-pool layout.

Generalizes stage 4's "one pool-A rank per block" to "N pool-A ranks per block, DDP'd
within the block group". Each block group holds the same V/U + CI fns + opt state on
every rank; the ranks in the group share the per-step layerwise compute via batch DP.

Cross-pool comm is essentially identical to stage 4's `TwoPoolLayout`: the block leader
(rank 0 within the group) is the canonical sender/receiver for cross-pool messages.
Non-leader ranks in the block group only participate in the in-block all-reduce.

Topology constraint for this MVP: doesn't require alignment between in-block DP size
and pool B's DP size. Cross-pool CI is full-batch sliced per pool-B rank; pool B's V/U
grads come back to the block leader and get broadcast within the block group.
"""

# pyright: reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from nano_param_decomp.run import ComponentLinear
from nano_param_decomp.two_pool_stage2 import ModuleCIFn


@dataclass(frozen=True)
class BlockDDPWorld:
    """Declarative global topology for the in-block-DDP 2-pool layout.

    Pool A is organized into block groups. Each block group is a tuple of pool-A ranks
    that share replication of a set of sites. The block group's rank 0 is the "leader"
    and handles cross-pool send/recv on behalf of the group.

    Pool B is a flat DP-N group, same as stage 4.
    """

    world_size: int

    block_groups: tuple[tuple[int, ...], ...]  # one tuple per block
    block_owned_sites: tuple[tuple[str, ...], ...]  # parallel to block_groups

    pool_b_ranks: tuple[int, ...]

    all_sites: tuple[str, ...]
    batch_global: int

    pool_b_group: dist.ProcessGroup
    block_group_groups: tuple[dist.ProcessGroup, ...]  # one per block group

    @property
    def n_blocks(self) -> int:
        return len(self.block_groups)

    @property
    def n_per_block(self) -> int:
        size = len(self.block_groups[0])
        assert all(len(bg) == size for bg in self.block_groups), (
            "heterogeneous block group sizes not supported in MVP"
        )
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
        """Batch slice per pool-A rank — driven by in-block DP size."""
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
    """Construct a BlockDDPWorld after dist.init_process_group on every rank.

    Calls `dist.new_group` for the pool-B group and once per block group — all ranks
    must execute these collectively, even non-members.
    """
    world_size = dist.get_world_size()
    assert len(block_groups) == len(block_owned_sites), (
        f"block_groups ({len(block_groups)}) and block_owned_sites "
        f"({len(block_owned_sites)}) must be parallel"
    )
    pool_a_ranks = [r for bg in block_groups for r in bg]
    assert len(pool_a_ranks) + len(pool_b_ranks) == world_size
    assert set(pool_a_ranks).isdisjoint(set(pool_b_ranks))
    assert len(set(pool_a_ranks)) == len(pool_a_ranks), "duplicate rank in block_groups"

    all_sites = tuple(s for sites in block_owned_sites for s in sites)
    # Every rank must call new_group, even if not a member, so the call order matches.
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
    """This rank's view of the block-DDP world + cross-pool comm + in-block all-reduce."""

    world: BlockDDPWorld
    my_rank: int
    my_pool: Literal["a", "b"]

    # Pool A perspective
    my_block_idx: int | None
    my_within_block_idx: int | None
    my_is_block_leader: bool
    my_owned_sites: tuple[str, ...]

    # Pool B perspective
    my_slice_idx: int | None
    my_is_pool_leader: bool

    @classmethod
    def from_world(cls, world: BlockDDPWorld, my_rank: int) -> "BlockDDPLayout":
        for bg_idx, bg in enumerate(world.block_groups):
            if my_rank in bg:
                within = bg.index(my_rank)
                return cls(
                    world=world,
                    my_rank=my_rank,
                    my_pool="a",
                    my_block_idx=bg_idx,
                    my_within_block_idx=within,
                    my_is_block_leader=(within == 0),
                    my_owned_sites=world.block_owned_sites[bg_idx],
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

    # --- Queries ---

    def is_my_site(self, site: str) -> bool:
        return self.my_pool == "a" and site in self.my_owned_sites

    def i_lead_site(self, site: str) -> bool:
        """Am I the block leader for this site (the cross-pool comm canonical sender)?"""
        return self.is_my_site(site) and self.my_is_block_leader

    def my_batch_slice_a(self) -> slice:
        """For pool A: which batch slice this rank's layerwise loss operates on."""
        assert self.my_pool == "a" and self.my_within_block_idx is not None
        b = self.world.batch_local_a
        return slice(self.my_within_block_idx * b, (self.my_within_block_idx + 1) * b)

    def my_batch_slice_b(self) -> slice:
        """For pool B: which batch slice this rank's PPGD operates on."""
        assert self.my_pool == "b" and self.my_slice_idx is not None
        b = self.world.batch_local_b
        return slice(self.my_slice_idx * b, (self.my_slice_idx + 1) * b)

    def slice_for_b_idx(self, slice_idx: int) -> slice:
        b = self.world.batch_local_b
        return slice(slice_idx * b, (slice_idx + 1) * b)

    # --- Cross-pool comm (same shape as stage 4; block leader is the canonical actor) ---

    def send_owned_ci_to_pool_b(self, ci_owned: dict[str, Tensor]) -> None:
        """Pool A. Only the block leader for each site sends; other ranks idle."""
        assert self.my_pool == "a"
        for site in self.world.all_sites:
            if not self.i_lead_site(site):
                continue
            for slice_idx, b_rank in enumerate(self.world.pool_b_ranks):
                sl = self.slice_for_b_idx(slice_idx)
                dist.send(ci_owned[site][sl].detach().contiguous(), dst=b_rank)

    def recv_ci_from_owners(
        self,
        wrappers: dict[str, ComponentLinear],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Tensor]:
        """Pool B. Each site comes from that site's block leader."""
        assert self.my_pool == "b"
        out: dict[str, Tensor] = {}
        b_local = self.world.batch_local_b
        for site in self.world.all_sites:
            leader = self.world.block_leader_of_site(site)
            buf = torch.empty(b_local, seq_len, wrappers[site].C, device=device, dtype=dtype)
            dist.recv(buf, src=leader)
            out[site] = buf
        return out

    def send_pool_b_grads_to_owners(
        self,
        v_grads: dict[str, Tensor],
        u_grads: dict[str, Tensor],
        ci_grads: dict[str, Tensor],
    ) -> None:
        """Pool B. Each B rank sends per-site ci_scratch.grad to that site's block leader.
        B leader additionally sends pool-B-reduced V/U grads to each block leader.
        """
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
        wrappers: dict[str, ComponentLinear],  # type: ignore[name-defined]
        ci_lower_owned: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        """Pool A. Block leader recvs V/U grads from B leader and per-slice ci_grads from
        every B rank. Then broadcasts everything within the block group so all members
        have the same B-contributed grads.
        """
        assert self.my_pool == "a"
        v_grads: dict[str, Tensor] = {}
        u_grads: dict[str, Tensor] = {}
        ci_grads: dict[str, Tensor] = {}

        b_leader = self.world.pool_b_ranks[0]

        # 1) Block leader recvs V/U from B leader; otherwise allocate empty buffer.
        for site in self.world.all_sites:
            if not self.is_my_site(site):
                continue
            v_buf = torch.empty_like(wrappers[site].V)
            u_buf = torch.empty_like(wrappers[site].U)
            if self.my_is_block_leader:
                dist.recv(v_buf, src=b_leader)
                dist.recv(u_buf, src=b_leader)
            v_grads[site] = v_buf
            u_grads[site] = u_buf

        # 2) Block leader recvs per-slice ci_grads and concats; others allocate buffer.
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
                        (self.world.batch_local_b, S, C),
                        dtype=full.dtype,
                        device=full.device,
                    )
                    dist.recv(buf, src=b_rank)
                    slices.append(buf)
                ci_grads[site].copy_(torch.cat(slices, dim=0))

        # 3) Broadcast B-contributed grads within block group so every rank has them.
        assert self.my_block_idx is not None
        block_group = self.world.block_group_groups[self.my_block_idx]
        block_leader_rank = self.world.block_groups[self.my_block_idx][0]
        for site in self.world.all_sites:
            if not self.is_my_site(site):
                continue
            dist.broadcast(v_grads[site], src=block_leader_rank, group=block_group)
            dist.broadcast(u_grads[site], src=block_leader_rank, group=block_group)
            dist.broadcast(ci_grads[site], src=block_leader_rank, group=block_group)

        return v_grads, u_grads, ci_grads

    def all_reduce_grads_in_block(self, params: Iterable[nn.Parameter]) -> None:
        """Mean-reduce param grads across the block group (in-block DDP sync).

        Each rank's home backward produced partial grads — averaging across the block
        group's ranks combines per-slice layerwise contributions into the full-batch
        gradient.
        """
        assert self.my_pool == "a" and self.my_block_idx is not None
        block_group = self.world.block_group_groups[self.my_block_idx]
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=block_group)

    def send_updated_weights_to_pool_b(
        self,
        v_owned: dict[str, Tensor],
        u_owned: dict[str, Tensor],
    ) -> None:
        """Pool A. Only the block leader sends — others have identical V/U after
        the in-block all-reduce + optimizer step."""
        assert self.my_pool == "a"
        for site in self.world.all_sites:
            if not self.i_lead_site(site):
                continue
            for b_rank in self.world.pool_b_ranks:
                dist.send(v_owned[site].detach().contiguous(), dst=b_rank)
                dist.send(u_owned[site].detach().contiguous(), dst=b_rank)

    def recv_updated_weights_from_owners(
        self,
        wrappers: dict[str, "ComponentLinear"],  # type: ignore[name-defined]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        assert self.my_pool == "b"
        v_new: dict[str, Tensor] = {}
        u_new: dict[str, Tensor] = {}
        for site in self.world.all_sites:
            leader = self.world.block_leader_of_site(site)
            v_buf = torch.empty_like(wrappers[site].V)
            u_buf = torch.empty_like(wrappers[site].U)
            dist.recv(v_buf, src=leader)
            dist.recv(u_buf, src=leader)
            v_new[site] = v_buf
            u_new[site] = u_buf
        return v_new, u_new


# --- Layout-aware install (mirrors two_pool/install.py for the block-DDP case) ---


def install_components_for_block_ddp(
    target: nn.Module,
    layout: BlockDDPLayout,
    c_per_site: dict[str, int],
) -> dict[str, "ComponentLinear"]:  # type: ignore[name-defined]
    """Install ComponentLinear at the sites this rank has.

    Pool A: at every site in `layout.my_owned_sites` — same set for every rank in the
    block group (V/U replicated within the group, sharded across blocks).
    Pool B: at every site (full replica for full-model PPGD).
    """
    from nano_param_decomp.run import ComponentLinear

    paths = layout.my_owned_sites if layout.my_pool == "a" else layout.world.all_sites

    for p in target.parameters():
        p.requires_grad_(False)

    wrappers: dict[str, ComponentLinear] = {}
    for path in paths:
        C = c_per_site[path]
        parent_path, _, attr = path.rpartition(".")
        parent = target.get_submodule(parent_path) if parent_path else target
        linear = target.get_submodule(path)
        assert isinstance(linear, nn.Linear), f"{path} is not nn.Linear"
        wrapper = ComponentLinear(linear, C)
        setattr(parent, attr, wrapper)
        wrappers[path] = wrapper
    return wrappers


def build_ci_fns_for_block_ddp(
    layout: BlockDDPLayout,
    wrappers: dict[str, ComponentLinear],
    c_per_site: dict[str, int],
    hidden: int,
    leaky_alpha: float,
) -> dict[str, ModuleCIFn]:
    """Per-site CI fns — pool A only, owned sites only. Replicated within the block group
    (same seed since all ranks build the same model from the same seed).
    """
    from nano_param_decomp.two_pool_stage2 import build_ci_fns

    if layout.my_pool != "a":
        return {}
    d_in_per_site = {s: int(wrappers[s].W_target.shape[1]) for s in layout.my_owned_sites}
    owned_c = {s: c_per_site[s] for s in layout.my_owned_sites}
    return build_ci_fns(d_in_per_site, owned_c, hidden=hidden, leaky_alpha=leaky_alpha)
