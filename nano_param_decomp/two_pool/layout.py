"""World / TwoPoolLayout — the single source of truth for 2-pool topology.

`World` is purely declarative — same value across the cluster, no per-rank fields. It
captures who's in which pool, which sites are owned by whom, batch size, and the dist
process groups.

`TwoPoolLayout` wraps a World, adds this rank's perspective (my_pool, my_owned_sites,
my_slice_idx, etc), and hangs the cross-pool comm methods off itself.

Both are immutable. Build once at startup, pass everywhere.
"""

# pyright: reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false

from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass(frozen=True)
class World:
    """Purely declarative global topology. Identical content on every rank.

    Process group handles are technically per-process objects but each represents the
    same logical group; storing them on `World` is conceptually correct.
    """

    world_size: int
    pool_a_ranks: tuple[int, ...]
    pool_b_ranks: tuple[int, ...]
    all_sites: tuple[str, ...]            # canonical iteration order
    site_owner: dict[str, int]            # site path → pool-A rank
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
        assert self.batch_global % self.n_pool_b == 0, (
            f"batch_global={self.batch_global} not divisible by n_pool_b={self.n_pool_b}"
        )
        return self.batch_global // self.n_pool_b


def build_world(
    pool_a_ranks: list[int],
    pool_b_ranks: list[int],
    all_sites: list[str],
    site_owner: dict[str, int],
    batch_global: int,
) -> World:
    """Construct a World — call after dist.init_process_group on every rank."""
    world_size = dist.get_world_size()
    assert len(pool_a_ranks) + len(pool_b_ranks) == world_size, (
        f"pool ranks ({len(pool_a_ranks)} + {len(pool_b_ranks)}) != world_size ({world_size})"
    )
    assert set(pool_a_ranks).isdisjoint(set(pool_b_ranks))
    for site, owner in site_owner.items():
        assert owner in pool_a_ranks, f"site {site} owner {owner} not in pool A"
        assert site in all_sites, f"site {site} not in all_sites"
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
    """This rank's perspective on the world + the cross-pool comm orchestration."""

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

    # --- Queries ---

    def is_my_site(self, site: str) -> bool:
        return self.my_pool == "a" and self.world.site_owner[site] == self.my_rank

    def owner_of(self, site: str) -> int:
        return self.world.site_owner[site]

    def my_batch_slice(self) -> slice:
        assert self.my_pool == "b" and self.my_slice_idx is not None
        b = self.world.batch_local_b
        return slice(self.my_slice_idx * b, (self.my_slice_idx + 1) * b)

    def slice_for_b_idx(self, slice_idx: int) -> slice:
        b = self.world.batch_local_b
        return slice(slice_idx * b, (slice_idx + 1) * b)

    # --- Cross-pool comm ---
    # Each method asserts the pool that may call it. Iteration over sites uses the
    # canonical `world.all_sites` order so all ranks step through send/recv in lockstep.

    def send_owned_ci_to_pool_b(self, ci_owned: dict[str, Tensor]) -> None:
        """Pool A. For each of my owned sites, send the per-B-rank batch slice of ci."""
        assert self.my_pool == "a"
        for site in self.world.all_sites:
            if not self.is_my_site(site):
                continue
            for slice_idx, b_rank in enumerate(self.world.pool_b_ranks):
                sl = self.slice_for_b_idx(slice_idx)
                dist.send(ci_owned[site][sl].detach().contiguous(), dst=b_rank)

    def recv_ci_from_owners(
        self,
        wrappers: dict[str, "ComponentLinear"],  # type: ignore[name-defined]
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Tensor]:
        """Pool B. Recv per-site (batch-slice) ci from each site's owning A rank."""
        assert self.my_pool == "b"
        out: dict[str, Tensor] = {}
        b_local = self.world.batch_local_b
        for site in self.world.all_sites:
            owner = self.world.site_owner[site]
            buf = torch.empty(b_local, seq_len, wrappers[site].C, device=device, dtype=dtype)
            dist.recv(buf, src=owner)
            out[site] = buf
        return out

    def send_pool_b_grads_to_owners(
        self,
        v_grads: dict[str, Tensor],
        u_grads: dict[str, Tensor],
        ci_grads: dict[str, Tensor],
    ) -> None:
        """Pool B. B leader sends pool-B-reduced V/U grads to each site's owner. Every B
        rank sends its per-slice ci_scratch.grad to each site's owner.
        """
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
        wrappers: dict[str, "ComponentLinear"],  # type: ignore[name-defined]
        ci_lower_owned: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        """Pool A. For each owned site: recv V/U grads (from B leader) + per-slice ci
        grads (from each B rank, concat in batch dim).
        """
        assert self.my_pool == "a"
        v_grads: dict[str, Tensor] = {}
        u_grads: dict[str, Tensor] = {}
        ci_grads: dict[str, Tensor] = {}

        b_leader = self.world.pool_b_ranks[0]
        for site in self.world.all_sites:
            if not self.is_my_site(site):
                continue
            v_buf = torch.empty_like(wrappers[site].V)
            u_buf = torch.empty_like(wrappers[site].U)
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
            for _slice_idx, b_rank in enumerate(self.world.pool_b_ranks):
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
        """Pool A. After optimizer step, broadcast each owned site's V/U to every B rank."""
        assert self.my_pool == "a"
        for site in self.world.all_sites:
            if not self.is_my_site(site):
                continue
            for b_rank in self.world.pool_b_ranks:
                dist.send(v_owned[site].detach().contiguous(), dst=b_rank)
                dist.send(u_owned[site].detach().contiguous(), dst=b_rank)

    def recv_updated_weights_from_owners(
        self,
        wrappers: dict[str, "ComponentLinear"],  # type: ignore[name-defined]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Pool B. Recv updated V/U for every site from its owner."""
        assert self.my_pool == "b"
        v_new: dict[str, Tensor] = {}
        u_new: dict[str, Tensor] = {}
        for site in self.world.all_sites:
            owner = self.world.site_owner[site]
            v_buf = torch.empty_like(wrappers[site].V)
            u_buf = torch.empty_like(wrappers[site].U)
            dist.recv(v_buf, src=owner)
            dist.recv(u_buf, src=owner)
            v_new[site] = v_buf
            u_new[site] = u_buf
        return v_new, u_new
