"""Block-selected CI expert execution, local or sharded over the expert operand row."""

from dataclasses import dataclass

import jax
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from param_decomp.core.components import SelectedCI
from param_decomp.core.placement import PlacedRule
from param_decomp.routed.experts import (
    ExpertShardedJobs,
    GroupedMatmulBackend,
    RoutedJobs,
    combine_jobs,
    ep_combine_jobs,
    ep_gather_tokens,
    ep_grouped_matmul,
    ep_unsort_jobs,
    expert_sharded_jobs,
    gather_tokens,
    grouped_matmul,
    routed_jobs,
    unsort_jobs,
)


@dataclass(frozen=True)
class TokenSelection:
    """Local token schedules shared by a chunk's expert blocks and output heads."""

    jobs: tuple[RoutedJobs, ...]
    weights: tuple[Float[Array, "T k"], ...]
    lead: tuple[int, ...]

    def gather(self, selection: int, h: Float[Array, "b t d"]) -> Float[Array, "J d"]:
        return gather_tokens(h.reshape(-1, h.shape[-1]), self.jobs[selection])

    def block_matmul(
        self, selection: int, x_jobs: Array, table: Array, backend: GroupedMatmulBackend
    ) -> Array:
        return grouped_matmul(x_jobs, table, self.jobs[selection].group_sizes, backend)

    def combine(self, selection: int, y: Float[Array, "J d"]) -> Float[Array, "b t d"]:
        return combine_jobs(y, self.jobs[selection], self.weights[selection]).reshape(
            *self.lead, y.shape[-1]
        )

    def selected_ci(self, selection: int, y: Array, n_blocks: int) -> SelectedCI:
        per_token = unsort_jobs(y, self.jobs[selection])
        values = per_token.reshape(*self.lead, per_token.shape[-2] * per_token.shape[-1])
        ids = self.jobs[selection].top_idx
        return SelectedCI(values, ids.reshape(*self.lead, ids.shape[-1]), n_blocks)


@dataclass(frozen=True)
class BlockShardedSelection:
    """Expert-parallel computation with token-ordered selected outputs."""

    jobs: tuple[ExpertShardedJobs, ...]
    weights: tuple[Float[Array, "b t k"], ...]
    shard_axis: str

    def gather(self, selection: int, h: Float[Array, "b t d"]) -> Float[Array, "b s J d"]:
        return ep_gather_tokens(h, self.jobs[selection], self.shard_axis)

    def block_matmul(
        self, selection: int, x_jobs: Array, table: Array, backend: GroupedMatmulBackend
    ) -> Array:
        return ep_grouped_matmul(x_jobs, table, self.jobs[selection], self.shard_axis, backend)

    def combine(self, selection: int, y: Float[Array, "b s J d"]) -> Float[Array, "b t d"]:
        # Residual streams replicate over expert owners, so their partial outputs sum.
        return ep_combine_jobs(
            y,
            self.jobs[selection],
            self.weights[selection],
            self.shard_axis,
            P(jax.typeof(y).sharding.spec[0], None, None),
        )

    def selected_ci(self, selection: int, y: Array, n_blocks: int) -> SelectedCI:
        jobs = self.jobs[selection]
        values = ep_unsort_jobs(y, jobs, self.shard_axis)
        b, t, k, c = values.shape
        return SelectedCI(values.reshape(b, t, k * c), jobs.top_idx, n_blocks)


Selection = TokenSelection | BlockShardedSelection


def token_selection(
    ids: Int[Array, "r b t k"], weights: Float[Array, "r b t k"], table_size: int
) -> TokenSelection:
    """Every expert local: one token schedule per selection over the flattened tokens."""
    n_selections, *lead, k = ids.shape
    return TokenSelection(
        jobs=tuple(routed_jobs(ids[r].reshape(-1, k), table_size) for r in range(n_selections)),
        weights=tuple(weights[r].reshape(-1, k) for r in range(n_selections)),
        lead=tuple(lead),
    )


def block_sharded_selection(
    ids: Int[Array, "r b t k"],
    weights: Float[Array, "r b t k"],
    table_size: int,
    expert_operands: PlacedRule,
) -> BlockShardedSelection:
    """Experts sharded over the one mesh axis the expert operand row assigns them."""
    assignment = expert_operands.assignment("expert")
    assert len(assignment) == 1, (
        f"expert-parallel CI requires one expert mesh axis, got {assignment}"
    )
    shard_axis = assignment[0]
    n_shards = expert_operands.mesh.shape[shard_axis]
    return BlockShardedSelection(
        jobs=tuple(expert_sharded_jobs(ids[r], table_size, n_shards) for r in range(len(ids))),
        weights=tuple(weights[r] for r in range(len(ids))),
        shard_axis=shard_axis,
    )
