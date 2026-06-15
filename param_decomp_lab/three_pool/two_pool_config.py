"""Serializable, normalized topology for 2-pool training.

The 2-pool variant merges the 3-pool CI and PPGD pools into a single **Pool A**
(adversary + CI fn co-located on the same ranks, same batch slice), leaving the
chunkwise pool (**Pool B**) untouched. So the only authored per-rank batches are
Pool A's and the chunkwise pool's, plus the site→chunk split.

  * **Pool A** — replicated CI fn + full V/U replica + persistent PPGD sources, DP
    across batch. ``n_a = batch / pool_a.per_rank_batch``. The CI forward and the
    adversary forward run on the SAME rank and SAME batch slice, so the CI↔PPGD
    cross-pool edge (mask send + g_CI return) of the 3-pool design disappears
    entirely — the adversary's g_CI is the local ``.grad`` of the CI forward's own
    ``lower_leaky``.
  * **Chunkwise pool** — V/U sharded into chunks of sites; each chunk replicated
    across ``chunk_dp = batch / chunkwise.per_rank_batch`` ranks (DDP within chunk).
    Unchanged from the 3-pool.

The single surviving cross-pool edge is Pool A ↔ chunkwise: masks out + g_CI back
(the old CI↔chunk edge) and the V/U sync (updated V/U from B's owners → A's replica;
V/U grads A → B). Its batch geometry uses the same cross-divisibility rule
(``pool_a.per_rank_batch`` and ``chunkwise.per_rank_batch`` cross-divide).

``resolve(ordered_sites, batch_size)`` returns the pure ``TwoPoolResolvedLayout``;
``two_pool_optimize.py`` builds the runtime ``Chunk`` objects from it. Canonical
order: chunks first (rank 0 = chunk-0 leader), then Pool A.
"""

from dataclasses import dataclass
from typing import Self

from pydantic import model_validator

from param_decomp_config.base import BaseConfig
from param_decomp_lab.three_pool.config import ChunkwiseSpec, PoolSpec


@dataclass(frozen=True)
class TwoPoolResolvedLayout:
    """Pure canonical rank assignment derived from a ``TwoPoolTopology`` + the
    expanded site list + the global batch. ``chunks`` is per-chunk ``(ranks, sites)``.

    Canonical order: chunks first (rank 0 = chunk-0 leader), then Pool A.
    """

    pool_a_ranks: tuple[int, ...]
    chunks: tuple[tuple[tuple[int, ...], tuple[str, ...]], ...]
    world_size: int


class TwoPoolTopology(BaseConfig):
    """Normalized 2-pool topology. Pairs with a regular ``PDConfig``.

    Authors per-rank batch for Pool A (CI + adversary) and the chunkwise pool, plus
    the site→chunk split; every rank id is derived by ``resolve`` in canonical order.
    The cross-divisibility constraint keeps the Pool A ↔ chunk batch overlap a whole,
    aligned sub-slice.
    """

    pool_a: PoolSpec
    chunkwise: ChunkwiseSpec
    use_fused_kl: bool = True

    @model_validator(mode="after")
    def validate_cross_divisibility(self) -> Self:
        bl_a = self.pool_a.per_rank_batch
        bl_chunk = self.chunkwise.per_rank_batch
        # n_a / chunk_dp are batch/bl; they cross-divide iff the bl's do. Cross-divisible
        # per-rank batches ⇒ every Pool A↔chunk batch overlap is a whole, aligned
        # sub-slice (one-to-K fan-out in either direction).
        assert bl_chunk % bl_a == 0 or bl_a % bl_chunk == 0, (
            f"pool_a.per_rank_batch ({bl_a}) and chunkwise.per_rank_batch ({bl_chunk}) must "
            f"cross-divide (one divides the other) so each Pool A↔chunk batch overlap is a "
            f"whole sub-slice. Tip: make one a multiple of the other."
        )
        return self

    def resolve(self, ordered_sites: list[str], batch_size: int) -> TwoPoolResolvedLayout:
        """Derive the canonical rank assignment for ``ordered_sites`` + ``batch_size``.

        Canonical order: chunk-0 ranks (rank 0 = chunk-0 leader), …, chunk-N ranks,
        Pool A ranks. Each per-rank batch must divide ``batch_size``.
        """
        assert batch_size % self.pool_a.per_rank_batch == 0
        assert batch_size % self.chunkwise.per_rank_batch == 0
        assert ordered_sites, "resolve needs at least one decomposed site"

        n_a = batch_size // self.pool_a.per_rank_batch
        chunk_dp = batch_size // self.chunkwise.per_rank_batch
        spc = self.chunkwise.sites_per_chunk or len(ordered_sites)
        site_chunks = [ordered_sites[i : i + spc] for i in range(0, len(ordered_sites), spc)]
        assert len(site_chunks) == self.chunkwise.n_chunks, (
            f"chunkwise.n_chunks ({self.chunkwise.n_chunks}) != actual chunk count "
            f"({len(site_chunks)}) for {len(ordered_sites)} sites at sites_per_chunk={spc}"
        )

        r = 0
        chunks: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
        for sites in site_chunks:
            chunks.append((tuple(range(r, r + chunk_dp)), tuple(sites)))
            r += chunk_dp
        pool_a_ranks = tuple(range(r, r + n_a))
        r += n_a
        return TwoPoolResolvedLayout(
            pool_a_ranks=pool_a_ranks,
            chunks=tuple(chunks),
            world_size=r,
        )

    def world_size(self, batch_size: int, n_chunks: int) -> int:
        """World size from per-rank batches + chunk count, without resolving sites."""
        n_a = batch_size // self.pool_a.per_rank_batch
        chunk_dp = batch_size // self.chunkwise.per_rank_batch
        return n_chunks * chunk_dp + n_a
