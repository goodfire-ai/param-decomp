"""Unit tests for ``ThreePoolTopology.resolve`` — the canonical rank assignment.

CPU-only, no torch. Pins that the resolver derives ranks in canonical order (chunks
first → rank 0 is chunk-0's leader, then CI, then PPGD) and that the derived world
size matches the b512 Llama production layout that used to be hand-authored.
"""

from param_decomp_lab.three_pool.config import (
    ChunkwiseSpec,
    PoolSpec,
    ThreePoolTopology,
)


def test_resolve_b512_llama_single_chunk() -> None:
    """The committed b512 Llama-L18 layout (3 MLP sites in one 32-wide chunk, CI 32,
    PPGD 32 → 96 GPUs) is reproduced by the resolver from per-rank batches alone."""
    topo = ThreePoolTopology(
        ci=PoolSpec(per_rank_batch=16),
        ppgd=PoolSpec(per_rank_batch=16),
        chunkwise=ChunkwiseSpec(per_rank_batch=16, sites_per_chunk=None, n_chunks=1),
    )
    sites = [
        "layers.18.mlp.gate_proj",
        "layers.18.mlp.up_proj",
        "layers.18.mlp.down_proj",
    ]
    layout = topo.resolve(sites, batch_size=512)

    assert layout.world_size == 96
    assert layout.chunks == ((tuple(range(0, 32)), tuple(sites)),)
    assert layout.ci_ranks == tuple(range(32, 64))
    assert layout.ppgd_ranks == tuple(range(64, 96))
    # Rank 0 is always chunk-0's first rank (the rank-0 convention, by construction).
    assert layout.chunks[0][0][0] == 0


def test_resolve_multi_chunk_gpt2_style() -> None:
    """A multi-chunk case: 4 sites split into 2 chunks of 2, chunk_dp=2, n_ci=n_ppgd=1
    over batch 16 → canonical ranks chunk0 (0,1), chunk1 (2,3), CI 4, PPGD 5."""
    topo = ThreePoolTopology(
        ci=PoolSpec(per_rank_batch=16),
        ppgd=PoolSpec(per_rank_batch=16),
        chunkwise=ChunkwiseSpec(per_rank_batch=8, sites_per_chunk=2, n_chunks=2),
    )
    sites = ["h.0.attn.q_proj", "h.0.attn.k_proj", "h.1.attn.q_proj", "h.1.attn.k_proj"]
    layout = topo.resolve(sites, batch_size=16)

    assert layout.world_size == 6
    assert layout.chunks == (
        ((0, 1), ("h.0.attn.q_proj", "h.0.attn.k_proj")),
        ((2, 3), ("h.1.attn.q_proj", "h.1.attn.k_proj")),
    )
    assert layout.ci_ranks == (4,)
    assert layout.ppgd_ranks == (5,)
    assert layout.chunks[0][0][0] == 0


def test_resolve_ranks_partition_world_with_no_gaps() -> None:
    """The derived rank ids tile ``range(world_size)`` exactly once — no overlap, dup,
    or gap (the invalid states the old hand-authored rank lists could express)."""
    topo = ThreePoolTopology(
        ci=PoolSpec(per_rank_batch=8),
        ppgd=PoolSpec(per_rank_batch=16),
        chunkwise=ChunkwiseSpec(per_rank_batch=4, sites_per_chunk=1, n_chunks=3),
    )
    sites = ["a", "b", "c"]
    layout = topo.resolve(sites, batch_size=16)

    all_ranks = [r for ranks, _ in layout.chunks for r in ranks]
    all_ranks += list(layout.ci_ranks)
    all_ranks += list(layout.ppgd_ranks)
    assert sorted(all_ranks) == list(range(layout.world_size))
