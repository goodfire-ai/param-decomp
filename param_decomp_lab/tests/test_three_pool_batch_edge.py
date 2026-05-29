"""Unit tests for ``BatchEdge`` — the cross-pool batch-slice geometry.

CPU-only, no torch.distributed. Exercises both fan directions: CI-coarse
(``n_ci`` divides ``n_down``) and CI-fine / inverted (``n_down`` divides
``n_ci``), plus the square case. The invariant under test is that the
``(ci_slice, down_slice)`` overlaps tile each pool's local batch tensor exactly
once and agree on the global rows they cover from both sides — which is what
makes the cross-pool sends/recvs stitch back to the single-pool batch.
"""

import pytest

from param_decomp_lab.three_pool.layout import BatchEdge

# (n_ci, n_down, batch_global), spanning coarse / square / inverted regimes.
EDGE_CASES = [
    (2, 4, 8),  # CI coarse (legacy forward path), fanout 2
    (2, 2, 8),  # square
    (4, 2, 8),  # CI fine (inverted), fanout 2
    (4, 4, 8),  # square, all-equal
    (8, 4, 8),  # CI fine, fanout 2, b_down > b_ci
    (4, 8, 8),  # CI coarse, fanout 2
    (1, 4, 8),  # single CI rank fans to all
    (6, 6, 12),  # square non-power-of-two
]


@pytest.mark.parametrize(("n_ci", "n_down", "batch_global"), EDGE_CASES)
def test_overlaps_tile_local_tensors_and_agree_globally(
    n_ci: int, n_down: int, batch_global: int
) -> None:
    edge = BatchEdge(n_ci=n_ci, n_down=n_down, batch_global=batch_global)

    # Every downstream slice is fully tiled by its CI overlaps (no gaps/overlaps).
    for j in range(n_down):
        covered = sorted(
            (s.start, s.stop)
            for s in (edge.overlap_within_down(i, j) for i in edge.ci_slices_for_down_slice(j))
        )
        cursor = 0
        for start, stop in covered:
            assert start == cursor, f"gap/overlap in down slice {j}: {covered}"
            cursor = stop
        assert cursor == edge.b_down

    # Every CI slice is fully tiled by its downstream overlaps.
    for i in range(n_ci):
        covered = sorted(
            (s.start, s.stop)
            for s in (edge.overlap_within_ci(i, j) for j in edge.down_slices_for_ci_slice(i))
        )
        cursor = 0
        for start, stop in covered:
            assert start == cursor, f"gap/overlap in CI slice {i}: {covered}"
            cursor = stop
        assert cursor == edge.b_ci

    # Each overlap covers the SAME global rows from both sides, with matching length.
    for i in range(n_ci):
        for j in edge.down_slices_for_ci_slice(i):
            ci_sub = edge.overlap_within_ci(i, j)
            down_sub = edge.overlap_within_down(i, j)
            assert (ci_sub.stop - ci_sub.start) == (down_sub.stop - down_sub.start)
            ci_global = (i * edge.b_ci + ci_sub.start, i * edge.b_ci + ci_sub.stop)
            down_global = (j * edge.b_down + down_sub.start, j * edge.b_down + down_sub.stop)
            assert ci_global == down_global


@pytest.mark.parametrize(("n_ci", "n_down", "batch_global"), EDGE_CASES)
def test_pairing_is_symmetric(n_ci: int, n_down: int, batch_global: int) -> None:
    """``j in down_slices_for_ci_slice(i)`` iff ``i in ci_slices_for_down_slice(j)``."""
    edge = BatchEdge(n_ci=n_ci, n_down=n_down, batch_global=batch_global)
    for i in range(n_ci):
        for j in edge.down_slices_for_ci_slice(i):
            assert i in edge.ci_slices_for_down_slice(j)
    for j in range(n_down):
        for i in edge.ci_slices_for_down_slice(j):
            assert j in edge.down_slices_for_ci_slice(i)


def test_fanout_direction() -> None:
    coarse = BatchEdge(n_ci=2, n_down=4, batch_global=8)
    assert coarse.ci_is_coarse and coarse.fanout == 2
    # One CI rank feeds 2 down ranks; each down rank reads from 1 CI rank.
    assert coarse.down_slices_for_ci_slice(0) == (0, 1)
    assert coarse.ci_slices_for_down_slice(3) == (1,)

    fine = BatchEdge(n_ci=4, n_down=2, batch_global=8)
    assert not fine.ci_is_coarse and fine.fanout == 2
    # One down rank gathers from 2 CI ranks; each CI rank feeds 1 down rank.
    assert fine.ci_slices_for_down_slice(0) == (0, 1)
    assert fine.down_slices_for_ci_slice(3) == (1,)


def test_rejects_non_cross_divisible() -> None:
    with pytest.raises(AssertionError):
        BatchEdge(n_ci=6, n_down=4, batch_global=12)
