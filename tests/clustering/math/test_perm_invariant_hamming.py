from itertools import permutations

import numpy as np
import pytest

from param_decomp.clustering.math.perm_invariant_hamming import perm_invariant_hamming_matrix

# pyright complains about the types when calling perm_invariant_hamming
# pyright: reportCallIssue=false


def brute_force_min_hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Exhaustive check for small k."""
    k = int(max(a.max(), b.max()) + 1)
    best = len(a)
    for perm in permutations(range(k)):
        mapping = np.array(perm)
        best = min(best, int((mapping[a] != b).sum()))
    return best


def test_identity() -> None:
    """a == b should give distance 0."""
    a = np.array([0, 1, 2, 1, 0])
    b = a.copy()
    X = np.array([a, b])
    D = perm_invariant_hamming_matrix(X)
    # Distance between row 1 and row 0 should be 0
    assert D[1, 0] == 0


def test_all_one_group() -> None:
    """All rows belong to one group in both arrays (possibly different labels)."""
    a = np.zeros(10, dtype=int)
    b = np.ones(10, dtype=int)  # different label but identical grouping
    X = np.array([a, b])
    D = perm_invariant_hamming_matrix(X)
    assert D[1, 0] == 0


def test_permuted_labels() -> None:
    a = np.array([0, 2, 1, 1, 0])
    b = np.array([1, 0, 0, 2, 1])
    X = np.array([a, b])
    D = perm_invariant_hamming_matrix(X)
    assert D[1, 0] == 1


def test_swap_two_labels() -> None:
    a = np.array([0, 0, 1, 1])
    b = np.array([1, 1, 0, 0])
    X = np.array([a, b])
    D = perm_invariant_hamming_matrix(X)
    assert D[1, 0] == 0


def test_random_small_bruteforce() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = 7
        k = 3
        a = rng.integers(0, k, size=n)
        b = rng.integers(0, k, size=n)
        X = np.array([a, b])
        D = perm_invariant_hamming_matrix(X)
        d_alg = D[1, 0]
        d_true = brute_force_min_hamming(a, b)
        assert d_alg == d_true


def test_shape_mismatch() -> None:
    a = np.array([0, 1, 2])
    b = np.array([0, 1])
    with pytest.raises((ValueError, IndexError)):
        # This should fail when trying to create the matrix due to shape mismatch
        X = np.array([a, b])
        perm_invariant_hamming_matrix(X)


def test_matrix_multiple_pairs() -> None:
    """Test the matrix function with multiple label vectors."""
    a = np.array([0, 0, 1, 1])
    b = np.array([2, 2, 3, 3])  # Should be distance 0 (perfect mapping)
    c = np.array([0, 1, 0, 1])  # Should be distance 2 from both a and b
    X = np.array([a, b, c])
    D = perm_invariant_hamming_matrix(X)

    assert D[1, 0] == 0  # a and b should have distance 0
    assert D[2, 0] == 2  # a and c should have distance 2
    assert D[2, 1] == 2  # b and c should have distance 2


def test_matrix_upper_triangle_nan() -> None:
    """Test that upper triangle and diagonal are NaN."""
    a = np.array([0, 1, 0])
    b = np.array([1, 0, 1])
    c = np.array([0, 0, 1])
    X = np.array([a, b, c])
    D = perm_invariant_hamming_matrix(X)

    # Diagonal should be NaN
    assert np.isnan(D[0, 0])
    assert np.isnan(D[1, 1])
    assert np.isnan(D[2, 2])

    # Upper triangle should be NaN
    assert np.isnan(D[0, 1])
    assert np.isnan(D[0, 2])
    assert np.isnan(D[1, 2])

    # Lower triangle should have actual distances
    assert not np.isnan(D[1, 0])
    assert not np.isnan(D[2, 0])
    assert not np.isnan(D[2, 1])


def test_unused_labels() -> None:
    """Test when arrays don't use all labels 0..k-1."""
    a = np.array([0, 0, 3, 3])  # skips 1, 2
    b = np.array([1, 1, 2, 2])
    X = np.array([a, b])
    D = perm_invariant_hamming_matrix(X)
    assert D[1, 0] == 0
