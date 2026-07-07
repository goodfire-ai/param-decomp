"""Host-side math of compare_runs: matching recovery, Pearson-from-stats, Jaccard."""

import numpy as np

from param_decomp_lab.experiments.lm.compare_runs import (
    _hungarian_max,
    _jaccard_under_perm,
    _pearson_from_stats,
    _weight_similarity,
)


def _random_vu(rng: np.random.Generator, d_in: int, c: int, d_out: int):
    return (
        rng.standard_normal((d_in, c)).astype(np.float32),
        rng.standard_normal((c, d_out)).astype(np.float32),
    )


def test_weight_matching_recovers_permutation_and_joint_sign_flips():
    rng = np.random.default_rng(0)
    d_in, c, d_out = 16, 24, 12
    v_a, u_a = _random_vu(rng, d_in, c, d_out)
    perm = rng.permutation(c)
    signs = rng.choice([-1.0, 1.0], size=c).astype(np.float32)
    noise = 1.0 + 0.01 * rng.standard_normal((d_in, c)).astype(np.float32)
    v_b = (v_a * signs)[:, perm] * noise
    u_b = (u_a * signs[:, None])[perm]

    s = _weight_similarity((v_a, u_a), (v_b, u_b))
    recovered, matched = _hungarian_max(s)
    assert (perm[recovered] == np.arange(c)).all() or (recovered == np.argsort(perm)).all()
    assert matched.min() > 0.95

    null = _weight_similarity((v_a, u_a), (v_b, u_b), u_perm_b=rng.permutation(c))
    _, matched_null = _hungarian_max(null)
    assert matched_null.mean() < 0.5 * matched.mean()


def test_pearson_from_stats_matches_corrcoef():
    rng = np.random.default_rng(1)
    n, c_a, c_b = 500, 5, 7
    a = rng.standard_normal((n, c_a))
    b = rng.standard_normal((n, c_b))
    b[:, 0] = a[:, 0]  # a perfectly correlated pair
    b[:, 1] = 0.0  # a zero-variance component

    r = _pearson_from_stats(a.sum(0), (a**2).sum(0), b.sum(0), (b**2).sum(0), a.T @ b, n)
    expected = np.corrcoef(a.T, b.T)[:c_a, c_a:]
    varying = np.ones(c_b, bool)
    varying[1] = False
    np.testing.assert_allclose(r[:, varying], expected[:, varying], atol=1e-10)
    assert (r[:, 1] == 0.0).all()


def test_jaccard_under_perm():
    fire_a = np.array([10.0, 0.0])
    fire_b = np.array([5.0, 10.0])
    joint = np.array([[2.0, 0.0], [0.0, 0.0]])
    perm = np.array([0, 1])
    jaccard, independence = _jaccard_under_perm(fire_a, fire_b, joint, perm, n=100)
    np.testing.assert_allclose(jaccard, [2.0 / 13.0, 0.0])
    d_a, d_b = 0.1, 0.05
    np.testing.assert_allclose(independence[0], d_a * d_b / (d_a + d_b - d_a * d_b), atol=1e-12)
    assert independence[1] == 0.0
