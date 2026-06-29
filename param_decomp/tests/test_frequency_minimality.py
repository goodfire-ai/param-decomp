"""Frequency-minimality penalty `Σ_c f_c·log2(1 + a'·f_c)` (SPEC S7/S8).

`importance_minimality_terms` returns `(lp, freq)`: `lp = Σ_c f_c` (the bare per-token
firing-rate mean) and `freq` the batch-invariant frequency penalty with `a' =
reference_token_count`. These pin the properties that motivate the split from the old
rolled `lp + beta·log2(1 + B·T·f_c)`: batch-invariance, the `f=0 → 0` cutoff, and that
`a' = B·T` reproduces the old implicit-`B·T` value exactly (so coefficients transfer).
"""

import math

import jax.numpy as jnp

from param_decomp.configs import (
    FrequencyMinimalityConfig,
    ImportanceMinimalityLossConfig,
)
from param_decomp.losses import annealed_imp_min_param, imp_min_terms, importance_minimality_terms


def test_closed_form():
    # pnorm=1, eps=0: per_component_sums = column sums; f = sums / n; a' = 8.
    ci = {"a": jnp.array([[1.0, 2.0], [3.0, 4.0]])}  # n=2, sums=[4,6], f=[2,3]
    lp, freq = importance_minimality_terms(ci, jnp.asarray(1.0), eps=0.0, reference_token_count=8)
    assert math.isclose(float(lp), 2.0 + 3.0, rel_tol=1e-6)
    expected = 2.0 * math.log2(1 + 8 * 2.0) + 3.0 * math.log2(1 + 8 * 3.0)
    assert math.isclose(float(freq), expected, rel_tol=1e-6)


def test_zero_frequency_zero_contribution():
    # A component that never fires (f=0) contributes exactly 0 to freq.
    ci = {"a": jnp.array([[0.0, 5.0], [0.0, 5.0]])}  # f = [0, 5]
    _, freq = importance_minimality_terms(ci, jnp.asarray(1.0), eps=0.0, reference_token_count=16)
    expected = 0.0 + 5.0 * math.log2(1 + 16 * 5.0)
    assert math.isclose(float(freq), expected, rel_tol=1e-6)


def test_batch_invariance():
    # Same per-token frequency at two batch sizes => same freq (the whole point of a').
    base = jnp.array([[0.1, 0.4, 0.7]])  # one row of per-token values
    small = {"a": jnp.tile(base, (4, 1))}  # n=4
    large = {"a": jnp.tile(base, (64, 1))}  # n=64, identical f_c
    _, freq_small = importance_minimality_terms(
        small, jnp.asarray(1.0), 0.0, reference_token_count=1024
    )
    _, freq_large = importance_minimality_terms(
        large, jnp.asarray(1.0), 0.0, reference_token_count=1024
    )
    assert jnp.allclose(freq_small, freq_large, rtol=1e-5)


def test_a_prime_bt_reproduces_old_rolled_log_term():
    # Old rolled imp-min: Σ_c f_c + beta·f_c·log2(1 + sum_c), sum_c = f_c·B·T implicit.
    # New split: lp = Σ_c f_c, freq = Σ_c f_c·log2(1 + a'·f_c). With a' = B·T,
    # beta·freq == the old log term exactly. Here B·T = n (the leading count).
    ci = {"a": jnp.array([[0.2, 0.5, 0.9], [0.4, 0.1, 0.7]]), "b": jnp.array([[0.3], [0.6]])}
    p, eps, beta = 2.0, 1e-9, 0.7
    n = 2  # rows per site

    old = jnp.zeros(())
    for v in ci.values():
        sums = ((v + eps) ** p).sum(axis=0)
        mean = sums / n
        old = old + (mean + beta * mean * jnp.log2(1 + sums)).sum()

    lp, freq = importance_minimality_terms(ci, jnp.asarray(p), eps=eps, reference_token_count=n)
    assert jnp.allclose(lp + beta * freq, old, rtol=1e-6)


def test_lp_independent_of_reference_token_count():
    ci = {"a": jnp.array([[0.5, 1.5], [2.5, 3.5]])}
    lp_a, _ = importance_minimality_terms(ci, jnp.asarray(1.5), 1e-6, reference_token_count=32)
    lp_b, _ = importance_minimality_terms(ci, jnp.asarray(1.5), 1e-6, reference_token_count=999)
    assert jnp.allclose(lp_a, lp_b)


def test_dispatch_no_frequency_gives_zero_freq():
    cfg = ImportanceMinimalityLossConfig(coeff=1.0, pnorm=2.0, p_anneal_final_p=2.0)
    assert cfg.frequency is None
    ci = {"a": jnp.array([[0.1, 0.9], [0.4, 0.6]])}
    param = annealed_imp_min_param(jnp.asarray(0.0), 100, cfg)
    lp, freq = imp_min_terms(ci, cfg, param)
    assert float(freq) == 0.0
    assert float(lp) > 0.0


def test_dispatch_with_frequency_matches_direct():
    cfg = ImportanceMinimalityLossConfig(
        coeff=1.0,
        pnorm=2.0,
        p_anneal_final_p=2.0,
        frequency=FrequencyMinimalityConfig(coeff=0.5, reference_token_count=128),
    )
    ci = {"a": jnp.array([[0.1, 0.9], [0.4, 0.6]])}
    param = annealed_imp_min_param(jnp.asarray(0.0), 100, cfg)
    lp, freq = imp_min_terms(ci, cfg, param)
    lp_d, freq_d = importance_minimality_terms(ci, param, cfg.eps, reference_token_count=128)
    assert jnp.allclose(lp, lp_d)
    assert jnp.allclose(freq, freq_d)
