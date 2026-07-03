"""smooth-L0 (Geman–McClure) importance-minimality penalty (SPEC S7/S8/S9').

The penalty shares the per-site `lp` mean (plus the optional frequency term) with the
`L_p` penalty and differs ONLY in the per-value shape `phi_gamma(c) = c^2/(c^2+gamma^2)`.
These checks pin the properties that motivate it over `L_p`: flat at the origin
(`phi'(0)=0`, no singularity to clip), bounded gradient (`|phi'| <= 0.65/gamma`),
redescent for clearly-on components, and the half-saturation crossover `phi(gamma) = 1/2`.
"""

import jax
import jax.numpy as jnp

from param_decomp.configs import (
    FrequencyMinimalityConfig,
    SmoothL0ImportanceMinimalityLossConfig,
)
from param_decomp.losses import (
    annealed_gamma,
    annealed_imp_min_param,
    imp_min_terms,
    smooth_l0_importance_minimality_terms,
)


def _phi(c: jax.Array, gamma: float) -> jax.Array:
    return c**2 / (c**2 + gamma**2)


def test_phi_shape_invariants():
    for gamma in (1.0, 0.1):
        assert float(_phi(jnp.array(0.0), gamma)) == 0.0  # off -> exactly 0
        assert abs(float(_phi(jnp.array(gamma), gamma)) - 0.5) < 1e-6  # half-saturation
        assert float(_phi(jnp.array(10.0 * gamma), gamma)) > 0.99  # clearly-on -> ~1


def test_phi_gradient_flat_at_origin_and_bounded():
    """phi'(0) = 0 (no L_p cliff) and the peak |phi'| ~ 0.65/gamma sits at c = gamma/sqrt(3)."""
    for gamma in (1.0, 0.1):
        dphi = jax.grad(lambda c, g=gamma: _phi(c, g))
        assert float(dphi(jnp.array(0.0))) == 0.0
        cs = jnp.linspace(0.0, 5.0 * gamma, 4096)
        grads = jnp.abs(jax.vmap(dphi)(cs))
        peak = float(grads.max())
        assert peak <= 0.65 / gamma + 1e-3
        c_peak = float(cs[jnp.argmax(grads)])
        assert abs(c_peak - gamma / jnp.sqrt(3.0)) < 0.02 * gamma
        # redescent: gradient at a clearly-on point is far below the peak.
        assert float(dphi(jnp.array(5.0 * gamma))) < 0.2 * peak


def test_terms_match_manual_per_site_structure():
    ci = {
        "a": jnp.array([[0.0, 0.5, 1.0], [0.2, 0.0, 0.9]]),
        "b": jnp.array([[0.3], [0.7]]),
    }
    gamma = 0.1
    n_positions = 2  # both sites have 2 rows; a' = B·T reproduces the old `log2(1 + sum)`
    lp, freq = smooth_l0_importance_minimality_terms(
        ci, jnp.asarray(gamma), reference_token_count=n_positions
    )

    exp_lp = jnp.zeros(())
    exp_freq = jnp.zeros(())
    for v in ci.values():
        sums = _phi(v, gamma).sum(axis=0)
        means = sums / v.shape[0]
        exp_lp = exp_lp + means.sum()
        exp_freq = exp_freq + (means * jnp.log2(1.0 + n_positions * means)).sum()
    assert jnp.allclose(lp, exp_lp)
    assert jnp.allclose(freq, exp_freq)


def test_anneal_and_dispatch():
    cfg = SmoothL0ImportanceMinimalityLossConfig(
        coeff=2e-4,
        gamma=1.0,
        frequency=FrequencyMinimalityConfig(coeff=1e-4, reference_token_count=64),
        gamma_anneal_start_frac=0.0,
        gamma_anneal_final_gamma=0.1,
        gamma_anneal_end_frac=1.0,
    )
    total = 100
    assert abs(float(annealed_gamma(jnp.asarray(0.0), total, cfg)) - 1.0) < 1e-6
    assert abs(float(annealed_gamma(jnp.asarray(50.0), total, cfg)) - 0.55) < 1e-6
    assert abs(float(annealed_gamma(jnp.asarray(total), total, cfg)) - 0.1) < 1e-6

    ci = {"a": jnp.array([[0.0, 0.5, 1.0], [0.2, 0.0, 0.9]])}
    param = annealed_imp_min_param(jnp.asarray(float(total)), total, cfg)
    via_dispatch = imp_min_terms(ci, cfg, param)
    direct = smooth_l0_importance_minimality_terms(ci, param, reference_token_count=64)
    assert jnp.allclose(via_dispatch[0], direct[0])
    assert jnp.allclose(via_dispatch[1], direct[1])
