"""Smooth-L0 (Geman–McClure) importance-minimality penalty (SPEC S7/S8/S9).

The penalty sums per-component mean activity and optionally adds a frequency term. These
checks pin the properties that motivate it over the retired `L_p` penalty: flat at the
origin (`phi'(0)=0`, no singularity to clip), bounded gradient
(`|phi'| <= 0.65/gamma`),
redescent for clearly-on components, and the half-saturation crossover `phi(gamma) = 1/2`.
"""

import jax
import jax.numpy as jnp

from param_decomp.core.configs import (
    FrequencyMinimalityConfig,
    ImportanceMinimalityLossConfig,
)
from param_decomp.core.losses import (
    imp_min_terms,
    importance_minimality_terms,
    scheduled_value_at,
)
from param_decomp.core.schedule import Knot, ScheduleConfig, get_scheduled_value


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
    activity, freq = importance_minimality_terms(
        ci, jnp.asarray(gamma), reference_datapoint_count=n_positions
    )

    expected_activity = jnp.zeros(())
    exp_freq = jnp.zeros(())
    for v in ci.values():
        sums = _phi(v, gamma).sum(axis=0)
        means = sums / v.shape[0]
        expected_activity = expected_activity + means.sum()
        exp_freq = exp_freq + (means * jnp.log2(1.0 + n_positions * means)).sum()
    assert jnp.allclose(activity, expected_activity)
    assert jnp.allclose(freq, exp_freq)


def test_anneal_and_dispatch():
    cfg = ImportanceMinimalityLossConfig(
        coeff=2e-4,
        gamma=ScheduleConfig(max_val=1.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.1))),
        frequency=FrequencyMinimalityConfig(coeff=1e-4, reference_datapoint_count=64),
    )
    total = 100
    for step in (0, 50, total - 1):
        param = scheduled_value_at(jnp.asarray(step / (total - 1), jnp.float32), cfg.gamma)
        expected = get_scheduled_value(step, total, cfg.gamma)
        assert abs(float(param) - expected) < 1e-6
    assert abs(get_scheduled_value(0, total, cfg.gamma) - 1.0) < 1e-6
    assert abs(get_scheduled_value(total - 1, total, cfg.gamma) - 0.1) < 1e-6

    last = total - 1
    ci = {"a": jnp.array([[0.0, 0.5, 1.0], [0.2, 0.0, 0.9]])}
    param = scheduled_value_at(jnp.asarray(last / (total - 1), jnp.float32), cfg.gamma)
    via_dispatch = imp_min_terms(ci, cfg, param)
    direct = importance_minimality_terms(ci, param, reference_datapoint_count=64)
    assert jnp.allclose(via_dispatch[0], direct[0])
    assert jnp.allclose(via_dispatch[1], direct[1])
