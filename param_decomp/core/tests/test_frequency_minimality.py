"""Frequency-minimality penalty `Σ_c Φ(f_c)`, `Φ(f) = f·log2(1 + a'·f)` (SPEC S7/S8/S8'').

The closed-form tests pin the properties that motivate the split from the old rolled
`lp + beta·log2(1 + B·T·f_c)`: batch-invariance, the `f=0 → 0` cutoff, and that
`a' = B·T` reproduces the old implicit-`B·T` value exactly (so coefficients transfer).
The EMA tests pin S8'': debiased smoothing of `f_c` (step-0 identity, closed form,
settling near `Φ(mean f)` under alternating batches) and the surrogate gradient
(full single-batch scale at stationarity, zero gradient into the EMA state).
"""

import math

import jax
import jax.numpy as jnp
import pytest

from param_decomp.core.configs import (
    FrequencyMinimalityConfig,
    ImportanceMinimalityLossConfig,
)
from param_decomp.core.losses import (
    BatchFrequency,
    EmaFrequency,
    annealed_imp_min_param,
    ema_frequency_penalty,
    imp_min_terms,
    importance_minimality_terms,
    lp_term,
    per_component_frequencies,
    resolve_frequency,
)
from param_decomp.core.schedule import ScheduleConfig


def test_legacy_reference_token_count_alias():
    cfg = FrequencyMinimalityConfig.model_validate({"coeff": 0.5, "reference_token_count": 32768})
    assert cfg.reference_datapoint_count == 32768
    assert cfg.model_dump() == {
        "coeff": 0.5,
        "reference_datapoint_count": 32768,
        "ema_halflife_steps": None,
    }


def test_closed_form():
    # pnorm=1, eps=0: per_component_sums = column sums; f = sums / n; a' = 8.
    ci = {"a": jnp.array([[1.0, 2.0], [3.0, 4.0]])}  # n=2, sums=[4,6], f=[2,3]
    lp, freq = importance_minimality_terms(
        ci, jnp.asarray(1.0), eps=0.0, reference_datapoint_count=8
    )
    assert math.isclose(float(lp), 2.0 + 3.0, rel_tol=1e-6)
    expected = 2.0 * math.log2(1 + 8 * 2.0) + 3.0 * math.log2(1 + 8 * 3.0)
    assert math.isclose(float(freq), expected, rel_tol=1e-6)


def test_zero_frequency_zero_contribution():
    # A component that never fires (f=0) contributes exactly 0 to freq.
    ci = {"a": jnp.array([[0.0, 5.0], [0.0, 5.0]])}  # f = [0, 5]
    _, freq = importance_minimality_terms(
        ci, jnp.asarray(1.0), eps=0.0, reference_datapoint_count=16
    )
    expected = 0.0 + 5.0 * math.log2(1 + 16 * 5.0)
    assert math.isclose(float(freq), expected, rel_tol=1e-6)


def test_batch_invariance():
    # Same per-token frequency at two batch sizes => same freq (the whole point of a').
    base = jnp.array([[0.1, 0.4, 0.7]])  # one row of per-token values
    small = {"a": jnp.tile(base, (4, 1))}  # n=4
    large = {"a": jnp.tile(base, (64, 1))}  # n=64, identical f_c
    _, freq_small = importance_minimality_terms(
        small, jnp.asarray(1.0), 0.0, reference_datapoint_count=1024
    )
    _, freq_large = importance_minimality_terms(
        large, jnp.asarray(1.0), 0.0, reference_datapoint_count=1024
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

    lp, freq = importance_minimality_terms(ci, jnp.asarray(p), eps=eps, reference_datapoint_count=n)
    assert jnp.allclose(lp + beta * freq, old, rtol=1e-6)


def test_lp_independent_of_reference_datapoint_count():
    ci = {"a": jnp.array([[0.5, 1.5], [2.5, 3.5]])}
    lp_a, _ = importance_minimality_terms(ci, jnp.asarray(1.5), 1e-6, reference_datapoint_count=32)
    lp_b, _ = importance_minimality_terms(ci, jnp.asarray(1.5), 1e-6, reference_datapoint_count=999)
    assert jnp.allclose(lp_a, lp_b)


def test_dispatch_no_frequency_gives_zero_freq():
    cfg = ImportanceMinimalityLossConfig(coeff=1.0, pnorm=ScheduleConfig.constant(2.0))
    assert cfg.frequency is None
    ci = {"a": jnp.array([[0.1, 0.9], [0.4, 0.6]])}
    param = annealed_imp_min_param(jnp.asarray(0.0), cfg)
    lp, freq = imp_min_terms(ci, cfg, param)
    assert float(freq) == 0.0
    assert float(lp) > 0.0


def test_frequencies_seam_matches_direct_terms():
    # The train step's two readouts (lp_term + the resolved role's term over one
    # per_component_frequencies pass) equal the closed-form pair.
    cfg = ImportanceMinimalityLossConfig(
        coeff=1.0,
        pnorm=ScheduleConfig.constant(2.0),
        frequency=FrequencyMinimalityConfig(coeff=0.5, reference_datapoint_count=128),
    )
    ci = {"a": jnp.array([[0.1, 0.9], [0.4, 0.6]])}
    param = annealed_imp_min_param(jnp.asarray(0.0), cfg)
    frequencies = per_component_frequencies(ci, cfg, param)
    role = resolve_frequency(cfg.frequency)
    assert isinstance(role, BatchFrequency)
    term = role.term(frequencies)
    lp_d, freq_d = importance_minimality_terms(ci, param, cfg.eps, reference_datapoint_count=128)
    assert jnp.allclose(lp_term(frequencies), lp_d)
    assert jnp.allclose(term.freq, freq_d)


def _ema_cfg(halflife: float) -> ImportanceMinimalityLossConfig:
    return ImportanceMinimalityLossConfig(
        coeff=1.0,
        pnorm=ScheduleConfig.constant(1.0),
        eps=0.0,
        frequency=FrequencyMinimalityConfig(
            coeff=0.5, reference_datapoint_count=64, ema_halflife_steps=halflife
        ),
    )


def test_ema_halflife_cap_refused():
    with pytest.raises(ValueError):
        FrequencyMinimalityConfig(coeff=0.5, reference_datapoint_count=64, ema_halflife_steps=1e7)


def test_ema_finite_at_extreme_halflife():
    # Forming decay = 2^(-1/h) rounds to 1 past h ~ 1e16 (f64) and the subtractive
    # 1-decay^(step+1) cancels to 0 (NaN from the debias division); the direct -ln(2)/h
    # form is stable through the configured cap (1e6), incl. the step-0 identity.
    cfg = _ema_cfg(halflife=1e6)
    ci = {"a": jnp.array([[0.2, 0.8], [0.4, 0.1]])}
    param = annealed_imp_min_param(jnp.asarray(0.0), cfg)
    frequencies = per_component_frequencies(ci, cfg, param)
    ema = {"a": jnp.zeros(2, jnp.float32)}
    role = resolve_frequency(cfg.frequency)
    assert isinstance(role, EmaFrequency)
    term = role.term(frequencies, ema, jnp.asarray(0.0))
    assert jnp.isfinite(term.freq)
    assert jnp.allclose(term.freq, term.freq_batch, rtol=1e-4)


def test_ema_long_scan_rounding_bounded():
    # fp32 rounding accumulated over a full run: 400k constant-f updates of the real
    # recurrence, debiased exactly in f64. Measured ~3e-5 at the 1e6 cap and ~4e-4 at
    # 1e4 (fully stationary after 40 halflives); at 1e9 the drift reached 0.4%, which
    # is why the cap sits at 1e6.
    f = jnp.full((2,), 0.5, jnp.float32)
    steps = 400_000
    for halflife, bound in ((1e6, 1e-4), (1e4, 1e-3)):
        ema = jax.lax.fori_loop(
            0,
            steps,
            lambda i, ema, h=halflife: ema_frequency_penalty(
                {"a": f}, {"a": ema}, i.astype(jnp.float32), h, 64
            )[1]["a"],
            jnp.zeros((2,), jnp.float32),
        )
        debias = -math.expm1(-math.log(2.0) / halflife * steps)
        assert jnp.allclose(ema / debias, 0.5, rtol=bound), (halflife, ema / debias)


def test_ema_stationary_equals_batch_penalty():
    # Constant batches: ema_t = (1 - decay^t)·f, so the debiased f̂ is f at EVERY step.
    cfg = _ema_cfg(halflife=8.0)
    ci = {"a": jnp.array([[0.2, 0.8], [0.4, 0.1]])}
    param = annealed_imp_min_param(jnp.asarray(0.0), cfg)
    frequencies = per_component_frequencies(ci, cfg, param)
    ema = {name: jnp.zeros(v.shape[-1], jnp.float32) for name, v in ci.items()}
    role = resolve_frequency(cfg.frequency)
    assert isinstance(role, EmaFrequency)
    for step in range(6):
        term = role.term(frequencies, ema, jnp.asarray(float(step)))
        ema = term.new_freq_ema
        assert jnp.allclose(term.freq, term.freq_batch, rtol=1e-5), step


def test_ema_matches_closed_form():
    # Varying frequencies: ema_t = (1-decay)·Σ_i decay^(t-i)·f_i, f̂ debiased by 1-decay^(t+1).
    halflife, a = 4.0, 32
    decay = 0.5 ** (1.0 / halflife)
    fs = [jnp.array([0.1, 0.5]), jnp.array([0.3, 0.2]), jnp.array([0.05, 0.9])]
    ema = {"a": jnp.zeros(2, jnp.float32)}
    freq = jnp.zeros((), jnp.float32)
    for step, f in enumerate(fs):
        freq, ema = ema_frequency_penalty({"a": f}, ema, jnp.asarray(float(step)), halflife, a)
    expected_ema = (1 - decay) * sum(decay ** (2 - i) * f for i, f in enumerate(fs))
    assert jnp.allclose(ema["a"], expected_ema, rtol=1e-6)
    f_hat = expected_ema / (1 - decay**3)
    assert jnp.allclose(freq, jnp.sum(f_hat * jnp.log2(1 + a * f_hat)), rtol=1e-6)


def test_ema_smooths_toward_mean():
    # Alternating high/low frequencies: the smoothed penalty settles near Φ(mean f),
    # far from either single-batch penalty.
    halflife, a = 16.0, 1024

    def phi(f: jax.Array) -> jax.Array:
        return jnp.sum(f * jnp.log2(1 + a * f))

    f_hi, f_lo = jnp.array([0.2]), jnp.array([0.0])
    f_mean = (f_hi + f_lo) / 2
    ema = {"a": jnp.zeros(1, jnp.float32)}
    freq = jnp.zeros((), jnp.float32)
    for step in range(300):
        f = f_hi if step % 2 == 0 else f_lo
        freq, ema = ema_frequency_penalty({"a": f}, ema, jnp.asarray(float(step)), halflife, a)
    assert abs(float(freq - phi(f_mean))) < 0.1 * float(phi(f_hi) - phi(f_mean))


def test_ema_gradient_matches_unsmoothed_at_stationarity():
    # At f̂ = f_batch (step 0, zero-init ema) the surrogate's gradient wrt ci equals the
    # un-smoothed penalty's gradient — full scale, no shrink by 1-decay.
    cfg = _ema_cfg(halflife=50.0)
    ci_val = jnp.array([[0.1, 0.9], [0.4, 0.6]])
    param = annealed_imp_min_param(jnp.asarray(0.0), cfg)
    role = resolve_frequency(cfg.frequency)
    assert isinstance(role, EmaFrequency)

    def ema_freq(ci: jax.Array) -> jax.Array:
        frequencies = per_component_frequencies({"a": ci}, cfg, param)
        return role.term(frequencies, {"a": jnp.zeros(2)}, jnp.asarray(0.0)).freq

    def batch_freq(ci: jax.Array) -> jax.Array:
        _, freq = importance_minimality_terms({"a": ci}, param, 0.0, reference_datapoint_count=64)
        return freq

    assert jnp.allclose(jax.grad(ema_freq)(ci_val), jax.grad(batch_freq)(ci_val), rtol=1e-5)


def test_ema_state_carries_no_gradient():
    # The CI fn must not be able to steer the frequency estimate: new_ema is grad-free.
    cfg = _ema_cfg(halflife=50.0)
    param = annealed_imp_min_param(jnp.asarray(0.0), cfg)
    role = resolve_frequency(cfg.frequency)
    assert isinstance(role, EmaFrequency)

    def ema_mass(ci: jax.Array) -> jax.Array:
        frequencies = per_component_frequencies({"a": ci}, cfg, param)
        ft = role.term(frequencies, {"a": jnp.zeros(2)}, jnp.asarray(0.0))
        return jnp.sum(ft.new_freq_ema["a"])

    grad = jax.grad(ema_mass)(jnp.array([[0.1, 0.9], [0.4, 0.6]]))
    assert jnp.allclose(grad, 0.0)
