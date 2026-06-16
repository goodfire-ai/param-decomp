"""CPU tests for the JAX-native slow (plot-type) eval pass.

Pins the reduction semantics against hand-rolled numpy (component activation density and
mean-CI per component are exact under micro-batching), the `pre_sigmoid`-vs-`lower`
distinction, the `n_batches_accum` cap on the histogram sample, and that the renderer
emits valid PNGs under the exact torch `slow_eval/figures/*` keys.
"""

import jax
import numpy as np

from jax_single_pool.ci_fn import CIArch, init_ci_fn, lower_leaky_hard_sigmoid
from jax_single_pool.llama8b import (
    llama_decomposed_lm,
    llama_site_specs,
    mlp_family_site_cs,
)
from jax_single_pool.slow_eval import (
    accumulate_site_reductions,
    make_slow_eval_step,
    render_slow_eval_figures,
)
from jax_single_pool.tests.test_llama8b import (
    _tiny_cfg,  # pyright: ignore[reportPrivateUsage]
    _tiny_target,  # pyright: ignore[reportPrivateUsage]
)


def _tiny_setup(threshold: float):
    cfg = _tiny_cfg()
    tgt = _tiny_target(cfg, 4, jax.random.PRNGKey(0))
    C = 8
    sites = llama_site_specs(cfg, mlp_family_site_cs(4, 5, C))
    lm = llama_decomposed_lm(cfg, sites)
    ci_fn = init_ci_fn(CIArch(16, 1, 2, 32), lm.sites, jax.random.PRNGKey(2))
    step = make_slow_eval_step(lm, threshold)
    return cfg, lm, tgt, ci_fn, step, C


def test_reductions_match_hand_rolled_per_component():
    cfg, lm, tgt, ci_fn, step, C = _tiny_setup(threshold=0.0)
    b, t = 3, 16
    residual = jax.random.normal(jax.random.PRNGKey(4), (b, t, cfg.n_embd)) * 0.5

    reductions = accumulate_site_reductions(step, ci_fn, tgt, [residual], n_batches_accum=None)

    site_inputs = lm.site_inputs(tgt, residual)
    lower = {s: lower_leaky_hard_sigmoid(ci_fn.site_logits(site_inputs)[s]) for s in lm.site_names}
    for site in lm.site_names:
        flat = np.asarray(lower[site]).reshape(-1, C).astype(np.float32)
        r = reductions[site]
        assert r.n_positions == b * t
        np.testing.assert_allclose(r.density_counts, (flat > 0.0).sum(0), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(r.ci_sums, flat.sum(0), rtol=1e-4, atol=1e-4)


def test_density_threshold_caps_counts_at_n_positions():
    cfg, lm, tgt, ci_fn, step, _ = _tiny_setup(threshold=-1.0)  # everything "alive"
    residual = jax.random.normal(jax.random.PRNGKey(7), (2, 16, cfg.n_embd)) * 0.5
    reductions = accumulate_site_reductions(step, ci_fn, tgt, [residual], n_batches_accum=None)
    for r in reductions.values():
        np.testing.assert_array_equal(r.density_counts, np.full_like(r.density_counts, 2 * 16))


def test_cross_batch_sum_accumulates_linearly():
    cfg, lm, tgt, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    res_a = jax.random.normal(jax.random.PRNGKey(4), (2, 16, cfg.n_embd)) * 0.5
    res_b = jax.random.normal(jax.random.PRNGKey(5), (2, 16, cfg.n_embd)) * 0.5

    one = accumulate_site_reductions(step, ci_fn, tgt, [res_a], None)
    two = accumulate_site_reductions(step, ci_fn, tgt, [res_a, res_b], None)
    other = accumulate_site_reductions(step, ci_fn, tgt, [res_b], None)
    for site in lm.site_names:
        assert two[site].n_positions == one[site].n_positions + other[site].n_positions
        np.testing.assert_allclose(
            two[site].ci_sums, one[site].ci_sums + other[site].ci_sums, rtol=1e-4, atol=1e-4
        )


def test_n_batches_accum_caps_histogram_sample_only():
    cfg, lm, tgt, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    batches = [
        jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(9), i), (2, 16, cfg.n_embd))
        for i in range(3)
    ]
    capped = accumulate_site_reductions(step, ci_fn, tgt, batches, n_batches_accum=1)
    full = accumulate_site_reductions(step, ci_fn, tgt, batches, n_batches_accum=None)
    for site in lm.site_names:
        # the cap only limits the histogram raw-value sample; counts/sums span all batches
        assert capped[site].n_positions == full[site].n_positions == 3 * 2 * 16
        assert capped[site].lower_sample.size == 2 * 16 * 8  # one batch
        assert full[site].lower_sample.size == 3 * 2 * 16 * 8


def test_pre_sigmoid_differs_from_lower():
    cfg, lm, tgt, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.normal(jax.random.PRNGKey(4), (2, 16, cfg.n_embd))
    reductions = accumulate_site_reductions(step, ci_fn, tgt, [residual], None)
    for r in reductions.values():
        # lower is clamped to [0, 1]; logits are unbounded — they cannot be identical
        assert r.lower_sample.min() >= 0.0 and r.lower_sample.max() <= 1.0
        assert not np.allclose(r.lower_sample, r.logits_sample)


def test_render_emits_torch_keyed_pngs():
    cfg, lm, tgt, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.normal(jax.random.PRNGKey(4), (2, 16, cfg.n_embd))
    reductions = accumulate_site_reductions(step, ci_fn, tgt, [residual], None)
    figures = render_slow_eval_figures(reductions)
    assert set(figures) == {
        "figures/causal_importance_values",
        "figures/causal_importance_values_pre_sigmoid",
        "figures/component_activation_density",
        "figures/ci_mean_per_component",
        "figures/ci_mean_per_component_log",
    }
    for png in figures.values():
        assert png[:4] == b"\x89PNG", "renderer must emit valid PNG bytes"


def test_finite_reductions():
    cfg, lm, tgt, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.normal(jax.random.PRNGKey(4), (2, 16, cfg.n_embd))
    reductions = accumulate_site_reductions(step, ci_fn, tgt, [residual], None)
    for r in reductions.values():
        assert np.all(np.isfinite(r.density_counts))
        assert np.all(np.isfinite(r.ci_sums))
        assert np.all(np.isfinite(r.lower_sample))
        assert np.all(np.isfinite(r.logits_sample))
