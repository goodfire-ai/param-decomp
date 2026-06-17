"""CPU tests for the in-loop attention-pattern recon eval metrics.

Pins the metric-local `attn_pattern_for` target dispatch (shape + causal/softmax sanity on
both LM targets), the all-false-routes clean target (KL=0 when masked==clean), and the
host-side token-weighted accumulation (combined = Σ sum_kl / Σ n_distributions).
"""

import jax
import numpy as np
import pytest

from jax_single_pool.attn_patterns_eval import (
    accumulate_attn_patterns,
    attn_pattern_for,
    attn_patterns_log_entries,
    make_ci_attn_patterns_step,
    make_stochastic_attn_patterns_step,
)
from jax_single_pool.ci_fn import CIArch, init_ci_fn
from jax_single_pool.llama8b import (
    init_decomp_vu,
    llama_decomposed_lm,
    llama_site_specs,
)
from jax_single_pool.llama_simple_mlp import (
    canonical_site_cs as simple_canonical,
)
from jax_single_pool.llama_simple_mlp import (
    llama_simple_mlp_decomposed_lm,
)
from jax_single_pool.llama_simple_mlp import (
    site_specs as simple_site_specs,
)
from jax_single_pool.lm import SiteC
from jax_single_pool.tests.test_llama8b import (
    _tiny_cfg as _llama_cfg,  # pyright: ignore[reportPrivateUsage]
)
from jax_single_pool.tests.test_llama8b import (
    _tiny_target as _llama_target,  # pyright: ignore[reportPrivateUsage]
)
from jax_single_pool.tests.test_llama_simple_mlp import (
    _tiny_cfg as _simple_cfg,  # pyright: ignore[reportPrivateUsage]
)
from jax_single_pool.tests.test_llama_simple_mlp import (
    _tiny_target_and_prefix as _simple_target_and_prefix,  # pyright: ignore[reportPrivateUsage]
)


def test_attn_pattern_for_shape_and_causal_softmax_llama():
    cfg = _llama_cfg()
    target = _llama_target(cfg, 0, jax.random.PRNGKey(0))
    pattern_fn = attn_pattern_for(target)
    b, t = 2, 9
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    q = jax.random.normal(jax.random.PRNGKey(1), (b, t, qd))
    k = jax.random.normal(jax.random.PRNGKey(2), (b, t, kvd))
    pattern = np.asarray(pattern_fn(q, k))

    assert pattern.shape == (b, cfg.n_head, t, t)
    np.testing.assert_allclose(pattern.sum(-1), 1.0, rtol=1e-5, atol=1e-5)
    upper = np.triu(np.ones((t, t), bool), k=1)
    assert np.allclose(pattern[:, :, upper], 0.0), "future positions must carry zero mass"
    assert pattern.dtype == np.float32


def test_attn_pattern_for_shape_and_causal_softmax_simple_mlp():
    cfg = _simple_cfg()
    target, _ = _simple_target_and_prefix(cfg, 0, jax.random.PRNGKey(0))
    pattern_fn = attn_pattern_for(target)
    b, t = 2, 7
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    q = jax.random.normal(jax.random.PRNGKey(1), (b, t, qd))
    k = jax.random.normal(jax.random.PRNGKey(2), (b, t, kvd))
    pattern = np.asarray(pattern_fn(q, k))

    assert pattern.shape == (b, cfg.n_head, t, t)
    np.testing.assert_allclose(pattern.sum(-1), 1.0, rtol=1e-5, atol=1e-5)
    upper = np.triu(np.ones((t, t), bool), k=1)
    assert np.allclose(pattern[:, :, upper], 0.0)


def test_attn_pattern_for_refuses_non_attention_target():
    with pytest.raises(AssertionError, match="only applies to attention targets"):
        attn_pattern_for(object())


def _llama_attn_setup():
    cfg = _llama_cfg()
    target = _llama_target(cfg, 4, jax.random.PRNGKey(0))
    site_cs = (
        SiteC("layers.4.self_attn.q_proj", 6),
        SiteC("layers.4.self_attn.k_proj", 6),
        SiteC("layers.5.self_attn.q_proj", 8),
        SiteC("layers.5.self_attn.k_proj", 8),
    )
    from jax_single_pool.llama8b import canonical_site_cs

    sites = llama_site_specs(cfg, canonical_site_cs(site_cs))
    lm = llama_decomposed_lm(cfg, sites)
    components = init_decomp_vu(sites, jax.random.PRNGKey(1))
    ci_fn = init_ci_fn(CIArch(16, 1, 2, 32), lm.sites, jax.random.PRNGKey(2))
    return cfg, lm, target, components, ci_fn


def test_ci_step_clean_equals_masked_when_ci_all_one_gives_finite_kl():
    cfg, lm, target, components, ci_fn = _llama_attn_setup()
    pattern_fn = attn_pattern_for(target)
    step = make_ci_attn_patterns_step(lm, pattern_fn)
    b, t = 2, 12
    residual = jax.random.normal(jax.random.PRNGKey(4), (b, t, cfg.n_embd)) * 0.5

    sum_kl, n_dist = step(components, ci_fn, target, residual, jax.random.PRNGKey(0))

    q_sites = [s for s in lm.site_names if s.endswith("q_proj")]
    assert set(sum_kl) == set(q_sites) == set(n_dist)
    for q in q_sites:
        assert int(n_dist[q]) == b * cfg.n_head * t
        assert np.isfinite(float(sum_kl[q]))
        assert float(sum_kl[q]) >= 0.0, "KL is non-negative"


def test_accumulate_is_token_weighted_and_combines():
    cfg, lm, target, components, ci_fn = _llama_attn_setup()
    pattern_fn = attn_pattern_for(target)
    step = make_ci_attn_patterns_step(lm, pattern_fn)
    res_a = jax.random.normal(jax.random.PRNGKey(4), (2, 10, cfg.n_embd)) * 0.5
    res_b = jax.random.normal(jax.random.PRNGKey(5), (2, 10, cfg.n_embd)) * 0.5

    one = accumulate_attn_patterns(step, components, ci_fn, target, [res_a], jax.random.PRNGKey(0))
    other = accumulate_attn_patterns(
        step, components, ci_fn, target, [res_b], jax.random.PRNGKey(0)
    )
    two = accumulate_attn_patterns(
        step, components, ci_fn, target, [res_a, res_b], jax.random.PRNGKey(0)
    )
    for site in one:
        assert two[site].n_distributions == one[site].n_distributions + other[site].n_distributions
        np.testing.assert_allclose(
            two[site].sum_kl, one[site].sum_kl + other[site].sum_kl, rtol=1e-4, atol=1e-4
        )

    entries = attn_patterns_log_entries("CIMaskedAttnPatternsReconLoss", two)
    combined = entries["CIMaskedAttnPatternsReconLoss"]
    total_sum = sum(r.sum_kl for r in two.values())
    total_n = sum(r.n_distributions for r in two.values())
    np.testing.assert_allclose(combined, total_sum / total_n, rtol=1e-6)
    for site, r in two.items():
        np.testing.assert_allclose(
            entries[f"CIMaskedAttnPatternsReconLoss/{site}"],
            r.sum_kl / r.n_distributions,
            rtol=1e-6,
        )


def test_stochastic_step_runs_and_scales_n_by_draws():
    cfg, lm, target, components, ci_fn = _llama_attn_setup()
    pattern_fn = attn_pattern_for(target)
    n_draws = 3
    step = make_stochastic_attn_patterns_step(lm, pattern_fn, n_draws, "continuous")
    b, t = 2, 8
    residual = jax.random.normal(jax.random.PRNGKey(4), (b, t, cfg.n_embd)) * 0.5

    sum_kl, n_dist = step(components, ci_fn, target, residual, jax.random.PRNGKey(0))
    for q in (s for s in lm.site_names if s.endswith("q_proj")):
        assert int(n_dist[q]) == b * cfg.n_head * t * n_draws
        assert np.isfinite(float(sum_kl[q])) and float(sum_kl[q]) >= 0.0


def test_simple_mlp_step_runs_end_to_end():
    cfg = _simple_cfg()
    target, _ = _simple_target_and_prefix(cfg, 0, jax.random.PRNGKey(0))
    site_cs = simple_canonical((SiteC("h.0.attn.q_proj", 6), SiteC("h.0.attn.k_proj", 6)))
    sites = simple_site_specs(cfg, site_cs)
    lm = llama_simple_mlp_decomposed_lm(cfg, sites)
    components = init_decomp_vu(sites, jax.random.PRNGKey(1))
    ci_fn = init_ci_fn(CIArch(16, 1, 2, 32), lm.sites, jax.random.PRNGKey(2))
    step = make_ci_attn_patterns_step(lm, attn_pattern_for(target))
    b, t = 2, 10
    residual = jax.random.normal(jax.random.PRNGKey(4), (b, t, cfg.n_embd)) * 0.5

    sum_kl, n_dist = step(components, ci_fn, target, residual, jax.random.PRNGKey(0))
    assert set(sum_kl) == {"h.0.attn.q_proj"}
    assert int(n_dist["h.0.attn.q_proj"]) == b * cfg.n_head * t
    assert np.isfinite(float(sum_kl["h.0.attn.q_proj"]))
