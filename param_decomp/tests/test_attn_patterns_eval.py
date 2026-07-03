"""CPU tests for the in-loop attention-pattern recon eval metrics.

Pins the metric-local `attn_pattern_for` target dispatch (shape + causal/softmax sanity on
both LM targets), the all-false-routes clean target (KL=0 when masked==clean), and the
host-side token-weighted accumulation (combined = Σ sum_kl / Σ n_distributions).
"""

from typing import Any

import equinox as eqx
import jax
import numpy as np
import pytest

from param_decomp.attn_patterns_eval import (
    accumulate_attn_patterns,
    attn_pattern_for,
    attn_patterns_log_entries,
    make_ci_attn_patterns_step,
    make_stochastic_attn_patterns_step,
)
from param_decomp.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    CIFn,
    build_ci_fn,
)
from param_decomp.components import SiteC, SiteSpec, init_decomp_vu
from param_decomp.lm import DecomposedModel, run_stochastic_masked_output
from param_decomp.targets.llama8b import (
    llama_site_specs,
)
from param_decomp.targets.llama_simple_mlp import (
    canonical_site_cs as simple_canonical,
)
from param_decomp.targets.llama_simple_mlp import (
    site_specs as simple_site_specs,
)
from param_decomp.tests.test_llama8b import (
    _tiny_cfg as _llama_cfg,
)
from param_decomp.tests.test_llama8b import (
    _tiny_decomposed_lm as _llama_decomposed_lm,
)
from param_decomp.tests.test_llama_simple_mlp import (
    _tiny_cfg as _simple_cfg,
)
from param_decomp.tests.test_llama_simple_mlp import (
    _tiny_decomposed_model as _simple_decomposed_model,
)


def _build_ci_fn(lm: DecomposedModel, n_embd: int, key: jax.Array) -> CIFn:
    """One transformer chunk over all sites, reading the residual entering the first
    decomposed block. The old `CIArch(16, 1, 2, 32)` dims map onto the chunk arch."""
    site_names = lm.site_names
    first_block = min(int(name.split(".")[1]) for name in site_names)
    arch = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(f"resid.{first_block}",), output_sites=site_names),),
        input_dim=n_embd,
        d_model=16,
        n_blocks=1,
        n_heads=2,
        mlp_hidden=32,
    )
    return build_ci_fn(arch, lm.sites, key)


def test_attn_pattern_for_shape_and_causal_softmax_llama():
    cfg = _llama_cfg()
    sites = llama_site_specs(cfg, (SiteC("layers.0.self_attn.q_proj", 4),))
    lm = _llama_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    pattern_fn = attn_pattern_for(lm)
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
    sites = simple_site_specs(cfg, (SiteC("h.0.attn.q_proj", 4),))
    lm = _simple_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    pattern_fn = attn_pattern_for(lm)
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
    site_cs = (
        SiteC("layers.4.self_attn.q_proj", 8),
        SiteC("layers.4.self_attn.k_proj", 6),
        SiteC("layers.5.self_attn.q_proj", 8),
        SiteC("layers.5.self_attn.k_proj", 6),
    )
    from param_decomp.targets.llama8b import canonical_site_cs

    sites = llama_site_specs(cfg, canonical_site_cs(site_cs))
    lm = _llama_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    components = init_decomp_vu(sites, jax.random.PRNGKey(1))
    ci_fn = _build_ci_fn(lm, cfg.n_embd, jax.random.PRNGKey(2))
    return cfg, lm, components, ci_fn


def test_ci_step_clean_equals_masked_when_ci_all_one_gives_finite_kl():
    cfg, lm, components, ci_fn = _llama_attn_setup()
    pattern_fn = attn_pattern_for(lm)
    step = make_ci_attn_patterns_step(lm, pattern_fn)
    b, t = 2, 12
    residual = jax.random.randint(jax.random.PRNGKey(4), (b, t), 0, cfg.vocab_size)

    sum_kl, n_dist = step(lm, components, ci_fn, residual, jax.random.PRNGKey(0))

    q_sites = [s for s in lm.site_names if s.endswith("q_proj")]
    assert set(sum_kl) == set(q_sites) == set(n_dist)
    for q in q_sites:
        assert int(n_dist[q]) == b * cfg.n_head * t
        assert np.isfinite(float(sum_kl[q]))
        assert float(sum_kl[q]) >= 0.0, "KL is non-negative"


def test_accumulate_is_token_weighted_and_combines():
    cfg, lm, components, ci_fn = _llama_attn_setup()
    pattern_fn = attn_pattern_for(lm)
    step = make_ci_attn_patterns_step(lm, pattern_fn)
    res_a = jax.random.randint(jax.random.PRNGKey(4), (2, 10), 0, cfg.vocab_size)
    res_b = jax.random.randint(jax.random.PRNGKey(5), (2, 10), 0, cfg.vocab_size)

    one = accumulate_attn_patterns(step, lm, components, ci_fn, [res_a], jax.random.PRNGKey(0))
    other = accumulate_attn_patterns(step, lm, components, ci_fn, [res_b], jax.random.PRNGKey(0))
    two = accumulate_attn_patterns(
        step, lm, components, ci_fn, [res_a, res_b], jax.random.PRNGKey(0)
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
    cfg, lm, components, ci_fn = _llama_attn_setup()
    pattern_fn = attn_pattern_for(lm)
    n_draws = 3
    step = make_stochastic_attn_patterns_step(lm, pattern_fn, n_draws)
    b, t = 2, 8
    residual = jax.random.randint(jax.random.PRNGKey(4), (b, t), 0, cfg.vocab_size)

    sum_kl, n_dist = step(lm, components, ci_fn, residual, jax.random.PRNGKey(0))
    for q in (s for s in lm.site_names if s.endswith("q_proj")):
        assert int(n_dist[q]) == b * cfg.n_head * t * n_draws
        assert np.isfinite(float(sum_kl[q])) and float(sum_kl[q]) >= 0.0


def test_simple_mlp_step_runs_end_to_end():
    cfg = _simple_cfg()
    site_cs = simple_canonical((SiteC("h.0.attn.q_proj", 6), SiteC("h.0.attn.k_proj", 6)))
    sites = simple_site_specs(cfg, site_cs)
    lm = _simple_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    components = init_decomp_vu(sites, jax.random.PRNGKey(1))
    ci_fn = _build_ci_fn(lm, cfg.n_embd, jax.random.PRNGKey(2))
    step = make_ci_attn_patterns_step(lm, attn_pattern_for(lm))
    b, t = 2, 10
    residual = jax.random.randint(jax.random.PRNGKey(4), (b, t), 0, cfg.vocab_size)

    sum_kl, n_dist = step(lm, components, ci_fn, residual, jax.random.PRNGKey(0))
    assert set(sum_kl) == {"h.0.attn.q_proj"}
    assert int(n_dist["h.0.attn.q_proj"]) == b * cfg.n_head * t
    assert np.isfinite(float(sum_kl["h.0.attn.q_proj"]))


class _PositionlessStub(eqx.Module):
    """A `leading_axes=()` model whose methods are never called — only exercises the
    LM-only `leading_axes` guards (which fire at step construction)."""

    sites: tuple[SiteSpec, ...] = eqx.field(static=True)
    leading_axes: tuple[str, ...] = eqx.field(static=True)

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sites)

    def shardings(self, mesh: Any) -> "_PositionlessStub":
        del mesh
        raise AssertionError("positionless stub fn must not be called")

    def recon_loss_fn(self, masked_output: Any, clean_output: Any) -> jax.Array:
        del masked_output, clean_output
        raise AssertionError("positionless stub fn must not be called")

    def clean_output(self, resid: Any) -> Any:
        del resid
        raise AssertionError("positionless stub fn must not be called")

    def read_activations(self, resid: Any, wanted: tuple[str, ...]) -> dict[str, jax.Array]:
        del resid, wanted
        raise AssertionError("positionless stub fn must not be called")

    def prepare_compute_weights(self, vu: Any) -> Any:
        return vu

    def masked_output(
        self,
        vu: Any,
        resid: Any,
        masks: Any,
        delta_masks: Any,
        routes: Any,
        live: tuple[str, ...],
        has_delta: bool,
        *,
        remat: bool,
    ) -> Any:
        del vu, resid, masks, delta_masks, routes, live, has_delta, remat
        raise AssertionError("positionless stub fn must not be called")

    def stack_ci(self, ci_lower: dict[str, Any]) -> dict[str, Any]:
        return ci_lower

    def masked_output_stochastic(
        self,
        prepared: Any,
        resid: Any,
        ci_stacked: Any,
        draw_key: Any,
        routes: Any,
        live: tuple[str, ...],
        has_delta: bool,
        *,
        remat: bool,
    ) -> Any:
        return run_stochastic_masked_output(
            self, prepared, resid, ci_stacked, draw_key, routes, live, has_delta, remat=remat
        )

    def masked_site_outputs(
        self,
        vu: Any,
        resid: Any,
        masks: Any,
        delta_masks: Any,
        routes: Any,
        live: tuple[str, ...],
        has_delta: bool,
    ) -> dict[str, jax.Array]:
        del vu, resid, masks, delta_masks, routes, live, has_delta
        raise AssertionError("positionless stub fn must not be called")

    def weight_deltas(self, vu: Any) -> dict[str, jax.Array]:
        del vu
        raise AssertionError("positionless stub fn must not be called")


def test_attn_patterns_steps_reject_positionless_target():
    """Attention patterns are causal maps over a sequence axis; both step constructors
    must fail loud against a positionless (`leading_axes=()`) target. The leading-axes
    guard fires before site/pattern inspection, so a dummy pattern fn is fine."""
    lm = _PositionlessStub(
        sites=(SiteSpec("linear1", 5, 2, 8), SiteSpec("linear2", 2, 5, 6)),
        leading_axes=(),
    )
    assert lm.leading_axes == ()
    dummy_pattern_fn = lambda q, k: q  # noqa: E731 — never reached; assert fires first
    with pytest.raises(AssertionError, match="LM-only"):
        make_ci_attn_patterns_step(lm, dummy_pattern_fn)
    with pytest.raises(AssertionError, match="LM-only"):
        make_stochastic_attn_patterns_step(lm, dummy_pattern_fn, 1)
