"""CPU tests for the in-loop hidden-acts recon eval metrics (SPEC S31).

Pins the per-site MSE shape/count bookkeeping on both steps and the host-side
token-weighted accumulation. Batch != seq throughout: the steps must derive the waist
`*leading` from the CI output, not from the token inputs (whose `shape[:-1]` drops the
sequence axis — the regression that crashed the stochastic step's delta/route broadcast).
"""

import jax
import numpy as np

from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    CIFn,
    MHACIAttention,
    build_ci_fn,
)
from param_decomp.core.components import SiteC, init_component_stacks
from param_decomp.core.hidden_acts_eval import (
    accumulate_hidden_acts,
    hidden_acts_log_entries,
    make_ci_hidden_acts_step,
    make_stochastic_hidden_acts_step,
)
from param_decomp.core.model import DecomposedModel
from param_decomp.targets.llama_simple_mlp import (
    canonical_site_cs,
    parse_site_name,
    site_specs,
)
from param_decomp.targets.testing import (
    tiny_simple_mlp_cfg,
    tiny_simple_mlp_decomposed_model,
)

_BATCH, _SEQ = 2, 12


def _build_ci_fn(model: DecomposedModel, n_embd: int, key: jax.Array) -> CIFn:
    site_names = model.site_names
    first_block = min(parse_site_name(n)[0] for n in site_names)
    arch = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(f"resid.{first_block}",), output_sites=site_names),),
        input_dim=n_embd,
        d_model=16,
        n_blocks=1,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )
    return build_ci_fn(arch, model.sites, key)


def _setup():
    cfg = tiny_simple_mlp_cfg()
    site_cs = canonical_site_cs(
        (
            SiteC("h.2.attn.q_proj", 8),
            SiteC("h.2.attn.v_proj", 12),
            SiteC("h.2.mlp.c_fc", 8),
            SiteC("h.3.mlp.down_proj", 16),
        )
    )
    model = tiny_simple_mlp_decomposed_model(cfg, site_specs(cfg, site_cs), jax.random.PRNGKey(0))
    components = init_component_stacks(model.sites, jax.random.PRNGKey(1))
    ci_fn = _build_ci_fn(model, cfg.n_embd, jax.random.PRNGKey(2))
    tokens = jax.random.randint(jax.random.PRNGKey(3), (_BATCH, _SEQ), 0, cfg.vocab_size)
    site_d_out = {
        "h.2.attn.q_proj": cfg.n_head * cfg.head_dim,
        "h.2.attn.v_proj": cfg.n_kv_head * cfg.head_dim,
        "h.2.mlp.c_fc": cfg.n_intermediate,
        "h.3.mlp.down_proj": cfg.n_embd,
    }
    return model, components, ci_fn, tokens, site_d_out


def test_ci_step_per_site_sums_and_counts():
    model, components, ci_fn, tokens, site_d_out = _setup()
    step = make_ci_hidden_acts_step(model)

    sum_mse, n_elements = step(model, components, ci_fn, tokens, jax.random.PRNGKey(0))

    assert set(sum_mse) == set(model.site_names) == set(n_elements)
    for site in model.site_names:
        assert int(n_elements[site]) == _BATCH * _SEQ * site_d_out[site]
        assert np.isfinite(float(sum_mse[site]))
        assert float(sum_mse[site]) >= 0.0


def test_stochastic_step_per_site_sums_and_counts():
    model, components, ci_fn, tokens, site_d_out = _setup()
    n_mask_samples = 3
    step = make_stochastic_hidden_acts_step(model, n_mask_samples)

    sum_mse, n_elements = step(model, components, ci_fn, tokens, jax.random.PRNGKey(0))

    assert set(sum_mse) == set(model.site_names) == set(n_elements)
    for site in model.site_names:
        assert int(n_elements[site]) == _BATCH * _SEQ * site_d_out[site] * n_mask_samples
        assert np.isfinite(float(sum_mse[site]))
        assert float(sum_mse[site]) >= 0.0


def test_accumulate_and_log_entries_token_weighted():
    model, components, ci_fn, tokens, _ = _setup()
    step = make_ci_hidden_acts_step(model)

    one = accumulate_hidden_acts(step, model, components, ci_fn, [tokens], jax.random.PRNGKey(0))
    two = accumulate_hidden_acts(
        step, model, components, ci_fn, [tokens, tokens], jax.random.PRNGKey(0)
    )

    for site, r in two.items():
        assert r.n_elements == 2 * one[site].n_elements
        np.testing.assert_allclose(r.sum_mse, 2 * one[site].sum_mse, rtol=1e-6)

    entries = hidden_acts_log_entries("CIHiddenActsReconLoss", two)
    assert set(entries) == {"CIHiddenActsReconLoss"} | {f"CIHiddenActsReconLoss/{s}" for s in two}
    total_sum = sum(r.sum_mse for r in two.values())
    total_n = sum(r.n_elements for r in two.values())
    np.testing.assert_allclose(entries["CIHiddenActsReconLoss"], total_sum / total_n)
