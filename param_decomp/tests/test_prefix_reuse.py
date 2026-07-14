"""Prefix reuse (SPEC S3/S18 amendment 2026-07-13): a `ResidualStart` forward from
`prefix_residual` must reproduce the token forward on every model path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.components import DecompVU
from param_decomp.lm import ResidualStart
from param_decomp.targets.llama8b import LlamaDecomposedModel
from param_decomp.tests.test_llama8b import _mlp_sites, _tiny_cfg, _tiny_decomposed_lm


def _tiny_vu(lm: "LlamaDecomposedModel", key: jax.Array) -> DecompVU:
    ks = iter(jax.random.split(key, 64))
    return DecompVU(
        vu={
            s.name: (
                jax.random.normal(next(ks), (s.d_in, s.C)) * 0.02,
                jax.random.normal(next(ks), (s.C, s.d_out)) * 0.02,
            )
            for s in lm.sites
        }
    )


def _assert_close(a: jax.Array, b: jax.Array, what: str) -> None:
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-5, err_msg=what)


@pytest.mark.parametrize("first,last", [(4, 5), (2, 2)])
def test_residual_start_matches_token_forward(first: int, last: int):
    cfg = _tiny_cfg()
    sites = _mlp_sites(cfg, first, last, C=8)
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    assert hasattr(lm, "prefix_residual") and lm.split_layer > 0
    assert lm.split_layer == first

    key = jax.random.PRNGKey(1)
    tokens = jax.random.randint(key, (2, 6), 0, cfg.vocab_size)
    start = ResidualStart(lm.prefix_residual(tokens))

    _assert_close(lm.clean_output(tokens), lm.clean_output(start), "clean_output")

    logits_tok, collect_tok = lm.clean_output_and_site_outputs(tokens)
    logits_res, collect_res = lm.clean_output_and_site_outputs(start)
    _assert_close(logits_tok, logits_res, "clean_output_and_site_outputs logits")
    for s in lm.site_names:
        _assert_close(collect_tok[s], collect_res[s], f"clean site output {s}")

    wanted = (f"resid.{first}", *lm.site_names)
    taps_tok = lm.read_activations(tokens, wanted)
    taps_res = lm.read_activations(start, wanted)
    for k in wanted:
        _assert_close(taps_tok[k], taps_res[k], f"tap {k}")

    prepared = lm.prepare_compute_weights(_tiny_vu(lm, jax.random.PRNGKey(2)))
    leading = tokens.shape
    mkey = jax.random.PRNGKey(3)
    masks = {
        s.name: jax.random.uniform(jax.random.fold_in(mkey, i), (*leading, s.C))
        for i, s in enumerate(lm.sites)
    }
    delta_masks = {
        s.name: jax.random.uniform(jax.random.fold_in(mkey, 100 + i), leading)
        for i, s in enumerate(lm.sites)
    }
    routes = {
        s.name: jax.random.uniform(jax.random.fold_in(mkey, 200 + i), leading) < 0.5
        for i, s in enumerate(lm.sites)
    }
    live = lm.site_names

    for r in (None, routes):
        out_tok = lm.masked_output(prepared, tokens, masks, delta_masks, r, live, True, remat=False)
        out_res = lm.masked_output(prepared, start, masks, delta_masks, r, live, True, remat=False)
        _assert_close(out_tok, out_res, f"masked_output routes={'on' if r else 'off'}")

    site_tok = lm.masked_site_outputs(prepared, tokens, masks, delta_masks, None, live, True)
    site_res = lm.masked_site_outputs(prepared, start, masks, delta_masks, None, live, True)
    for s in live:
        _assert_close(site_tok[s], site_res[s], f"masked site output {s}")

    acts_tok = lm.masked_component_activations(
        prepared, tokens, masks, delta_masks, None, live, True
    )
    acts_res = lm.masked_component_activations(
        prepared, start, masks, delta_masks, None, live, True
    )
    for s in live:
        _assert_close(acts_tok[s], acts_res[s], f"component activations {s}")


def test_prefix_is_stop_gradient_and_layer_zero_site_disables():
    cfg = _tiny_cfg()
    lm0 = _tiny_decomposed_lm(cfg, _mlp_sites(cfg, 0, 1, C=8), jax.random.PRNGKey(0))
    assert lm0.split_layer == 0
    with pytest.raises(AssertionError):
        lm0.prefix_residual(jnp.zeros((1, 4), jnp.int32))

    lm = _tiny_decomposed_lm(cfg, _mlp_sites(cfg, 3, 3, C=8), jax.random.PRNGKey(0))
    tokens = jax.random.randint(jax.random.PRNGKey(1), (1, 4), 0, cfg.vocab_size)
    resid = lm.prefix_residual(tokens)
    assert resid.shape == (1, 4, cfg.n_embd)
