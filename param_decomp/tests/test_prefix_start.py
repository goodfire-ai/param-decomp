"""Prefix-start equivalence (SPEC D5): a masked forward seeded from the clean pass's
`resid.{first live block}` tap equals the input-edge (`start_from_inputs`) forward, on
the deterministic AND stochastic mask paths, and `clean_output_and_taps` matches the
separate clean/taps calls."""

import jax
import jax.numpy as jnp

from param_decomp.components import init_decomp_vu
from param_decomp.targets.llama_simple_mlp import site_specs
from param_decomp.tests.test_llama8b import _mlp_sites, _tiny_cfg, _tiny_decomposed_lm
from param_decomp.tests.test_llama_simple_mlp import (
    _MIXED_SITE_CS,
)
from param_decomp.tests.test_llama_simple_mlp import (
    _tiny_cfg as _tiny_simple_mlp_cfg,
)
from param_decomp.tests.test_llama_simple_mlp import (
    _tiny_decomposed_model as _tiny_simple_mlp,
)

_ATOL = 1e-6  # fp32 CPU; the split scan preserves op order, only fusion may differ


def _lm_and_masks():
    cfg = _tiny_cfg()
    key = jax.random.PRNGKey(0)
    lm_key, vu_key, tok_key, mask_key = jax.random.split(key, 4)
    sites = _mlp_sites(cfg, 4, 5, 8)
    lm = _tiny_decomposed_lm(cfg, sites, lm_key)
    vu = init_decomp_vu(sites, vu_key)
    prepared = lm.prepare_compute_weights(vu)
    tokens = jax.random.randint(tok_key, (2, 16), 0, cfg.vocab_size)
    names = lm.site_names
    mask_keys = iter(jax.random.split(mask_key, 2 * len(names)))
    masks = {s.name: jax.random.uniform(next(mask_keys), (2, 16, s.C), jnp.float32) for s in sites}
    delta_masks = {s: jax.random.uniform(next(mask_keys), (2, 16), jnp.float32) for s in names}
    return lm, prepared, tokens, names, masks, delta_masks


def test_masked_output_tap_start_matches_input_start():
    lm, prepared, tokens, names, masks, delta_masks = _lm_and_masks()
    _, taps = lm.clean_output_and_taps(tokens, lm.start_taps(names))

    from_inputs = lm.masked_output(
        prepared, lm.start_from_inputs(tokens), masks, delta_masks, None, names, True,
        remat=False,
    )  # fmt: skip
    from_tap = lm.masked_output(
        prepared, lm.masked_start(tokens, taps, names), masks, delta_masks, None, names, True,
        remat=False,
    )  # fmt: skip
    assert jnp.allclose(from_inputs, from_tap, atol=_ATOL), jnp.abs(from_inputs - from_tap).max()

    from_tap_remat = lm.masked_output(
        prepared, lm.masked_start(tokens, taps, names), masks, delta_masks, None, names, True,
        remat=True,
    )  # fmt: skip
    assert jnp.allclose(from_inputs, from_tap_remat, atol=_ATOL)


def test_stochastic_tap_start_matches_input_start():
    """The stochastic per-(layer,kind) keys fold over ABSOLUTE layer indices, so the same
    `draw_key` draws the same masks under either start (D5)."""
    lm, prepared, tokens, names, _, _ = _lm_and_masks()
    _, taps = lm.clean_output_and_taps(tokens, lm.start_taps(names))
    ci_key, draw_key = jax.random.split(jax.random.PRNGKey(7))
    ci_keys = iter(jax.random.split(ci_key, len(names)))
    ci_lower = {
        s.name: jax.random.uniform(next(ci_keys), (2, 16, s.C), jnp.float32) for s in lm.sites
    }
    ci_stacked = lm.stack_ci(ci_lower)

    from_inputs = lm.masked_output_stochastic(
        prepared, lm.start_from_inputs(tokens), ci_stacked, draw_key, None, names, True,
        remat=False,
    )  # fmt: skip
    from_tap = lm.masked_output_stochastic(
        prepared, lm.masked_start(tokens, taps, names), ci_stacked, draw_key, None, names, True,
        remat=False,
    )  # fmt: skip
    assert jnp.allclose(from_inputs, from_tap, atol=_ATOL), jnp.abs(from_inputs - from_tap).max()


def test_clean_output_and_taps_matches_separate_calls():
    lm, _, tokens, names, _, _ = _lm_and_masks()
    wanted = ("resid.2", *lm.start_taps(names))
    output, taps = lm.clean_output_and_taps(tokens, wanted)
    assert set(taps) == set(wanted)
    assert jnp.allclose(output, lm.clean_output(tokens), atol=_ATOL)
    # scan-emitted taps vs the unrolled `read_activations` loop: same math, different
    # fusion — D4-level reassociation tolerance, not bit parity.
    separate = lm.read_activations(tokens, wanted)
    for tap in wanted:
        assert jnp.allclose(taps[tap], separate[tap], rtol=1e-4, atol=1e-5), tap


def test_simple_mlp_tap_start_matches_input_start():
    """The unrolled-loop target, including remat=True — the start's static int must be
    destructured OUTSIDE the `jax.checkpoint` (a tuple'd int would flatten to a tracer)."""
    key = jax.random.PRNGKey(3)
    lm_key, vu_key, tok_key, mask_key = jax.random.split(key, 4)
    cfg = _tiny_simple_mlp_cfg()
    sites = site_specs(cfg, _MIXED_SITE_CS)
    lm = _tiny_simple_mlp(cfg, sites, lm_key)
    names = lm.site_names
    vu = init_decomp_vu(sites, vu_key)
    tokens = jax.random.randint(tok_key, (2, 8), 0, cfg.vocab_size)
    mask_keys = iter(jax.random.split(mask_key, 2 * len(names)))
    masks = {s.name: jax.random.uniform(next(mask_keys), (2, 8, s.C), jnp.float32) for s in sites}
    delta_masks = {s: jax.random.uniform(next(mask_keys), (2, 8), jnp.float32) for s in names}
    _, taps = lm.clean_output_and_taps(tokens, lm.start_taps(names))

    from_inputs = lm.masked_output(
        vu, lm.start_from_inputs(tokens), masks, delta_masks, None, names, True, remat=False
    )
    for remat in (False, True):
        from_tap = lm.masked_output(
            vu, lm.masked_start(tokens, taps, names), masks, delta_masks, None, names, True,
            remat=remat,
        )  # fmt: skip
        assert jnp.allclose(from_inputs, from_tap, atol=_ATOL), (
            remat,
            jnp.abs(from_inputs - from_tap).max(),
        )
