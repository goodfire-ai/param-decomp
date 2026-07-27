"""The family tap grammar: wire-key pins (the historical string forms, byte-identical)
and fail-closed parsing on a bound grammar."""

import pytest

from param_decomp.core.family import ArchFamily
from param_decomp.targets import glu_transformer, llama_simple_mlp
from param_decomp.targets.transformer_taps import TransformerTapGrammar, resid_tap_key

D_RESID = 64
D_IN = {"layers.2.mlp.down_proj": 48, "h.3.attn.q_proj": 32}


def _grammar(family: ArchFamily, n_layer: int = 32) -> TransformerTapGrammar:
    return TransformerTapGrammar(family, n_layer, D_RESID, d_in_of=lambda name: D_IN[name])


def test_wire_forms_are_the_historical_keys():
    assert resid_tap_key(18) == "resid.18"
    assert resid_tap_key(7) == "resid.7"
    # site-input taps are the site name verbatim — no mint, no wrapping
    glu = _grammar(glu_transformer.FAMILY)
    simple = _grammar(llama_simple_mlp.FAMILY)
    assert glu.block_of("layers.18.mlp.down_proj") == 18
    assert simple.block_of("h.3.attn.q_proj") == 3
    assert simple.block_of("h.0.mlp.c_fc") == 0
    assert simple.block_of("h.2.attn.v_proj") == 2
    assert glu.block_of(resid_tap_key(0)) == 0
    assert glu.block_of(resid_tap_key(31)) == 31


def test_widths():
    glu = _grammar(glu_transformer.FAMILY)
    simple = _grammar(llama_simple_mlp.FAMILY)
    assert glu.width_of("resid.5") == D_RESID
    assert glu.width_of("layers.2.mlp.down_proj") == D_IN["layers.2.mlp.down_proj"]
    assert simple.width_of("h.3.attn.q_proj") == D_IN["h.3.attn.q_proj"]


def test_resid_tap_beyond_the_block_range_dies():
    glu = _grammar(glu_transformer.FAMILY, n_layer=32)
    with pytest.raises(AssertionError, match=r"'resid\.99' out of range.*blocks 0\.\.31"):
        glu.block_of("resid.99")


def test_site_tap_beyond_the_block_range_dies():
    glu = _grammar(glu_transformer.FAMILY, n_layer=32)
    with pytest.raises(AssertionError, match=r"out of range.*blocks 0\.\.31"):
        glu.block_of("layers.99.mlp.down_proj")


def test_non_integer_resid_suffix_dies_named():
    glu = _grammar(glu_transformer.FAMILY)
    with pytest.raises(AssertionError, match=r"malformed residual tap 'resid\.abc'"):
        glu.block_of("resid.abc")


def test_keys_outside_the_family_vocabulary_die_named():
    glu = _grammar(glu_transformer.FAMILY)
    simple = _grammar(llama_simple_mlp.FAMILY)
    with pytest.raises(AssertionError, match="not a glu_transformer site"):
        glu.block_of("residd.5")
    with pytest.raises(AssertionError, match="not a glu_transformer site"):
        glu.width_of("layers.3.self_attn.zebra_proj")
    with pytest.raises(AssertionError, match="not a simple_mlp site"):
        simple.block_of("layers.3.self_attn.q_proj")  # the OTHER family's site name
