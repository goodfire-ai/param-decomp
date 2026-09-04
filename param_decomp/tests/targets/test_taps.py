"""The family tap grammar: wire-key pins (the historical string forms, byte-identical)
and fail-closed parsing on a bound grammar."""

import pytest

from param_decomp.core.family import ArchFamily
from param_decomp.targets import glu_transformer, llama_simple_mlp
from param_decomp.targets.transformer_taps import (
    BlockTap,
    PostAttentionResidual,
    ResidualBoundary,
    SiteOutput,
    TransformerTapGrammar,
    attention_input_tap_key,
    attention_output_tap_key,
    mlp_hidden_tap_key,
    mlp_input_tap_key,
    post_attention_tap_key,
    resid_tap_key,
    site_output_tap_key,
)

D_RESID = 64
D_ATTENTION_OUTPUT = 32
D_MLP_HIDDEN = 48
D_OUT = {"layers.2.mlp.down_proj": 64, "h.3.attn.q_proj": 40}


def _grammar(family: ArchFamily, n_layer: int = 32) -> TransformerTapGrammar:
    return TransformerTapGrammar(
        family=family,
        n_layer=n_layer,
        d_resid=D_RESID,
        d_attention_output=D_ATTENTION_OUTPUT,
        d_mlp_hidden=D_MLP_HIDDEN,
        d_out_of=lambda name: D_OUT.get(name, D_RESID),
    )


def test_wire_forms_name_physical_activations_once():
    assert resid_tap_key(18) == "resid.18"
    assert post_attention_tap_key(7) == "post_attn.7"
    assert attention_input_tap_key(18) == "attn_in.18"
    assert attention_output_tap_key(18) == "attn_out.18"
    assert mlp_input_tap_key(18) == "mlp_in.18"
    assert mlp_hidden_tap_key(18) == "mlp_hidden.18"
    assert site_output_tap_key("layers.2.mlp.down_proj") == "layers.2.mlp.down_proj.out"

    glu = _grammar(glu_transformer.FAMILY)
    simple = _grammar(llama_simple_mlp.FAMILY)
    assert glu.block_of(attention_input_tap_key(18)) == 18
    assert simple.block_of(attention_input_tap_key(3)) == 3
    assert simple.block_of(mlp_input_tap_key(0)) == 0
    assert simple.block_of(mlp_hidden_tap_key(2)) == 2
    assert glu.block_of(resid_tap_key(0)) == 0
    assert glu.block_of(resid_tap_key(31)) == 31
    assert glu.block_of(resid_tap_key(32)) == 32


def test_widths():
    glu = _grammar(glu_transformer.FAMILY)
    simple = _grammar(llama_simple_mlp.FAMILY)
    assert glu.width_of("resid.5") == D_RESID
    assert glu.width_of(mlp_hidden_tap_key(2)) == D_MLP_HIDDEN
    assert simple.width_of(attention_input_tap_key(3)) == D_RESID
    assert glu.width_of("post_attn.2") == D_RESID
    assert glu.width_of("layers.2.mlp.down_proj.out") == D_OUT["layers.2.mlp.down_proj"]


def test_block_tap_keys_name_each_physical_vector_once():
    grammar = _grammar(glu_transformer.FAMILY)
    assert grammar.block_tap_keys((2,)) == (
        attention_input_tap_key(2),
        attention_output_tap_key(2),
        mlp_input_tap_key(2),
        mlp_hidden_tap_key(2),
    )
    with pytest.raises(AssertionError):
        grammar.block_tap_keys((2, 2))


def test_resid_tap_beyond_the_block_range_dies():
    glu = _grammar(glu_transformer.FAMILY, n_layer=32)
    with pytest.raises(AssertionError, match=r"'resid\.99' out of range.*0\.\.32"):
        glu.block_of("resid.99")


def test_block_tap_beyond_the_block_range_dies():
    glu = _grammar(glu_transformer.FAMILY, n_layer=32)
    with pytest.raises(AssertionError, match="unknown transformer activation"):
        glu.block_of("attn_in.99")


def test_non_integer_resid_suffix_dies_named():
    glu = _grammar(glu_transformer.FAMILY)
    with pytest.raises(AssertionError, match=r"malformed residual boundary 'resid\.abc'"):
        glu.block_of("resid.abc")


def test_keys_outside_the_family_vocabulary_die_named():
    glu = _grammar(glu_transformer.FAMILY)
    simple = _grammar(llama_simple_mlp.FAMILY)
    with pytest.raises(AssertionError, match="unknown transformer activation"):
        glu.block_of("residd.5")
    with pytest.raises(AssertionError, match="not a glu_transformer site"):
        glu.width_of("layers.3.self_attn.zebra_proj.out")
    with pytest.raises(AssertionError, match="unknown transformer activation"):
        simple.block_of("layers.3.self_attn.q_proj")


def test_resolution_returns_typed_sources_in_request_order():
    grammar = _grammar(glu_transformer.FAMILY)
    keys = ("resid.32", "post_attn.2", mlp_hidden_tap_key(2), "layers.2.mlp.down_proj.out")
    sources = grammar.resolve(keys, lambda point: point)
    assert isinstance(sources[0], ResidualBoundary)
    assert isinstance(sources[1], PostAttentionResidual)
    assert isinstance(sources[2], BlockTap)
    assert isinstance(sources[3], SiteOutput)
