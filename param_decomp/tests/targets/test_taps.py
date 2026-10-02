"""The family tap grammar: wire-key pins (the historical string forms, byte-identical)
and fail-closed parsing on a bound grammar."""

import pytest

from param_decomp.core.family import ArchFamily
from param_decomp.targets import llama_simple_mlp, transformer
from param_decomp.targets.transformer_taps import (
    BlockCaptures,
    BlockTap,
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
D_OUT = 40


def _grammar(family: ArchFamily, n_layer: int = 32) -> TransformerTapGrammar:
    block = BlockCaptures(
        tap_widths={
            "attn_in": D_RESID,
            "attn_out": D_ATTENTION_OUTPUT,
            "post_attn": D_RESID,
            "mlp_in": D_RESID,
            "mlp_hidden": D_MLP_HIDDEN,
        },
        site_output_widths=dict.fromkeys(family.matrices, D_OUT),
    )
    return TransformerTapGrammar(family=family, d_resid=D_RESID, blocks=(block,) * n_layer)


def _mixer_blocks_grammar() -> TransformerTapGrammar:
    """Block 0 computes no MLP hidden and carries only the down matrix."""
    return TransformerTapGrammar(
        family=transformer.FAMILY,
        d_resid=D_RESID,
        blocks=(
            BlockCaptures(
                tap_widths={"attn_in": D_RESID, "attn_out": D_ATTENTION_OUTPUT, "mlp_in": D_RESID},
                site_output_widths={"down": D_RESID},
            ),
        ),
    )


def test_wire_forms_name_physical_activations_once():
    assert resid_tap_key(18) == "resid.18"
    assert post_attention_tap_key(7) == "post_attn.7"
    assert attention_input_tap_key(18) == "attn_in.18"
    assert attention_output_tap_key(18) == "attn_out.18"
    assert mlp_input_tap_key(18) == "mlp_in.18"
    assert mlp_hidden_tap_key(18) == "mlp_hidden.18"
    assert site_output_tap_key("layers.2.mlp.down_proj") == "layers.2.mlp.down_proj.out"

    glu = _grammar(transformer.FAMILY)
    simple = _grammar(llama_simple_mlp.FAMILY)
    assert glu.block_of(attention_input_tap_key(18)) == 18
    assert simple.block_of(attention_input_tap_key(3)) == 3
    assert simple.block_of(mlp_input_tap_key(0)) == 0
    assert simple.block_of(mlp_hidden_tap_key(2)) == 2
    assert glu.block_of(resid_tap_key(0)) == 0
    assert glu.block_of(resid_tap_key(31)) == 31
    assert glu.block_of(resid_tap_key(32)) == 32


def test_widths():
    glu = _grammar(transformer.FAMILY)
    simple = _grammar(llama_simple_mlp.FAMILY)
    assert glu.width_of("resid.5") == D_RESID
    assert glu.width_of(mlp_hidden_tap_key(2)) == D_MLP_HIDDEN
    assert simple.width_of(attention_input_tap_key(3)) == D_RESID
    assert glu.width_of("post_attn.2") == D_RESID
    assert glu.width_of("layers.2.mlp.down_proj.out") == D_OUT


def test_site_input_tap_keys_name_each_physical_vector_once():
    grammar = _grammar(transformer.FAMILY)
    assert grammar.site_input_tap_keys((2,)) == (
        attention_input_tap_key(2),
        attention_output_tap_key(2),
        mlp_input_tap_key(2),
        mlp_hidden_tap_key(2),
    )
    with pytest.raises(AssertionError):
        grammar.site_input_tap_keys((2, 2))


def test_undeclared_block_vectors_and_matrices_die_named():
    grammar = _mixer_blocks_grammar()
    assert grammar.site_input_tap_keys((0,)) == (
        attention_input_tap_key(0),
        attention_output_tap_key(0),
        mlp_input_tap_key(0),
    )
    with pytest.raises(AssertionError, match="computes no 'mlp_hidden' vector in block 0"):
        grammar.width_of(mlp_hidden_tap_key(0))
    with pytest.raises(AssertionError, match="computes no 'post_attn' vector in block 0"):
        grammar.width_of(post_attention_tap_key(0))
    with pytest.raises(AssertionError, match="block 0 carries no 'q' matrix"):
        grammar.width_of("layers.0.self_attn.q_proj.out")
    assert grammar.width_of("layers.0.mlp.down_proj.out") == D_RESID


def test_resid_tap_beyond_the_block_range_dies():
    glu = _grammar(transformer.FAMILY, n_layer=32)
    with pytest.raises(AssertionError, match=r"'resid\.99' out of range.*0\.\.32"):
        glu.block_of("resid.99")


def test_block_tap_beyond_the_block_range_dies():
    glu = _grammar(transformer.FAMILY, n_layer=32)
    with pytest.raises(AssertionError, match=r"block tap 'attn_in\.99' out of range.*0\.\.31"):
        glu.block_of("attn_in.99")


def test_non_integer_resid_suffix_dies_named():
    glu = _grammar(transformer.FAMILY)
    with pytest.raises(AssertionError, match=r"malformed residual boundary 'resid\.abc'"):
        glu.block_of("resid.abc")


def test_keys_outside_the_family_vocabulary_die_named():
    glu = _grammar(transformer.FAMILY)
    simple = _grammar(llama_simple_mlp.FAMILY)
    with pytest.raises(AssertionError, match="unknown transformer activation"):
        glu.block_of("residd.5")
    with pytest.raises(AssertionError, match="not a glu_transformer site"):
        glu.width_of("layers.3.self_attn.zebra_proj.out")
    with pytest.raises(AssertionError, match="unknown transformer activation"):
        simple.block_of("layers.3.self_attn.q_proj")


def test_resolution_returns_typed_sources_in_request_order():
    grammar = _grammar(transformer.FAMILY)
    keys = ("resid.32", "post_attn.2", mlp_hidden_tap_key(2), "layers.2.mlp.down_proj.out")
    sources = grammar.resolve(keys, lambda point: point)
    assert isinstance(sources[0], ResidualBoundary)
    assert sources[1] == BlockTap(name="post_attn", block=2)
    assert isinstance(sources[2], BlockTap)
    assert isinstance(sources[3], SiteOutput)
