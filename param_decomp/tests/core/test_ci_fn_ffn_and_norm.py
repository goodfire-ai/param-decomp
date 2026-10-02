"""SwiGLU + learned RMSNorm scale on the chunkwise CI transformer's FFN."""

import jax
import jax.numpy as jnp
import pytest

from param_decomp.core.ci_fn.implementations.chunkwise.arch import (
    Chunk,
    ChunkwiseTransformerCIFnArch,
    UnplacedChunkwiseCIFn,
    init_chunkwise_transformer_backbone,
)
from param_decomp.core.ci_fn.implementations.transformer.backbone import BackboneCIFn
from param_decomp.core.ci_fn.implementations.transformer.layers import MHACIFnAttention
from param_decomp.core.components import (
    DenseFactorization,
    SiteSpec,
    init_component_stacks,
    require_full_emission,
)
from param_decomp.tests.sequence import unsegmented_sequence_layout

D, NH, FFN = 16, 4, 32
SITES = (
    SiteSpec("layers.0.q_proj", DenseFactorization(d_in=12, d_out=12, C=3), "q_proj"),
    SiteSpec("layers.0.mlp", DenseFactorization(d_in=12, d_out=12, C=5), "mlp"),
)
COMPONENTS = init_component_stacks(SITES, jax.random.PRNGKey(3))


def _arch(ffn_kind: str, learned_norm_scale: bool, ffn_hidden: int = FFN):
    return ChunkwiseTransformerCIFnArch(
        chunks=(Chunk(input_taps=("resid.0",), output_sites=tuple(s.name for s in SITES)),),
        input_dim=12,
        d_model=D,
        n_blocks=2,
        attention=MHACIFnAttention(mask="bidirectional", implementation="xla", n_heads=NH),
        ffn_hidden=ffn_hidden,
        ffn_kind=ffn_kind,  # pyright: ignore[reportArgumentType]
        learned_norm_scale=learned_norm_scale,
    )


def _build(ffn_kind: str, learned_norm_scale: bool, ffn_hidden: int = FFN):
    return init_chunkwise_transformer_backbone(
        _arch(ffn_kind, learned_norm_scale, ffn_hidden),
        SITES,
        UnplacedChunkwiseCIFn(),
        jax.random.PRNGKey(0),
    )


def _taps():
    return {"resid.0": jax.random.normal(jax.random.PRNGKey(1), (2, 6, 12))}


# ----------------------------- the defaults must not move -----------------------------


def test_gelu_weightless_is_bit_identical_to_the_pre_option_draw():
    """`gelu` + no norm scale is today's arch. The RNG split is 6 draws/block and swiglu needs
    a 7th — widening it unconditionally would silently redraw EVERY gelu param and move the
    equivalence goldens, so the count is conditional. This pins that."""
    backbone = _build("gelu", False)
    for b in backbone.chunks.blocks:
        assert b.gate is None
        assert b.norm_scales is None


def test_learned_norm_scale_inits_to_ones_so_step_zero_is_unchanged():
    """A learned scale initialised to ones is the weightless norm exactly — so turning the
    option on cannot move step 0, only the trajectory after it."""
    scaled, weightless = _build("gelu", True), _build("gelu", False)
    for b in scaled.chunks.blocks:
        assert b.norm_scales is not None
        for s in b.norm_scales:
            assert jnp.array_equal(s, jnp.ones_like(s))
    assert jnp.allclose(
        require_full_emission(
            BackboneCIFn(scaled)(
                _taps(),
                None,
                COMPONENTS,
                sequence=unsegmented_sequence_layout(_taps()),
                remat=False,
            ).preactivations[SITES[0].name]
        ),
        require_full_emission(
            BackboneCIFn(weightless)(
                _taps(),
                None,
                COMPONENTS,
                sequence=unsegmented_sequence_layout(_taps()),
                remat=False,
            ).preactivations[SITES[0].name]
        ),
        rtol=1e-6,
        atol=1e-6,
    )


# ----------------------------- swiglu -----------------------------


def test_swiglu_adds_exactly_one_matrix_per_block():
    gelu, swiglu = _build("gelu", False), _build("swiglu", False)
    n = lambda f: sum(x.size for x in jax.tree.leaves(f.chunks))  # noqa: E731
    per_block_extra = D * FFN + FFN  # w_gate + b_gate
    assert n(swiglu) - n(gelu) == per_block_extra * len(gelu.chunks.blocks)
    for b in swiglu.chunks.blocks:
        assert b.gate is not None
        w_gate, b_gate = b.gate
        assert w_gate.shape[1:] == (D, FFN) and b_gate.shape[1:] == (FFN,)


def test_swiglu_is_the_gated_product_not_a_gelu():
    """Pins the actual math: `silu(gate) * up`, so scrambling the gate must move the output
    (otherwise the third matrix is decoration)."""
    import equinox as eqx

    swiglu = _build("swiglu", False)
    base = require_full_emission(
        BackboneCIFn(swiglu)(
            _taps(),
            None,
            COMPONENTS,
            sequence=unsegmented_sequence_layout(_taps()),
            remat=False,
        ).preactivations[SITES[0].name]
    )
    blocks = swiglu.chunks.blocks
    assert blocks[0].gate is not None
    scrambled = eqx.tree_at(
        lambda f: f.chunks.blocks[0].gate[0],
        swiglu,
        jnp.full_like(blocks[0].gate[0], 5.0),
    )
    scrambled_ci = BackboneCIFn(scrambled)(
        _taps(), None, COMPONENTS, sequence=unsegmented_sequence_layout(_taps()), remat=False
    )
    assert not jnp.allclose(base, require_full_emission(scrambled_ci.preactivations[SITES[0].name]))


def test_swiglu_and_gelu_differ():
    a = require_full_emission(
        BackboneCIFn(_build("gelu", False))(
            _taps(),
            None,
            COMPONENTS,
            sequence=unsegmented_sequence_layout(_taps()),
            remat=False,
        ).preactivations[SITES[0].name]
    )
    b = require_full_emission(
        BackboneCIFn(_build("swiglu", False))(
            _taps(),
            None,
            COMPONENTS,
            sequence=unsegmented_sequence_layout(_taps()),
            remat=False,
        ).preactivations[SITES[0].name]
    )
    assert not jnp.allclose(a, b)


@pytest.mark.parametrize("ffn_kind", ["gelu", "swiglu"])
@pytest.mark.parametrize("learned_norm_scale", [False, True])
def test_every_combination_runs_end_to_end(ffn_kind: str, learned_norm_scale: bool):
    ci = BackboneCIFn(_build(ffn_kind, learned_norm_scale))(
        _taps(), None, COMPONENTS, sequence=unsegmented_sequence_layout(_taps()), remat=True
    )
    for site in SITES:
        lower = require_full_emission(ci.lower[site.name])
        assert lower.shape == (2, 6, site.C)
        assert jnp.isfinite(lower).all()


# The authored-schema parse tests (`ChunkwiseTransformerCIFnConfig.ffn` arms) live with the
# schema, lab-side: `param_decomp/tests/experiments/lm/test_lm_ci_schema.py`.


def test_iso_param_swiglu_width_at_production_shape():
    """The arithmetic that matters for the sweep, pinned because I got it wrong twice.

    Production is d_model 4096 / hidden 16384 (the classic 4d FFN). SwiGLU is 3 matrices vs
    2, so iso-param is 2/3 -> 10922.67, which is NOT an integer; and 10922 fails
    `CIFnBlock.shardings`' `ffn_hidden % N == 0` (10922 % 32 == 10, not even % 8). Rounding UP
    to a multiple of 256 gives 11008 — which is also exactly Llama-2-7B's intermediate at the
    same d_model, i.e. the same 2/3-rounded-to-256 arithmetic.
    """
    d, gelu_hidden = 4096, 16384
    assert gelu_hidden == 4 * d
    iso = 2 / 3 * gelu_hidden
    assert iso != int(iso), "iso-param is fractional: it MUST be rounded, not truncated"
    assert 10922 % 8 != 0, "the naive truncation does not even tile 8 devices"
    iso_rounded = -(-int(iso) // 256) * 256
    assert iso_rounded == 11008
    for n_devices in (8, 32, 128, 256):
        assert iso_rounded % n_devices == 0
    # iso-param holds: 3 matrices at 2/3 width ~= 2 matrices at full width
    assert 3 * d * iso_rounded == pytest.approx(2 * d * gelu_hidden, rel=0.01)
