"""The streamed model-output edge: `StreamedLinearOutput` + the `targets.losses`
streamed kernels against their materialized twins.

Parity is checked at both scales the edge serves: a tiny vocab (every chunking, both
frozen-weight dtypes, gradients) and the production 248,320-entry vocab (bf16 inputs,
fp32 accumulators — the reassociation regime the large-capacity reference uses). The kernels
fp32-accumulate each chunk's logits from the native-dtype operands, so the kernel-level
references form the full logits the same way (`_fp32_logits`) and the two spellings
differ only by fp32 reassociation in the softmax reductions; gradients differ by bf16
ulps through the shared matmul transpose. The model-level tests pin the GLU and qwen36_moe
edge flips, where the materialized edge's bf16-rounded logits are the one seam between
the edges; the CE/KL eval-tier flip lives with the eval suite
(`experiments/lm/test_eval.py`)."""

import dataclasses
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from jax.typing import DTypeLike
from jaxtyping import Array, TypeCheckError

from param_decomp.core.components import ComponentStacks, SiteC, init_component_stacks
from param_decomp.core.model import MaterializedMasking, PlacedModel, StochasticMasking
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.targets.llama_simple_mlp import site_specs as simple_mlp_site_specs
from param_decomp.targets.lm_output import (
    MaterializedOutputEdge,
    StreamedLinearOutput,
    StreamedOutputEdge,
)
from param_decomp.targets.losses import (
    _CEAccumulator,
    _KLAccumulator,
    kl_per_position,
    lm_output_position_kl,
    lm_output_position_next_token_ce,
    streamed_kl_per_position,
    streamed_position_ce,
    streamed_position_kl,
    streamed_position_next_token_ce,
)
from param_decomp.targets.qwen36_moe import (
    Qwen36MoeDecomposedModel,
    full_site_cs,
    qwen36_35b_a3b_config,
    qwen36_moe_site_specs,
)
from param_decomp.targets.testing import (
    TINY_QWEN36_CS,
    constant_mask_values,
    site_masks,
    tiny_glu_cfg,
    tiny_glu_decomposed_lm,
    tiny_qwen36_cfg,
    tiny_qwen36_decomposed_model,
    tiny_simple_mlp_cfg,
    tiny_simple_mlp_decomposed_model,
)
from param_decomp.targets.transformer import (
    TransformerDecomposedModel,
    build_decomposed_lm,
    glu_site_specs,
)


def _fp32_logits(activations: Array, head: Array) -> Array:
    """The full logits formed as the streamed kernels form each chunk: native-dtype
    operands, fp32 accumulation."""
    return jnp.dot(activations, head.T, preferred_element_type=jnp.float32)


def _materialized_next_token_ce(logits: Array, token_ids: Array) -> Array:
    """Mean next-token CE for fixtures with unsegmented sequences."""
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    labels = jnp.take_along_axis(log_probs[:, :-1], token_ids[:, 1:, None], axis=-1)[..., 0]
    return -labels.mean()


def _random_pair(
    key: Array, shape: tuple[int, int, int], vocab: int, dtype: DTypeLike
) -> tuple[Array, Array, Array, Array]:
    k_clean, k_masked, k_head, k_tokens = random.split(key, 4)
    batch, seq, d_model = shape
    return (
        random.normal(k_clean, (batch, seq, d_model), dtype),
        random.normal(k_masked, (batch, seq, d_model), dtype),
        random.normal(k_head, (vocab, d_model), dtype),
        random.randint(k_tokens, (batch, seq), 0, vocab),
    )


@pytest.mark.parametrize("record_type", [_CEAccumulator, _KLAccumulator])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("defect", ["shape", "dtype"])
def test_accumulators_check_every_field(
    record_type: type[_CEAccumulator] | type[_KLAccumulator],
    compiled: bool,
    defect: Literal["shape", "dtype"],
):
    values = jnp.ones((2, 3), dtype=jnp.float32)
    match defect:
        case "shape":
            invalid = jnp.ones((2, 1), dtype=jnp.float32)
        case "dtype":
            invalid = values.astype(jnp.bfloat16)
    fields = dataclasses.fields(record_type)
    construct = jax.jit(record_type) if compiled else record_type
    for field in fields:
        arguments = {item.name: invalid if item == field else values for item in fields}
        with pytest.raises(TypeCheckError):
            construct(**arguments)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("n_chunks", [1, 4, 64])
def test_streamed_kl_and_ce_match_materialized_tiny_vocab(dtype: DTypeLike, n_chunks: int):
    vocab = 64
    h_clean, h_masked, head, tokens = _random_pair(random.PRNGKey(0), (3, 5, 16), vocab, dtype)
    clean = StreamedLinearOutput(activations=h_clean, head=head, n_chunks=n_chunks)
    masked = StreamedLinearOutput(activations=h_masked, head=head, n_chunks=n_chunks)

    kl_expected = kl_per_position(_fp32_logits(h_masked, head), _fp32_logits(h_clean, head))
    np.testing.assert_allclose(
        np.asarray(streamed_kl_per_position(masked, clean)),
        np.asarray(kl_expected),
        rtol=1e-6,
        atol=1e-6,
    )
    ce_expected = _materialized_next_token_ce(_fp32_logits(h_masked, head), tokens)
    np.testing.assert_allclose(
        np.asarray(streamed_position_next_token_ce(masked, tokens).mean()),
        np.asarray(ce_expected),
        rtol=1e-6,
        atol=1e-6,
    )


def test_streamed_gradients_match_materialized():
    vocab, n_chunks = 64, 4
    h_clean, h_masked, head, tokens = _random_pair(
        random.PRNGKey(1), (3, 5, 16), vocab, jnp.bfloat16
    )
    logits_clean = _fp32_logits(h_clean, head)

    def package(activations: Array) -> StreamedLinearOutput:
        return StreamedLinearOutput(
            activations=activations.astype(jnp.bfloat16), head=head, n_chunks=n_chunks
        )

    def assert_close_frobenius(got: Array, expected: Array) -> None:
        # both paths pull the softmax cotangent back through one bf16 matmul transpose;
        # the streamed chunking reassociates its products, so pointwise ulp bounds don't
        # hold on the smallest elements — the norm-level one does (the placed suites'
        # criterion; observed ~2e-3)
        got_np, expected_np = np.asarray(got), np.asarray(expected)
        error = np.linalg.norm(got_np - expected_np) / np.linalg.norm(expected_np)
        assert error < 1e-2, error

    clean = package(h_clean)
    grad_kl_materialized = jax.grad(
        lambda h: kl_per_position(_fp32_logits(h.astype(jnp.bfloat16), head), logits_clean)
    )(h_masked.astype(jnp.float32))
    grad_kl_streamed = jax.jit(jax.grad(lambda h: streamed_kl_per_position(package(h), clean)))(
        h_masked.astype(jnp.float32)
    )
    assert_close_frobenius(grad_kl_streamed, grad_kl_materialized)

    grad_ce_materialized = jax.grad(
        lambda h: _materialized_next_token_ce(_fp32_logits(h.astype(jnp.bfloat16), head), tokens)
    )(h_masked.astype(jnp.float32))
    grad_ce_streamed = jax.jit(
        jax.grad(lambda h: streamed_position_next_token_ce(package(h), tokens).mean())
    )(h_masked.astype(jnp.float32))
    assert_close_frobenius(grad_ce_streamed, grad_ce_materialized)


def test_streamed_kernels_match_at_the_production_vocab():
    """The large-vocabulary regime: the full 248,320-entry axis, bf16 logits chunks, fp32
    online accumulators — the reassociation the tiny-vocab cases cannot exercise."""
    vocab = qwen36_35b_a3b_config().vocab_size
    n_chunks = 32
    assert vocab % n_chunks == 0, (vocab, n_chunks)
    h_clean, h_masked, head, tokens = _random_pair(
        random.PRNGKey(2), (2, 4, 64), vocab, jnp.bfloat16
    )
    clean = StreamedLinearOutput(activations=h_clean, head=head, n_chunks=n_chunks)
    masked = StreamedLinearOutput(activations=h_masked, head=head, n_chunks=n_chunks)

    kl_expected = kl_per_position(_fp32_logits(h_masked, head), _fp32_logits(h_clean, head))
    np.testing.assert_allclose(
        np.asarray(streamed_kl_per_position(masked, clean)),
        np.asarray(kl_expected),
        rtol=1e-5,
        atol=1e-5,
    )
    ce_expected = _materialized_next_token_ce(_fp32_logits(h_masked, head), tokens)
    np.testing.assert_allclose(
        np.asarray(streamed_position_next_token_ce(masked, tokens).mean()),
        np.asarray(ce_expected),
        rtol=1e-5,
        atol=1e-5,
    )


def test_per_position_kernels_match_materialized_reductions():
    h_clean, h_masked, head, tokens = _random_pair(random.PRNGKey(3), (2, 6, 8), 32, jnp.float32)
    clean = StreamedLinearOutput(activations=h_clean, head=head, n_chunks=4)
    masked = StreamedLinearOutput(activations=h_masked, head=head, n_chunks=4)
    position_kl = streamed_position_kl(masked, clean)
    assert position_kl.shape == (2, 6)
    np.testing.assert_allclose(
        np.asarray(position_kl.mean()), np.asarray(streamed_kl_per_position(masked, clean))
    )
    position_ce = streamed_position_next_token_ce(masked, tokens)
    assert position_ce.shape == (2, 5)
    np.testing.assert_allclose(
        np.asarray(position_ce.mean()),
        np.asarray(_materialized_next_token_ce(_fp32_logits(h_masked, head), tokens)),
    )


def test_per_position_kernels_agree_across_the_output_edge():
    h_clean, h_masked, head, tokens = _random_pair(random.PRNGKey(4), (2, 6, 8), 32, jnp.float32)
    clean = StreamedLinearOutput(activations=h_clean, head=head, n_chunks=4)
    masked = StreamedLinearOutput(activations=h_masked, head=head, n_chunks=4)
    clean_logits, masked_logits = _fp32_logits(h_clean, head), _fp32_logits(h_masked, head)
    np.testing.assert_allclose(
        np.asarray(lm_output_position_kl(masked_logits, clean_logits)),
        np.asarray(lm_output_position_kl(masked, clean)),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(lm_output_position_next_token_ce(masked_logits, tokens)),
        np.asarray(lm_output_position_next_token_ce(masked, tokens)),
        rtol=1e-5,
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("n_chunks", [1, 4, 32])
def test_streamed_ce_preserves_vocabulary_boundaries(dtype: DTypeLike, n_chunks: int):
    labels = jnp.asarray(
        [[jnp.iinfo(jnp.int32).min, -1, 0, 7, 8, 15, 16, 23, 24, 31, 32, jnp.iinfo(jnp.int32).max]],
        jnp.int32,
    )
    activations, _, head, _ = _random_pair(random.PRNGKey(51), (1, labels.size, 8), 32, dtype)
    output = StreamedLinearOutput(activations, head, n_chunks)
    actual = jax.jit(streamed_position_ce)(output, labels)
    valid = (labels >= 0) & (labels < head.shape[0])
    expected = -jnp.take_along_axis(
        jax.nn.log_softmax(_fp32_logits(activations, head), axis=-1),
        jnp.clip(labels, 0, head.shape[0] - 1)[..., None],
        axis=-1,
    )[..., 0]
    assert actual.dtype == jnp.float32
    np.testing.assert_array_equal(jnp.isnan(actual), ~valid)
    np.testing.assert_allclose(actual[valid], expected[valid], rtol=1e-6, atol=1e-6)


def test_streamed_ce_rejects_a_partial_final_vocabulary_chunk():
    output = StreamedLinearOutput(jnp.ones((1, 2, 4)), jnp.ones((10, 4)), 3)
    with pytest.raises(AssertionError):
        streamed_position_ce(output, jnp.asarray([[0, 9]], jnp.int32))


def test_streamed_ce_yields_nan_for_an_out_of_range_label_like_its_twin():
    """A label in no vocab chunk must not come out as a finite CE (it would be logZ): it is
    NaN — for an overflowing label where the materialized `take_along_axis` fill puts one,
    and for a negative label too (the twin wraps those) — and in-range positions of the
    same batch are untouched."""
    h_clean, _, head, tokens = _random_pair(random.PRNGKey(5), (2, 4, 8), 32, jnp.float32)
    out_of_range = tokens.at[0, 1].set(32).at[1, 3].set(-1)
    package = StreamedLinearOutput(activations=h_clean, head=head, n_chunks=4)
    streamed = np.asarray(streamed_position_next_token_ce(package, out_of_range))
    materialized = -np.asarray(
        jnp.take_along_axis(
            jax.nn.log_softmax(h_clean @ head.T, axis=-1)[:, :-1],
            out_of_range[:, 1:, None],
            axis=-1,
        )[..., 0]
    )
    bad = np.zeros_like(streamed, dtype=bool)
    bad[0, 0] = bad[1, 2] = True
    np.testing.assert_array_equal(np.isnan(streamed), bad)
    assert np.isnan(materialized[0, 0]) and np.isfinite(materialized[1, 2])
    np.testing.assert_allclose(streamed[~bad], materialized[~bad], rtol=1e-6, atol=1e-6)


def test_recon_loss_fn_dispatches_on_the_output_edge_and_refuses_a_mix():
    h_clean, h_masked, head, _ = _random_pair(random.PRNGKey(4), (2, 3, 8), 32, jnp.float32)
    clean = StreamedLinearOutput(activations=h_clean, head=head, n_chunks=4)
    masked = StreamedLinearOutput(activations=h_masked, head=head, n_chunks=4)
    streamed = Qwen36MoeDecomposedModel.recon_loss_fn(masked, clean)
    materialized = Qwen36MoeDecomposedModel.recon_loss_fn(h_masked @ head.T, h_clean @ head.T)
    np.testing.assert_allclose(np.asarray(streamed), np.asarray(materialized), rtol=1e-6, atol=1e-6)
    with pytest.raises(AssertionError, match="mixed model-output edges"):
        Qwen36MoeDecomposedModel.recon_loss_fn(masked, h_clean @ head.T)


# ── the qwen36_moe edge flip, model level ─────────────────────────────────────


def _model_and_vu(key: Array) -> tuple[Qwen36MoeDecomposedModel, ComponentStacks]:
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, TINY_QWEN36_CS))
    model_key, vu_key = jax.random.split(key)
    return tiny_qwen36_decomposed_model(cfg, sites, model_key), init_component_stacks(sites, vu_key)


def _streamed_twin(model: Qwen36MoeDecomposedModel, n_chunks: int) -> Qwen36MoeDecomposedModel:
    return dataclasses.replace(model, output_edge=StreamedOutputEdge(n_vocab_chunks=n_chunks))


def test_qwen36_edges_share_one_forward():
    """The edge flip changes only the output's spelling: the streamed package's
    materialized product is bit-identical to the materialized edge's logits."""
    model, _vu = _model_and_vu(jax.random.PRNGKey(0))
    assert model.output_edge == MaterializedOutputEdge()
    tokens = jax.random.randint(jax.random.PRNGKey(7), (2, 12), 0, model.cfg.vocab_size)

    logits = PlacedModel(model=model, placement=None).clean_forward(LMBatch(tokens)).output
    package = (
        PlacedModel(model=_streamed_twin(model, 4), placement=None)
        .clean_forward(LMBatch(tokens))
        .output
    )
    assert isinstance(logits, jax.Array)
    assert isinstance(package, StreamedLinearOutput)
    assert package.n_chunks == 4
    np.testing.assert_array_equal(
        np.asarray(package.activations @ package.head.T), np.asarray(logits)
    )


def test_qwen36_masked_recon_grads_agree_across_the_edge_flip():
    """d(recon KL)/d(V/U) through the stochastic masked forward, streamed vs
    materialized — the training path's gradient, edge-flip invariant."""
    model, components = _model_and_vu(jax.random.PRNGKey(0))
    tokens = jax.random.randint(jax.random.PRNGKey(7), (2, 12), 0, model.cfg.vocab_size)

    def loss(target: Qwen36MoeDecomposedModel, value: ComponentStacks) -> Array:
        placed = PlacedModel(model=target, placement=None)
        clean = jax.tree.map(
            jax.lax.stop_gradient,
            placed.clean_forward(LMBatch(tokens)),
        )
        ci = site_masks(
            target, clean.conditioning.selection, constant_mask_values(target, tokens.shape, 0.4)
        )
        masked = placed.masked_forward(
            placed.prepare_compute_weights(value),
            clean.conditioning,
            masking=placed.model.prepare_masking(
                StochasticMasking(ci=ci, draw_key=jax.random.PRNGKey(3))
            ),
            routes=None,
            remat=True,
        ).output
        return placed.recon_loss_fn(masked, clean.output)

    grads_materialized = jax.jit(jax.grad(loss, argnums=1))(model, components)
    grads_streamed = jax.jit(jax.grad(loss, argnums=1))(_streamed_twin(model, 4), components)
    for got, expected in zip(
        jax.tree.leaves(grads_streamed), jax.tree.leaves(grads_materialized), strict=True
    ):
        got_np, expected_np = np.asarray(got), np.asarray(expected)
        denom = np.linalg.norm(expected_np)
        error = np.linalg.norm(got_np - expected_np) / (denom if denom > 0 else 1.0)
        assert error < 2e-2, error


@pytest.fixture(params=["llama", "simple_mlp"])
def glu_model(request: pytest.FixtureRequest) -> TransformerDecomposedModel:
    family: Literal["llama", "simple_mlp"] = request.param
    match family:
        case "llama":
            cfg = dataclasses.replace(tiny_glu_cfg(), n_layer=3)
            sites = glu_site_specs(
                cfg,
                (
                    SiteC("layers.1.self_attn.q_proj", 8),
                    SiteC("layers.1.mlp.down_proj", 8),
                ),
            )
            return tiny_glu_decomposed_lm(cfg, sites, random.PRNGKey(0))
        case "simple_mlp":
            cfg = dataclasses.replace(tiny_simple_mlp_cfg(), n_layer=3)
            sites = simple_mlp_site_specs(
                cfg,
                (SiteC("h.1.attn.q_proj", 8), SiteC("h.1.mlp.down_proj", 8)),
            )
            return tiny_simple_mlp_decomposed_model(cfg, sites, random.PRNGKey(0))


@pytest.mark.parametrize("capture", [False, True])
def test_glu_edges_preserve_clean_and_masked_forwards(
    glu_model: TransformerDecomposedModel, capture: bool
):
    model = glu_model
    assert model.output_edge == MaterializedOutputEdge()
    streamed = dataclasses.replace(model, output_edge=StreamedOutputEdge(n_vocab_chunks=4))
    tokens = random.randint(random.PRNGKey(7), (2, 8), 0, model.embed.shape[0])
    components = init_component_stacks(model.sites, random.PRNGKey(1))
    captures = frozenset(model.site_output_keys(model.site_names)) if capture else frozenset()
    masking = MaterializedMasking(
        component_masks={site.name: jnp.full((2, 8, site.C), 0.4) for site in model.sites}
    )

    def forwards(target: TransformerDecomposedModel):
        return (
            target.clean_forward(
                LMBatchWithDocuments.from_unsegmented_sequences(tokens), captures, placement=None
            ),
            target.masked_forward(
                target.prepare_compute_weights(components, None),
                LMBatchWithDocuments.from_unsegmented_sequences(tokens),
                masking=target.prepare_masking(masking),
                routes=None,
                placement=None,
                capture_keys=captures,
                remat=True,
            ),
        )

    reference = eqx.filter_jit(forwards)(model)
    actual = eqx.filter_jit(forwards)(streamed)
    for got, expected in zip(actual, reference, strict=True):
        assert isinstance(got.output, StreamedLinearOutput)
        assert isinstance(expected.output, jax.Array)
        assert got.output.n_chunks == 4
        np.testing.assert_array_equal(
            np.asarray(got.output.activations @ got.output.head.T), np.asarray(expected.output)
        )
        np.testing.assert_array_equal(np.asarray(got.output.head), np.asarray(model.head_weight))
        assert set(got.captures) == set(expected.captures) == captures
        for key in captures:
            np.testing.assert_array_equal(
                np.asarray(got.captures[key]), np.asarray(expected.captures[key])
            )


@pytest.mark.parametrize("remat", [False, True])
def test_glu_masked_recon_values_and_gradients_agree_across_edges(
    glu_model: TransformerDecomposedModel, remat: bool
):
    model = glu_model
    tokens = random.randint(random.PRNGKey(7), (2, 8), 0, model.embed.shape[0])
    components = init_component_stacks(model.sites, random.PRNGKey(1))

    def loss(target: TransformerDecomposedModel, value: ComponentStacks) -> Array:
        placed = PlacedModel(model=target, placement=None)
        clean = jax.tree.map(
            jax.lax.stop_gradient,
            placed.clean_forward(LMBatchWithDocuments.from_unsegmented_sequences(tokens)),
        )
        ci = {site.name: jnp.full((2, 8, site.C), 0.4) for site in target.sites}
        masked = placed.masked_forward(
            placed.prepare_compute_weights(value),
            clean.conditioning,
            masking=placed.model.prepare_masking(
                StochasticMasking(ci=ci, draw_key=random.PRNGKey(3))
            ),
            routes=None,
            remat=remat,
        ).output
        return placed.recon_loss_fn(masked, clean.output)

    value_and_grad = eqx.filter_jit(jax.value_and_grad(loss, argnums=1))
    expected_loss, expected_grads = value_and_grad(model, components)
    actual_loss, actual_grads = value_and_grad(
        dataclasses.replace(model, output_edge=StreamedOutputEdge(n_vocab_chunks=4)), components
    )
    np.testing.assert_allclose(
        np.asarray(actual_loss), np.asarray(expected_loss), rtol=2e-2, atol=1e-6
    )
    for got, expected in zip(
        jax.tree.leaves(actual_grads), jax.tree.leaves(expected_grads), strict=True
    ):
        got_np, expected_np = np.asarray(got), np.asarray(expected)
        denominator = np.linalg.norm(expected_np)
        assert denominator > 0
        error = np.linalg.norm(got_np - expected_np) / denominator
        assert error < 2e-2, error


def test_glu_builder_accepts_streamed_output_at_construction():
    cfg = dataclasses.replace(tiny_glu_cfg(), n_layer=1)
    weights = tiny_glu_decomposed_lm(cfg, (), random.PRNGKey(0))
    edge = StreamedOutputEdge(n_vocab_chunks=4)
    model = build_decomposed_lm(
        embed=weights.embed,
        layers=weights.layers,
        norm=weights.norm,
        lm_head=weights.lm_head,
        inv_freq=weights.inv_freq,
        cfg=cfg,
        sites=(),
        output_edge=edge,
    )
    assert model.output_edge == edge
    tokens = jnp.zeros((1, 4), dtype=jnp.int32)
    output = model.clean_forward(
        LMBatchWithDocuments.from_unsegmented_sequences(tokens), placement=None
    ).output
    assert isinstance(output, StreamedLinearOutput)
    assert output.n_chunks == 4
    np.testing.assert_array_equal(
        np.asarray(output.activations @ output.head.T),
        np.asarray(
            weights.clean_forward(
                LMBatchWithDocuments.from_unsegmented_sequences(tokens), placement=None
            ).output
        ),
    )
