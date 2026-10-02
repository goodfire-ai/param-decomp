"""Packed documents reproduce independent forwards and derivatives."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.attention import causal_attention_head_first
from param_decomp.core.components import SiteC, init_component_stacks
from param_decomp.core.model import ForwardResult, MaterializedMasking
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.pretrain.models import (
    GPT2SimpleConfig,
    LlamaSimpleConfig,
    LlamaSimpleMLPConfig,
    init_model,
)
from param_decomp.sequence import SequenceLayout
from param_decomp.targets.lm_output import LMOutput
from param_decomp.targets.testing import materialized_logits, tiny_glu_cfg, tiny_glu_decomposed_lm
from param_decomp.targets.transformer import (
    TransformerConfig,
    TransformerDecomposedModel,
    glu_site_specs,
)
from param_decomp.tests.targets.test_qwen3 import _tiny_decomposed_qwen, _tiny_qwen_cfg


def _packed_batch() -> LMBatchWithDocuments:
    return LMBatchWithDocuments(
        LMBatch(jnp.array([[1, 2, 3, 4, 5, 6]], jnp.int32)),
        SequenceLayout(jnp.array([[0, 0, 0, 1, 1, 1]], jnp.int32)),
    )


def test_causal_attention_head_first_isolates_outputs_and_value_gradients():
    q, k, v = jax.random.normal(jax.random.key(0), (3, 1, 2, 6, 4))
    sequence = _packed_batch().sequence
    packed = causal_attention_head_first(q, k, v, sequence, None, "xla")
    independent = jnp.concatenate(
        [
            causal_attention_head_first(
                q[:, :, start : start + 3],
                k[:, :, start : start + 3],
                v[:, :, start : start + 3],
                SequenceLayout(jnp.zeros((1, 3), jnp.int32)),
                None,
                "xla",
            )
            for start in (0, 3)
        ],
        axis=2,
    )
    np.testing.assert_allclose(packed, independent, atol=1e-6)
    gradient = jax.grad(
        lambda values: causal_attention_head_first(q, k, values, sequence, None, "xla")[
            :, :, 3:
        ].sum()
    )(v)
    np.testing.assert_array_equal(gradient[:, :, :3], 0)
    single = causal_attention_head_first(
        q, k, v, SequenceLayout(jnp.zeros((1, 6), jnp.int32)), None, "xla"
    )
    reference = jax.nn.dot_product_attention(
        q.transpose(0, 2, 1, 3),
        k.transpose(0, 2, 1, 3),
        v.transpose(0, 2, 1, 3),
        is_causal=True,
        implementation="xla",
    ).transpose(0, 2, 1, 3)
    np.testing.assert_array_equal(single, reference)


@pytest.mark.parametrize("qwen", [False, True], ids=["llama", "qwen"])
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_glu_packed_outputs_captures_and_embedding_gradients(
    qwen: bool, capture: bool, masked: bool
):
    cfg = replace(_tiny_qwen_cfg() if qwen else tiny_glu_cfg(), n_layer=3)
    sites = glu_site_specs(
        cfg, (SiteC("layers.1.self_attn.q_proj", 4), SiteC("layers.1.mlp.down_proj", 4))
    )
    model = (
        _tiny_decomposed_qwen(cfg, sites, jax.random.key(0))
        if isinstance(cfg, TransformerConfig)
        else tiny_glu_decomposed_lm(cfg, sites, jax.random.key(0))
    )
    components = init_component_stacks(sites, jax.random.key(1))
    weights = model.prepare_compute_weights(components, None)
    packed = _packed_batch()
    keys = frozenset({"resid.1", "resid.3"}) if capture else frozenset()

    def forward(
        target: TransformerDecomposedModel, inputs: LMBatchWithDocuments
    ) -> ForwardResult[LMOutput, LMBatchWithDocuments]:
        if masked:
            return target.masked_forward(
                weights,
                inputs,
                masking=target.prepare_masking(
                    MaterializedMasking(
                        component_masks={
                            site.name: jnp.full((*inputs.batch.token_ids.shape, site.C), 0.5)
                            for site in sites
                        },
                        weight_delta_masks=None,
                    )
                ),
                routes=None,
                placement=None,
                capture_keys=keys,
                remat=True,
            )
        return target.clean_forward(inputs, keys, placement=None)

    result = eqx.filter_jit(forward)(model, packed)
    independent = [
        forward(
            model,
            LMBatchWithDocuments.from_unsegmented_sequences(
                packed.batch.token_ids[:, start : start + 3]
            ),
        )
        for start in (0, 3)
    ]
    np.testing.assert_allclose(
        materialized_logits(result.output),
        jnp.concatenate([materialized_logits(item.output) for item in independent], axis=1),
        atol=2e-6,
        rtol=2e-5,
    )
    for key in keys:
        np.testing.assert_allclose(
            result.captures[key],
            jnp.concatenate([item.captures[key] for item in independent], axis=1),
            atol=2e-6,
            rtol=2e-5,
        )
    assert result.sequence is not None
    np.testing.assert_array_equal(result.sequence.document_ids, packed.sequence.document_ids)

    def suffix_loss(embed: jax.Array) -> jax.Array:
        target = eqx.tree_at(lambda m: m.embed, model, embed)
        return jnp.sum(materialized_logits(forward(target, packed).output)[:, 3:] ** 2)

    gradient = jax.grad(suffix_loss)(model.embed)
    np.testing.assert_array_equal(gradient[jnp.array([1, 2, 3])], 0)


@pytest.mark.parametrize("model_type", ["GPT2Simple", "LlamaSimple", "LlamaSimpleMLP"])
def test_pretrain_packed_outputs_and_parameter_gradients(model_type: str):
    match model_type:
        case "GPT2Simple":
            cfg = GPT2SimpleConfig(
                model_type="GPT2Simple", block_size=8, vocab_size=16, n_layer=1, n_head=2, n_embd=8
            )
        case "LlamaSimple":
            cfg = LlamaSimpleConfig(
                model_type="LlamaSimple",
                block_size=8,
                vocab_size=16,
                n_layer=1,
                n_head=2,
                n_embd=8,
                n_intermediate=16,
                n_ctx=8,
                n_key_value_heads=1,
            )
        case "LlamaSimpleMLP":
            cfg = LlamaSimpleMLPConfig(
                model_type="LlamaSimpleMLP",
                block_size=8,
                vocab_size=16,
                n_layer=1,
                n_head=2,
                n_embd=8,
                n_intermediate=16,
                n_ctx=8,
                n_key_value_heads=1,
            )
        case other:
            raise AssertionError(other)
    model = init_model(cfg, jax.random.key(0), "xla")
    packed = _packed_batch()
    separate = tuple(
        LMBatchWithDocuments.from_unsegmented_sequences(
            packed.batch.token_ids[:, start : start + 3]
        )
        for start in (0, 3)
    )
    np.testing.assert_allclose(
        model(packed), jnp.concatenate([model(batch) for batch in separate], axis=1), atol=1e-6
    )
    packed_gradient = eqx.filter_grad(lambda target: jnp.sum(target(packed) ** 2))(model)
    separate_gradient = eqx.filter_grad(
        lambda target: sum(jnp.sum(target(batch) ** 2) for batch in separate)
    )(model)
    for actual, expected in zip(
        jax.tree.leaves(packed_gradient), jax.tree.leaves(separate_gradient), strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=2e-5)


@pytest.mark.multidevice
def test_glu_document_layout_follows_batch_sharding():
    from jax.sharding import AxisType, Mesh

    from param_decomp.core.placement import from_config
    from param_decomp.core.sharding import place_target, shard_batch

    cfg = replace(tiny_glu_cfg(), n_layer=1)
    sites = glu_site_specs(cfg, (SiteC("layers.0.self_attn.q_proj", 4),))
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.key(0))
    one_row = _packed_batch()
    batch = jax.tree.map(lambda x: jnp.tile(x, (4, 1)), one_row)
    reference = model.clean_forward(batch, placement=None).output
    mesh = Mesh(
        np.array(jax.devices()[:4]).reshape(2, 2, 1),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    rules = from_config("zero1", mesh, sites)
    placed = place_target(model, rules)
    with jax.set_mesh(mesh):
        sharded = jax.tree.map(lambda x: shard_batch(x, mesh, batch_axis=0), batch)
        result = eqx.filter_jit(lambda target, inputs: target.clean_forward(inputs))(
            placed, sharded
        )
    np.testing.assert_allclose(
        materialized_logits(result.output), materialized_logits(reference), atol=1e-6, rtol=1e-5
    )
    assert result.sequence is not None
    assert result.sequence.document_ids.sharding == sharded.sequence.document_ids.sharding
