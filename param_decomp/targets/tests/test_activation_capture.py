"""Typed selective capture: point parity, strict 1:1 binding, and empty-plan compilation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from param_decomp.core.components import SiteC, init_component_stacks
from param_decomp.core.model import BATCH_AXES, MaterializedMasking, PlacedModel
from param_decomp.core.placement import from_config
from param_decomp.core.sharding import hsdp_mesh, target_shardings_audit
from param_decomp.targets.glu_transformer import (
    GLU_ANATOMY,
    GatedMLP,
    GLUDecomposedModel,
    GLULayer,
    _capture_source_for_point,
    _clean_mlp_out,
    canonical_site_cs,
    glu_site_specs,
    site_name,
)
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm
from param_decomp.targets.transformer_taps import (
    attention_input_tap_key,
    attention_output_tap_key,
    mlp_hidden_tap_key,
    mlp_input_tap_key,
    post_attention_tap_key,
    site_output_tap_key,
)
from param_decomp.vendored_jax.llama import rms_norm


def _model():
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(
        cfg,
        canonical_site_cs(
            tuple(
                SiteC(site_name(block, kind), 4)
                for block in (2, 3)
                for kind in ("q", "k", "v", "o", "gate", "up", "down")
            )
        ),
    )
    return cfg, tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))


def _all_point_classes(block: int, n_layer: int) -> tuple[str, ...]:
    sites = tuple(site_name(block, kind) for kind in ("q", "k", "v", "o", "gate", "up", "down"))
    return (
        "resid.0",
        f"resid.{block + 1}",
        f"resid.{n_layer}",
        post_attention_tap_key(block),
        attention_input_tap_key(block),
        attention_output_tap_key(block),
        mlp_input_tap_key(block),
        mlp_hidden_tap_key(block),
        *(site_output_tap_key(site) for site in sites),
    )


def test_frozen_target_persistence_and_linear_operands_follow_the_table():
    cfg, model = _model()
    mesh = Mesh(
        np.asarray(jax.devices()).reshape(1, 1, 1),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    rules = from_config("owner", mesh, model.sites)

    shardings = model.shardings(rules)
    assert isinstance(shardings.embed, NamedSharding)
    assert isinstance(shardings.norm, NamedSharding)
    assert isinstance(shardings.lm_head, NamedSharding)
    assert isinstance(shardings.inv_freq, NamedSharding)
    assert shardings.embed.spec == P(None, "fsdp")
    assert shardings.norm.spec == P(None)
    assert shardings.lm_head.spec == P(None, "fsdp")
    assert shardings.inv_freq.spec == P(None)
    mlp_shardings = shardings.stacked.mlp
    assert isinstance(mlp_shardings, GatedMLP)
    for sharding, expected in (
        (shardings.stacked.attn.wq, P(None, "tp", "fsdp")),
        (shardings.stacked.attn.wo, P(None, "fsdp", "tp")),
        (mlp_shardings.Wg, P(None, "tp", "fsdp")),
        (mlp_shardings.Wd, P(None, "fsdp", "tp")),
    ):
        assert isinstance(sharding, NamedSharding)
        assert sharding.spec == expected

    audit = target_shardings_audit(PlacedModel(model=model, placement=rules))
    assert len(audit) == len(jax.tree.leaves(model))
    assert audit["target.embed"][0].spec == P(None, "fsdp")
    assert audit["target.stacked.mlp.Wg"][0].spec == P(None, "tp", "fsdp")

    tokens = jax.random.randint(jax.random.PRNGKey(7), (2, 8), 0, cfg.vocab_size)
    jaxpr = jax.make_jaxpr(
        lambda target, inputs: target.clean_forward(inputs, placement=rules).output
    )(model, tokens)
    scan = next(equation for equation in jaxpr.jaxpr.eqns if equation.primitive.name == "scan")
    matrix_constraints = [
        equation.params["dst_sharding"].spec
        for equation in scan.params["jaxpr"].jaxpr.eqns
        if equation.primitive.name == "reshard" and len(equation.params["dst_sharding"].spec) == 2
    ]
    # One weight-operand reshard per frozen linear, in execution order
    # q, k, v, o, gate, up, down — columns gather their fsdp shard to P(None, "tp"),
    # rows to P("tp", None).
    assert matrix_constraints == [
        P(None, "tp"),
        P(None, "tp"),
        P(None, "tp"),
        P("tp", None),
        P(None, "tp"),
        P(None, "tp"),
        P("tp", None),
    ]
    activation_constraints = [
        equation.params["dst_sharding"].spec
        for equation in scan.params["jaxpr"].jaxpr.eqns
        if equation.primitive.name == "reshard" and len(equation.params["dst_sharding"].spec) >= 3
    ]
    intermediate = P(("replicate", "fsdp"), None, "tp")
    heads = P(("replicate", "fsdp"), None, "tp", None)
    external = P(("replicate", "fsdp"), None, None)
    # Fewer explicit intermediate reshards than the constraint-based lowering had:
    # the linears' outputs are now typed at the einsum (out_sharding), so only the
    # activation-row pins remain as reshard equations.
    assert activation_constraints.count(intermediate) == 3
    assert activation_constraints.count(heads) == 3
    assert activation_constraints.count(external) == 5


def _frozen_routes_masking(
    model: GLUDecomposedModel, leading: tuple[int, ...]
) -> MaterializedMasking:
    """Total masks with all-False routes: every position takes the frozen `x @ W` path —
    the representable frozen forward now that masks must cover every site."""
    return MaterializedMasking(
        component_masks={s.name: jnp.ones((*leading, s.C)) for s in model.sites},
        routes={s.name: jnp.zeros(leading, bool) for s in model.sites},
    )


def test_clean_and_frozen_masked_paths_agree_at_every_declared_point_class():
    cfg, model = _model()
    keys = frozenset(_all_point_classes(2, cfg.n_layer))
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 8), 0, cfg.vocab_size)
    clean_forward_result = model.clean_forward(tokens, keys, placement=None)
    clean_captures = clean_forward_result.captures

    components = init_component_stacks(model.sites, jax.random.PRNGKey(2))
    masked_forward_result = model.masked_forward(
        model.prepare_compute_weights(components, None),
        tokens,
        masking=_frozen_routes_masking(model, (2, 8)),
        placement=None,
        capture_keys=keys,
        remat=True,
    )
    masked_captures = masked_forward_result.captures

    assert set(clean_captures) == set(masked_captures) == keys
    for key in keys:
        assert jnp.array_equal(masked_captures[key], clean_captures[key]), key
        assert clean_captures[key].shape[-1] == model._capture_grammar().width_of(key)
    assert jnp.array_equal(masked_forward_result.output, clean_forward_result.output)


def test_new_residual_point_classes_match_direct_block_algebra():
    cfg, model = _model()
    block = 2
    keys = frozenset(("resid.0", post_attention_tap_key(block), f"resid.{cfg.n_layer}"))
    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, 8), 0, cfg.vocab_size)
    captures = model.clean_forward(tokens, keys, placement=None).captures

    residual = model.embed_tokens(tokens, None)
    assert jnp.array_equal(captures["resid.0"], residual)
    expected_post_attention = None
    for index, layer in enumerate(model.layers):
        residual = residual + layer.attn(rms_norm(residual, layer.ln1, model.eps), model.inv_freq)
        if index == block:
            expected_post_attention = residual
        residual = residual + _clean_mlp_out(layer, rms_norm(residual, layer.ln2, model.eps), None)

    assert expected_post_attention is not None
    assert jnp.allclose(
        captures[post_attention_tap_key(block)], expected_post_attention, rtol=1e-5, atol=1e-5
    )
    assert jnp.allclose(captures[f"resid.{cfg.n_layer}"], residual, rtol=1e-5, atol=1e-5)


def test_forward_result_pytree_reconstruction_is_inert():
    _cfg, model = _model()
    tokens = jnp.ones((1, 4), jnp.int32)
    clean_forward_result = model.clean_forward(tokens, frozenset({"resid.2"}), placement=None)

    erased = jax.tree.map(lambda _value: None, clean_forward_result)
    assert erased.output is None
    assert erased.captures == dict.fromkeys(clean_forward_result.captures)

    shaped = jax.eval_shape(lambda tree: tree, clean_forward_result)
    assert all(isinstance(value, jax.ShapeDtypeStruct) for value in shaped.captures.values())


def test_shared_input_has_one_canonical_capture_key():
    _cfg, model = _model()
    qkv = tuple(site_name(2, kind) for kind in ("q", "k", "v"))
    tokens = jnp.ones((1, 4), jnp.int32)
    for site in qkv:
        with pytest.raises(AssertionError, match="unknown transformer activation"):
            model.clean_forward(tokens, frozenset({site}), placement=None)

    clean_forward_result = model.clean_forward(
        tokens, frozenset({attention_input_tap_key(2)}), placement=None
    )
    assert clean_forward_result.captures.keys() == {attention_input_tap_key(2)}


@pytest.mark.multidevice
def test_capture_values_are_batch_sharded_at_the_producer():
    cfg, model = _model()
    capture_keys = frozenset({"resid.2", attention_input_tap_key(2)})
    mesh = hsdp_mesh(1, jax.device_count(), 1)
    tokens = jax.random.randint(
        jax.random.PRNGKey(3), (2 * mesh.devices.size, 8), 0, cfg.vocab_size
    )

    components = model.prepare_compute_weights(
        init_component_stacks(model.sites, jax.random.PRNGKey(5)), None
    )
    masking = _frozen_routes_masking(model, tokens.shape)
    with jax.set_mesh(mesh):
        clean_forward_result = jax.jit(
            lambda m, x: m.clean_forward(x, capture_keys, placement=None)
        )(model, tokens)
        masked_forward_result = jax.jit(
            lambda m, prepared, x: m.masked_forward(
                prepared,
                x,
                masking=masking,
                capture_keys=capture_keys,
                placement=None,
                remat=True,
            )
        )(model, components, tokens)

    expected = NamedSharding(mesh, P(BATCH_AXES, None, None))
    for forward_result in (clean_forward_result, masked_forward_result):
        assert forward_result.captures
        assert all(
            value.sharding.is_equivalent_to(expected, value.ndim)
            for value in forward_result.captures.values()
        )


def test_capture_sources_are_unique_and_request_aligned():
    _cfg, model = _model()
    keys = (
        site_output_tap_key(site_name(3, "down")),
        "resid.1",
        attention_input_tap_key(2),
    )
    sources = model._capture_grammar().resolve(
        keys, lambda point: _capture_source_for_point(GLU_ANATOMY, point)
    )
    assert len(keys) == len(sources) == len(set(sources))


def test_no_capture_wrapper_lowers_to_the_compact_clean_graph():
    _cfg, model = _model()
    tokens = jnp.ones((1, 4), jnp.int32)

    def compact_clean(m: GLUDecomposedModel, x: Array) -> Array:
        def block(residual: Array, layer: GLULayer) -> tuple[Array, None]:
            residual = residual + layer.attn(rms_norm(residual, layer.ln1, m.eps), m.inv_freq)
            residual = residual + _clean_mlp_out(layer, rms_norm(residual, layer.ln2, m.eps), None)
            return residual, None

        residual = m.embed_tokens(x, None)
        residual, _ = jax.lax.scan(block, residual, m.stacked)
        residual = rms_norm(residual, m.norm, m.eps)
        return residual @ m.head_weight.T

    direct = jax.jit(lambda m, x: compact_clean(m, x)).lower(model, tokens).as_text()
    public = (
        jax.jit(lambda m, x: m.clean_forward(x, placement=None).output)
        .lower(model, tokens)
        .as_text()
    )
    assert public == direct


def test_resolution_fails_at_first_trace_for_unknown_points():
    _cfg, model = _model()
    tokens = jnp.ones((1, 4), jnp.int32)
    with pytest.raises(AssertionError, match="unknown transformer activation"):
        jax.jit(
            lambda m, x: (
                m.clean_forward(x, frozenset({"python_local_variable"}), placement=None).output
            )
        ).lower(model, tokens)


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) < 8, reason="requires eight local devices")
def test_attention_refuses_a_tp_that_does_not_divide_the_kv_head_count():
    """The qkv activation row shards the head axis over tp, and GQA keeps distinct
    query and key/value head counts (tiny cfg: 4 q heads, 2 KV heads; Llama-8B: 32/8) —
    so a tp that divides the flat qd/kvd widths but not a head count must refuse at
    first trace rather than silently pad-shard k/v."""
    _cfg, model = _model()
    tokens = jnp.zeros((8, 8), jnp.int32)

    def trace(replicate: int, fsdp: int, tp: int) -> None:
        devices = np.asarray(jax.devices()[: replicate * fsdp * tp])
        mesh = Mesh(
            devices.reshape(replicate, fsdp, tp),
            ("replicate", "fsdp", "tp"),
            axis_types=(AxisType.Explicit,) * 3,
        )
        rules = from_config("zero1", mesh, model.sites)
        jax.make_jaxpr(lambda m, x: m.clean_forward(x, placement=rules).output)(model, tokens)

    trace(1, 4, 2)  # both head counts tile tp=2
    with pytest.raises(AssertionError, match=r"'kv_head' \(dim 2\) does not tile"):
        trace(1, 2, 4)  # tp=4 divides qd=32 and kvd=16, but not the 2 KV heads
