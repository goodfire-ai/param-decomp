"""Arithmetic examples pin useful target FLOPs and the differentiated prefix."""

import math
from dataclasses import replace
from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from jax.core import ShapedArray
from jax.extend import core

from param_decomp.core.components import DenseFactorization, SiteC, SiteSpec
from param_decomp.core.decomposed_linear import SiteWeights, site_forward
from param_decomp.core.flops.target import GradientTarget, linear_flops
from param_decomp.target_ports.qwen3_5_moe import gated_delta_rule
from param_decomp.targets.qwen36_moe import (
    _delta_rule_flops,
    layer_is_full_attention,
    qwen36_moe_flops,
    qwen36_moe_site_specs,
    site_name,
)
from param_decomp.targets.testing import tiny_qwen36_cfg
from param_decomp.targets.transformer import (
    GLU_ANATOMY,
    TransformerConfig,
    glu_site_specs,
    transformer_flops,
)


def _dense_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=20,
        n_layer=1,
        n_head=2,
        n_kv_head=1,
        n_embd=8,
        n_intermediate=12,
        head_dim=4,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        max_position_embeddings=16,
        tie_word_embeddings=False,
    )


def test_clean_glu_counts_projections_causal_attention_and_head() -> None:
    cfg = _dense_config()
    cost = transformer_flops(
        cfg,
        GLU_ANATOMY,
        (),
        batch_size=2,
        sequence_length=3,
        gradients="none",
        include_frozen_paths=False,
        capture_keys=frozenset(),
    ).flops
    linear_weights = 8 * 8 + 2 * 8 * 4 + 8 * 8 + 3 * 8 * 12 + 8 * 20
    causal_attention = 2 * 2 * 2 * (3 * 4 // 2) * 4 * 2
    assert cost.forward == 2 * (2 * 3) * linear_weights + causal_attention
    assert cost.backward == 0


@pytest.mark.parametrize("frozen_paths", [False, True])
def test_last_projection_does_not_differentiate_frozen_prefix(frozen_paths: bool) -> None:
    cfg = _dense_config()
    sites = glu_site_specs(cfg, (SiteC("layers.0.mlp.down_proj", 5),))
    components = transformer_flops(
        cfg,
        GLU_ANATOMY,
        sites,
        batch_size=2,
        sequence_length=3,
        gradients="components",
        include_frozen_paths=frozen_paths,
        capture_keys=frozenset(),
    ).flops
    sources = transformer_flops(
        cfg,
        GLU_ANATOMY,
        sites,
        batch_size=2,
        sequence_length=3,
        gradients="sources",
        include_frozen_paths=frozen_paths,
        capture_keys=frozenset(),
    ).flops
    v_product = 2 * 6 * 12 * 5
    u_product = 2 * 6 * 5 * 8
    head_product = 2 * 6 * 8 * 20
    assert components.backward == v_product + 2 * u_product + head_product
    assert sources.backward == u_product + head_product
    assert components.forward == sources.forward


def test_query_components_do_not_differentiate_frozen_key_and_value_projections() -> None:
    cfg = _dense_config()
    sites = glu_site_specs(cfg, (SiteC("layers.0.self_attn.q_proj", 5),))
    cost = transformer_flops(
        cfg,
        GLU_ANATOMY,
        sites,
        batch_size=2,
        sequence_length=3,
        gradients="components",
        include_frozen_paths=False,
        capture_keys=frozenset(),
    ).flops
    v_product = 2 * 6 * 8 * 5
    u_product = 2 * 6 * 5 * 8
    attention_products = 2 * 2 * 2 * 6 * 4 * 2
    downstream_frozen_products = 2 * 6 * (8 * 8 + 3 * 8 * 12 + 8 * 20)
    assert cost.backward == (
        v_product + 2 * u_product + attention_products + downstream_frozen_products
    )


def test_unused_moe_experts_only_increase_router_work() -> None:
    cfg = tiny_qwen36_cfg()
    larger = replace(cfg, n_experts=cfg.n_experts * 2)
    costs = [
        qwen36_moe_flops(
            arch,
            (),
            batch_size=2,
            sequence_length=3,
            gradients="none",
            include_frozen_paths=False,
            capture_keys=frozenset(),
        ).flops
        for arch in (cfg, larger)
    ]
    assert costs[1].forward - costs[0].forward == (2 * 6 * cfg.n_embd * cfg.n_experts * cfg.n_layer)
    assert costs[0].backward == costs[1].backward == 0


def test_invalid_site_is_not_silently_ignored() -> None:
    cfg = _dense_config()
    sites = glu_site_specs(replace(cfg, n_layer=2), (SiteC("layers.1.mlp.down_proj", 5),))
    with pytest.raises(ValueError, match="outside the target architecture"):
        transformer_flops(
            cfg,
            GLU_ANATOMY,
            sites,
            batch_size=1,
            sequence_length=1,
            gradients="components",
            include_frozen_paths=False,
            capture_keys=frozenset(),
        )


def _contraction_flops(graph: core.ClosedJaxpr) -> int:
    total = 0
    for equation in graph.jaxpr.eqns:
        if equation.primitive.name == "dot_general":
            (contracted_axes, _), _ = equation.params["dimension_numbers"]
            lhs = equation.invars[0].aval
            output = equation.outvars[0].aval
            assert isinstance(lhs, ShapedArray)
            assert isinstance(output, ShapedArray)
            total += 2 * math.prod(output.shape) * math.prod(lhs.shape[a] for a in contracted_axes)
    return total


@pytest.mark.parametrize("gradients", ["components", "sources"])
@pytest.mark.parametrize("input_needs_gradient", [False, True])
@pytest.mark.parametrize("frozen_blend", ["absent", "delta", "routing"])
def test_site_rule_matches_production_autodiff(
    gradients: GradientTarget,
    input_needs_gradient: bool,
    frozen_blend: Literal["absent", "delta", "routing"],
) -> None:
    arrays = tuple(
        jax.ShapeDtypeStruct(shape, jnp.float32)
        for shape in ((2, 3, 7), (7, 5), (5, 11), (11, 7), (2, 3, 5), (2, 3))
    )

    def objective(
        x: jax.Array,
        v: jax.Array,
        u: jax.Array,
        weight: jax.Array,
        mask: jax.Array,
        delta: jax.Array,
    ) -> jax.Array:
        match frozen_blend:
            case "absent":
                delta_mask, route = None, None
            case "delta":
                delta_mask, route = delta, None
            case "routing":
                delta_mask, route = None, delta > 0.5
        return site_forward(
            x, SiteWeights(weight, v, u, None, None), mask, delta_mask, route
        ).output.sum()

    match gradients:
        case "components":
            differentiated_arguments = (1, 2, 4, 5)
        case "sources":
            differentiated_arguments = (4, 5)
        case "none":
            raise AssertionError("This test exercises the two training derivatives")
    if input_needs_gradient:
        differentiated_arguments = (0, *differentiated_arguments)
    forward = _contraction_flops(jax.make_jaxpr(objective)(*arrays))
    differentiated = jax.value_and_grad(objective, argnums=differentiated_arguments)
    forward_and_backward = _contraction_flops(jax.make_jaxpr(differentiated)(*arrays))

    site = SiteSpec("projection", DenseFactorization(d_in=7, d_out=11, C=5), "projection")
    actual = linear_flops(
        site.name,
        2 * 3,
        7,
        11,
        site,
        gradients=gradients,
        input_changed=input_needs_gradient,
        include_frozen_paths=frozen_blend != "absent",
        output_gradient=True,
        auxiliary_gradient=False,
        component_key=site.name,
    ).flops
    assert actual.forward == forward
    assert actual.backward == forward_and_backward - forward


@pytest.mark.parametrize("sequence_length", [1, 2])
def test_initial_delta_state_has_no_decay_gradient(sequence_length: int) -> None:
    query = jnp.ones((1, sequence_length, 1, 2))
    key = jnp.full_like(query, 0.5)
    value = jnp.ones((1, sequence_length, 1, 3))
    decay = jnp.full((1, sequence_length, 1), -0.5)
    strength = jnp.full_like(decay, 0.5)

    def objective(g: jax.Array) -> jax.Array:
        return gated_delta_rule(query, key, value, g, strength, "sequential").sum()

    gradient = jax.grad(objective)(decay)
    assert jnp.all(gradient[:, 0] == 0)
    assert bool(jnp.any(gradient != 0)) == (sequence_length > 1)

    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(
        cfg,
        tuple(
            SiteC(site_name(layer, "gdn_a"), 3)
            for layer in range(cfg.n_layer)
            if not layer_is_full_attention(cfg, layer)
        ),
    )
    cost = qwen36_moe_flops(
        cfg,
        sites,
        batch_size=1,
        sequence_length=sequence_length,
        gradients="sources",
        include_frozen_paths=True,
        capture_keys=frozenset(),
    ).flops
    assert (cost.backward > 0) == (sequence_length > 1)


@pytest.mark.parametrize("frozen_paths", [False, True])
def test_shared_forward_reuses_clean_prefix_and_first_frozen_branch(frozen_paths: bool) -> None:
    cfg = _dense_config()
    sites = glu_site_specs(cfg, (SiteC("layers.0.mlp.down_proj", 5),))
    clean = transformer_flops(
        cfg,
        GLU_ANATOMY,
        (),
        batch_size=2,
        sequence_length=3,
        gradients="none",
        include_frozen_paths=False,
        capture_keys=frozenset(),
    ).flops
    shared = transformer_flops(
        cfg,
        GLU_ANATOMY,
        sites,
        batch_size=2,
        sequence_length=3,
        include_frozen_paths=frozen_paths,
        capture_keys=frozenset(),
        gradients="none",
    ).shared_forward
    head = 2 * 6 * 8 * 20
    down = 2 * 6 * 12 * 8
    assert shared == clean.forward - head - (0 if frozen_paths else down)


@pytest.mark.parametrize("frozen_paths", [False, True])
def test_shared_forward_includes_parallel_frozen_attention_projections(frozen_paths: bool) -> None:
    cfg = _dense_config()
    sites = glu_site_specs(cfg, (SiteC("layers.0.self_attn.q_proj", 5),))
    shared = transformer_flops(
        cfg,
        GLU_ANATOMY,
        sites,
        batch_size=2,
        sequence_length=3,
        include_frozen_paths=frozen_paths,
        capture_keys=frozenset(),
        gradients="none",
    ).shared_forward
    key_and_value = 2 * 6 * 8 * 4 * 2
    frozen_query = 2 * 6 * 8 * 8
    assert shared == key_and_value + (frozen_query if frozen_paths else 0)


def test_qwen_pinned_router_only_needs_selected_logits_without_router_auxiliary() -> None:
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(cfg, (SiteC(site_name(0, "gdn_q"), 3),))
    full_capture_keys = frozenset(f"router_logits.{layer}" for layer in range(cfg.n_layer))
    costs = [
        qwen36_moe_flops(
            cfg,
            sites,
            batch_size=2,
            sequence_length=3,
            gradients="sources",
            include_frozen_paths=True,
            capture_keys=captures,
        ).flops
        for captures in (frozenset(), full_capture_keys)
    ]
    extra_router = 2 * 6 * cfg.n_embd * (cfg.n_experts - cfg.n_experts_per_token) * cfg.n_layer
    assert costs[1].forward - costs[0].forward == extra_router
    assert costs[1].backward - costs[0].backward == extra_router


def test_selected_router_softmax_has_identical_values_and_gradients() -> None:
    logits = jnp.array([0.7, -0.5, 1.1, 0.2])
    indices = jnp.array([2, 0])
    cotangent = jnp.array([0.3, -0.9])

    def full(logits: jax.Array) -> jax.Array:
        selected = jax.nn.softmax(logits)[indices]
        return selected / selected.sum()

    def selected(logits: jax.Array) -> jax.Array:
        return jax.nn.softmax(logits[indices])

    assert jnp.allclose(full(logits), selected(logits))
    full_gradient = jax.grad(lambda x: full(x) @ cotangent)(logits)
    selected_gradient = jax.grad(lambda x: selected(x) @ cotangent)(logits)
    assert jnp.allclose(full_gradient, selected_gradient, atol=1e-7)


def test_qwen_without_decomposition_reuses_entire_clean_forward() -> None:
    cfg = tiny_qwen36_cfg()
    cost = qwen36_moe_flops(
        cfg,
        (),
        gradients="none",
        batch_size=2,
        sequence_length=3,
        include_frozen_paths=False,
        capture_keys=frozenset(),
    ).flops
    shared = qwen36_moe_flops(
        cfg,
        (),
        batch_size=2,
        sequence_length=3,
        include_frozen_paths=False,
        capture_keys=frozenset(),
        gradients="none",
    ).shared_forward
    assert shared == cost.forward


@pytest.mark.parametrize("sequence_length", [1, 2, 5])
def test_delta_recurrence_skips_zero_initial_state_prediction(sequence_length: int) -> None:
    cfg = tiny_qwen36_cfg()
    cost = _delta_rule_flops(
        cfg,
        2,
        sequence_length,
        gradients="none",
        query_changed=False,
        key_changed=False,
        value_changed=False,
        decay_changed=False,
        strength_changed=False,
        auxiliary_gradient=False,
    ).flops
    product = (
        2 * 2 * cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim
    )
    assert cost.forward == (3 * sequence_length - 1) * product
    assert cost.backward == 0


def test_causal_convolution_does_not_count_padding_taps() -> None:
    cfg = tiny_qwen36_cfg()
    costs = [
        qwen36_moe_flops(
            replace(cfg, linear_conv_kernel_dim=width),
            (),
            batch_size=2,
            sequence_length=2,
            gradients="none",
            include_frozen_paths=False,
            capture_keys=frozenset(),
        ).flops
        for width in (2, 4)
    ]
    assert costs[0] == costs[1]


@pytest.mark.parametrize("capture", ["resid.1", "mlp_in.0", "layers.0.mlp.down_proj.out"])
def test_second_source_backward_reuses_head_and_stops_at_source(
    capture: Literal["resid.1", "mlp_in.0", "layers.0.mlp.down_proj.out"],
) -> None:
    cfg = _dense_config()
    sites = glu_site_specs(cfg, (SiteC("layers.0.mlp.down_proj", 5),))
    extra = transformer_flops(
        cfg,
        GLU_ANATOMY,
        sites,
        batch_size=2,
        sequence_length=3,
        include_frozen_paths=True,
        capture_keys=frozenset({capture}),
        gradients="sources",
    ).additional_backward
    match capture:
        case "resid.1" | "layers.0.mlp.down_proj.out":
            expected = 2 * 6 * 5 * 8
        case "mlp_in.0":
            expected = 0
    assert extra == expected


def test_second_source_backward_shares_parallel_branch_cotangents() -> None:
    cfg = _dense_config()
    sites = glu_site_specs(
        cfg,
        (
            SiteC("layers.0.self_attn.q_proj", 5),
            SiteC("layers.0.self_attn.v_proj", 7),
        ),
    )
    extra = transformer_flops(
        cfg,
        GLU_ANATOMY,
        sites,
        batch_size=2,
        sequence_length=3,
        include_frozen_paths=True,
        capture_keys=frozenset({"layers.0.self_attn.q_proj.out"}),
        gradients="sources",
    ).additional_backward
    assert extra == 2 * 6 * 5 * 8


def test_second_source_backward_needs_no_extra_work_without_auxiliaries() -> None:
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(cfg, (SiteC(site_name(0, "gdn_q"), 3),))
    assert (
        qwen36_moe_flops(
            cfg,
            sites,
            batch_size=2,
            sequence_length=3,
            include_frozen_paths=True,
            capture_keys=frozenset(),
            gradients="sources",
        ).additional_backward
        == 0
    )


def test_decay_auxiliary_adds_backward_absent_from_single_token_output() -> None:
    cfg = tiny_qwen36_cfg()
    name = site_name(0, "gdn_a")
    sites = qwen36_moe_site_specs(cfg, (SiteC(name, 3),))
    costs = [
        qwen36_moe_flops(
            cfg,
            sites,
            batch_size=2,
            sequence_length=1,
            gradients="sources",
            include_frozen_paths=True,
            capture_keys=captures,
        ).flops
        for captures in (frozenset(), frozenset({f"{name}.out"}))
    ]
    assert costs[0].backward == 0
    assert costs[1].backward == 2 * 2 * 3 * cfg.linear_num_value_heads


@pytest.mark.parametrize("sequence_length", [1, 2, 3])
@pytest.mark.parametrize("differentiated", [(0,), (1,), (2,), (3,), (4,), (0, 1, 2, 3, 4)])
def test_delta_reverse_dependencies_match_unrolled_autodiff(
    sequence_length: int, differentiated: tuple[int, ...]
) -> None:
    cfg = replace(
        tiny_qwen36_cfg(),
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=2,
        linear_value_head_dim=3,
    )
    shapes = (
        (1, sequence_length, 1, 2),
        (1, sequence_length, 1, 2),
        (1, sequence_length, 1, 3),
        (1, sequence_length, 1),
        (1, sequence_length, 1),
    )
    arrays = tuple(jax.ShapeDtypeStruct(shape, jnp.float32) for shape in shapes)

    def objective(
        q: jax.Array, k: jax.Array, v: jax.Array, decay: jax.Array, strength: jax.Array
    ) -> jax.Array:
        state = jnp.zeros((1, 1, 2, 3))
        loss = jnp.array(0.0)
        for position in range(sequence_length):
            if position > 0:
                state = state * jnp.exp(decay[:, position, :, None, None])
                prediction = jnp.einsum("bhkv,bhk->bhv", state, k[:, position])
                correction = v[:, position] - prediction
            else:
                correction = v[:, position]
            correction = correction * strength[:, position, :, None]
            state = state + jnp.einsum("bhk,bhv->bhkv", k[:, position], correction)
            loss = loss + jnp.einsum("bhkv,bhk->bhv", state, q[:, position]).sum()
        return loss

    analytical = _delta_rule_flops(
        cfg,
        1,
        sequence_length,
        gradients="sources",
        query_changed=0 in differentiated,
        key_changed=1 in differentiated,
        value_changed=2 in differentiated,
        decay_changed=3 in differentiated,
        strength_changed=4 in differentiated,
        auxiliary_gradient=False,
    ).flops
    forward = _contraction_flops(jax.make_jaxpr(objective)(*arrays))
    derivative = jax.value_and_grad(objective, argnums=differentiated)
    combined = _contraction_flops(jax.make_jaxpr(derivative)(*arrays))
    assert analytical.forward == forward
    assert analytical.backward == combined - forward


def test_component_input_projection_and_weight_adjoint_are_reusable() -> None:
    cfg = _dense_config()
    name = "layers.0.mlp.down_proj"
    sites = glu_site_specs(cfg, (SiteC(name, 5),))
    reusable = transformer_flops(
        cfg,
        GLU_ANATOMY,
        sites,
        batch_size=2,
        sequence_length=3,
        include_frozen_paths=True,
        capture_keys=frozenset(),
        gradients="components",
    ).reusable_components
    assert set(reusable) == {name}
    assert reusable[name].forward == reusable[name].backward == 2 * 6 * 12 * 5


def test_first_decay_projection_has_distinct_reusable_identity() -> None:
    cfg = tiny_qwen36_cfg()
    name = site_name(0, "gdn_a")
    sites = qwen36_moe_site_specs(cfg, (SiteC(name, 3),))
    costs = [
        qwen36_moe_flops(
            cfg,
            sites,
            batch_size=2,
            sequence_length=3,
            include_frozen_paths=True,
            capture_keys=captures,
            gradients="components",
        ).reusable_components
        for captures in (frozenset(), frozenset({f"{name}.out"}))
    ]
    tail = f"{name}/after_first_token"
    first = f"{name}/first_token"
    assert costs[0][tail] == costs[1][tail]
    assert first not in costs[0]
    assert costs[1][first].forward == costs[1][first].backward == 2 * 2 * cfg.n_embd * 3


def test_router_auxiliary_reuses_all_later_layer_source_cotangents() -> None:
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(cfg, (SiteC(site_name(0, "gdn_q"), 3),))
    costs = [
        qwen36_moe_flops(
            arch,
            sites,
            batch_size=2,
            sequence_length=3,
            include_frozen_paths=True,
            capture_keys=frozenset({"router_logits.0"}),
            gradients="sources",
        ).additional_backward
        for arch in (cfg, replace(cfg, n_layer=cfg.n_layer * 2))
    ]
    assert costs[0] == costs[1]
    assert costs[0] > 0
