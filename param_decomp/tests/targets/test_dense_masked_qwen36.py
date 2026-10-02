"""Routed Qwen expert execution agrees with the dense masked implementation."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.core.components import ComponentStacks, SelectedCI, SiteCI, init_component_stacks
from param_decomp.core.model import MaterializedMasking
from param_decomp.lm.batch import LMBatch
from param_decomp.targets.qwen36_moe import (
    Qwen36MoeDecomposedModel,
    full_site_cs,
    qwen36_moe_site_specs,
)
from param_decomp.targets.testing import (
    TINY_QWEN36_CS,
    materialized_logits,
    random_mask_values,
    site_masks,
    tiny_qwen36_cfg,
    tiny_qwen36_decomposed_model,
)
from param_decomp.targets.transformer_taps import mlp_input_tap_key, site_output_tap_key


def _models() -> tuple[
    Qwen36MoeDecomposedModel, Qwen36MoeDecomposedModel, ComponentStacks, jax.Array
]:
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(
        cfg,
        full_site_cs(
            cfg, {kind: 16 if kind.startswith("experts_") else 8 for kind in TINY_QWEN36_CS}
        ),
    )
    routed = tiny_qwen36_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    dense = replace(routed, expert_implementation="dense_masked")
    components = init_component_stacks(sites, jax.random.PRNGKey(1))
    tokens = jax.random.randint(jax.random.PRNGKey(2), (2, 4), 0, cfg.vocab_size)
    return dense, routed, components, tokens


def _assert_close(got: object, expected: object) -> None:
    for actual, reference in zip(jax.tree.leaves(got), jax.tree.leaves(expected), strict=True):
        actual_np, reference_np = np.asarray(actual), np.asarray(reference)
        np.testing.assert_allclose(actual_np, reference_np, rtol=2e-3, atol=2e-5)


def test_routed_frozen_forward_and_captures_match_dense():
    dense, routed, _, tokens = _models()
    keys = frozenset(
        [mlp_input_tap_key(i) for i in range(dense.cfg.n_layer)]
        + [site_output_tap_key(site) for site in dense.site_names]
    )
    expected = jax.jit(lambda model: model.clean_forward(LMBatch(tokens), keys, placement=None))(
        dense
    )
    actual = jax.jit(lambda model: model.clean_forward(LMBatch(tokens), keys, placement=None))(
        routed
    )
    _assert_close(actual, expected)


@pytest.mark.parametrize("delta_enabled", [False, True])
@pytest.mark.parametrize("route_enabled", [False, True])
def test_routed_masked_forward_and_gradients_match_dense(delta_enabled: bool, route_enabled: bool):
    dense, routed, components, tokens = _models()
    conditioning = dense.clean_forward(LMBatch(tokens), placement=None).conditioning
    values = random_mask_values(dense, tokens.shape, jax.random.PRNGKey(3))
    masks = site_masks(dense, conditioning.selection, values)
    deltas = {site: jnp.full((2, 1), 0.3) for site in dense.site_names} if delta_enabled else None
    routes = (
        {site: jnp.array([[True], [False]]) for site in dense.site_names} if route_enabled else None
    )

    def loss(model: Qwen36MoeDecomposedModel, vu: ComponentStacks, coefficients: dict[str, SiteCI]):
        output = model.masked_forward(
            model.prepare_compute_weights(vu, None),
            conditioning,
            masking=model.prepare_masking(
                MaterializedMasking(component_masks=coefficients, weight_delta_masks=deltas)
            ),
            routes=routes,
            placement=None,
            remat=False,
        ).output
        logits = materialized_logits(output)
        return jnp.mean(jnp.cos(logits)), logits

    evaluate = jax.jit(jax.value_and_grad(loss, argnums=(1, 2), has_aux=True, allow_int=True))
    expected = evaluate(dense, components, masks)
    actual = evaluate(routed, components, masks)
    # Integer routing leaves have float0 cotangents, so compare numerical leaves only.
    leaves = zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True)
    for got, want in leaves:
        if got.dtype != jax.dtypes.float0:
            np.testing.assert_allclose(got, want, rtol=3e-3, atol=3e-5)


def test_dense_component_capture_preserves_selected_emission():
    dense, routed, components, tokens = _models()

    def capture(model: Qwen36MoeDecomposedModel):
        return model.component_activation_forward(
            model.prepare_compute_weights(components, None),
            LMBatch(tokens),
            sites=model.site_names,
            capture_keys=frozenset(),
            placement=None,
        )

    expected_forward, expected = jax.jit(capture)(dense)
    actual_forward, actual = jax.jit(capture)(routed)
    _assert_close(actual_forward, expected_forward)
    for site in dense.site_names:
        assert type(actual[site]) is type(expected[site])
        if ".experts." in site:
            assert isinstance(expected[site], SelectedCI)
        _assert_close(actual[site], expected[site])


@pytest.mark.multidevice
@pytest.mark.parametrize("tp", [1, 2])
def test_placed_dense_target_matches_replicated_reference(tp: int):
    from jax.sharding import AxisType, Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from param_decomp.core.init_placed import init_component_stacks_placed
    from param_decomp.core.placement import from_config
    from param_decomp.core.sharding import place_target

    if len(jax.devices()) < 2 * tp:
        pytest.skip("requires enough simulated devices for data2 and the expert partition")
    dense, _, components, tokens = _models()
    mesh = Mesh(
        np.asarray(jax.devices()[: 2 * tp]).reshape(2, tp),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    rules = from_config("zero1-replicated-resident-moe", mesh, dense.sites)
    expected_forward, expected_acts = dense.component_activation_forward(
        dense.prepare_compute_weights(components, None),
        LMBatch(tokens),
        sites=dense.site_names,
        capture_keys=frozenset(),
        placement=None,
    )
    placed_model = place_target(dense, rules)
    placed_components = init_component_stacks_placed(dense.sites, jax.random.PRNGKey(1), rules)
    placed_batch = LMBatch(jax.device_put(tokens, NamedSharding(mesh, P("data", None))))
    values = random_mask_values(dense, tokens.shape, jax.random.PRNGKey(5))
    masks = site_masks(dense, expected_forward.conditioning.selection, values)
    with jax.set_mesh(mesh):
        prepared = placed_model.model.prepare_compute_weights(placed_components, rules)
        forward, activations = jax.jit(
            lambda weights, batch: placed_model.model.component_activation_forward(
                weights, batch, sites=dense.site_names, capture_keys=frozenset(), placement=rules
            )
        )(prepared, placed_batch)
        _assert_close(forward.output, expected_forward.output)
        for site in dense.site_names:
            actual, expected = activations[site], expected_acts[site]
            if isinstance(expected, SelectedCI):
                assert isinstance(actual, SelectedCI)
                assert jax.typeof(actual.values).sharding.spec == P("data", None, None)
            _assert_close(actual, expected)

        values_placed = jax.tree.map(
            lambda x: jax.device_put(x, NamedSharding(mesh, P("data", None, None))), values
        )
        mask_placed = site_masks(dense, forward.conditioning.selection, values_placed)

        def placed_loss(vu: ComponentStacks):
            out = placed_model.masked_forward(
                placed_model.model.prepare_compute_weights(vu, rules),
                forward.conditioning,
                masking=placed_model.model.prepare_masking(
                    MaterializedMasking(component_masks=mask_placed, weight_delta_masks=None)
                ),
                routes=None,
                remat=False,
            ).output
            return jnp.mean(jnp.cos(materialized_logits(out)))

        got_value, got_grad = jax.jit(jax.value_and_grad(placed_loss))(placed_components)

    def reference_loss(vu: ComponentStacks):
        out = dense.masked_forward(
            dense.prepare_compute_weights(vu, None),
            expected_forward.conditioning,
            masking=dense.prepare_masking(
                MaterializedMasking(component_masks=masks, weight_delta_masks=None)
            ),
            routes=None,
            placement=None,
            remat=False,
        ).output
        return jnp.mean(jnp.cos(materialized_logits(out)))

    expected_value, expected_grad = jax.jit(jax.value_and_grad(reference_loss))(components)
    _assert_close(got_value, expected_value)
    _assert_close(got_grad, expected_grad)


@pytest.mark.multidevice
@pytest.mark.parametrize("tp", [1, 2])
def test_dense_selected_expansion_preserves_expert_sharding_and_gradients(tp: int):
    from jax.sharding import AxisType, Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from param_decomp.targets.qwen36_moe import _dense_selected_values

    if len(jax.devices()) < 4:
        pytest.skip("requires four devices")
    indices = jnp.asarray([[[0, 2], [1, 2], [2, 0]]] * 4)
    values = jax.random.normal(jax.random.key(63), (4, 3, 4))
    cotangent = jax.random.normal(jax.random.key(64), (4, 3, 4, 2))

    def reference(v: jax.Array) -> jax.Array:
        blocks = v.reshape(4, 3, 2, 2)
        return (blocks[..., :, None, :] * jax.nn.one_hot(indices, 4)[..., None]).sum(axis=-3)

    expected = reference(values)
    expected_grad = jax.grad(lambda v: (reference(v) * cotangent).sum())(values)
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(4 // tp, tp),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    with jax.set_mesh(mesh):
        selected = SelectedCI(
            jax.device_put(values, NamedSharding(mesh, P("data", None, None))),
            jax.device_put(indices, NamedSharding(mesh, P("data", None, None))),
            4,
        )
        weights = jax.device_put(jnp.zeros((4, 5, 2)), NamedSharding(mesh, P("tp", None, None)))
        cotangent = jax.device_put(cotangent, NamedSharding(mesh, P("data", None, "tp", None)))
        actual = jax.jit(_dense_selected_values)(selected, weights)
        actual_grad = jax.jit(
            jax.grad(
                lambda v: (
                    _dense_selected_values(SelectedCI(v, selected.block_indices, 4), weights)
                    * cotangent
                ).sum()
            )
        )(selected.values)
        assert actual.sharding.spec == P("data", None, "tp", None)
        assert actual_grad.sharding == selected.values.sharding
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(actual_grad, expected_grad, atol=1e-6)
