"""Merged sources preserve selected semantics, RNG, and adversarial gradient gating."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from param_decomp.core.adversary import BlockedSourceComponents, SiteSource
from param_decomp.core.components import SelectedCI, site_ci_values
from param_decomp.core.masking import mixed_persistent_stochastic_masking


@pytest.mark.parametrize("adv_fraction", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("placed", [False, pytest.param(True, marks=pytest.mark.multidevice)])
def test_merged_selected_masks_and_gradients_match_token_oracle(
    adv_fraction: float, placed: bool
) -> None:
    if placed and len(jax.devices()) < 4:
        pytest.skip("requires four devices")
    b, t, k, e, c = 8, 3, 2, 4, 2
    key = jax.random.key(17)
    ids = jnp.array([[[0, 1], [2, 0], [1, 3]]] * b, jnp.int32)
    values = jnp.linspace(0.1, 0.9, b * t * k * c).reshape(b, t, k * c)
    table = jnp.linspace(0.2, 0.8, b * e * c).reshape(b, 1, e, c)
    delta = jnp.linspace(0.1, 0.9, b).reshape(b, 1)
    routes = jnp.zeros((b, t), jnp.bool_)
    assignment_key, uniform_key = jax.random.split(key)
    component_key, delta_key = jax.random.split(uniform_key)
    assigned = jax.random.bernoulli(assignment_key, adv_fraction, (b, 1))
    if adv_fraction == 0.5:
        assert bool(assigned.any()) and not bool(assigned.all())
    noise = jax.random.uniform(jax.random.fold_in(component_key, 0), values.shape)
    delta_noise = jax.random.uniform(jax.random.fold_in(delta_key, 0), (b, t))

    def oracle(ci: Array, source: Array, delta_source: Array) -> Array:
        selected = jnp.einsum("btke,btec->btkc", jax.nn.one_hot(ids, e), source).reshape(
            b, t, k * c
        )
        mask = ci + (1 - ci) * jnp.where(assigned[..., None], selected, noise)
        delta_mask = jnp.where(assigned, delta_source, delta_noise)
        return jnp.sin(mask).sum() + jnp.sin(delta_mask).sum()

    expected, expected_grads = jax.jit(jax.value_and_grad(oracle, argnums=(0, 1, 2)))(
        values, table, delta
    )

    def execute(vals: Array, sources: Array, deltas: Array) -> tuple[Array, Array]:
        ci = SelectedCI(vals, ids_placed if placed else ids, e)
        masking, mixed_routes = mixed_persistent_stochastic_masking(
            key,
            {"site": ci},
            {"site": SiteSource(BlockedSourceComponents(sources), deltas)},
            (b, t),
            jnp.asarray(adv_fraction),
            {"site": routes},
        )
        pair = masking.ingredients["site"]
        assert isinstance(pair.ci, SelectedCI)
        assert len(jax.tree.leaves(pair)) == len(jax.tree.leaves(pair.ci)) + 2
        assert pair.delta.shape == (b, t)
        assert mixed_routes is not None
        return (
            jnp.sin(site_ci_values(pair.compose())).sum() + jnp.sin(pair.delta).sum(),
            mixed_routes["site"],
        )

    differentiated = jax.jit(jax.value_and_grad(execute, argnums=(0, 1, 2), has_aux=True))
    if placed:
        mesh = Mesh(
            np.asarray(jax.devices()[:4]).reshape(2, 2),
            ("data", "tp"),
            axis_types=(AxisType.Explicit,) * 2,
        )
        with jax.set_mesh(mesh):
            ids_placed = jax.device_put(ids, NamedSharding(mesh, P("data", None, None)))
            (actual, actual_routes), grads = differentiated(
                jax.device_put(values, NamedSharding(mesh, P("data", None, None))),
                jax.device_put(table, NamedSharding(mesh, P("data", None, "tp", None))),
                jax.device_put(delta, NamedSharding(mesh, P("data", None))),
            )
    else:
        (actual, actual_routes), grads = differentiated(values, table, delta)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    np.testing.assert_array_equal(actual_routes, jnp.broadcast_to(assigned, (b, t)))
    for actual_grad, expected_grad in zip(grads, expected_grads, strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, atol=1e-6, rtol=2e-6)
    np.testing.assert_array_equal(np.asarray(grads[1])[~np.asarray(assigned[:, 0])], 0.0)
    np.testing.assert_array_equal(np.asarray(grads[2])[~np.asarray(assigned[:, 0])], 0.0)
