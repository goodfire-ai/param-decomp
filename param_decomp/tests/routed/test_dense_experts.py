"""Routed expert operations agree with the dense definition."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.routed.dense import (
    dense_combine_experts,
    dense_expert_matmul,
    dense_project_and_combine_experts,
    dense_select_experts,
    token_expert_matmul,
)
from param_decomp.routed.experts import (
    combine_jobs,
    gather_tokens,
    grouped_matmul,
    routed_jobs,
)


def test_routed_values_and_gradients_match_dense() -> None:
    ids = jnp.asarray([[0, 2], [2, 0], [1, 2]], dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(71), 3)
    x = jax.random.normal(keys[0], (3, 5))
    table = jax.random.normal(keys[1], (4, 5, 7))
    mixing = jax.nn.softmax(jax.random.normal(keys[2], (3, 2)), axis=-1)
    jobs = routed_jobs(ids, 4)

    def dense(x: jax.Array, table: jax.Array, mixing: jax.Array) -> jax.Array:
        return dense_combine_experts(token_expert_matmul(x, table), ids, mixing)

    def routed(x: jax.Array, table: jax.Array, mixing: jax.Array) -> jax.Array:
        values = grouped_matmul(gather_tokens(x, jobs), table, jobs.group_sizes, "ragged_dot")
        return combine_jobs(values, jobs, mixing)

    np.testing.assert_allclose(routed(x, table, mixing), dense(x, table, mixing), atol=1e-6)
    for actual, expected in zip(
        jax.grad(lambda *args: jnp.sin(routed(*args)).sum(), argnums=(0, 1, 2))(x, table, mixing),
        jax.grad(lambda *args: jnp.sin(dense(*args)).sum(), argnums=(0, 1, 2))(x, table, mixing),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.multidevice
@pytest.mark.parametrize("tp", [1, 2])
def test_dense_selected_boundary_preserves_token_order_and_gradients(tp: int) -> None:
    if len(jax.devices()) < 4:
        pytest.skip("requires four devices")
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(4 // tp, tp),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    ids = jnp.asarray([[[0, 2], [1, 2], [2, 0]]] * 4)
    x = jnp.arange(60.0).reshape(4, 3, 5) / 60
    weights = jnp.arange(40.0).reshape(4, 5, 2) / 40

    def reference(x: jax.Array, weights: jax.Array) -> jax.Array:
        table = jnp.einsum("btd,edh->bteh", x, weights)
        return jnp.take_along_axis(table, ids[..., None], axis=-2)

    expected = reference(x, weights)
    expected_grad = jax.grad(lambda x, w: jnp.sin(reference(x, w)).sum(), argnums=(0, 1))(
        x, weights
    )
    with jax.set_mesh(mesh):
        ids = jax.device_put(ids, NamedSharding(mesh, P("data", None, None)))
        x = jax.device_put(x, NamedSharding(mesh, P("data", None, None)))
        weights = jax.device_put(weights, NamedSharding(mesh, P("tp", None, None)))

        @jax.jit
        def selected_table(x: jax.Array, weights: jax.Array) -> jax.Array:
            return dense_select_experts(token_expert_matmul(x, weights), ids)

        actual = selected_table(x, weights)
        assert actual.sharding.spec == P("data", None, None, None)
        actual_grad = jax.jit(
            jax.grad(lambda x, w: jnp.sin(selected_table(x, w)).sum(), argnums=(0, 1))
        )(x, weights)
        for gradient, operand in zip(actual_grad, (x, weights), strict=True):
            assert gradient.sharding == operand.sharding
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    for actual_array, expected_array in zip(actual_grad, expected_grad, strict=True):
        np.testing.assert_allclose(actual_array, expected_array, atol=1e-6)


@pytest.mark.multidevice
@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_dense_combine_matches_fp32_dot_without_changing_expert_rounding(
    tp: int, dtype: jax.typing.DTypeLike
) -> None:
    if len(jax.devices()) < 4:
        pytest.skip("requires four devices")
    values = jax.random.normal(jax.random.key(81), (4, 3, 4, 7)).astype(dtype)
    mixing = jax.nn.softmax(jax.random.normal(jax.random.key(82), (4, 3, 2)), axis=-1)
    ids = jnp.asarray([[[0, 2], [1, 2], [2, 0]]] * 4, dtype=jnp.int32)

    def reference(v: jax.Array, w: jax.Array) -> jax.Array:
        routing = jnp.einsum("btke,btk->bte", jax.nn.one_hot(ids, 4), w)
        return jnp.einsum("bted,bte->btd", v, routing, preferred_element_type=jnp.float32).astype(
            v.dtype
        )

    expected = reference(values, mixing)
    expected_grad = jax.grad(
        lambda v, w: jnp.sin(reference(v, w).astype(jnp.float32)).sum(), argnums=(0, 1)
    )(values, mixing)
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(4 // tp, tp),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    with jax.set_mesh(mesh):
        v = jax.device_put(values, NamedSharding(mesh, P("data", None, "tp", None)))
        w = jax.device_put(mixing, NamedSharding(mesh, P("data", None, None)))
        indices = jax.device_put(ids, NamedSharding(mesh, P("data", None, None)))
        actual = jax.jit(dense_combine_experts)(v, indices, w)
        actual_grad = jax.jit(
            jax.grad(
                lambda v, w: jnp.sin(
                    dense_combine_experts(v, indices, w).astype(jnp.float32)
                ).sum(),
                argnums=(0, 1),
            )
        )(v, w)
        assert actual.sharding.spec == P("data", None, None)
        assert actual_grad[0].sharding == v.sharding
        assert actual_grad[1].sharding == w.sharding
    for got, want in zip((actual, *actual_grad), (expected, *expected_grad), strict=True):
        np.testing.assert_allclose(
            np.asarray(got, dtype=np.float32),
            np.asarray(want, dtype=np.float32),
            atol=2e-5,
            rtol=2e-5,
        )


@pytest.mark.multidevice
@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("positions", [17, 64, 67, 512])
def test_chunked_expert_projection_preserves_values_gradients_and_ownership(
    tp: int, dtype: jax.typing.DTypeLike, positions: int
) -> None:
    if len(jax.devices()) < 4:
        pytest.skip("requires four devices")
    hidden = jax.random.normal(jax.random.key(91), (4, positions, 4, 16)).astype(dtype)
    projection = (jax.random.normal(jax.random.key(92), (4, 16, 24)) / 4).astype(dtype)
    ids = jax.lax.top_k(jax.random.normal(jax.random.key(93), (4, positions, 4)), 2)[1]
    mixing = jax.nn.softmax(jax.random.normal(jax.random.key(94), ids.shape), axis=-1)
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(4 // tp, tp),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )

    def reference(h: jax.Array, p: jax.Array, w: jax.Array) -> jax.Array:
        return dense_combine_experts(dense_expert_matmul(h, p), ids, w)

    def chunked(h: jax.Array, p: jax.Array, w: jax.Array) -> jax.Array:
        return dense_project_and_combine_experts(h, p, ids, w)

    with jax.set_mesh(mesh):
        hidden = jax.device_put(hidden, NamedSharding(mesh, P("data", None, "tp", None)))
        projection = jax.device_put(projection, NamedSharding(mesh, P("tp", None, None)))
        ids = jax.device_put(ids, NamedSharding(mesh, P("data", None, None)))
        mixing = jax.device_put(mixing, NamedSharding(mesh, P("data", None, None)))
        expected = jax.jit(reference)(hidden, projection, mixing)
        actual = jax.jit(chunked)(hidden, projection, mixing)
        expected_grads = jax.jit(
            jax.grad(
                lambda h, p, w: jnp.sin(reference(h, p, w).astype(jnp.float32)).sum(),
                argnums=(0, 1, 2),
            )
        )(hidden, projection, mixing)
        actual_grads = jax.jit(
            jax.grad(
                lambda h, p, w: jnp.sin(chunked(h, p, w).astype(jnp.float32)).sum(),
                argnums=(0, 1, 2),
            )
        )(hidden, projection, mixing)
        assert actual.sharding == expected.sharding == NamedSharding(mesh, P("data", None, None))
        for grad, primal in zip(actual_grads, (hidden, projection, mixing), strict=True):
            assert grad.sharding == primal.sharding

    tolerance = 0.01 if dtype == jnp.bfloat16 else 2e-6
    for got, want in zip((actual, *actual_grads), (expected, *expected_grads), strict=True):
        reference_array = np.asarray(want, dtype=np.float32)
        difference = np.asarray(got, dtype=np.float32) - reference_array
        assert np.linalg.norm(difference) <= tolerance * np.linalg.norm(reference_array) + 1e-6
