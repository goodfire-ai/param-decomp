"""The factored output's numeric consumers enforce its array relationships."""

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from param_decomp.targets.lm_output import (
    MaterializedOutputEdge,
    OutputEdge,
    StreamedLinearOutput,
    StreamedOutputEdge,
    linear_output,
)
from param_decomp.targets.losses import (
    _accumulator_like,
    streamed_position_ce,
    streamed_position_kl,
)


@pytest.mark.parametrize("edge", [MaterializedOutputEdge(), StreamedOutputEdge(2)])
def test_output_edge_requires_placed_heads_and_rejects_invalid_operands(edge: OutputEdge):
    activations = jnp.ones((2, 3, 4), jnp.bfloat16)
    host_head = np.ones((8, 4), dtype=np.float32)
    head = jax.device_put(host_head, activations.sharding)
    output = linear_output(activations, head, edge)
    if isinstance(output, StreamedLinearOutput):
        assert output.head is head
    else:
        assert output.shape == (2, 3, 8)
    with pytest.raises(TypeCheckError):
        linear_output(activations, host_head, edge)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeCheckError):
        linear_output(activations, jnp.ones((8, 5)), edge)
    with pytest.raises(TypeCheckError):
        linear_output(activations.astype(jnp.int32), head, edge)
    with pytest.raises(TypeCheckError):
        linear_output(activations, jnp.ones((8, 4), dtype=jnp.int32), edge)


@pytest.mark.parametrize("edge", [MaterializedOutputEdge(), StreamedOutputEdge(2)])
@pytest.mark.parametrize("shape", [(4,), (3, 4), (1, 2, 3, 4)])
@pytest.mark.parametrize("compiled", [False, True])
def test_both_lm_output_edges_require_batch_and_sequence_axes(
    edge: OutputEdge, shape: tuple[int, ...], compiled: bool
):
    head = jnp.ones((8, 4))

    def project(activations: jax.Array):
        return linear_output(activations, head, edge)

    operation = jax.jit(project) if compiled else project
    with pytest.raises(TypeCheckError):
        operation(jnp.ones(shape))


def test_streamed_consumers_reject_invalid_shapes_and_integer_logits():
    labels = jnp.zeros((2, 3), jnp.int32)
    with pytest.raises(TypeCheckError):
        streamed_position_ce(StreamedLinearOutput(jnp.ones((2, 3, 4)), jnp.ones((8, 5)), 2), labels)
    with pytest.raises(TypeCheckError):
        streamed_position_ce(
            StreamedLinearOutput(jnp.ones((2, 3, 4), jnp.int32), jnp.ones((8, 4)), 2), labels
        )
    with pytest.raises(TypeCheckError):
        streamed_position_ce(
            StreamedLinearOutput(jnp.ones((2, 3, 4)), jnp.ones((8, 4)), 2),
            labels.astype(jnp.float32),
        )


@pytest.mark.parametrize("leading", [(), (3,), (2, 3), (2, 1, 3)])
def test_streamed_losses_preserve_all_leading_axes_in_float32(leading: tuple[int, ...]):
    output = jax.eval_shape(
        lambda activations, head: StreamedLinearOutput(activations, head, 2),
        jax.ShapeDtypeStruct((*leading, 4), jnp.bfloat16),
        jax.ShapeDtypeStruct((8, 4), jnp.bfloat16),
    )
    labels = jax.ShapeDtypeStruct(leading, jnp.int32)
    assert jax.eval_shape(streamed_position_ce, output, labels) == jax.ShapeDtypeStruct(
        leading, jnp.float32
    )
    assert jax.eval_shape(streamed_position_kl, output, output) == jax.ShapeDtypeStruct(
        leading, jnp.float32
    )

    assert jax.eval_shape(_accumulator_like, output.activations) == jax.ShapeDtypeStruct(
        leading, jnp.float32
    )


@pytest.mark.parametrize("metric", ["ce", "kl"])
def test_streamed_consumers_remain_composable_with_vmap_and_grad(metric: Literal["ce", "kl"]):
    head = jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 32
    activations = jnp.ones((2, 3, 4), jnp.bfloat16)
    labels = jnp.zeros((2, 3), jnp.int32)

    def compare(x: jax.Array, labels: jax.Array) -> jax.Array:
        output = StreamedLinearOutput(x, head, 2)
        match metric:
            case "ce":
                return streamed_position_ce(output, labels).sum()
            case "kl":
                return streamed_position_kl(output, StreamedLinearOutput(x * 0.5, head, 2)).sum()

    expected = jax.grad(compare)(activations, labels)
    actual = jax.jit(jax.vmap(jax.grad(compare)))(activations, labels)
    np.testing.assert_allclose(actual.astype(jnp.float32), expected.astype(jnp.float32))
