"""Dense expert computation on token/expert axes; selection is a routing mask."""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int


def dense_expert_matmul(
    x: Float[Array, "... E d"], weights: Float[Array, "E d h"]
) -> Float[Array, "... E h"]:
    """Apply every expert matrix; a singleton input expert axis broadcasts to all experts."""
    sharding = jax.typeof(x).sharding
    input_spec = sharding.spec.partitions
    expert_axis = jax.typeof(weights).sharding.spec.partitions[0]
    return jnp.einsum(
        "...ed,edh->...eh",
        x,
        weights,
        out_sharding=None
        if sharding.mesh.empty
        else NamedSharding(sharding.mesh, P(*input_spec[:-2], expert_axis, None)),
    )


def dense_select_experts(
    values: Float[Array, "... E c"], indices: Int[Array, "... k"]
) -> Float[Array, "... k c"]:
    """Read selected experts into token order, replicated over the expert mesh axis."""
    sharding = jax.typeof(values).sharding
    return jnp.einsum(
        "...ec,...ke->...kc",
        values,
        jax.nn.one_hot(indices, values.shape[-2], dtype=values.dtype),
        precision=jax.lax.Precision.HIGHEST,
        out_sharding=None
        if sharding.mesh.empty
        else NamedSharding(sharding.mesh, P(*sharding.spec.partitions[:-2], None, None)),
    )


def token_expert_matmul(
    x: Float[Array, "... d"], weights: Float[Array, "E d h"]
) -> Float[Array, "... E h"]:
    return dense_expert_matmul(x[..., None, :], weights)


def dense_combine_experts(
    values: Float[Array, "... E d"],
    top_idx: Int[Array, "... k"],
    weights: Float[Array, "... k"],
) -> Float[Array, "... d"]:
    """Sum only selected experts with their fp32 routing weights."""
    sharding = jax.typeof(values).sharding
    spec = sharding.spec.partitions
    selected = jax.nn.one_hot(top_idx, values.shape[-2], dtype=jnp.float32)
    routing = jnp.einsum(
        "...ke,...k->...e",
        selected,
        weights.astype(jnp.float32),
        out_sharding=None
        if sharding.mesh.empty
        else NamedSharding(sharding.mesh, P(*spec[:-2], spec[-2])),
    )
    # Expose conversion, multiplication and reduction together for fusion.
    return jnp.sum(values.astype(jnp.float32) * routing[..., None], axis=-2).astype(values.dtype)


def dense_project_and_combine_experts(
    hidden: Float[Array, "b t E h"],
    projection: Float[Array, "E h d"],
    top_idx: Int[Array, "b t k"],
    mixing: Float[Array, "b t k"],
) -> Float[Array, "b t d"]:
    """Bound expert-output storage by mapping over 32-position sequence chunks."""

    @jax.checkpoint
    def project_and_combine(
        inputs: tuple[Float[Array, "b E h"], Int[Array, "b k"], Float[Array, "b k"]],
    ) -> Float[Array, "b d"]:
        values, indices, weights = inputs
        # Round each expert output before the fp32 routing reduction.
        projected = dense_expert_matmul(values, projection)
        return dense_combine_experts(projected, indices, weights)

    sequence_first = tuple(jnp.swapaxes(x, 0, 1) for x in (hidden, top_idx, mixing))
    combined = jax.lax.map(project_and_combine, sequence_first, batch_size=32)
    return jnp.swapaxes(combined, 0, 1)
