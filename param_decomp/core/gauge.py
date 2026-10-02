"""A component's two activations.

Component `c` adds `(x·V_c) U_c` to its site's output. Rescaling `U_c → a U_c` and
`V_c → V_c / a` leaves every contribution unchanged, so the **gauge-variant activation**
`x·V_c` depends on how `U` and `V` happen to be scaled, while the **gauge-invariant
activation** `(x·V_c)·‖U_c‖` is the size of the contribution itself."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Float32


def u_norms_of(U: Float[Array, "... C d_out"]) -> Float32[Array, "... C"]:
    """`‖U_c‖` per component: row norms of `U`, accumulated in float32 without a float32
    copy of `U`."""
    sharding = jax.typeof(U).sharding
    return jnp.sqrt(
        jnp.einsum(
            "...cd,...cd->...c",
            U,
            U,
            preferred_element_type=jnp.float32,
            out_sharding=sharding.update(spec=P(*sharding.spec[:-1])),
        )
    )


def gauge_invariant_activation(
    gauge_variant_activation: Float[Array, "*shape"], u_norm: Float32[Array, "*broadcast"]
) -> Float32[Array, "*shape"]:
    """`(x·V_c)·‖U_c‖`, in float32; `u_norm` broadcasts against the activations."""
    return gauge_variant_activation.astype(jnp.float32) * u_norm


def gauge_variant_activation(
    gauge_invariant_activation: Float[Array, "*shape"], u_norm: Float32[Array, "*broadcast"]
) -> Float32[Array, "*shape"]:
    """`x·V_c` from `(x·V_c)·‖U_c‖`, in float32. A component with `U_c = 0` contributes
    nothing at any gauge-variant activation, so it has no inverse and fails."""
    u_norm = eqx.error_if(u_norm, jnp.any(u_norm == 0), "a component with U_c = 0 has no x·V_c")
    return gauge_invariant_activation.astype(jnp.float32) / u_norm
