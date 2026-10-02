"""CI clipping operations and their gradient rules."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


@jax.custom_vjp
def lower_leaky_hard_sigmoid(x: Array) -> Array:
    return jnp.clip(x, 0.0, 1.0)


def _lhs_f(x: Array) -> tuple[Array, Array]:
    return jnp.clip(x, 0.0, 1.0), x


def _lhs_b(x: Array, g: Array) -> tuple[Array]:
    leak = jnp.where(g < 0, 0.01 * g, 0.0)
    return (jnp.where(x <= 0, leak, jnp.where(x <= 1, g, 0.0)),)


lower_leaky_hard_sigmoid.defvjp(_lhs_f, _lhs_b)


@jax.custom_vjp
def symmetric_leaky_hard_sigmoid(x: Array) -> Array:
    """Clamp to [0, 1]; outside it, leak only the gradients whose descent step points
    back inside."""
    return jnp.clip(x, 0.0, 1.0)


def _symmetric_lhs_b(x: Array, g: Array) -> tuple[Array]:
    below = jnp.where(g < 0, 0.01 * g, 0.0)
    above = jnp.where(g > 0, 0.01 * g, 0.0)
    return (jnp.where(x <= 0, below, jnp.where(x >= 1, above, g)),)


symmetric_leaky_hard_sigmoid.defvjp(_lhs_f, _symmetric_lhs_b)


def upper_leaky_hard_sigmoid(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """`x>1 ? 1+alpha*(x-1) : clamp(x,0,1)` — ordinary autodiff of this expression
    (torch builds its backward the same way; only the lower squashing is a custom VJP)."""
    alpha = 0.01
    return jnp.where(x > 1, 1 + alpha * (x - 1), jnp.clip(x, 0.0, 1.0))
