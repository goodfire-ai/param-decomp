"""Canary that the jaxtyping+beartype runtime type-checker is actually firing.

jaxtyping+beartype silently no-ops if the `@jaxtyped(typechecker=beartype)` decorator is
misconfigured or a dependency bump breaks it — checks just stop running, with no error.
This locks in that the core loss contract still rejects bad dtype/shape/rank at runtime.
"""

import jax.numpy as jnp
import pytest
from jax import Array
from jaxtyping import TypeCheckError

from param_decomp.losses import kl_per_position


def test_valid_input_accepted() -> None:
    out = kl_per_position(jnp.zeros((2, 3, 5)), jnp.zeros((2, 3, 5)))
    assert out.shape == ()


@pytest.mark.parametrize(
    "masked, clean",
    [
        (jnp.zeros((2, 3, 5), jnp.int32), jnp.zeros((2, 3, 5), jnp.int32)),  # dtype: not Float
        (jnp.zeros((2, 3, 5)), jnp.zeros((2, 3, 7))),  # shape: vocab bound 5 != 7 across args
        (jnp.zeros(()), jnp.zeros(())),  # rank: no logit axis
    ],
)
def test_violation_raises(masked: Array, clean: Array) -> None:
    with pytest.raises(TypeCheckError):
        kl_per_position(masked, clean)
