"""Grad-check for the custom VJP of `lower_leaky_hard_sigmoid` (SPEC S6, risk R-5).

The equivalence fixtures feed CI values in pre-computed, so the custom backward is
never exercised there. This pins it directly: the cotangent is a grad-sign gate (leak
`alpha*g` below zero only when `g<0`) plus `<=` tie-breaks at the clamp boundaries
(`ci_fn.py` `_lhs_b`). `x=0` falls to the lower branch, `x=1` to the middle branch.
"""

import jax
import jax.numpy as jnp
import pytest

from jax_single_pool.ci_fn import lower_leaky_hard_sigmoid

ALPHA = 0.01

# (x, incoming g, expected cotangent) covering {x<0, x in (0,1], x>1} x {g<0, g>0}
# plus the exact boundary points x=0 (lower branch wins) and x=1 (middle branch wins).
CASES = [
    (-2.0, -3.0, ALPHA * -3.0),  # x<0, g<0 -> alpha*g
    (-2.0, 3.0, 0.0),  # x<0, g>0 -> 0
    (0.5, -3.0, -3.0),  # x in (0,1], g<0 -> g
    (0.5, 3.0, 3.0),  # x in (0,1], g>0 -> g
    (4.0, -3.0, 0.0),  # x>1, g<0 -> 0
    (4.0, 3.0, 0.0),  # x>1, g>0 -> 0
    (0.0, -3.0, ALPHA * -3.0),  # boundary x=0, g<0: lower branch -> alpha*g
    (0.0, 3.0, 0.0),  # boundary x=0, g>0: lower branch -> 0
    (1.0, -3.0, -3.0),  # boundary x=1, g<0: middle branch -> g
    (1.0, 3.0, 3.0),  # boundary x=1, g>0: middle branch -> g
]


@pytest.mark.parametrize("x, g, expected", CASES)
def test_lower_leaky_hard_cotangent(x: float, g: float, expected: float):
    primal, vjp_fn = jax.vjp(lower_leaky_hard_sigmoid, jnp.float32(x))
    (cotangent,) = vjp_fn(jnp.float32(g))
    assert primal.dtype == jnp.float32 and cotangent.dtype == jnp.float32
    assert cotangent == jnp.float32(expected), (x, g, cotangent, expected)


def test_lower_leaky_hard_grid_vectorized():
    """Same six cases plus boundaries, as one array vjp — checks the broadcasting path."""
    xs = jnp.array([c[0] for c in CASES], dtype=jnp.float32)
    gs = jnp.array([c[1] for c in CASES], dtype=jnp.float32)
    expected = jnp.array([c[2] for c in CASES], dtype=jnp.float32)
    _, vjp_fn = jax.vjp(lower_leaky_hard_sigmoid, xs)
    (cotangents,) = vjp_fn(gs)
    assert jnp.array_equal(cotangents, expected), (cotangents, expected)
