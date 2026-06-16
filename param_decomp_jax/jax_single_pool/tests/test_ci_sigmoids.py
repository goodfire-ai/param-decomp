"""CI-fn output-squashing parity (torch `pd.sigmoid_type` → JAX `squashing_fns`).

The torch sigmoids are simple closed forms; their reference outputs are encoded here as
numpy expressions (the SAME math torch's `param_decomp.ci_sigmoids` runs), so this pins
JAX↔torch forward parity for every supported `sigmoid_type` without importing torch.
The one custom-VJP — `lower_leaky_hard` — also has its backward checked against torch's
`LowerLeakyHardSigmoidFunction.backward`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_single_pool.ci_fn import (
    hard_sigmoid,
    leaky_hard_sigmoid,
    lower_leaky_hard_sigmoid,
    normal_sigmoid,
    squashing_fns,
    swish_hard_sigmoid,
    upper_leaky_hard_sigmoid,
)

X = np.linspace(-3.0, 3.0, 61, dtype=np.float64)
ALPHA = 0.01


def _torch_normal(x):
    return 1.0 / (1.0 + np.exp(-x))


def _torch_hard(x):
    return np.clip(x, 0.0, 1.0)


def _torch_leaky_hard(x):
    return np.where(x > 0, np.minimum(x, 1.0), ALPHA * x)


def _torch_upper_leaky_hard(x):
    return np.where(x > 1, 1 + ALPHA * (x - 1), np.clip(x, 0.0, 1.0))


def _torch_swish(x, beta):
    return x * (1.0 / (1.0 + np.exp(-beta * x)))


def _torch_upside_down_swish(x, beta):
    return x * (1.0 / (1.0 + np.exp(beta * x)))


def _torch_swish_hard(x):
    beta, scale, xshift, yshift = 10.0, 0.5, 0.5, 0.5
    x = x - xshift
    return (
        yshift
        + (_torch_upside_down_swish(x - scale, beta) - _torch_swish(x, beta))
        + (_torch_swish(x + scale, beta) - _torch_upside_down_swish(x, beta))
    )


@pytest.mark.parametrize(
    "jax_fn, torch_ref",
    [
        (normal_sigmoid, _torch_normal),
        (hard_sigmoid, _torch_hard),
        (leaky_hard_sigmoid, _torch_leaky_hard),
        (upper_leaky_hard_sigmoid, _torch_upper_leaky_hard),
        (lower_leaky_hard_sigmoid, _torch_hard),  # forward is exactly clamp(x,0,1)
        (swish_hard_sigmoid, _torch_swish_hard),
    ],
)
def test_forward_matches_torch_math(jax_fn, torch_ref):
    got = np.asarray(jax_fn(jnp.asarray(X, dtype=jnp.float32)), dtype=np.float64)
    np.testing.assert_allclose(got, torch_ref(X), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("g_sign", [1.0, -1.0])
def test_lower_leaky_hard_backward_matches_torch(g_sign):
    """torch `LowerLeakyHardSigmoidFunction.backward`: for x<=0 the leak `alpha*g` fires
    ONLY when g<0; for 0<x<=1 grad passes through; for x>1 grad is zeroed."""
    g = g_sign * np.ones_like(X)
    _, vjp = jax.vjp(lower_leaky_hard_sigmoid, jnp.asarray(X, dtype=jnp.float32))
    got = np.asarray(vjp(jnp.asarray(g, dtype=jnp.float32))[0], dtype=np.float64)
    expected = np.where(
        X <= 0,
        np.where(g < 0, ALPHA * g, 0.0),
        np.where(X <= 1, g, 0.0),
    )
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_squashing_dispatch_matches_torch_component_model():
    """leaky_hard is the asymmetric production pair; every other type uses one fn twice
    (torch component_model.py:180-186)."""
    lower, upper = squashing_fns("leaky_hard")
    assert lower is lower_leaky_hard_sigmoid and upper is upper_leaky_hard_sigmoid

    for st, fn in [
        ("normal", normal_sigmoid),
        ("hard", hard_sigmoid),
        ("upper_leaky_hard", upper_leaky_hard_sigmoid),
        ("swish_hard", swish_hard_sigmoid),
    ]:
        lower, upper = squashing_fns(st)
        assert lower is fn and upper is fn
