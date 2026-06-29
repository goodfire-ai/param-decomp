"""fp8 decomposition-GEMM dot: forward + gradient closeness to the bf16 reference, for
both scaling modes and an N-D (batched) left operand."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from jax import random
from jaxtyping import Array

from param_decomp import fp8


def _bf16_dot(a: Array, b: Array) -> Array:
    ca = a.ndim - 1
    return jax.lax.dot_general(
        a, b, (((ca,), (0,)), ((), ())), preferred_element_type=jnp.float32
    ).astype(a.dtype)


def _rel(x: Array, y: Array) -> float:
    x, y = x.astype(jnp.float32), y.astype(jnp.float32)
    return float(jnp.linalg.norm(x - y) / (jnp.linalg.norm(y) + 1e-9))


@pytest.mark.parametrize("dot", [fp8._dot_per_tensor, fp8._dot_per_row])
@pytest.mark.parametrize("shape_a", [(512, 256), (8, 64, 256)])
def test_fp8_dot_matches_bf16(dot: Callable[[Array, Array], Array], shape_a: tuple[int, ...]):
    ka, kb = random.split(random.PRNGKey(0))
    a = random.normal(ka, shape_a, jnp.bfloat16)
    b = random.normal(kb, (256, 384), jnp.bfloat16)

    # forward
    assert _rel(dot(a, b), _bf16_dot(a, b)) < 0.1

    # gradients (sum-loss) vs the bf16 reference — fp8 grad noise tolerance
    loss_fp8 = lambda a, b: jnp.sum(dot(a, b).astype(jnp.float32))
    loss_bf = lambda a, b: jnp.sum(_bf16_dot(a, b).astype(jnp.float32))
    ga_fp8, gb_fp8 = jax.grad(loss_fp8, (0, 1))(a, b)
    ga_bf, gb_bf = jax.grad(loss_bf, (0, 1))(a, b)
    assert ga_fp8.shape == a.shape and gb_fp8.shape == b.shape
    assert _rel(ga_fp8, ga_bf) < 0.35
    assert _rel(gb_fp8, gb_bf) < 0.35


def test_per_row_beats_per_tensor_with_outliers():
    """Per-row scaling should resist an outlier row that wrecks the per-tensor scale."""
    ka, kb = random.split(random.PRNGKey(1))
    a = random.normal(ka, (256, 256), jnp.bfloat16)
    a = a.at[0].set(a[0] * 50.0)  # one outlier row
    b = random.normal(kb, (256, 256), jnp.bfloat16)
    ref = _bf16_dot(a, b)
    assert _rel(fp8._dot_per_row(a, b), ref) <= _rel(fp8._dot_per_tensor(a, b), ref)


def test_configure_switch():
    fp8.configure("off", "per_row")
    assert not fp8.components_enabled()
    fp8.configure("components", "per_tensor")
    assert fp8.components_enabled()
    fp8.configure("off", "per_row")  # restore
