"""A draw follows its consumer's dtype, and the bf16 and fp32 streams from one key
are unrelated samples — not roundings of each other."""

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.core.linear_plan import uniform_like

_SHAPE = (4, 16, 32)


def _draw(key: jax.Array, dtype: jnp.dtype) -> np.ndarray:
    return np.asarray(uniform_like(key, jnp.zeros(_SHAPE, dtype)).astype(jnp.float32))


def test_uniform_like_draws_in_the_reference_dtype():
    key = jax.random.PRNGKey(11)
    for dtype in (jnp.bfloat16, jnp.float32):
        assert uniform_like(key, jnp.zeros(_SHAPE, dtype)).dtype == dtype


def test_same_key_same_dtype_is_the_same_sample():
    key = jax.random.PRNGKey(11)
    for dtype in (jnp.bfloat16, jnp.float32):
        assert np.array_equal(_draw(key, dtype), _draw(key, dtype))


def test_same_key_bf16_and_fp32_are_unrelated_samples():
    """The bf16 stream is not the fp32 stream rounded: comparing the fp32 draw rounded to
    bf16 against the bf16 draw finds almost no equal elements (two independent U[0,1]
    samples land in the same bf16 bucket at roughly the bf16 mantissa resolution)."""
    key = jax.random.PRNGKey(11)
    bf16_draw = _draw(key, jnp.bfloat16)
    fp32_draw_rounded = np.asarray(
        jnp.asarray(_draw(key, jnp.float32)).astype(jnp.bfloat16).astype(jnp.float32)
    )
    assert not np.array_equal(bf16_draw, fp32_draw_rounded)
    assert np.mean(bf16_draw == fp32_draw_rounded) < 0.05


def test_pinning_the_draw_dtype_makes_the_sample_dtype_invariant():
    """Consumer-directed draws: `dtype=` overrides the reference's dtype, so a bf16 consumer can draw
    the fp32 sample and cast — both compute arms then score the same sample."""
    key = jax.random.PRNGKey(11)
    pinned_for_bf16 = uniform_like(key, jnp.zeros(_SHAPE, jnp.bfloat16), dtype=jnp.float32)
    assert pinned_for_bf16.dtype == jnp.float32
    assert np.array_equal(np.asarray(pinned_for_bf16), _draw(key, jnp.float32))
