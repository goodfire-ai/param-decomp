"""The logit softcap of `placed_dot_product_attention`: on the XLA path each logit `s` becomes
`cap · tanh(s / cap)` before the masked softmax."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.attention import placed_dot_product_attention

N_HEADS, HEAD_DIM, T = 4, 8, 16


def _operands(n_kv_heads: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Logits of a few units, so a cap of 2 bends them; the second row's last three keys are
    masked out, which leaves every query some key to attend under either mask."""
    keys = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(keys[0], (2, T, N_HEADS, HEAD_DIM), jnp.float32) * 3.0
    k = jax.random.normal(keys[1], (2, T, n_kv_heads, HEAD_DIM), jnp.float32) * 3.0
    v = jax.random.normal(keys[2], (2, T, n_kv_heads, HEAD_DIM), jnp.float32)
    mask = jnp.ones((2, 1, T, T), bool).at[1, :, :, -3:].set(False)
    return q, k, v, mask


def _attention(
    operands: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    is_causal: bool,
    logit_softcap: float | None,
) -> np.ndarray:
    q, k, v, mask = operands
    out = placed_dot_product_attention(
        q,
        k,
        v,
        mask,
        is_causal=is_causal,
        implementation="xla",
        qkv_sharding=None,
        logit_softcap=logit_softcap,
        sink_logits=None,
    )
    return np.asarray(out)


def _reference(
    operands: tuple[jax.Array, jax.Array, jax.Array, jax.Array], is_causal: bool, cap: float
) -> np.ndarray:
    q, k, v, mask = (np.asarray(a, np.float64) for a in operands)
    group = N_HEADS // k.shape[2]
    k, v = np.repeat(k, group, axis=2), np.repeat(v, group, axis=2)
    logits = cap * np.tanh(np.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(HEAD_DIM) / cap)
    allowed = mask.astype(bool) & (np.tril(np.ones((T, T), bool)) if is_causal else True)
    logits = np.where(allowed, logits, -np.inf)
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bkhd->bqhd", weights, v)


@pytest.mark.parametrize("n_kv_heads", [N_HEADS, 2])
@pytest.mark.parametrize("is_causal", [True, False])
def test_capped_attention_is_the_masked_softmax_of_capped_logits(n_kv_heads: int, is_causal: bool):
    operands = _operands(n_kv_heads)
    np.testing.assert_allclose(
        _attention(operands, is_causal, 2.0), _reference(operands, is_causal, 2.0), atol=1e-5
    )


@pytest.mark.parametrize("n_kv_heads", [N_HEADS, 2])
@pytest.mark.parametrize("is_causal", [True, False])
def test_a_cap_far_above_every_logit_leaves_attention_unchanged(n_kv_heads: int, is_causal: bool):
    operands = _operands(n_kv_heads)
    np.testing.assert_allclose(
        _attention(operands, is_causal, 1e6), _attention(operands, is_causal, None), atol=1e-5
    )
