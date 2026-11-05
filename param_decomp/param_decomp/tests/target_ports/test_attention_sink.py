"""The attention sink of `placed_dot_product_attention`: each query head's softmax gains one
slot at that head's sink logit, whose value is zero."""

import math
from collections.abc import Callable, Iterator
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from param_decomp.attention import AttentionImplementation, placed_dot_product_attention
from param_decomp.sequence import SequenceLayout

N_HEADS, HEAD_DIM, T = 4, 8, 16
Operands = tuple[Array, Array, Array, Array, Array]


@pytest.fixture(autouse=True)
def float32_products() -> Iterator[None]:
    """Comparisons against float64 references hold float32 products exact, as on CPU; a GPU
    would round them to TF32 by default."""
    with jax.default_matmul_precision("highest"):
        yield


def _operands(n_kv_heads: int) -> Operands:
    """Logits and sink logits of a few units, so neither the sink nor the keys dominate. The
    second row's last three keys are masked out, and its query 9 sees no key at all: under a
    sink it attends to nothing."""
    keys = jax.random.split(jax.random.key(0), 4)
    q = jax.random.normal(keys[0], (2, T, N_HEADS, HEAD_DIM), jnp.float32) * 3.0
    k = jax.random.normal(keys[1], (2, T, n_kv_heads, HEAD_DIM), jnp.float32) * 3.0
    v = jax.random.normal(keys[2], (2, T, n_kv_heads, HEAD_DIM), jnp.float32)
    sink_logits = jax.random.normal(keys[3], (N_HEADS,), jnp.float32) * 2.0
    mask = jnp.ones((2, 1, T, T), bool).at[1, :, :, -3:].set(False).at[1, :, 9, :].set(False)
    return q, k, v, mask, sink_logits


def _attention(operands: Operands, is_causal: bool, cap: float | None) -> Array:
    q, k, v, mask, sink_logits = operands
    return placed_dot_product_attention(
        q,
        k,
        v,
        mask,
        is_causal=is_causal,
        implementation="xla",
        qkv_sharding=None,
        logit_softcap=cap,
        sink_logits=sink_logits,
    )


def _reference(operands: Operands, is_causal: bool, cap: float | None) -> np.ndarray:
    """The softmax over the sink and every allowed key, in float64; the sink's value is zero."""
    q, k, v, mask, sink_logits = (np.asarray(a, np.float64) for a in operands)
    group = N_HEADS // k.shape[2]
    k, v = np.repeat(k, group, axis=2), np.repeat(v, group, axis=2)
    logits = np.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(HEAD_DIM)
    if cap is not None:
        logits = cap * np.tanh(logits / cap)
    allowed = mask.astype(bool) & (np.tril(np.ones((T, T), bool)) if is_causal else True)
    logits = np.where(allowed, logits, -np.inf)
    sink = np.broadcast_to(sink_logits[None, :, None, None], (*logits.shape[:-1], 1))
    every = np.concatenate([sink, logits], axis=-1)
    weights = np.exp(every - every.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bkhd->bqhd", weights[..., 1:], v)


@pytest.mark.parametrize("cap", [None, 2.0])
@pytest.mark.parametrize("n_kv_heads", [N_HEADS, 2])
@pytest.mark.parametrize("is_causal", [True, False])
def test_a_sink_is_a_zero_value_slot_at_its_logit(
    is_causal: bool, n_kv_heads: int, cap: float | None
):
    """A cap bends the keys' logits and leaves the sink's as it is."""
    operands = _operands(n_kv_heads)
    out = np.asarray(_attention(operands, is_causal, cap))
    np.testing.assert_allclose(out, _reference(operands, is_causal, cap), atol=1e-5)
    np.testing.assert_array_equal(out[1, 9], 0.0)


@pytest.mark.parametrize("cap", [None, 2.0])
@pytest.mark.parametrize("is_causal", [True, False])
def test_a_sink_far_below_every_logit_leaves_attention_unchanged(
    is_causal: bool, cap: float | None
):
    q, k, v, mask, _ = _operands(2)
    mask = mask.at[1, :, 9, 0].set(True)
    plain = placed_dot_product_attention(
        q,
        k,
        v,
        mask,
        is_causal=is_causal,
        implementation="xla",
        qkv_sharding=None,
        logit_softcap=cap,
        sink_logits=None,
    )
    sunk = _attention((q, k, v, mask, jnp.full((N_HEADS,), -1e4)), is_causal, cap)
    np.testing.assert_allclose(sunk, plain, atol=1e-5)


def _jnp_reference(
    q: Array,
    k: Array,
    v: Array,
    sink_logits: Array,
    mask: Array,
    is_causal: bool,
    cap: float | None,
) -> Array:
    """`_reference` in jnp, so autodiff gives its gradients."""
    group = N_HEADS // k.shape[2]
    k, v = jnp.repeat(k, group, axis=2), jnp.repeat(v, group, axis=2)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(HEAD_DIM)
    if cap is not None:
        logits = cap * jnp.tanh(logits / cap)
    if is_causal:
        mask = mask & jnp.tril(jnp.ones((T, T), bool))
    logits = jnp.where(mask, logits, -jnp.inf)
    sink = jnp.broadcast_to(sink_logits[None, :, None, None], (*logits.shape[:-1], 1))
    weights = jax.nn.softmax(jnp.concatenate([sink, logits], axis=-1), axis=-1)
    return jnp.einsum("bhqk,bkhd->bqhd", weights[..., 1:], v)


@pytest.mark.parametrize("cap", [None, 2.0])
@pytest.mark.parametrize("n_kv_heads", [N_HEADS, 2])
@pytest.mark.parametrize("is_causal", [True, False])
def test_every_operand_and_the_sink_get_the_reference_gradient(
    is_causal: bool, n_kv_heads: int, cap: float | None
):
    """The uncapped sink rides the kernel's own backward as an extra key; its gradient, and
    the queries', keys' and values', are the softmax's."""
    q, k, v, mask, sink_logits = _operands(n_kv_heads)
    cotangent = jax.random.normal(jax.random.key(5), q.shape)

    def objective(attend: Callable[..., Array]) -> Callable[..., Array]:
        return lambda *operands: jnp.sum(attend(*operands) * cotangent)

    def placed(q: Array, k: Array, v: Array, sink_logits: Array) -> Array:
        return _attention((q, k, v, mask, sink_logits), is_causal, cap)

    def reference(q: Array, k: Array, v: Array, sink_logits: Array) -> Array:
        return _jnp_reference(q, k, v, sink_logits, mask, is_causal, cap)

    argnums = (0, 1, 2, 3)
    got = jax.grad(objective(placed), argnums=argnums)(q, k, v, sink_logits)
    want = jax.grad(objective(reference), argnums=argnums)(q, k, v, sink_logits)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g, w, atol=1e-4, rtol=1e-4)
    assert np.all(np.asarray(got[3]) != 0.0)


def _on_hopper_or_blackwell() -> bool:
    device = jax.devices()[0]
    return device.platform == "gpu" and str(device.compute_capability).split(".")[0] in ("9", "10")


@pytest.mark.skipif(not _on_hopper_or_blackwell(), reason="cuDNN flash attention needs sm90+")
@pytest.mark.parametrize("head_dim", [120, 128])
@pytest.mark.parametrize("is_causal", [True, False])
def test_cudnn_with_a_sink_is_as_accurate_as_xla(head_dim: int, is_causal: bool):
    """The CI fn's head dims (Qwen3-1.7B's CI: 120) through cuDNN in bf16, with document
    boundaries off the tile grid: each output and gradient lies within 2%, or twice the XLA
    path's own bf16 error, of the attention computed in float32. The sink's gradient sums
    over every query, which moves it by about 1% in bf16."""
    batch, seq_len, heads = 2, 256, 4
    keys = jax.random.split(jax.random.key(0), 5)
    q, k, v, out_grad = (
        jax.random.normal(key, (batch, seq_len, heads, head_dim), jnp.bfloat16) for key in keys[:4]
    )
    sink_logits = jax.random.normal(keys[4], (heads,), jnp.bfloat16) * 2
    document_ids = jnp.zeros((batch, seq_len), jnp.int32).at[:, 100:].set(1).at[0, 177:].set(2)
    mask = SequenceLayout(document_ids).attention_mask()[:, None, :, :]

    def attend(implementation: AttentionImplementation) -> Callable[..., Array]:
        def run(q: Array, k: Array, v: Array, sink_logits: Array) -> Array:
            return placed_dot_product_attention(
                q, k, v, mask, is_causal=is_causal, implementation=implementation,
                qkv_sharding=None, logit_softcap=None, sink_logits=sink_logits,
            )  # fmt: skip

        return run

    def output_and_grads(
        implementation: AttentionImplementation, dtype: jnp.dtype
    ) -> tuple[Array, ...]:
        operands = (a.astype(dtype) for a in (q, k, v, sink_logits))
        out, vjp = jax.vjp(attend(implementation), *operands)
        return out, *vjp(out_grad.astype(dtype))

    def error(value: Array, exact: Array) -> float:
        return float(jnp.linalg.norm(value.astype(jnp.float32) - exact) / jnp.linalg.norm(exact))

    exact = jax.jit(lambda: output_and_grads("xla", jnp.dtype(jnp.float32)))()
    xla = jax.jit(lambda: output_and_grads("xla", jnp.dtype(jnp.bfloat16)))()
    got = jax.jit(lambda: output_and_grads("flash", jnp.dtype(jnp.bfloat16)))()
    for g, x, e in zip(got, xla, exact, strict=True):
        assert error(g, e) < max(2e-2, 2 * error(x, e))


@pytest.mark.skipif(not _on_hopper_or_blackwell(), reason="cuDNN flash attention needs sm90+")
def test_cudnn_with_a_sink_inside_the_flash_shard_map():
    """The flash path's shard map, on a one-GPU explicit mesh: the sink logits rest replicated
    over the batch axis, and their gradient is the one outside the map."""
    batch, seq_len, heads, head_dim = 2, 256, 4, 120
    keys = jax.random.split(jax.random.key(1), 5)
    q, k, v, out_grad = (
        jax.random.normal(key, (batch, seq_len, heads, head_dim), jnp.bfloat16) for key in keys[:4]
    )
    sink_logits = jax.random.normal(keys[4], (heads,), jnp.float32)
    mask = SequenceLayout(jnp.zeros((batch, seq_len), jnp.int32)).attention_mask()[:, None]
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("batch", "head"),
        axis_types=(AxisType.Explicit,) * 2,
    )

    def output_and_grads(
        sharding: NamedSharding | None,
        q: Array,
        k: Array,
        v: Array,
        sink_logits: Array,
        mask: Array,
    ) -> tuple[Array, ...]:
        def run(q: Array, k: Array, v: Array, sink_logits: Array) -> Array:
            return placed_dot_product_attention(
                q, k, v, mask, is_causal=True, implementation="flash", qkv_sharding=sharding,
                logit_softcap=None, sink_logits=sink_logits,
            )  # fmt: skip

        out, vjp = jax.vjp(run, q, k, v, sink_logits)
        return out, *vjp(out_grad)

    want = jax.jit(partial(output_and_grads, None))(q, k, v, sink_logits, mask)
    with jax.set_mesh(mesh):
        sharding = NamedSharding(mesh, P("batch", None, "head", None))
        placed = (
            *(jax.device_put(a, sharding) for a in (q, k, v)),
            jax.device_put(sink_logits, NamedSharding(mesh, P())),
            jax.device_put(mask, NamedSharding(mesh, P("batch", None, None, None))),
        )
        got = jax.jit(partial(output_and_grads, sharding))(*placed)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(np.asarray(g, np.float32), np.asarray(w, np.float32), atol=1e-2)
