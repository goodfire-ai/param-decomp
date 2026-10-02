"""Attention's saved backward inputs stay on the query's batch/head shard."""

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


@pytest.mark.multidevice
@pytest.mark.parametrize("implementation", ("xla", "flash"))
@pytest.mark.parametrize("head_parts", (1, 2))
@pytest.mark.parametrize("kv_heads", (4, 8))
@pytest.mark.parametrize("causal", (False, True))
def test_attention_scan_saved_residuals_match_unsharded_gradients(
    implementation: AttentionImplementation, head_parts: int, kv_heads: int, causal: bool
):
    if implementation == "flash" and jax.default_backend() != "gpu":
        pytest.skip("requires actual cuDNN flash attention")
    count = 2 * head_parts
    if len(jax.local_devices()) < count:
        pytest.skip(f"requires {count} local devices")
    batch, tokens, q_heads, width = 8, 128, 8, 64
    keys = jax.random.split(jax.random.key(80), 4)
    scale = (jnp.arange(batch, dtype=jnp.bfloat16) + 1)[:, None, None, None] / 4
    q = jax.random.normal(keys[0], (batch, tokens, q_heads, width), dtype=jnp.bfloat16) * scale
    k = jax.random.normal(keys[1], (batch, tokens, kv_heads, width), dtype=jnp.bfloat16) * scale
    v = jax.random.normal(keys[2], k.shape, dtype=jnp.bfloat16)
    cotangent = jax.random.normal(keys[3], q.shape, dtype=jnp.bfloat16)
    # Distinct examples and document boundaries expose incorrect rank-zero LSE reuse.
    split = jnp.arange(batch)[:, None] * 3 + tokens // 2
    documents = (jnp.arange(tokens)[None, :] >= split).astype(jnp.int32)
    mask = SequenceLayout(documents).attention_mask()[:, None]

    def objective(
        q: Array,
        k: Array,
        v: Array,
        mask: Array,
        cotangent: Array,
        *,
        sharding: NamedSharding | None,
        remat: bool,
        backend: AttentionImplementation,
    ) -> Array:
        def block(q: Array, _: None) -> tuple[Array, None]:
            value = placed_dot_product_attention(
                q, k, v, mask, is_causal=causal, implementation=backend, qkv_sharding=sharding
            )
            return q * 0.5 + value, None

        body = jax.checkpoint(block) if remat else block
        output, _ = jax.lax.scan(body, q, None, length=3)
        return jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32)) / output.size

    reference = jax.jit(
        jax.value_and_grad(
            partial(objective, sharding=None, remat=False, backend=implementation),
            argnums=(0, 1, 2),
        )
    )(q, k, v, mask, cotangent)
    oracle = jax.jit(
        jax.value_and_grad(
            partial(objective, sharding=None, remat=False, backend="xla"), argnums=(0, 1, 2)
        )
    )(q, k, v, mask, cotangent)
    mesh = Mesh(
        np.asarray(jax.local_devices()[:count]).reshape(2, head_parts),
        ("batch", "head"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    spec = P("batch", None, "head", None)
    with jax.set_mesh(mesh):
        sharding = NamedSharding(mesh, spec)
        inputs = tuple(jax.device_put(x, sharding) for x in (q, k, v))
        placed_mask = jax.device_put(mask, NamedSharding(mesh, P("batch", None, None, None)))
        placed_cotangent = jax.device_put(cotangent, sharding)
        for remat in (False, True):
            differentiate = jax.jit(
                jax.value_and_grad(
                    partial(objective, sharding=sharding, remat=remat, backend=implementation),
                    argnums=(0, 1, 2),
                )
            )
            compiled = differentiate.lower(*inputs, placed_mask, placed_cotangent).compile()
            actual = compiled(*inputs, placed_mask, placed_cotangent)
            for expected in (reference, oracle):
                for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
                    actual_values = np.asarray(a, np.float32)
                    expected_values = np.asarray(e, np.float32)
                    assert np.isfinite(actual_values).all()
                    error = np.linalg.norm(actual_values - expected_values) / max(
                        float(np.linalg.norm(expected_values)), 1e-12
                    )
                    assert error < 0.025, error
            if implementation == "flash":
                hlo = compiled.as_text()
                assert hlo is not None
                backward = [
                    line
                    for line in hlo.splitlines()
                    if 'custom_call_target="__cudnn$fmha' in line and "Backward" in line
                ]
                assert backward, "test must execute real fused attention backward"
                local_heads = q_heads // head_parts
                for line in backward:
                    assert f"f32[{batch // 2},{local_heads},{tokens}]" in line
                    assert f"f32[{batch},{local_heads},{tokens}]" not in line


@pytest.mark.multidevice
def test_flash_shard_map_forward_and_pullback_with_cpu_attention(monkeypatch: pytest.MonkeyPatch):
    """Exercise Flash's partition boundary using the XLA kernel, without claiming cuDNN parity."""
    if jax.default_backend() != "cpu" or jax.local_device_count() < 4:
        pytest.skip("requires four CPU devices")
    import param_decomp.attention as attention

    batch, tokens, query_heads, kv_heads, width = 4, 8, 4, 2, 4
    q_key, k_key, v_key = jax.random.split(jax.random.key(11), 3)
    q = jax.random.normal(q_key, (batch, tokens, query_heads, width))
    k = jax.random.normal(k_key, (batch, tokens, kv_heads, width))
    v = jax.random.normal(v_key, k.shape)
    documents = (jnp.arange(tokens)[None, :] >= jnp.arange(batch)[:, None] + 2).astype(jnp.int32)
    mask = SequenceLayout(documents).attention_mask()[:, None]

    def objective(
        q: Array, k: Array, v: Array, mask: Array, *, sharding: NamedSharding | None
    ) -> Array:
        y = placed_dot_product_attention(
            q,
            k,
            v,
            mask,
            is_causal=True,
            implementation="flash",
            qkv_sharding=sharding,
        )
        return jnp.sum(jnp.sin(y))

    monkeypatch.setattr(attention, "jax_attention_implementation", lambda *_args: "xla")
    expected = jax.jit(jax.value_and_grad(partial(objective, sharding=None), argnums=(0, 1, 2)))(
        q, k, v, mask
    )
    mesh = Mesh(
        np.asarray(jax.local_devices()[:4]).reshape(2, 2),
        ("batch", "head"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    sharding = NamedSharding(mesh, P("batch", None, "head", None))
    with jax.set_mesh(mesh):
        operands = tuple(jax.device_put(value, sharding) for value in (q, k, v))
        placed_mask = jax.device_put(mask, NamedSharding(mesh, P("batch", None, None, None)))
        actual = jax.jit(
            jax.value_and_grad(partial(objective, sharding=sharding), argnums=(0, 1, 2))
        )(*operands, placed_mask)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, e, rtol=2e-5, atol=2e-5)
    for gradient, operand in zip(actual[1], operands, strict=True):
        assert gradient.sharding.is_equivalent_to(operand.sharding, ndim=4)
