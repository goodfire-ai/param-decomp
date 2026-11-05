"""Attention never falls back from requested flash execution to XLA."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import jax
import jax.numpy as jnp
import pytest
from jax.typing import DTypeLike
from jaxtyping import Array

from param_decomp.attention import (
    AttentionImplementation,
    causal_attention_head_first,
    jax_attention_implementation,
    placed_dot_product_attention,
)
from param_decomp.sequence import SequenceLayout


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_explicit_flash_keeps_the_requested_backend(dtype: DTypeLike):
    assert jax_attention_implementation("flash", "gpu", jnp.dtype(dtype)) == "cudnn"
    assert jax_attention_implementation("mosaic", "gpu", jnp.dtype(dtype)) == "mosaic"


@pytest.mark.parametrize("requested", ["flash", "mosaic"])
@pytest.mark.parametrize(
    ("backend", "dtype"),
    [("cpu", jnp.bfloat16), ("cpu", jnp.float32), ("gpu", jnp.float32)],
)
def test_flash_refuses_incompatible_execution_without_falling_back(
    requested: AttentionImplementation, backend: str, dtype: DTypeLike
):
    with pytest.raises(AssertionError, match="Select 'xla' explicitly"):
        jax_attention_implementation(requested, backend, jnp.dtype(dtype))


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_non_flash_execution_requires_the_explicit_choice(backend: str):
    assert jax_attention_implementation("xla", backend, jnp.dtype(jnp.float32)) == "xla"


def test_legacy_auto_cannot_reach_jax_as_an_implicit_default():
    with pytest.raises(AssertionError, match="must explicitly select"):
        jax_attention_implementation(
            cast(AttentionImplementation, cast(object, "auto")), "cpu", jnp.dtype(jnp.float32)
        )


def test_target_attention_requires_explicit_xla_on_cpu():
    q = jnp.ones((1, 2, 4, 8), dtype=jnp.bfloat16)
    sequence = SequenceLayout(jnp.zeros((1, 4), dtype=jnp.int32))
    with pytest.raises(AssertionError, match="cuDNN flash attention requires a GPU"):
        causal_attention_head_first(q, q, q, sequence, None, "flash")
    result = causal_attention_head_first(q, q, q, sequence, None, "xla")
    assert result.shape == q.shape
    assert jnp.all(jnp.isfinite(result))


def test_flash_kernel_rejection_propagates_without_retry(monkeypatch: pytest.MonkeyPatch):
    q = jnp.ones((1, 2, 5, 8), dtype=jnp.bfloat16)
    sequence = SequenceLayout(jnp.zeros((1, 5), dtype=jnp.int32))
    calls: list[str] = []

    def reject_shape(
        query: Array,
        key: Array,
        value: Array,
        *,
        is_causal: bool,
        mask: Array,
        implementation: Literal["cudnn", "xla"],
    ):
        assert query.shape[1] == 5 and is_causal
        assert key.shape == value.shape == query.shape
        assert mask.shape == (query.shape[0], 1, query.shape[1], query.shape[1])
        calls.append(implementation)
        raise NotImplementedError("unsupported flash shape")

    monkeypatch.setattr(
        "param_decomp.attention.get_default_device", lambda: _CompileOnlyGPUDevice()
    )
    monkeypatch.setattr(jax.nn, "dot_product_attention", reject_shape)
    with pytest.raises(NotImplementedError, match="unsupported flash shape"):
        causal_attention_head_first(q, q, q, sequence, None, "flash")
    assert calls == ["cudnn"]


@dataclass(frozen=True)
class _CompileOnlyGPUDevice:
    platform: str = "gpu"


def test_flash_respects_compile_target_when_host_backend_is_cpu(monkeypatch: pytest.MonkeyPatch):
    assert jax.default_backend() == "cpu"
    calls: list[str] = []

    def record_implementation(
        query: Array,
        key: Array,
        value: Array,
        *,
        is_causal: bool,
        mask: Array,
        implementation: Literal["cudnn", "xla"],
    ) -> Array:
        assert is_causal and query.shape == key.shape == value.shape
        assert mask.shape == (query.shape[0], 1, query.shape[1], query.shape[1])
        calls.append(implementation)
        return query

    monkeypatch.setattr(jax.nn, "dot_product_attention", record_implementation)
    operands = jax.ShapeDtypeStruct((1, 2, 4, 8), jnp.bfloat16)
    sequence = SequenceLayout(jnp.zeros((1, 4), dtype=jnp.int32))
    with jax.default_device(_CompileOnlyGPUDevice()):
        result = jax.eval_shape(
            lambda q: causal_attention_head_first(q, q, q, sequence, None, "flash"), operands
        )
        assert jax.default_backend() == "cpu"
    assert result.shape == operands.shape
    assert calls == ["cudnn"]


def _on_hopper_or_blackwell() -> bool:
    device = jax.devices()[0]
    return device.platform == "gpu" and str(device.compute_capability).split(".")[0] in ("9", "10")


@pytest.mark.skipif(
    not _on_hopper_or_blackwell(), reason="tokamax Mosaic GPU attention needs sm90 or sm100"
)
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("head_dim", [34, 128])
def test_mosaic_matches_xla_attention_with_documents(head_dim: int, is_causal: bool):
    """The CI fn's shapes: a head dim cuDNN rejects (34) and a standard one, causal and
    bidirectional, with document boundaries off the tile grid."""
    batch, seq_len, heads = 2, 256, 4
    keys = jax.random.split(jax.random.key(0), 4)
    q, k, v, out_grad = (
        jax.random.normal(key, (batch, seq_len, heads, head_dim), jnp.bfloat16) for key in keys
    )
    document_ids = jnp.zeros((batch, seq_len), jnp.int32).at[:, 100:].set(1).at[0, 177:].set(2)
    mask = SequenceLayout(document_ids).attention_mask()[:, None, :, :]

    def attend(implementation: AttentionImplementation) -> Callable[..., Array]:
        def run(q: Array, k: Array, v: Array) -> Array:
            return placed_dot_product_attention(
                q, k, v, mask, is_causal=is_causal, implementation=implementation,
                qkv_sharding=None, logit_softcap=None, sink_logits=None,
            )  # fmt: skip

        return run

    def output_and_grads(implementation: AttentionImplementation) -> tuple[Array, ...]:
        out, vjp = jax.vjp(attend(implementation), q, k, v)
        return out, *vjp(out_grad)

    got = jax.jit(lambda: output_and_grads("mosaic"))()
    want = jax.jit(lambda: output_and_grads("xla"))()
    for g, w in zip(got, want, strict=True):
        g, w = g.astype(jnp.float32), w.astype(jnp.float32)
        assert float(jnp.linalg.norm(g - w) / jnp.linalg.norm(w)) < 1e-2
