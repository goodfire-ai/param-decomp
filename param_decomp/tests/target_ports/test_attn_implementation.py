"""Attention never falls back from requested flash execution to XLA."""

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
)
from param_decomp.sequence import SequenceLayout


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_explicit_flash_keeps_the_requested_backend(dtype: DTypeLike):
    assert jax_attention_implementation("flash", "gpu", jnp.dtype(dtype)) == "cudnn"


@pytest.mark.parametrize(
    ("backend", "dtype"),
    [("cpu", jnp.bfloat16), ("cpu", jnp.float32), ("gpu", jnp.float32)],
)
def test_flash_refuses_incompatible_execution_without_falling_back(backend: str, dtype: DTypeLike):
    with pytest.raises(AssertionError, match="Select 'xla' explicitly"):
        jax_attention_implementation("flash", backend, jnp.dtype(dtype))


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
