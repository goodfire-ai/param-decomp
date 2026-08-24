"""Pin the SDPA backend dispatch: cuDNN only where cuDNN flash attention can run —
GPU, half precision, and a sequence length in the family we run through it (multiples
of 64). fp32 falls back to the XLA composite (cuDNN's SDPA rejects it), and so do the
tPD target stream's natural prompt lengths (found live: `Unsupported sequence length
Q 5, KV 5` from `check_is_flash_attention` on run p-5de41a1c)."""

import jax.numpy as jnp
import pytest

from param_decomp.vendored_jax.llama import attn_implementation


@pytest.mark.parametrize(
    ("backend", "dtype", "seq_len", "expected"),
    [
        ("gpu", jnp.bfloat16, 2048, "cudnn"),
        ("gpu", jnp.float16, 512, "cudnn"),
        ("gpu", jnp.float32, 2048, "xla"),
        ("cpu", jnp.bfloat16, 2048, "xla"),
        ("cpu", jnp.float32, 2048, "xla"),
        ("gpu", jnp.bfloat16, 5, "xla"),  # tPD target stream: natural prompt length
        ("gpu", jnp.bfloat16, 96, "xla"),  # not a multiple of 64
        ("gpu", jnp.bfloat16, 64, "cudnn"),
    ],
)
def test_attn_implementation_dispatch(
    backend: str, dtype: jnp.dtype, seq_len: int, expected: str
) -> None:
    assert attn_implementation("auto", backend, jnp.dtype(dtype), seq_len) == expected


def test_explicit_xla_ignores_cudnn_compatibility() -> None:
    assert attn_implementation("xla", "gpu", jnp.dtype(jnp.bfloat16), 512) == "xla"


def test_explicit_cudnn_fails_closed_when_incompatible() -> None:
    with pytest.raises(AssertionError):
        attn_implementation("cudnn", "cpu", jnp.dtype(jnp.bfloat16), 512)
