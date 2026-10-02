"""Scaled dot-product attention with explicit backend and batch/head sharding."""

from typing import Literal, get_args

import jax
import jax.numpy as jnp
from jax.extend.backend import get_default_device
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float

from param_decomp.sequence import SequenceLayout

AttentionImplementation = Literal["flash", "xla"]


def jax_attention_implementation(
    requested: AttentionImplementation, backend: str, dtype: jnp.dtype
) -> Literal["cudnn", "xla"]:
    """Translate the requested algorithm to JAX's backend, without fallback."""
    assert requested in get_args(AttentionImplementation), (
        f"attention implementation must explicitly select 'flash' or 'xla', got {requested!r}"
    )
    match requested:
        case "flash":
            assert backend == "gpu" and dtype in (jnp.float16, jnp.bfloat16), (
                f"cuDNN flash attention requires a GPU and float16/bfloat16 operands; "
                f"got backend={backend!r}, dtype={dtype}. "
                "Select 'xla' explicitly to allow non-flash attention."
            )
            return "cudnn"
        case "xla":
            return "xla"


def placed_dot_product_attention(
    q: Float[Array, "b t qh hd"],
    k: Float[Array, "b s kvh hd"],
    v: Float[Array, "b s kvh hd"],
    mask: Bool[Array, "b 1 t s"],
    *,
    is_causal: bool,
    implementation: AttentionImplementation,
    qkv_sharding: NamedSharding | None,
) -> Float[Array, "b t qh hd"]:
    backend = jax_attention_implementation(implementation, get_default_device().platform, q.dtype)

    def attention(q: Array, k: Array, v: Array, mask: Array) -> Array:
        return jax.nn.dot_product_attention(
            q, k, v, mask=mask, is_causal=is_causal, implementation=backend
        )

    if qkv_sharding is None:
        return attention(q, k, v, mask)
    spec = qkv_sharding.spec
    assert len(spec) == 4 and spec[1] is None and spec[3] is None, (
        f"Attention may shard batch and heads only, got {spec}"
    )
    match implementation:
        case "flash":
            # The saved output and softmax statistics must share Q's local batch.
            # Enclose both directions of cuDNN's custom VJP in the same shard map.
            return jax.shard_map(
                attention,
                mesh=qkv_sharding.mesh,
                in_specs=(spec, spec, spec, P(spec[0], None, None, None)),
                out_specs=spec,
            )(q, k, v, mask)
        case "xla":
            return jax.sharding.auto_axes(out_sharding=qkv_sharding)(attention)(q, k, v, mask)


def causal_attention_head_first(
    q: Float[Array, "b qh t hd"],
    k: Float[Array, "b kvh t hd"],
    v: Float[Array, "b kvh t hd"],
    sequence: SequenceLayout,
    qkv_sharding: NamedSharding | None,
    implementation: AttentionImplementation,
) -> Float[Array, "b qh t hd"]:
    """Document-isolated causal attention with heads before tokens.

    ``qkv_sharding`` describes the token-before-head layout used by the kernel.
    """
    qt, kt, vt = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
    out = placed_dot_product_attention(
        qt,
        kt,
        vt,
        sequence.attention_mask()[:, None, :, :],
        is_causal=True,
        implementation=implementation,
        qkv_sharding=qkv_sharding,
    )
    return out.transpose(0, 2, 1, 3)
