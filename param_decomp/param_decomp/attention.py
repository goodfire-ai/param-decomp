"""Scaled dot-product attention with explicit backend and batch/head sharding."""

import math
from functools import partial
from typing import Literal, get_args

import jax
import jax.numpy as jnp
from jax._src.cudnn import fused_attention_stablehlo as cudnn_attention
from jax.extend.backend import get_default_device
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float, Float32

from param_decomp.sequence import SequenceLayout

AttentionImplementation = Literal["flash", "mosaic", "xla"]
"""`flash` is cuDNN's fused kernel; `mosaic` is tokamax's Pallas Mosaic GPU flash kernel
(Hopper and Blackwell), which also takes head dims cuDNN rejects (not a multiple of 8);
`xla` materializes the scores, and is the one that takes a logit softcap. `flash` and `xla`
take an attention sink. No choice falls back to another."""


def jax_attention_implementation(
    requested: AttentionImplementation, backend: str, dtype: jnp.dtype
) -> Literal["cudnn", "mosaic", "xla"]:
    """Translate the requested algorithm to the kernel that runs it, without fallback."""
    assert requested in get_args(AttentionImplementation), (
        f"attention implementation must explicitly select one of "
        f"{get_args(AttentionImplementation)}, got {requested!r}"
    )
    match requested:
        case "flash":
            _assert_flash_operands("cuDNN", backend, dtype)
            return "cudnn"
        case "mosaic":
            _assert_flash_operands("tokamax Mosaic GPU", backend, dtype)
            return "mosaic"
        case "xla":
            return "xla"


def _assert_flash_operands(kernel: str, backend: str, dtype: jnp.dtype) -> None:
    assert backend == "gpu" and dtype in (jnp.float16, jnp.bfloat16), (
        f"{kernel} flash attention requires a GPU and float16/bfloat16 operands; "
        f"got backend={backend!r}, dtype={dtype}. "
        "Select 'xla' explicitly to allow non-flash attention."
    )


def placed_dot_product_attention(
    q: Float[Array, "b t qh hd"],
    k: Float[Array, "b s kvh hd"],
    v: Float[Array, "b s kvh hd"],
    mask: Bool[Array, "b 1 t s"],
    *,
    is_causal: bool,
    implementation: AttentionImplementation,
    qkv_sharding: NamedSharding | None,
    logit_softcap: float | None,
    sink_logits: Float[Array, " qh"] | None,
) -> Float[Array, "b t qh hd"]:
    """`logit_softcap`, when set, softmaxes `cap · tanh(s / cap)` of each logit
    `s = q·k / √hd`; `None` softmaxes the logits as they are. Only the materialized XLA path
    applies a cap (cuDNN's fused kernel has no logit hook), so it needs `implementation="xla"`.

    `sink_logits`, when set, gives each query head one more softmax slot, its sink, whose
    logit is that head's entry at every query and whose value is zero: weight on it attends
    to nothing. A cap leaves the sink's logit as it is. cuDNN (`_cudnn_attention_with_sink`)
    and XLA take a sink; tokamax's Mosaic kernel does not."""
    backend = jax_attention_implementation(implementation, get_default_device().platform, q.dtype)

    def attention(q: Array, k: Array, v: Array, mask: Array, sink_logits: Array | None) -> Array:
        match backend, logit_softcap, sink_logits:
            case _, None, None:
                return _kernel_attention(q, k, v, mask, is_causal=is_causal, backend=backend)
            case "cudnn", None, jax.Array() as sink:
                return _cudnn_attention_with_sink(q, k, v, mask, sink, is_causal=is_causal)
            case "xla", cap, sink:
                return _materialized_attention(q, k, v, mask, sink, is_causal=is_causal, cap=cap)
            case _:
                raise AssertionError(
                    f"implementation={implementation!r} takes neither a logit softcap (xla "
                    "only) nor an attention sink (flash and xla only)"
                )

    if qkv_sharding is None:
        return attention(q, k, v, mask, sink_logits)
    spec = qkv_sharding.spec
    assert len(spec) == 4 and spec[1] is None and spec[3] is None, (
        f"Attention may shard batch and heads only, got {spec}"
    )
    match implementation:
        case "flash" | "mosaic":
            # The saved output and softmax statistics must share Q's local batch.
            # Enclose both directions of the kernel's custom VJP in the same shard map.
            # tokamax's Pallas outputs declare no varying mesh axes (vma), which the
            # checker refuses inside a manual map; the specs here are the contract.
            # A sink's logits go where their query heads are.
            sink_spec = P(spec[2])
            if sink_logits is not None:
                sink_logits = jax.reshard(sink_logits, NamedSharding(qkv_sharding.mesh, sink_spec))
            return jax.shard_map(
                attention,
                mesh=qkv_sharding.mesh,
                in_specs=(spec, spec, spec, P(spec[0], None, None, None), sink_spec),
                out_specs=spec,
                check_vma=implementation == "flash",
            )(q, k, v, mask, sink_logits)
        case "xla":
            return jax.sharding.auto_axes(out_sharding=qkv_sharding)(attention)(
                q, k, v, mask, sink_logits
            )


def _kernel_attention(
    q: Float[Array, "b t qh hd"],
    k: Float[Array, "b s kvh hd"],
    v: Float[Array, "b s kvh hd"],
    mask: Bool[Array, "b 1 t s"],
    *,
    is_causal: bool,
    backend: Literal["cudnn", "mosaic", "xla"],
) -> Float[Array, "b t qh hd"]:
    match backend:
        case "cudnn" | "xla":
            return jax.nn.dot_product_attention(
                q, k, v, mask=mask, is_causal=is_causal, implementation=backend
            )
        case "mosaic":
            # Imported at trace time: tokamax must import after the launch env and
            # before pyarrow (see `experiments/lm/run.py`).
            import tokamax

            return tokamax.dot_product_attention(
                q, k, v, mask=mask, is_causal=is_causal, implementation="mosaic"
            )


# cuDNN's fused attention with a sink, at no extra quadratic cost. With `lse` the logsumexp
# of a query's logits over its keys, the sink's logsumexp is `lse_s = logaddexp(lse, s)` and
# its output is the plain output times `w = exp(lse - lse_s)`. The flash backward reads the
# softmax only through the logsumexp and the output it is handed, so handing cuDNN's
# backward `lse_s` and the sink's output gives the exact gradients for q, k and v; the
# sink's own is `-Σ (1 - w) · (dO · O)`. JAX exposes cuDNN's forward statistics and backward
# only through its private module, pinned by the lock; the GPU test
# (`tests/target_ports/test_attention_sink.py`) checks this path against float32 attention.
def _cudnn_absent_operands(q: Array) -> tuple[Array, Array, Array, Array, Array, Array]:
    """The optional operands JAX's cuDNN rules take after the bias (query and key/value
    sequence lengths and segment offsets, key and value page tables), all absent here."""
    absent = jnp.zeros(0, q.dtype)
    return absent, absent, absent, absent, absent, absent


type _CudnnSettings = tuple[
    float, int, float, tuple[bool, bool], cudnn_attention.MaskType, int, None, int
]


def _cudnn_nondiff_args(q: Array, is_causal: bool) -> _CudnnSettings:
    """JAX's cuDNN attention settings, as its public wrapper builds them for a boolean mask
    and no dropout: scale, seed, dropout rate, (has bias, has bias gradient), mask type,
    layout, sliding window, cuDNN version."""
    mask_type = cudnn_attention.MaskType.CAUSAL if is_causal else cudnn_attention.MaskType.NO_MASK
    return (
        1 / math.sqrt(q.shape[-1]),
        42,
        0.0,
        (True, False),
        mask_type,
        cudnn_attention.AttentionLayout.BTNH.value,
        None,
        cudnn_attention.check_cudnn_version(),
    )


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def _cudnn_sink_attention(
    q: Array, k: Array, v: Array, bias: Array, sink_logits: Float32[Array, " qh"], is_causal: bool
) -> Array:
    out, _ = _cudnn_sink_attention_fwd(q, k, v, bias, sink_logits, is_causal)
    return out


def _cudnn_sink_attention_fwd(
    q: Array, k: Array, v: Array, bias: Array, sink_logits: Float32[Array, " qh"], is_causal: bool
) -> tuple[Array, tuple[Array, ...]]:
    out, lse = cudnn_attention._dot_product_attention_fwd(
        q,
        k,
        v,
        bias,
        *_cudnn_absent_operands(q),
        *_cudnn_nondiff_args(q, is_causal),
        return_residual=True,
    )
    b, t, n_heads, _ = q.shape
    assert lse.shape == (b, n_heads, t) and lse.dtype == jnp.float32, (lse.shape, lse.dtype)
    sink_lse = jnp.logaddexp(lse, sink_logits[None, :, None])
    keys_share = jnp.exp(lse - sink_lse).transpose(0, 2, 1)[..., None]
    out = (out.astype(jnp.float32) * keys_share).astype(q.dtype)
    return out, (q, k, v, bias, sink_lse, out, keys_share)


def _cudnn_sink_attention_bwd(
    is_causal: bool, residuals: tuple[Array, ...], out_grad: Array
) -> tuple[Array | None, ...]:
    q, k, v, bias, sink_lse, out, keys_share = residuals
    # The backward rule's `is_training` slot receives the cuDNN version, as JAX's own
    # custom VJP passes it, and its `return_residual` is off.
    grads = cudnn_attention._dot_product_attention_bwd_rule(
        *_cudnn_nondiff_args(q, is_causal),
        False,
        (q, k, v, bias, *_cudnn_absent_operands(q), sink_lse, out),
        out_grad,
    )
    out_dot_grad = jnp.sum(out.astype(jnp.float32) * out_grad.astype(jnp.float32), axis=-1)
    sink_grad = -jnp.sum((1 - keys_share[..., 0]) * out_dot_grad, axis=(0, 1))
    return grads[0], grads[1], grads[2], None, sink_grad


_cudnn_sink_attention.defvjp(_cudnn_sink_attention_fwd, _cudnn_sink_attention_bwd)


def _cudnn_attention_with_sink(
    q: Float[Array, "b t qh hd"],
    k: Float[Array, "b s kvh hd"],
    v: Float[Array, "b s kvh hd"],
    mask: Bool[Array, "b 1 t s"],
    sink_logits: Float[Array, " qh"],
    *,
    is_causal: bool,
) -> Float[Array, "b t qh hd"]:
    """cuDNN attention whose every query head also softmaxes over its sink. Inside a shard
    map the logits vary over fewer mesh axes than the queries (they rest replicated across
    batch shards); cast to the queries' axes before the custom VJP, autodiff sums their
    gradient across those shards."""
    bias = cudnn_attention.combine_bias_and_mask(None, mask, q.dtype)
    sink_logits = sink_logits.astype(jnp.float32)
    missing = (
        jax.typeof(q).manual_axis_type.varying - jax.typeof(sink_logits).manual_axis_type.varying
    )
    if missing:
        sink_logits = jax.lax.pcast(sink_logits, tuple(sorted(missing)), to="varying")
    return _cudnn_sink_attention(q, k, v, bias, sink_logits, is_causal)


def _materialized_attention(
    q: Float[Array, "b t qh hd"],
    k: Float[Array, "b s kvh hd"],
    v: Float[Array, "b s kvh hd"],
    mask: Bool[Array, "b 1 t s"],
    sink_logits: Float[Array, " qh"] | None,
    *,
    is_causal: bool,
    cap: float | None,
) -> Float[Array, "b t qh hd"]:
    """Softmax attention over the logits `q·k / √hd`, each `s` capped to `cap · tanh(s / cap)`
    when a cap is given, its logits and softmax in fp32; a sink joins the softmax at its own,
    uncapped logit, with a zero value."""
    group = q.shape[2] // k.shape[2]
    k, v = jnp.repeat(k, group, axis=2), jnp.repeat(v, group, axis=2)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32)
    logits = logits / math.sqrt(q.shape[-1])
    if cap is not None:
        assert cap > 0, cap
        logits = cap * jnp.tanh(logits / cap)
    if is_causal:
        mask = mask & jnp.tril(jnp.ones((q.shape[1], k.shape[1]), bool))[None, None]
    logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
    if sink_logits is None:
        weights = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
        return jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    b, n_heads, t, _ = logits.shape
    sink = jnp.broadcast_to(
        sink_logits.astype(jnp.float32)[None, :, None, None], (b, n_heads, t, 1)
    )
    weights = jax.nn.softmax(jnp.concatenate([sink, logits], axis=-1), axis=-1).astype(v.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", weights[..., 1:], v)


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
        logit_softcap=None,
        sink_logits=None,
    )
    return out.transpose(0, 2, 1, 3)
