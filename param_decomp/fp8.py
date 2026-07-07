"""fp8 matmul for the decomposition GEMMs (`x@V`, `acts@U`).

e4m3 forward / e5m2 backward, with per-row (per-token / per-output-channel) or per-tensor
dynamic scaling. On Hopper/Blackwell XLA fuses the `cast→dot_general` into a cublasLt f8
kernel. A process-level switch (`configure`) selects scope + mode; it is read at TRACE time
(a Python constant during tracing), so it bakes as a static choice — set it once, before the
step is traced (one setting per run).

The dot is N-D in the left operand (`a: [*leading, K]`, `b: [K, N]`, contract `a`'s last
with `b`'s first) so the batch-sharded leading axes are never reshaped (no reshard).

Per-row scaling quantizes along the NON-contracted axes (TE-style): an outlier in one row
no longer sets the whole-tensor scale. The backward re-quantizes the operands along the
backward's contracted axis (the "transposed re-quant"); residuals are the bf16 operands.

`mxfp8` instead routes through `jax.nn.scaled_dot_general` (cuDNN block-scaled microscaling,
32-element blocks, built-in backward — no custom_vjp). Hardware-accelerated ONLY on
Blackwell (B200); Hopper has no kernel for it (orders of magnitude slower) and CPU has no
lowering at all, so it is B200-opt-in, never a default.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import DTypeLike
from jaxtyping import Array

E4M3 = jnp.float8_e4m3fn
E5M2 = jnp.float8_e5m2
_MAX_E4M3 = 448.0
_MAX_E5M2 = 57344.0
_EPS = 1e-12


@dataclass(frozen=True)
class _Settings:
    components: bool
    mode: str  # "per_tensor" | "per_row" | "mxfp8"


_settings = _Settings(components=False, mode="per_row")


def configure(scope: str, mode: str) -> None:
    """Set the process-level fp8 switch. `scope`: "off" | "components". `mode`:
    "per_tensor" | "per_row" | "mxfp8". Call before the train step is first traced."""
    assert scope in ("off", "components"), scope
    assert mode in ("per_tensor", "per_row", "mxfp8"), mode
    global _settings
    _settings = _Settings(components=(scope == "components"), mode=mode)


def components_enabled() -> bool:
    return _settings.components


def settings_repr() -> str:
    return f"fp8(components={_settings.components}, mode={_settings.mode!r})"


def _amax(x: Array, axes: int | tuple[int, ...]) -> Array:
    return jnp.max(jnp.abs(x), axis=axes, keepdims=True).astype(jnp.float32)


def _amax_scalar(x: Array) -> Array:
    return jnp.max(jnp.abs(x)).astype(jnp.float32)


def _cast(x: Array, scale: Array, dt: DTypeLike) -> Array:
    return (x.astype(jnp.float32) * scale).astype(dt)


# ───────────────────────── per-tensor (scalar scales) ─────────────────────────


@jax.custom_vjp
def _dot_per_tensor(a: Array, b: Array) -> Array:
    ca = a.ndim - 1
    sa = _MAX_E4M3 / (_amax_scalar(a) + _EPS)
    sb = _MAX_E4M3 / (_amax_scalar(b) + _EPS)
    out = lax.dot_general(
        _cast(a, sa, E4M3), _cast(b, sb, E4M3), (((ca,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )  # fmt: skip
    return (out / (sa * sb)).astype(a.dtype)


def _dot_per_tensor_fwd(a: Array, b: Array) -> tuple[Array, tuple[Array, Array]]:
    return _dot_per_tensor(a, b), (a, b)


def _dot_per_tensor_bwd(res: tuple[Array, Array], g: Array) -> tuple[Array, Array]:
    a, b = res
    ca = a.ndim - 1
    lead = tuple(range(ca))
    sa = _MAX_E4M3 / (_amax_scalar(a) + _EPS)
    sb = _MAX_E4M3 / (_amax_scalar(b) + _EPS)
    sg = _MAX_E5M2 / (_amax_scalar(g) + _EPS)
    qa, qb, qg = _cast(a, sa, E4M3), _cast(b, sb, E4M3), _cast(g, sg, E5M2)
    # grad_a = g @ b^T  (contract N: g last, b axis1) -> [*lead, K]
    ga = lax.dot_general(
        qg, qb, (((g.ndim - 1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )
    ga = (ga / (sg * sb)).astype(a.dtype)
    # grad_b = a^T @ g  (contract leading) -> [K, N]
    gb = lax.dot_general(qa, qg, ((lead, lead), ((), ())), preferred_element_type=jnp.float32)
    gb = (gb / (sa * sg)).astype(b.dtype)
    return ga, gb


_dot_per_tensor.defvjp(_dot_per_tensor_fwd, _dot_per_tensor_bwd)


# ───────────────────────── per-row (per non-contracted slice) ─────────────────────────


@jax.custom_vjp
def _dot_per_row(a: Array, b: Array) -> Array:
    ca = a.ndim - 1
    sa = _MAX_E4M3 / (_amax(a, ca) + _EPS)  # [*lead, 1]
    sb = _MAX_E4M3 / (_amax(b, 0) + _EPS)  # [1, N]
    out = lax.dot_general(
        _cast(a, sa, E4M3), _cast(b, sb, E4M3), (((ca,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )  # fmt: skip
    return (out / (sa * sb)).astype(a.dtype)


def _dot_per_row_fwd(a: Array, b: Array) -> tuple[Array, tuple[Array, Array]]:
    return _dot_per_row(a, b), (a, b)


def _dot_per_row_bwd(res: tuple[Array, Array], g: Array) -> tuple[Array, Array]:
    a, b = res
    ca = a.ndim - 1
    lead = tuple(range(ca))
    K = a.shape[-1]
    N = b.shape[-1]
    # grad_a = g @ b^T  (contract N) -> [*lead, K]
    sg = _MAX_E5M2 / (_amax(g, g.ndim - 1) + _EPS)  # [*lead, 1]
    sb2 = _MAX_E4M3 / (_amax(b, 1) + _EPS)  # [K, 1]
    ga = lax.dot_general(
        _cast(g, sg, E5M2), _cast(b, sb2, E4M3), (((g.ndim - 1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )  # fmt: skip
    ga = (ga * (1.0 / sg) * (1.0 / sb2).reshape(K)).astype(a.dtype)
    # grad_b = a^T @ g  (contract leading) -> [K, N]
    sa2 = _MAX_E4M3 / (_amax(a, lead) + _EPS)  # [1.., 1, K]
    sg2 = _MAX_E5M2 / (_amax(g, lead) + _EPS)  # [1.., 1, N]
    gb = lax.dot_general(
        _cast(a, sa2, E4M3), _cast(g, sg2, E5M2), ((lead, lead), ((), ())),
        preferred_element_type=jnp.float32,
    )  # fmt: skip
    gb = (gb * (1.0 / sa2).reshape(K, 1) * (1.0 / sg2).reshape(1, N)).astype(b.dtype)
    return ga, gb


_dot_per_row.defvjp(_dot_per_row_fwd, _dot_per_row_bwd)


# ───────────────────────── mxfp8 (Blackwell block-scaled) ─────────────────────────


def _dot_mxfp8(a: Array, b: Array) -> Array:
    ca = a.ndim - 1
    cfg = jax.nn.get_scaled_dot_general_config("mxfp8")
    out = jax.nn.scaled_dot_general(
        a, b, (((ca,), (0,)), ((), ())),
        preferred_element_type=jnp.bfloat16, configs=[cfg, cfg, cfg],
    )  # fmt: skip
    return out.astype(a.dtype)


def matmul(a: Array, b: Array) -> Array:
    """fp8 `a @ b` (contract `a`'s last axis with `b`'s first), per the configured mode."""
    match _settings.mode:
        case "per_row":
            return _dot_per_row(a, b)
        case "per_tensor":
            return _dot_per_tensor(a, b)
        case "mxfp8":
            return _dot_mxfp8(a, b)
        case _:
            raise AssertionError(_settings.mode)
