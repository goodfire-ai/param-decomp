"""CI-fn interface + the chunkwise-transformer impl.

A CI fn maps named INPUT taps to a `CI` bundle over OUTPUT sites:
`dict[InputTap, Array] -> CI` (preactivations + the two squashings). The input keyspace (opaque
tap keys — the lab authors them, the target resolves and captures them) is
independent of the output keyspace (the decomposition sites). The output sites MUST
partition the model's sites — every site needs exactly one CI value — asserted at
construction. Core treats both keyspaces as OPAQUE dict keys: look up inputs, scatter
outputs, validate the partition. It never parses a key.

The SAME preactivations are squashed two ways (SPEC S5/S6) in ONE place (`CI.from_preactivations`):
`lower` (clip[0,1], leaky-below) feeds recon / PPGD / routing masks; `upper`
(leaky-above-1) feeds importance-minimality. `preactivations` is kept too — the CI histograms /
heatmaps plot the pre-squash view. Params are fp32 masters (SPEC N1); the trainer casts
for bf16 compute.

The chunkwise-transformer (`ChunkwiseTransformerCIFn`) is the LM impl: each chunk reads
one or more residual taps (RMS-normed per tap, then concatenated) and emits CI for the
matrix sites it covers, via an independent pre-norm bidirectional-RoPE transformer. The
per-chunk transformers are stacked along a leading `n_chunks` axis and run under a
`jax.lax.scan` over that axis (so the chunk iteration lowers as a loop — one chunk's FSDP
weight gather live at a time, not all `n_chunks` hoisted into the flat entry computation).
The positionless toys use the MLP impls below (`LayerwiseMLPCIFn` /
`GlobalMLPCIFn`); every impl satisfies the same `CIFn` protocol and is equally core — the
architectures differ by domain (sequence vs positionless), not by status. A POSITIONED target
that cannot afford attention over its positions runs the same chunkwise impl at `n_blocks=0`,
which is position-local by construction (see `ChunkwiseTransformerCIArch`).
"""

import math
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int, PRNGKeyArray

from param_decomp.core.axes import Axes, MeshAxis, SemanticAxis
from param_decomp.core.components import SiteSpec, activation_axes
from param_decomp.core.linear_plan import placed_linear, value_mesh
from param_decomp.core.model import CaptureKeys
from param_decomp.core.placement import (
    CIFnPlacement,
    CIWeightFamily,
    CIWeightPlacement,
    PlacedRule,
    PlacementRules,
    materialize_reduced_weights,
    ns_staging_sharding,
)
from param_decomp.core.precision import COMPUTE_DT, cast_floating
from param_decomp.vendored_jax.llama import (
    apply_rope,
    attn_implementation,
    rms_norm,
    rope_cos_sin,
)

CI_FN_RMS_EPS = float(jnp.finfo(jnp.float32).eps)
"""Matches torch's `F.rms_norm` default eps (`finfo(fp32).eps` ~1.19e-7); RMS upcasts to
fp32 internally, so this is the dtype that governs (SPEC S4)."""

SiteDict = dict[str, Float[Array, "*leading C"]]
"""Per-output-site tensor keyed by OUTPUT site name."""

_StoredCIAxes = tuple[SemanticAxis, SemanticAxis, SemanticAxis]
# Attention keeps per-projection axes: q/o carry the query head axis, k/v the K/V head
# axis (narrower under GQA). Each name covers both spellings of that dimension — the
# flat `n * head_dim` projection width and the head COUNT of its split view.
CI_ATTN_Q_AXES: _StoredCIAxes = ("stack", "q_head", "d_model")
CI_ATTN_KV_AXES: _StoredCIAxes = ("stack", "kv_head", "d_model")
CI_ATTN_OUT_AXES: _StoredCIAxes = ("stack", "d_model", "q_head")
CI_FFN_IN_AXES: _StoredCIAxes = ("stack", "d_model", "ffn_hidden")
CI_FFN_OUT_AXES: _StoredCIAxes = ("stack", "ffn_hidden", "d_model")
CI_INPUT_AXES: _StoredCIAxes = ("stack", "input", "d_model")
CI_OUTPUT_AXES: _StoredCIAxes = ("stack", "d_model", "C")


def _vector_sharding(
    row: PlacedRule, axes: tuple[SemanticAxis, SemanticAxis], shape: tuple[int, ...]
) -> NamedSharding:
    row.validate_shape(axes, shape)
    return row.sharding_for(axes)


# ----------------------------- squashings (SPEC S5/S6) -----------------------------


@jax.custom_vjp
def lower_leaky_hard_sigmoid(x: Array) -> Array:
    return jnp.clip(x, 0.0, 1.0)


def _lhs_f(x: Array) -> tuple[Array, Array]:
    return jnp.clip(x, 0.0, 1.0), x


def _lhs_b(x: Array, g: Array) -> tuple[Array]:
    leak = jnp.where(g < 0, 0.01 * g, 0.0)
    return (jnp.where(x <= 0, leak, jnp.where(x <= 1, g, 0.0)),)


lower_leaky_hard_sigmoid.defvjp(_lhs_f, _lhs_b)


def upper_leaky_hard_sigmoid(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """`x>1 ? 1+alpha*(x-1) : clamp(x,0,1)` — ordinary autodiff of this expression
    (torch builds its backward the same way; only the lower squashing is a custom VJP)."""
    alpha = 0.01
    return jnp.where(x > 1, 1 + alpha * (x - 1), jnp.clip(x, 0.0, 1.0))


# ----------------------------- the CI bundle + protocol -----------------------------


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CI:
    """The CI fn output: raw preactivations + both squashings, all keyed by output site. `preactivations`
    is kept (a consumed view — the histograms / heatmaps plot pre-squash). The squashing
    lives only in `from_preactivations`, so no impl re-triplicates it."""

    preactivations: SiteDict
    lower: SiteDict
    upper: SiteDict

    @staticmethod
    def from_preactivations(preactivations: SiteDict) -> "CI":
        return CI(
            preactivations=preactivations,
            lower={k: lower_leaky_hard_sigmoid(v) for k, v in preactivations.items()},
            upper={k: upper_leaky_hard_sigmoid(v) for k, v in preactivations.items()},
        )


@runtime_checkable
class CIFn(Protocol):
    """`dict[InputTap, Array] -> CI`. `output_names` partition the model sites (asserted at
    construction); input taps are unconstrained. `has_position_axis` must equal the
    paired `DecomposedModel.has_position_axis` (asserted at trainer construction)."""

    @property
    def capture_keys(self) -> CaptureKeys: ...

    output_names: tuple[str, ...]
    has_position_axis: bool

    def __call__(
        self, taps: dict[str, Array], *, remat: bool, placement: CIFnPlacement | None
    ) -> CI: ...


class PlacedCIFn(eqx.Module):
    """A CI fn paired with ITS placement — resolved exactly once, at run assembly
    (`resolve_ci_placement`), so no downstream code ever holds an unresolved
    (fn, placement rows) combination. `placement is None` means this fn runs unplaced —
    a decided state, not an omission. The fn's arrays are pytree children (traced,
    differentiated, cast); the placement is static and rides the treedef, so the pair
    threads through jit/vjp as one value and cannot desync."""

    fn: CIFn
    placement: CIFnPlacement | None = eqx.field(static=True)


def evaluate_ci(placed_ci_fn: PlacedCIFn, taps: dict[str, Array], *, remat: bool) -> CI:
    """Run fp32-master CI parameters and their captured inputs in compute precision."""
    return evaluate_compute_ci(materialize_ci_compute_weights(placed_ci_fn), taps, remat=remat)


def evaluate_compute_ci(compute_ci_fn: PlacedCIFn, taps: dict[str, Array], *, remat: bool) -> CI:
    return compute_ci_fn.fn(
        cast_floating(taps, COMPUTE_DT), remat=remat, placement=compute_ci_fn.placement
    )


def evaluate_padded_ci(
    placed_ci_fn: PlacedCIFn,
    taps: dict[str, Array],
    valid_token_count: Int[Array, ""],
    *,
    remat: bool,
) -> CI:
    """Evaluate a chunkwise CI fn without allowing right-padding into attention."""
    compute_ci_fn = materialize_ci_compute_weights(placed_ci_fn)
    if not isinstance(compute_ci_fn.fn, ChunkwiseTransformerCIFn):
        raise TypeError(
            "masked right-padding requires a chunkwise-transformer CI function; "
            f"got {type(compute_ci_fn.fn)}"
        )
    return compute_ci_fn.fn.evaluate_padded(
        cast_floating(taps, COMPUTE_DT),
        valid_token_count,
        remat=remat,
        placement=compute_ci_fn.placement,
    )


def materialize_ci_compute_weights(placed_ci_fn: PlacedCIFn) -> PlacedCIFn:
    compute_ci_fn = cast_floating(placed_ci_fn.fn, COMPUTE_DT)
    if isinstance(compute_ci_fn, ChunkwiseTransformerCIFn):
        chunks = _reconstruct_ci_compute_weights(compute_ci_fn.chunks, placed_ci_fn.placement)
        return PlacedCIFn(
            fn=eqx.tree_at(lambda f: f.chunks, compute_ci_fn, chunks),
            placement=placed_ci_fn.placement,
        )
    # Bypass protection: `resolve_ci_placement` never pairs an MLP fn with rows, so a
    # placed non-chunkwise bundle can only be a hand-built mispairing.
    assert placed_ci_fn.placement is None, (
        f"CI placement rows require a chunkwise transformer, got {type(placed_ci_fn.fn)}"
    )
    return PlacedCIFn(fn=compute_ci_fn, placement=None)


def ci_preactivations(placed_ci_fn: PlacedCIFn, taps: dict[str, Array], *, remat: bool) -> SiteDict:
    """Evaluate CI in compute precision and expose fp32 preactivations for metric
    reductions — through the compute lifecycle (materialized residents), which a
    placed run's persistence-layout weights require."""
    compute_ci_fn = materialize_ci_compute_weights(placed_ci_fn)
    preactivations = evaluate_compute_ci(compute_ci_fn, taps, remat=remat).preactivations
    return cast_floating(preactivations, jnp.float32)


# ----------------------------- transformer building blocks -----------------------------


def _weightless_rms_norm(x: Array, eps: float) -> Array:
    return rms_norm(x, jnp.ones((x.shape[-1],), x.dtype), eps)


def _constrain_ci_activation(
    x: Array, placement: CIFnPlacement | None, feature_axis: SemanticAxis
) -> Array:
    if placement is None:
        return x
    axes = activation_axes(x.ndim, feature_axis)
    placement.activations.validate_shape(axes, x.shape)
    return jax.sharding.reshard(x, placement.activations.sharding_for(axes))


def _rms_norm_maybe_scaled(x: Array, scale: Array | None, eps: float) -> Array:
    """`scale is None` is the weightless norm — `ones` in x's dtype, i.e. today's numerics
    exactly (and no bf16→fp32 promotion, which an fp32 scale leaf would cause)."""
    if scale is None:
        return _weightless_rms_norm(x, eps)
    return rms_norm(x, scale, eps)


def _ci_linear(
    x: Array,
    weight: Array,
    placement: CIFnPlacement | None,
    family: CIWeightFamily,
    stored_axes: tuple[SemanticAxis, SemanticAxis],
    *,
    transposed: bool,
) -> Array:
    assert weight.ndim == 2, weight.shape
    operand = jnp.swapaxes(weight, -1, -2) if transposed else weight
    if placement is None:
        return x @ operand
    plan = placement.linear_plan(family, stored_axes, x.ndim, transposed=transposed)
    return placed_linear(x, operand, plan)


@dataclass(frozen=True)
class MHACIAttention:
    """Every query head carries its own K/V head."""

    n_heads: int

    @property
    def n_kv_heads(self) -> int:
        return self.n_heads


@dataclass(frozen=True)
class GQACIAttention:
    """`n_heads // n_kv_heads` query heads share each K/V head, so `wk`/`wv` narrow to
    `n_kv_heads * head_dim`. head_dim, the RoPE tables, `wq`/`wo` and every sharding are
    identical to MHA — only the K/V projections change."""

    n_heads: int
    n_kv_heads: int

    def __post_init__(self) -> None:
        assert self.n_heads % self.n_kv_heads == 0, (
            "n_heads must be divisible by n_kv_heads (each K/V head serves an equal group "
            f"of query heads): {self.n_heads} % {self.n_kv_heads}"
        )
        assert self.n_kv_heads < self.n_heads, (
            f"n_kv_heads == n_heads ({self.n_heads}) is MHA — use MHACIAttention rather than "
            "a degenerate GQA"
        )


CIAttention = MHACIAttention | GQACIAttention
"""The CI transformer's attention. Both arms answer `n_heads` and `n_kv_heads`, so the call
site never dispatches — but MHA derives its K/V count from the TYPE instead of leaving
`n_kv_heads == n_heads` as a convention a reader has to know, and cannot carry an explicit
one. GQA's grouping invariant is checked at construction, not at init."""


CIFfnKind = Literal["gelu", "swiglu"]
"""`gelu`: `Linear+b → GELU → Linear+b`. `swiglu`: a second projection gates the first —
`silu(h@wg + bg) * (h@w1 + b1) → Linear+b`. SwiGLU is a THIRD matrix, so it grows the MLP
~50% at a fixed `ffn_hidden`; iso-param means setting `ffn_hidden` to ~2/3. Nothing here
rescales it — the width is the config author's to state."""


class CIBlock(eqx.Module):
    """Pre-norm block: RMSNorm → bidirectional RoPE attention → residual;
    RMSNorm → FFN (`gelu` or `swiglu`) → residual.

    `attention` is the resolved variant: under `GQACIAttention` the K/V projections narrow to
    `n_kv_heads * head_dim` and `jax.nn.dot_product_attention` broadcasts each K/V head over
    its group of query heads. Both arms answer `n_kv_heads`, so nothing here dispatches.

    `gate is None` ⟺ the GELU FFN; a present gate ⟺ SwiGLU. The gate's `(w, b)` ride in one
    optional tuple because they vary together, and its presence IS the FFN discriminator — so
    there's no separate tag to desync from the params.

    `norm_scales is None` ⟺ the weightless norms (today's behaviour); present ⟺ learned
    per-channel scales, `(pre-attn, pre-MLP)`."""

    wq: Array
    wk: Array
    wv: Array
    wo: Array
    w1: Array
    b1: Array
    w2: Array
    b2: Array
    gate: tuple[Array, Array] | None
    norm_scales: tuple[Array, Array] | None
    attention: CIAttention = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def shardings(self, mesh: Mesh, placement: CIFnPlacement) -> "CIBlock":
        """Place stacked attention and FFN parameters at their persistence rows.

        Every large weight's sharding derives from its semantic axes and the placement
        table."""
        attention = placement.attention.optimizer_state
        ffn = placement.ffn.optimizer_state
        attention.validate_shape(CI_ATTN_Q_AXES, self.wq.shape)
        attention.validate_shape(CI_ATTN_KV_AXES, self.wk.shape)
        attention.validate_shape(CI_ATTN_KV_AXES, self.wv.shape)
        attention.validate_shape(CI_ATTN_OUT_AXES, self.wo.shape)
        ffn.validate_shape(CI_FFN_IN_AXES, self.w1.shape)
        ffn.validate_shape(CI_FFN_OUT_AXES, self.w2.shape)
        attn_q = NamedSharding(mesh, attention.spec_for(CI_ATTN_Q_AXES))
        attn_kv = NamedSharding(mesh, attention.spec_for(CI_ATTN_KV_AXES))
        attn_out = NamedSharding(mesh, attention.spec_for(CI_ATTN_OUT_AXES))
        ffn_in = NamedSharding(mesh, ffn.spec_for(CI_FFN_IN_AXES))
        ffn_out = NamedSharding(mesh, ffn.spec_for(CI_FFN_OUT_AXES))
        vectors = placement.vectors
        b1 = _vector_sharding(vectors, ("stack", "ffn_hidden"), self.b1.shape)
        b2 = _vector_sharding(vectors, ("stack", "d_model"), self.b2.shape)
        placed = eqx.tree_at(
            lambda b: (b.wq, b.wk, b.wv, b.wo, b.w1, b.b1, b.w2, b.b2),
            self,
            (attn_q, attn_kv, attn_kv, attn_out, ffn_in, b1, ffn_out, b2),
        )
        if self.gate is not None:
            # The swiglu gate is a second `[nc, d_model, ffn_hidden]` up-proj: same
            # Megatron-on-ffn_hidden placement as w1, same ÷N divisibility requirement.
            placement.ffn.optimizer_state.validate_shape(CI_FFN_IN_AXES, self.gate[0].shape)
            gate_bias = _vector_sharding(vectors, ("stack", "ffn_hidden"), self.gate[1].shape)
            placed = eqx.tree_at(lambda b: b.gate, placed, (ffn_in, gate_bias))
        if self.norm_scales is not None:
            norm = _vector_sharding(vectors, ("stack", "d_model"), self.norm_scales[0].shape)
            placed = eqx.tree_at(lambda b: b.norm_scales, placed, (norm, norm))
        return placed

    def __call__(
        self,
        x: Float[Array, "b t d"],
        inv_freq: Array,
        *,
        placement: CIFnPlacement | None,
        valid_token_count: Int[Array, ""] | None,
    ) -> Array:
        t = x.shape[1]
        attn_scale, mlp_scale = (None, None) if self.norm_scales is None else self.norm_scales
        h = _rms_norm_maybe_scaled(x, attn_scale, self.eps)

        def heads(  # [b, t, d] -> [b, nh, t, hd]  (RoPE layout)
            w: Array, n_head: int, stored_axes: _StoredCIAxes
        ) -> Array:
            proj = _ci_linear(
                h,
                w,
                placement,
                "attention",
                stored_axes[1:],
                transposed=True,
            )
            if not value_mesh(proj).empty:
                # Type the head split: the flat head dim's assignment (tp) lands on the
                # HEAD axis — an untyped reshape may park it on head_dim, which the
                # attention contraction then cannot resolve. Both head counts tile their
                # assignments by construction (`resolve_ci_placement` refuses otherwise),
                # so the assignment carries through unconditionally.
                mesh = value_mesh(proj)
                proj_spec = jax.typeof(proj).sharding.spec
                return jax.lax.reshape(
                    proj,
                    (*proj.shape[:2], n_head, proj.shape[2] // n_head),
                    out_sharding=NamedSharding(mesh, P(*proj_spec[:2], proj_spec[2], None)),
                ).transpose(0, 2, 1, 3)
            return einops.rearrange(proj, "b t (nh hd) -> b nh t hd", nh=n_head)

        q = heads(self.wq, self.attention.n_heads, CI_ATTN_Q_AXES)
        kv = self.attention.n_kv_heads
        k, v = heads(self.wk, kv, CI_ATTN_KV_AXES), heads(self.wv, kv, CI_ATTN_KV_AXES)
        cos, sin = rope_cos_sin(inv_freq, t, x.dtype)
        q, k = apply_rope(q, k, cos, sin)  # cos/sin broadcast over the head axis: any count
        qt, kt, vt = (einops.rearrange(a, "b nh t hd -> b t nh hd") for a in (q, k, v))
        # cuDNN flash on GPU (its partitioner requires device-local heads — true here, no
        # head-sharding); XLA elsewhere (CPU tests have no cuDNN). Bidirectional. Fewer K/V
        # heads than query heads is GQA, grouped natively by dot_product_attention.
        impl = attn_implementation("auto", jax.default_backend(), qt.dtype, t)
        sequence_lengths = (
            None
            if valid_token_count is None
            else jnp.full((x.shape[0],), valid_token_count, dtype=jnp.int32)
        )
        y = jax.nn.dot_product_attention(
            qt,
            kt,
            vt,
            is_causal=False,
            query_seq_lengths=sequence_lengths,
            key_value_seq_lengths=sequence_lengths,
            implementation=impl,
        )
        x = x + _ci_linear(
            einops.rearrange(y, "b t nh hd -> b t (nh hd)"),
            self.wo,
            placement,
            "attention",
            CI_ATTN_OUT_AXES[1:],
            transposed=True,
        )
        h = _rms_norm_maybe_scaled(x, mlp_scale, self.eps)
        up = (
            _ci_linear(h, self.w1, placement, "ffn", CI_FFN_IN_AXES[1:], transposed=False) + self.b1
        )
        if self.gate is None:
            hidden = jax.nn.gelu(up, approximate=False)
        else:
            w_gate, b_gate = self.gate
            gate = (
                _ci_linear(h, w_gate, placement, "ffn", CI_FFN_IN_AXES[1:], transposed=False)
                + b_gate
            )
            hidden = jax.nn.silu(gate) * up
        return (
            x
            + _ci_linear(
                hidden,
                self.w2,
                placement,
                "ffn",
                CI_FFN_OUT_AXES[1:],
                transposed=False,
            )
            + self.b2
        )


# ----------------------------- chunkwise transformer -----------------------------


@dataclass(frozen=True)
class Chunk:
    """One resolved chunk: the input taps to concatenate → CI for a group of output sites.
    Authored lab-side (from `blocks_per_chunk` + topology); core treats both keyspaces as
    opaque keys. `input_taps` may name several residual taps (e.g. the residual entering the
    chunk plus earlier read points) — RMS-normed per tap and concatenated as the input."""

    input_taps: tuple[str, ...]
    output_sites: tuple[str, ...]


@dataclass(frozen=True)
class _ChunkMeta:
    """Per-chunk static routing, index-aligned with the stacked `chunks` leading axis."""

    input_taps: tuple[str, ...]  # taps to RMS-norm + concatenate as this chunk's input
    output_sites: tuple[str, ...]  # output sites this chunk scores, in C-per-slot order


@dataclass(frozen=True)
class ChunkwiseTransformerCIArch:
    """Resolved chunkwise-transformer arch: explicit chunks + the CI transformer's dims.

    `input_dim` is the per-chunk concatenated input width — a plain linear-layer input
    dimension. The lab computes it from the taps it authored (their widths summed); core
    stays agnostic to what the taps mean, so no transformer concept (residual width) leaks
    in. All chunks share one `input_dim` (the vmap homogeneity requirement).

    `attention` is the resolved variant (the schema's `attention` union, translated).

    `n_blocks=0` degenerates to `RMS-normed taps → in_proj → per-site output heads`: the FFN
    lives inside the block alongside attention, so dropping blocks leaves an affine map on the
    NORMALIZED tap — a direction-only probe, with no learned nonlinearity, no hidden layer, and
    no sensitivity to tap magnitude at all. It is position-local (blocks are the only thing
    reading ACROSS positions) and it runs, so it serves as a cheap baseline, but a positioned
    target that wants a real per-position CI fn wants `LayerwiseMLPCIArch(has_position_axis=True)`.
    Pinned by `param_decomp/tests/core/test_ci_fn_zero_blocks.py` (locality) and
    `param_decomp/tests/core/test_ci_fn_positioned_mlp.py` (the magnitude contrast)."""

    chunks: tuple[Chunk, ...]
    input_dim: int
    d_model: int
    n_blocks: int
    attention: CIAttention
    ffn_hidden: int
    ffn_kind: CIFfnKind
    learned_norm_scale: bool

    @property
    def capture_keys(self) -> CaptureKeys:
        """The activation taps consumed by any chunk."""
        return frozenset(tap for chunk in self.chunks for tap in chunk.input_taps)


class ChunkTransformer(eqx.Module):
    """ONE chunk: its (already RMS-normed, concatenated) input `[*leading, total_d_in]` →
    a TUPLE of per-output-site preactivations (`out` of `[*leading, C_j]` per site-slot j), via
    in_proj → RoPE blocks → one output head PER site-slot.

    One head per site-slot (`out_ws[j] [d_model, C_j]` / `out_bs[j] [C_j]`) instead of a
    single glued `[d_model, ΣC]` head: each head's output IS that site's CI, born already
    split per site (matching `x@V` / the mask, SPEC §4.1 `site_out`). Under pure HSDP the C
    axis is replicated (not sharded), so the split is a pure layout convenience; it was
    load-bearing under the prior TP layout (a tp-sharded glued ΣC axis sliced mid-site),
    and is kept harmlessly.

    In the bundle every array below carries a leading `n_chunks` axis and the module is
    run under `jax.lax.scan` over that axis, so this body is written for a single chunk."""

    in_proj_w: Float[Array, "total_d_in d_model"]
    in_proj_b: Float[Array, " d_model"]
    blocks: list[CIBlock]
    out_ws: tuple[Float[Array, "d_model _C"], ...]
    out_bs: tuple[Float[Array, " _C"], ...]

    def shardings(self, mesh: Mesh, placement: CIFnPlacement) -> "ChunkTransformer":
        """Place the complete stacked CI transformer from its semantic placement rows."""
        input_row = placement.input.optimizer_state
        output_row = placement.output.optimizer_state
        input_row.validate_shape(CI_INPUT_AXES, self.in_proj_w.shape)
        for w in self.out_ws:
            output_row.validate_shape(CI_OUTPUT_AXES, w.shape)
        in_proj_sh = NamedSharding(mesh, input_row.spec_for(CI_INPUT_AXES))
        out_ws_sh = NamedSharding(mesh, output_row.spec_for(CI_OUTPUT_AXES))
        vectors = placement.vectors
        in_proj_b = _vector_sharding(vectors, ("stack", "d_model"), self.in_proj_b.shape)
        out_bs = tuple(
            _vector_sharding(vectors, ("stack", "C"), bias.shape) for bias in self.out_bs
        )
        return eqx.tree_at(
            lambda ct: (ct.in_proj_w, ct.in_proj_b, ct.blocks, ct.out_ws, ct.out_bs),
            self,
            (
                in_proj_sh,
                in_proj_b,
                [b.shardings(mesh, placement) for b in self.blocks],
                tuple(out_ws_sh for _ in self.out_ws),
                out_bs,
            ),
        )

    def __call__(
        self,
        x: Float[Array, "*leading total_d_in"],
        inv_freq: Array,
        *,
        placement: CIFnPlacement | None,
        valid_token_count: Int[Array, ""] | None,
    ) -> tuple[Float[Array, "*leading _C"], ...]:
        x = (
            _ci_linear(
                x,
                self.in_proj_w,
                placement,
                "input",
                CI_INPUT_AXES[1:],
                transposed=False,
            )
            + self.in_proj_b
        )
        for block in self.blocks:
            x = block(
                x,
                inv_freq,
                placement=placement,
                valid_token_count=valid_token_count,
            )
        return tuple(
            _ci_linear(
                x,
                w,
                placement,
                "output",
                CI_OUTPUT_AXES[1:],
                transposed=False,
            )
            + b
            for w, b in zip(self.out_ws, self.out_bs, strict=True)
        )


def ns_compute_shardings(
    ci_fn: "ChunkwiseTransformerCIFn", mesh: Mesh, placement: CIFnPlacement
) -> "ChunkwiseTransformerCIFn":
    """Per-leaf muon-NS staging shardings for the chunkwise stack: every muon-labeled
    (3D) weight position carries its family's `ns_compute` waypoint row verbatim
    (`ns_staging_sharding`); every other position rides through untouched (the muon
    partition masks them out). Shaped for the stacked-muon `waypoints` callable, so
    `ci_fn` may be the muon-MASKED update tree — only positions selected below are
    read."""

    def staging(weights: CIWeightPlacement) -> NamedSharding:
        return ns_staging_sharding(weights.ns_compute, mesh)

    attention, ffn = staging(placement.attention), staging(placement.ffn)

    def where(f: "ChunkwiseTransformerCIFn") -> tuple[Array, ...]:
        locations: list[Array] = [f.chunks.in_proj_w, *f.chunks.out_ws]
        for block in f.chunks.blocks:
            locations += [block.wq, block.wk, block.wv, block.wo, block.w1, block.w2]
            if block.gate is not None:
                locations.append(block.gate[0])
        return tuple(locations)

    values: list[NamedSharding] = [staging(placement.input)]
    values += [staging(placement.output)] * len(ci_fn.chunks.out_ws)
    for block in ci_fn.chunks.blocks:
        values += [attention, attention, attention, attention, ffn, ffn]
        if block.gate is not None:
            values.append(ffn)
    return eqx.tree_at(where, ci_fn, tuple(values))


def _reconstruct_ci_compute_weights(
    chunks: "ChunkTransformer", placement: CIFnPlacement | None
) -> "ChunkTransformer":
    """Transition persistent CI weights to their declared resident-compute rows.

    Equal rows are communication-free. Staged weights may remain FSDP-sharded so operand
    materialization occurs per block inside the scan. No-op off-mesh."""
    if jax.sharding.get_abstract_mesh().empty:
        return chunks
    assert placement is not None, "on-mesh CI compute-weight materialization requires placement"
    attention, ffn, in_proj, output = (
        placement.attention,
        placement.ffn,
        placement.input,
        placement.output,
    )

    def pin(x: Array, weights: CIWeightPlacement, axes: Axes) -> Array:
        # x arrives already compute-dtype (the whole fn is cast before reconstruction), so
        # the gather's collective moves bf16 bytes, not the f32 master's —
        # materialize_reduced_weights barriers the cast ahead of the collective.
        return materialize_reduced_weights(
            x,
            source=weights.optimizer_state,
            destination=weights.compute_weights,
            axes=axes,
        )

    def pin_block(blk: CIBlock) -> CIBlock:
        # The vector leaves (biases, the swiglu gate bias, norm scales) have no separate
        # compute row: they stay at their persistence layout (the ci_fn/vectors row).
        pinned = eqx.tree_at(
            lambda b: (b.wq, b.wk, b.wv, b.wo, b.w1, b.w2),
            blk,
            (
                pin(blk.wq, attention, CI_ATTN_Q_AXES),
                pin(blk.wk, attention, CI_ATTN_KV_AXES),
                pin(blk.wv, attention, CI_ATTN_KV_AXES),
                pin(blk.wo, attention, CI_ATTN_OUT_AXES),
                pin(blk.w1, ffn, CI_FFN_IN_AXES),
                pin(blk.w2, ffn, CI_FFN_OUT_AXES),
            ),
        )
        if blk.gate is not None:
            pinned = eqx.tree_at(lambda b: b.gate[0], pinned, pin(blk.gate[0], ffn, CI_FFN_IN_AXES))
        return pinned

    pinned_blocks = [pin_block(blk) for blk in chunks.blocks]
    return eqx.tree_at(
        lambda ct: (ct.in_proj_w, ct.blocks, ct.out_ws),
        chunks,
        (
            pin(chunks.in_proj_w, in_proj, CI_INPUT_AXES),
            pinned_blocks,
            tuple(pin(w, output, CI_OUTPUT_AXES) for w in chunks.out_ws),
        ),
    )


class ChunkwiseTransformerCIFn(eqx.Module):
    """Per-chunk `ChunkTransformer`s stacked along a leading `n_chunks` axis, iterated by a
    `jax.lax.scan` over that axis (lowers as a loop so one chunk's FSDP weight gather is live
    at a time, not all `n_chunks` at once). Each chunk's input is its `chunk_input_taps`
    RMS-normed per tap and concatenated. Requires homogeneous chunks (equal total input width
    and an identical per-slot C tuple — same C-per-output-site ORDER) so the stack, including
    the per-slot output heads, is rectangular — asserted at init."""

    chunks: ChunkTransformer  # arrays stacked along leading n_chunks
    inv_freq: Array  # shared across chunks (RoPE buffer); NOT mapped

    capture_keys: CaptureKeys = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)  # all sites, flat
    chunk_meta: tuple[_ChunkMeta, ...] = eqx.field(static=True)  # per-chunk routing
    eps: float = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)

    def shardings(self, mesh: Mesh, placement: CIFnPlacement) -> "ChunkwiseTransformerCIFn":
        """The stacked per-chunk transformer's HSDP layout (`ChunkTransformer.shardings`,
        leading `n_chunks` axis un-sharded); `inv_freq` (a 1-D RoPE buffer) replicates."""
        return eqx.tree_at(
            lambda f: (f.chunks, f.inv_freq),
            self,
            (self.chunks.shardings(mesh, placement), NamedSharding(mesh, P())),
        )

    def __call__(
        self,
        taps: dict[str, Array],
        *,
        remat: bool,
        placement: CIFnPlacement | None,
    ) -> CI:
        return self._evaluate(
            taps,
            remat=remat,
            placement=placement,
            valid_token_count=None,
        )

    def evaluate_padded(
        self,
        taps: dict[str, Array],
        valid_token_count: Int[Array, ""],
        *,
        remat: bool,
        placement: CIFnPlacement | None,
    ) -> CI:
        return self._evaluate(
            taps,
            remat=remat,
            placement=placement,
            valid_token_count=valid_token_count,
        )

    def _evaluate(
        self,
        taps: dict[str, Array],
        *,
        remat: bool,
        placement: CIFnPlacement | None,
        valid_token_count: Int[Array, ""] | None,
    ) -> CI:
        per_chunk_in = [
            jnp.concatenate(
                [
                    # Both boundaries matter: the first keeps the cached tap TP-replicated
                    # through RMS reduction; the second prevents the input-projection slice
                    # from being sunk backward through that reduction.
                    _constrain_ci_activation(
                        _weightless_rms_norm(
                            _constrain_ci_activation(taps[k], placement, "feature"), self.eps
                        ),
                        placement,
                        "feature",
                    )
                    for k in m.input_taps
                ],
                axis=-1,
            )
            for m in self.chunk_meta
        ]
        stacked_in = jnp.stack(per_chunk_in, axis=0)  # [n_chunks, *leading, total_d_in]
        inv_freq = jax.lax.stop_gradient(self.inv_freq)
        # `lax.scan` (not `filter_vmap`) over the leading `n_chunks` axis so XLA lowers the
        # chunk iteration as a loop: one chunk's FSDP weight all-gather (∝ ΣC/tp) is live at
        # a time, then freed, instead of every chunk's gathered weights materialized at once
        # (the vmap unrolls, hoisting all n_chunks gathers into the flat entry computation).
        # Same math as the vmap — scan stacks per-iteration outputs exactly as vmap maps
        # them; results match up to fp32 reassociation (XLA picks different matmul layouts).
        chunk_arrays, chunk_static = eqx.partition(self.chunks, eqx.is_array)

        def run_chunk(
            _: None, scanned: tuple[ChunkTransformer, Array]
        ) -> tuple[None, tuple[Array, ...]]:
            chunk_array, chunk_input = scanned
            chunk = eqx.combine(chunk_array, chunk_static)
            return None, chunk(
                chunk_input,
                inv_freq,
                placement=placement,
                valid_token_count=valid_token_count,
            )

        # Per-CHUNK remat: checkpoint the scan BODY so the backward recomputes one chunk at a
        # time, keeping only the carry — NOT all `n_chunks` chunks' attention scores + MLP
        # hidden states stacked `[n_chunks, ...]`. (Whole-CI-fn checkpointing does not bound
        # the scan: the recompute still stacks every chunk — the `[n_chunks, *, seq, seq]`
        # f32 score slab that dominated the full-model step. Same fix shape as the target's
        # per-layer remat.)
        # Each per-slot head stacks over the chunk axis: `stacked_per_slot[j]` is
        # `[n_chunks, *leading, C_j]`. No glued ΣC axis, so no slice — site `(chunk i, slot j)`
        # is `stacked_per_slot[j][i]` directly (chunks are slot-homogeneous in C-per-site
        # ORDER, asserted at init, so slot j carries one C_j across every chunk).
        # Per-CHUNK checkpoint of the scan BODY in BOTH modes — `remat` controls ONLY whether
        # the chunk ACTIVATIONS are recomputed; it NEVER controls the ÷fsdp→full weight gather.
        # `remat=True` → nothing_saveable: recompute activations AND re-gather (min memory, the
        # `[n_chunks, *, seq, seq]` f32 score slab never stacks). `remat=False` → dots_saveable:
        # SAVE the activation matmuls, still re-gather the weights (a collective, not a dot) — i.e.
        # plain FSDP. WITHOUT any checkpoint the backward would instead stack every chunk's full
        # gathered weights `[n_chunks, …]` as residuals → DDP-stack OOM, so we always checkpoint.
        policy = (
            jax.checkpoint_policies.nothing_saveable
            if remat
            else jax.checkpoint_policies.dots_saveable
        )

        body = jax.checkpoint(run_chunk, policy=policy)
        _, stacked_per_slot = jax.lax.scan(body, None, (chunk_arrays, stacked_in))
        preactivations: SiteDict = {}
        for chunk_idx, m in enumerate(self.chunk_meta):
            for slot, site in enumerate(m.output_sites):
                preactivations[site] = stacked_per_slot[slot][chunk_idx]
        return CI.from_preactivations(preactivations)


def _init_chunk_transformer(
    arch: ChunkwiseTransformerCIArch,
    total_d_in: int,
    slot_cs: tuple[int, ...],
    key: PRNGKeyArray,
) -> ChunkTransformer:
    """One chunk's params, same Kaiming scheme as the old global transformer: relu-gain
    (√2) on in_proj / MLP-in, linear gain (1) on out / MLP-out, PyTorch-default
    `U(±1/√fan_in)` on the attention projections, zero biases.

    The per-site output heads are SLICES of a single glued `[d, ΣC]` Kaiming draw (drawn with
    the same `out_key`, `gain 1`): head j = columns `[offset_j : offset_j + C_j]`. This keeps
    the RNG consumption (one `(d, ΣC)` normal + one `(ΣC,)` zero bias) and the values bit-for-
    bit identical to the old single glued head, so the equivalence goldens are unchanged —
    the math is the same, only the partitioning differs.

    Each consumer takes its OWN explicit key — the split count lives next to its use
    (`n_blocks + 2` at the top = in_proj + out + one per block; 6 within a block, 7 with
    swiglu's gate), so it can't silently drift out of sync with the number of draws."""
    relu_gain = 2.0**0.5
    d, ffn = arch.d_model, arch.ffn_hidden
    d_kv = (d // arch.attention.n_heads) * arch.attention.n_kv_heads  # narrower under GQA

    def kaiming(k: PRNGKeyArray, shape: tuple[int, ...], fan_in: int, gain: float) -> Array:
        return jax.random.normal(k, shape) * (gain / fan_in**0.5)

    def attn_default(k: PRNGKeyArray, shape: tuple[int, ...], fan_in: int) -> Array:
        bound = 1.0 / fan_in**0.5
        return jax.random.uniform(k, shape, minval=-bound, maxval=bound)

    def block(bkey: PRNGKeyArray) -> CIBlock:
        # 6 draws for gelu, 7 for swiglu's extra gate — NOT 7 unconditionally: the split
        # count determines every derived key, so widening it would silently redraw every
        # gelu param and move the equivalence goldens.
        match arch.ffn_kind:
            case "gelu":
                kq, kk, kv, ko, k1, k2 = jax.random.split(bkey, 6)
                gate = None
            case "swiglu":
                kq, kk, kv, ko, k1, k2, kg = jax.random.split(bkey, 7)
                gate = (kaiming(kg, (d, ffn), d, relu_gain), jnp.zeros((ffn,)))
        norm_scales = (jnp.ones((d,)), jnp.ones((d,))) if arch.learned_norm_scale else None
        return CIBlock(
            wq=attn_default(kq, (d, d), d),
            wk=attn_default(kk, (d_kv, d), d),
            wv=attn_default(kv, (d_kv, d), d),
            wo=attn_default(ko, (d, d), d),
            w1=kaiming(k1, (d, ffn), d, relu_gain),
            b1=jnp.zeros((ffn,)),
            w2=kaiming(k2, (ffn, d), ffn, 1.0),
            b2=jnp.zeros((d,)),
            gate=gate,
            norm_scales=norm_scales,
            attention=arch.attention,
            eps=CI_FN_RMS_EPS,
        )

    in_key, out_key, *block_keys = jax.random.split(key, arch.n_blocks + 2)
    c_chunk = sum(slot_cs)
    glued_w = kaiming(out_key, (d, c_chunk), d, 1.0)
    glued_b = jnp.zeros((c_chunk,))
    offsets = [0]
    for c in slot_cs:
        offsets.append(offsets[-1] + c)
    return ChunkTransformer(
        in_proj_w=kaiming(in_key, (total_d_in, d), total_d_in, relu_gain),
        in_proj_b=jnp.zeros((d,)),
        blocks=[block(bk) for bk in block_keys],
        out_ws=tuple(glued_w[:, offsets[j] : offsets[j + 1]] for j in range(len(slot_cs))),
        out_bs=tuple(glued_b[offsets[j] : offsets[j + 1]] for j in range(len(slot_cs))),
    )


def init_chunkwise_transformer_ci_fn(
    arch: ChunkwiseTransformerCIArch, sites: tuple[SiteSpec, ...], key: PRNGKeyArray
) -> ChunkwiseTransformerCIFn:
    """Validate the output partition + chunk homogeneity, then build STACKED chunk params.

    - partition: the chunks' output sites are disjoint and cover every model site.
    - homogeneity: equal tap count (→ equal total input width) and an identical per-SLOT C
      tuple (same C-per-output-site in the same ORDER) across every chunk, so the per-chunk
      params — including the per-slot output heads — stack rectangularly along the scanned
      `n_chunks` axis. The per-slot heads stack slot-by-slot, so a mismatched C ORDER would
      silently misalign sites across chunks: fail fast.
    """
    site_c = {s.name: s.C for s in sites}
    covered = [name for ch in arch.chunks for name in ch.output_sites]
    assert sorted(covered) == sorted(s.name for s in sites), "chunks must partition sites"
    assert len(covered) == len(set(covered)), "chunks overlap on an output site"
    slot_cs_per_chunk = {tuple(site_c[n] for n in ch.output_sites) for ch in arch.chunks}
    assert len(slot_cs_per_chunk) == 1, (
        f"chunks not homogeneous in per-slot C tuple (the per-slot heads stack slot-by-slot "
        f"across chunks — equal C-per-site ORDER required): {slot_cs_per_chunk}"
    )
    (slot_cs,) = slot_cs_per_chunk
    assert all(ch.input_taps for ch in arch.chunks), "each chunk needs at least one input tap"
    # Per-chunk cat width must equal `arch.input_dim` (lab guarantees it; the runtime
    # `jnp.stack` / in_proj einsum fails loud if a chunk's taps don't sum to it).

    assert arch.n_blocks >= 0, (
        f"n_blocks must be >= 0 ({arch.n_blocks}); 0 is the legitimate position-local arch — "
        "in_proj + output heads, no attention — see ChunkwiseTransformerCIArch"
    )
    n_heads = arch.attention.n_heads
    hd = arch.d_model // n_heads
    assert arch.d_model % n_heads == 0 and hd % 2 == 0, (arch.d_model, n_heads)
    inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / hd))

    # vmap over the per-chunk keys instead of unrolling n_chunks python-side inits and
    # stacking: bit-identical draws (same fold_in key per chunk), same stacked layout, but
    # the init graph is ONE chunk's RNG body — the unrolled form's XLA compile time grows
    # with chunk count (multi-minute at tens of chunks).
    chunk_keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(len(arch.chunks)))
    stacked: ChunkTransformer = eqx.filter_vmap(
        lambda k: _init_chunk_transformer(arch, arch.input_dim, slot_cs, k)
    )(chunk_keys)

    return ChunkwiseTransformerCIFn(
        chunks=stacked,
        inv_freq=inv_freq,
        capture_keys=arch.capture_keys,
        output_names=tuple(name for ch in arch.chunks for name in ch.output_sites),
        chunk_meta=tuple(_ChunkMeta(ch.input_taps, ch.output_sites) for ch in arch.chunks),
        eps=CI_FN_RMS_EPS,
        has_position_axis=True,
    )


# ------------- per-site / global MLPs (pointwise over every leading axis) -------------


# The MLP arches bind their config to a target at the composition root. Their input taps
# (`input_names` / `input_taps`) are therefore resolved exactly once, like the chunkwise
# architecture's authored tap union, and every downstream consumer reads the same
# authoritative field.
@dataclass(frozen=True)
class LayerwiseMLPCIArch:
    """Hidden widths shared by every per-site MLP.

    `has_position_axis` is the TARGET's shape, not a property of the MLP: the stack is
    pointwise over every leading axis, so the same weights serve `[batch, d]` and
    `[batch, position, d]` alike. It is declared here so the CI fn and the model can be
    checked to agree (`core.run_state.init_decomposition`)."""

    hidden_dims: tuple[int, ...]
    has_position_axis: bool
    input_names: tuple[str, ...]

    @property
    def capture_keys(self) -> CaptureKeys:
        return frozenset(self.input_names)


class SiteMLP(eqx.Module):
    """`hidden_dims` Linear+GELU layers then a linear head: Kaiming-`relu` (`gain √2`)
    hidden layers with zero bias, linear-gain (`1`) final head."""

    weights: list[Float[Array, "d_in d_out"]]
    biases: list[Float[Array, " d_out"]]

    def shardings(self, mesh: Mesh) -> "SiteMLP":
        """Each `[d_in, d_out]` weight shards its OUTPUT axis (axis 1) over the data axes
        (`("replicate","fsdp")`) — the master + Adam state shard ÷(replicate·fsdp). 1-D
        biases replicate. The MLP is single-shot (no scan), so there is no compute
        reconstruction; GSPMD gathers as needed (trivial at the toy's small device count).
        Asserts every output dim tiles its actual shard count — not the total device count,
        which over-counts by ×tp."""
        shard_axes: tuple[MeshAxis, ...] = ("replicate", "fsdp")
        shard_out = NamedSharding(mesh, P(None, shard_axes))
        repl = NamedSharding(mesh, P())
        n = math.prod(mesh.shape[a] for a in shard_axes)
        for layer_idx, w in enumerate(self.weights):
            assert w.shape[1] % n == 0, (
                f"SiteMLP.weights[{layer_idx}].d_out {w.shape[1]} not ÷ N={n}"
            )
        return eqx.tree_at(
            lambda m: (m.weights, m.biases),
            self,
            ([shard_out] * len(self.weights), [repl] * len(self.biases)),
        )

    def __call__(self, x: Float[Array, "*leading d_in"]) -> Float[Array, "*leading C"]:
        n_hidden = len(self.weights) - 1
        on_mesh = not value_mesh(x).empty
        for layer_idx, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            if on_mesh:
                # The ZeRO-stored d_out shard uses the same axes as the batch, so the
                # operand must materialize replicated, and the output must be typed
                # for the weight-grad transpose to resolve against the axis-typed
                # batch.
                w = jax.sharding.reshard(w, P(None, None))
                leading_spec = jax.typeof(x).sharding.spec[:-1]
                x = jnp.einsum("...i,io->...o", x, w, out_sharding=P(*leading_spec, None)) + b
            else:
                x = einops.einsum(x, w, "... i, i o -> ... o") + b
            if layer_idx < n_hidden:
                x = jax.nn.gelu(x, approximate=False)
        return x


class LayerwiseMLPCIFn(eqx.Module):
    """One MLP per site, with input taps aligned to output sites by position."""

    site_mlps: dict[str, SiteMLP]
    input_names: tuple[str, ...] = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)

    @property
    def capture_keys(self) -> CaptureKeys:
        return frozenset(self.input_names)

    def shardings(self, mesh: Mesh) -> "LayerwiseMLPCIFn":
        return eqx.tree_at(
            lambda f: f.site_mlps,
            self,
            {name: mlp.shardings(mesh) for name, mlp in self.site_mlps.items()},
        )

    def site_preactivations(self, taps: dict[str, Array]) -> dict[str, Array]:
        assert set(taps) == set(self.input_names), (
            f"tap keys {sorted(taps)} != CI fn inputs {sorted(self.input_names)}"
        )
        return {
            output_name: self.site_mlps[output_name](taps[input_name])
            for input_name, output_name in zip(self.input_names, self.output_names, strict=True)
        }

    def __call__(
        self, taps: dict[str, Array], *, remat: bool, placement: CIFnPlacement | None
    ) -> CI:
        del remat  # single-shot (no scan to bound) -> remat is a no-op for the MLP CI fns
        assert placement is None, f"{type(self).__name__} is unplaced (no CI placement rows)"
        return CI.from_preactivations(self.site_preactivations(taps))


def _init_mlp_stack(dims: tuple[int, ...], key: PRNGKeyArray) -> SiteMLP:
    """One `Linear+GELU` stack `dims[0] -> ... -> dims[-1]`: Kaiming `relu`-gain (`√2`) on
    the hidden layers, linear gain (`1`) on the final head, zero biases."""
    relu_gain = 2.0**0.5
    layer_keys = jax.random.split(key, len(dims) - 1)
    weights: list[Array] = []
    biases: list[Array] = []
    for layer_idx, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        gain = relu_gain if layer_idx < len(dims) - 2 else 1.0
        weights.append(jax.random.normal(layer_keys[layer_idx], (d_in, d_out)) * (gain / d_in**0.5))
        biases.append(jnp.zeros((d_out,)))
    return SiteMLP(weights=weights, biases=biases)


def init_layerwise_mlp_ci_fn(
    arch: LayerwiseMLPCIArch, sites: tuple[SiteSpec, ...], key: PRNGKeyArray
) -> LayerwiseMLPCIFn:
    """Per-site MLP init: each site's MLP maps `d_in -> hidden_dims... -> C`."""
    assert arch.hidden_dims, "MLP CI fn needs at least one hidden layer"
    site_mlps = {
        spec.name: _init_mlp_stack(
            (spec.d_in, *arch.hidden_dims, spec.C), jax.random.fold_in(key, site_idx)
        )
        for site_idx, spec in enumerate(sites)
    }
    output_names = tuple(s.name for s in sites)
    assert len(arch.input_names) == len(output_names), (arch.input_names, output_names)
    assert len(set(arch.input_names)) == len(arch.input_names), arch.input_names
    return LayerwiseMLPCIFn(
        site_mlps=site_mlps,
        input_names=arch.input_names,
        output_names=output_names,
        has_position_axis=arch.has_position_axis,
    )


@dataclass(frozen=True)
class TapSpec:
    """One input tap: its capture key and feature width. The key is opaque to core (the
    lab authors it, the target resolves it); the width rides alongside so the consumer
    can size and assert its input without deriving it from a site."""

    key: str
    width: int


@dataclass(frozen=True)
class GlobalMLPCIArch:
    """Hidden widths of the single global MLP shared across ALL sites, plus the input
    taps it concatenates. The taps are DECOUPLED from the output sites: several sites may
    read one physical tap (an LM block's q/k/v share its attention input), so the taps
    are unique keys with explicit widths, never a per-site alignment
    (`LayerwiseMLPCIArch` keeps that alignment — there it is real)."""

    hidden_dims: tuple[int, ...]
    has_position_axis: bool
    input_taps: tuple[TapSpec, ...]

    @property
    def capture_keys(self) -> CaptureKeys:
        return frozenset(tap.key for tap in self.input_taps)


class GlobalMLPCIFn(eqx.Module):
    """ONE shared MLP over all sites behind the `CIFn` protocol. The taps are
    concatenated in `input_taps` order into `[*leading, Σ width]`, mapped to `[*leading,
    Σ C]`, and split back per output site by `c_sizes` in `output_names` order — so every
    site's preactivations depend on every tap."""

    mlp: SiteMLP
    input_taps: tuple[TapSpec, ...] = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)
    c_sizes: tuple[int, ...] = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)

    @property
    def capture_keys(self) -> CaptureKeys:
        return frozenset(tap.key for tap in self.input_taps)

    def shardings(self, mesh: Mesh) -> "GlobalMLPCIFn":
        return eqx.tree_at(lambda f: f.mlp, self, self.mlp.shardings(mesh))

    def site_preactivations(self, taps: dict[str, Array]) -> dict[str, Array]:
        assert set(taps) == {tap.key for tap in self.input_taps}, (
            f"tap keys {sorted(taps)} != CI fn inputs {sorted(t.key for t in self.input_taps)}"
        )
        for tap in self.input_taps:
            assert taps[tap.key].shape[-1] == tap.width, (
                f"tap {tap.key} width {taps[tap.key].shape[-1]} != expected {tap.width}"
            )
        concatenated = jnp.concatenate([taps[tap.key] for tap in self.input_taps], axis=-1)
        preactivations = self.mlp(concatenated)
        offsets = [0]
        for c in self.c_sizes:
            offsets.append(offsets[-1] + c)
        return {
            name: preactivations[..., offsets[i] : offsets[i + 1]]
            for i, name in enumerate(self.output_names)
        }

    def __call__(
        self, taps: dict[str, Array], *, remat: bool, placement: CIFnPlacement | None
    ) -> CI:
        del remat  # single-shot (no scan to bound) -> remat is a no-op for the MLP CI fns
        assert placement is None, f"{type(self).__name__} is unplaced (no CI placement rows)"
        return CI.from_preactivations(self.site_preactivations(taps))


def init_global_mlp_ci_fn(
    arch: GlobalMLPCIArch, sites: tuple[SiteSpec, ...], key: PRNGKeyArray
) -> GlobalMLPCIFn:
    """Global MLP init: one stack `Σ tap width -> hidden_dims... -> Σ C`, same Kaiming
    scheme as the per-site MLP."""
    assert arch.hidden_dims, "global MLP CI fn needs at least one hidden layer"
    tap_keys = tuple(tap.key for tap in arch.input_taps)
    assert tap_keys and len(set(tap_keys)) == len(tap_keys), tap_keys
    c_sizes = tuple(s.C for s in sites)
    dims = (sum(tap.width for tap in arch.input_taps), *arch.hidden_dims, sum(c_sizes))
    return GlobalMLPCIFn(
        mlp=_init_mlp_stack(dims, key),
        input_taps=arch.input_taps,
        output_names=tuple(s.name for s in sites),
        c_sizes=c_sizes,
        has_position_axis=arch.has_position_axis,
    )


# ----------------------------- construction (placement-agnostic) -----------------------------


CIFnArch = ChunkwiseTransformerCIArch | LayerwiseMLPCIArch | GlobalMLPCIArch
"""Every CI-fn architecture. Construction goes through `build_ci_fn`; sharding/placement is
a separate, scale-driven concern (see `init_placed`), never coupled to arch type."""


def resolve_ci_placement(arch: CIFnArch, rules: PlacementRules | None) -> CIFnPlacement | None:
    """THE one CI-placement resolution, at run assembly: the chunkwise transformer
    consumes the run's CI rows; the MLP archs run unplaced. Downstream code receives
    the already-paired `PlacedCIFn` (or, on the muon path, this resolved value) —
    never the raw rules next to a fn.

    Resolution is also where the attention head split's divisibility is refused: the
    split parks each projection's flat assignment on its head-COUNT axis
    (`CIBlock.__call__`'s `heads`), so both counts must tile — under GQA `kv_head` is
    the narrow one. There is no replication fallback; a user who WANTS replicated K/V
    heads authors an explicit table with `kv_head` unmapped."""
    match arch:
        case ChunkwiseTransformerCIArch():
            if rules is None:
                return None
            activations = rules.ci_fn.activations
            activations.validate_shape(("q_head",), (arch.attention.n_heads,))
            activations.validate_shape(("kv_head",), (arch.attention.n_kv_heads,))
            return rules.ci_fn
        case LayerwiseMLPCIArch() | GlobalMLPCIArch():
            return None


def build_ci_fn(arch: CIFnArch, sites: tuple[SiteSpec, ...], key: PRNGKeyArray) -> CIFn:
    """Construct the CI fn for `arch`, host-side and unsharded. Placement is applied by the
    caller by SCALE (mesh × C-divisibility), never by which arch this is."""
    match arch:
        case ChunkwiseTransformerCIArch():
            return init_chunkwise_transformer_ci_fn(arch, sites, key)
        case LayerwiseMLPCIArch():
            return init_layerwise_mlp_ci_fn(arch, sites, key)
        case GlobalMLPCIArch():
            return init_global_mlp_ci_fn(arch, sites, key)
