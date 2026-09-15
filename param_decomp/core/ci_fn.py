"""CI-fn interface + the chunkwise-transformer impl.

A CI fn maps named INPUT taps to a `CI` bundle over OUTPUT sites:
`dict[InputTap, Array] -> CI` (preactivations + the two squashings). The input keyspace (opaque
tap keys — the lab authors them, the target resolves and captures them) is
independent of the output keyspace (the decomposition sites). The output sites MUST
partition the model's sites — every site needs exactly one CI value — asserted at
construction. Core treats both keyspaces as OPAQUE dict keys: look up inputs, scatter
outputs, validate the partition. It never parses a key.

The SAME preactivations are squashed two ways in ONE place (`CI.from_preactivations`):
`lower` (clip[0,1], leaky-below) feeds recon / PPGD / routing masks; `upper`
(leaky-above-1) feeds importance-minimality. `preactivations` is kept too — the CI histograms /
heatmaps plot the pre-squash view. Params are fp32 masters; the trainer casts
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
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal, Protocol, runtime_checkable

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int, PRNGKeyArray

from param_decomp.core.axes import Axes, MeshAxis, SemanticAxis
from param_decomp.core.components import (
    ExpertBlocked,
    NarrowCI,
    SiteCI,
    SiteSpec,
    activation_axes,
    map_site_ci,
)
from param_decomp.core.linear_plan import placed_linear, value_mesh
from param_decomp.core.model import CaptureKeys
from param_decomp.core.placement import (
    CIFnPlacement,
    CIFnRows,
    CIWeightFamily,
    CIWeightPlacement,
    PlacedRule,
    PlacementRules,
    StackCensus,
    batch_axes,
    materialize_reduced_weights,
    ns_staging_sharding,
    resolve_stack_census,
    strip_stack_pad,
    validate_stacked_leaf,
)
from param_decomp.core.precision import COMPUTE_DT, cast_floating
from param_decomp.routed.experts import (
    ExpertShardedJobs,
    GroupedMatmulBackend,
    RoutedJobs,
    combine_jobs,
    ep_combine_jobs,
    ep_gather_tokens,
    ep_grouped_matmul,
    ep_unsort_jobs,
    expert_sharded_jobs,
    gather_tokens,
    grouped_matmul,
    routed_jobs,
    unsort_jobs,
)
from param_decomp.target_ports.llama import (
    apply_rope,
    attn_implementation,
    rms_norm,
    rope_cos_sin,
)

CI_FN_RMS_EPS = float(jnp.finfo(jnp.float32).eps)
"""Matches torch's `F.rms_norm` default eps (`finfo(fp32).eps` ~1.19e-7); RMS upcasts to
fp32 internally, so this is the dtype that governs."""


SiteDict = dict[str, SiteCI]
"""Per-output-site CI value keyed by OUTPUT site name: full `[*leading, C]` arrays for
dense sites, `NarrowCI` bundles for expert-blocked narrow-emitting sites
(`components.SiteCI` / `components.NarrowCI`)."""


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


# ----------------------------- squashings -----------------------------


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
    def from_preactivations(preactivations: "Mapping[str, SiteCI]") -> "CI":
        return CI(
            preactivations=dict(preactivations),
            lower={k: map_site_ci(lower_leaky_hard_sigmoid, v) for k, v in preactivations.items()},
            upper={k: map_site_ci(upper_leaky_hard_sigmoid, v) for k, v in preactivations.items()},
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
    """The CI fn's entry: fp32 masters → compute dtype → every stacked leaf to its
    compute row with the chunk stack's persist pads stripped on the way
    (`_reconstruct_ci_compute_weights`), so the fn the forward scans carries ONLY real
    chunks (`_resident_chunk_stack`)."""
    compute_ci_fn = cast_floating(placed_ci_fn.fn, COMPUTE_DT)
    placement = placed_ci_fn.placement
    match compute_ci_fn:
        case ChunkwiseTransformerCIFn():
            chunks = _reconstruct_ci_compute_weights(compute_ci_fn.chunks, placement)
            return PlacedCIFn(
                fn=_resident_chunk_stack(compute_ci_fn, chunks, placement), placement=placement
            )
        case MoEChunkwiseTransformerCIFn():
            chunks = _reconstruct_moe_ci_compute_weights(compute_ci_fn.chunks, placement)
            return PlacedCIFn(
                fn=_resident_chunk_stack(compute_ci_fn, chunks, placement), placement=placement
            )
        case LayerwiseMLPCIFn() | GlobalMLPCIFn():
            # Bypass protection: `resolve_ci_placement` never pairs an MLP fn with rows,
            # so a placed non-chunkwise bundle can only be a hand-built mispairing.
            assert placement is None, (
                f"CI placement rows require a chunkwise transformer, got {type(compute_ci_fn)}"
            )
            return PlacedCIFn(fn=compute_ci_fn, placement=None)
        case _:
            raise AssertionError(f"unknown CI fn {type(compute_ci_fn)}")


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


def _attention_half(
    x: Float[Array, "b t d"],
    *,
    wq: Array,
    wk: Array,
    wv: Array,
    wo: Array,
    attention: CIAttention,
    inv_freq: Array,
    norm_scale: Array | None,
    eps: float,
    placement: CIFnPlacement | None,
    valid_token_count: Int[Array, ""] | None,
) -> Array:
    """The pre-norm bidirectional-RoPE attention half of a CI block, residual included —
    shared verbatim by the dense and MoE blocks. `valid_token_count` masks right-padding
    out of attention (the padded prompt-analysis path)."""
    t = x.shape[1]
    h = _rms_norm_maybe_scaled(x, norm_scale, eps)

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

    q = heads(wq, attention.n_heads, CI_ATTN_Q_AXES)
    kv = attention.n_kv_heads
    k, v = heads(wk, kv, CI_ATTN_KV_AXES), heads(wv, kv, CI_ATTN_KV_AXES)
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
    if value_mesh(qt).empty:
        y = jax.nn.dot_product_attention(
            qt,
            kt,
            vt,
            is_causal=False,
            query_seq_lengths=sequence_lengths,
            key_value_seq_lengths=sequence_lengths,
            implementation=impl,
        )
    else:
        # The XLA arm's internals (vmap + einsum) don't preserve explicit-sharding
        # typing across their batch dims, so run the call under auto axes and re-type
        # the output with the operands' own (identical) sharding. The sequence lengths
        # ride as an operand (None when unpadded) so both arms share one padding spelling.
        def attention_under_auto(operands: tuple[Array, Array, Array, Array | None]) -> Array:
            q, k, v, lengths = operands
            return jax.nn.dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                query_seq_lengths=lengths,
                key_value_seq_lengths=lengths,
                implementation=impl,
            )

        out_sharding = NamedSharding(value_mesh(qt), jax.typeof(qt).sharding.spec)
        y = jax.sharding.auto_axes(attention_under_auto, out_sharding=out_sharding)(
            (qt, kt, vt, sequence_lengths)
        )
        assert isinstance(y, jax.Array), type(y)
    return x + _ci_linear(
        einops.rearrange(y, "b t nh hd -> b t (nh hd)"),
        wo,
        placement,
        "attention",
        CI_ATTN_OUT_AXES[1:],
        transposed=True,
    )


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
        attn_scale, mlp_scale = (None, None) if self.norm_scales is None else self.norm_scales
        x = _attention_half(
            x,
            wq=self.wq,
            wk=self.wk,
            wv=self.wv,
            wo=self.wo,
            attention=self.attention,
            inv_freq=inv_freq,
            norm_scale=attn_scale,
            eps=self.eps,
            placement=placement,
            valid_token_count=valid_token_count,
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
    split per site (matching `x@V` / the mask, `site_out`). Under pure HSDP the C
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
    """The stacked chunk module's entry: every weight leaf from its persist row to its
    compute row with the chunk stack's pads stripped on the way
    (`materialize_reduced_weights` — the leaves arrive compute-dtype, the whole fn is
    cast first), every vector leaf's pads stripped in place (`strip_stack_pad`: the
    vectors row is their persist AND compute layout, whole on the stack axis). The
    leaves are enumerated by name; `_resident_chunk_stack` checks every leaf's extent so
    one this list misses cannot reach the scan padded. No-op off-mesh."""
    if jax.sharding.get_abstract_mesh().empty:
        return chunks
    assert placement is not None, "on-mesh CI compute-weight materialization requires placement"
    census = placement.chunks

    def enter(x: Array, weights: CIWeightPlacement, axes: Axes) -> Array:
        return materialize_reduced_weights(
            x,
            census=census,
            source=weights.optimizer_state,
            destination=weights.compute_weights,
            axes=axes,
        )

    def vector(x: Array) -> Array:
        return strip_stack_pad(x, census)

    def block(blk: CIBlock) -> CIBlock:
        return replace(
            blk,
            wq=enter(blk.wq, placement.attention, CI_ATTN_Q_AXES),
            wk=enter(blk.wk, placement.attention, CI_ATTN_KV_AXES),
            wv=enter(blk.wv, placement.attention, CI_ATTN_KV_AXES),
            wo=enter(blk.wo, placement.attention, CI_ATTN_OUT_AXES),
            w1=enter(blk.w1, placement.ffn, CI_FFN_IN_AXES),
            b1=vector(blk.b1),
            w2=enter(blk.w2, placement.ffn, CI_FFN_OUT_AXES),
            b2=vector(blk.b2),
            gate=None
            if blk.gate is None
            else (enter(blk.gate[0], placement.ffn, CI_FFN_IN_AXES), vector(blk.gate[1])),
            norm_scales=None
            if blk.norm_scales is None
            else (vector(blk.norm_scales[0]), vector(blk.norm_scales[1])),
        )

    return replace(
        chunks,
        in_proj_w=enter(chunks.in_proj_w, placement.input, CI_INPUT_AXES),
        in_proj_b=vector(chunks.in_proj_b),
        blocks=[block(blk) for blk in chunks.blocks],
        out_ws=tuple(enter(w, placement.output, CI_OUTPUT_AXES) for w in chunks.out_ws),
        out_bs=tuple(vector(b) for b in chunks.out_bs),
    )


class ChunkwiseTransformerCIFn(eqx.Module):
    """Per-chunk `ChunkTransformer`s stacked along a leading `n_chunks` axis, iterated by a
    `jax.lax.scan` over that axis (lowers as a loop so one chunk's FSDP weight gather is live
    at a time, not all `n_chunks` at once). Each chunk's input is its `chunk_input_taps`
    RMS-normed per tap and concatenated. Requires homogeneous chunks (equal total input width
    and an identical per-slot C tuple — same C-per-output-site ORDER) so the stack, including
    the per-slot output heads, is rectangular — asserted at init."""

    chunks: ChunkTransformer  # arrays stacked along leading n_chunks (+ stack_pad)
    inv_freq: Array  # shared across chunks (RoPE buffer); NOT mapped

    capture_keys: CaptureKeys = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)  # all sites, flat
    chunk_meta: tuple[_ChunkMeta, ...] = eqx.field(static=True)  # per-chunk routing
    stack_pad: int = eqx.field(static=True)
    """The persist-layer PAD slots trailing the real chunks on every `chunks` leaf
    (`pad_ci_fn`) — enumerated here, never shape-inferred; `chunk_meta` never names
    them. `0` on every unplaced fn and on the compute residents the forward scans."""
    eps: float = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)

    def shardings(self, mesh: Mesh, placement: CIFnPlacement) -> "ChunkwiseTransformerCIFn":
        """The stacked per-chunk transformer's persist layout (`ChunkTransformer.shardings`
        at the padded chunk extent); `inv_freq` (a 1-D RoPE buffer) replicates."""
        _validate_chunk_stack(self, placement.chunks, self.chunks.in_proj_w.shape[0])
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
        assert self.stack_pad == 0, (
            f"the chunk scan runs the compute residents, which carry no persist pads "
            f"(got stack_pad={self.stack_pad}); enter through materialize_ci_compute_weights"
        )
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
        stack_pad=0,
        eps=CI_FN_RMS_EPS,
        has_position_axis=True,
    )


# ------------------- MoE chunkwise transformer (routed narrow emission) -------------------
# The chunkwise transformer's sibling for MoE targets: blocks mirror the target's MoE
# stage (non-causal attention + concat-wide routed-expert FFN banks reusing the target's
# CAPTURED per-token routing), and expert-blocked output sites are scored NARROWLY —
# per-(router, expert) head blocks fused into the expert slots, emitting `NarrowCI`
# bundles rather than the unmaterializable full `[*leading, C]`.


CI_EXPERT_FFN_IN_AXES: Axes = ("stack", "expert", "d_model", "ffn_hidden")
CI_EXPERT_FFN_OUT_AXES: Axes = ("stack", "expert", "ffn_hidden", "d_model")
CI_EXPERT_HEAD_AXES: Axes = ("stack", "expert", "ffn_hidden", "C_block")


@dataclass(frozen=True)
class RoutingTap:
    """One target MoE layer's captured routing, as two input taps: `ids_key` resolves to
    the top-k expert ids `[*leading, k]` (integer) and `weights_key` to the renormalized
    routing weights `[*leading, k]`, in the target's stored top-k order. Captured from
    the CLEAN forward like every other CI input tap, so the values arrive stop-gradded;
    the CI fn never routes for itself."""

    ids_key: str
    weights_key: str


@dataclass(frozen=True)
class FullSlot:
    """A chunk output site scored by a dense `[d_model, C]` head — full emission."""

    site: str


@dataclass(frozen=True)
class NarrowSlot:
    """An expert-blocked output site, scored narrowly under router `router` (an index
    into the chunk's `routing`): its head blocks dispatch on exactly that target layer's
    top-k — the per-(layer, expert) parameter identity."""

    site: str
    router: int


MoESlot = FullSlot | NarrowSlot


@dataclass(frozen=True)
class MoEChunk:
    """One resolved MoE chunk: the input taps to concatenate, the routing tap of EVERY
    target layer the chunk covers (each block's concat-wide expert banks dispatch on
    all of them, and each routing also enters the chunk input as a dense weight
    vector), and the output slots in emission order. Authored lab-side; core treats
    every key as opaque."""

    input_taps: tuple[str, ...]
    routing: tuple[RoutingTap, ...]
    slots: tuple[MoESlot, ...]

    @property
    def output_sites(self) -> tuple[str, ...]:
        return tuple(slot.site for slot in self.slots)


@dataclass(frozen=True)
class MoEChunkwiseTransformerCIArch:
    """Resolved MoE chunkwise-transformer arch. Each chunk covers one target stage and
    runs `n_blocks` blocks of non-causal RoPE attention + a CONCAT-WIDE MoE FFN: one
    `n_experts` swiglu bank per covered target layer (`len(chunk.routing)` banks, each
    expert `d_model -> expert_ffn_hidden`), every bank dispatched by ITS layer's
    captured routing in the same block — so expert slot (layer, e) holds parameters
    that activate exactly when the target routed (layer, e) — plus an always-on dense
    swiglu shared expert (`shared_ffn_hidden`). There is no learned router and no gelu
    arm: the FFN mirrors the target's MoE shape by construction. `n_blocks >= 1` is the
    size lever: each block's expert banks cost one target stage's expert parameters.

    `input_dim` is the full concatenated chunk-input width: the RMS-normed activation
    taps' widths PLUS one dense `n_experts`-wide routing-weight vector per covered
    layer (the lab sums both). `d_model`, `attention`, and `learned_norm_scale` mean
    exactly what they mean on `ChunkwiseTransformerCIArch`."""

    chunks: tuple[MoEChunk, ...]
    input_dim: int
    d_model: int
    n_blocks: int
    attention: CIAttention
    n_experts: int
    expert_ffn_hidden: int
    shared_ffn_hidden: int
    learned_norm_scale: bool
    grouped_matmul_backend: GroupedMatmulBackend
    """The arm the expert banks' grouped matmuls run — authored where the arch is
    resolved (the experiment authors the target family's production arm; toy-shape
    builders author the `ragged_dot` oracle, whose shapes the split arm's 64-tiling
    kernel refuses)."""

    @property
    def capture_keys(self) -> CaptureKeys:
        """Every chunk's activation taps plus both halves of every routing tap."""
        return frozenset(
            key
            for chunk in self.chunks
            for key in (
                *chunk.input_taps,
                *(k for tap in chunk.routing for k in (tap.ids_key, tap.weights_key)),
            )
        )


def _ci_expert_shard_axis(placement: CIFnPlacement) -> MeshAxis:
    """The ONE mesh axis the CI fn's expert grid shards over, read off the moe
    expert-FFN operand row — the same axis the target's experts shard over (the
    co-location premise). Fail-closed: a multi-axis or absent assignment has no EP
    spelling here."""
    moe = placement.moe
    assert moe is not None, "the MoE CI dispatch needs the table's moe families"
    assignment = moe.expert_ffn.operands.assignment("expert")
    assert len(assignment) == 1, (
        f"the placed MoE CI fn needs the expert axis on exactly one mesh axis; "
        f"ci_fn/expert_ffn.operands assigns expert -> {assignment!r}"
    )
    (axis,) = assignment
    return axis


@dataclass(frozen=True)
class _TokenDispatch:
    """One chunk's captured routing as job schedules, UNPLACED: per-router `RoutedJobs`
    over the flattened token axis. One schedule per (chunk, router) serves every block's
    bank for that router and its sites' fused heads."""

    jobs: tuple[RoutedJobs, ...]
    weights: tuple[Float[Array, "T k"], ...]
    lead: tuple[int, ...]
    backend: GroupedMatmulBackend

    def gather(self, router: int, h: Float[Array, "b t d"]) -> Float[Array, "J d"]:
        return gather_tokens(h.reshape(-1, h.shape[-1]), self.jobs[router])

    def expert_matmul(self, router: int, x_jobs: Array, experts: Array) -> Array:
        return grouped_matmul(x_jobs, experts, self.jobs[router].group_sizes, self.backend)

    def combine(self, router: int, y: Float[Array, "J d"]) -> Float[Array, "b t d"]:
        return combine_jobs(y, self.jobs[router], self.weights[router]).reshape(
            *self.lead, y.shape[-1]
        )

    def narrow_values(self, router: int, y: Float[Array, "J c"]) -> Float[Array, "b t k_c"]:
        per_token = unsort_jobs(y, self.jobs[router])  # [T, k, c]
        return per_token.reshape(*self.lead, per_token.shape[-2] * per_token.shape[-1])

    def router_ids(self, router: int) -> Int[Array, "b t k"]:
        ids = self.jobs[router].top_idx
        return ids.reshape(*self.lead, ids.shape[-1])


@dataclass(frozen=True)
class _ExpertShardedDispatch:
    """The expert-parallel sibling on the explicit mesh: per-router `ExpertShardedJobs`
    co-located with the target's expert shard. The narrow combine's unsort sums over
    the expert-shard axis — an ACTIVATION all-reduce over tp, the same collective the
    routed trunk combine already pays."""

    jobs: tuple[ExpertShardedJobs, ...]
    weights: tuple[Float[Array, "b t k"], ...]
    shard_axis: str
    backend: GroupedMatmulBackend

    def gather(self, router: int, h: Float[Array, "b t d"]) -> Float[Array, "b s J d"]:
        return ep_gather_tokens(h, self.jobs[router], self.shard_axis)

    def expert_matmul(self, router: int, x_jobs: Array, experts: Array) -> Array:
        return ep_grouped_matmul(x_jobs, experts, self.jobs[router], self.shard_axis, self.backend)

    def combine(self, router: int, y: Float[Array, "b s J d"]) -> Float[Array, "b t d"]:
        # the CI fn's waist stays replicated over tp in every arm (only the target's
        # masked forwards have a sequence-parallel spelling), so the combine is an all-reduce.
        return ep_combine_jobs(
            y,
            self.jobs[router],
            self.weights[router],
            self.shard_axis,
            P(jax.typeof(y).sharding.spec[0], None, None),
        )

    def narrow_values(self, router: int, y: Float[Array, "b s J c"]) -> Float[Array, "b t k_c"]:
        per_token = ep_unsort_jobs(y, self.jobs[router], self.shard_axis)  # [b, t, k, c]
        b, t, k, c = per_token.shape
        return per_token.reshape(b, t, k * c)

    def router_ids(self, router: int) -> Int[Array, "b t k"]:
        return self.jobs[router].top_idx


MoEDispatch = _TokenDispatch | _ExpertShardedDispatch
"""One chunk's routed dispatch — both arms answer the same five moves, so blocks and
heads never branch on placement."""


def _moe_ci_dispatch(
    ids: Int[Array, "R b t k"],
    weights: Float[Array, "R b t k"],
    n_experts: int,
    placement: CIFnPlacement | None,
    backend: GroupedMatmulBackend,
) -> MoEDispatch:
    n_routers, *lead, k = ids.shape
    if placement is None:
        return _TokenDispatch(
            jobs=tuple(routed_jobs(ids[r].reshape(-1, k), n_experts) for r in range(n_routers)),
            weights=tuple(weights[r].reshape(-1, k) for r in range(n_routers)),
            lead=tuple(lead),
            backend=backend,
        )
    shard_axis = _ci_expert_shard_axis(placement)
    n_shards = placement.activations.mesh.shape[shard_axis]
    return _ExpertShardedDispatch(
        jobs=tuple(expert_sharded_jobs(ids[r], n_experts, n_shards) for r in range(n_routers)),
        weights=tuple(weights[r] for r in range(n_routers)),
        shard_axis=shard_axis,
        backend=backend,
    )


class MoECIBlock(eqx.Module):
    """Pre-norm block: RMSNorm → bidirectional RoPE attention → residual; RMSNorm →
    concat-wide MoE FFN → residual. The attention half is `CIBlock`'s exactly
    (`_attention_half`). The FFN is the target STAGE's MoE shape: one `E`-expert swiglu
    bank per covered target layer, bank r dispatched by router r's captured routing —
    every token activates its `R·k` routed slots — fp32 routing-weighted combines
    scaled 1/R (each router's renormalized weights sum to 1, so R banks would write
    residual mass R where a target block writes 1), plus the always-on dense swiglu
    shared expert. No FFN biases — the expert leaves are exactly the routed-matmul
    operand shapes. The banks are LENGTH-R TUPLES of `[E, ., .]` leaves, not one
    `[R, E, ., .]` axis: the stacked bundle's leaves stay rank-4 — the grouped-matmul
    rhs, the muon 4D canonical fold, and the placement rows all consume that rank
    directly. `norm_scales` as on `CIBlock`."""

    wq: Array
    wk: Array
    wv: Array
    wo: Array
    expert_gate: tuple[Float[Array, "E d_model expert_ffn"], ...]
    expert_up: tuple[Float[Array, "E d_model expert_ffn"], ...]
    expert_down: tuple[Float[Array, "E expert_ffn d_model"], ...]
    shared_gate: Float[Array, "d_model shared_ffn"]
    shared_up: Float[Array, "d_model shared_ffn"]
    shared_down: Float[Array, "shared_ffn d_model"]
    norm_scales: tuple[Array, Array] | None
    attention: CIAttention = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def shardings(self, mesh: Mesh, placement: CIFnPlacement) -> "MoECIBlock":
        """Attention at the ci_fn/attention rows, expert banks at the moe expert_ffn
        rows (expert co-located with the target's expert shard), the shared swiglu at
        the ci_fn/ffn rows, norm scales at the vectors row."""
        moe = placement.moe
        assert moe is not None, "MoECIBlock placement needs the table's moe families"
        attention = placement.attention.optimizer_state
        ffn = placement.ffn.optimizer_state
        expert_ffn = moe.expert_ffn.optimizer_state
        attention.validate_shape(CI_ATTN_Q_AXES, self.wq.shape)
        attention.validate_shape(CI_ATTN_KV_AXES, self.wk.shape)
        attention.validate_shape(CI_ATTN_KV_AXES, self.wv.shape)
        attention.validate_shape(CI_ATTN_OUT_AXES, self.wo.shape)
        for gate, up, down in zip(self.expert_gate, self.expert_up, self.expert_down, strict=True):
            expert_ffn.validate_shape(CI_EXPERT_FFN_IN_AXES, gate.shape)
            expert_ffn.validate_shape(CI_EXPERT_FFN_IN_AXES, up.shape)
            expert_ffn.validate_shape(CI_EXPERT_FFN_OUT_AXES, down.shape)
        ffn.validate_shape(CI_FFN_IN_AXES, self.shared_gate.shape)
        ffn.validate_shape(CI_FFN_IN_AXES, self.shared_up.shape)
        ffn.validate_shape(CI_FFN_OUT_AXES, self.shared_down.shape)
        bank_in = NamedSharding(mesh, expert_ffn.spec_for(CI_EXPERT_FFN_IN_AXES))
        bank_out = NamedSharding(mesh, expert_ffn.spec_for(CI_EXPERT_FFN_OUT_AXES))
        placed = eqx.tree_at(
            lambda b: (
                b.wq,
                b.wk,
                b.wv,
                b.wo,
                b.expert_gate,
                b.expert_up,
                b.expert_down,
                b.shared_gate,
                b.shared_up,
                b.shared_down,
            ),
            self,
            (
                NamedSharding(mesh, attention.spec_for(CI_ATTN_Q_AXES)),
                NamedSharding(mesh, attention.spec_for(CI_ATTN_KV_AXES)),
                NamedSharding(mesh, attention.spec_for(CI_ATTN_KV_AXES)),
                NamedSharding(mesh, attention.spec_for(CI_ATTN_OUT_AXES)),
                tuple(bank_in for _ in self.expert_gate),
                tuple(bank_in for _ in self.expert_up),
                tuple(bank_out for _ in self.expert_down),
                NamedSharding(mesh, ffn.spec_for(CI_FFN_IN_AXES)),
                NamedSharding(mesh, ffn.spec_for(CI_FFN_IN_AXES)),
                NamedSharding(mesh, ffn.spec_for(CI_FFN_OUT_AXES)),
            ),
        )
        if self.norm_scales is not None:
            vectors = placement.vectors
            norm = _vector_sharding(vectors, ("stack", "d_model"), self.norm_scales[0].shape)
            placed = eqx.tree_at(lambda b: b.norm_scales, placed, (norm, norm))
        return placed

    def __call__(
        self,
        x: Float[Array, "b t d"],
        inv_freq: Array,
        dispatch: MoEDispatch,
        *,
        placement: CIFnPlacement | None,
    ) -> tuple[Float[Array, "b t d"], tuple[Array, ...]]:
        """Returns the residual plus each router's expert hidden states in JOB space —
        the last block's are the fused narrow heads' inputs, on the same jobs the
        slots' swiglus already ran."""
        attn_scale, mlp_scale = (None, None) if self.norm_scales is None else self.norm_scales
        x = _attention_half(
            x,
            wq=self.wq,
            wk=self.wk,
            wv=self.wv,
            wo=self.wo,
            attention=self.attention,
            inv_freq=inv_freq,
            norm_scale=attn_scale,
            eps=self.eps,
            placement=placement,
            # The MoE arch has no padded consumer: prompt analysis (the one padded
            # caller) dispatches on the dense chunkwise arch only.
            valid_token_count=None,
        )
        h = _rms_norm_maybe_scaled(x, mlp_scale, self.eps)
        n_routers = len(self.expert_gate)
        routed = jnp.zeros_like(x)
        hidden_jobs: list[Array] = []
        for router in range(n_routers):
            x_jobs = dispatch.gather(router, h)
            gate = dispatch.expert_matmul(router, x_jobs, self.expert_gate[router])
            up = dispatch.expert_matmul(router, x_jobs, self.expert_up[router])
            hidden = jax.nn.silu(gate) * up
            down = dispatch.expert_matmul(router, hidden, self.expert_down[router])
            routed = routed + dispatch.combine(router, down)
            hidden_jobs.append(hidden)
        shared_gate = _ci_linear(
            h, self.shared_gate, placement, "ffn", CI_FFN_IN_AXES[1:], transposed=False
        )
        shared_up = _ci_linear(
            h, self.shared_up, placement, "ffn", CI_FFN_IN_AXES[1:], transposed=False
        )
        shared = _ci_linear(
            jax.nn.silu(shared_gate) * shared_up,
            self.shared_down,
            placement,
            "ffn",
            CI_FFN_OUT_AXES[1:],
            transposed=False,
        )
        return x + routed / n_routers + shared, tuple(hidden_jobs)


class DenseCIHead(eqx.Module):
    """One full-emission site head: `x_final @ w + b -> [*leading, C]`."""

    w: Float[Array, "d_model C"]
    b: Float[Array, " C"]


class ExpertCIHead(eqx.Module):
    """One narrow-emission site head, FUSED into the expert slots: block (e, :) reads
    the LAST trunk block's (router, e) expert hidden state on that slot's jobs — the
    same jobs the slot's swiglu already ran, no extra gather — and emits the slot's `c`
    preactivations, token-major `[*leading, k·c]`. The head's parameters activate
    exactly when its (layer, expert) is routed. Biasless like the expert banks: the
    leaf is exactly the routed-matmul operand shape."""

    w: Float[Array, "E expert_ffn c"]
    router: int = eqx.field(static=True)


MoECIHead = DenseCIHead | ExpertCIHead
"""Per-slot head union: the static discriminator rides the treedef, so the stacked
bundle stays rectangular per slot while slots differ in emission."""


class MoEChunkTransformer(eqx.Module):
    """ONE MoE chunk: its (already assembled, concatenated) input `[b, t, total_d_in]`
    → in_proj → `n_blocks` `MoECIBlock`s (every block dispatching on all `R` captured
    routings) → one head per output slot: `DenseCIHead`s read the final residual
    full-width; `ExpertCIHead`s read the last block's job-space expert hiddens and emit
    `NarrowCI` bundles carrying the dispatch's own router indices. In the bundle every
    array leaf carries a leading `n_chunks` axis and the module runs under a
    `jax.lax.scan` over that axis, exactly as `ChunkTransformer` does."""

    in_proj_w: Float[Array, "total_d_in d_model"]
    in_proj_b: Float[Array, " d_model"]
    blocks: list[MoECIBlock]
    heads: tuple[MoECIHead, ...]

    def shardings(self, mesh: Mesh, placement: CIFnPlacement) -> "MoEChunkTransformer":
        """in_proj at ci_fn/input, dense heads at ci_fn/output, expert heads at the
        ci_fn moe expert_head rows; biases at the vectors row."""
        moe = placement.moe
        assert moe is not None, "MoEChunkTransformer placement needs the table's moe families"
        input_row = placement.input.optimizer_state
        output_row = placement.output.optimizer_state
        head_row = moe.expert_head.optimizer_state
        input_row.validate_shape(CI_INPUT_AXES, self.in_proj_w.shape)
        vectors = placement.vectors
        placed_heads: list[MoECIHead] = []
        for head in self.heads:
            match head:
                case DenseCIHead():
                    output_row.validate_shape(CI_OUTPUT_AXES, head.w.shape)
                    placed_heads.append(
                        DenseCIHead(
                            w=NamedSharding(mesh, output_row.spec_for(CI_OUTPUT_AXES)),  # pyright: ignore[reportArgumentType]
                            b=_vector_sharding(vectors, ("stack", "C"), head.b.shape),  # pyright: ignore[reportArgumentType]
                        )
                    )
                case ExpertCIHead():
                    head_row.validate_shape(CI_EXPERT_HEAD_AXES, head.w.shape)
                    placed_heads.append(
                        ExpertCIHead(
                            w=NamedSharding(mesh, head_row.spec_for(CI_EXPERT_HEAD_AXES)),  # pyright: ignore[reportArgumentType]
                            router=head.router,
                        )
                    )
        return eqx.tree_at(
            lambda ct: (ct.in_proj_w, ct.in_proj_b, ct.blocks, ct.heads),
            self,
            (
                NamedSharding(mesh, input_row.spec_for(CI_INPUT_AXES)),
                _vector_sharding(vectors, ("stack", "d_model"), self.in_proj_b.shape),
                [b.shardings(mesh, placement) for b in self.blocks],
                tuple(placed_heads),
            ),
        )

    def __call__(
        self,
        x: Float[Array, "b t total_d_in"],
        inv_freq: Array,
        dispatch: MoEDispatch,
        *,
        placement: CIFnPlacement | None,
    ) -> tuple[SiteCI, ...]:
        x = (
            _ci_linear(x, self.in_proj_w, placement, "input", CI_INPUT_AXES[1:], transposed=False)
            + self.in_proj_b
        )
        hidden_jobs: tuple[Array, ...] = ()
        for block in self.blocks:
            x, hidden_jobs = block(x, inv_freq, dispatch, placement=placement)
        outputs: list[SiteCI] = []
        for head in self.heads:
            match head:
                case DenseCIHead(w=w, b=b):
                    outputs.append(
                        _ci_linear(x, w, placement, "output", CI_OUTPUT_AXES[1:], transposed=False)
                        + b
                    )
                case ExpertCIHead(w=w, router=router):
                    values_jobs = dispatch.expert_matmul(router, hidden_jobs[router], w)
                    outputs.append(
                        NarrowCI(
                            values=dispatch.narrow_values(router, values_jobs),
                            router_indices=dispatch.router_ids(router),
                            n_experts=w.shape[0],
                        )
                    )
        return tuple(outputs)


@dataclass(frozen=True)
class _MoEChunkMeta:
    """Per-chunk static routing, index-aligned with the stacked `chunks` leading axis."""

    input_taps: tuple[str, ...]
    routing: tuple[RoutingTap, ...]
    slots: tuple[MoESlot, ...]


class MoEChunkwiseTransformerCIFn(eqx.Module):
    """`ChunkwiseTransformerCIFn`'s MoE sibling: stacked `MoEChunkTransformer`s under a
    `jax.lax.scan` with per-chunk remat. Expert sites emit `NarrowCI` bundles — each
    narrow site's values leave this fn already married to the router indices that key
    them (the slot's `router` tap); dense sites emit full-width arrays. Each chunk's
    input concatenates its RMS-normed activation taps with one dense `[.., E]`
    routing-weight vector per covered layer (F2): the weights are scattered by the
    captured ids and scaled by a FIXED k (a k-sparse vector's own RMS would mis-scale
    it), so a uniformly-routed token's entries are O(1) like the normed taps'."""

    chunks: MoEChunkTransformer  # arrays stacked along leading n_chunks (+ stack_pad)
    inv_freq: Array  # shared across chunks (RoPE buffer); NOT mapped

    capture_keys: CaptureKeys = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)
    chunk_meta: tuple[_MoEChunkMeta, ...] = eqx.field(static=True)
    stack_pad: int = eqx.field(static=True)
    """As on `ChunkwiseTransformerCIFn`: the enumerated persist pad trailing the real
    chunks on every `chunks` leaf."""
    n_experts: int = eqx.field(static=True)
    grouped_matmul_backend: GroupedMatmulBackend = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)

    def shardings(self, mesh: Mesh, placement: CIFnPlacement) -> "MoEChunkwiseTransformerCIFn":
        """The stacked per-chunk transformer's persist layout (`MoEChunkTransformer.
        shardings` at the padded chunk extent); `inv_freq` replicates."""
        _validate_chunk_stack(self, placement.chunks, self.chunks.in_proj_w.shape[0])
        return eqx.tree_at(
            lambda f: (f.chunks, f.inv_freq),
            self,
            (self.chunks.shardings(mesh, placement), NamedSharding(mesh, P())),
        )

    def _chunk_input(
        self, meta: _MoEChunkMeta, taps: dict[str, Array], placement: CIFnPlacement | None
    ) -> Array:
        parts = [
            # Both boundaries matter, as on the dense chunkwise fn: keep the cached tap
            # TP-replicated through the RMS reduction, and stop the in_proj slice from
            # sinking backward through it.
            _constrain_ci_activation(
                _weightless_rms_norm(
                    _constrain_ci_activation(taps[key], placement, "feature"), self.eps
                ),
                placement,
                "feature",
            )
            for key in meta.input_taps
        ]
        for tap in meta.routing:
            ids = taps[tap.ids_key]
            weights = taps[tap.weights_key]
            k = ids.shape[-1]
            dense = jnp.sum(
                jax.nn.one_hot(ids, self.n_experts, dtype=weights.dtype) * weights[..., None],
                axis=-2,
            )
            parts.append(_constrain_ci_activation(dense * k, placement, "feature"))
        return jnp.concatenate(parts, axis=-1)

    def __call__(
        self,
        taps: dict[str, Array],
        *,
        remat: bool,
        placement: CIFnPlacement | None,
    ) -> CI:
        assert self.stack_pad == 0, (
            f"the chunk scan runs the compute residents, which carry no persist pads "
            f"(got stack_pad={self.stack_pad}); enter through materialize_ci_compute_weights"
        )
        per_chunk_in = [self._chunk_input(meta, taps, placement) for meta in self.chunk_meta]
        stacked_in = jnp.stack(per_chunk_in, axis=0)  # [n_chunks, b, t, total_d_in]
        stacked_ids = jnp.stack(
            [jnp.stack([taps[tap.ids_key] for tap in m.routing]) for m in self.chunk_meta]
        )  # [n_chunks, R, b, t, k]
        stacked_weights = jnp.stack(
            [jnp.stack([taps[tap.weights_key] for tap in m.routing]) for m in self.chunk_meta]
        )
        inv_freq = jax.lax.stop_gradient(self.inv_freq)
        chunk_arrays, chunk_static = eqx.partition(self.chunks, eqx.is_array)

        def run_chunk(
            _: None, scanned: tuple[MoEChunkTransformer, Array, Array, Array]
        ) -> tuple[None, tuple[SiteCI, ...]]:
            chunk_array, chunk_input, ids, weights = scanned
            chunk = eqx.combine(chunk_array, chunk_static)
            dispatch = _moe_ci_dispatch(
                ids, weights, self.n_experts, placement, self.grouped_matmul_backend
            )
            return None, chunk(chunk_input, inv_freq, dispatch, placement=placement)

        # Per-CHUNK checkpoint of the scan body in BOTH modes, exactly as the dense
        # chunkwise fn spells it: `remat` controls only whether chunk ACTIVATIONS are
        # recomputed, never the entry weight gather.
        policy = (
            jax.checkpoint_policies.nothing_saveable
            if remat
            else jax.checkpoint_policies.dots_saveable
        )
        body = jax.checkpoint(run_chunk, policy=policy)
        _, stacked_per_slot = jax.lax.scan(
            body, None, (chunk_arrays, stacked_in, stacked_ids, stacked_weights)
        )
        preactivations: SiteDict = {}
        for chunk_idx, meta in enumerate(self.chunk_meta):
            for slot, moe_slot in enumerate(meta.slots):
                # Slices both emissions: a bare array, or a NarrowCI whose leaves each
                # carry the scanned n_chunks axis.
                preactivations[moe_slot.site] = jax.tree.map(
                    lambda a, i=chunk_idx: a[i], stacked_per_slot[slot]
                )
        return CI.from_preactivations(preactivations)


def moe_ns_compute_shardings(
    ci_fn: MoEChunkwiseTransformerCIFn, mesh: Mesh, placement: CIFnPlacement
) -> MoEChunkwiseTransformerCIFn:
    """`ns_compute_shardings`' MoE sibling: every muon-labeled weight position carries
    its family's `ns_compute` waypoint row (expert banks and fused heads at the moe
    rows); every other position rides through untouched."""
    moe = placement.moe
    assert moe is not None, "MoE CI muon staging needs the table's moe families"

    def staging(weights: CIWeightPlacement) -> NamedSharding:
        return ns_staging_sharding(weights.ns_compute, mesh)

    attention, ffn = staging(placement.attention), staging(placement.ffn)
    expert_ffn, expert_head = staging(moe.expert_ffn), staging(moe.expert_head)

    def where(f: MoEChunkwiseTransformerCIFn) -> tuple[Array, ...]:
        locations: list[Array] = [f.chunks.in_proj_w]
        locations += [head.w for head in f.chunks.heads]
        for block in f.chunks.blocks:
            locations += [block.wq, block.wk, block.wv, block.wo]
            locations += [*block.expert_gate, *block.expert_up, *block.expert_down]
            locations += [block.shared_gate, block.shared_up, block.shared_down]
        return tuple(locations)

    values: list[NamedSharding] = [staging(placement.input)]
    for head in ci_fn.chunks.heads:
        match head:
            case DenseCIHead():
                values.append(staging(placement.output))
            case ExpertCIHead():
                values.append(expert_head)
    for block in ci_fn.chunks.blocks:
        values += [attention] * 4
        values += [expert_ffn] * (
            len(block.expert_gate) + len(block.expert_up) + len(block.expert_down)
        )
        values += [ffn] * 3
    return eqx.tree_at(where, ci_fn, tuple(values))


def _reconstruct_moe_ci_compute_weights(
    chunks: MoEChunkTransformer, placement: CIFnPlacement | None
) -> MoEChunkTransformer:
    """`_reconstruct_ci_compute_weights` over the MoE chunk module: the expert banks and
    fused heads enter through the MoE families' rows. No-op off-mesh."""
    if jax.sharding.get_abstract_mesh().empty:
        return chunks
    assert placement is not None, "on-mesh CI compute-weight materialization requires placement"
    moe = placement.moe
    assert moe is not None, "MoE CI compute materialization needs the table's moe families"
    census = placement.chunks

    def enter(x: Array, weights: CIWeightPlacement, axes: Axes) -> Array:
        return materialize_reduced_weights(
            x,
            census=census,
            source=weights.optimizer_state,
            destination=weights.compute_weights,
            axes=axes,
        )

    def vector(x: Array) -> Array:
        return strip_stack_pad(x, census)

    def head(h: MoECIHead) -> MoECIHead:
        match h:
            case DenseCIHead():
                return replace(h, w=enter(h.w, placement.output, CI_OUTPUT_AXES), b=vector(h.b))
            case ExpertCIHead():
                return replace(h, w=enter(h.w, moe.expert_head, CI_EXPERT_HEAD_AXES))

    def block(blk: MoECIBlock) -> MoECIBlock:
        return replace(
            blk,
            wq=enter(blk.wq, placement.attention, CI_ATTN_Q_AXES),
            wk=enter(blk.wk, placement.attention, CI_ATTN_KV_AXES),
            wv=enter(blk.wv, placement.attention, CI_ATTN_KV_AXES),
            wo=enter(blk.wo, placement.attention, CI_ATTN_OUT_AXES),
            expert_gate=tuple(
                enter(w, moe.expert_ffn, CI_EXPERT_FFN_IN_AXES) for w in blk.expert_gate
            ),
            expert_up=tuple(enter(w, moe.expert_ffn, CI_EXPERT_FFN_IN_AXES) for w in blk.expert_up),
            expert_down=tuple(
                enter(w, moe.expert_ffn, CI_EXPERT_FFN_OUT_AXES) for w in blk.expert_down
            ),
            shared_gate=enter(blk.shared_gate, placement.ffn, CI_FFN_IN_AXES),
            shared_up=enter(blk.shared_up, placement.ffn, CI_FFN_IN_AXES),
            shared_down=enter(blk.shared_down, placement.ffn, CI_FFN_OUT_AXES),
            norm_scales=None
            if blk.norm_scales is None
            else (vector(blk.norm_scales[0]), vector(blk.norm_scales[1])),
        )

    return replace(
        chunks,
        in_proj_w=enter(chunks.in_proj_w, placement.input, CI_INPUT_AXES),
        in_proj_b=vector(chunks.in_proj_b),
        blocks=[block(blk) for blk in chunks.blocks],
        heads=tuple(head(h) for h in chunks.heads),
    )


def _moe_slot_signature(
    chunk: MoEChunk, site_spec: dict[str, SiteSpec], n_experts: int
) -> tuple[tuple[object, ...], ...]:
    """One chunk's per-slot (emission, shape) signature — equal across chunks ⟺ the
    stacked bundle is rectangular and every slot means the same thing in every chunk."""
    signature: list[tuple[object, ...]] = []
    for slot in chunk.slots:
        spec = site_spec[slot.site]
        match slot:
            case FullSlot():
                signature.append(("full", spec.C))
            case NarrowSlot(router=router):
                factorization = spec.factorization
                assert isinstance(factorization, ExpertBlocked), (
                    f"narrow slot {slot.site!r} needs an expert-blocked site, "
                    f"got {type(factorization).__name__}"
                )
                assert factorization.n_experts == n_experts, (
                    slot.site,
                    factorization.n_experts,
                    n_experts,
                )
                assert 0 <= router < len(chunk.routing), (slot.site, router, len(chunk.routing))
                signature.append(("narrow", router, factorization.c_per_expert))
    return tuple(signature)


def _init_moe_chunk_transformer(
    arch: MoEChunkwiseTransformerCIArch,
    slot_signature: tuple[tuple[object, ...], ...],
    n_routers: int,
    key: PRNGKeyArray,
) -> MoEChunkTransformer:
    """One MoE chunk's params under the chunkwise Kaiming scheme: relu-gain (√2) on
    in_proj / gate / up projections, linear gain (1) on down projections and heads,
    PyTorch-default `U(±1/√fan_in)` on the attention projections, zero biases. Each
    consumer takes its OWN explicit key — the split counts live next to their use."""
    relu_gain = 2.0**0.5
    d, di, ds = arch.d_model, arch.expert_ffn_hidden, arch.shared_ffn_hidden
    n_experts = arch.n_experts
    d_kv = (d // arch.attention.n_heads) * arch.attention.n_kv_heads

    def kaiming(k: PRNGKeyArray, shape: tuple[int, ...], fan_in: int, gain: float) -> Array:
        return jax.random.normal(k, shape) * (gain / fan_in**0.5)

    def attn_default(k: PRNGKeyArray, shape: tuple[int, ...], fan_in: int) -> Array:
        bound = 1.0 / fan_in**0.5
        return jax.random.uniform(k, shape, minval=-bound, maxval=bound)

    def block(bkey: PRNGKeyArray) -> MoECIBlock:
        # 4 attention + 3 per router bank + 3 shared draws; the split count derives
        # every key, so it lives here, next to the draws.
        kq, kk, kv, ko, *rest = jax.random.split(bkey, 4 + 3 * n_routers + 3)
        gate_keys, up_keys = rest[:n_routers], rest[n_routers : 2 * n_routers]
        down_keys = rest[2 * n_routers : 3 * n_routers]
        ksg, ksu, ksd = rest[3 * n_routers :]
        norm_scales = (jnp.ones((d,)), jnp.ones((d,))) if arch.learned_norm_scale else None
        return MoECIBlock(
            wq=attn_default(kq, (d, d), d),
            wk=attn_default(kk, (d_kv, d), d),
            wv=attn_default(kv, (d_kv, d), d),
            wo=attn_default(ko, (d, d), d),
            expert_gate=tuple(kaiming(k, (n_experts, d, di), d, relu_gain) for k in gate_keys),
            expert_up=tuple(kaiming(k, (n_experts, d, di), d, relu_gain) for k in up_keys),
            expert_down=tuple(kaiming(k, (n_experts, di, d), di, 1.0) for k in down_keys),
            shared_gate=kaiming(ksg, (d, ds), d, relu_gain),
            shared_up=kaiming(ksu, (d, ds), d, relu_gain),
            shared_down=kaiming(ksd, (ds, d), ds, 1.0),
            norm_scales=norm_scales,
            attention=arch.attention,
            eps=CI_FN_RMS_EPS,
        )

    in_key, heads_key, *block_keys = jax.random.split(key, arch.n_blocks + 2)
    head_keys = jax.random.split(heads_key, len(slot_signature))
    heads: list[MoECIHead] = []
    for slot_sig, head_key in zip(slot_signature, head_keys, strict=True):
        match slot_sig:
            case ("full", int() as c_full):
                heads.append(
                    DenseCIHead(w=kaiming(head_key, (d, c_full), d, 1.0), b=jnp.zeros((c_full,)))
                )
            case ("narrow", int() as router, int() as c):
                heads.append(
                    ExpertCIHead(w=kaiming(head_key, (n_experts, di, c), di, 1.0), router=router)
                )
            case _:
                raise AssertionError(slot_sig)
    return MoEChunkTransformer(
        in_proj_w=kaiming(in_key, (arch.input_dim, d), arch.input_dim, relu_gain),
        in_proj_b=jnp.zeros((d,)),
        blocks=[block(bk) for bk in block_keys],
        heads=tuple(heads),
    )


def init_moe_chunkwise_transformer_ci_fn(
    arch: MoEChunkwiseTransformerCIArch, sites: tuple[SiteSpec, ...], key: PRNGKeyArray
) -> MoEChunkwiseTransformerCIFn:
    """Validate the output partition and chunk homogeneity as the dense chunkwise init
    does — plus: every `NarrowSlot` names an expert-blocked site whose factorization
    matches the arch's `n_experts`, every `FullSlot` a dense site, and every chunk
    covers one shared router count — then build stacked chunk params under the same
    Kaiming scheme."""
    site_spec = {s.name: s for s in sites}
    covered = [slot.site for chunk in arch.chunks for slot in chunk.slots]
    assert sorted(covered) == sorted(site_spec), "chunks must partition sites"
    assert len(covered) == len(set(covered)), "chunks overlap on an output site"
    assert arch.n_blocks >= 1, (
        f"the MoE chunkwise arch needs n_blocks >= 1 ({arch.n_blocks}): the fused narrow "
        "heads read the last block's expert hiddens"
    )
    router_counts = {len(chunk.routing) for chunk in arch.chunks}
    assert len(router_counts) == 1, f"chunks not homogeneous in router count: {router_counts}"
    (n_routers,) = router_counts
    signatures = {_moe_slot_signature(chunk, site_spec, arch.n_experts) for chunk in arch.chunks}
    assert len(signatures) == 1, (
        f"chunks not homogeneous in per-slot (emission, shape) signature (the per-slot "
        f"heads stack slot-by-slot across chunks): {signatures}"
    )
    (slot_signature,) = signatures
    assert all(chunk.input_taps for chunk in arch.chunks), "each chunk needs an input tap"

    n_heads = arch.attention.n_heads
    hd = arch.d_model // n_heads
    assert arch.d_model % n_heads == 0 and hd % 2 == 0, (arch.d_model, n_heads)
    inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / hd))

    chunk_keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(len(arch.chunks)))
    stacked: MoEChunkTransformer = eqx.filter_vmap(
        lambda k: _init_moe_chunk_transformer(arch, slot_signature, n_routers, k)
    )(chunk_keys)

    return MoEChunkwiseTransformerCIFn(
        chunks=stacked,
        inv_freq=inv_freq,
        capture_keys=arch.capture_keys,
        output_names=tuple(name for chunk in arch.chunks for name in chunk.output_sites),
        chunk_meta=tuple(
            _MoEChunkMeta(chunk.input_taps, chunk.routing, chunk.slots) for chunk in arch.chunks
        ),
        stack_pad=0,
        n_experts=arch.n_experts,
        grouped_matmul_backend=arch.grouped_matmul_backend,
        eps=CI_FN_RMS_EPS,
        has_position_axis=True,
    )


# ----------------------- the chunk stack's persist pads -----------------------
# Both chunkwise fns stack every array leaf along a leading chunk axis; a placement whose
# persist rows cut that axis pads it (`StackCensus`, resolved in `resolve_ci_placement`)
# exactly as the V/U groups pad theirs: trailing all-zero slots on EVERY leaf of the
# stacked chunk module, appended by the placed init (`pad_ci_fn`), stripped at the entry
# (`_reconstruct_ci_compute_weights`), never named by `chunk_meta`, never scanned.

ChunkStackedCIFn = ChunkwiseTransformerCIFn | MoEChunkwiseTransformerCIFn


def _validate_chunk_stack(fn: ChunkStackedCIFn, census: StackCensus, stack_extent: int) -> None:
    """BOUNDARY VALIDATION of the resolved chunk census against the fn actually held —
    its static chunk routing, its pad enumeration, and its leaves' stack extent.
    Disagreement is an upstream bug and dies here."""
    assert census.stack_len == len(fn.chunk_meta), (
        f"placement expects a {census.stack_len}-chunk CI fn; this fn routes "
        f"{len(fn.chunk_meta)} chunks"
    )
    assert census.stack_pad == fn.stack_pad, (
        f"placement expects a chunk-stack pad of {census.stack_pad}; this fn enumerates "
        f"{fn.stack_pad}"
    )
    assert stack_extent == census.padded_stack_len, (
        "chunk leaves disagree with the padded census extent",
        stack_extent,
        census,
    )


def _pad_chunk_stack[Chunks: eqx.Module](chunks: Chunks, pad: int) -> Chunks:
    return jax.tree.map(
        lambda leaf: jnp.concatenate([leaf, jnp.zeros((pad, *leaf.shape[1:]), leaf.dtype)]),
        chunks,
    )


def _resident_chunk_stack[Fn: ChunkStackedCIFn](
    fn: Fn, resident_chunks: eqx.Module, placement: CIFnPlacement | None
) -> Fn:
    """What the scan consumes: the entry's resident chunk module on a fn that enumerates
    no pads. Boundary check of the entry's leaf enumeration — every resident leaf carries
    exactly the real chunk extent, so a leaf the entry did not name cannot reach the scan
    padded."""
    if placement is None:
        assert fn.stack_pad == 0, f"an unplaced CI fn carries no persist pads: {fn.stack_pad}"
        return replace(fn, chunks=resident_chunks)
    census = placement.chunks
    _validate_chunk_stack(fn, census, fn.chunks.in_proj_w.shape[0])
    for leaf in jax.tree.leaves(resident_chunks):
        assert leaf.shape[0] == census.stack_len, (
            "a chunk leaf reached the scan at the padded extent",
            leaf.shape,
            census,
        )
    return replace(fn, chunks=resident_chunks, stack_pad=0)


def pad_ci_fn(fn: CIFn, placement: CIFnPlacement | None) -> CIFn:
    """The one constructor of a padded persist tree: append the placement's chunk-stack
    pad as trailing all-zero slots on every leaf of the stacked chunk module and
    enumerate it on `stack_pad`. The MLP fns carry no chunk stack and run unplaced."""
    match fn:
        case ChunkwiseTransformerCIFn() | MoEChunkwiseTransformerCIFn():
            assert placement is not None, f"{type(fn).__name__} is placed by its rows"
            assert fn.stack_pad == 0, f"already padded: {fn.stack_pad}"
            census = placement.chunks
            assert census.stack_len == len(fn.chunk_meta), (census, len(fn.chunk_meta))
            if census.stack_pad == 0:
                return fn
            return replace(
                fn, chunks=_pad_chunk_stack(fn.chunks, census.stack_pad), stack_pad=census.stack_pad
            )
        case LayerwiseMLPCIFn() | GlobalMLPCIFn():
            assert placement is None, f"{type(fn).__name__} runs unplaced"
            return fn
        case _:
            raise AssertionError(f"unknown CI fn {type(fn)}")


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
        (`placement.batch_axes`) — the master + Adam state shard ÷N. 1-D biases
        replicate. The MLP is single-shot (no scan), so there is no compute
        reconstruction; GSPMD gathers as needed (trivial at the toy's small device count).
        Asserts every output dim tiles its actual shard count — not the total device count,
        which over-counts by ×tp."""
        shard_out = NamedSharding(mesh, P(None, batch_axes(mesh)))
        repl = NamedSharding(mesh, P())
        n = math.prod(mesh.shape[a] for a in batch_axes(mesh))
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


CIFnArch = (
    ChunkwiseTransformerCIArch
    | MoEChunkwiseTransformerCIArch
    | LayerwiseMLPCIArch
    | GlobalMLPCIArch
)
"""Every CI-fn architecture. Construction goes through `build_ci_fn`; sharding/placement is
a separate, scale-driven concern (see `init_placed`), never coupled to arch type."""


@dataclass(frozen=True)
class _WeightLeaf:
    """One arch-known stacked weight leaf: the family whose rows it enters through, its
    semantic axes, and its per-chunk shape (the stack axis prepended at validation)."""

    family: CIWeightPlacement
    axes: Axes
    per_chunk: tuple[int, ...]


@dataclass(frozen=True)
class _VectorLeaf:
    """One arch-known stacked vector leaf, resting at the vectors row."""

    axes: Axes
    per_chunk: tuple[int, ...]


_ChunkLeaf = _WeightLeaf | _VectorLeaf


def _validate_chunk_leaves(
    census: StackCensus, rows: CIFnRows, leaves: tuple[_ChunkLeaf, ...]
) -> None:
    """Construction-time tiling of every arch-known persist leaf at the PADDED chunk
    extent — the CI twin of `_resolve_group_census`'s shape validation, so a mesh the
    CI masters (or a padded chunk stack's entry waypoint) cannot tile refuses where the
    arch and the rows first meet. The per-slot heads' C is a site fact and is validated
    where the fn is placed (`shardings`)."""
    for leaf in leaves:
        shape = (census.padded_stack_len, *leaf.per_chunk)
        match leaf:
            case _WeightLeaf(family=family, axes=axes):
                validate_stacked_leaf(
                    census, family.optimizer_state, family.compute_weights, axes, shape
                )
            case _VectorLeaf(axes=axes):
                rows.vectors.validate_shape(axes, shape)


def _attention_leaves(
    rows: CIFnRows, attention: CIAttention, d_model: int
) -> tuple[_ChunkLeaf, ...]:
    d_kv = (d_model // attention.n_heads) * attention.n_kv_heads
    return (
        _WeightLeaf(rows.attention, CI_ATTN_Q_AXES, (d_model, d_model)),
        _WeightLeaf(rows.attention, CI_ATTN_KV_AXES, (d_kv, d_model)),
        _WeightLeaf(rows.attention, CI_ATTN_OUT_AXES, (d_model, d_model)),
    )


def _validate_ci_head_split(rows: CIFnRows, attention: CIAttention) -> None:
    """The attention head split's divisibility: the split parks each projection's flat
    assignment on its head-COUNT axis (`CIBlock.__call__`'s `heads`), so both counts
    must tile — under GQA `kv_head` is the narrow one. There is no replication
    fallback; a user who WANTS replicated K/V heads authors an explicit table with
    `kv_head` unmapped."""
    rows.activations.validate_shape(("q_head",), (attention.n_heads,))
    rows.activations.validate_shape(("kv_head",), (attention.n_kv_heads,))


def resolve_ci_placement(arch: CIFnArch, rules: PlacementRules | None) -> CIFnPlacement | None:
    """THE one CI-placement resolution, at run assembly: the chunkwise transformers
    consume the run's CI rows (the MoE arch additionally requires the table's MoE
    families) and resolve their chunk stack's census — the pad that makes the persist
    stack tile every row its leaves rest at (`CIFnRows.chunk_persist_rows`, plus the MoE
    families' masters); the MLP archs run unplaced. Downstream code receives the
    already-paired `PlacedCIFn` (or, on the muon path, this resolved value) — never the
    raw rows next to a fn.

    Resolution is also the construction-time refusal point for everything the arch
    alone determines: the head split (`_validate_ci_head_split`) and every arch-known
    master leaf's tiling at the padded extent (`_validate_chunk_leaves`)."""
    match arch:
        case ChunkwiseTransformerCIArch():
            if rules is None:
                return None
            rows = rules.ci_fn
            _validate_ci_head_split(rows, arch.attention)
            census = resolve_stack_census(len(arch.chunks), rows.chunk_persist_rows)
            d, ffn = arch.d_model, arch.ffn_hidden
            leaves: tuple[_ChunkLeaf, ...] = (
                _WeightLeaf(rows.input, CI_INPUT_AXES, (arch.input_dim, d)),
                _VectorLeaf(("stack", "d_model"), (d,)),
            )
            if arch.n_blocks > 0:
                leaves += (
                    *_attention_leaves(rows, arch.attention, d),
                    _WeightLeaf(rows.ffn, CI_FFN_IN_AXES, (d, ffn)),
                    _WeightLeaf(rows.ffn, CI_FFN_OUT_AXES, (ffn, d)),
                    _VectorLeaf(("stack", "ffn_hidden"), (ffn,)),
                )
            _validate_chunk_leaves(census, rows, leaves)
            return CIFnPlacement.resolved(rows, census)
        case MoEChunkwiseTransformerCIArch():
            if rules is None:
                return None
            rows = rules.ci_fn
            moe = rows.moe
            assert moe is not None, (
                "the MoE chunkwise CI arch needs a placement table carrying the ci_fn moe "
                "families (expert_ffn + expert_head) — the zero1-replicated-resident-moe "
                "preset; this table binds none"
            )
            _validate_ci_head_split(rows, arch.attention)
            # The tp split of the CI expert grid must cut whole experts, exactly as the
            # target's fused axes do.
            moe.expert_ffn.operands.validate_shape(("expert",), (arch.n_experts,))
            moe.expert_head.operands.validate_shape(("expert",), (arch.n_experts,))
            census = resolve_stack_census(
                len(arch.chunks),
                (
                    *rows.chunk_persist_rows,
                    moe.expert_ffn.optimizer_state,
                    moe.expert_head.optimizer_state,
                ),
            )
            d, di, ds, n_experts = (
                arch.d_model,
                arch.expert_ffn_hidden,
                arch.shared_ffn_hidden,
                arch.n_experts,
            )
            _validate_chunk_leaves(
                census,
                rows,
                (
                    _WeightLeaf(rows.input, CI_INPUT_AXES, (arch.input_dim, d)),
                    _VectorLeaf(("stack", "d_model"), (d,)),
                    *_attention_leaves(rows, arch.attention, d),
                    _WeightLeaf(moe.expert_ffn, CI_EXPERT_FFN_IN_AXES, (n_experts, d, di)),
                    _WeightLeaf(moe.expert_ffn, CI_EXPERT_FFN_OUT_AXES, (n_experts, di, d)),
                    _WeightLeaf(rows.ffn, CI_FFN_IN_AXES, (d, ds)),
                    _WeightLeaf(rows.ffn, CI_FFN_OUT_AXES, (ds, d)),
                ),
            )
            return CIFnPlacement.resolved(rows, census)
        case LayerwiseMLPCIArch() | GlobalMLPCIArch():
            return None


def build_ci_fn(arch: CIFnArch, sites: tuple[SiteSpec, ...], key: PRNGKeyArray) -> CIFn:
    """Construct the CI fn for `arch`, host-side and unsharded. Placement is applied by the
    caller by SCALE (mesh × C-divisibility), never by which arch this is."""
    match arch:
        case ChunkwiseTransformerCIArch():
            return init_chunkwise_transformer_ci_fn(arch, sites, key)
        case MoEChunkwiseTransformerCIArch():
            return init_moe_chunkwise_transformer_ci_fn(arch, sites, key)
        case LayerwiseMLPCIArch():
            return init_layerwise_mlp_ci_fn(arch, sites, key)
        case GlobalMLPCIArch():
            return init_global_mlp_ci_fn(arch, sites, key)
