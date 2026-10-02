"""Shared attention, dense transformer blocks, and seeded initialization."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, PRNGKeyArray

from param_decomp.attention import AttentionImplementation, placed_dot_product_attention
from param_decomp.core.axes import SemanticAxis
from param_decomp.core.components import activation_axes
from param_decomp.core.flops.types import ForwardBackwardFlops, MatrixParameters, ParameterCensus
from param_decomp.core.linear_plan import LinearPlan, placed_linear, value_mesh
from param_decomp.core.muon_stacked import StageInPlace
from param_decomp.core.placement import CIFnWeightPlacement, PlacedRule
from param_decomp.sequence import SequenceLayout
from param_decomp.target_ports.llama import apply_rope, rms_norm, rope_cos_sin

MatrixAxes = tuple[SemanticAxis, SemanticAxis]

# Attention keeps per-projection axes: q/o carry the query head axis, k/v the K/V head
# axis (narrower under GQA). Each name covers both spellings of that dimension — the
# flat `n * head_dim` projection width and the head COUNT of its split view.
CI_FN_ATTN_Q_AXES: MatrixAxes = ("q_head", "d_model")
CI_FN_ATTN_KV_AXES: MatrixAxes = ("kv_head", "d_model")
CI_FN_ATTN_OUT_AXES: MatrixAxes = ("d_model", "q_head")
CI_FN_FFN_IN_AXES: MatrixAxes = ("d_model", "ffn_hidden")
CI_FN_FFN_OUT_AXES: MatrixAxes = ("ffn_hidden", "d_model")
CI_FN_INPUT_AXES: MatrixAxes = ("input", "d_model")
CI_FN_OUTPUT_AXES: MatrixAxes = ("d_model", "C")


@dataclass(frozen=True)
class PlacedProjection:
    """One projection's weight rows paired with the activation row its linear runs
    between."""

    weights: CIFnWeightPlacement
    activations: PlacedRule

    def plan(
        self, stored_axes: MatrixAxes, activation_ndim: int, *, transposed: bool
    ) -> LinearPlan:
        """One stored matrix applied to an activation of rank `activation_ndim`,
        optionally transposed."""
        assert isinstance(self.activations.mesh, Mesh), type(self.activations.mesh)

        def spec(row: PlacedRule) -> P:
            stored = row.spec_for(stored_axes)
            return P(*reversed(stored)) if transposed else stored

        operand_axes = tuple(reversed(stored_axes)) if transposed else stored_axes
        input_axes = activation_axes(activation_ndim, operand_axes[0])
        output_axes = activation_axes(activation_ndim, operand_axes[1])
        return LinearPlan(
            mesh=self.activations.mesh,
            input=self.activations.spec_for(input_axes),
            operand_input=self.activations.spec_for(input_axes),
            resident_weight=spec(self.weights.compute_weights),
            operand=spec(self.weights.operands),
            output=self.activations.spec_for(output_axes),
            weight_reduced=self.weights.compute_weight_provenance(stored_axes),
        )


@dataclass(frozen=True)
class LocalProjection:
    """A projection with no placement: the plain matmul `x @ w`."""


ProjectionPlacement = PlacedProjection | LocalProjection


@dataclass(frozen=True)
class CIFnTransformerStructure[Value]:
    """One value for each part of the CI transformer layers: the input projection,
    the attention projections, the dense FFN, and the output heads."""

    input: Value
    attention: Value
    ffn: Value
    output: Value

    @staticmethod
    def uniform[Other](value: Other) -> "CIFnTransformerStructure[Other]":
        return CIFnTransformerStructure(input=value, attention=value, ffn=value, output=value)

    def map[Mapped](self, f: Callable[[Value], Mapped]) -> "CIFnTransformerStructure[Mapped]":
        return CIFnTransformerStructure(
            input=f(self.input), attention=f(self.attention), ffn=f(self.ffn), output=f(self.output)
        )


CI_FN_TRANSFORMER_MATRIX_AXES: CIFnTransformerStructure[tuple[MatrixAxes, ...]] = (
    CIFnTransformerStructure(
        input=(CI_FN_INPUT_AXES,),
        attention=(CI_FN_ATTN_Q_AXES, CI_FN_ATTN_KV_AXES, CI_FN_ATTN_OUT_AXES),
        ffn=(CI_FN_FFN_IN_AXES, CI_FN_FFN_OUT_AXES),
        output=(CI_FN_OUTPUT_AXES,),
    )
)
"""Each part's per-matrix axes, before any independent axes a variant stacks them on."""

CIFnTransformerPlacement = CIFnTransformerStructure[ProjectionPlacement]
"""Where each part of the CI transformer layers runs. A placed CI
architecture builds it from its own rows; an unplaced one uses
`LOCAL_CI_FN_TRANSFORMER_PLACEMENT`."""

LOCAL_CI_FN_TRANSFORMER_PLACEMENT: CIFnTransformerPlacement = CIFnTransformerStructure.uniform(
    LocalProjection()
)

CIFnTransformerMuonStaging = CIFnTransformerStructure[NamedSharding | StageInPlace]
"""Where stacked Muon stages each part."""

CI_FN_TRANSFORMER_STAGING_IN_PLACE: CIFnTransformerMuonStaging = CIFnTransformerStructure.uniform(
    StageInPlace()
)


def ci_fn_linear(
    x: Array,
    weight: Array,
    linear: ProjectionPlacement,
    stored_axes: MatrixAxes,
    *,
    transposed: bool,
) -> Array:
    """`x @ weight` (or its transpose), planned from `linear` when placed."""
    assert weight.ndim == 2, weight.shape
    operand = jnp.swapaxes(weight, -1, -2) if transposed else weight
    match linear:
        case LocalProjection():
            return x @ operand
        case PlacedProjection():
            return placed_linear(
                x, operand, linear.plan(stored_axes, x.ndim, transposed=transposed)
            )


def split_ci_fn_heads(projection: Array, n_heads: int) -> Array:
    """Split projection width into heads, preserving sharding on the head count."""
    mesh = value_mesh(projection)
    if mesh.empty:
        return einops.rearrange(projection, "b t (nh hd) -> b nh t hd", nh=n_heads)
    spec = jax.typeof(projection).sharding.spec
    return jax.lax.reshape(
        projection,
        (*projection.shape[:2], n_heads, projection.shape[2] // n_heads),
        out_sharding=NamedSharding(mesh, P(*spec[:2], spec[2], None)),
    ).transpose(0, 2, 1, 3)


CI_FN_RMS_EPS = float(jnp.finfo(jnp.float32).eps)
"""Matches torch's `F.rms_norm` default eps (`finfo(fp32).eps` ~1.19e-7); RMS upcasts to
fp32 internally, so this is the dtype that governs."""


def weightless_rms_norm(x: Array, eps: float) -> Array:
    return rms_norm(x, jnp.ones((x.shape[-1],), x.dtype), eps)


def normalized_tap_concatenation(
    taps: Sequence[Array], constrain_activation: Callable[[Array], Array], eps: float
) -> Array:
    """Every tap RMS-normalized independently, then concatenated along features. Each tap
    is pinned to the activation row on both sides of its RMS reduction: before it, so the
    reduction reads the whole feature axis, and after it, so the input projection's operand
    reshard starts from the normalized tap."""
    return jnp.concatenate(
        [constrain_activation(weightless_rms_norm(constrain_activation(tap), eps)) for tap in taps],
        axis=-1,
    )


def rms_norm_maybe_scaled(x: Array, scale: Array | None, eps: float) -> Array:
    """`scale is None` is the weightless norm — `ones` in x's dtype, i.e. today's numerics
    exactly (and no bf16→fp32 promotion, which an fp32 scale leaf would cause)."""
    if scale is None:
        return weightless_rms_norm(x, eps)
    return rms_norm(x, scale, eps)


CIFnAttentionMask = Literal["bidirectional", "causal"]


@dataclass(frozen=True)
class MHACIFnAttention:
    """Every query head carries its own K/V head."""

    n_heads: int
    implementation: AttentionImplementation
    mask: CIFnAttentionMask

    @property
    def n_kv_heads(self) -> int:
        return self.n_heads


@dataclass(frozen=True)
class GQACIFnAttention:
    """`n_heads // n_kv_heads` query heads share each K/V head, so `wk`/`wv` narrow to
    `n_kv_heads * head_dim`. head_dim, the RoPE tables, `wq`/`wo` and every sharding are
    identical to MHA — only the K/V projections change."""

    n_heads: int
    n_kv_heads: int
    implementation: AttentionImplementation
    mask: CIFnAttentionMask

    def __post_init__(self) -> None:
        assert self.n_heads % self.n_kv_heads == 0, (
            "n_heads must be divisible by n_kv_heads (each K/V head serves an equal group "
            f"of query heads): {self.n_heads} % {self.n_kv_heads}"
        )
        assert self.n_kv_heads < self.n_heads, (
            f"n_kv_heads == n_heads ({self.n_heads}) is MHA — use MHACIFnAttention rather than "
            "a degenerate GQA"
        )


CIFnAttention = MHACIFnAttention | GQACIFnAttention
"""The CI transformer's attention. Both arms answer `n_heads` and `n_kv_heads`, so the call
site never dispatches — but MHA derives its K/V count from the TYPE instead of leaving
`n_kv_heads == n_heads` as a convention a reader has to know, and cannot carry an explicit
one. GQA's grouping invariant is checked at construction, not at init."""
CIFnFfnKind = Literal["gelu", "swiglu"]
"""`gelu`: `Linear+b → GELU → Linear+b`. `swiglu`: a second projection gates the first —
`silu(h@wg + bg) * (h@w1 + b1) → Linear+b`. SwiGLU is a THIRD matrix, so it grows the MLP
~50% at a fixed `ffn_hidden`; iso-param means setting `ffn_hidden` to ~2/3. Nothing here
rescales it — the width is the config author's to state."""


def attention_flops(
    attention: CIFnAttention, width: int, batch_size: int, sequence_length: int
) -> int:
    kv_width = width // attention.n_heads * attention.n_kv_heads
    projections = 2 * batch_size * sequence_length * (2 * width**2 + 2 * width * kv_width)
    match attention.mask:
        case "bidirectional":
            n_pairs = sequence_length**2
        case "causal":
            n_pairs = sequence_length * (sequence_length + 1) // 2
    scores_and_values = 4 * batch_size * n_pairs * width
    return projections + scores_and_values


def gelu_ffn_flops(width: int, hidden: int, n_tokens: int) -> int:
    return 4 * n_tokens * width * hidden


def swiglu_ffn_flops(width: int, hidden: int, n_tokens: int) -> int:
    return 6 * n_tokens * width * hidden


def attention_half(
    x: Float[Array, "b t d"],
    *,
    wq: Array,
    wk: Array,
    wv: Array,
    wo: Array,
    attention: CIFnAttention,
    inv_freq: Array,
    norm_scale: Array | None,
    eps: float,
    placement: CIFnTransformerPlacement,
    sequence: SequenceLayout,
) -> Array:
    """The pre-norm RoPE attention half of a CI block, residual included —
    shared by dense and block-selected blocks, respecting the supplied sequence layout."""
    assert sequence.document_ids.shape == x.shape[:2], (sequence.document_ids.shape, x.shape)
    h = rms_norm_maybe_scaled(x, norm_scale, eps)

    def heads(  # [b, t, d] -> [b, nh, t, hd]  (RoPE layout)
        w: Array, n_head: int, stored_axes: MatrixAxes
    ) -> Array:
        proj = ci_fn_linear(h, w, placement.attention, stored_axes, transposed=True)
        return split_ci_fn_heads(proj, n_head)

    q = heads(wq, attention.n_heads, CI_FN_ATTN_Q_AXES)
    kv = attention.n_kv_heads
    k, v = heads(wk, kv, CI_FN_ATTN_KV_AXES), heads(wv, kv, CI_FN_ATTN_KV_AXES)
    cos, sin = rope_cos_sin(inv_freq, sequence.position_ids(), x.dtype)
    q, k = apply_rope(q, k, cos, sin)  # cos/sin broadcast over the head axis: any count
    qt, kt, vt = (einops.rearrange(a, "b nh t hd -> b t nh hd") for a in (q, k, v))
    mask = sequence.attention_mask()[:, None, :, :]
    mesh = value_mesh(qt)
    qkv_sharding = None if mesh.empty else NamedSharding(mesh, jax.typeof(qt).sharding.spec)
    y = placed_dot_product_attention(
        qt,
        kt,
        vt,
        mask,
        qkv_sharding=qkv_sharding,
        is_causal=attention.mask == "causal",
        implementation=attention.implementation,
    )
    return x + ci_fn_linear(
        einops.rearrange(y, "b t nh hd -> b t (nh hd)"),
        wo,
        placement.attention,
        CI_FN_ATTN_OUT_AXES,
        transposed=True,
    )


class CIFnBlock(eqx.Module):
    """Pre-norm block: RMSNorm → configured RoPE attention → residual;
    RMSNorm → FFN (`gelu` or `swiglu`) → residual.

    `attention` is the resolved variant: under `GQACIFnAttention` the K/V projections narrow to
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
    attention: CIFnAttention = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __call__(
        self,
        x: Float[Array, "b t d"],
        inv_freq: Array,
        *,
        placement: CIFnTransformerPlacement,
        sequence: SequenceLayout,
    ) -> Array:
        attn_scale, mlp_scale = (None, None) if self.norm_scales is None else self.norm_scales
        x = attention_half(
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
            sequence=sequence,
        )
        h = rms_norm_maybe_scaled(x, mlp_scale, self.eps)
        ffn = placement.ffn
        up = ci_fn_linear(h, self.w1, ffn, CI_FN_FFN_IN_AXES, transposed=False) + self.b1
        if self.gate is None:
            hidden = jax.nn.gelu(up, approximate=False)
        else:
            w_gate, b_gate = self.gate
            gate = ci_fn_linear(h, w_gate, ffn, CI_FN_FFN_IN_AXES, transposed=False) + b_gate
            hidden = jax.nn.silu(gate) * up
        return (
            x + ci_fn_linear(hidden, self.w2, ffn, CI_FN_FFN_OUT_AXES, transposed=False) + self.b2
        )


class TransformerArch(Protocol):
    """Backbone dimensions used by dense CI transformer initialization."""

    @property
    def d_model(self) -> int: ...

    @property
    def n_blocks(self) -> int: ...

    @property
    def attention(self) -> CIFnAttention: ...

    @property
    def ffn_hidden(self) -> int: ...

    @property
    def ffn_kind(self) -> CIFnFfnKind: ...

    @property
    def learned_norm_scale(self) -> bool: ...


@dataclass(frozen=True)
class TransformerParameters:
    """Initialized dense transformer parameters before architecture-specific stacking."""

    in_proj_w: Float[Array, "total_d_in d_model"]
    in_proj_b: Float[Array, " d_model"]
    blocks: list[CIFnBlock]
    out_ws: tuple[Float[Array, "d_model _C"], ...]
    out_bs: tuple[Float[Array, " _C"], ...]


def init_transformer_parameters(
    arch: TransformerArch,
    total_d_in: int,
    output_widths: tuple[int, ...],
    key: PRNGKeyArray,
) -> TransformerParameters:
    """Initialize input and FFN-in projections with Kaiming relu gain, output heads
    and FFN-out with linear gain, attention with uniform U(±1/√fan_in), and zero biases.

    Heads partition one concatenated random draw. Each block consumes six keys for
    GELU or seven for SwiGLU; the split count determines every projection.
    """
    relu_gain = 2.0**0.5
    d, ffn = arch.d_model, arch.ffn_hidden
    d_kv = (d // arch.attention.n_heads) * arch.attention.n_kv_heads  # narrower under GQA

    def kaiming(k: PRNGKeyArray, shape: tuple[int, ...], fan_in: int, gain: float) -> Array:
        return jax.random.normal(k, shape) * (gain / fan_in**0.5)

    def attn_default(k: PRNGKeyArray, shape: tuple[int, ...], fan_in: int) -> Array:
        bound = 1.0 / fan_in**0.5
        return jax.random.uniform(k, shape, minval=-bound, maxval=bound)

    def block(bkey: PRNGKeyArray) -> CIFnBlock:
        match arch.ffn_kind:
            case "gelu":
                kq, kk, kv, ko, k1, k2 = jax.random.split(bkey, 6)
                gate = None
            case "swiglu":
                kq, kk, kv, ko, k1, k2, kg = jax.random.split(bkey, 7)
                gate = (kaiming(kg, (d, ffn), d, relu_gain), jnp.zeros((ffn,)))
        norm_scales = (jnp.ones((d,)), jnp.ones((d,))) if arch.learned_norm_scale else None
        return CIFnBlock(
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
    total_output_width = sum(output_widths)
    concatenated_weights = kaiming(out_key, (d, total_output_width), d, 1.0)
    concatenated_biases = jnp.zeros((total_output_width,))
    offsets = [0]
    for c in output_widths:
        offsets.append(offsets[-1] + c)
    return TransformerParameters(
        in_proj_w=kaiming(in_key, (total_d_in, d), total_d_in, relu_gain),
        in_proj_b=jnp.zeros((d,)),
        blocks=[block(bk) for bk in block_keys],
        out_ws=tuple(
            concatenated_weights[:, offsets[j] : offsets[j + 1]] for j in range(len(output_widths))
        ),
        out_bs=tuple(
            concatenated_biases[offsets[j] : offsets[j + 1]] for j in range(len(output_widths))
        ),
    )


def input_attention_norm_parameter_census(
    *,
    input_dim: int,
    width: int,
    n_blocks: int,
    attention: CIFnAttention,
    learned_norm_scale: bool,
    n_chunks: int,
) -> ParameterCensus:
    kv_width = width // attention.n_heads * attention.n_kv_heads
    matrices = [MatrixParameters(input_dim, width, n_chunks)]
    if n_blocks:
        matrices.extend(
            (
                MatrixParameters(width, width, 2 * n_blocks * n_chunks),
                MatrixParameters(width, kv_width, 2 * n_blocks * n_chunks),
            )
        )
    n_vector_parameters = n_chunks * width * (1 + 2 * n_blocks * learned_norm_scale)
    return ParameterCensus(tuple(matrices), n_vector_parameters)


def dense_transformer_flops(
    arch: TransformerArch,
    *,
    input_dim: int,
    n_stacks: int,
    output_width: int,
    batch_size: int,
    sequence_length: int,
) -> ForwardBackwardFlops:
    """`n_stacks` independent transformers over one token grid, whose output heads
    together emit `output_width` preactivations per token. The taps receive no
    gradient, so the input projections run no backward input contraction."""
    n_tokens = batch_size * sequence_length
    input_projection = 2 * n_tokens * input_dim * arch.d_model
    attention = attention_flops(arch.attention, arch.d_model, batch_size, sequence_length)
    match arch.ffn_kind:
        case "gelu":
            ffn = gelu_ffn_flops(arch.d_model, arch.ffn_hidden, n_tokens)
        case "swiglu":
            ffn = swiglu_ffn_flops(arch.d_model, arch.ffn_hidden, n_tokens)
    heads = 2 * n_tokens * arch.d_model * output_width
    forward = n_stacks * (input_projection + arch.n_blocks * (attention + ffn)) + heads
    return ForwardBackwardFlops(forward, 2 * forward - n_stacks * input_projection)


def dense_transformer_parameters(
    arch: TransformerArch, *, input_dim: int, n_stacks: int, output_widths: tuple[int, ...]
) -> ParameterCensus:
    """`n_stacks` independent transformers whose output heads have `output_widths`."""
    input_attention_norms = input_attention_norm_parameter_census(
        input_dim=input_dim,
        width=arch.d_model,
        n_blocks=arch.n_blocks,
        attention=arch.attention,
        learned_norm_scale=arch.learned_norm_scale,
        n_chunks=n_stacks,
    )
    matrices = list(input_attention_norms.matrices)
    n_vector_parameters = input_attention_norms.n_vector_parameters
    match arch.ffn_kind:
        case "gelu":
            n_projections = 2
            n_bias_parameters = arch.ffn_hidden + arch.d_model
        case "swiglu":
            n_projections = 3
            n_bias_parameters = 2 * arch.ffn_hidden + arch.d_model
    n_blocks = arch.n_blocks * n_stacks
    if n_blocks:
        matrices.append(MatrixParameters(arch.d_model, arch.ffn_hidden, n_projections * n_blocks))
    n_vector_parameters += n_bias_parameters * n_blocks
    for width in output_widths:
        matrices.append(MatrixParameters(arch.d_model, width, 1))
        n_vector_parameters += width
    return ParameterCensus(tuple(matrices), n_vector_parameters)
