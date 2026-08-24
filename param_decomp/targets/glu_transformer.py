"""The SHARED machinery of the vendored HF GLU-transformer decomposition targets: site
grammar, frozen modules, the scan/masked-forward engine, and HF safetensors loading. The
model FAMILIES live in their own files — `llama31.py`, `qwen3.py` —
each contributing its arch config, its `FrozenAttn` variant (via the `_prep_qk` pre-RoPE
hook, e.g. Qwen3's QK-norm), and its HF attn loader; nothing here switches on a family.

The decomposed sites are any per-layer weight matrices (SPEC §1/§3) named torch-style:
`layers.{i}.self_attn.{q,k,v,o}_proj` and `layers.{i}.mlp.{gate,up,down}_proj`, each
with its own C. `GLUDecomposedModel` (an `eqx.Module`) carries the full frozen model —
embedding through every layer to the LM head — as array fields, threaded into the jitted
step as a pytree arg; layers without sites run the plain frozen block.

q/k/v sites are decomposed BEFORE `_prep_qk`/RoPE/SDPA (the masked site output feeds the
attention math); the o site applies to the attention output. V/U masters are fp32
keyed per site (`ComponentStacks`); frozen weights are stored bf16 (SPEC N1) — the trainer
casts for compute.

Real HF weights load straight from the cached safetensors (no torch dep).

Sharding (the production HSDP memory story): frozen target matrices derive Megatron
column/row layouts from `PlacementRules.target`. Q/K/V and gate/up shard their output;
O and down consume that shard and reduce to the replicated external waist. Decomposed
linears instead retain their replicated public waist and shard only the internal C axis.
Embed / lm_head / norm / inv_freq replicate. The bf16 component compute weights are
materialized to the `fsdp`-sharded residents once per step in `prepare_compute_weights`
(the cross-`replicate` gather, typed `reduced` over the gathered axes, off the per-layer
hot path). V/U, CI-fn, and source placement are the engine's concern (`placement`,
`init_placed`), not this module's.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, cast, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int
from safetensors import safe_open

from param_decomp.core import family
from param_decomp.core.axes import Axes, SemanticAxis
from param_decomp.core.components import (
    ComponentStacks,
    SiteC,
    SiteDims,
    SiteSpec,
    activation_axes,
    site_slots_for,
)
from param_decomp.core.decomposed_linear import (
    PlannedComponentLinear,
    SiteForward,
    site_forward,
)
from param_decomp.core.family import ArchFamily
from param_decomp.core.linear_plan import (
    LinearPlan,
    placed_linear,
    slice_leading,
    uniform_like,
    unreduce,
    value_mesh,
)
from param_decomp.core.model import (
    EMPTY_CAPTURE_KEYS,
    CaptureKeys,
    ForwardResult,
    Masking,
    MaterializedMasking,
    StochasticMasking,
)
from param_decomp.core.nonlinearity import (
    AttentionHeads,
    Neurons,
    NonlinearityPartition,
)
from param_decomp.core.placement import (
    PlacedRule,
    PlacementRules,
    component_stacks_to_compute_weights,
    constrain_activation,
    materialize_stored_weight,
    placed_target_linear,
    target_linear_plan,
)
from param_decomp.targets.losses import kl_per_position
from param_decomp.targets.transformer_taps import (
    BlockTap,
    PostAttentionResidual,
    ResidualBoundary,
    SiteOutput,
    TransformerPoint,
    TransformerTapGrammar,
    attention_input_tap_key,
    attention_output_tap_key,
    mlp_hidden_tap_key,
    mlp_input_tap_key,
    site_output_tap_key,
)
from param_decomp.vendored_jax.llama import (
    AttentionImplementation,
    apply_rope,
    causal_sdpa,
    repeat_kv,
    rms_norm,
    rope_cos_sin,
)


class GLUArch(Protocol):
    """The arch-config surface the shared machinery reads. Family configs satisfy it
    structurally — the vendored `LlamaConfig` (llama3 rope-scaling fields on top of
    these) and the family-neutral `GLUConfig`."""

    @property
    def vocab_size(self) -> int: ...
    @property
    def n_layer(self) -> int: ...
    @property
    def n_head(self) -> int: ...
    @property
    def n_kv_head(self) -> int: ...
    @property
    def n_embd(self) -> int: ...
    @property
    def n_intermediate(self) -> int: ...
    @property
    def rms_norm_eps(self) -> float: ...
    @property
    def head_dim(self) -> int: ...
    @property
    def n_rep(self) -> int: ...
    @property
    def n_ctx(self) -> int: ...


class HFGLUArch(GLUArch, Protocol):
    """A GLU architecture loaded from a Hugging Face causal-LM checkpoint."""

    @property
    def tie_word_embeddings(self) -> bool: ...


@dataclass(frozen=True)
class GLUConfig:
    """A family-neutral GLU-transformer arch config (plain RoPE, no family extras).
    Families with more knobs bring their own config type satisfying `GLUArch`."""

    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int
    n_intermediate: int
    head_dim: int
    rope_theta: float
    rms_norm_eps: float
    max_position_embeddings: int
    tie_word_embeddings: bool

    @property
    def n_rep(self) -> int:
        return self.n_head // self.n_kv_head

    @property
    def n_ctx(self) -> int:
        """The context bound under its role name; `max_position_embeddings` is HF's."""
        return self.max_position_embeddings


def default_inv_freq(head_dim: int, rope_theta: float) -> Float[Array, " hd2"]:
    """Plain (unscaled) RoPE inverse frequencies — HF's `rope_type: default`."""
    return 1.0 / (rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))


# GLU = SwiGLU MLP (Llama-3.1, Qwen3). The family's matrix vocabulary — the authored
# c-spec keys (lab-side) are typed by it, so a c-spec key and a target matrix cannot drift.
GluMatrix = Literal["q", "k", "v", "o", "gate", "up", "down"]

KIND_ORDER: tuple[str, ...] = get_args(GluMatrix)
"""Within-layer canonical site order = computation order, DERIVED from the `GluMatrix`
vocabulary. The canonical site order (`glu_site_specs`) is layer-ascending, then this."""
ATTN_KINDS = ("q", "k", "v", "o")
MLP_KINDS = ("gate", "up", "down")
assert KIND_ORDER == ATTN_KINDS + MLP_KINDS, KIND_ORDER


@dataclass(frozen=True)
class GatedMLPKinds:
    """The gated anatomy's MLP matrix vocabulary, by structural role."""

    gate: str
    up: str
    down: str

    @property
    def kinds(self) -> tuple[str, ...]:
        return (self.gate, self.up, self.down)

    @property
    def hidden(self) -> tuple[str, ...]:
        return (self.gate, self.up)


@dataclass(frozen=True)
class PlainMLPKinds:
    """The plain (two-matrix) anatomy's MLP matrix vocabulary, by structural role."""

    fc: str
    down: str

    @property
    def kinds(self) -> tuple[str, ...]:
        return (self.fc, self.down)

    @property
    def hidden(self) -> tuple[str, ...]:
        return (self.fc,)


@dataclass(frozen=True)
class Anatomy:
    """A transformer family's site vocabulary bound to the engine's structural roles.

    The family names its matrices however it likes (`q` vs `q_proj`, `gate` vs `c_fc`);
    the engine only ever asks role questions — which kind is the o projection, which
    kinds produce the MLP hidden, which consume the row placement. The MLP arm is the
    one anatomical fork (`GatedMLPKinds` | `PlainMLPKinds`), matching the layer's
    `GatedMLP | PlainMLP` weights."""

    family: ArchFamily
    q: str
    k: str
    v: str
    o: str
    mlp: GatedMLPKinds | PlainMLPKinds

    def __post_init__(self) -> None:
        assert self.kind_order == self.family.matrices, (self.kind_order, self.family.matrices)

    @property
    def attn_kinds(self) -> tuple[str, str, str, str]:
        return (self.q, self.k, self.v, self.o)

    @property
    def kind_order(self) -> tuple[str, ...]:
        return (*self.attn_kinds, *self.mlp.kinds)

    @property
    def row_kinds(self) -> frozenset[str]:
        """Kinds whose frozen linear consumes the column shard (o and down)."""
        return frozenset((self.o, self.mlp.down))

    def site_output_tap(self, kind: str) -> "_GLUTap":
        match self.mlp:
            case GatedMLPKinds(gate=gate, up=up, down=down):
                mlp_taps = {
                    gate: _GLUTap.GATE_OUTPUT,
                    up: _GLUTap.UP_OUTPUT,
                    down: _GLUTap.DOWN_OUTPUT,
                }
            case PlainMLPKinds(fc=fc, down=down):
                mlp_taps = {fc: _GLUTap.FC_OUTPUT, down: _GLUTap.DOWN_OUTPUT}
        tap_of_kind = {
            self.q: _GLUTap.Q_OUTPUT,
            self.k: _GLUTap.K_OUTPUT,
            self.v: _GLUTap.V_OUTPUT,
            self.o: _GLUTap.O_OUTPUT,
            **mlp_taps,
        }
        assert kind in tap_of_kind, f"unknown {self.family.key} site kind {kind!r}"
        return tap_of_kind[kind]


def _hidden_acts_reconstruction_dependencies(
    anatomy: Anatomy, point: TransformerPoint
) -> frozenset[str]:
    """Same-block decomposed kinds whose output can influence ``point``."""
    match point:
        case ResidualBoundary():
            return frozenset()
        case PostAttentionResidual():
            return frozenset(anatomy.attn_kinds)
        case BlockTap(name=name):
            match name:
                case "attn_in":
                    return frozenset()
                case "attn_out":
                    return frozenset((anatomy.q, anatomy.k, anatomy.v))
                case "mlp_in":
                    return frozenset(anatomy.attn_kinds)
                case "mlp_hidden":
                    return frozenset((*anatomy.attn_kinds, *anatomy.mlp.hidden))
                case _:
                    raise AssertionError(name)
        case SiteOutput(name=name):
            _block, kind = anatomy.family.parse(name)
            if kind in (anatomy.q, anatomy.k, anatomy.v):
                return frozenset((kind,))
            if kind == anatomy.o:
                return frozenset(anatomy.attn_kinds)
            if kind in anatomy.mlp.hidden:
                return frozenset((*anatomy.attn_kinds, kind))
            if kind == anatomy.mlp.down:
                return frozenset(anatomy.kind_order)
            raise AssertionError(kind)


SITE_NAME_PATTERN = re.compile(
    r"^layers\.(\d+)\.(?:self_attn\.(q|k|v|o)|mlp\.(gate|up|down))_proj$"
)


def site_name(layer: int, kind: str) -> str:
    assert kind in KIND_ORDER, kind
    submodule = "self_attn" if kind in ATTN_KINDS else "mlp"
    return f"layers.{layer}.{submodule}.{kind}_proj"


def parse_site_name(name: str) -> tuple[int, str]:
    """`layers.{i}.{self_attn,mlp}.{kind}_proj` -> (layer, kind); rejects anything else
    (including kind/submodule mismatches like `self_attn.gate_proj`)."""
    match = SITE_NAME_PATTERN.match(name)
    assert match is not None, (
        f"not a glu_transformer site: {name!r} (sites are layers.{{i}}.self_attn.{{q|k|v|o}}_proj"
        f" / layers.{{i}}.mlp.{{gate|up|down}}_proj)"
    )
    layer, attn_kind, mlp_kind = match.groups()
    return int(layer), attn_kind if attn_kind is not None else mlp_kind


FAMILY = ArchFamily("glu_transformer", KIND_ORDER, site_name, parse_site_name)
"""This family's matrix grammar as data — the vocabulary + name renderer the tiled
`glu_transformer` c-specs resolve against."""

GLU_ANATOMY = Anatomy(
    family=FAMILY, q="q", k="k", v="v", o="o", mlp=GatedMLPKinds(gate="gate", up="up", down="down")
)
"""The HF GLU families' vocabulary bound to the engine's structural roles."""


def anatomy_site_dims(anatomy: Anatomy, cfg: GLUArch, kind: str) -> SiteDims:
    """Dimensions of one per-layer matrix in right-mult orientation, by structural role."""
    d, di = cfg.n_embd, cfg.n_intermediate
    qd = cfg.n_head * cfg.head_dim
    kvd = cfg.n_kv_head * cfg.head_dim
    if kind == anatomy.q:
        return SiteDims(d_in=d, d_out=qd)
    if kind in (anatomy.k, anatomy.v):
        return SiteDims(d_in=d, d_out=kvd)
    if kind == anatomy.o:
        return SiteDims(d_in=qd, d_out=d)
    if kind in anatomy.mlp.hidden:
        return SiteDims(d_in=d, d_out=di)
    if kind == anatomy.mlp.down:
        return SiteDims(d_in=di, d_out=d)
    raise AssertionError(f"unknown kind {kind!r}")


def site_dims(cfg: GLUArch, kind: str) -> SiteDims:
    return anatomy_site_dims(GLU_ANATOMY, cfg, kind)


def canonical_site_cs(site_cs: tuple[SiteC, ...]) -> tuple[SiteC, ...]:
    return family.canonical_site_cs(FAMILY, site_cs)


def mlp_family_site_cs(first_layer: int, last_layer: int, C: int) -> tuple[SiteC, ...]:
    """The gate/up/down sites of a contiguous layer range at one C (the native-config
    target family), in canonical order."""
    assert first_layer <= last_layer, (first_layer, last_layer)
    return tuple(
        SiteC(site_name(layer, kind), C)
        for layer in range(first_layer, last_layer + 1)
        for kind in MLP_KINDS
    )


def anatomy_nonlinearity_partition(
    anatomy: Anatomy, cfg: GLUArch, kind: str
) -> NonlinearityPartition | None:
    """The nonlinearity units one per-layer matrix writes into, by structural role: the
    MLP hidden writers face elementwise neurons, q/k/v face attention heads, and the
    residual writers (o, down) face none."""
    if kind in anatomy.mlp.hidden:
        return Neurons()
    if kind == anatomy.q:
        return AttentionHeads(cfg.n_head)
    if kind in (anatomy.k, anatomy.v):
        return AttentionHeads(cfg.n_kv_head)
    if kind in (anatomy.o, anatomy.mlp.down):
        return None
    raise AssertionError(f"unknown kind {kind!r}")


def nonlinearity_partition(cfg: GLUArch, kind: str) -> NonlinearityPartition | None:
    return anatomy_nonlinearity_partition(GLU_ANATOMY, cfg, kind)


def glu_site_specs(cfg: GLUArch, site_cs: tuple[SiteC, ...]) -> tuple[SiteSpec, ...]:
    return family.site_specs(
        FAMILY,
        site_cs,
        lambda kind: site_dims(cfg, kind),
        lambda kind: nonlinearity_partition(cfg, kind),
        cfg.n_layer,
    )


# ----------------------------- frozen layers -----------------------------


class FrozenAttn(eqx.Module):
    """Plain GQA attention (Llama, LlamaSimpleMLP). A family with extra pre-RoPE math
    subclasses and overrides `_prep_qk` (and `shardings` for any extra fields) — e.g.
    `qwen3.Qwen3FrozenAttn`'s per-head QK-norm."""

    wq: Float[Array, "qd d"]
    wk: Float[Array, "kvd d"]
    wv: Float[Array, "kvd d"]
    wo: Float[Array, "d qd"]
    n_head: int = eqx.field(static=True)
    n_kv_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    n_rep: int = eqx.field(static=True)
    implementation: AttentionImplementation = eqx.field(static=True)

    def _prep_qk(
        self, q: Float[Array, "b t h hd"], k: Float[Array, "b t kvh hd"]
    ) -> tuple[Array, Array]:
        """Family hook between the head reshape and RoPE; identity for plain attention."""
        return q, k

    def shardings(self, placement: PlacementRules, axes: Axes) -> "FrozenAttn":
        assert axes in (("d_out", "d_in"), ("layer", "d_out", "d_in")), axes
        column = placement.target.column.persist
        row = placement.target.row.persist
        for w in (self.wq, self.wk, self.wv):
            column.validate_shape(axes, w.shape)
        row.validate_shape(axes, self.wo.shape)
        return eqx.tree_at(
            lambda a: (a.wq, a.wk, a.wv, a.wo),
            self,
            (
                column.sharding_for(axes),
                column.sharding_for(axes),
                column.sharding_for(axes),
                row.sharding_for(axes),
            ),
        )

    def core(
        self,
        q_flat: Float[Array, "b t qd"],
        k_flat: Float[Array, "b t kvd"],
        v_flat: Float[Array, "b t kvd"],
        inv_freq: Array,
        activation_row: PlacedRule | None,
    ) -> Float[Array, "b t qd"]:
        """RoPE + causal SDPA between the q/k/v projections and the o projection —
        the seam the decomposed q/k/v site outputs feed into."""
        b, t, _ = q_flat.shape
        assert q_flat.shape[-1] == self.n_head * self.head_dim, q_flat.shape
        assert k_flat.shape[-1] == self.n_kv_head * self.head_dim, k_flat.shape
        assert v_flat.shape[-1] == self.n_kv_head * self.head_dim, v_flat.shape
        # Native GQA preserves distinct query and key/value head counts on their own
        # semantic axes, and BOTH must tile their assignments — checked BEFORE the
        # head-split reshapes so a tp that divides the flat qd/kvd widths but not a head
        # count dies on OUR divisibility gate, not on the reshape's sharding rule.
        q_axes: Axes = ("batch", "position", "q_head", "head_dim")
        kv_axes: Axes = ("batch", "position", "kv_head", "head_dim")
        qkv_spec = None
        if activation_row is not None:
            activation_row.validate_shape(q_axes, (b, t, self.n_head, self.head_dim))
            activation_row.validate_shape(kv_axes, (b, t, self.n_kv_head, self.head_dim))
            # cuDNN SDPA requires identical sharding on its direct operands, so the two
            # head axes must resolve to ONE spec at this row.
            assert activation_row.spec_for(q_axes) == activation_row.spec_for(kv_axes), (
                f"{activation_row.label}: q_head and kv_head must carry the same mesh "
                f"assignment (cuDNN SDPA shards q/k/v identically): "
                f"{activation_row.rule.get('q_head')!r} != {activation_row.rule.get('kv_head')!r}"
            )
            qkv_spec = activation_row.sharding_for(q_axes)
        q = q_flat.reshape(b, t, self.n_head, self.head_dim)
        k = k_flat.reshape(b, t, self.n_kv_head, self.head_dim)
        q, k = self._prep_qk(q, k)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v_flat.reshape(b, t, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        cos, sin = rope_cos_sin(inv_freq, t, q_flat.dtype)
        q, k = apply_rope(q, k, cos, sin)
        output = (
            causal_sdpa(q, k, v, qkv_spec, self.implementation)
            .transpose(0, 2, 1, 3)
            .reshape(b, t, self.n_head * self.head_dim)
        )
        return constrain_activation(output, activation_row)

    def __call__(self, x: Float[Array, "b t d"], inv_freq: Array) -> Array:
        return self.core(x @ self.wq.T, x @ self.wk.T, x @ self.wv.T, inv_freq, None) @ self.wo.T

    def pattern(
        self, q_flat: Float[Array, "b t qd"], k_flat: Float[Array, "b t kvd"], inv_freq: Array
    ) -> Float[Array, "b h t t"]:
        """Post-softmax causal attention map from flat Q/K projections — the target-owned
        recipe behind the attn-patterns eval (`GLUDecomposedModel.attention_pattern_from_qk`). Same
        `_prep_qk`/RoPE/GQA math as `core`; scores in fp32, scaled by `1/√head_dim`,
        causal-masked, softmaxed (no score materialization concerns — eval-only)."""
        b, t, _ = q_flat.shape
        assert q_flat.shape[-1] == self.n_head * self.head_dim, q_flat.shape
        assert k_flat.shape[-1] == self.n_kv_head * self.head_dim, k_flat.shape
        q = q_flat.reshape(b, t, self.n_head, self.head_dim)
        k = k_flat.reshape(b, t, self.n_kv_head, self.head_dim)
        q, k = self._prep_qk(q, k)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        cos, sin = rope_cos_sin(inv_freq, t, q_flat.dtype)
        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32))
        scores = scores / self.head_dim**0.5
        causal = jnp.triu(jnp.ones((t, t), bool), k=1)
        return jax.nn.softmax(jnp.where(causal, -jnp.inf, scores), axis=-1)


class GatedMLP(eqx.Module):
    """SwiGLU MLP weights: `silu(x@Wg.T) * (x@Wu.T) @ Wd.T` (Llama-3.1, Qwen3)."""

    Wg: Float[Array, "di d"]
    Wu: Float[Array, "di d"]
    Wd: Float[Array, "d di"]

    def shardings(self, placement: PlacementRules, axes: Axes) -> "GatedMLP":
        column = placement.target.column.persist
        row = placement.target.row.persist
        for w in (self.Wg, self.Wu):
            column.validate_shape(axes, w.shape)
        row.validate_shape(axes, self.Wd.shape)
        return eqx.tree_at(
            lambda m: (m.Wg, m.Wu, m.Wd),
            self,
            (column.sharding_for(axes), column.sharding_for(axes), row.sharding_for(axes)),
        )


def _gelu_tanh(x: Array) -> Array:
    """The plain MLP's activation — torch `NewGELU` (tanh approximation), exactly
    `jax.nn.gelu(approximate=True)`; pinned by the simple_mlp torch-fixture test."""
    return jax.nn.gelu(x, approximate=True)


class PlainMLP(eqx.Module):
    """Two-matrix GELU(tanh) MLP weights: `gelu(x@Wfc.T) @ Wdown.T` (LlamaSimpleMLP)."""

    Wfc: Float[Array, "di d"]
    Wdown: Float[Array, "d di"]

    def shardings(self, placement: PlacementRules, axes: Axes) -> "PlainMLP":
        column = placement.target.column.persist
        row = placement.target.row.persist
        column.validate_shape(axes, self.Wfc.shape)
        row.validate_shape(axes, self.Wdown.shape)
        return eqx.tree_at(
            lambda m: (m.Wfc, m.Wdown),
            self,
            (column.sharding_for(axes), row.sharding_for(axes)),
        )


class GLULayer(eqx.Module):
    """One layer's frozen weights — norms, attention, MLP. Decomposed sites read
    their frozen target W from here at forward time; layers without sites run the
    plain frozen block from the same fields. Weights pass as a runtime arg — never
    baked into the HLO as a multi-GB constant. The MLP is the enumerated anatomy
    arm (`GatedMLP | PlainMLP`) — static treedef, so one model scans one anatomy."""

    ln1: Float[Array, " d"]
    ln2: Float[Array, " d"]
    attn: FrozenAttn
    mlp: GatedMLP | PlainMLP

    def shardings(self, placement: PlacementRules) -> "GLULayer":
        axes = ("layer", "d_out", "d_in")
        repl = NamedSharding(placement.mesh, P())
        return eqx.tree_at(
            lambda layer: (layer.ln1, layer.ln2, layer.attn, layer.mlp),
            self,
            (
                repl,
                repl,
                self.attn.shardings(placement, axes),
                self.mlp.shardings(placement, axes),
            ),
        )


V_WEIGHT_AXES: tuple[SemanticAxis, SemanticAxis] = ("d_in", "C")
U_WEIGHT_AXES: tuple[SemanticAxis, SemanticAxis] = ("C", "d_out")


def _frozen_site_weight(anatomy: Anatomy, layer: GLULayer, kind: str) -> Array:
    attn_weight_of = {
        anatomy.q: layer.attn.wq,
        anatomy.k: layer.attn.wk,
        anatomy.v: layer.attn.wv,
        anatomy.o: layer.attn.wo,
    }
    if kind in attn_weight_of:
        return attn_weight_of[kind]
    match layer.mlp, anatomy.mlp:
        case GatedMLP() as mlp, GatedMLPKinds() as kinds:
            return {kinds.gate: mlp.Wg, kinds.up: mlp.Wu, kinds.down: mlp.Wd}[kind]
        case PlainMLP() as mlp, PlainMLPKinds() as kinds:
            return {kinds.fc: mlp.Wfc, kinds.down: mlp.Wdown}[kind]
        case _:
            raise AssertionError((type(layer.mlp), anatomy.mlp))


# ----------------------------- forwards -----------------------------


def _clean_mlp_out(layer: GLULayer, mlp_in: Array, placement: PlacementRules | None) -> Array:
    """Frozen target MLP — exactly `W` applied, not the `V@U + (W−V@U)` identity, so
    non-decomposed layers carry no V/U gradient and no decomposition rounding (SPEC S2/S3)."""
    column = None if placement is None else placement.target.column
    row = None if placement is None else placement.target.row
    match layer.mlp:
        case GatedMLP(Wg=Wg, Wu=Wu, Wd=Wd):
            gate = placed_target_linear(mlp_in, Wg, column)
            up = placed_target_linear(mlp_in, Wu, column)
            return placed_target_linear(jax.nn.silu(gate) * up, Wd, row)
        case PlainMLP(Wfc=Wfc, Wdown=Wdown):
            fc = placed_target_linear(mlp_in, Wfc, column)
            return placed_target_linear(_gelu_tanh(fc), Wdown, row)


def _clean_block(
    layer: GLULayer,
    x: Array,
    inv_freq: Array,
    eps: float,
    placement: PlacementRules | None,
) -> Array:
    attn = layer.attn
    h = rms_norm(x, layer.ln1, eps)
    column = None if placement is None else placement.target.column
    row = None if placement is None else placement.target.row
    attention_output = attn.core(
        placed_target_linear(h, attn.wq, column),
        placed_target_linear(h, attn.wk, column),
        placed_target_linear(h, attn.wv, column),
        inv_freq,
        None if column is None else column.output,
    )
    x = x + placed_target_linear(attention_output, attn.wo, row)
    return x + _clean_mlp_out(layer, rms_norm(x, layer.ln2, eps), placement)


def _stack_layers(layers: list[GLULayer]) -> GLULayer:
    """Stack a per-layer `GLULayer` list into one whose array leaves carry a leading
    layer axis — the `xs` for a `lax.scan` over the (homogeneous) block stack. Static
    fields (attn head counts) ride in the treedef, shared across iterations."""
    return jax.tree.map(lambda *per_layer: jnp.stack(per_layer), *layers)


class _GLUTap(Enum):
    RESIDUAL_IN = "residual_in"
    QKV_INPUT = "qkv_input"
    Q_OUTPUT = "q_output"
    K_OUTPUT = "k_output"
    V_OUTPUT = "v_output"
    ATTENTION_OUTPUT = "attention_output"
    O_OUTPUT = "o_output"
    POST_ATTENTION_RESIDUAL = "post_attention_residual"
    MLP_INPUT = "mlp_input"
    GATE_OUTPUT = "gate_output"
    UP_OUTPUT = "up_output"
    FC_OUTPUT = "fc_output"
    DOWN_INPUT = "down_input"
    DOWN_OUTPUT = "down_output"
    RESIDUAL_OUT = "residual_out"


@dataclass(frozen=True, kw_only=True)
class _GLUCaptureSource:
    block: int
    tap: _GLUTap


GLUCaptureSources = tuple[_GLUCaptureSource, ...]


@dataclass(frozen=True, kw_only=True)
class _GatedHiddenActs:
    gate_output: Array
    up_output: Array


@dataclass(frozen=True, kw_only=True)
class _PlainHiddenActs:
    fc_output: Array


@dataclass(frozen=True, kw_only=True)
class _GLUBlockActivations:
    """The target's capturable activations for one block; never returned wholesale.
    The MLP hidden-producing outputs ride the anatomy arm (`hidden`)."""

    residual_in: Array
    qkv_input: Array
    q_output: Array
    k_output: Array
    v_output: Array
    attention_output: Array
    o_output: Array
    post_attention_residual: Array
    mlp_input: Array
    hidden: _GatedHiddenActs | _PlainHiddenActs
    down_input: Array
    down_output: Array
    residual_out: Array


_UNUSED_CAPTURE_SLOT = -1


@dataclass(frozen=True, kw_only=True)
class _ScanCaptureLayout:
    """One exact-size scan-carry buffer for each requested value kind."""

    slot_by_block_per_tap: tuple[tuple[_GLUTap, tuple[int, ...]], ...]
    sources: tuple[_GLUCaptureSource, ...]


def _capture_source_for_point(anatomy: Anatomy, point: TransformerPoint) -> _GLUCaptureSource:
    match point:
        case ResidualBoundary(boundary=0):
            return _GLUCaptureSource(block=0, tap=_GLUTap.RESIDUAL_IN)
        case ResidualBoundary(boundary=boundary):
            return _GLUCaptureSource(block=boundary - 1, tap=_GLUTap.RESIDUAL_OUT)
        case PostAttentionResidual(block=block):
            return _GLUCaptureSource(block=block, tap=_GLUTap.POST_ATTENTION_RESIDUAL)
        case BlockTap(name=name, block=block):
            value_by_name = {
                "attn_in": _GLUTap.QKV_INPUT,
                "attn_out": _GLUTap.ATTENTION_OUTPUT,
                "mlp_in": _GLUTap.MLP_INPUT,
                "mlp_hidden": _GLUTap.DOWN_INPUT,
            }
            return _GLUCaptureSource(block=block, tap=value_by_name[name])
        case SiteOutput(name=name, block=block):
            _layer, kind = anatomy.family.parse(name)
            return _GLUCaptureSource(block=block, tap=anatomy.site_output_tap(kind))


def _scan_capture_layout(sources: GLUCaptureSources, n_layer: int) -> _ScanCaptureLayout:
    tap_slots: list[tuple[_GLUTap, tuple[int, ...]]] = []
    for tap in _GLUTap:
        blocks = tuple(source.block for source in sources if source.tap is tap)
        if not blocks or tap is _GLUTap.RESIDUAL_IN:
            continue
        slot_by_block = [_UNUSED_CAPTURE_SLOT] * n_layer
        for slot, block in enumerate(blocks):
            slot_by_block[block] = slot
        tap_slots.append((tap, tuple(slot_by_block)))
    return _ScanCaptureLayout(slot_by_block_per_tap=tuple(tap_slots), sources=sources)


def _captured_activation(block_activations: _GLUBlockActivations, tap: _GLUTap) -> Array:
    match tap:
        case _GLUTap.RESIDUAL_IN:
            return block_activations.residual_in
        case _GLUTap.QKV_INPUT:
            return block_activations.qkv_input
        case _GLUTap.Q_OUTPUT:
            return block_activations.q_output
        case _GLUTap.K_OUTPUT:
            return block_activations.k_output
        case _GLUTap.V_OUTPUT:
            return block_activations.v_output
        case _GLUTap.ATTENTION_OUTPUT:
            return block_activations.attention_output
        case _GLUTap.O_OUTPUT:
            return block_activations.o_output
        case _GLUTap.POST_ATTENTION_RESIDUAL:
            return block_activations.post_attention_residual
        case _GLUTap.MLP_INPUT:
            return block_activations.mlp_input
        case _GLUTap.GATE_OUTPUT:
            hidden = block_activations.hidden
            assert isinstance(hidden, _GatedHiddenActs), tap
            return hidden.gate_output
        case _GLUTap.UP_OUTPUT:
            hidden = block_activations.hidden
            assert isinstance(hidden, _GatedHiddenActs), tap
            return hidden.up_output
        case _GLUTap.FC_OUTPUT:
            hidden = block_activations.hidden
            assert isinstance(hidden, _PlainHiddenActs), tap
            return hidden.fc_output
        case _GLUTap.DOWN_INPUT:
            return block_activations.down_input
        case _GLUTap.DOWN_OUTPUT:
            return block_activations.down_output
        case _GLUTap.RESIDUAL_OUT:
            return block_activations.residual_out


def _allocate_capture_buffers(
    layout: _ScanCaptureLayout,
    residual: Array,
    width_of: Callable[[_GLUTap], int],
) -> dict[str, Array]:
    if value_mesh(residual).empty:
        spec = None
    else:
        residual_spec = jax.typeof(residual).sharding.spec
        spec = P(None, *residual_spec[:-1], None)
    return {
        tap.value: jnp.zeros(
            (
                sum(slot >= 0 for slot in slots),
                *residual.shape[:-1],
                width_of(tap),
            ),
            residual.dtype,
            **({} if spec is None else {"out_sharding": spec}),
        )
        for tap, slots in layout.slot_by_block_per_tap
    }


def _slot_index_arrays(layout: _ScanCaptureLayout) -> dict[str, Array]:
    return {
        tap.value: jnp.asarray(slots, dtype=jnp.int32)
        for tap, slots in layout.slot_by_block_per_tap
    }


def _write_block_captures(
    layout: _ScanCaptureLayout,
    buffers: dict[str, Array],
    slot_indices: dict[str, Array],
    block_activations: _GLUBlockActivations,
) -> dict[str, Array]:
    updated_buffers = dict(buffers)
    for tap, _slot_tuple in layout.slot_by_block_per_tap:
        buffer_key = tap.value
        slot_index = slot_indices[buffer_key]
        captured_value = _captured_activation(block_activations, tap)
        if not value_mesh(buffers[buffer_key]).empty:
            # The buffer write requires exact type equality; captures land in the
            # buffer's batch-sharded, feature-replicated layout.
            buffer_spec = jax.typeof(buffers[buffer_key]).sharding.spec
            captured_value = jax.sharding.reshard(captured_value, P(*buffer_spec[1:]))
        updated_buffers[buffer_key] = jax.lax.cond(
            slot_index != _UNUSED_CAPTURE_SLOT,
            lambda buffer, value=captured_value, index=slot_index: (
                jax.lax.dynamic_update_index_in_dim(buffer, value, index, axis=0)
            ),
            lambda buffer: buffer,
            buffers[buffer_key],
        )
    return updated_buffers


def _read_capture_buffers(
    captured_by_source: dict[_GLUCaptureSource, Array],
    layout: _ScanCaptureLayout,
    buffers: dict[str, Array],
) -> None:
    """Index scan-buffer values by capture source after the layer scan.

    The embedding residual is recorded directly by the caller before the scan.
    """
    slot_by_block_per_tap = dict(layout.slot_by_block_per_tap)
    for source in layout.sources:
        if source.tap is not _GLUTap.RESIDUAL_IN:
            captured_by_source[source] = buffers[source.tap.value][
                slot_by_block_per_tap[source.tap][source.block]
            ]


def _captures_in_request_order(
    sources: GLUCaptureSources, captured_by_source: dict[_GLUCaptureSource, Array]
) -> tuple[Array, ...]:
    assert set(captured_by_source) == set(sources), (captured_by_source.keys(), sources)
    return tuple(captured_by_source[source] for source in sources)


def _stack_per_kind_vu(
    anatomy: Anatomy, components: ComponentStacks, n_layers: int
) -> dict[str, dict[str, Array]]:
    """Per decomposed KIND, the layer-stacked `(V, U)` arrays — the MASK-INDEPENDENT part of
    the scan inputs (a leading layer axis, one homogeneous body across layers). Mask/
    delta/route are attached per-forward by `_attach_per_kind_masks`; the V/U stack + the
    compute-weight materialization (the ÷N→÷fsdp cross-node gather) are the same for EVERY
    forward in a step, so they are built ONCE via `prepare_compute_weights` and shared.
    Runs on the COMPUTE stacks (stack axis unsharded), so a partially decomposed model's
    per-site slicing never indexes a sharded axis."""
    for name, group, _slot in components.site_slots:
        assert group == anatomy.family.parse(name)[1], (
            f"engine sites must be grouped by matrix kind, got {group!r} for {name}"
        )
    slot_of = {name: (group, slot) for name, group, slot in components.site_slots}
    per_kind: dict[str, dict[str, Array]] = {}
    for kind in tuple(components.stacks):
        names = [anatomy.family.name_of(layer, kind) for layer in range(n_layers)]
        slots = [slot_of[n][1] for n in names if n in slot_of]
        if len(names) == len(slots):
            assert slots == list(range(n_layers)), (kind, slots)
            Vs, Us = components.stacks[kind]
        else:
            # Manual per-site slicing has no jax reduced-rule: drop the tag, slice and
            # stitch, then RE-TAG the stitched stack — a pure typing move (the value is
            # already replicated over the tagged axes) that keeps the deferred backward
            # reduction at this entry boundary instead of inside the layer scan.
            reduced_tags = tuple(
                frozenset(jax.typeof(s).sharding.spec.reduced) for s in components.stacks[kind]
            )
            components = eqx.tree_at(
                lambda c, kind=kind: c.stacks[kind],
                components,
                tuple(unreduce(s) for s in components.stacks[kind]),
            )
            present = next(n for n in names if n in slot_of)
            sample_v = components.site(present).V
            sample_u = components.site(present).U
            Vs = jnp.stack(
                [components.site(n).V if n in slot_of else jnp.zeros_like(sample_v) for n in names]
            )
            Us = jnp.stack(
                [components.site(n).U if n in slot_of else jnp.zeros_like(sample_u) for n in names]
            )

            def retag(stacked: Array, tag: frozenset[str]) -> Array:
                if not tag:
                    return stacked
                spec = jax.typeof(stacked).sharding.spec
                return jax.sharding.reshard(stacked, P(*spec, reduced=tag))

            Vs = retag(Vs, reduced_tags[0])
            Us = retag(Us, reduced_tags[1])
        per_kind[kind] = {"V": Vs, "U": Us}
    return per_kind


def _attach_per_kind_masks(
    anatomy: Anatomy,
    prepared_weights: dict[str, dict[str, Array]],
    n_layers: int,
    leading: tuple[int, ...],
    component_masks: dict[str, Array],
    weight_delta_masks: dict[str, Array] | None,
    routes: dict[str, Array] | None,
) -> dict[str, dict[str, Array]]:
    """Attach the per-forward `(mask[, delta][, route])` stacks to the shared, already
    stacked + ÷fsdp-reconstructed `prepared_weights` per-kind `(V, U)` weights. The masks
    cover every decomposed site; the homogeneous layer stacks fill dummy mask/delta/route
    values for non-decomposed layers, which segmentation excludes from this masked block."""
    # Dummy mask/delta/route shapes match the REAL entries (the source scope sets the leading
    # shape: `sc` broadcasts over batch as `(1, T)`, not the full `(B, T)`).
    a_mask = next(iter(component_masks.values())) if component_masks else None
    mask_lead = a_mask.shape[:-1] if a_mask is not None else leading
    a_delta = (
        next(iter(weight_delta_masks.values()), None) if weight_delta_masks is not None else None
    )
    a_route = next(iter(routes.values())) if (routes and len(routes)) else None

    def filler(build: Callable[[], Array], sample: Array | None) -> Array:
        """Dummy entries for non-decomposed layers, typed like the real entries — a
        homogeneous stack requires one sharding across its parts."""
        value = build()
        if sample is None or value_mesh(sample).empty:
            return value
        return jax.sharding.reshard(value, jax.typeof(sample).sharding)

    per_kind: dict[str, dict[str, Array]] = {}
    for kind, vu_entry in prepared_weights.items():
        C = vu_entry["V"].shape[-1]
        mask_dt = a_mask.dtype if a_mask is not None else vu_entry["V"].dtype
        names = [anatomy.family.name_of(layer, kind) for layer in range(n_layers)]
        masks_k = jnp.stack(
            [
                component_masks[name]
                if name in component_masks
                else filler(lambda C=C, mask_dt=mask_dt: jnp.ones((*mask_lead, C), mask_dt), a_mask)
                for name in names
            ]
        )
        entry: dict[str, Array] = {**vu_entry, "mask": masks_k}
        if weight_delta_masks is not None:
            delta_shape = a_delta.shape if a_delta is not None else leading
            delta_dtype = a_delta.dtype if a_delta is not None else mask_dt
            entry["delta"] = jnp.stack(
                [
                    weight_delta_masks[name]
                    if name in weight_delta_masks
                    else filler(
                        lambda shape=delta_shape, dt=delta_dtype: jnp.zeros(shape, dt), a_delta
                    )
                    for name in names
                ]
            )
        if routes is not None:
            r_shape = a_route.shape if a_route is not None else leading
            r_dt = a_route.dtype if a_route is not None else jnp.bool_
            entry["route"] = jnp.stack(
                [
                    routes[name]
                    if name in routes
                    else filler(lambda s=r_shape, d=r_dt: jnp.zeros(s, d), a_route)
                    for name in names
                ]
            )
        per_kind[kind] = entry
    return per_kind


def _stack_ci_per_kind(
    anatomy: Anatomy, ci_lower: dict[str, Array], n_layers: int
) -> dict[str, Array]:
    """Stack the per-site CI envelope into per-kind `[n_layer, *leading, C]` — built ONCE per
    step and SHARED across every stochastic recon forward (the CI envelope is identical for
    all). Mirrors `_stack_per_kind_vu`: the shared stack replaces N per-forward mask stacks."""
    kinds: dict[str, Array] = {}
    sample_by_kind: dict[str, Array] = {}
    for name, v in ci_lower.items():
        sample_by_kind.setdefault(anatomy.family.parse(name)[1], v)
    for kind, sample in sample_by_kind.items():
        names = [anatomy.family.name_of(layer, kind) for layer in range(n_layers)]
        kinds[kind] = jnp.stack(
            [ci_lower[name] if name in ci_lower else jnp.zeros_like(sample) for name in names]
        )
    return kinds


def _attach_per_kind_stochastic(
    anatomy: Anatomy,
    prepared_weights: dict[str, dict[str, Array]],
    n_layers: int,
    leading: tuple[int, ...],
    ci_stacked: dict[str, Array],
    draw_key: Array,
    routes: dict[str, Array] | None,
) -> dict[str, dict[str, Array]]:
    """Stochastic recon: attach the SHARED per-kind `ci` stack + per-(layer,kind) RNG keys
    instead of pre-built mask/delta stacks. `decomposed_site_output` draws `source = uniform(key)` and
    builds `mask = ci + (1−ci)·source` INSIDE the checkpointed block, so the per-forward mask
    is recomputed in the backward (faithful by checkpoint determinism — same key fwd+bwd) and
    never held. Only the (tiny) keys are per-forward; the `[n_layer,*,C]` ci stack
    is shared, so N forwards' mask stacks collapse to one ci stack (the memory win)."""
    src_base, delta_base = jax.random.split(draw_key)
    a_route = next(iter(routes.values())) if (routes and len(routes)) else None

    def route_filler(r_shape: tuple[int, ...], r_dt: jnp.dtype) -> Array:
        value = jnp.zeros(r_shape, r_dt)
        if a_route is None or value_mesh(a_route).empty:
            return value
        return jax.sharding.reshard(value, jax.typeof(a_route).sharding)

    per_kind: dict[str, dict[str, Array]] = {}
    for kind, vu_entry in prepared_weights.items():
        names = [anatomy.family.name_of(layer, kind) for layer in range(n_layers)]
        kind_idx = anatomy.kind_order.index(kind)
        src_keys = jnp.stack(
            [
                jax.random.fold_in(jax.random.fold_in(src_base, kind_idx), layer)
                for layer in range(n_layers)
            ]
        )
        delta_keys = jnp.stack(
            [
                jax.random.fold_in(jax.random.fold_in(delta_base, kind_idx), layer)
                for layer in range(n_layers)
            ]
        )
        entry: dict[str, Array] = {
            **vu_entry,
            "ci": ci_stacked[kind],
            "src_key": src_keys,
            "delta_key": delta_keys,
        }
        if routes is not None:
            r_shape = a_route.shape if a_route is not None else leading
            r_dt = a_route.dtype if a_route is not None else jnp.bool_
            entry["route"] = jnp.stack(
                [routes[name] if name in routes else route_filler(r_shape, r_dt) for name in names]
            )
        per_kind[kind] = entry
    return per_kind


class TiedHead(eqx.Module):
    """The output projection reads the embedding parameter — one stored table, no second
    frozen leaf. A zero-leaf pytree node, so tied models carry no head array anywhere
    (jit args, shardings trees, audits). Families with untied heads carry the array."""


class GLUDecomposedModel(eqx.Module):
    """The GLU-transformer `DecomposedModel` (the `model.py` contract; SPEC §1), shared
    across the HF GLU families — a family's identity lives in its `stacked.attn` module
    (its `FrozenAttn` variant) and `inv_freq`, never in a switch here.

    Carries the FROZEN full model (embedding, all blocks, final norm, lm_head) as array
    fields — so it threads into the jitted step as a pytree arg, its weights traced not
    baked. A `TiedHead` in the `lm_head` slot spells weight tying: the output projection
    is the embedding parameter. The TRAINABLE V/U (`vu: ComponentStacks`) is passed to
    the forward methods explicitly: separate lifecycle (own optimizer + checkpoint,
    C-sharded while these weights replicate), so it is NOT a field here.

    Forward methods take token `inputs` and embed internally. Blocks with no decomposed
    site run the plain frozen path — so a subset decomposition just leaves the rest
    frozen.

    `sites` / `has_position_axis` / `n_ctx` / `eps` are static config."""

    embed: Float[Array, "vocab d"]
    stacked: GLULayer  # the per-layer weights stacked on a leading layer axis (the scan
    # `xs`), stored pre-stacked: a saved jit input, never re-stacked inside a forward.
    n_layer: int = eqx.field(static=True)
    norm: Float[Array, " d"]
    lm_head: Float[Array, "vocab d"] | TiedHead
    inv_freq: Float[Array, " hd2"]
    sites: tuple[SiteSpec, ...] = eqx.field(static=True)
    anatomy: Anatomy = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    n_ctx: int = eqx.field(static=True)

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sites)

    @property
    def head_weight(self) -> Float[Array, "vocab d"]:
        """The output projection's stored table — the embedding when the head is tied."""
        match self.lm_head:
            case TiedHead():
                return self.embed
            case head:
                return head

    @property
    def layers(self) -> list[GLULayer]:
        """Per-layer view of `stacked` (slices the leading layer axis). For non-hot
        consumers (equivalence harness); the forwards use `stacked`."""
        return [jax.tree.map(lambda a, idx=i: a[idx], self.stacked) for i in range(self.n_layer)]

    def attention_pattern_from_qk(
        self,
        q_site: str,
        q_flat: Float[Array, "b t qd"],
        k_flat: Float[Array, "b t kvd"],
    ) -> Float[Array, "b h t t"]:
        """Post-softmax causal attention map from a layer's flat Q/K projections — the
        target-owned recipe the attn-patterns eval consumes (`attn_patterns_eval`'s
        `AttnPatternModel` protocol). Delegates to that LAYER's own attention module, so
        family behavior (Qwen3's per-layer QK-norm) comes along for free."""
        layer = self.anatomy.family.parse(q_site)[0]
        attn = jax.tree.map(lambda a, idx=layer: a[idx], self.stacked.attn)
        return attn.pattern(q_flat, k_flat, self.inv_freq)

    def shardings(self, placement: PlacementRules) -> "GLUDecomposedModel":
        embedding_axes: Axes = ("vocab", "d_model")
        normalization_axes: Axes = ("d_model",)
        position_axes: Axes = ("rope_frequency",)
        output_axes: Axes = ("vocab", "d_model")
        placement.target.embedding.persist.validate_shape(embedding_axes, self.embed.shape)
        placement.target.normalization.validate_shape(normalization_axes, self.norm.shape)
        placement.target.position_encoding.validate_shape(position_axes, self.inv_freq.shape)
        head_sharding: NamedSharding | TiedHead
        match self.lm_head:
            case TiedHead() as tied:
                # The output path reads the embedding under the OUTPUT placement, so the
                # one stored table must satisfy both roles' persistence layouts.
                assert placement.target.embedding.persist.spec_for(embedding_axes) == (
                    placement.target.output.persist.spec_for(output_axes)
                ), "tied embedding/output weights require one persistence layout"
                head_sharding = tied
            case head:
                placement.target.output.persist.validate_shape(output_axes, head.shape)
                head_sharding = placement.target.output.persist.sharding_for(output_axes)
        return eqx.tree_at(
            lambda m: (m.embed, m.norm, m.lm_head, m.inv_freq, m.stacked),
            self,
            (
                placement.target.embedding.persist.sharding_for(embedding_axes),
                placement.target.normalization.sharding_for(normalization_axes),
                head_sharding,
                placement.target.position_encoding.sharding_for(position_axes),
                self.stacked.shardings(placement),
            ),
        )

    @staticmethod
    def recon_loss_fn(masked_output: Array, clean_output: Array) -> Array:
        return kl_per_position(masked_output, clean_output)

    def embed_tokens(
        self, tokens: Int[Array, "b t"], placement: PlacementRules | None
    ) -> Float[Array, "b t d"]:
        # Every token-consuming forward funnels through here: enforce the family's
        # pretraining context bound at the single entry point.
        assert tokens.shape[1] <= self.n_ctx, (tokens.shape, self.n_ctx)
        if placement is None:
            if value_mesh(tokens).empty:
                return self.embed[tokens]
            # Axis-typed tokens: the gather output follows the token sharding
            # (replicated table), which the gather rule cannot infer on its own.
            token_sharding = jax.typeof(tokens).sharding
            return self.embed.at[tokens].get(
                out_sharding=NamedSharding(token_sharding.mesh, P(*token_sharding.spec, None))
            )
        weight = materialize_stored_weight(
            self.embed,
            placement.target.embedding.persist,
            placement.target.embedding.operand,
            axes=("vocab", "d_model"),
        )
        # The gather's output sharding is ambiguous (sharded indices, sharded table);
        # type the residual at the external waist directly — the layer scan's carry
        # must enter with the type its body maintains anyway. An off-mesh trace
        # (untyped tokens) takes the plain gather.
        if value_mesh(tokens).empty:
            return constrain_activation(weight[tokens], placement.activations.external)
        external = placement.activations.external
        axes = activation_axes(tokens.ndim + 1, "feature")
        return weight.at[tokens].get(out_sharding=external.sharding_for(axes))

    def _output_logits(self, residual: Array, placement: PlacementRules | None) -> Array:
        if placement is None:
            weight = self.head_weight
        else:
            # Legal for a tied head too: `shardings` pinned the embedding's persistence
            # layout to the output's, so the stored table satisfies the output role.
            weight = materialize_stored_weight(
                self.head_weight,
                placement.target.output.persist,
                placement.target.output.operand,
                axes=("vocab", "d_model"),
            )
        return residual @ weight.T

    def _capture_grammar(self) -> TransformerTapGrammar:
        anatomy = self.anatomy

        def site_dimensions(name: str) -> tuple[int, int]:
            _layer, kind = anatomy.family.parse(name)
            weight = _frozen_site_weight(anatomy, self.stacked, kind)
            return weight.shape[2], weight.shape[1]

        return TransformerTapGrammar(
            family=anatomy.family,
            n_layer=self.n_layer,
            d_resid=self.embed.shape[1],
            d_attention_output=site_dimensions(anatomy.family.name_of(0, anatomy.o))[0],
            d_mlp_hidden=site_dimensions(anatomy.family.name_of(0, anatomy.mlp.down))[0],
            d_out_of=lambda name: site_dimensions(name)[1],
        )

    def site_output_keys(self, sites: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(site_output_tap_key(site) for site in sites)

    def assert_hidden_acts_reconstruction_points(self, keys: tuple[str, ...]) -> None:
        self._capture_grammar().assert_hidden_acts_reconstruction_points(
            keys,
            self.site_names,
            lambda point: _hidden_acts_reconstruction_dependencies(self.anatomy, point),
        )

    def _block_activations(
        self, residual_in: Array, layer: GLULayer, placement: PlacementRules | None
    ) -> _GLUBlockActivations:
        attn = layer.attn
        column = None if placement is None else placement.target.column
        row = None if placement is None else placement.target.row
        h1 = rms_norm(residual_in, layer.ln1, self.eps)
        q = placed_target_linear(h1, attn.wq, column)
        k = placed_target_linear(h1, attn.wk, column)
        v = placed_target_linear(h1, attn.wv, column)
        attention_output = attn.core(
            q, k, v, self.inv_freq, None if column is None else column.output
        )
        o = placed_target_linear(attention_output, attn.wo, row)
        post_attention = residual_in + o
        h2 = rms_norm(post_attention, layer.ln2, self.eps)
        match layer.mlp:
            case GatedMLP(Wg=Wg, Wu=Wu, Wd=Wd):
                gate = placed_target_linear(h2, Wg, column)
                up = placed_target_linear(h2, Wu, column)
                down_input = jax.nn.silu(gate) * up
                down = placed_target_linear(down_input, Wd, row)
                hidden: _GatedHiddenActs | _PlainHiddenActs = _GatedHiddenActs(
                    gate_output=gate, up_output=up
                )
            case PlainMLP(Wfc=Wfc, Wdown=Wdown):
                fc = placed_target_linear(h2, Wfc, column)
                down_input = _gelu_tanh(fc)
                down = placed_target_linear(down_input, Wdown, row)
                hidden = _PlainHiddenActs(fc_output=fc)
        residual_out = post_attention + down
        return _GLUBlockActivations(
            residual_in=residual_in,
            qkv_input=h1,
            q_output=q,
            k_output=k,
            v_output=v,
            attention_output=attention_output,
            o_output=o,
            post_attention_residual=post_attention,
            mlp_input=h2,
            hidden=hidden,
            down_input=down_input,
            down_output=down,
            residual_out=residual_out,
        )

    def _value_width(self, value: _GLUTap) -> int:
        d = self.embed.shape[1]
        mlp = self.stacked.mlp
        match value:
            case (
                _GLUTap.RESIDUAL_IN
                | _GLUTap.QKV_INPUT
                | _GLUTap.O_OUTPUT
                | _GLUTap.POST_ATTENTION_RESIDUAL
                | _GLUTap.MLP_INPUT
                | _GLUTap.DOWN_OUTPUT
                | _GLUTap.RESIDUAL_OUT
            ):
                return d
            case _GLUTap.Q_OUTPUT:
                return self.stacked.attn.wq.shape[1]
            case _GLUTap.K_OUTPUT:
                return self.stacked.attn.wk.shape[1]
            case _GLUTap.V_OUTPUT:
                return self.stacked.attn.wv.shape[1]
            case _GLUTap.ATTENTION_OUTPUT:
                return self.stacked.attn.wo.shape[2]
            case _GLUTap.GATE_OUTPUT:
                assert isinstance(mlp, GatedMLP), value
                return mlp.Wg.shape[1]
            case _GLUTap.UP_OUTPUT:
                assert isinstance(mlp, GatedMLP), value
                return mlp.Wu.shape[1]
            case _GLUTap.FC_OUTPUT:
                assert isinstance(mlp, PlainMLP), value
                return mlp.Wfc.shape[1]
            case _GLUTap.DOWN_INPUT:
                match mlp:
                    case GatedMLP(Wd=Wd):
                        return Wd.shape[2]
                    case PlainMLP(Wdown=Wdown):
                        return Wdown.shape[2]

    def _clean_output(self, inputs: Int[Array, "b t"], placement: PlacementRules | None) -> Array:
        """Untouched graph used when no captures are requested."""

        def block(residual: Array, layer: GLULayer) -> tuple[Array, None]:
            return _clean_block(layer, residual, self.inv_freq, self.eps, placement), None

        residual = self.embed_tokens(inputs, placement)
        residual, _ = jax.lax.scan(block, residual, self.stacked)
        residual = rms_norm(residual, self.norm, self.eps)
        return self._output_logits(residual, placement)

    def clean_forward(
        self,
        inputs: Int[Array, "b t"],
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        *,
        placement: PlacementRules | None,
    ) -> ForwardResult:
        if not capture_keys:
            return ForwardResult.from_producer(
                output=self._clean_output(inputs, placement), capture_keys=(), capture_values=()
            )
        ordered_capture_keys = tuple(sorted(capture_keys))
        capture_sources = self._capture_grammar().resolve(
            ordered_capture_keys, lambda point: _capture_source_for_point(self.anatomy, point)
        )

        residual = self.embed_tokens(inputs, placement)
        captured_by_source: dict[_GLUCaptureSource, Array] = {}
        embedding_residual_source = _GLUCaptureSource(block=0, tap=_GLUTap.RESIDUAL_IN)
        if embedding_residual_source in capture_sources:
            captured_by_source[embedding_residual_source] = residual

        layout = _scan_capture_layout(capture_sources, self.n_layer)
        buffers = _allocate_capture_buffers(layout, residual, self._value_width)
        slot_indices_by_tap = _slot_index_arrays(layout)

        def block(
            state: tuple[Array, dict[str, Array]],
            layer_and_slots: tuple[GLULayer, dict[str, Array]],
        ) -> tuple[tuple[Array, dict[str, Array]], None]:
            x, buffers_ = state
            layer, slots = layer_and_slots
            block_activations = self._block_activations(x, layer, placement)
            return (
                block_activations.residual_out,
                _write_block_captures(layout, buffers_, slots, block_activations),
            ), None

        (residual, buffers), _ = jax.lax.scan(
            block, (residual, buffers), (self.stacked, slot_indices_by_tap)
        )
        _read_capture_buffers(captured_by_source, layout, buffers)
        residual = rms_norm(residual, self.norm, self.eps)
        return ForwardResult.from_producer(
            output=self._output_logits(residual, placement),
            capture_keys=ordered_capture_keys,
            capture_values=_captures_in_request_order(capture_sources, captured_by_source),
        )

    def _run_masked_forward(
        self,
        prepared_weights: dict[str, dict[str, Array]],
        inputs: Int[Array, "b t"],
        masking: Masking,
        remat: bool,
        capture_keys: tuple[str, ...],
        *,
        collect_component_activations: bool,
        placement: PlacementRules | None,
    ) -> tuple[ForwardResult, dict[str, Array]]:
        """One segmented masked forward for output, captures and optional ``x@V`` diagnostics.

        Empty capture keys keep the compact frozen blocks; non-empty keys allocate only
        their exact-size capture slots.
        """
        capture_sources = (
            self._capture_grammar().resolve(
                capture_keys, lambda point: _capture_source_for_point(self.anatomy, point)
            )
            if capture_keys
            else ()
        )
        residual = self.embed_tokens(inputs, placement)
        leading = residual.shape[:-1]
        site_set = frozenset(self.site_names)
        match masking:
            case StochasticMasking(ci_stacked=ci_stacked, draw_key=draw_key, routes=routes):
                assert routes is None or set(routes) == site_set, (
                    sorted(routes or {}),
                    sorted(site_set),
                )
                uses_weight_deltas = True
                per_kind = _attach_per_kind_stochastic(
                    self.anatomy,
                    prepared_weights,
                    self.n_layer,
                    leading,
                    ci_stacked,
                    draw_key,
                    routes,
                )
            case MaterializedMasking(
                component_masks=component_masks,
                weight_delta_masks=weight_delta_masks,
                routes=routes,
            ):
                # Exact key equality for every non-None dict: the homogeneous layer
                # stacks fill dummy entries for names outside these dicts, so a missing
                # DECOMPOSED site would otherwise silently get a zero delta / no route.
                assert set(component_masks) == site_set, (
                    sorted(component_masks),
                    sorted(site_set),
                )
                assert weight_delta_masks is None or set(weight_delta_masks) == site_set, (
                    sorted(weight_delta_masks or {}),
                    sorted(site_set),
                )
                assert routes is None or set(routes) == site_set, (
                    sorted(routes or {}),
                    sorted(site_set),
                )
                uses_weight_deltas = weight_delta_masks is not None
                per_kind = _attach_per_kind_masks(
                    self.anatomy,
                    prepared_weights,
                    self.n_layer,
                    leading,
                    component_masks,
                    weight_delta_masks,
                    routes,
                )
        decomposed_kinds = frozenset(per_kind)

        def layer_is_decomposed(layer: int) -> bool:
            flags = {
                self.anatomy.family.name_of(layer, kind) in site_set for kind in decomposed_kinds
            }
            assert len(flags) == 1, (
                f"layer {layer} is partially decomposed ({flags}); the segmented masked "
                f"forward requires whole layers across {len(decomposed_kinds)} kinds"
            )
            return flags.pop()

        decomposed_layers = [layer for layer in range(self.n_layer) if layer_is_decomposed(layer)]
        assert decomposed_layers, "a DecomposedModel has at least one decomposed layer"
        first_decomposed, last_decomposed = decomposed_layers[0], decomposed_layers[-1] + 1
        assert decomposed_layers == list(range(first_decomposed, last_decomposed)), (
            f"decomposed layers must be contiguous, got {decomposed_layers}"
        )

        def decomposed_site_output(
            site_input: Array,
            frozen_weight: Array,
            frozen_linear: LinearPlan | None,
            decomposition_inputs: dict[str, Array],
            *,
            site_placement: PlacementRules | PlannedComponentLinear | None,
        ) -> SiteForward:
            v = decomposition_inputs["V"]
            u = decomposition_inputs["U"]
            if "ci" in decomposition_inputs:
                ci_lower = decomposition_inputs["ci"]
                random_source = uniform_like(decomposition_inputs["src_key"], ci_lower)
                component_mask = ci_lower + (1.0 - ci_lower) * random_source
                weight_delta_mask = (
                    uniform_like(decomposition_inputs["delta_key"], ci_lower, drop_last_axis=True)
                    if uses_weight_deltas
                    else None
                )
                return site_forward(
                    site_input,
                    v,
                    u,
                    frozen_weight,
                    component_mask,
                    weight_delta_mask,
                    decomposition_inputs.get("route"),
                    site_placement,
                    frozen_linear,
                )
            return site_forward(
                site_input,
                v,
                u,
                frozen_weight,
                decomposition_inputs["mask"],
                decomposition_inputs.get("delta"),
                decomposition_inputs.get("route"),
                site_placement,
                frozen_linear,
            )

        def execute_site(
            site_input: Array,
            kind: str,
            frozen_weight: Array,
            per_kind_inputs: dict[str, dict[str, Array]],
            *,
            frozen_linear: LinearPlan | None,
            site_placement: PlacementRules | PlannedComponentLinear | None,
        ) -> tuple[Array, Array | None]:
            if kind not in decomposed_kinds:
                return (
                    site_input @ frozen_weight.T
                    if frozen_linear is None
                    else placed_linear(site_input, frozen_weight.T, frozen_linear),
                    None,
                )
            decomposition_inputs = per_kind_inputs[kind]
            result = decomposed_site_output(
                site_input,
                frozen_weight,
                frozen_linear,
                decomposition_inputs,
                site_placement=site_placement,
            )
            component_activation = (
                result.component_activation if collect_component_activations else None
            )
            return result.output, component_activation

        def site_output_and_component_activation(
            site_input: Array,
            kind: str,
            frozen_weight: Array,
            per_kind_inputs: dict[str, dict[str, Array]],
        ) -> tuple[Array, Array | None]:
            if placement is None:
                return execute_site(
                    site_input,
                    kind,
                    frozen_weight,
                    per_kind_inputs,
                    frozen_linear=None,
                    site_placement=None,
                )
            assert kind in self.anatomy.kind_order, kind
            frozen_placement = (
                placement.target.row if kind in self.anatomy.row_kinds else placement.target.column
            )
            if kind in decomposed_kinds:
                external_axes = activation_axes(site_input.ndim, "feature")
                component_axes = activation_axes(site_input.ndim, "C")
                component_placement = PlannedComponentLinear(
                    v=placement.target_native_component_linear_plan(
                        frozen_placement,
                        V_WEIGHT_AXES,
                        external_axes,
                        component_axes,
                    ),
                    u=placement.target_native_component_linear_plan(
                        frozen_placement,
                        U_WEIGHT_AXES,
                        component_axes,
                        external_axes,
                    ),
                    component=placement.activations.component,
                    output=frozen_placement.output,
                )
            else:
                component_placement = None
            return execute_site(
                site_input,
                kind,
                frozen_weight,
                per_kind_inputs,
                frozen_linear=target_linear_plan(site_input, frozen_placement),
                site_placement=component_placement,
            )

        def compute_decomposed_block_activations(
            residual_in: Array,
            layer: GLULayer,
            per_kind_inputs: dict[str, dict[str, Array]],
        ) -> tuple[_GLUBlockActivations, dict[str, Array] | None]:
            anatomy = self.anatomy
            attn = layer.attn
            execute = site_output_and_component_activation
            h1 = rms_norm(residual_in, layer.ln1, self.eps)
            q, qa = execute(h1, anatomy.q, attn.wq, per_kind_inputs)
            k, ka = execute(h1, anatomy.k, attn.wk, per_kind_inputs)
            v, va = execute(h1, anatomy.v, attn.wv, per_kind_inputs)
            column = None if placement is None else placement.target.column
            attention_output = attn.core(
                q, k, v, self.inv_freq, None if column is None else column.output
            )
            o, oa = execute(attention_output, anatomy.o, attn.wo, per_kind_inputs)
            post_attention = residual_in + o
            h2 = rms_norm(post_attention, layer.ln2, self.eps)
            match layer.mlp, anatomy.mlp:
                case GatedMLP() as mlp, GatedMLPKinds() as mlp_kinds:
                    gate, ga = execute(h2, mlp_kinds.gate, mlp.Wg, per_kind_inputs)
                    up, ua = execute(h2, mlp_kinds.up, mlp.Wu, per_kind_inputs)
                    down_input = jax.nn.silu(gate) * up
                    down, da = execute(down_input, mlp_kinds.down, mlp.Wd, per_kind_inputs)
                    hidden: _GatedHiddenActs | _PlainHiddenActs = _GatedHiddenActs(
                        gate_output=gate, up_output=up
                    )
                    mlp_activations = {mlp_kinds.gate: ga, mlp_kinds.up: ua, mlp_kinds.down: da}
                case PlainMLP() as mlp, PlainMLPKinds() as mlp_kinds:
                    fc, fa = execute(h2, mlp_kinds.fc, mlp.Wfc, per_kind_inputs)
                    down_input = _gelu_tanh(fc)
                    down, da = execute(down_input, mlp_kinds.down, mlp.Wdown, per_kind_inputs)
                    hidden = _PlainHiddenActs(fc_output=fc)
                    mlp_activations = {mlp_kinds.fc: fa, mlp_kinds.down: da}
                case _:
                    raise AssertionError((type(layer.mlp), anatomy.mlp))
            residual_out = post_attention + down
            block_activations = _GLUBlockActivations(
                residual_in=residual_in,
                qkv_input=h1,
                q_output=q,
                k_output=k,
                v_output=v,
                attention_output=attention_output,
                o_output=o,
                post_attention_residual=post_attention,
                mlp_input=h2,
                hidden=hidden,
                down_input=down_input,
                down_output=down,
                residual_out=residual_out,
            )
            if not collect_component_activations:
                return block_activations, None
            component_activations_by_kind = {
                kind: activation
                for kind, activation in {
                    anatomy.q: qa,
                    anatomy.k: ka,
                    anatomy.v: va,
                    anatomy.o: oa,
                    **mlp_activations,
                }.items()
                if activation is not None
            }
            return block_activations, component_activations_by_kind

        policy = (
            jax.checkpoint_policies.nothing_saveable
            if remat
            else jax.checkpoint_policies.dots_saveable
        )

        def run_scan(body: Any, carry: Any, xs: Any) -> tuple[Any, Any]:
            return jax.lax.scan(jax.checkpoint(body, policy=policy), carry, xs)

        def slice_layers(lo: int, hi: int) -> GLULayer:
            return jax.tree.map(lambda a: a[lo:hi], self.stacked)

        captured_by_source: dict[_GLUCaptureSource, Array] = {}
        initial = _GLUCaptureSource(block=0, tap=_GLUTap.RESIDUAL_IN)
        if initial in capture_sources:
            captured_by_source[initial] = residual

        layout = _scan_capture_layout(capture_sources, self.n_layer) if capture_sources else None
        buffers = (
            {} if layout is None else _allocate_capture_buffers(layout, residual, self._value_width)
        )
        slot_indices_by_tap = {} if layout is None else _slot_index_arrays(layout)
        component_activation_segments: list[dict[str, Array]] = []
        bounds = sorted({0, first_decomposed, last_decomposed, self.n_layer})

        for lo, hi in zip(bounds, bounds[1:], strict=False):
            if lo == hi:
                continue
            segment_decomposed = (
                first_decomposed <= lo
                and hi <= last_decomposed
                and first_decomposed < last_decomposed
            )
            per_kind_segment_inputs: dict[str, dict[str, Array]] = {}
            if segment_decomposed:
                per_kind_segment_inputs = {
                    kind: {key: slice_leading(value, lo, hi) for key, value in entry.items()}
                    for kind, entry in per_kind.items()
                }
                xs: Any = (slice_layers(lo, hi), per_kind_segment_inputs)
            else:
                xs = slice_layers(lo, hi)

            if layout is not None:
                segment_slots = {key: value[lo:hi] for key, value in slot_indices_by_tap.items()}
                xs = (*xs, segment_slots) if segment_decomposed else (xs, segment_slots)

                def capture_body(
                    is_decomposed: bool,
                    capture_layout: _ScanCaptureLayout,
                ) -> Callable[..., Any]:
                    def body(
                        state: tuple[Array, dict[str, Array]],
                        layer_input: Any,
                    ) -> tuple[tuple[Array, dict[str, Array]], dict[str, Array] | None]:
                        x, buffers_ = state
                        if is_decomposed:
                            layer, per_kind_layer_inputs, slots = layer_input
                            block_activations, block_component_activations = (
                                compute_decomposed_block_activations(
                                    x, layer, per_kind_layer_inputs
                                )
                            )
                        else:
                            layer, slots = layer_input
                            block_activations = self._block_activations(x, layer, placement)
                            block_component_activations = None
                        updated_buffers = _write_block_captures(
                            capture_layout, buffers_, slots, block_activations
                        )
                        return (
                            block_activations.residual_out,
                            updated_buffers,
                        ), block_component_activations

                    return body

                (residual, buffers), segment_component_activations = run_scan(
                    capture_body(segment_decomposed, layout),
                    (residual, buffers),
                    xs,
                )
            else:

                def plain_block(
                    x: Array,
                    layer_input: Any,
                    segment_decomposed: bool = segment_decomposed,
                ) -> tuple[Array, dict[str, Array] | None]:
                    if segment_decomposed:
                        layer, per_kind_layer_inputs = layer_input
                        block_activations, block_component_activations = (
                            compute_decomposed_block_activations(x, layer, per_kind_layer_inputs)
                        )
                        return block_activations.residual_out, block_component_activations
                    return _clean_block(layer_input, x, self.inv_freq, self.eps, placement), None

                residual, segment_component_activations = run_scan(
                    plain_block,
                    residual,
                    xs,
                )

            if segment_component_activations is not None:
                component_activation_segments.append(segment_component_activations)

        if layout is not None:
            _read_capture_buffers(captured_by_source, layout, buffers)

        captures = _captures_in_request_order(capture_sources, captured_by_source)
        component_activations: dict[str, Array] = {}
        if collect_component_activations:
            assert component_activation_segments, (
                "component activations require a non-empty decomposed segment"
            )
            stacks = {
                kind: jnp.concatenate(
                    [part[kind] for part in component_activation_segments], axis=0
                )
                for kind in component_activation_segments[0]
            }
            for site in self.site_names:
                layer, kind = self.anatomy.family.parse(site)
                component_activations[site] = stacks[kind][layer - first_decomposed]
            assert set(component_activations) == site_set, (
                sorted(component_activations),
                sorted(site_set),
            )

        residual = rms_norm(residual, self.norm, self.eps)
        forward_result = ForwardResult.from_producer(
            output=self._output_logits(residual, placement),
            capture_keys=capture_keys,
            capture_values=captures,
        )
        return forward_result, component_activations

    def prepare_compute_weights(
        self, vu: ComponentStacks, placement: PlacementRules | None
    ) -> dict[str, dict[str, Array]]:
        """The ÷N→÷fsdp cross-`replicate` gather runs ONCE per step in ENTRY (off the hot
        path), landing a SMALL ÷fsdp-resident stack typed `reduced` over the gathered axes
        (`materialize_reduced_weights`) — the per-layer scan body then gathers ONE layer's
        `fsdp` shard transiently. The caller casts to compute dtype first, so the entry
        collective moves bf16 bytes."""
        compute = (
            vu
            if placement is None
            else component_stacks_to_compute_weights(vu, placement.components)
        )
        return _stack_per_kind_vu(self.anatomy, compute, self.n_layer)

    def component_activation_forward(
        self,
        prepared_weights: dict[str, dict[str, Array]],
        inputs: Int[Array, "b t"],
        /,
        *,
        capture_keys: CaptureKeys,
        placement: PlacementRules | None,
    ) -> tuple[ForwardResult, dict[str, Array]]:
        """Run one frozen forward for requested captures and every decomposed site's ``x @ V``."""
        anatomy = self.anatomy
        component_input_keys: list[str] = []
        site_locations = tuple(anatomy.family.parse(site) for site in self.site_names)
        for layer, kind in site_locations:
            if kind in (anatomy.q, anatomy.k, anatomy.v):
                component_input_keys.append(attention_input_tap_key(layer))
            elif kind == anatomy.o:
                component_input_keys.append(attention_output_tap_key(layer))
            elif kind in anatomy.mlp.hidden:
                component_input_keys.append(mlp_input_tap_key(layer))
            elif kind == anatomy.mlp.down:
                component_input_keys.append(mlp_hidden_tap_key(layer))
            else:
                raise AssertionError(kind)

        full_forward_result = self.clean_forward(
            inputs,
            capture_keys | frozenset(component_input_keys),
            placement=placement,
        )
        component_activations: dict[str, Array] = {}
        for site, key, (layer, kind) in zip(
            self.site_names, component_input_keys, site_locations, strict=True
        ):
            V = unreduce(prepared_weights[kind]["V"])[layer]
            site_input = full_forward_result.captures[key].astype(V.dtype)
            if placement is None:
                component_activations[site] = site_input @ V
            else:
                component_activations[site] = placed_linear(
                    site_input,
                    V,
                    placement.component_linear_plan(
                        V_WEIGHT_AXES,
                        activation_axes(site_input.ndim, "feature"),
                        activation_axes(site_input.ndim, "C"),
                    ),
                )
        requested_forward_result = ForwardResult(
            output=full_forward_result.output,
            captures={key: full_forward_result.captures[key] for key in sorted(capture_keys)},
        )
        return requested_forward_result, component_activations

    def stack_ci(self, ci_lower: dict[str, Array]) -> dict[str, Array]:
        return _stack_ci_per_kind(self.anatomy, ci_lower, self.n_layer)

    def masked_forward(
        self,
        prepared_weights: dict[str, dict[str, Array]],
        inputs: Int[Array, "b t"],
        /,
        *,
        masking: Masking,
        placement: PlacementRules | None,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        remat: bool,
    ) -> ForwardResult:
        masked_forward_result, _component_activations = self._run_masked_forward(
            prepared_weights,
            inputs,
            masking,
            remat,
            tuple(sorted(capture_keys)),
            collect_component_activations=False,
            placement=placement,
        )
        return masked_forward_result

    def masked_component_activations(
        self,
        prepared_weights: dict[str, dict[str, Array]],
        inputs: Int[Array, "b t"],
        masking: MaterializedMasking,
        *,
        placement: PlacementRules | None,
    ) -> dict[str, Array]:
        _forward_result, activations = self._run_masked_forward(
            prepared_weights,
            inputs,
            masking,
            False,
            (),
            collect_component_activations=True,
            placement=placement,
        )
        return activations

    def target_weight_sq_norms(self) -> dict[str, Array]:
        """Per-slot `‖W_s‖²` of each frozen stack, slot-aligned with `weight_deltas`
        (the S17 relative-error scales, read once at setup)."""
        norms: dict[str, list[Array]] = {}
        for name, group, _slot in site_slots_for(self.sites):
            layer, kind = self.anatomy.family.parse(name)
            frozen_weight = _frozen_site_weight(
                self.anatomy, jax.tree.map(lambda a, li=layer: a[li], self.stacked), kind
            )
            norms.setdefault(group, []).append(jnp.sum(frozen_weight.astype(jnp.float32) ** 2))
        return {group: jnp.stack(per_slot) for group, per_slot in norms.items()}

    def weight_deltas(self, vu: ComponentStacks) -> dict[str, Array]:
        """fp32 `W − V@U` per persistence stack from fp32 masters (SPEC N2; faithfulness
        input). Whole-stack einsum per group — never `vu.site()`, whose per-site slices
        of a stack-sharded persist layout redistribute cross-node."""
        out: dict[str, Array] = {}
        for group, (Vs, Us) in vu.stacks.items():
            slot_names = [name for name, g, _slot in vu.site_slots if g == group]
            Ws = jnp.stack(
                [
                    _frozen_site_weight(
                        self.anatomy, jax.tree.map(lambda a, li=layer: a[li], self.stacked), kind
                    )
                    for layer, kind in map(self.anatomy.family.parse, slot_names)
                ]
            )
            if not value_mesh(Vs).empty:
                # Land the delta PIECE-WISE, derived from the faithfulness operands'
                # own typing: g and d_out keep their assignments, and the C contraction
                # reduce-SCATTERS onto d_in (the matrix delta row's spelling) — a
                # replicated-d_in output would instead make XLA all-gather the full-C
                # f32 operands (replicate-count times the resident bytes).
                v_spec = jax.typeof(Vs).sharding.spec
                u_spec = jax.typeof(Us).sharding.spec
                delta_spec = P(v_spec[0], u_spec[2], v_spec[2])
                mesh = value_mesh(Vs)
                # Materialize the frozen slot stack BEFORE the subtract: without the
                # barrier GSPMD propagates the delta layout backward through the concat
                # and lowers the frozen slices as cross-node redistribution.
                Ws = jax.sharding.reshard(
                    jax.lax.optimization_barrier(Ws), NamedSharding(mesh, delta_spec)
                )
                vu_product = jnp.einsum(
                    "gic,gco->goi",
                    Vs.astype(jnp.float32),
                    Us.astype(jnp.float32),
                    out_sharding=NamedSharding(mesh, delta_spec),
                )
            else:
                vu_product = jnp.einsum(
                    "gic,gco->goi", Vs.astype(jnp.float32), Us.astype(jnp.float32)
                )
            out[group] = Ws.astype(jnp.float32) - vu_product
        return out


# ----------------------------- HF weight loading -----------------------------


def hf_snapshot_dir(model_name: str) -> Path:
    """Newest local snapshot of `model_name`, from the STANDARD HF hub cache
    (`HF_HUB_CACHE`, else `~/.cache/huggingface/hub`). Cluster launchers should export
    `HF_HUB_CACHE` to a shared world-readable cache — a home `~/.cache` hub is silently
    mutable, and a wiped entry strands running jobs that reload weights on requeue."""
    import os

    cache = Path(os.environ.get("HF_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub")))
    repo = "models--" + model_name.replace("/", "--")
    snaps = sorted((cache / repo / "snapshots").iterdir())
    assert snaps, f"no snapshot for {model_name} under {cache}"
    return snaps[-1]


class HFWeights:
    """Lazy keyed access to the safetensors of an HF checkpoint, cast to the FAMILY's
    frozen-weights dtype (SPEC N1: bf16 storage — the families pass it; this module holds
    no dtype opinion)."""

    def __init__(self, snapshot: Path, dtype: DTypeLike):
        index_path = snapshot / "model.safetensors.index.json"
        single_path = snapshot / "model.safetensors"
        match index_path.exists(), single_path.exists():
            case True, False:
                index = json.loads(index_path.read_text())
                self._key_to_file = index["weight_map"]
            case False, True:
                with safe_open(str(single_path), framework="numpy") as weights:
                    self._key_to_file = dict.fromkeys(weights.keys(), single_path.name)
            case has_index, has_single:
                raise AssertionError(
                    f"expected exactly one HF safetensors layout under {snapshot}, got "
                    f"index={has_index}, single={has_single}"
                )
        self._snapshot = snapshot
        self._dtype = dtype
        self._open: dict[str, Any] = {}

    def get(self, key: str) -> Array:
        fname = self._key_to_file[key]
        if fname not in self._open:
            self._open[fname] = safe_open(str(self._snapshot / fname), framework="numpy")
        host_array = np.asarray(self._open[fname].get_tensor(key), dtype=self._dtype)
        return cast(Array, cast(object, host_array))


AttnLoader = Callable[[HFWeights, int], FrozenAttn]
"""`(weights, layer_idx) -> the family's FrozenAttn` — each family file supplies its own
(Llama: plain projections; Qwen3: plus the q/k_norm keys)."""


def load_glu_blocks(w: HFWeights, cfg: GLUArch, load_attn: AttnLoader) -> list[GLULayer]:
    pre = "model.layers"
    return [
        GLULayer(
            ln1=w.get(f"{pre}.{i}.input_layernorm.weight"),
            ln2=w.get(f"{pre}.{i}.post_attention_layernorm.weight"),
            attn=load_attn(w, i),
            mlp=GatedMLP(
                Wg=w.get(f"{pre}.{i}.mlp.gate_proj.weight"),
                Wu=w.get(f"{pre}.{i}.mlp.up_proj.weight"),
                Wd=w.get(f"{pre}.{i}.mlp.down_proj.weight"),
            ),
        )
        for i in range(cfg.n_layer)
    ]


def build_engine_model(
    embed: Array,
    layers: list[GLULayer],
    norm: Array,
    lm_head: Float[Array, "vocab d"] | TiedHead,
    inv_freq: Array,
    cfg: GLUArch,
    sites: tuple[SiteSpec, ...],
    anatomy: Anatomy,
) -> GLUDecomposedModel:
    """Assemble an engine model from the frozen full-model arrays + decomposition config
    for ANY declared anatomy. `sites` must be canonical-ordered with dims matching `cfg`."""
    site_cs = tuple(SiteC(s.name, s.C) for s in sites)
    expected = family.site_specs(
        anatomy.family,
        family.canonical_site_cs(anatomy.family, site_cs),
        lambda kind: anatomy_site_dims(anatomy, cfg, kind),
        lambda kind: anatomy_nonlinearity_partition(anatomy, cfg, kind),
        cfg.n_layer,
    )
    assert sites == expected, f"sites are not the canonical specs for this config: {sites}"
    return GLUDecomposedModel(
        embed=embed,
        stacked=_stack_layers(layers),
        n_layer=len(layers),
        norm=norm,
        lm_head=lm_head,
        inv_freq=inv_freq,
        sites=sites,
        anatomy=anatomy,
        has_position_axis=True,
        eps=cfg.rms_norm_eps,
        n_ctx=cfg.n_ctx,
    )


def build_decomposed_lm(
    embed: Array,
    layers: list[GLULayer],
    norm: Array,
    lm_head: Array | TiedHead,
    inv_freq: Array,
    cfg: GLUArch,
    sites: tuple[SiteSpec, ...],
) -> GLUDecomposedModel:
    """`build_engine_model` at the GLU anatomy — the HF families' entry point."""
    return build_engine_model(embed, layers, norm, lm_head, inv_freq, cfg, sites, GLU_ANATOMY)


def load_decomposed_glu_from_hf(
    model_name: str,
    cfg: HFGLUArch,
    sites: tuple[SiteSpec, ...],
    load_attn: AttnLoader,
    inv_freq: Array,
    weights_dtype: DTypeLike,
) -> GLUDecomposedModel:
    """Load a GLU-transformer `DecomposedModel` from the cached HF snapshot: the full
    frozen model (embedding, all blocks, final norm, lm_head) as fields plus the static
    decomposition config (`sites`). The FAMILY contributes `load_attn` + `inv_freq` (its
    RoPE flavor); blocks without a decomposed site run the plain frozen path."""
    w = HFWeights(hf_snapshot_dir(model_name), weights_dtype)
    return build_decomposed_lm(
        embed=w.get("model.embed_tokens.weight"),
        layers=load_glu_blocks(w, cfg, load_attn),
        norm=w.get("model.norm.weight"),
        lm_head=TiedHead() if cfg.tie_word_embeddings else w.get("lm_head.weight"),
        inv_freq=inv_freq,
        cfg=cfg,
        sites=sites,
    )
