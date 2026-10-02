"""Shared dense transformer decomposition: frozen modules, site grammar, masked
execution, and weight loading. Llama, Qwen3, and LlamaSimpleMLP supply their architecture,
attention variant, and gated or plain MLP. Execution follows those structural choices.

The decomposed sites are any per-layer weight matrices named torch-style:
`layers.{i}.self_attn.{q,k,v,o}_proj` and `layers.{i}.mlp.{gate,up,down}_proj`, each
with its own C. `TransformerDecomposedModel` (an `eqx.Module`) carries the full frozen model —
embedding through every layer to the LM head — as array fields, threaded into the jitted
step as a pytree arg; layers without sites run the plain frozen block.

q/k/v sites are decomposed BEFORE `_prep_qk`/RoPE/SDPA (the masked site output feeds the
attention math); the o site applies to the attention output. V/U masters are fp32
keyed per site (`ComponentStacks`); frozen weights are stored bf16 — the trainer
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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Literal, Protocol, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Float32, Int, PRNGKeyArray, jaxtyped
from safetensors import safe_open

from param_decomp.attention import AttentionImplementation, causal_attention_head_first
from param_decomp.core import family
from param_decomp.core.axes import Axes, SemanticAxis
from param_decomp.core.components import (
    ComponentStacks,
    DenseFactorization,
    SiteC,
    SiteCI,
    SiteDims,
    SiteSpec,
    activation_axes,
    component_stacks_from_site_arrays,
    require_full_emission,
    site_stack_indices_for,
)
from param_decomp.core.decomposed_linear import (
    DenseComponentOverride,
    PlannedComponentLinear,
    SiteWeights,
    overridden_site_forward,
    site_forward,
)
from param_decomp.core.family import ArchFamily
from param_decomp.core.flops.target import (
    GradientTarget,
    TargetPassFlops,
    causal_attention_flops,
    linear_flops,
    sum_target_flops,
    validate_batch,
)
from param_decomp.core.gauge import u_norms_of
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
    ComponentOverrides,
    ForwardResult,
    Masking,
    MaterializedMasking,
    SiteOverrides,
    SiteRoutes,
    SourceMasking,
    StochasticMasking,
    validate_routes,
)
from param_decomp.core.nonlinearity import (
    ComponentSide,
    KVHeads,
    Neurons,
    NonlinearityAlignment,
    QueryHeads,
)
from param_decomp.core.placement import (
    PlacedRule,
    PlacementRules,
    TargetLinearPlacement,
    component_stacks_to_compute_weights,
    constrain_activation,
    materialize_stored_weight,
    placed_target_linear,
    target_linear_plan,
)
from param_decomp.core.pytree import ShardingTree
from param_decomp.core.source_mask import SourceMaskIngredients
from param_decomp.lm.batch import LMBatchWithDocuments
from param_decomp.sequence import SequenceLayout
from param_decomp.target_ports.llama import apply_rope, repeat_kv, rms_norm, rope_cos_sin
from param_decomp.targets.host import cpu_staging
from param_decomp.targets.lm_output import (
    LMOutput,
    OutputEdge,
    linear_output,
    pin_lm_output_batch,
)
from param_decomp.targets.losses import lm_output_kl_per_position
from param_decomp.targets.transformer_taps import (
    BlockCaptures,
    BlockTap,
    BlockTapName,
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


class TransformerArch(Protocol):
    """The arch-config surface the shared machinery reads. Family configs satisfy it
    structurally — the vendored `LlamaConfig` (llama3 rope-scaling fields on top of
    these) and the family-neutral `TransformerConfig`."""

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


class HFTransformerArch(TransformerArch, Protocol):
    """The shared architecture fields required by Hugging Face causal-LM loaders."""

    @property
    def tie_word_embeddings(self) -> bool: ...


@dataclass(frozen=True)
class TransformerConfig:
    """A family-neutral transformer arch config (plain RoPE, no family extras).
    Families with more knobs bring their own config type satisfying `TransformerArch`."""

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
    one anatomical fork (`GatedMLPKinds` | `PlainMLPKinds`); a model's MLP weights carry
    the same kinds, and the model rejects an anatomy that disagrees with them."""

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

    def site_input_key(self, site: str) -> str:
        layer, kind = self.family.parse(site)
        key_of_kind = {
            self.q: attention_input_tap_key(layer),
            self.k: attention_input_tap_key(layer),
            self.v: attention_input_tap_key(layer),
            self.o: attention_output_tap_key(layer),
            **{hidden: mlp_input_tap_key(layer) for hidden in self.mlp.hidden},
            self.mlp.down: mlp_hidden_tap_key(layer),
        }
        assert kind in key_of_kind, f"unknown {self.family.key} site kind {kind!r}"
        return key_of_kind[kind]

    def site_depth(self, site: str) -> int:
        """Where a site reads in the forward: sites of one depth read the same input, and
        every site of a greater depth runs after it."""
        layer, kind = self.family.parse(site)
        depth_of_kind = {
            self.q: 0,
            self.k: 0,
            self.v: 0,
            self.o: 1,
            **{hidden: 2 for hidden in self.mlp.hidden},
            self.mlp.down: 3,
        }
        assert kind in depth_of_kind, f"unknown {self.family.key} site kind {kind!r}"
        return 4 * layer + depth_of_kind[kind]

    def site_output_tap(self, kind: str) -> "_TransformerTap":
        match self.mlp:
            case GatedMLPKinds(gate=gate, up=up, down=down):
                mlp_taps = {
                    gate: _TransformerTap.GATE_OUTPUT,
                    up: _TransformerTap.UP_OUTPUT,
                    down: _TransformerTap.DOWN_OUTPUT,
                }
            case PlainMLPKinds(fc=fc, down=down):
                mlp_taps = {fc: _TransformerTap.FC_OUTPUT, down: _TransformerTap.DOWN_OUTPUT}
        tap_of_kind = {
            self.q: _TransformerTap.Q_OUTPUT,
            self.k: _TransformerTap.K_OUTPUT,
            self.v: _TransformerTap.V_OUTPUT,
            self.o: _TransformerTap.O_OUTPUT,
            **mlp_taps,
        }
        assert kind in tap_of_kind, f"unknown {self.family.key} site kind {kind!r}"
        return tap_of_kind[kind]


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

GLU_MLP_KINDS = GatedMLPKinds(gate="gate", up="up", down="down")

GLU_ANATOMY = Anatomy(family=FAMILY, q="q", k="k", v="v", o="o", mlp=GLU_MLP_KINDS)
"""The HF GLU families' vocabulary bound to the engine's structural roles."""


def anatomy_site_dims(anatomy: Anatomy, cfg: TransformerArch, kind: str) -> SiteDims:
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


def capture_grammar(
    anatomy: Anatomy, n_layer: int, d_resid: int, dims_of: Callable[[str], SiteDims]
) -> TransformerTapGrammar:
    """Every block computes the same vectors: all four site inputs and the post-attention
    residual, plus each matrix's output."""
    block = BlockCaptures(
        tap_widths={
            "attn_in": d_resid,
            "attn_out": dims_of(anatomy.o).d_in,
            "post_attn": d_resid,
            "mlp_in": d_resid,
            "mlp_hidden": dims_of(anatomy.mlp.down).d_in,
        },
        site_output_widths={kind: dims_of(kind).d_out for kind in anatomy.kind_order},
    )
    return TransformerTapGrammar(family=anatomy.family, d_resid=d_resid, blocks=(block,) * n_layer)


def transformer_flops(
    cfg: TransformerArch,
    anatomy: Anatomy,
    sites: tuple[SiteSpec, ...],
    *,
    batch_size: int,
    sequence_length: int,
    gradients: GradientTarget,
    include_frozen_paths: bool,
    capture_keys: frozenset[str],
) -> TargetPassFlops:
    """Count projections and attention, following the two residual branches explicitly."""
    validate_batch(batch_size, sequence_length)
    by_name = {site.name: site for site in sites}
    if len(by_name) != len(sites):
        raise ValueError("Decomposed site names must be unique")
    names = {
        anatomy.family.name_of(layer, kind)
        for layer in range(cfg.n_layer)
        for kind in anatomy.kind_order
    }
    if unknown := by_name.keys() - names:
        raise ValueError(f"Sites outside the target architecture: {sorted(unknown)}")
    grammar = capture_grammar(
        anatomy, cfg.n_layer, cfg.n_embd, lambda kind: anatomy_site_dims(anatomy, cfg, kind)
    )
    captured = frozenset(grammar.parse(key) for key in capture_keys)

    def tap_captured(name: BlockTapName, layer: int) -> bool:
        return BlockTap(name=name, block=layer) in captured

    def output_captured(layer: int, kind: str) -> bool:
        return (
            SiteOutput(name=anatomy.family.name_of(layer, kind), block=layer, kind=kind) in captured
        )

    auxiliary_projections: dict[str, bool] = {}
    auxiliary_attention: dict[int, bool] = {}
    residual_auxiliary = False
    for layer in reversed(range(cfg.n_layer)):
        residual_auxiliary = residual_auxiliary or ResidualBoundary(boundary=layer + 1) in captured
        down = anatomy.family.name_of(layer, anatomy.mlp.down)
        auxiliary_projections[down] = residual_auxiliary or output_captured(layer, anatomy.mlp.down)
        hidden_auxiliary = auxiliary_projections[down] or tap_captured("mlp_hidden", layer)
        for kind in anatomy.mlp.hidden:
            name = anatomy.family.name_of(layer, kind)
            auxiliary_projections[name] = hidden_auxiliary or output_captured(layer, kind)
            residual_auxiliary = residual_auxiliary or auxiliary_projections[name]
        residual_auxiliary = (
            residual_auxiliary or tap_captured("mlp_in", layer) or tap_captured("post_attn", layer)
        )
        output = anatomy.family.name_of(layer, anatomy.o)
        auxiliary_projections[output] = residual_auxiliary or output_captured(layer, anatomy.o)
        auxiliary_attention[layer] = auxiliary_projections[output] or tap_captured(
            "attn_out", layer
        )
        for kind in (anatomy.q, anatomy.k, anatomy.v):
            name = anatomy.family.name_of(layer, kind)
            auxiliary_projections[name] = auxiliary_attention[layer] or output_captured(layer, kind)
            residual_auxiliary = residual_auxiliary or auxiliary_projections[name]
        residual_auxiliary = (
            residual_auxiliary
            or tap_captured("attn_in", layer)
            or ResidualBoundary(boundary=layer) in captured
        )

    n_rows = batch_size * sequence_length
    terms: list[TargetPassFlops] = []

    def project(layer: int, kind: str, input_changed: bool) -> bool:
        name = anatomy.family.name_of(layer, kind)
        dims = anatomy_site_dims(anatomy, cfg, kind)
        terms.append(
            linear_flops(
                name,
                n_rows,
                dims.d_in,
                dims.d_out,
                by_name.get(name),
                gradients=gradients,
                input_changed=input_changed,
                include_frozen_paths=include_frozen_paths,
                output_gradient=True,
                auxiliary_gradient=auxiliary_projections[name],
                component_key=name,
            )
        )
        return input_changed or name in by_name

    residual_changed = False
    for layer in range(cfg.n_layer):
        query_changed = project(layer, anatomy.q, residual_changed)
        key_changed = project(layer, anatomy.k, residual_changed)
        value_changed = project(layer, anatomy.v, residual_changed)
        terms.append(
            causal_attention_flops(
                batch_size,
                sequence_length,
                cfg.n_head,
                cfg.head_dim,
                gradients=gradients,
                query_changed=query_changed,
                key_changed=key_changed,
                value_changed=value_changed,
                auxiliary_gradient=auxiliary_attention[layer],
            )
        )
        attention_changed = query_changed or key_changed or value_changed
        attention_output_changed = project(layer, anatomy.o, attention_changed)
        residual_changed = residual_changed or attention_output_changed
        match anatomy.mlp:
            case GatedMLPKinds(gate=gate, up=up, down=down):
                gate_changed = project(layer, gate, residual_changed)
                up_changed = project(layer, up, residual_changed)
                hidden_changed = gate_changed or up_changed
            case PlainMLPKinds(fc=fc, down=down):
                hidden_changed = project(layer, fc, residual_changed)
        mlp_output_changed = project(layer, down, hidden_changed)
        residual_changed = residual_changed or mlp_output_changed
    terms.append(
        linear_flops(
            "lm_head",
            n_rows,
            cfg.n_embd,
            cfg.vocab_size,
            None,
            gradients=gradients,
            input_changed=residual_changed,
            include_frozen_paths=False,
            output_gradient=True,
            auxiliary_gradient=False,
            component_key="lm_head",
        )
    )
    return sum_target_flops(terms)


def site_dims(cfg: TransformerArch, kind: str) -> SiteDims:
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


def anatomy_nonlinearity_alignment(
    anatomy: Anatomy, cfg: TransformerArch, kind: str
) -> NonlinearityAlignment:
    """The nonlinearity-facing side and partition for each architectural role."""
    if kind in anatomy.mlp.hidden:
        return NonlinearityAlignment("output", Neurons())
    if kind == anatomy.q:
        return NonlinearityAlignment("output", QueryHeads(cfg.n_head))
    if kind in (anatomy.k, anatomy.v):
        assert cfg.n_head % cfg.n_kv_head == 0, (cfg.n_head, cfg.n_kv_head)
        return NonlinearityAlignment("output", KVHeads(cfg.n_kv_head, cfg.n_head // cfg.n_kv_head))
    if kind == anatomy.o:
        return NonlinearityAlignment("input", QueryHeads(cfg.n_head))
    if kind == anatomy.mlp.down:
        return NonlinearityAlignment("input", Neurons())
    raise AssertionError(f"unknown kind {kind!r}")


def nonlinearity_alignment(cfg: TransformerArch, kind: str) -> NonlinearityAlignment:
    return anatomy_nonlinearity_alignment(GLU_ANATOMY, cfg, kind)


def glu_site_specs(cfg: TransformerArch, site_cs: tuple[SiteC, ...]) -> tuple[SiteSpec, ...]:
    return family.site_specs(
        FAMILY,
        site_cs,
        lambda kind, c: site_dims(cfg, kind).dense(c),
        lambda kind: nonlinearity_alignment(cfg, kind),
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

    def _projection_shardings(
        self, placement: PlacementRules, axes: Axes
    ) -> tuple[NamedSharding, NamedSharding]:
        assert axes in (("d_out", "d_in"), ("layer", "d_out", "d_in")), axes
        column = placement.target.column.persist
        row = placement.target.row.persist
        for w in (self.wq, self.wk, self.wv):
            column.validate_shape(axes, w.shape)
        row.validate_shape(axes, self.wo.shape)
        return column.sharding_for(axes), row.sharding_for(axes)

    def shardings(self, placement: PlacementRules, axes: Axes) -> ShardingTree:
        column, row = self._projection_shardings(placement, axes)
        return eqx.tree_at(
            lambda a: (a.wq, a.wk, a.wv, a.wo),
            self,
            (column, column, column, row),
        )

    def core(
        self,
        q_flat: Float[Array, "b t qd"],
        k_flat: Float[Array, "b t kvd"],
        v_flat: Float[Array, "b t kvd"],
        inv_freq: Array,
        sequence: SequenceLayout,
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
                f"{activation_row.label_for_log}: q_head and kv_head must carry the same mesh "
                f"assignment (cuDNN SDPA shards q/k/v identically): "
                f"{activation_row.assignment('q_head')!r} != "
                f"{activation_row.assignment('kv_head')!r}"
            )
            qkv_spec = activation_row.sharding_for(q_axes)
        q = q_flat.reshape(b, t, self.n_head, self.head_dim)
        k = k_flat.reshape(b, t, self.n_kv_head, self.head_dim)
        q, k = self._prep_qk(q, k)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v_flat.reshape(b, t, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        cos, sin = rope_cos_sin(inv_freq, sequence.position_ids(), q_flat.dtype)
        q, k = apply_rope(q, k, cos, sin)
        output = (
            causal_attention_head_first(q, k, v, sequence, qkv_spec, self.implementation)
            .transpose(0, 2, 1, 3)
            .reshape(b, t, self.n_head * self.head_dim)
        )
        return constrain_activation(output, activation_row)

    def __call__(
        self, x: Float[Array, "b t d"], inv_freq: Array, sequence: SequenceLayout
    ) -> Array:
        return (
            self.core(x @ self.wq.T, x @ self.wk.T, x @ self.wv.T, inv_freq, sequence, None)
            @ self.wo.T
        )

    def pattern(
        self,
        q_flat: Float[Array, "b t qd"],
        k_flat: Float[Array, "b t kvd"],
        inv_freq: Array,
        sequence: SequenceLayout,
    ) -> Float[Array, "b h t t"]:
        """Post-softmax causal attention map from flat Q/K projections — the target-owned
        recipe behind the attn-patterns eval (`TransformerDecomposedModel.attention_pattern_from_qk`). Same
        `_prep_qk`/RoPE/GQA math as `core`; scores in fp32, scaled by `1/√head_dim`,
        causal-masked, softmaxed. Requesting this diagnostic materializes [B, H, T, T]
        scores independently of the forward attention backend."""
        b, t, _ = q_flat.shape
        assert q_flat.shape[-1] == self.n_head * self.head_dim, q_flat.shape
        assert k_flat.shape[-1] == self.n_kv_head * self.head_dim, k_flat.shape
        q = q_flat.reshape(b, t, self.n_head, self.head_dim)
        k = k_flat.reshape(b, t, self.n_kv_head, self.head_dim)
        q, k = self._prep_qk(q, k)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        cos, sin = rope_cos_sin(inv_freq, sequence.position_ids(), q_flat.dtype)
        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32))
        scores = scores / self.head_dim**0.5
        allowed = sequence.attention_mask() & jnp.tril(jnp.ones((t, t), bool))
        return jax.nn.softmax(jnp.where(allowed[:, None, :, :], scores, -jnp.inf), axis=-1)


class GatedMLP(eqx.Module):
    """SwiGLU MLP weights: `silu(x@Wg.T) * (x@Wu.T) @ Wd.T` (Llama-3.1, Qwen3)."""

    kinds: GatedMLPKinds = eqx.field(static=True)
    Wg: Float[Array, "di d"]
    Wu: Float[Array, "di d"]
    Wd: Float[Array, "d di"]

    def shardings(self, placement: PlacementRules, axes: Axes) -> ShardingTree:
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

    kinds: PlainMLPKinds = eqx.field(static=True)
    Wfc: Float[Array, "di d"]
    Wdown: Float[Array, "d di"]

    def shardings(self, placement: PlacementRules, axes: Axes) -> ShardingTree:
        column = placement.target.column.persist
        row = placement.target.row.persist
        column.validate_shape(axes, self.Wfc.shape)
        row.validate_shape(axes, self.Wdown.shape)
        return eqx.tree_at(
            lambda m: (m.Wfc, m.Wdown),
            self,
            (column.sharding_for(axes), row.sharding_for(axes)),
        )


class TransformerLayer(eqx.Module):
    """One layer's frozen weights — norms, attention, MLP. Decomposed sites read
    their frozen target W from here at forward time; layers without sites run the
    plain frozen block from the same fields. Weights pass as a runtime arg — never
    baked into the HLO as a multi-GB constant. The MLP is the enumerated anatomy
    arm (`GatedMLP | PlainMLP`) — static treedef, so one model scans one anatomy."""

    ln1: Float[Array, " d"]
    ln2: Float[Array, " d"]
    attn: FrozenAttn
    mlp: GatedMLP | PlainMLP

    def shardings(self, placement: PlacementRules) -> ShardingTree:
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


def _frozen_site_weight(anatomy: Anatomy, layer: TransformerLayer, kind: str) -> Array:
    attn_weight_of = {
        anatomy.q: layer.attn.wq,
        anatomy.k: layer.attn.wk,
        anatomy.v: layer.attn.wv,
        anatomy.o: layer.attn.wo,
    }
    if kind in attn_weight_of:
        return attn_weight_of[kind]
    match layer.mlp:
        case GatedMLP(kinds=kinds, Wg=Wg, Wu=Wu, Wd=Wd):
            return {kinds.gate: Wg, kinds.up: Wu, kinds.down: Wd}[kind]
        case PlainMLP(kinds=kinds, Wfc=Wfc, Wdown=Wdown):
            return {kinds.fc: Wfc, kinds.down: Wdown}[kind]


# ----------------------------- forwards -----------------------------


def _stack_layers(layers: list[TransformerLayer]) -> TransformerLayer:
    """Stack a per-layer `TransformerLayer` list into one whose array leaves carry a leading
    layer axis — the `xs` for a `lax.scan` over the (homogeneous) block stack. Static
    fields (attn head counts) ride in the treedef, shared across iterations."""
    return jax.tree.map(lambda *per_layer: jnp.stack(per_layer), *layers)


class _TransformerTap(Enum):
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


def _block_taps(anatomy: Anatomy) -> frozenset[_TransformerTap]:
    """Exactly the taps one block of this anatomy computes; `_run_transformer_block` asserts its
    result against it. The MLP arm reaches the tap vocabulary only through the anatomy's
    own hidden kinds, so a new arm is described once, where its kinds are declared."""
    hidden_outputs = tuple(anatomy.site_output_tap(kind) for kind in anatomy.mlp.hidden)
    return frozenset(
        (
            _TransformerTap.RESIDUAL_IN,
            _TransformerTap.QKV_INPUT,
            _TransformerTap.Q_OUTPUT,
            _TransformerTap.K_OUTPUT,
            _TransformerTap.V_OUTPUT,
            _TransformerTap.ATTENTION_OUTPUT,
            _TransformerTap.O_OUTPUT,
            _TransformerTap.POST_ATTENTION_RESIDUAL,
            _TransformerTap.MLP_INPUT,
            *hidden_outputs,
            _TransformerTap.DOWN_INPUT,
            _TransformerTap.DOWN_OUTPUT,
            _TransformerTap.RESIDUAL_OUT,
        )
    )


class _SiteExecutor(Protocol):
    """How `_run_transformer_block` turns one matrix site into its output — the block's only
    injected concern. It also carries the anatomy the kernel names sites by and the
    placement the attention seam must agree with (the q/k/v outputs' activation row)."""

    @property
    def anatomy(self) -> Anatomy: ...
    @property
    def placement(self) -> PlacementRules | None: ...
    def __call__(self, site_input: Array, kind: str, frozen_weight: Array) -> Array: ...


def _target_rule(anatomy: Anatomy, placement: PlacementRules, kind: str) -> TargetLinearPlacement:
    """The Megatron layout one frozen matrix takes: the residual writers (o, down) consume
    the column shard and reduce onto the waist; every other kind shards its output."""
    assert kind in anatomy.kind_order, kind
    return placement.target.row if kind in anatomy.row_kinds else placement.target.column


@dataclass(frozen=True)
class _FrozenSiteExecutor:
    """Every site applies exactly its frozen `W` — not the `V@U + (W−V@U)` identity, so
    non-decomposed layers carry no V/U gradient and no decomposition rounding."""

    anatomy: Anatomy
    placement: PlacementRules | None

    def __call__(self, site_input: Array, kind: str, frozen_weight: Array) -> Array:
        rule = None if self.placement is None else _target_rule(self.anatomy, self.placement, kind)
        return placed_target_linear(site_input, frozen_weight, rule)


@dataclass(frozen=True)
class _SitePlans:
    """A decomposed site's two routes under a placement — the target `x@W.T` and the
    `V`/`U` pair — both derived from the same target-matrix rule."""

    target: LinearPlan
    component: PlannedComponentLinear


def _site_plans(
    anatomy: Anatomy, placement: PlacementRules, kind: str, site_input: Array
) -> _SitePlans:
    rule = _target_rule(anatomy, placement, kind)
    external_axes = activation_axes(site_input.ndim, "feature")
    component_axes = activation_axes(site_input.ndim, "C")
    return _SitePlans(
        target=target_linear_plan(site_input, rule),
        component=PlannedComponentLinear(
            v=placement.target_native_component_linear_plan(
                rule, V_WEIGHT_AXES, external_axes, component_axes
            ),
            u=placement.target_native_component_linear_plan(
                rule, U_WEIGHT_AXES, component_axes, external_axes
            ),
            component=placement.activations.component,
            output=rule.output,
        ),
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _MaterializedSiteMask:
    mask: Array
    delta: Array | None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _StochasticSiteMask:
    ci: Array
    src_key: Array
    delta_key: Array


type _SiteMask = _MaterializedSiteMask | _StochasticSiteMask


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _RoutedSiteInputs:
    v: Array
    u: Array
    masking: _SiteMask
    route: Array | None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _OverriddenSiteInputs:
    v: Array
    u: Array
    masking: _SiteMask
    override: DenseComponentOverride | None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _FrozenLayerInputs:
    layer: TransformerLayer


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _DecomposedLayerInputs:
    layer: TransformerLayer
    sites: "_RoutedSites | _OverriddenSites"


def _undecomposed_site_output(
    anatomy: Anatomy,
    placement: PlacementRules | None,
    site_input: Array,
    kind: str,
    frozen_weight: Array,
) -> Array:
    rule = None if placement is None else _target_rule(anatomy, placement, kind)
    return placed_target_linear(site_input, frozen_weight, rule)


def _site_mask_and_delta(masking: _SiteMask) -> tuple[Array, Array | None]:
    """The site's component and delta masks; a stochastic recipe draws them here, inside
    the checkpointed block, so backward redraws from the same keys."""
    match masking:
        case _StochasticSiteMask(ci=ci, src_key=src_key, delta_key=delta_key):
            mask = ci + (1.0 - ci) * uniform_like(src_key, ci)
            return mask, uniform_like(delta_key, ci, drop_last_axis=True)
        case _MaterializedSiteMask(mask=mask, delta=delta):
            return mask, delta


def _decomposed_site_weights(
    anatomy: Anatomy,
    placement: PlacementRules | None,
    site_input: Array,
    kind: str,
    frozen_weight: Array,
    v: Array,
    u: Array,
) -> SiteWeights:
    plans = None if placement is None else _site_plans(anatomy, placement, kind, site_input)
    return SiteWeights(
        frozen_weight,
        v,
        u,
        None if plans is None else plans.target,
        None if plans is None else plans.component,
    )


@dataclass
class _RoutedSiteExecutor:
    """Sites whose kind is decomposed run `site_forward` against the frozen `W` under
    their route; the rest apply `W`. Every decomposed site's `x@V` lands in
    `component_activations` — already a live value of that site's forward, so holding
    it adds no operation; whether the scan body emits it as `ys` stays the caller's
    decision."""

    anatomy: Anatomy
    placement: PlacementRules | None
    per_kind_inputs: dict[str, _RoutedSiteInputs]
    component_activations: dict[str, Array] = field(default_factory=dict)

    def __call__(self, site_input: Array, kind: str, frozen_weight: Array) -> Array:
        inputs = self.per_kind_inputs.get(kind)
        if inputs is None:
            return _undecomposed_site_output(
                self.anatomy, self.placement, site_input, kind, frozen_weight
            )
        mask, delta = _site_mask_and_delta(inputs.masking)
        weights = _decomposed_site_weights(
            self.anatomy, self.placement, site_input, kind, frozen_weight, inputs.v, inputs.u
        )
        result = site_forward(site_input, weights, mask, delta, inputs.route)
        self.component_activations[kind] = result.component_activation
        return result.output


@dataclass
class _OverriddenSiteExecutor:
    """`_RoutedSiteExecutor` with every position decomposed and each overridden kind
    running `overridden_site_forward`."""

    anatomy: Anatomy
    placement: PlacementRules | None
    per_kind_inputs: dict[str, _OverriddenSiteInputs]
    component_activations: dict[str, Array] = field(default_factory=dict)

    def __call__(self, site_input: Array, kind: str, frozen_weight: Array) -> Array:
        inputs = self.per_kind_inputs.get(kind)
        if inputs is None:
            return _undecomposed_site_output(
                self.anatomy, self.placement, site_input, kind, frozen_weight
            )
        mask, delta = _site_mask_and_delta(inputs.masking)
        weights = _decomposed_site_weights(
            self.anatomy, self.placement, site_input, kind, frozen_weight, inputs.v, inputs.u
        )
        match inputs.override:
            case None:
                result = site_forward(site_input, weights, mask, delta, None)
            case DenseComponentOverride() as override:
                result = overridden_site_forward(site_input, weights, mask, override, delta)
        self.component_activations[kind] = result.component_activation
        return result.output


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _RoutedSites:
    """A routed masked forward's per-kind decomposed-site inputs."""

    per_kind: dict[str, _RoutedSiteInputs]

    def executor(self, anatomy: Anatomy, placement: PlacementRules | None) -> _RoutedSiteExecutor:
        return _RoutedSiteExecutor(anatomy, placement, self.per_kind)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _OverriddenSites:
    """An overridden masked forward's per-kind decomposed-site inputs."""

    per_kind: dict[str, _OverriddenSiteInputs]

    def executor(
        self, anatomy: Anatomy, placement: PlacementRules | None
    ) -> _OverriddenSiteExecutor:
        return _OverriddenSiteExecutor(anatomy, placement, self.per_kind)


def _run_transformer_block(
    layer: TransformerLayer,
    residual_in: Array,
    inv_freq: Array,
    eps: float,
    sequence: SequenceLayout,
    *,
    execute_site: _SiteExecutor,
) -> dict[_TransformerTap, Array]:
    """The transformer block's equations, stated ONCE. Every value the block computes
    comes back keyed by tap: each is already a live operand of the next equation, so the
    dict adds nothing to a caller that reads none — an empty capture request lowers to
    exactly the compact frozen graph. The frozen and masked forwards differ only in
    `execute_site`."""
    anatomy = execute_site.anatomy
    attn = layer.attn
    qkv_input = rms_norm(residual_in, layer.ln1, eps)
    q = execute_site(qkv_input, anatomy.q, attn.wq)
    k = execute_site(qkv_input, anatomy.k, attn.wk)
    v = execute_site(qkv_input, anatomy.v, attn.wv)
    column = None if execute_site.placement is None else execute_site.placement.target.column
    attention_output = attn.core(
        q, k, v, inv_freq, sequence, None if column is None else column.output
    )
    o = execute_site(attention_output, anatomy.o, attn.wo)
    post_attention_residual = residual_in + o
    mlp_input = rms_norm(post_attention_residual, layer.ln2, eps)
    match layer.mlp:
        case GatedMLP(kinds=kinds, Wg=Wg, Wu=Wu, Wd=Wd):
            gate = execute_site(mlp_input, kinds.gate, Wg)
            up = execute_site(mlp_input, kinds.up, Wu)
            down_input = jax.nn.silu(gate) * up
            hidden = {_TransformerTap.GATE_OUTPUT: gate, _TransformerTap.UP_OUTPUT: up}
            down_weight = Wd
        case PlainMLP(kinds=kinds, Wfc=Wfc, Wdown=Wdown):
            fc = execute_site(mlp_input, kinds.fc, Wfc)
            down_input = _gelu_tanh(fc)
            hidden = {_TransformerTap.FC_OUTPUT: fc}
            down_weight = Wdown
    down_output = execute_site(down_input, kinds.down, down_weight)
    taps = {
        _TransformerTap.RESIDUAL_IN: residual_in,
        _TransformerTap.QKV_INPUT: qkv_input,
        _TransformerTap.Q_OUTPUT: q,
        _TransformerTap.K_OUTPUT: k,
        _TransformerTap.V_OUTPUT: v,
        _TransformerTap.ATTENTION_OUTPUT: attention_output,
        _TransformerTap.O_OUTPUT: o,
        _TransformerTap.POST_ATTENTION_RESIDUAL: post_attention_residual,
        _TransformerTap.MLP_INPUT: mlp_input,
        **hidden,
        _TransformerTap.DOWN_INPUT: down_input,
        _TransformerTap.DOWN_OUTPUT: down_output,
        _TransformerTap.RESIDUAL_OUT: post_attention_residual + down_output,
    }
    assert taps.keys() == _block_taps(anatomy), sorted(tap.value for tap in taps)
    return taps


@dataclass(frozen=True, kw_only=True)
class _TransformerCaptureSource:
    block: int
    tap: _TransformerTap


TransformerCaptureSources = tuple[_TransformerCaptureSource, ...]


_UNUSED_CAPTURE_SLOT = -1


@dataclass(frozen=True, kw_only=True)
class _ScanCaptureLayout:
    """One exact-size scan-carry buffer for each requested value kind."""

    slot_by_block_per_tap: tuple[tuple[_TransformerTap, tuple[int, ...]], ...]
    sources: tuple[_TransformerCaptureSource, ...]


def _capture_source_for_point(
    anatomy: Anatomy, point: TransformerPoint
) -> _TransformerCaptureSource:
    match point:
        case ResidualBoundary(boundary=0):
            return _TransformerCaptureSource(block=0, tap=_TransformerTap.RESIDUAL_IN)
        case ResidualBoundary(boundary=boundary):
            return _TransformerCaptureSource(block=boundary - 1, tap=_TransformerTap.RESIDUAL_OUT)
        case BlockTap(name=name, block=block):
            return _TransformerCaptureSource(block=block, tap=_block_tap_value(name))
        case SiteOutput(block=block, kind=kind):
            return _TransformerCaptureSource(block=block, tap=anatomy.site_output_tap(kind))


def _block_tap_value(name: BlockTapName) -> _TransformerTap:
    match name:
        case "attn_in":
            return _TransformerTap.QKV_INPUT
        case "attn_out":
            return _TransformerTap.ATTENTION_OUTPUT
        case "post_attn":
            return _TransformerTap.POST_ATTENTION_RESIDUAL
        case "mlp_in":
            return _TransformerTap.MLP_INPUT
        case "mlp_hidden":
            return _TransformerTap.DOWN_INPUT


def _scan_capture_layout(sources: TransformerCaptureSources, n_layer: int) -> _ScanCaptureLayout:
    tap_slots: list[tuple[_TransformerTap, tuple[int, ...]]] = []
    for tap in _TransformerTap:
        blocks = tuple(source.block for source in sources if source.tap is tap)
        if not blocks or tap is _TransformerTap.RESIDUAL_IN:
            continue
        slot_by_block = [_UNUSED_CAPTURE_SLOT] * n_layer
        for slot, block in enumerate(blocks):
            slot_by_block[block] = slot
        tap_slots.append((tap, tuple(slot_by_block)))
    return _ScanCaptureLayout(slot_by_block_per_tap=tuple(tap_slots), sources=sources)


def _allocate_capture_buffers(
    layout: _ScanCaptureLayout,
    residual: Array,
    width_of: Callable[[_TransformerTap], int],
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
    taps: dict[_TransformerTap, Array],
) -> dict[str, Array]:
    updated_buffers = dict(buffers)
    for tap, _slot_tuple in layout.slot_by_block_per_tap:
        buffer_key = tap.value
        slot_index = slot_indices[buffer_key]
        captured_value = taps[tap]
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
    layout: _ScanCaptureLayout,
    buffers: dict[str, Array],
    embedding_residual: Array,
) -> tuple[Array, ...]:
    """Read captures in request order; the embedding residual precedes the scan."""
    slot_by_block_per_tap = dict(layout.slot_by_block_per_tap)
    return tuple(
        embedding_residual
        if source.tap is _TransformerTap.RESIDUAL_IN
        else buffers[source.tap.value][slot_by_block_per_tap[source.tap][source.block]]
        for source in layout.sources
    )


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
    for name, group, _slot in components.site_stack_indices:
        assert group == anatomy.family.parse(name)[1], (
            f"engine sites must be grouped by matrix kind, got {group!r} for {name}"
        )
    slot_of = {name: (group, slot) for name, group, slot in components.site_stack_indices}
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


def _stack_routes(anatomy: Anatomy, n_layers: int, kind: str, routes: SiteRoutes) -> Array:
    """Pad the undecomposed layers' routes with false, preserving the supplied routes'
    placement."""
    sample = next(iter(routes.values()))

    def missing() -> Array:
        value = jnp.zeros(sample.shape, sample.dtype)
        if value_mesh(sample).empty:
            return value
        return jax.sharding.reshard(value, jax.typeof(sample).sharding)

    names = [anatomy.family.name_of(layer, kind) for layer in range(n_layers)]
    return jnp.stack([routes[name] if name in routes else missing() for name in names])


def _prepare_per_kind_masks(
    anatomy: Anatomy,
    n_layers: int,
    component_masks: Mapping[str, Array],
    weight_delta_masks: Mapping[str, Array] | None,
) -> dict[str, _SiteMask]:
    """Prepare concrete masks, padding the undecomposed layers of each kind."""
    if not component_masks:
        return {}
    a_mask = next(iter(component_masks.values()))
    mask_lead = a_mask.shape[:-1]
    a_delta = (
        next(iter(weight_delta_masks.values()), None) if weight_delta_masks is not None else None
    )

    def filler(build: Callable[[], Array], sample: Array | None) -> Array:
        """Dummy entries for non-decomposed layers, typed like the real entries — a
        homogeneous stack requires one sharding across its parts."""
        value = build()
        if sample is None or value_mesh(sample).empty:
            return value
        return jax.sharding.reshard(value, jax.typeof(sample).sharding)

    per_kind: dict[str, _SiteMask] = {}
    samples = {anatomy.family.parse(name)[1]: mask for name, mask in component_masks.items()}
    for kind, sample in samples.items():
        C = sample.shape[-1]
        mask_dt = a_mask.dtype
        names = [anatomy.family.name_of(layer, kind) for layer in range(n_layers)]
        masks_k = jnp.stack(
            [
                component_masks[name]
                if name in component_masks
                else filler(lambda C=C, mask_dt=mask_dt: jnp.ones((*mask_lead, C), mask_dt), a_mask)
                for name in names
            ]
        )
        delta = None
        if weight_delta_masks is not None:
            delta_shape = a_delta.shape if a_delta is not None else mask_lead
            delta_dtype = a_delta.dtype if a_delta is not None else mask_dt
            delta = jnp.stack(
                [
                    weight_delta_masks[name]
                    if name in weight_delta_masks
                    else filler(
                        lambda shape=delta_shape, dt=delta_dtype: jnp.zeros(shape, dt), a_delta
                    )
                    for name in names
                ]
            )
        per_kind[kind] = _MaterializedSiteMask(masks_k, delta)
    return per_kind


def _stack_sites_per_kind[T: SiteCI | SourceMaskIngredients](
    anatomy: Anatomy, values: Mapping[str, T], n_layers: int
) -> dict[str, T]:
    """Stack each kind across layers, zero-filling absent sites leaf by leaf."""
    kinds: dict[str, T] = {}
    sample_by_kind: dict[str, T] = {}
    for name, value in values.items():
        sample_by_kind.setdefault(anatomy.family.parse(name)[1], value)
    for kind, sample in sample_by_kind.items():
        names = [anatomy.family.name_of(layer, kind) for layer in range(n_layers)]
        layers = [
            values[name] if name in values else jax.tree.map(jnp.zeros_like, sample)
            for name in names
        ]
        kinds[kind] = jax.tree.map(lambda *leaves: jnp.stack(leaves), *layers)
    return kinds


def _prepare_per_kind_stochastic(
    anatomy: Anatomy,
    n_layers: int,
    ci_stacked: Mapping[str, SiteCI],
    draw_key: Array,
) -> dict[str, _SiteMask]:
    """Prepare draw keys alongside shared CI; checkpointed blocks draw the masks."""
    src_base, delta_base = jax.random.split(draw_key)

    per_kind: dict[str, _SiteMask] = {}
    for kind in ci_stacked:
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
        per_kind[kind] = _StochasticSiteMask(
            require_full_emission(ci_stacked[kind]), src_keys, delta_keys
        )
    return per_kind


def _prepare_per_kind_sources(
    ingredients_stacked: dict[str, SourceMaskIngredients],
) -> dict[str, _SiteMask]:
    """Compose source masks before segmented execution; padded layers remain unread."""

    per_kind: dict[str, _SiteMask] = {}
    for kind in ingredients_stacked:
        mask = ingredients_stacked[kind].compose()
        assert isinstance(mask, Array), "this family's sites are dense-only"
        per_kind[kind] = _MaterializedSiteMask(mask, ingredients_stacked[kind].delta)
    return per_kind


def _scatter_component_overrides(
    overrides: ComponentOverrides, waist: tuple[int, ...]
) -> DenseComponentOverride:
    """One site's rows as a dense override over its waist. The rows may be traced, so an
    out-of-bounds or repeated row fails when the scatter runs."""
    assert overrides.indices.shape[1] == len(waist), (overrides.indices.shape, waist)
    rows = tuple(overrides.indices.T)
    in_bounds = jnp.all((overrides.indices >= 0) & (overrides.indices < jnp.array(waist)))
    hits = jnp.zeros(waist, jnp.int32).at[rows].add(1)
    hits = eqx.error_if(hits, ~in_bounds, "a component override indexes outside its site's waist")
    hits = eqx.error_if(hits, jnp.any(hits > 1), "a component override row repeats")
    values = jnp.zeros(waist, overrides.values.dtype).at[rows].set(overrides.values)
    return DenseComponentOverride(hits == 1, values)


def _stack_kind_overrides(
    anatomy: Anatomy,
    n_layers: int,
    kind: str,
    overrides: SiteOverrides,
    waist: tuple[int, ...],
) -> DenseComponentOverride | None:
    """One kind's dense overrides stacked over every layer, empty on layers without any;
    None when no site of the kind is overridden."""
    names = [anatomy.family.name_of(layer, kind) for layer in range(n_layers)]
    overridden = [overrides[name] for name in names if name in overrides]
    if not overridden:
        return None
    dtype = overridden[0].values.dtype
    assert all(site.values.dtype == dtype for site in overridden), [
        site.values.dtype for site in overridden
    ]
    none_overridden = DenseComponentOverride(jnp.zeros(waist, jnp.bool_), jnp.zeros(waist, dtype))
    per_layer = [
        _scatter_component_overrides(overrides[name], waist)
        if name in overrides
        else none_overridden
        for name in names
    ]
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *per_layer)


class TransformerPreparedMasking(eqx.Module):
    """Per-kind layer stacks of mask operands, ready for segmented execution."""

    per_kind: dict[str, _SiteMask]


class TransformerPreparedWeights(eqx.Module):
    """Per decomposed kind, the layer-stacked compute-layout `V` and `U`, and the placement
    the target's forwards execute them under."""

    per_kind: dict[str, dict[str, Array]]
    anatomy: Anatomy = eqx.field(static=True)
    placement: PlacementRules | None = eqx.field(static=True)

    def component_activations(
        self, site: str, x: Float[Array, "*leading d_in"]
    ) -> Float[Array, "*leading C"]:
        layer, kind = self.anatomy.family.parse(site)
        V = unreduce(self.per_kind[kind]["V"])[layer]
        x = x.astype(V.dtype)
        match self.placement:
            case None:
                return x @ V
            case PlacementRules() as placement:
                return placed_linear(
                    x,
                    V,
                    placement.component_linear_plan(
                        V_WEIGHT_AXES,
                        activation_axes(x.ndim, "feature"),
                        activation_axes(x.ndim, "C"),
                    ),
                )

    def u_norms(self, site: str) -> Float32[Array, " C"]:
        layer, kind = self.anatomy.family.parse(site)
        return u_norms_of(unreduce(self.per_kind[kind]["U"])[layer])


class TiedHead(eqx.Module):
    """The output projection reads the embedding parameter — one stored table, no second
    frozen leaf. A zero-leaf pytree node, so tied models carry no head array anywhere
    (jit args, shardings trees, audits). Families with untied heads carry the array."""


class TransformerDecomposedModel(eqx.Module):
    """Shared dense transformer implementation of `DecomposedModel`.
    Architecture differences live in the attention module, MLP variant, site anatomy,
    and rotary frequencies.

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
    stacked: TransformerLayer  # the per-layer weights stacked on a leading layer axis (the scan
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
    output_edge: OutputEdge = eqx.field(static=True)

    def __check_init__(self) -> None:
        assert self.anatomy.mlp == self.stacked.mlp.kinds, (
            self.anatomy.mlp,
            self.stacked.mlp.kinds,
        )

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
    def layers(self) -> list[TransformerLayer]:
        """Per-layer view of `stacked` (slices the leading layer axis). For non-hot
        consumers (equivalence harness); the forwards use `stacked`."""
        return [jax.tree.map(lambda a, idx=i: a[idx], self.stacked) for i in range(self.n_layer)]

    def frozen_site_weight(self, site: str) -> Float[Array, "d_out d_in"]:
        """The frozen target matrix `site` decomposes. Off the hot path: it slices one
        layer out of `stacked`."""
        layer, kind = self.anatomy.family.parse(site)
        return _frozen_site_weight(
            self.anatomy, jax.tree.map(lambda array: array[layer], self.stacked), kind
        )

    def attention_pattern_from_qk(
        self,
        q_site: str,
        q_flat: Float[Array, "b t qd"],
        k_flat: Float[Array, "b t kvd"],
        sequence: SequenceLayout,
    ) -> Float[Array, "b h t t"]:
        """Post-softmax causal attention map from a layer's flat Q/K projections — the
        target-owned recipe the attn-patterns eval consumes (`attn_patterns_eval`'s
        `AttnPatternModel` protocol). Delegates to that LAYER's own attention module, so
        family behavior (Qwen3's per-layer QK-norm) comes along for free."""
        layer = self.anatomy.family.parse(q_site)[0]
        attn = jax.tree.map(lambda a, idx=layer: a[idx], self.stacked.attn)
        return attn.pattern(q_flat, k_flat, self.inv_freq, sequence)

    def shardings(self, placement: PlacementRules) -> ShardingTree:
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
    def recon_loss_fn(masked_output: LMOutput, clean_output: LMOutput) -> Float[Array, ""]:
        return lm_output_kl_per_position(masked_output, clean_output)

    @staticmethod
    def pin_output_batch(output: LMOutput, mesh: Mesh | None) -> LMOutput:
        return pin_lm_output_batch(output, mesh)

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

    @jaxtyped(typechecker=beartype)
    def _output(
        self, residual: Float[Array, "batch seq d_model"], placement: PlacementRules | None
    ) -> LMOutput:
        if placement is None:
            weight = self.head_weight
        else:
            # Legal for a tied head too: `shardings` pins the embedding's persistence
            # layout to the output's, so the stored table satisfies the output role.
            weight = materialize_stored_weight(
                self.head_weight,
                placement.target.output.persist,
                placement.target.output.operand,
                axes=("vocab", "d_model"),
            )
        return linear_output(residual, weight, self.output_edge)

    def _capture_grammar(self) -> TransformerTapGrammar:
        def stacked_site_dims(kind: str) -> SiteDims:
            _n_layer, d_out, d_in = _frozen_site_weight(self.anatomy, self.stacked, kind).shape
            return SiteDims(d_in=d_in, d_out=d_out)

        return capture_grammar(self.anatomy, self.n_layer, self.embed.shape[1], stacked_site_dims)

    def site_output_keys(self, sites: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(site_output_tap_key(site) for site in sites)

    def _value_width(self, value: _TransformerTap) -> int:
        d = self.embed.shape[1]
        mlp = self.stacked.mlp
        match value:
            case (
                _TransformerTap.RESIDUAL_IN
                | _TransformerTap.QKV_INPUT
                | _TransformerTap.O_OUTPUT
                | _TransformerTap.POST_ATTENTION_RESIDUAL
                | _TransformerTap.MLP_INPUT
                | _TransformerTap.DOWN_OUTPUT
                | _TransformerTap.RESIDUAL_OUT
            ):
                return d
            case _TransformerTap.Q_OUTPUT:
                return self.stacked.attn.wq.shape[1]
            case _TransformerTap.K_OUTPUT:
                return self.stacked.attn.wk.shape[1]
            case _TransformerTap.V_OUTPUT:
                return self.stacked.attn.wv.shape[1]
            case _TransformerTap.ATTENTION_OUTPUT:
                return self.stacked.attn.wo.shape[2]
            case _TransformerTap.GATE_OUTPUT:
                assert isinstance(mlp, GatedMLP), value
                return mlp.Wg.shape[1]
            case _TransformerTap.UP_OUTPUT:
                assert isinstance(mlp, GatedMLP), value
                return mlp.Wu.shape[1]
            case _TransformerTap.FC_OUTPUT:
                assert isinstance(mlp, PlainMLP), value
                return mlp.Wfc.shape[1]
            case _TransformerTap.DOWN_INPUT:
                match mlp:
                    case GatedMLP(Wd=Wd):
                        return Wd.shape[2]
                    case PlainMLP(Wdown=Wdown):
                        return Wdown.shape[2]

    def _clean_output(
        self, inputs: LMBatchWithDocuments, placement: PlacementRules | None
    ) -> LMOutput:
        """Untouched graph used when no captures are requested — reading one tap of the
        kernel's result leaves the block exactly the compact frozen graph."""
        frozen_sites = _FrozenSiteExecutor(self.anatomy, placement)

        def block(residual: Array, layer: TransformerLayer) -> tuple[Array, None]:
            taps = _run_transformer_block(
                layer, residual, self.inv_freq, self.eps, inputs.sequence, execute_site=frozen_sites
            )
            return taps[_TransformerTap.RESIDUAL_OUT], None

        inputs.validate_shapes()
        residual = self.embed_tokens(inputs.batch.token_ids, placement)
        residual, _ = jax.lax.scan(block, residual, self.stacked)
        residual = rms_norm(residual, self.norm, self.eps)
        return self._output(residual, placement)

    def clean_forward(
        self,
        inputs: LMBatchWithDocuments,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        *,
        placement: PlacementRules | None,
    ) -> ForwardResult[LMOutput, LMBatchWithDocuments]:
        if not capture_keys:
            return ForwardResult.from_producer(
                leading_shape=inputs.batch.token_ids.shape,
                output=self._clean_output(inputs, placement),
                capture_keys=(),
                capture_values=(),
                conditioning=inputs,
                sequence=inputs.sequence,
            )
        ordered_capture_keys = tuple(sorted(capture_keys))
        capture_sources = self._capture_grammar().resolve(
            ordered_capture_keys, lambda point: _capture_source_for_point(self.anatomy, point)
        )

        inputs.validate_shapes()
        embedding_residual = self.embed_tokens(inputs.batch.token_ids, placement)
        residual = embedding_residual

        layout = _scan_capture_layout(capture_sources, self.n_layer)
        buffers = _allocate_capture_buffers(layout, residual, self._value_width)
        slot_indices_by_tap = _slot_index_arrays(layout)
        frozen_sites = _FrozenSiteExecutor(self.anatomy, placement)

        def block(
            state: tuple[Array, dict[str, Array]],
            layer_and_slots: tuple[TransformerLayer, dict[str, Array]],
        ) -> tuple[tuple[Array, dict[str, Array]], None]:
            x, buffers_ = state
            layer, slots = layer_and_slots
            taps = _run_transformer_block(
                layer, x, self.inv_freq, self.eps, inputs.sequence, execute_site=frozen_sites
            )
            return (
                taps[_TransformerTap.RESIDUAL_OUT],
                _write_block_captures(layout, buffers_, slots, taps),
            ), None

        (residual, buffers), _ = jax.lax.scan(
            block, (residual, buffers), (self.stacked, slot_indices_by_tap)
        )
        residual = rms_norm(residual, self.norm, self.eps)
        return ForwardResult.from_producer(
            leading_shape=inputs.batch.token_ids.shape,
            output=self._output(residual, placement),
            capture_keys=ordered_capture_keys,
            capture_values=_read_capture_buffers(layout, buffers, embedding_residual),
            conditioning=inputs,
            sequence=inputs.sequence,
        )

    def _run_masked_forward(
        self,
        prepared_weights: TransformerPreparedWeights,
        inputs: LMBatchWithDocuments,
        sites: _RoutedSites | _OverriddenSites,
        remat: bool,
        capture_keys: tuple[str, ...],
        *,
        collect_component_activations: bool,
        placement: PlacementRules | None,
    ) -> tuple[ForwardResult[LMOutput, LMBatchWithDocuments], dict[str, Array]]:
        """One segmented masked forward for output, captures and optional ``x@V`` diagnostics.

        Empty capture keys keep the compact frozen blocks; non-empty keys allocate only
        their exact-size capture slots.
        """
        assert (
            placement is None
            or placement.activations.masked_external is placement.activations.external
        ), (
            "sequence_sharding 'sequence_parallel' is only built for qwen36_moe; this "
            "target's masked forward would silently ignore the row"
        )
        capture_sources = (
            self._capture_grammar().resolve(
                capture_keys, lambda point: _capture_source_for_point(self.anatomy, point)
            )
            if capture_keys
            else ()
        )
        inputs.validate_shapes()
        embedding_residual = self.embed_tokens(inputs.batch.token_ids, placement)
        residual = embedding_residual
        site_set = frozenset(self.site_names)
        assert prepared_weights.placement == placement, "weights prepared under other rules"
        decomposed_kinds = frozenset(sites.per_kind)

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

        frozen_sites = _FrozenSiteExecutor(self.anatomy, placement)
        layout = _scan_capture_layout(capture_sources, self.n_layer)
        buffers = _allocate_capture_buffers(layout, residual, self._value_width)
        slot_indices_by_tap = _slot_index_arrays(layout)

        def block(
            state: tuple[Array, dict[str, Array]],
            inputs: tuple[_FrozenLayerInputs | _DecomposedLayerInputs, dict[str, Array]],
        ) -> tuple[tuple[Array, dict[str, Array]], dict[str, Array] | None]:
            residual_in, buffers_ = state
            layer_inputs, slots = inputs
            match layer_inputs:
                case _FrozenLayerInputs(layer=layer):
                    taps = _run_transformer_block(
                        layer,
                        residual_in,
                        self.inv_freq,
                        self.eps,
                        sequence,
                        execute_site=frozen_sites,
                    )
                    collected = None
                case _DecomposedLayerInputs(layer=layer, sites=layer_sites):
                    executor = layer_sites.executor(self.anatomy, placement)
                    taps = _run_transformer_block(
                        layer, residual_in, self.inv_freq, self.eps, sequence, execute_site=executor
                    )
                    activations = executor.component_activations
                    assert set(activations) == decomposed_kinds, (
                        sorted(activations),
                        sorted(decomposed_kinds),
                    )
                    collected = activations if collect_component_activations else None
            return (
                taps[_TransformerTap.RESIDUAL_OUT],
                _write_block_captures(layout, buffers_, slots, taps),
            ), collected

        policy = (
            jax.checkpoint_policies.nothing_saveable
            if remat
            else jax.checkpoint_policies.dots_saveable
        )

        def slice_layers(lo: int, hi: int) -> TransformerLayer:
            return jax.tree.map(lambda value: value[lo:hi], self.stacked)

        sequence = inputs.sequence
        component_stacks: dict[str, Array] | None = None
        for lo, hi in (
            (0, first_decomposed),
            (first_decomposed, last_decomposed),
            (last_decomposed, self.n_layer),
        ):
            if lo == hi:
                continue
            layer = slice_layers(lo, hi)
            layer_inputs = (
                _DecomposedLayerInputs(
                    layer, jax.tree.map(partial(slice_leading, lo=lo, hi=hi), sites)
                )
                if (lo, hi) == (first_decomposed, last_decomposed)
                else _FrozenLayerInputs(layer)
            )
            slots = {key: value[lo:hi] for key, value in slot_indices_by_tap.items()}
            (residual, buffers), collected = jax.lax.scan(
                jax.checkpoint(block, policy=policy),
                (residual, buffers),
                (layer_inputs, slots),
            )
            if collected is not None:
                component_stacks = collected

        captures = _read_capture_buffers(layout, buffers, embedding_residual)
        component_activations: dict[str, Array] = {}
        if collect_component_activations:
            assert component_stacks is not None, (
                "component activations require a non-empty decomposed segment"
            )
            for site in self.site_names:
                layer, kind = self.anatomy.family.parse(site)
                component_activations[site] = component_stacks[kind][layer - first_decomposed]

        residual = rms_norm(residual, self.norm, self.eps)
        forward_result = ForwardResult.from_producer(
            leading_shape=inputs.batch.token_ids.shape,
            output=self._output(residual, placement),
            capture_keys=capture_keys,
            capture_values=captures,
            conditioning=inputs,
            sequence=inputs.sequence,
        )
        return forward_result, component_activations

    def prepare_compute_weights(
        self, vu: ComponentStacks, placement: PlacementRules | None
    ) -> TransformerPreparedWeights:
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
        return TransformerPreparedWeights(
            _stack_per_kind_vu(self.anatomy, compute, self.n_layer), self.anatomy, placement
        )

    def component_activation_forward(
        self,
        prepared_weights: TransformerPreparedWeights,
        inputs: LMBatchWithDocuments,
        /,
        *,
        sites: tuple[str, ...],
        capture_keys: CaptureKeys,
        placement: PlacementRules | None,
    ) -> tuple[ForwardResult[LMOutput, LMBatchWithDocuments], dict[str, SiteCI]]:
        """Run one frozen forward for requested captures and each requested site's ``x @ V``."""
        assert set(sites) <= set(self.site_names), (sorted(sites), self.site_names)
        assert prepared_weights.placement == placement, "weights prepared under other rules"
        anatomy = self.anatomy
        component_input_keys = tuple(anatomy.site_input_key(site) for site in sites)

        full_forward_result = self.clean_forward(
            inputs,
            capture_keys | frozenset(component_input_keys),
            placement=placement,
        )
        component_activations: dict[str, SiteCI] = {
            site: prepared_weights.component_activations(site, full_forward_result.captures[key])
            for site, key in zip(sites, component_input_keys, strict=True)
        }
        requested_forward_result = ForwardResult(
            leading_shape=inputs.batch.token_ids.shape,
            output=full_forward_result.output,
            captures={key: full_forward_result.captures[key] for key in sorted(capture_keys)},
            conditioning=inputs,
            sequence=inputs.sequence,
        )
        return requested_forward_result, component_activations

    def prepare_masking(self, masking: Masking) -> TransformerPreparedMasking:
        match masking:
            case StochasticMasking(ci=ci, draw_key=draw_key):
                return self.prepare_stochastic_masking(ci)(draw_key)
            case SourceMasking(ingredients=ingredients):
                assert set(ingredients) == set(self.site_names)
                return TransformerPreparedMasking(
                    _prepare_per_kind_sources(
                        _stack_sites_per_kind(self.anatomy, ingredients, self.n_layer)
                    )
                )
            case MaterializedMasking(component_masks=masks, weight_delta_masks=deltas):
                assert set(masks) == set(self.site_names)
                return TransformerPreparedMasking(
                    _prepare_per_kind_masks(
                        self.anatomy,
                        self.n_layer,
                        {name: require_full_emission(mask) for name, mask in masks.items()},
                        deltas,
                    )
                )

    def prepare_stochastic_masking(
        self, ci: Mapping[str, SiteCI]
    ) -> Callable[[Array], TransformerPreparedMasking]:
        anatomy, n_layer = self.anatomy, self.n_layer
        assert set(ci) == set(self.site_names)
        stacked = _stack_sites_per_kind(anatomy, ci, n_layer)

        def draw(draw_key: Array) -> TransformerPreparedMasking:
            return TransformerPreparedMasking(
                _prepare_per_kind_stochastic(anatomy, n_layer, stacked, draw_key)
            )

        return draw

    def _routed_sites(
        self,
        prepared_weights: TransformerPreparedWeights,
        masking: TransformerPreparedMasking,
        routes: SiteRoutes | None,
    ) -> _RoutedSites:
        validate_routes(routes, self.site_names)
        stacks = prepared_weights.per_kind
        assert set(masking.per_kind) == set(stacks), (masking.per_kind.keys(), stacks.keys())
        return _RoutedSites(
            {
                kind: _RoutedSiteInputs(
                    stacks[kind]["V"],
                    stacks[kind]["U"],
                    entry,
                    None
                    if routes is None
                    else _stack_routes(self.anatomy, self.n_layer, kind, routes),
                )
                for kind, entry in masking.per_kind.items()
            }
        )

    def masked_forward(
        self,
        prepared_weights: TransformerPreparedWeights,
        inputs: LMBatchWithDocuments,
        /,
        *,
        masking: TransformerPreparedMasking,
        routes: SiteRoutes | None,
        placement: PlacementRules | None,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        remat: bool,
    ) -> ForwardResult[LMOutput, LMBatchWithDocuments]:
        masked_forward_result, _component_activations = self._run_masked_forward(
            prepared_weights,
            inputs,
            self._routed_sites(prepared_weights, masking, routes),
            remat,
            tuple(sorted(capture_keys)),
            collect_component_activations=False,
            placement=placement,
        )
        return masked_forward_result

    def overridden_forward(
        self,
        prepared_weights: TransformerPreparedWeights,
        inputs: LMBatchWithDocuments,
        /,
        *,
        masking: TransformerPreparedMasking,
        overrides: SiteOverrides,
        placement: PlacementRules | None,
        capture_keys: CaptureKeys,
        remat: bool,
    ) -> ForwardResult[LMOutput, LMBatchWithDocuments]:
        """`masked_forward` with `overrides` replacing chosen masked component
        activations after `masking` has produced them."""
        assert overrides and set(overrides) <= set(self.site_names), (
            sorted(overrides),
            self.site_names,
        )
        if placement is not None:
            raise NotImplementedError("component overrides on a placed forward")
        stacks = prepared_weights.per_kind
        assert set(masking.per_kind) == set(stacks), (masking.per_kind.keys(), stacks.keys())
        leading_shape = inputs.batch.token_ids.shape
        sites = _OverriddenSites(
            {
                kind: _OverriddenSiteInputs(
                    stacks[kind]["V"],
                    stacks[kind]["U"],
                    entry,
                    _stack_kind_overrides(
                        self.anatomy,
                        self.n_layer,
                        kind,
                        overrides,
                        (*leading_shape, stacks[kind]["V"].shape[-1]),
                    ),
                )
                for kind, entry in masking.per_kind.items()
            }
        )
        overridden_forward_result, _component_activations = self._run_masked_forward(
            prepared_weights,
            inputs,
            sites,
            remat,
            tuple(sorted(capture_keys)),
            collect_component_activations=False,
            placement=placement,
        )
        return overridden_forward_result

    def masked_component_activations(
        self,
        prepared_weights: TransformerPreparedWeights,
        inputs: LMBatchWithDocuments,
        masking: MaterializedMasking,
        *,
        placement: PlacementRules | None,
    ) -> dict[str, Array]:
        _forward_result, activations = self._run_masked_forward(
            prepared_weights,
            inputs,
            self._routed_sites(prepared_weights, self.prepare_masking(masking), None),
            False,
            (),
            collect_component_activations=True,
            placement=placement,
        )
        return activations

    def target_weight_sq_norms(self) -> dict[str, Array]:
        """Per-slot `‖W_s‖²` of each frozen stack, slot-aligned with `weight_deltas`
        (the relative-error scales, read once at setup)."""
        norms: dict[str, list[Array]] = {}
        for name, group, _slot in site_stack_indices_for(self.sites):
            frozen_weight = self.frozen_site_weight(name)
            norms.setdefault(group, []).append(jnp.sum(frozen_weight.astype(jnp.float32) ** 2))
        return {group: jnp.stack(per_slot) for group, per_slot in norms.items()}

    def weight_deltas(self, vu: ComponentStacks) -> dict[str, Array]:
        """fp32 `W − V@U` per persistence stack from fp32 masters (faithfulness
        input). Whole-stack einsum per group — never `vu.site()`, whose per-site slices
        of a stack-sharded persist layout redistribute cross-node."""
        out: dict[str, Array] = {}
        for group, (Vs, Us) in vu.stacks.items():
            slot_names = [name for name, g, _slot in vu.site_stack_indices if g == group]
            Ws = jnp.stack(
                [
                    _frozen_site_weight(
                        self.anatomy, jax.tree.map(lambda a, li=layer: a[li], self.stacked), kind
                    )
                    for layer, kind in map(self.anatomy.family.parse, slot_names)
                ]
            )
            if pad := vu.pad_of(group):
                # Persist-stack pad slots decompose a ZERO matrix: their deltas ride the
                # faithfulness lane as exact zeros (pad V/U are zero by invariant) and
                # exit at the loss reduction. The pad rows take the frozen stack's own
                # sharding — explicit-mode concatenate demands matching operand specs.
                zeros = jnp.zeros((pad, *Ws.shape[1:]), Ws.dtype)
                if not value_mesh(Ws).empty:
                    zeros = jax.sharding.reshard(zeros, jax.typeof(Ws).sharding)
                Ws = jnp.concatenate([Ws, zeros])
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


def nonlinearity_aligned_component_count(spec: SiteSpec) -> int:
    """Number of neuron/head-channel coordinates touching this matrix."""
    assert spec.alignment is not None, spec
    match spec.alignment.side:
        case "input":
            return spec.d_in
        case "output":
            return spec.d_out


def validate_nonlinearity_aligned_capacity(spec: SiteSpec) -> None:
    """Require enough matrix entries to give every overcomplete component nonempty support."""
    unit_count = nonlinearity_aligned_component_count(spec)
    residual_count = spec.d_in * spec.d_out // unit_count
    assert unit_count * residual_count >= spec.C, (
        f"{spec.name}: nonlinearity-aligned init supports at most {unit_count * residual_count} "
        f"nonempty components, got {spec.C}"
    )


def _gather_unit_rows(weight: Array, units: Array) -> Array:
    """Gather matrix rows and spell the surviving placement explicitly."""
    match jax.typeof(weight).sharding:
        case NamedSharding(mesh=mesh, spec=sharding):
            return weight.at[units].get(out_sharding=NamedSharding(mesh, P(None, *sharding[1:])))
        case other:
            raise AssertionError(f"an aval's sharding is always named, got {type(other).__name__}")


def nonlinearity_aligned_factors(
    weight: Array, factorization: DenseFactorization, side: ComponentSide, key: PRNGKeyArray
) -> tuple[Array, Array]:
    """Select architectural coordinates below width; partition them above width.

    At equality this is the canonical exact factorization. Below the architectural width,
    C distinct coordinates are sampled without replacement and copied whole. Above it,
    every coordinate is present and its vector across the opposite matrix dimension is
    partitioned among one or more components, so the component sum remains exactly W.
    """
    assert weight.shape == (factorization.d_out, factorization.d_in), (
        weight.shape,
        factorization,
    )
    match side:
        case "input":
            oriented_weight = weight.T
        case "output":
            oriented_weight = weight
    unit_count, residual_count = oriented_weight.shape
    assert unit_count * residual_count >= factorization.C
    oriented_weight = oriented_weight.astype(jnp.float32)

    def factors(aligned: Array, residual: Array) -> tuple[Array, Array]:
        match side:
            case "input":
                return aligned.T, residual
            case "output":
                return residual.T, aligned

    if unit_count == factorization.C:
        identity = jnp.eye(unit_count, dtype=jnp.float32)
        return factors(identity, oriented_weight)

    if unit_count > factorization.C:
        units = jax.random.permutation(key, unit_count)[: factorization.C]
        one_hot = jax.nn.one_hot(units, unit_count, dtype=jnp.float32)
        return factors(one_hot, _gather_unit_rows(oriented_weight, units))

    quotient, remainder = divmod(factorization.C, unit_count)
    unit_order = jax.random.permutation(key, unit_count)
    shard_counts = quotient + (jnp.arange(unit_count) < remainder)
    component_units = jnp.repeat(unit_order, shard_counts, total_repeat_length=factorization.C)

    unit_keys = jax.random.split(jax.random.fold_in(key, 1), unit_count)
    coordinates = jax.vmap(lambda k: jax.random.permutation(k, residual_count))(unit_keys)
    component_offsets = jnp.cumsum(shard_counts) - shard_counts
    coordinate_shards = (
        component_offsets[:, None]
        + jnp.arange(residual_count)[None, :] * shard_counts[:, None] // residual_count
    )
    ownership = jnp.zeros((factorization.C, residual_count), dtype=jnp.float32)
    ownership = ownership.at[coordinate_shards, coordinates].set(1)

    one_hot = jax.nn.one_hot(component_units, unit_count, dtype=jnp.float32)
    return factors(one_hot, _gather_unit_rows(oriented_weight, component_units) * ownership)


def _nonlinearity_aligned_site_factors(
    weight: Array, spec: SiteSpec, key: PRNGKeyArray
) -> tuple[Array, Array]:
    """Admit a dense site before constructing its nonlinearity-aligned factors."""
    assert isinstance(spec.factorization, DenseFactorization), spec
    assert spec.alignment is not None, spec
    return nonlinearity_aligned_factors(weight, spec.factorization, spec.alignment.side, key)


def nonlinearity_aligned_component_initializer(
    model: TransformerDecomposedModel, key: PRNGKeyArray
) -> ComponentStacks:
    """Initialize components along neurons or attention-head channels.

    With fewer components than architectural coordinates, sample distinct coordinates
    without replacement; the ordinary faithfulness warmup must fill the omitted weights.
    At equality, retain the canonical exact one-coordinate factorization. With surplus
    components, split randomly chosen coordinates across disjoint residual-coordinate
    shards; every component is nonempty and the component sum remains exactly the target.
    """
    keys = jax.random.split(key, len(model.sites))
    site_arrays: dict[str, tuple[Array, Array]] = {}
    for spec, site_key in zip(model.sites, keys, strict=True):
        validate_nonlinearity_aligned_capacity(spec)
        site_arrays[spec.name] = _nonlinearity_aligned_site_factors(
            model.frozen_site_weight(spec.name), spec, site_key
        )
    return component_stacks_from_site_arrays(model.sites, site_arrays)


# ----------------------------- HF weight loading -----------------------------


def hf_snapshot_dir(model_name: str) -> Path:
    """Newest local snapshot of `model_name`, from the STANDARD HF hub cache
    (`HF_HUB_CACHE`, else `~/.cache/huggingface/hub`). Multi-host callers should export
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
    frozen-weights dtype (bf16 storage — the families pass it; this module holds
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
        with cpu_staging():
            return jax.device_put(
                np.asarray(self._open[fname].get_tensor(key), dtype=self._dtype),
                jax.local_devices(backend="cpu")[0],
            )


type AttnLoader = Callable[[HFWeights, int], FrozenAttn]


def _validate_engine_sites(
    cfg: TransformerArch, sites: tuple[SiteSpec, ...], anatomy: Anatomy
) -> None:
    site_cs = tuple(SiteC(s.name, s.C) for s in sites)
    expected = family.site_specs(
        anatomy.family,
        family.canonical_site_cs(anatomy.family, site_cs),
        lambda kind, c: anatomy_site_dims(anatomy, cfg, kind).dense(c),
        lambda kind: anatomy_nonlinearity_alignment(anatomy, cfg, kind),
        cfg.n_layer,
    )
    assert sites == expected, f"sites are not the canonical specs for this config: {sites}"


def build_engine_model(
    embed: Array,
    layers: list[TransformerLayer],
    norm: Array,
    lm_head: Float[Array, "vocab d"] | TiedHead,
    inv_freq: Array,
    cfg: TransformerArch,
    sites: tuple[SiteSpec, ...],
    anatomy: Anatomy,
    output_edge: OutputEdge,
) -> TransformerDecomposedModel:
    """Assemble an engine model from the frozen full-model arrays + decomposition config
    for ANY declared anatomy. `sites` must be canonical-ordered with dims matching `cfg`."""
    _validate_engine_sites(cfg, sites, anatomy)
    return TransformerDecomposedModel(
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
        output_edge=output_edge,
    )


def build_decomposed_lm(
    embed: Array,
    layers: list[TransformerLayer],
    norm: Array,
    lm_head: Array | TiedHead,
    inv_freq: Array,
    cfg: TransformerArch,
    sites: tuple[SiteSpec, ...],
    output_edge: OutputEdge,
) -> TransformerDecomposedModel:
    """`build_engine_model` at the GLU anatomy — the HF families' entry point."""
    return build_engine_model(
        embed, layers, norm, lm_head, inv_freq, cfg, sites, GLU_ANATOMY, output_edge
    )


def load_glu_blocks(
    w: HFWeights, cfg: TransformerArch, load_attn: AttnLoader
) -> list[TransformerLayer]:
    pre = "model.layers"
    return [
        TransformerLayer(
            ln1=w.get(f"{pre}.{i}.input_layernorm.weight"),
            ln2=w.get(f"{pre}.{i}.post_attention_layernorm.weight"),
            attn=load_attn(w, i),
            mlp=GatedMLP(
                kinds=GLU_MLP_KINDS,
                Wg=w.get(f"{pre}.{i}.mlp.gate_proj.weight"),
                Wu=w.get(f"{pre}.{i}.mlp.up_proj.weight"),
                Wd=w.get(f"{pre}.{i}.mlp.down_proj.weight"),
            ),
        )
        for i in range(cfg.n_layer)
    ]


@cpu_staging()
def load_decomposed_glu_from_hf(
    model_name: str,
    cfg: HFTransformerArch,
    sites: tuple[SiteSpec, ...],
    load_attn: AttnLoader,
    inv_freq: Array,
    weights_dtype: DTypeLike,
    output_edge: OutputEdge,
) -> TransformerDecomposedModel:
    """Read and stack the frozen checkpoint as CPU JAX arrays before declared placement."""
    w = HFWeights(hf_snapshot_dir(model_name), weights_dtype)
    return build_engine_model(
        embed=w.get("model.embed_tokens.weight"),
        layers=load_glu_blocks(w, cfg, load_attn),
        norm=w.get("model.norm.weight"),
        lm_head=TiedHead() if cfg.tie_word_embeddings else w.get("lm_head.weight"),
        inv_freq=jax.device_put(inv_freq, jax.local_devices(backend="cpu")[0]),
        cfg=cfg,
        sites=sites,
        anatomy=GLU_ANATOMY,
        output_edge=output_edge,
    )
