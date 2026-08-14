"""The SHARED machinery of the vendored HF GLU-transformer decomposition targets: site
grammar, frozen modules, the scan/masked-forward engine, and HF safetensors loading. The
model FAMILIES live in their own files — `llama8b.py`, `qwen3_8b.py` —
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

Sharding (the production HSDP memory story): the frozen target declares its own
placement via `.shardings(mesh)` — FSDP-sharded on `fsdp` (the `d` dim of every
per-layer weight; embed / lm_head / norm / inv_freq replicate), applied by the engine's
`sharding.place_target`, gathered one layer at a time in the scan on NVLink. The bf16
COMPUTE weights are re-pinned to `fsdp`-only ONCE per step in
`_reconstruct_compute_weights` (the cross-`replicate` gather, off the per-layer hot
path). V/U, CI-fn, and source placement are the engine's concern (`placement`,
`init_placed`), not this module's.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import abc
from typing import Any, Literal, Protocol, Self, cast, get_args, override

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int
from safetensors import safe_open

from param_decomp.core import family
from param_decomp.core.components import (
    ComponentStacks,
    SiteC,
    SiteDims,
    SiteSpec,
    site_out,
)
from param_decomp.core.family import ArchFamily
from param_decomp.core.model import (
    EMPTY_CAPTURE_KEYS,
    CaptureKeys,
    ForwardResult,
    Masking,
    MaterializedMasking,
    StochasticMasking,
)
from param_decomp.core.sharding import assert_divisible
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
    rope_theta: float
    rms_norm_eps: float
    max_position_embeddings: int

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def n_rep(self) -> int:
        return self.n_head // self.n_kv_head


def default_inv_freq(head_dim: int, rope_theta: float) -> Float[Array, " hd2"]:
    """Plain (unscaled) RoPE inverse frequencies — HF's `rope_type: default`."""
    return 1.0 / (rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))


# GLU = SwiGLU MLP (llama8b, qwen3_8b). The family's matrix vocabulary — the authored
# c-spec keys (lab-side) are typed by it, so a c-spec key and a target matrix cannot drift.
GluMatrix = Literal["q", "k", "v", "o", "gate", "up", "down"]

KIND_ORDER: tuple[str, ...] = get_args(GluMatrix)
"""Within-layer canonical site order = computation order, DERIVED from the `GluMatrix`
vocabulary. The canonical site order (`glu_site_specs`) is layer-ascending, then this."""
ATTN_KINDS = ("q", "k", "v", "o")
MLP_KINDS = ("gate", "up", "down")
assert KIND_ORDER == ATTN_KINDS + MLP_KINDS, KIND_ORDER


def _hidden_acts_reconstruction_dependencies(point: TransformerPoint) -> frozenset[str]:
    """Same-block decomposed kinds whose output can influence ``point``."""
    match point:
        case ResidualBoundary():
            return frozenset()
        case PostAttentionResidual():
            return frozenset(ATTN_KINDS)
        case BlockTap(name=name):
            match name:
                case "attn_in":
                    return frozenset()
                case "attn_out":
                    return frozenset(("q", "k", "v"))
                case "mlp_in":
                    return frozenset(ATTN_KINDS)
                case "mlp_hidden":
                    return frozenset((*ATTN_KINDS, "gate", "up"))
                case _:
                    raise AssertionError(name)
        case SiteOutput(name=name):
            _block, kind = parse_site_name(name)
            match kind:
                case "q" | "k" | "v":
                    return frozenset((kind,))
                case "o":
                    return frozenset(ATTN_KINDS)
                case "gate":
                    return frozenset((*ATTN_KINDS, "gate"))
                case "up":
                    return frozenset((*ATTN_KINDS, "up"))
                case "down":
                    return frozenset(KIND_ORDER)
                case _:
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


def site_dims(cfg: GLUArch, kind: str) -> SiteDims:
    """Dimensions of one per-layer matrix in right-mult orientation."""
    d, di = cfg.n_embd, cfg.n_intermediate
    qd = cfg.n_head * cfg.head_dim
    kvd = cfg.n_kv_head * cfg.head_dim
    match kind:
        case "q":
            return SiteDims(d_in=d, d_out=qd)
        case "k" | "v":
            return SiteDims(d_in=d, d_out=kvd)
        case "o":
            return SiteDims(d_in=qd, d_out=d)
        case "gate" | "up":
            return SiteDims(d_in=d, d_out=di)
        case "down":
            return SiteDims(d_in=di, d_out=d)
        case _:
            raise AssertionError(f"unknown kind {kind!r}")


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


def glu_site_specs(cfg: GLUArch, site_cs: tuple[SiteC, ...]) -> tuple[SiteSpec, ...]:
    return family.site_specs(FAMILY, site_cs, lambda kind: site_dims(cfg, kind), cfg.n_layer)


# ----------------------------- frozen layers -----------------------------


class FrozenAttn(eqx.Module):
    """Plain GQA attention (Llama, LlamaSimpleMLP). A family with extra pre-RoPE math
    subclasses and overrides `_prep_qk` (and `shardings` for any extra fields) — e.g.
    `qwen3_8b.Qwen3FrozenAttn`'s per-head QK-norm."""

    wq: Float[Array, "qd d"]
    wk: Float[Array, "kvd d"]
    wv: Float[Array, "kvd d"]
    wo: Float[Array, "d qd"]
    n_head: int = eqx.field(static=True)
    n_kv_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    n_rep: int = eqx.field(static=True)

    def _prep_qk(
        self, q: Float[Array, "b t h hd"], k: Float[Array, "b t kvh hd"]
    ) -> tuple[Array, Array]:
        """Family hook between the head reshape and RoPE; identity for plain attention."""
        return q, k

    def shardings(self, mesh: Mesh) -> "FrozenAttn":
        """Stacked (leading `n_layer`, UNSHARDED — the scan axis) FSDP on `fsdp`: the `d` dim
        shards on `fsdp` (gathered per layer in the scan, on NVLink); the HEAD dim stays
        REPLICATED. `core` runs batch-parallel attention (q/k/v constrained batch over the
        full mesh, heads replicated — that identical spec is what cuDNN's partitioner
        requires), so the projections must come out heads-replicated. Attention weights are
        small, so FSDP-on-`fsdp` sharding is plenty."""
        in_fsdp = NamedSharding(mesh, P(None, None, "fsdp"))  # qkv [nc, head(repl), d on fsdp]
        out_fsdp = NamedSharding(mesh, P(None, "fsdp", None))  # wo [nc, d on fsdp, head(repl)]
        for w in (self.wq, self.wk, self.wv):
            assert_divisible(w.shape[2], mesh, "fsdp", "FrozenAttn qkv in (d)")
        assert_divisible(self.wo.shape[1], mesh, "fsdp", "FrozenAttn out-proj out (d)")
        return eqx.tree_at(
            lambda a: (a.wq, a.wk, a.wv, a.wo), self, (in_fsdp, in_fsdp, in_fsdp, out_fsdp)
        )

    def core(
        self,
        q_flat: Float[Array, "b t qd"],
        k_flat: Float[Array, "b t kvd"],
        v_flat: Float[Array, "b t kvd"],
        inv_freq: Array,
    ) -> Float[Array, "b t qd"]:
        """RoPE + causal SDPA between the q/k/v projections and the o projection —
        the seam the decomposed q/k/v site outputs feed into."""
        b, t, _ = q_flat.shape
        assert q_flat.shape[-1] == self.n_head * self.head_dim, q_flat.shape
        assert k_flat.shape[-1] == self.n_kv_head * self.head_dim, k_flat.shape
        assert v_flat.shape[-1] == self.n_kv_head * self.head_dim, v_flat.shape
        q = q_flat.reshape(b, t, self.n_head, self.head_dim)
        k = k_flat.reshape(b, t, self.n_kv_head, self.head_dim)
        q, k = self._prep_qk(q, k)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v_flat.reshape(b, t, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        cos, sin = rope_cos_sin(inv_freq, t, q_flat.dtype)
        q, k = apply_rope(q, k, cos, sin)
        # Native GQA: do NOT repeat_kv — `dot_product_attention` handles the q-heads:kv-heads
        # grouping internally. Repeating k/v to n_head and THEN sharding makes the SPMD
        # partitioner derive the repeated-k/v layout from the small n_kv_head source,
        # inconsistent with q -> cuDNN "Query, key and value should have same sharding" (forward
        # AND its rematerialized backward). Real GQA keeps q (n_head) and k/v (n_kv_head) as
        # independent head-parallel tensors with the SAME spec (different head COUNTS is fine).
        # cuDNN flash attention's custom partitioner requires q/k/v IDENTICALLY sharded.
        # Pure HSDP: pin all three batch-parallel over the FULL mesh, HEADS replicated
        # (`P(('replicate','fsdp'), None, None, None)` for `[b, heads, t, hd]`). The identical
        # q/k/v spec is exactly what cuDNN's flash partitioner demands; heads-replicated keeps
        # q (n_head) and k/v (n_kv_head) consistently sharded (no head TP that would split them
        # to different per-rank counts). The q/k/v projection outputs are `d_out`-replicated
        # (U FSDP's the C side, not d_out's head); this constraint pins the batch right here.
        # Guarded so it's a no-op off-mesh (CPU tests / single device); `run.py` sets the
        # global mesh.
        if not jax.sharding.get_abstract_mesh().empty:
            qkv_spec = jax.sharding.PartitionSpec(
                ("replicate", "fsdp"), None, None, None
            )  # batch-parallel: q/k/v IDENTICAL spec -> cuDNN happy; flash = no score materialization
            q, k, v = (jax.lax.with_sharding_constraint(a, qkv_spec) for a in (q, k, v))
        return causal_sdpa(q, k, v).transpose(0, 2, 1, 3).reshape(b, t, self.n_head * self.head_dim)

    def __call__(self, x: Float[Array, "b t d"], inv_freq: Array) -> Array:
        return self.core(x @ self.wq.T, x @ self.wk.T, x @ self.wv.T, inv_freq) @ self.wo.T

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


class GLULayer(eqx.Module):
    """One layer's frozen weights — norms, attention, MLP. Decomposed sites read
    their frozen target W from here at forward time; layers without sites run the
    plain frozen block from the same fields. Weights pass as a runtime arg — never
    baked into the HLO as a multi-GB constant."""

    ln1: Float[Array, " d"]
    ln2: Float[Array, " d"]
    attn: FrozenAttn
    Wg: Float[Array, "di d"]
    Wu: Float[Array, "di d"]
    Wd: Float[Array, "d di"]

    def shardings(self, mesh: Mesh) -> "GLULayer":
        """Stacked FSDP on `fsdp` (no TP): every MLP weight shards its `d`-dim on `fsdp`,
        the intermediate dim stays replicated; gathered back per layer inside the scan.
        Norms replicate; attn delegates to `FrozenAttn.shardings`."""
        in_fsdp = NamedSharding(mesh, P(None, None, "fsdp"))  # Wg/Wu [nc, di(repl), d on fsdp]
        out_fsdp = NamedSharding(mesh, P(None, "fsdp", None))  # Wd [nc, d on fsdp, di(repl)]
        repl = NamedSharding(mesh, P())
        assert_divisible(self.Wg.shape[2], mesh, "fsdp", "Wg in (d)")
        assert_divisible(self.Wd.shape[1], mesh, "fsdp", "Wd out (d)")
        return eqx.tree_at(
            lambda layer: (layer.ln1, layer.ln2, layer.attn, layer.Wg, layer.Wu, layer.Wd),
            self,
            (repl, repl, self.attn.shardings(mesh), in_fsdp, in_fsdp, out_fsdp),
        )


def _frozen_site_weight(layer: GLULayer, kind: str) -> Array:
    match kind:
        case "q":
            return layer.attn.wq
        case "k":
            return layer.attn.wk
        case "v":
            return layer.attn.wv
        case "o":
            return layer.attn.wo
        case "gate":
            return layer.Wg
        case "up":
            return layer.Wu
        case "down":
            return layer.Wd
        case _:
            raise AssertionError(f"unknown kind {kind!r}")


# ----------------------------- forwards -----------------------------


def _clean_mlp_out(layer: GLULayer, mlp_in: Array) -> Array:
    """Frozen target MLP — exactly `W` applied, not the `V@U + (W−V@U)` identity, so
    non-live sites carry no V/U gradient and no decomposition rounding (SPEC S2/S3)."""
    return (jax.nn.silu(mlp_in @ layer.Wg.T) * (mlp_in @ layer.Wu.T)) @ layer.Wd.T


def _clean_block(layer: GLULayer, x: Array, inv_freq: Array, eps: float) -> Array:
    x = x + layer.attn(rms_norm(x, layer.ln1, eps), inv_freq)
    return x + _clean_mlp_out(layer, rms_norm(x, layer.ln2, eps))


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
    GATE_UP_INPUT = "gate_up_input"
    GATE_OUTPUT = "gate_output"
    UP_OUTPUT = "up_output"
    DOWN_INPUT = "down_input"
    DOWN_OUTPUT = "down_output"
    RESIDUAL_OUT = "residual_out"


@dataclass(frozen=True, kw_only=True)
class _GLUCaptureSource:
    block: int
    tap: _GLUTap


GLUCaptureSources = tuple[_GLUCaptureSource, ...]


@dataclass(frozen=True, kw_only=True)
class _GLUBlockActivations:
    """The target's capturable activations for one block; never returned wholesale."""

    residual_in: Array
    qkv_input: Array
    q_output: Array
    k_output: Array
    v_output: Array
    attention_output: Array
    o_output: Array
    post_attention_residual: Array
    gate_up_input: Array
    gate_output: Array
    up_output: Array
    down_input: Array
    down_output: Array
    residual_out: Array


_UNUSED_CAPTURE_SLOT = -1


@dataclass(frozen=True, kw_only=True)
class _ScanCaptureLayout:
    """One exact-size scan-carry buffer for each requested value kind."""

    slot_by_block_per_tap: tuple[tuple[_GLUTap, tuple[int, ...]], ...]
    sources: tuple[_GLUCaptureSource, ...]


def _capture_source_for_point(point: TransformerPoint) -> _GLUCaptureSource:
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
                "mlp_in": _GLUTap.GATE_UP_INPUT,
                "mlp_hidden": _GLUTap.DOWN_INPUT,
            }
            return _GLUCaptureSource(block=block, tap=value_by_name[name])
        case SiteOutput(name=name, block=block):
            _layer, kind = parse_site_name(name)
            match kind:
                case "q":
                    value = _GLUTap.Q_OUTPUT
                case "k":
                    value = _GLUTap.K_OUTPUT
                case "v":
                    value = _GLUTap.V_OUTPUT
                case "o":
                    value = _GLUTap.O_OUTPUT
                case "gate":
                    value = _GLUTap.GATE_OUTPUT
                case "up":
                    value = _GLUTap.UP_OUTPUT
                case "down":
                    value = _GLUTap.DOWN_OUTPUT
                case _:
                    raise AssertionError(f"unknown GLU site kind {kind!r}")
            return _GLUCaptureSource(block=block, tap=value)


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
        case _GLUTap.GATE_UP_INPUT:
            return block_activations.gate_up_input
        case _GLUTap.GATE_OUTPUT:
            return block_activations.gate_output
        case _GLUTap.UP_OUTPUT:
            return block_activations.up_output
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
    return {
        tap.value: jnp.zeros(
            (
                sum(slot >= 0 for slot in slots),
                *residual.shape[:-1],
                width_of(tap),
            ),
            residual.dtype,
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


def _per_kind_dims(components: ComponentStacks) -> dict[str, tuple[int, int, int]]:
    """Per decomposed KIND, the `(d_in, C, d_out)` shared across its layers — asserting
    uniformity, the precondition for the layer-`lax.scan` masked forward (it stacks each
    kind across layers, so every layer's matrix of that kind must be the same shape)."""
    kind_dims: dict[str, tuple[int, int, int]] = {}
    for name, (d_in, d_out, c), _slot in components.site_slots:
        kind = parse_site_name(name)[1]
        dims = (d_in, c, d_out)
        assert kind_dims.setdefault(kind, dims) == dims, (
            f"per-kind dims must be uniform across layers for the scan masked forward: "
            f"{kind} {dims} != {kind_dims[kind]}"
        )
    return kind_dims


def _stack_per_kind_vu(components: ComponentStacks, n_layers: int) -> dict[str, dict[str, Array]]:
    """Per decomposed KIND, the layer-stacked `(V, U)` arrays — the MASK-INDEPENDENT part of
    the scan inputs (a leading layer axis, one homogeneous body across layers). Mask/live/
    delta/route are attached per-forward by `_attach_per_kind_masks`; the V/U stack +
    `_reconstruct_compute_weights` (the ÷N→÷fsdp cross-node gather) are the same for EVERY
    forward in a step, so they are built ONCE via `prepare_compute_weights` and shared.
    Per-kind dims (d_in, C, d_out) must be uniform across layers (asserted in `_per_kind_dims`)."""
    kind_dims = _per_kind_dims(components)
    slot_of = {name: (shape, slot) for name, shape, slot in components.site_slots}
    vu_dt = next(iter(components.stacks.values()))[0].dtype
    per_kind: dict[str, dict[str, Array]] = {}
    for kind, (d_in, C, d_out) in kind_dims.items():
        names = [site_name(layer, kind) for layer in range(n_layers)]
        present = [slot_of[n] for n in names if n in slot_of]
        shapes = {shape for shape, _ in present}
        slots = [slot for _, slot in present]
        stride = slots[1] - slots[0] if len(slots) > 1 else 1
        arithmetic = (
            len(names) == len(present)
            and len(shapes) == 1
            and stride >= 1
            and slots == list(range(slots[0], slots[0] + stride * len(slots), stride))
        )
        if arithmetic:
            # Full-kind fast path: one shape group, slots an arithmetic progression — the
            # per-kind scan stack is a STATIC (possibly strided) SLICE of the resting stack
            # (no restack, no zero-fill; the owner->compute reshard downstream is the only
            # data movement). Kinds sharing a shape group interleave layer-major (gate/up,
            # and q/o when qd == d), so their slots stride by the kinds-per-group count.
            (shape,) = shapes
            Vs_all, Us_all = components.stacks[shape]
            lo, hi = slots[0], slots[-1] + 1
            Vs = jax.lax.slice(Vs_all, (lo, 0, 0), (hi, *Vs_all.shape[1:]), (stride, 1, 1))
            Us = jax.lax.slice(Us_all, (lo, 0, 0), (hi, *Us_all.shape[1:]), (stride, 1, 1))
        else:
            Vs = jnp.stack(
                [
                    components.site(n).V if n in slot_of else jnp.zeros((d_in, C), vu_dt)
                    for n in names
                ]
            )
            Us = jnp.stack(
                [
                    components.site(n).U if n in slot_of else jnp.zeros((C, d_out), vu_dt)
                    for n in names
                ]
            )
        per_kind[kind] = {"V": Vs, "U": Us}
    return per_kind


def _attach_per_kind_masks(
    prepared_weights: dict[str, dict[str, Array]],
    n_layers: int,
    leading: tuple[int, ...],
    component_masks: dict[str, Array],
    weight_delta_masks: dict[str, Array] | None,
    routes: dict[str, Array] | None,
    live_sites: frozenset[str],
) -> dict[str, dict[str, Array]]:
    """Attach the per-forward `(live, mask[, delta][, route])` stacks to the shared, already
    stacked + ÷fsdp-reconstructed `prepared_weights` per-kind `(V, U)` weights. The
    homogeneous layer stacks include dummy mask/delta/route values outside the live segment;
    segmentation excludes those layers from this masked block. The explicit mappings exist
    only for live sites (recon builds them per chunk)."""
    # Dummy mask/delta/route shapes match the REAL entries (the source scope sets the leading
    # shape: `sc` broadcasts over batch as `(1, T)`, not the full `(B, T)`).
    a_mask = next(iter(component_masks.values())) if component_masks else None
    mask_lead = a_mask.shape[:-1] if a_mask is not None else leading
    a_delta = (
        next(iter(weight_delta_masks.values()), None) if weight_delta_masks is not None else None
    )
    a_route = next(iter(routes.values())) if (routes and len(routes)) else None

    per_kind: dict[str, dict[str, Array]] = {}
    for kind, vu_entry in prepared_weights.items():
        C = vu_entry["V"].shape[-1]
        mask_dt = a_mask.dtype if a_mask is not None else vu_entry["V"].dtype
        names = [site_name(layer, kind) for layer in range(n_layers)]
        live_flags = jnp.array([name in live_sites for name in names])
        masks_k = jnp.stack(
            [
                component_masks[name] if name in live_sites else jnp.ones((*mask_lead, C), mask_dt)
                for name in names
            ]
        )
        entry: dict[str, Array] = {**vu_entry, "live": live_flags, "mask": masks_k}
        if weight_delta_masks is not None:
            delta_shape = a_delta.shape if a_delta is not None else leading
            delta_dtype = a_delta.dtype if a_delta is not None else mask_dt
            entry["delta"] = jnp.stack(
                [
                    weight_delta_masks[name]
                    if name in live_sites
                    else jnp.zeros(delta_shape, delta_dtype)
                    for name in names
                ]
            )
        if routes is not None:
            r_shape = a_route.shape if a_route is not None else leading
            r_dt = a_route.dtype if a_route is not None else jnp.bool_
            entry["route"] = jnp.stack(
                [routes[name] if name in live_sites else jnp.zeros(r_shape, r_dt) for name in names]
            )
        per_kind[kind] = entry
    return per_kind


def _stack_ci_per_kind(ci_lower: dict[str, Array], n_layers: int) -> dict[str, Array]:
    """Stack the per-site CI envelope into per-kind `[n_layer, *leading, C]` — built ONCE per
    step and SHARED across every stochastic recon forward (the CI envelope is identical for
    all). Mirrors `_stack_per_kind_vu`: the shared stack replaces N per-forward mask stacks."""
    kinds: dict[str, Array] = {}
    sample_by_kind: dict[str, Array] = {}
    for name, v in ci_lower.items():
        sample_by_kind.setdefault(parse_site_name(name)[1], v)
    for kind, sample in sample_by_kind.items():
        names = [site_name(layer, kind) for layer in range(n_layers)]
        kinds[kind] = jnp.stack(
            [ci_lower[name] if name in ci_lower else jnp.zeros_like(sample) for name in names]
        )
    return kinds


def _attach_per_kind_stochastic(
    prepared_weights: dict[str, dict[str, Array]],
    n_layers: int,
    leading: tuple[int, ...],
    ci_stacked: dict[str, Array],
    draw_key: Array,
    routes: dict[str, Array] | None,
    live_sites: frozenset[str],
) -> dict[str, dict[str, Array]]:
    """Stochastic recon: attach the SHARED per-kind `ci` stack + per-(layer,kind) RNG keys
    instead of pre-built mask/delta stacks. `site_output_and_component_activation` draws `source = uniform(key)` and
    builds `mask = ci + (1−ci)·source` INSIDE the checkpointed block, so the per-forward mask
    is recomputed in the backward (faithful by checkpoint determinism — same key fwd+bwd) and
    never held. Only the (tiny) keys + live-flags are per-forward; the `[n_layer,*,C]` ci stack
    is shared, so N forwards' mask stacks collapse to one ci stack (the memory win)."""
    src_base, delta_base = jax.random.split(draw_key)
    a_route = next(iter(routes.values())) if (routes and len(routes)) else None
    per_kind: dict[str, dict[str, Array]] = {}
    for kind, vu_entry in prepared_weights.items():
        names = [site_name(layer, kind) for layer in range(n_layers)]
        live_flags = jnp.array([name in live_sites for name in names])
        kind_idx = KIND_ORDER.index(kind)
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
            **vu_entry, "live": live_flags, "ci": ci_stacked[kind],
            "src_key": src_keys, "delta_key": delta_keys,
        }  # fmt: skip
        if routes is not None:
            r_shape = a_route.shape if a_route is not None else leading
            r_dt = a_route.dtype if a_route is not None else jnp.bool_
            entry["route"] = jnp.stack(
                [routes[name] if name in live_sites else jnp.zeros(r_shape, r_dt) for name in names]
            )
        per_kind[kind] = entry
    return per_kind


def _reconstruct_compute_weights(
    per_kind: dict[str, dict[str, Array]],
) -> dict[str, dict[str, Array]]:
    """The ZeRO-1 weight reconstruction (pure-HSDP backup layout). The stacked
    `[n_layer, d_in, C]` / `[n_layer, C, d_out]` compute weights arrive with their FSDP dim
    in the persistence layout the run's placement rules chose (e.g. `owner`:
    stack ÷replicate, d ÷fsdp — see `ComponentStacks.shardings`). Reconstruct
    them to the `fsdp`-sharded (÷fsdp) COMPUTE layout here — BEFORE the layer scan — so:

      * the cross-`replicate` gather runs ONCE per step in ENTRY (off the hot path),
        landing a SMALL ÷fsdp-resident weight stack (`[n_layer, d_in/fsdp, C]`), NOT the full
        `[n_layer, d_in, C]` model;
      * the per-layer scan body then gathers ONE layer's `fsdp` shard to full d_in
        transiently (intra-node NVLink), freed each iteration — never a full-model resident.

    Cast to bf16 HERE (not f32) so the ÷fsdp-resident compute stack is half-size and XLA
    can't keep an f32 full copy alive. The fp32 masters + Adam stay ÷N (untouched — this is
    a separate read-only compute view). The leading `n_layer` axis (the scan `xs`) stays
    unsharded. No-op off-mesh (CPU / single device); `run.py` sets the global mesh."""
    if jax.sharding.get_abstract_mesh().empty:
        return per_kind
    v_spec = P(None, "fsdp", "tp")  # [n_layer, d_in ÷fsdp, C ÷tp] — d gathered/step, C stays ÷tp
    u_spec = P(None, "tp", "fsdp")  # [n_layer, C ÷tp, d_out ÷fsdp]
    out: dict[str, dict[str, Array]] = {}
    for kind, entry in per_kind.items():
        pinned = dict(entry)
        # optimization_barrier forces the bf16 cast to materialize BEFORE the ÷N→÷fsdp gather,
        # so the collective moves the compute dtype — XLA otherwise sinks the convert past the
        # all-gather and gathers the f32 master (2x the comm; the convert gathers in the HLO).
        pinned["V"] = jax.lax.with_sharding_constraint(
            jax.lax.optimization_barrier(entry["V"].astype(jnp.bfloat16)), v_spec
        )
        pinned["U"] = jax.lax.with_sharding_constraint(
            jax.lax.optimization_barrier(entry["U"].astype(jnp.bfloat16)), u_spec
        )
        out[kind] = pinned
    return out


GATHERED_WEIGHTS_CHECKPOINT_NAME = "gathered_compute_weights"
"""`checkpoint_name` tag on every full-layout (÷fsdp→full gathered) weight formed inside
the checkpointed scan body — the masked-forward remat policy excludes this name from the
saved set, so the backward RE-GATHERS the weights (cheap NVLink collectives) instead of
holding one full gathered copy per recon forward across the fwd→bwd liverange."""


def _gather_full_weight(w: Array, spec: P) -> Array:
    """Form the full-layout compute weight INSIDE the checkpointed scan body: the explicit
    per-layer sharding constraint anchors the ÷fsdp→full all-gather in the loop body (without
    it GSPMD is free to hoist the gather of the WHOLE per-chunk stack out of the `lax.scan`
    while-loop, where it stays resident from each forward until its backward), and the
    `checkpoint_name` tag keeps the gathered value out of every remat save set. Constraint is
    a no-op off-mesh (CPU tests / single device); the tag is transparent everywhere."""
    if not jax.sharding.get_abstract_mesh().empty:
        w = jax.lax.with_sharding_constraint(w, spec)
    return checkpoint_name(w, GATHERED_WEIGHTS_CHECKPOINT_NAME)


class _GLUTransformerCore(eqx.Module, abc.ABC):
    """The input-agnostic GLU-transformer machinery: frozen blocks, norm, unembed, and
    every forward entered at a residual (`*_from_residual`). Concrete models own the
    input adapter: `GLUDecomposedModel` embeds tokens; `ResidualGLUDecomposedModel`
    consumes residual activations directly (a depth-suffix run against a prefix mapper).

    The TRAINABLE V/U (`vu: ComponentStacks`) is passed to the forward methods
    explicitly: separate lifecycle (own optimizer + checkpoint, C-sharded while these
    weights replicate), so it is NOT a field here. Blocks with no decomposed site run
    the plain frozen path — a subset decomposition just leaves the rest frozen.

    `sites` / `has_position_axis` are static config."""

    stacked: GLULayer  # the per-layer weights stacked on a leading layer axis (the scan
    # `xs`), stored pre-stacked: a saved jit input, never re-stacked inside a forward.
    n_layer: int = eqx.field(static=True)
    norm: Float[Array, " d"]
    lm_head: Float[Array, "vocab d"]
    inv_freq: Float[Array, " hd2"]
    sites: tuple[SiteSpec, ...] = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sites)

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
        layer = parse_site_name(q_site)[0]
        attn = jax.tree.map(lambda a, idx=layer: a[idx], self.stacked.attn)
        return attn.pattern(q_flat, k_flat, self.inv_freq)

    def shardings(self, mesh: Mesh) -> Self:
        """FSDP-on-`fsdp` the per-layer weights (`stacked.shardings` — `d` on `fsdp`,
        head/intermediate replicated; the ~14 GB layer bulk shards `/fsdp`, gathered per layer
        inside the scan, on NVLink). lm_head / norm / inv_freq REPLICATE — the head is
        small and vocab-parallel logits aren't worth the complexity. (The old
        all-replicate justification — "the target is small vs activations" — is stale: at
        the full 32-layer model the replicated target + its backward/remat copies dominate the
        step's peak, which is what this shards away.)"""
        repl = NamedSharding(mesh, P())
        return eqx.tree_at(
            lambda m: (m.norm, m.lm_head, m.inv_freq, m.stacked),
            self,
            (repl, repl, repl, self.stacked.shardings(mesh)),
        )

    @staticmethod
    def recon_loss_fn(masked_output: Array, clean_output: Array) -> Array:
        return kl_per_position(masked_output, clean_output)

    @abc.abstractmethod
    def clean_forward(
        self, inputs: Any, capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS
    ) -> ForwardResult:
        """Each concrete model owns its input adapter (tokens vs residual activations);
        the machinery here is entered at `forward_from_residual`."""

    def _capture_grammar(self) -> TransformerTapGrammar:
        def site_dimensions(name: str) -> tuple[int, int]:
            _layer, kind = parse_site_name(name)
            match kind:
                case "q":
                    weight = self.stacked.attn.wq
                case "k":
                    weight = self.stacked.attn.wk
                case "v":
                    weight = self.stacked.attn.wv
                case "o":
                    weight = self.stacked.attn.wo
                case "gate":
                    weight = self.stacked.Wg
                case "up":
                    weight = self.stacked.Wu
                case "down":
                    weight = self.stacked.Wd
                case _:
                    raise AssertionError(f"unknown GLU site kind {kind!r}")
            return weight.shape[2], weight.shape[1]

        return TransformerTapGrammar(
            family=FAMILY,
            n_layer=self.n_layer,
            d_resid=self.norm.shape[0],
            d_attention_output=site_dimensions(FAMILY.name_of(0, "o"))[0],
            d_mlp_hidden=site_dimensions(FAMILY.name_of(0, "down"))[0],
            d_out_of=lambda name: site_dimensions(name)[1],
        )

    def site_output_keys(self, sites: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(site_output_tap_key(site) for site in sites)

    def assert_hidden_acts_reconstruction_points(self, keys: tuple[str, ...]) -> None:
        self._capture_grammar().assert_hidden_acts_reconstruction_points(
            keys, self.site_names, _hidden_acts_reconstruction_dependencies
        )

    def _block_activations(self, residual_in: Array, layer: GLULayer) -> _GLUBlockActivations:
        attn = layer.attn
        h1 = rms_norm(residual_in, layer.ln1, self.eps)
        q = h1 @ attn.wq.T
        k = h1 @ attn.wk.T
        v = h1 @ attn.wv.T
        attention_output = attn.core(q, k, v, self.inv_freq)
        o = attention_output @ attn.wo.T
        post_attention = residual_in + o
        h2 = rms_norm(post_attention, layer.ln2, self.eps)
        gate = h2 @ layer.Wg.T
        up = h2 @ layer.Wu.T
        down_input = jax.nn.silu(gate) * up
        down = down_input @ layer.Wd.T
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
            gate_up_input=h2,
            gate_output=gate,
            up_output=up,
            down_input=down_input,
            down_output=down,
            residual_out=residual_out,
        )

    def _value_width(self, value: _GLUTap) -> int:
        d = self.norm.shape[0]
        match value:
            case (
                _GLUTap.RESIDUAL_IN
                | _GLUTap.QKV_INPUT
                | _GLUTap.O_OUTPUT
                | _GLUTap.POST_ATTENTION_RESIDUAL
                | _GLUTap.GATE_UP_INPUT
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
                return self.stacked.Wg.shape[1]
            case _GLUTap.UP_OUTPUT:
                return self.stacked.Wu.shape[1]
            case _GLUTap.DOWN_INPUT:
                return self.stacked.Wd.shape[2]

    def _clean_output_from_residual(self, residual: Float[Array, "b t d"]) -> Array:
        """Untouched graph used when no captures are requested."""

        def block(residual: Array, layer: GLULayer) -> tuple[Array, None]:
            return _clean_block(layer, residual, self.inv_freq, self.eps), None

        residual, _ = jax.lax.scan(block, residual, self.stacked)
        residual = rms_norm(residual, self.norm, self.eps)
        return residual @ self.lm_head.T

    def forward_from_residual(
        self, residual: Float[Array, "b t d"], capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS
    ) -> ForwardResult:
        if not capture_keys:
            return ForwardResult.from_producer(
                output=self._clean_output_from_residual(residual),
                capture_keys=(),
                capture_values=(),
            )
        ordered_capture_keys = tuple(sorted(capture_keys))
        capture_sources = self._capture_grammar().resolve(
            ordered_capture_keys, _capture_source_for_point
        )

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
            block_activations = self._block_activations(x, layer)
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
            output=residual @ self.lm_head.T,
            capture_keys=ordered_capture_keys,
            capture_values=_captures_in_request_order(capture_sources, captured_by_source),
        )

    def _run_masked_forward(
        self,
        prepared_weights: dict[str, dict[str, Array]],
        residual: Float[Array, "b t d"],
        masking: Masking,
        remat: bool,
        capture_keys: tuple[str, ...],
        *,
        collect_component_activations: bool,
    ) -> tuple[ForwardResult, dict[str, Array]]:
        """One segmented masked forward for output, captures and optional ``x@V`` diagnostics.

        Empty capture keys keep the compact frozen blocks; non-empty keys allocate only
        their exact-size capture slots.
        """
        capture_sources = (
            self._capture_grammar().resolve(capture_keys, _capture_source_for_point)
            if capture_keys
            else ()
        )
        leading = residual.shape[:-1]
        match masking:
            case StochasticMasking(
                ci_stacked=ci_stacked,
                draw_key=draw_key,
                live_sites=live_site_names,
                routes=routes,
            ):
                live_sites = frozenset(live_site_names)
                uses_weight_deltas = True
                per_kind = _attach_per_kind_stochastic(
                    prepared_weights,
                    self.n_layer,
                    leading,
                    ci_stacked,
                    draw_key,
                    routes,
                    live_sites,
                )
            case MaterializedMasking(
                component_masks=component_masks,
                weight_delta_masks=weight_delta_masks,
                routes=routes,
            ):
                live_site_names = masking.live_sites
                live_sites = frozenset(live_site_names)
                uses_weight_deltas = weight_delta_masks is not None
                per_kind = _attach_per_kind_masks(
                    prepared_weights,
                    self.n_layer,
                    leading,
                    component_masks,
                    weight_delta_masks,
                    routes,
                    live_sites,
                )
        decomposed_kinds = frozenset(per_kind)

        def layer_is_live(layer: int) -> bool:
            flags = {site_name(layer, kind) in live_sites for kind in decomposed_kinds}
            assert len(flags) == 1, (
                f"layer {layer} is partially live ({flags}); the segmented masked forward "
                f"requires layer-aligned chunks across {len(decomposed_kinds)} kinds"
            )
            return flags.pop()

        live_layers = [layer for layer in range(self.n_layer) if layer_is_live(layer)]
        if live_layers:
            first_live, last_live = live_layers[0], live_layers[-1] + 1
            assert live_layers == list(range(first_live, last_live)), (
                f"live layers must be contiguous, got {live_layers}"
            )
        else:
            first_live = last_live = 0

        def decomposed_site_output(
            site_input: Array,
            frozen_weight: Array,
            decomposition_inputs: dict[str, Array],
        ) -> Array:
            v = _gather_full_weight(decomposition_inputs["V"], P(None, "tp"))
            u = _gather_full_weight(decomposition_inputs["U"], P("tp", None))
            if "ci" in decomposition_inputs:
                ci_lower = decomposition_inputs["ci"]
                random_source = jax.random.uniform(
                    decomposition_inputs["src_key"], ci_lower.shape, dtype=ci_lower.dtype
                )
                component_mask = ci_lower + (1.0 - ci_lower) * random_source
                weight_delta_mask = (
                    jax.random.uniform(
                        decomposition_inputs["delta_key"],
                        ci_lower.shape[:-1],
                        dtype=ci_lower.dtype,
                    )
                    if uses_weight_deltas
                    else None
                )
                return site_out(
                    site_input,
                    v,
                    u,
                    frozen_weight,
                    component_mask,
                    weight_delta_mask,
                    decomposition_inputs.get("route"),
                )
            return site_out(
                site_input,
                v,
                u,
                frozen_weight,
                decomposition_inputs["mask"],
                decomposition_inputs.get("delta"),
                decomposition_inputs.get("route"),
            )

        def site_output_and_component_activation(
            site_input: Array,
            kind: str,
            frozen_weight: Array,
            per_kind_inputs: dict[str, dict[str, Array]],
        ) -> tuple[Array, Array | None]:
            if kind not in decomposed_kinds:
                return site_input @ frozen_weight.T, None
            decomposition_inputs = per_kind_inputs[kind]
            site_output = decomposed_site_output(site_input, frozen_weight, decomposition_inputs)
            component_activation = (
                site_input @ decomposition_inputs["V"] if collect_component_activations else None
            )
            return site_output, component_activation

        def compute_live_block_activations(
            residual_in: Array,
            layer: GLULayer,
            per_kind_inputs: dict[str, dict[str, Array]],
        ) -> tuple[_GLUBlockActivations, dict[str, Array] | None]:
            attn = layer.attn
            h1 = rms_norm(residual_in, layer.ln1, self.eps)
            q, qa = site_output_and_component_activation(h1, "q", attn.wq, per_kind_inputs)
            k, ka = site_output_and_component_activation(h1, "k", attn.wk, per_kind_inputs)
            v, va = site_output_and_component_activation(h1, "v", attn.wv, per_kind_inputs)
            attention_output = attn.core(q, k, v, self.inv_freq)
            o, oa = site_output_and_component_activation(
                attention_output, "o", attn.wo, per_kind_inputs
            )
            post_attention = residual_in + o
            h2 = rms_norm(post_attention, layer.ln2, self.eps)
            gate, ga = site_output_and_component_activation(h2, "gate", layer.Wg, per_kind_inputs)
            up, ua = site_output_and_component_activation(h2, "up", layer.Wu, per_kind_inputs)
            down_input = jax.nn.silu(gate) * up
            down, da = site_output_and_component_activation(
                down_input, "down", layer.Wd, per_kind_inputs
            )
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
                gate_up_input=h2,
                gate_output=gate,
                up_output=up,
                down_input=down_input,
                down_output=down,
                residual_out=residual_out,
            )
            if not collect_component_activations:
                return block_activations, None
            kinds = ("q", "k", "v", "o", "gate", "up", "down")
            component_activations_by_kind = {
                kind: activation
                for kind, activation in zip(kinds, (qa, ka, va, oa, ga, ua, da), strict=True)
                if activation is not None
            }
            return block_activations, component_activations_by_kind

        base_policy = (
            jax.checkpoint_policies.nothing_saveable
            if remat
            else jax.checkpoint_policies.dots_saveable
        )
        never_save_gathered = jax.checkpoint_policies.save_anything_except_these_names(
            GATHERED_WEIGHTS_CHECKPOINT_NAME
        )

        def policy(prim: Any, *args: Any, **params: Any) -> bool:
            return bool(base_policy(prim, *args, **params)) and bool(
                never_save_gathered(prim, *args, **params)
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
        bounds = sorted({0, first_live, last_live, self.n_layer})

        for lo, hi in zip(bounds, bounds[1:], strict=False):
            if lo == hi:
                continue
            segment_live = first_live <= lo and hi <= last_live and first_live < last_live
            if segment_live:
                per_kind_segment_inputs = {
                    kind: {key: value[lo:hi] for key, value in entry.items()}
                    for kind, entry in per_kind.items()
                }
                xs: Any = (slice_layers(lo, hi), per_kind_segment_inputs)
            else:
                xs = slice_layers(lo, hi)

            if layout is not None:
                segment_slots = {key: value[lo:hi] for key, value in slot_indices_by_tap.items()}
                xs = (*xs, segment_slots) if segment_live else (xs, segment_slots)

                def captured_block(
                    state: tuple[Array, dict[str, Array]],
                    layer_input: Any,
                    segment_live: bool = segment_live,
                    layout: _ScanCaptureLayout = layout,
                ) -> tuple[tuple[Array, dict[str, Array]], dict[str, Array] | None]:
                    x, buffers_ = state
                    if segment_live:
                        layer, per_kind_layer_inputs, slots = layer_input
                        block_activations, block_component_activations = (
                            compute_live_block_activations(x, layer, per_kind_layer_inputs)
                        )
                    else:
                        layer, slots = layer_input
                        block_activations = self._block_activations(x, layer)
                        block_component_activations = None
                    updated_buffers = _write_block_captures(
                        layout, buffers_, slots, block_activations
                    )
                    return (
                        block_activations.residual_out,
                        updated_buffers,
                    ), block_component_activations

                (residual, buffers), segment_component_activations = run_scan(
                    captured_block, (residual, buffers), xs
                )
            else:

                def plain_block(
                    x: Array,
                    layer_input: Any,
                    segment_live: bool = segment_live,
                ) -> tuple[Array, dict[str, Array] | None]:
                    if segment_live:
                        layer, per_kind_layer_inputs = layer_input
                        block_activations, block_component_activations = (
                            compute_live_block_activations(x, layer, per_kind_layer_inputs)
                        )
                        return block_activations.residual_out, block_component_activations
                    return _clean_block(layer_input, x, self.inv_freq, self.eps), None

                residual, segment_component_activations = run_scan(plain_block, residual, xs)

            if segment_component_activations is not None:
                component_activation_segments.append(segment_component_activations)

        if layout is not None:
            _read_capture_buffers(captured_by_source, layout, buffers)

        captures = _captures_in_request_order(capture_sources, captured_by_source)
        component_activations: dict[str, Array] = {}
        if collect_component_activations:
            assert component_activation_segments, (
                "component activations require a non-empty live segment"
            )
            stacks = {
                kind: jnp.concatenate(
                    [part[kind] for part in component_activation_segments], axis=0
                )
                for kind in component_activation_segments[0]
            }
            for site in live_site_names:
                layer, kind = parse_site_name(site)
                component_activations[site] = stacks[kind][layer - first_live]
            assert set(component_activations) == set(live_site_names), (
                sorted(component_activations),
                sorted(live_site_names),
            )

        residual = rms_norm(residual, self.norm, self.eps)
        forward_result = ForwardResult.from_producer(
            output=residual @ self.lm_head.T,
            capture_keys=capture_keys,
            capture_values=captures,
        )
        return forward_result, component_activations

    def prepare_compute_weights(self, vu: ComponentStacks) -> dict[str, dict[str, Array]]:
        return _reconstruct_compute_weights(_stack_per_kind_vu(vu, self.n_layer))

    def component_activation_forward(
        self,
        prepared_weights: dict[str, dict[str, Array]],
        inputs: Array,
        /,
        *,
        capture_keys: CaptureKeys,
    ) -> tuple[ForwardResult, dict[str, Array]]:
        """Run one frozen forward for requested captures and every decomposed site's ``x @ V``."""
        component_input_keys: list[str] = []
        site_locations = tuple(parse_site_name(site) for site in self.site_names)
        for layer, kind in site_locations:
            match kind:
                case "q" | "k" | "v":
                    component_input_keys.append(attention_input_tap_key(layer))
                case "o":
                    component_input_keys.append(attention_output_tap_key(layer))
                case "gate" | "up":
                    component_input_keys.append(mlp_input_tap_key(layer))
                case "down":
                    component_input_keys.append(mlp_hidden_tap_key(layer))
                case _:
                    raise AssertionError(kind)

        full_forward_result = self.clean_forward(
            inputs,
            capture_keys | frozenset(component_input_keys),
        )
        component_activations: dict[str, Array] = {}
        for site, key, (layer, kind) in zip(
            self.site_names, component_input_keys, site_locations, strict=True
        ):
            V = prepared_weights[kind]["V"][layer]
            component_activations[site] = full_forward_result.captures[key].astype(V.dtype) @ V
        requested_forward_result = ForwardResult(
            output=full_forward_result.output,
            captures={key: full_forward_result.captures[key] for key in sorted(capture_keys)},
        )
        return requested_forward_result, component_activations

    def stack_ci(self, ci_lower: dict[str, Array]) -> dict[str, Array]:
        return _stack_ci_per_kind(ci_lower, self.n_layer)

    def masked_forward_from_residual(
        self,
        prepared_weights: dict[str, dict[str, Array]],
        residual: Float[Array, "b t d"],
        /,
        *,
        masking: Masking,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        remat: bool,
    ) -> ForwardResult:
        masked_forward_result, _component_activations = self._run_masked_forward(
            prepared_weights,
            residual,
            masking,
            remat,
            tuple(sorted(capture_keys)),
            collect_component_activations=False,
        )
        return masked_forward_result

    def masked_component_activations_from_residual(
        self,
        prepared_weights: dict[str, dict[str, Array]],
        residual: Float[Array, "b t d"],
        masking: MaterializedMasking,
    ) -> dict[str, Array]:
        _forward_result, activations = self._run_masked_forward(
            prepared_weights,
            residual,
            masking,
            False,
            (),
            collect_component_activations=True,
        )
        return activations

    def weight_deltas(self, vu: ComponentStacks) -> dict[str, Array]:
        """fp32 `W − V@U` per site from fp32 masters (SPEC N2; faithfulness input)."""
        weight_deltas: dict[str, Array] = {}
        for spec in self.sites:
            layer, kind = parse_site_name(spec.name)
            frozen_weight = _frozen_site_weight(
                jax.tree.map(lambda array, layer_index=layer: array[layer_index], self.stacked),
                kind,
            )
            site_components = vu.site(spec.name)
            weight_deltas[spec.name] = (
                frozen_weight.astype(jnp.float32)
                - (site_components.V.astype(jnp.float32) @ site_components.U.astype(jnp.float32)).T
            )
        return weight_deltas


# ----------------------------- HF weight loading -----------------------------


class GLUDecomposedModel(_GLUTransformerCore):
    """The token-input GLU transformer (`model.py` contract; SPEC §1): embeds internally,
    then runs the shared core. A family's identity lives in its `stacked.attn` module
    (its `FrozenAttn` variant) and `inv_freq`, never in a switch here."""

    embed: Float[Array, "vocab d"]

    def embed_tokens(self, tokens: Int[Array, "b t"]) -> Float[Array, "b t d"]:
        return self.embed[tokens]

    @override
    def shardings(self, mesh: Mesh) -> Self:
        """The core's shardings + a replicated embedding (~2 GB with the head: small, and
        vocab-parallel lookup isn't worth the complexity)."""
        core = super().shardings(mesh)
        return eqx.tree_at(lambda m: m.embed, core, NamedSharding(mesh, P()))

    @override
    def clean_forward(
        self, inputs: Int[Array, "b t"], capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS
    ) -> ForwardResult:
        return self.forward_from_residual(self.embed_tokens(inputs), capture_keys)

    def masked_forward(
        self,
        prepared_weights: dict[str, dict[str, Array]],
        inputs: Int[Array, "b t"],
        /,
        *,
        masking: Masking,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        remat: bool,
    ) -> ForwardResult:
        return self.masked_forward_from_residual(
            prepared_weights,
            self.embed_tokens(inputs),
            masking=masking,
            capture_keys=capture_keys,
            remat=remat,
        )

    def masked_component_activations(
        self,
        prepared_weights: dict[str, dict[str, Array]],
        inputs: Int[Array, "b t"],
        masking: MaterializedMasking,
    ) -> dict[str, Array]:
        return self.masked_component_activations_from_residual(
            prepared_weights, self.embed_tokens(inputs), masking
        )


class ResidualGLUDecomposedModel(_GLUTransformerCore):
    """A depth-suffix GLU transformer: consumes RESIDUAL activations as its input (the
    prefix runs upstream, e.g. once per batch as a data mapper), runs the remaining
    blocks, and unembeds. Same `model.py` contract — the batch edge is opaque to the
    engine, so a residual batch is as legal as a token batch. Nothing below the split is
    observable: its sites, taps, and captures live in ITS block coordinates
    (original block `k + i` = this model's block `i`)."""

    @override
    def clean_forward(
        self, inputs: Float[Array, "b t d"], capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS
    ) -> ForwardResult:
        return self.forward_from_residual(inputs, capture_keys)

    def masked_forward(
        self,
        prepared_weights: dict[str, dict[str, Array]],
        inputs: Float[Array, "b t d"],
        /,
        *,
        masking: Masking,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        remat: bool,
    ) -> ForwardResult:
        return self.masked_forward_from_residual(
            prepared_weights, inputs, masking=masking, capture_keys=capture_keys, remat=remat
        )

    def masked_component_activations(
        self,
        prepared_weights: dict[str, dict[str, Array]],
        inputs: Float[Array, "b t d"],
        masking: MaterializedMasking,
    ) -> dict[str, Array]:
        return self.masked_component_activations_from_residual(prepared_weights, inputs, masking)


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
    """Lazy keyed access to the sharded safetensors of an HF checkpoint, cast to the
    FAMILY's frozen-weights dtype (SPEC N1: bf16 storage — the families pass it; this
    module holds no dtype opinion)."""

    def __init__(self, snapshot: Path, dtype: DTypeLike):
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        self._key_to_file = index["weight_map"]
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
            Wg=w.get(f"{pre}.{i}.mlp.gate_proj.weight"),
            Wu=w.get(f"{pre}.{i}.mlp.up_proj.weight"),
            Wd=w.get(f"{pre}.{i}.mlp.down_proj.weight"),
        )
        for i in range(cfg.n_layer)
    ]


def build_decomposed_lm(
    embed: Array,
    layers: list[GLULayer],
    norm: Array,
    lm_head: Array,
    inv_freq: Array,
    cfg: GLUArch,
    sites: tuple[SiteSpec, ...],
) -> GLUDecomposedModel:
    """Assemble a `GLUDecomposedModel` from the frozen full-model arrays + decomposition
    config. `sites` must be canonical-ordered with dims matching `cfg`."""
    site_cs = tuple(SiteC(s.name, s.C) for s in sites)
    assert sites == glu_site_specs(cfg, canonical_site_cs(site_cs)), (
        f"sites are not the canonical specs for this config: {sites}"
    )
    return GLUDecomposedModel(
        embed=embed,
        stacked=_stack_layers(layers),
        n_layer=len(layers),
        norm=norm,
        lm_head=lm_head,
        inv_freq=inv_freq,
        sites=sites,
        has_position_axis=True,
        eps=cfg.rms_norm_eps,
    )


def load_decomposed_glu_from_hf(
    model_name: str,
    cfg: GLUArch,
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
        lm_head=w.get("lm_head.weight"),
        inv_freq=inv_freq,
        cfg=cfg,
        sites=sites,
    )
