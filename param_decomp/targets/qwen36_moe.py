"""The Qwen3.6-MoE architecture target (HF `qwen3_5_moe`; concrete support:
`Qwen/Qwen3.6-35B-A3B`) — the first MoE and first hybrid-attention slice, on its own
engine rather than `glu_transformer` (the hybrid layer schedule and the MoE waist share
no code shape with the GLU stack).

ARCHITECTURE. 40 pre-norm layers on a `full_attention_interval`-periodic schedule —
3 gated-DeltaNet linear-attention layers then 1 gated full-attention layer per stage —
each followed by an MoE MLP: 256 routed experts (top-8, softmax-then-topk with top-k
renormalization) plus one always-on shared expert behind a scalar sigmoid gate. Untied
head. Every `*_layernorm`/`q_norm`/`k_norm`/final norm is HF's ZERO-CENTERED RMSNorm
(`(1+w)`, fp32 multiply); the DeltaNet output norm alone is the plain-weight gated norm.
The numeric kernels are `param_decomp.target_ports.qwen3_5_moe`. The checkpoint's mrope
degenerates to plain partial RoPE for text-only position ids (identical T/H/W planes make
`apply_interleaved_mrope` a no-op), so this target implements partial RoPE directly —
pinned by `tests/qwen36_moe_hf_parity/`.

SITES. Both token mixers are FROZEN — never decomposed. The decomposed sites are the MoE
matrices, uniform across all 40 layers, with the expert axis STRUCTURAL inside each site:
`layers.{i}.mlp.experts.{gate,up,down}_proj` is the FUSED all-expert matrix (gate/up
`[E·di, d]`, down `[d, E·di]` — one honest linear map, the dense equivalent of the routed
MoE), and `layers.{i}.mlp.shared_expert.{gate,up,down}_proj` the shared expert's. Routing
weights fold into the fused down site's INPUT (`w_e · hidden_e`, exactly `w_e` applied
after each expert's down in exact arithmetic), so an unrouted expert's component
activations are exactly zero. The router and the shared-expert scalar gate stay frozen.

EXECUTION. The 256-expert compute is ROUTED — top-k jobs sorted by expert, grouped
matmuls over selected experts only (`param_decomp/routed/experts`) — frozen AND
decomposed. Wherever no `experts_*` kind is decomposed (the clean forward, masked
forwards decomposing only `shared_*` kinds) the frozen arm runs with the fp32
routing-weighted combine. A forward decomposing ANY `experts_*` kind runs the routed
DECOMPOSED arm on the same jobs schedule: per-job V_e/U_e contractions through the
C_block bottleneck, masks/CI gathered to job space (`[.., k·c]` live per token), the
frozen delta/route channels on the same grouped matmuls, routing weights folded into
the down-site input (`ExpertsExecution` enumerates the arms; `"dense"` is the
all-expert oracle the parity suite compares against). The shared expert is always
dense. The layer stack runs as one `lax.scan` over stages (the periodic unit), each
stage body unrolling its `interval` sublayers. Each decomposed kind must cover every
layer (whole-grid c-specs) — an enumerated gap, asserted loudly.

PLACEMENT (Plan A). On the two-axis `(data, tp)` mesh under the
`zero1-replicated-resident-moe` preset, every frozen weight persists at its operand
layout — expert-major fused axes and mixer heads ÷tp within the node; the 2 KV heads,
router, embeddings, and norms replicated — so no while body gathers a weight. The batch
shards over `data`; activations replicate over `tp` at the residual waist and shard
over it at the fused/hidden widths (Megatron column/row pairs). The routed frozen
expert arm becomes EP by activation slicing (`routed.experts.ExpertShardedJobs`): each
rank computes only its expert shard's jobs, and partial outputs reduce over `tp` in the
fp32 combine. V/U expert blocks co-locate with their frozen experts (`expert: tp`).
"""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import cache, partial
from typing import Literal, cast, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int

from param_decomp.core import family
from param_decomp.core.axes import Axes, MeshAxis, SemanticAxis
from param_decomp.core.components import (
    ComponentStacks,
    ExpertBlocked,
    Factorization,
    NarrowCI,
    SiteC,
    SiteCI,
    SiteDims,
    SiteSpec,
    activation_axes,
    map_site_ci,
    require_full_emission,
    site_ci_values,
    site_slots_for,
)
from param_decomp.core.decomposed_linear import (
    ExpertPlannedComponentLinear,
    PlannedComponentLinear,
    expert_block_site_forward,
    site_out,
)
from param_decomp.core.family import ArchFamily
from param_decomp.core.linear_plan import (
    ExpertContraction,
    placed_linear,
    uniform_like,
    unreduce,
    value_mesh,
)
from param_decomp.core.masking import compose_source_mask
from param_decomp.core.model import (
    EMPTY_CAPTURE_KEYS,
    CaptureKeys,
    ForwardResult,
    Masking,
    MaterializedMasking,
    SourceMasking,
    StochasticMasking,
)
from param_decomp.core.nonlinearity import Neurons, NonlinearityPartition
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
from param_decomp.routed.experts import (
    ExpertShardedJobs,
    GroupedMatmulBackend,
    RoutedJobs,
    combine_jobs,
    ep_combine_jobs,
    ep_gather_job_blocks,
    ep_gather_tokens,
    ep_grouped_matmul,
    ep_sort_job_values,
    ep_sort_jobs,
    ep_sum_jobs,
    ep_unsort_jobs,
    expert_sharded_jobs,
    gather_job_blocks,
    gather_tokens,
    grouped_matmul,
    routed_jobs,
    scatter_jobs,
    sort_job_values,
    sort_jobs,
    sum_jobs,
    unsort_jobs,
)
from param_decomp.target_ports.llama import (
    AttentionImplementation,
    attn_implementation,
    rope_cos_sin,
)
from param_decomp.target_ports.qwen3_5_moe import (
    GatedDeltaKernel,
    apply_partial_rope,
    causal_depthwise_conv1d_silu,
    gated_delta_rule,
    gated_rms_norm,
    rms_norm_zero_centered,
)
from param_decomp.targets.glu_transformer import HFWeights, default_inv_freq, hf_snapshot_dir
from param_decomp.targets.lm_output import LMOutput, StreamedLinearOutput, pin_lm_output_batch
from param_decomp.targets.losses import lm_output_kl_per_position
from param_decomp.targets.transformer_taps import mlp_input_tap_key, site_output_tap_key

# ----------------------------- config -----------------------------


@dataclass(frozen=True)
class Qwen36MoeConfig:
    """The `qwen3_5_moe` text-decoder architecture, exactly what this target implements:
    no vision tower, no MTP head, untied embeddings (the model carries an explicit
    `lm_head`). Field values mirror the HF `text_config`."""

    vocab_size: int
    n_layer: int
    full_attention_interval: int
    n_embd: int
    # full attention (every `interval`-th layer)
    n_head: int
    n_kv_head: int
    head_dim: int
    partial_rotary_factor: float
    rope_theta: float
    # gated DeltaNet (all other layers)
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_num_value_heads: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    # MoE MLP (every layer)
    n_experts: int
    n_experts_per_token: int
    moe_intermediate: int
    shared_expert_intermediate: int
    rms_norm_eps: float
    max_position_embeddings: int

    def __post_init__(self) -> None:
        assert self.full_attention_interval >= 2, self.full_attention_interval
        assert self.n_layer % self.full_attention_interval == 0, (
            self.n_layer,
            self.full_attention_interval,
        )
        assert self.n_head % self.n_kv_head == 0, (self.n_head, self.n_kv_head)
        assert self.linear_num_value_heads % self.linear_num_key_heads == 0, self
        rotary = self.head_dim * self.partial_rotary_factor
        assert rotary == int(rotary) and int(rotary) % 2 == 0, rotary
        assert self.n_experts_per_token <= self.n_experts, self

    @property
    def n_stages(self) -> int:
        return self.n_layer // self.full_attention_interval

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def n_ctx(self) -> int:
        """The context bound under its role name; `max_position_embeddings` is HF's."""
        return self.max_position_embeddings


def qwen36_35b_a3b_config() -> Qwen36MoeConfig:
    """Architecture of `Qwen/Qwen3.6-35B-A3B` (text decoder; `transformers` 4.57.1
    config, `layer_types` = 3×linear_attention then full_attention, repeating)."""
    return Qwen36MoeConfig(
        vocab_size=248320,
        n_layer=40,
        full_attention_interval=4,
        n_embd=2048,
        n_head=16,
        n_kv_head=2,
        head_dim=256,
        partial_rotary_factor=0.25,
        rope_theta=10_000_000.0,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=32,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        n_experts=256,
        n_experts_per_token=8,
        moe_intermediate=512,
        shared_expert_intermediate=512,
        rms_norm_eps=1e-6,
        max_position_embeddings=262144,
    )


def layer_is_full_attention(cfg: Qwen36MoeConfig, layer: int) -> bool:
    """HF's `layer_types` schedule: full attention closes each interval."""
    return (layer + 1) % cfg.full_attention_interval == 0


# ----------------------------- family / site grammar -----------------------------

# The decomposed matrix vocabulary: the MoE waist only. The expert axis is structural
# INSIDE the `experts_*` sites (fused all-expert matrices); the token mixers are frozen
# and have no site spelling at all.
Qwen36MoeMatrix = Literal[
    "experts_gate",
    "experts_up",
    "experts_down",
    "shared_gate",
    "shared_up",
    "shared_down",
]

KIND_ORDER: tuple[str, ...] = get_args(Qwen36MoeMatrix)
"""Within-layer canonical site order = computation order (routed experts, then the
shared expert), DERIVED from the `Qwen36MoeMatrix` vocabulary."""

_EXPERT_KINDS = frozenset({"experts_gate", "experts_up", "experts_down"})
"""The kinds sharing one expert arm: decomposing ANY of them switches that arm from
routed-frozen to the routed DECOMPOSED execution (undecomposed expert kinds inside it
run their frozen grouped matmuls on the same jobs schedule)."""

ExpertsExecution = Literal["routed", "dense"]
"""The enumerated decomposed-expert executions — correctness-identical up to fp32
reassociation. `routed` is the production arm (job-space compute over the k selected
experts per token — the memory-efficient arm for large component counts); `dense` computes every expert
for every token (`expert_block_site_forward` over the whole fused matrix,
verbatim) and exists as the parity oracle."""


@dataclass(frozen=True)
class MaterializedOutputEdge:
    """Every forward materializes its `[.., vocab]` logits — the default edge."""


@dataclass(frozen=True)
class StreamedOutputEdge:
    """Every forward returns the FACTORED output (`targets.lm_output.StreamedLinearOutput`:
    the final activations + the frozen unembedding reference) so the `[.., vocab]` logits
    never materialize; output comparisons stream over `n_vocab_chunks` vocab chunks.
    Reassociation-identical to the materialized edge — the streamed kernels form each
    chunk's logits with the same native-dtype matmul."""

    n_vocab_chunks: int


OutputEdge = MaterializedOutputEdge | StreamedOutputEdge
"""The enumerated model-output edges: which member of the `LMOutput` union
(`targets.lm_output`) every forward produces. Consumers dispatch on the output union,
never on this declaration."""

SITE_NAME_PATTERN = re.compile(
    r"^layers\.(\d+)\.mlp\.(?:experts\.(gate|up|down)|shared_expert\.(gate|up|down))_proj$"
)


def site_name(layer: int, kind: str) -> str:
    assert kind in KIND_ORDER, kind
    scope, role = kind.split("_")
    submodule = "experts" if scope == "experts" else "shared_expert"
    return f"layers.{layer}.mlp.{submodule}.{role}_proj"


def router_idx_tap_key(layer: int) -> str:
    """Layer `layer`'s captured top-k expert ids `[.., k]` (int32), in `_topk_routing`'s
    stored order — THE component-identity key of narrow CI emission (one spelling; the
    parse arm and the CI-fn resolver both read it here)."""
    return f"router_idx.{layer}"


def router_weights_tap_key(layer: int) -> str:
    """Layer `layer`'s captured renormalized routing weights `[.., k]` (fp32)."""
    return f"router_weights.{layer}"


def is_expert_kind(kind: str) -> bool:
    """Whether one matrix kind is expert-blocked (narrow-emitting) vs shared (dense)."""
    assert kind in KIND_ORDER, kind
    return kind in _EXPERT_KINDS


def parse_site_name(name: str) -> tuple[int, str]:
    """`layers.{i}.mlp.{experts,shared_expert}.{gate|up|down}_proj` -> (layer, kind);
    rejects anything else."""
    match = SITE_NAME_PATTERN.match(name)
    assert match is not None, (
        f"not a qwen36_moe site: {name!r} (sites are layers.{{i}}.mlp.experts."
        f"{{gate|up|down}}_proj / layers.{{i}}.mlp.shared_expert.{{gate|up|down}}_proj)"
    )
    layer, expert_role, shared_role = match.groups()
    kind = f"experts_{expert_role}" if expert_role is not None else f"shared_{shared_role}"
    return int(layer), kind


FAMILY = ArchFamily("qwen36_moe", KIND_ORDER, site_name, parse_site_name)
"""This family's matrix grammar as data — the vocabulary + name renderer qwen36_moe
c-specs resolve against."""


def site_dims(cfg: Qwen36MoeConfig, kind: str) -> SiteDims:
    """Dimensions of one per-layer site matrix in right-mult orientation. The `experts_*`
    sites are the FUSED all-expert matrices (expert-major on the `E·di` axis)."""
    d = cfg.n_embd
    fused = cfg.n_experts * cfg.moe_intermediate
    shared = cfg.shared_expert_intermediate
    match kind:
        case "experts_gate" | "experts_up":
            return SiteDims(d_in=d, d_out=fused)
        case "experts_down":
            return SiteDims(d_in=fused, d_out=d)
        case "shared_gate" | "shared_up":
            return SiteDims(d_in=d, d_out=shared)
        case "shared_down":
            return SiteDims(d_in=shared, d_out=d)
        case _:
            raise AssertionError(f"unknown kind {kind!r}")


def site_factorization(cfg: Qwen36MoeConfig, kind: str, C: int) -> Factorization:
    """How one kind's V/U factor its matrix: the `experts_*` sites are expert-local
    (`ExpertBlocked`, `c_per_expert = C // n_experts` components confined to each
    expert's block), the shared-expert sites are dense."""
    d = cfg.n_embd
    di = cfg.moe_intermediate
    match kind:
        case "experts_gate" | "experts_up" | "experts_down":
            assert C % cfg.n_experts == 0, (
                f"{kind}: C={C} must be a multiple of n_experts={cfg.n_experts}"
            )
            c_per_expert = C // cfg.n_experts
            d_in, d_out = (d, di) if kind != "experts_down" else (di, d)
            return ExpertBlocked(
                n_experts=cfg.n_experts, d_in=d_in, d_out=d_out, c_per_expert=c_per_expert
            )
        case "shared_gate" | "shared_up" | "shared_down":
            return site_dims(cfg, kind).dense(C)
        case _:
            raise AssertionError(f"unknown kind {kind!r}")


def site_contraction(kind: str) -> ExpertContraction | None:
    """The orientation of one kind's expert-blocked linears (None = a dense site):
    gate/up fuse the experts on their output, down on its input."""
    match kind:
        case "experts_gate" | "experts_up":
            return "fused_output"
        case "experts_down":
            return "fused_input"
        case "shared_gate" | "shared_up" | "shared_down":
            return None
        case _:
            raise AssertionError(f"unknown kind {kind!r}")


def nonlinearity_partition(kind: str) -> NonlinearityPartition | None:
    """gate/up writers face elementwise neurons (the silu·up product, per expert inside
    the fused axis); the residual-writing downs face none."""
    match kind:
        case "experts_gate" | "experts_up" | "shared_gate" | "shared_up":
            return Neurons()
        case "experts_down" | "shared_down":
            return None
        case _:
            raise AssertionError(f"unknown kind {kind!r}")


def canonical_site_cs(site_cs: tuple[SiteC, ...]) -> tuple[SiteC, ...]:
    return family.canonical_site_cs(FAMILY, site_cs)


def qwen36_moe_site_specs(cfg: Qwen36MoeConfig, site_cs: tuple[SiteC, ...]) -> tuple[SiteSpec, ...]:
    return family.site_specs(
        FAMILY,
        site_cs,
        lambda kind, c: site_factorization(cfg, kind, c),
        lambda kind: nonlinearity_partition(kind),
        cfg.n_layer,
    )


def full_site_cs(cfg: Qwen36MoeConfig, c_of: Mapping[str, int]) -> tuple[SiteC, ...]:
    """Every layer's sites for the selected kinds at their C, in canonical order — the
    whole-grid tiling this target's masked forward requires per decomposed kind."""
    assert set(c_of) <= set(KIND_ORDER), sorted(c_of)
    return tuple(
        SiteC(site_name(layer, kind), c_of[kind])
        for layer in range(cfg.n_layer)
        for kind in KIND_ORDER
        if kind in c_of
    )


# ----------------------------- frozen modules -----------------------------


_GATED_DELTA_KERNEL: GatedDeltaKernel = "chunkwise"
"""The wired gated-delta-rule arm; `sequential` is the parity oracle the tests hold it
against (`GatedDeltaKernel` enumerates both)."""


class FrozenGatedDeltaNet(eqx.Module):
    """The `qwen3_5_moe` gated-DeltaNet token mixer (split in_proj variant): depthwise
    causal conv + silu on q/k/v, per-v-head decay `−exp(A_log)·softplus(a + dt_bias)`
    and write strength `σ(b)`, l2-normed q/k, the fp32 delta-rule scan, then the
    plain-weight gated output norm and out projection.

    HF's fused `in_proj_qkv` (and its conv weight) is stored SPLIT into its q/k/v row
    blocks: the conv is per-channel and silu elementwise, so the split pieces compute
    bit-identical values, and each piece's head axis then shards over `tp` whole
    (2 k-heads / 4 v-heads per rank at the production mesh) — the fused axis's shard
    boundaries would cut across the q|k|v concatenation instead."""

    w_q: Float[Array, "kd d"]
    w_k: Float[Array, "kd d"]
    w_v: Float[Array, "vd d"]
    w_z: Float[Array, "vd d"]
    w_b: Float[Array, "vh d"]
    w_a: Float[Array, "vh d"]
    conv_q: Float[Array, "kd k"]
    conv_k: Float[Array, "kd k"]
    conv_v: Float[Array, "vd k"]
    a_log: Float[Array, " vh"]
    dt_bias: Float[Array, " vh"]
    norm_w: Float[Array, " dv"]
    w_out: Float[Array, "d vd"]
    n_k_heads: int = eqx.field(static=True)
    n_v_heads: int = eqx.field(static=True)
    k_head_dim: int = eqx.field(static=True)
    v_head_dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def shardings(self, placement: PlacementRules, lead: Axes) -> "FrozenGatedDeltaNet":
        """Everything shards by HEAD over the column/row rows' tp assignment: projection
        rows and their conv channels follow their heads, the per-v-head decay/write
        vectors follow theirs, and the out projection consumes v-head columns. The
        gated-norm weight is per-head-DIM, shared across heads — replicated."""
        column = placement.target.column.persist
        row = placement.target.row.persist
        matrix: Axes = (*lead, "d_out", "d_in")
        heads: Axes = (*lead, "d_out")
        for w in (self.w_q, self.w_k, self.w_v, self.w_z, self.w_b, self.w_a):
            column.validate_shape(matrix, w.shape)
        row.validate_shape(matrix, self.w_out.shape)

        def channels(conv: Array) -> NamedSharding:
            # conv channels follow their projection's d_out shard; the kernel axis has
            # no semantic-axis name and replicates, spelled by extending the spec.
            column.validate_shape(heads, conv.shape[:-1])
            return NamedSharding(placement.mesh, P(*column.spec_for(heads), None))

        return eqx.tree_at(
            lambda m: (
                m.w_q,
                m.w_k,
                m.w_v,
                m.w_z,
                m.w_b,
                m.w_a,
                m.conv_q,
                m.conv_k,
                m.conv_v,
                m.a_log,
                m.dt_bias,
                m.norm_w,
                m.w_out,
            ),
            self,
            (
                *(column.sharding_for(matrix) for _ in range(6)),
                channels(self.conv_q),
                channels(self.conv_k),
                channels(self.conv_v),
                column.sharding_for(heads),
                column.sharding_for(heads),
                NamedSharding(placement.mesh, P()),
                row.sharding_for(matrix),
            ),
        )

    def __call__(
        self, x: Float[Array, "b t d"], placement: PlacementRules | None
    ) -> Float[Array, "b t d"]:
        b, t, _ = x.shape
        value_dim = self.n_v_heads * self.v_head_dim
        column = None if placement is None else placement.target.column
        row = None if placement is None else placement.target.row
        q_flat = causal_depthwise_conv1d_silu(
            placed_target_linear(x, self.w_q, column), self.conv_q
        )
        k_flat = causal_depthwise_conv1d_silu(
            placed_target_linear(x, self.w_k, column), self.conv_k
        )
        v_flat = causal_depthwise_conv1d_silu(
            placed_target_linear(x, self.w_v, column), self.conv_v
        )
        z = placed_target_linear(x, self.w_z, column).reshape(b, t, self.n_v_heads, self.v_head_dim)
        beta = jax.nn.sigmoid(placed_target_linear(x, self.w_b, column))
        # fp32 before the exp/softplus: a bf16 A_log exponentiates to ±inf (HF's note).
        g = -jnp.exp(self.a_log.astype(jnp.float32)) * jax.nn.softplus(
            placed_target_linear(x, self.w_a, column).astype(jnp.float32)
            + self.dt_bias.astype(jnp.float32)
        )
        q = q_flat.reshape(b, t, self.n_k_heads, self.k_head_dim)
        k = k_flat.reshape(b, t, self.n_k_heads, self.k_head_dim)
        v = v_flat.reshape(b, t, self.n_v_heads, self.v_head_dim)
        rep = self.n_v_heads // self.n_k_heads
        if rep > 1:
            # out-head j reads in-head j//rep: a head-major broadcast-merge, so a
            # tp-sharded head axis stays rank-local (in-heads land on their out-heads'
            # rank whenever tp divides the k-head count); the spec is unchanged, but
            # jnp.repeat on an explicitly sharded axis demands it spelled.
            if value_mesh(q).empty:
                q = jnp.repeat(q, rep, axis=2)
                k = jnp.repeat(k, rep, axis=2)
            else:
                q = jnp.repeat(q, rep, axis=2, out_sharding=jax.typeof(q).sharding)
                k = jnp.repeat(k, rep, axis=2, out_sharding=jax.typeof(k).sharding)
        core = gated_delta_rule(q, k, v, g, beta, _GATED_DELTA_KERNEL)
        core = gated_rms_norm(core, self.norm_w, z, self.eps)
        return placed_target_linear(core.reshape(b, t, value_dim), self.w_out, row)


class FrozenGatedAttention(eqx.Module):
    """The `qwen3_5_moe` full-attention token mixer: the q projection emits query and a
    per-head sigmoid output gate (2·head_dim per head, query first), zero-centered
    per-head QK-norm before partial RoPE, GQA causal SDPA, gate, o projection."""

    wq: Float[Array, "qg d"]
    wk: Float[Array, "kvd d"]
    wv: Float[Array, "kvd d"]
    wo: Float[Array, "d qd"]
    q_norm: Float[Array, " hd"]
    k_norm: Float[Array, " hd"]
    n_head: int = eqx.field(static=True)
    n_kv_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    implementation: AttentionImplementation = eqx.field(static=True)

    def shardings(self, placement: PlacementRules, lead: Axes) -> "FrozenGatedAttention":
        """q/o shard by query head (`wq` rows are head-major query|gate pairs); the K/V
        projections and per-head norms REPLICATE — the checkpoint's 2 KV heads sit below
        any real tp degree, and a row split inside one head would break the SDPA head
        unit. The placed forward repeats K/V to the query-head count and shards the
        copies instead."""
        column = placement.target.column.persist
        row = placement.target.row.persist
        matrix: Axes = (*lead, "d_out", "d_in")
        column.validate_shape(matrix, self.wq.shape)
        row.validate_shape(matrix, self.wo.shape)
        repl = NamedSharding(placement.mesh, P())
        return eqx.tree_at(
            lambda m: (m.wq, m.wk, m.wv, m.wo, m.q_norm, m.k_norm),
            self,
            (
                column.sharding_for(matrix),
                repl,
                repl,
                row.sharding_for(matrix),
                repl,
                repl,
            ),
        )

    def __call__(
        self, x: Float[Array, "b t d"], inv_freq: Array, placement: PlacementRules | None
    ) -> Float[Array, "b t d"]:
        b, t, _ = x.shape
        column = None if placement is None else placement.target.column
        row = None if placement is None else placement.target.row
        query_and_gate = placed_target_linear(x, self.wq, column).reshape(
            b, t, self.n_head, 2 * self.head_dim
        )
        q = query_and_gate[..., : self.head_dim]
        gate = query_and_gate[..., self.head_dim :]
        q = rms_norm_zero_centered(q, self.q_norm, self.eps)
        k = rms_norm_zero_centered(
            (x @ self.wk.T).reshape(b, t, self.n_kv_head, self.head_dim), self.k_norm, self.eps
        )
        v = (x @ self.wv.T).reshape(b, t, self.n_kv_head, self.head_dim)
        cos, sin = rope_cos_sin(inv_freq, t, q.dtype)
        q, k = apply_partial_rope(q, k, cos, sin)
        if placement is not None:
            # The internal GQA head-group reshape cannot split a q-head axis sharded
            # finer than the KV head count, and cuDNN SDPA wants q/k/v identically
            # sharded — so under placement K/V repeat to the query-head count (out-head
            # j reads kv-head j // rep, the GQA grouping) and take q's head sharding;
            # the unplaced arm keeps the grouped-KV fast path.
            rep = self.n_head // self.n_kv_head
            k = jax.sharding.reshard(jnp.repeat(k, rep, axis=2), jax.typeof(q).sharding)
            v = jax.sharding.reshard(jnp.repeat(v, rep, axis=2), jax.typeof(q).sharding)
        # Pin the SDPA operand shapes: a mis-split query/gate or a bad head reshape
        # reaches cuDNN as an unequal-head-dim (MLA) graph and dies at device-side graph
        # validation — make it a Python error with the shapes in hand instead.
        expected_kv_heads = self.n_head if placement is not None else self.n_kv_head
        assert q.shape == (b, t, self.n_head, self.head_dim) and k.shape == v.shape == (
            b,
            t,
            expected_kv_heads,
            self.head_dim,
        ), (q.shape, k.shape, v.shape, self.n_head, expected_kv_heads, self.head_dim)
        out = jax.nn.dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            implementation=attn_implementation(
                self.implementation, jax.default_backend(), q.dtype, t
            ),
        )
        out = out * jax.nn.sigmoid(gate)
        return placed_target_linear(out.reshape(b, t, self.n_head * self.head_dim), self.wo, row)


class FrozenMoE(eqx.Module):
    """One layer's frozen MoE weights: input norm, router, the FUSED all-expert
    gate/up/down (expert-major fused axis — the decomposed sites' frozen matrices),
    the shared expert, and its scalar sigmoid gate."""

    ln: Float[Array, " d"]
    router: Float[Array, "E d"]
    experts_gate: Float[Array, "Edi d"]
    experts_up: Float[Array, "Edi d"]
    experts_down: Float[Array, "d Edi"]
    shared_gate: Float[Array, "si d"]
    shared_up: Float[Array, "si d"]
    shared_down: Float[Array, "d si"]
    shared_expert_gate: Float[Array, "1 d"]

    def shardings(self, placement: PlacementRules, n_experts: int) -> "FrozenMoE":
        """Column/row rows shard the fused expert axes over tp — expert-major, so a tp
        split IS an expert split (whole experts per rank; the divisibility is asserted
        here so a non-tiling expert count dies at placement, not at first trace). The
        router and scalar gate replicate: routing softmaxes over ALL experts on every
        rank."""
        column = placement.target.column.persist
        row = placement.target.row.persist
        matrix: Axes = ("layer", "d_out", "d_in")
        n_fused_shards = column.shard_count("d_out")
        assert n_experts % n_fused_shards == 0, (
            f"the fused expert axis shards ÷{n_fused_shards} (target/column.persist), "
            f"which must split WHOLE experts: n_experts={n_experts} does not tile"
        )
        assert n_experts % row.shard_count("d_in") == 0, (n_experts, row.rule)
        for w in (self.experts_gate, self.experts_up, self.shared_gate, self.shared_up):
            column.validate_shape(matrix, w.shape)
        for w in (self.experts_down, self.shared_down):
            row.validate_shape(matrix, w.shape)
        repl = NamedSharding(placement.mesh, P())
        return eqx.tree_at(
            lambda m: (
                m.ln,
                m.router,
                m.experts_gate,
                m.experts_up,
                m.experts_down,
                m.shared_gate,
                m.shared_up,
                m.shared_down,
                m.shared_expert_gate,
            ),
            self,
            (
                repl,
                repl,
                column.sharding_for(matrix),
                column.sharding_for(matrix),
                row.sharding_for(matrix),
                column.sharding_for(matrix),
                column.sharding_for(matrix),
                row.sharding_for(matrix),
                repl,
            ),
        )


class DeltaNetSublayer(eqx.Module):
    ln1: Float[Array, " d"]
    mixer: FrozenGatedDeltaNet


class AttnSublayer(eqx.Module):
    ln1: Float[Array, " d"]
    attn: FrozenGatedAttention


# ----------------------------- capture grammar -----------------------------


class _Tap(Enum):
    RESIDUAL_IN = "residual_in"
    MOE_INPUT = "moe_input"
    ROUTER_IDX = "router_idx"
    ROUTER_WEIGHTS = "router_weights"
    EXPERTS_GATE_OUTPUT = "experts_gate_output"
    EXPERTS_UP_OUTPUT = "experts_up_output"
    EXPERTS_DOWN_OUTPUT = "experts_down_output"
    SHARED_GATE_OUTPUT = "shared_gate_output"
    SHARED_UP_OUTPUT = "shared_up_output"
    SHARED_DOWN_OUTPUT = "shared_down_output"
    RESIDUAL_OUT = "residual_out"


_SITE_OUTPUT_TAP: dict[str, _Tap] = {
    "experts_gate": _Tap.EXPERTS_GATE_OUTPUT,
    "experts_up": _Tap.EXPERTS_UP_OUTPUT,
    "experts_down": _Tap.EXPERTS_DOWN_OUTPUT,
    "shared_gate": _Tap.SHARED_GATE_OUTPUT,
    "shared_up": _Tap.SHARED_UP_OUTPUT,
    "shared_down": _Tap.SHARED_DOWN_OUTPUT,
}

# The kinds whose masked output can influence each same-layer site-output tap; residual
# taps and the MoE input depend only on EARLIER layers.
_INTERMEDIATE_TAPS = frozenset(
    {
        _Tap.EXPERTS_GATE_OUTPUT,
        _Tap.EXPERTS_UP_OUTPUT,
        _Tap.SHARED_GATE_OUTPUT,
        _Tap.SHARED_UP_OUTPUT,
    }
)
"""Taps whose values live at the fused/hidden width — placed at the intermediate row
(feature over tp); every other tap is model-width at the external row."""


def _tap_feature_row(placement: PlacementRules, tap: _Tap) -> PlacedRule:
    if tap in _INTERMEDIATE_TAPS:
        return placement.target.intermediate
    return placement.activations.external


def _tap_dtype(tap: _Tap, residual_dtype: jnp.dtype) -> jnp.dtype:
    """Capture-buffer dtype per tap: activations at the residual dtype; the routing
    taps at `_topk_routing`'s own — int32 ids, fp32 renormalized weights (the CI fn
    casts the weights to compute dtype at entry; `cast_floating` skips the ids)."""
    match tap:
        case _Tap.ROUTER_IDX:
            return jnp.dtype(jnp.int32)
        case _Tap.ROUTER_WEIGHTS:
            return jnp.dtype(jnp.float32)
        case (
            _Tap.RESIDUAL_IN
            | _Tap.MOE_INPUT
            | _Tap.EXPERTS_GATE_OUTPUT
            | _Tap.EXPERTS_UP_OUTPUT
            | _Tap.EXPERTS_DOWN_OUTPUT
            | _Tap.SHARED_GATE_OUTPUT
            | _Tap.SHARED_UP_OUTPUT
            | _Tap.SHARED_DOWN_OUTPUT
            | _Tap.RESIDUAL_OUT
        ):
            return residual_dtype


_SAME_LAYER_DEPENDENCIES: dict[_Tap, frozenset[str]] = {
    _Tap.RESIDUAL_IN: frozenset(),
    _Tap.MOE_INPUT: frozenset(),
    # the frozen router reads the normed MoE input — upstream of every same-layer site
    _Tap.ROUTER_IDX: frozenset(),
    _Tap.ROUTER_WEIGHTS: frozenset(),
    _Tap.EXPERTS_GATE_OUTPUT: frozenset({"experts_gate"}),
    _Tap.EXPERTS_UP_OUTPUT: frozenset({"experts_up"}),
    _Tap.EXPERTS_DOWN_OUTPUT: frozenset({"experts_gate", "experts_up", "experts_down"}),
    _Tap.SHARED_GATE_OUTPUT: frozenset({"shared_gate"}),
    _Tap.SHARED_UP_OUTPUT: frozenset({"shared_up"}),
    _Tap.SHARED_DOWN_OUTPUT: frozenset({"shared_gate", "shared_up", "shared_down"}),
    _Tap.RESIDUAL_OUT: frozenset(KIND_ORDER),
}


@dataclass(frozen=True, kw_only=True)
class _CaptureSource:
    layer: int
    tap: _Tap


def _parse_capture_key(key: str, n_layer: int) -> _CaptureSource:
    """This target's closed activation vocabulary: `resid.{b}` boundaries, `mlp_in.{b}`
    (the normed MoE input), and `{site}.out` linear site outputs — the shared
    transformer-tap spellings (`transformer_taps`); anything else fails closed."""
    if key.startswith("resid."):
        boundary = int(key.removeprefix("resid."))
        assert 0 <= boundary <= n_layer, (key, n_layer)
        if boundary == 0:
            return _CaptureSource(layer=0, tap=_Tap.RESIDUAL_IN)
        return _CaptureSource(layer=boundary - 1, tap=_Tap.RESIDUAL_OUT)
    if key.startswith("mlp_in."):
        block = int(key.removeprefix("mlp_in."))
        assert 0 <= block < n_layer, (key, n_layer)
        return _CaptureSource(layer=block, tap=_Tap.MOE_INPUT)
    if key.endswith(".out"):
        layer, kind = parse_site_name(key.removesuffix(".out"))
        assert 0 <= layer < n_layer, (key, n_layer)
        return _CaptureSource(layer=layer, tap=_SITE_OUTPUT_TAP[kind])
    if key.startswith("router_idx.") or key.startswith("router_weights."):
        # The MoE CI fn's routing inputs: layer {l}'s top-k expert ids `[.., k]` (int32)
        # and renormalized routing weights `[.., k]` (fp32) — the values `_topk_routing`
        # already computes on every forward; capture is threading, not new compute.
        tap = _Tap.ROUTER_IDX if key.startswith("router_idx.") else _Tap.ROUTER_WEIGHTS
        layer = int(key.split(".", 1)[1])
        assert 0 <= layer < n_layer, (key, n_layer)
        return _CaptureSource(layer=layer, tap=tap)
    raise AssertionError(f"unknown qwen36_moe activation {key!r}")


def _capture_sources(keys: tuple[str, ...], n_layer: int) -> tuple[_CaptureSource, ...]:
    sources = tuple(_parse_capture_key(key, n_layer) for key in keys)
    assert len(set(sources)) == len(sources), (
        "multiple capture keys name one physical activation",
        keys,
    )
    return sources


_UNUSED_SLOT = -1


def _capture_layout(
    sources: tuple[_CaptureSource, ...], n_layer: int
) -> dict[str, tuple[int, ...]]:
    """One exact-size scan-carry buffer per requested tap kind: tap value -> per-layer
    slot (−1 unused). The embedding residual (`RESIDUAL_IN`) is recorded pre-scan."""
    layout: dict[str, tuple[int, ...]] = {}
    for tap in _Tap:
        layers = [source.layer for source in sources if source.tap is tap]
        if not layers or tap is _Tap.RESIDUAL_IN:
            continue
        slot_by_layer = [_UNUSED_SLOT] * n_layer
        for slot, layer in enumerate(layers):
            slot_by_layer[layer] = slot
        layout[tap.value] = tuple(slot_by_layer)
    return layout


@dataclass(frozen=True, kw_only=True)
class _LayerActs:
    """One layer's capturable activations (the MoE waist plus the residual boundary).

    The fused-width gate/up taps exist only when captured (None otherwise — no full-width
    materialization), and under BOTH routed expert arms (frozen and decomposed) they hold
    the SCATTERED job results: selected experts' outputs exactly as computed, ZEROS at
    unselected experts (whose outputs the routed arms never compute; the dense oracle
    materializes them)."""

    moe_input: Array
    router_indices: Array
    router_weights: Array
    experts_gate_output: Array | None
    experts_up_output: Array | None
    experts_down_output: Array
    shared_gate_output: Array
    shared_up_output: Array
    shared_down_output: Array
    residual_out: Array

    def of(self, tap: _Tap) -> Array:
        match tap:
            case _Tap.RESIDUAL_IN:
                raise AssertionError("the embedding residual is recorded pre-scan")
            case _Tap.MOE_INPUT:
                return self.moe_input
            case _Tap.ROUTER_IDX:
                return self.router_indices
            case _Tap.ROUTER_WEIGHTS:
                return self.router_weights
            case _Tap.EXPERTS_GATE_OUTPUT:
                assert self.experts_gate_output is not None, "tap not captured"
                return self.experts_gate_output
            case _Tap.EXPERTS_UP_OUTPUT:
                assert self.experts_up_output is not None, "tap not captured"
                return self.experts_up_output
            case _Tap.EXPERTS_DOWN_OUTPUT:
                return self.experts_down_output
            case _Tap.SHARED_GATE_OUTPUT:
                return self.shared_gate_output
            case _Tap.SHARED_UP_OUTPUT:
                return self.shared_up_output
            case _Tap.SHARED_DOWN_OUTPUT:
                return self.shared_down_output
            case _Tap.RESIDUAL_OUT:
                return self.residual_out


def _write_captures(
    buffers: dict[str, Array], slots: dict[str, Array], acts: _LayerActs
) -> dict[str, Array]:
    updated = dict(buffers)
    for buffer_key, buffer in buffers.items():
        slot = slots[buffer_key]
        value = acts.of(_Tap(buffer_key))
        updated[buffer_key] = jax.lax.cond(
            slot != _UNUSED_SLOT,
            lambda buf, v=value, s=slot: jax.lax.dynamic_update_index_in_dim(buf, v, s, axis=0),
            lambda buf: buf,
            buffer,
        )
    return updated


# ----------------------------- decomposed-site execution -----------------------------


GROUPED_MATMUL_BACKEND: GroupedMatmulBackend = "tokamax_split_vjp"
"""The arm the family's PRODUCTION builders author — the sm100 split-VJP kernel
triple — for the target model (`build_qwen36_moe_model`'s real and abstract callers)
and the MoE CI fn's expert banks (the experiment resolve). The backend is authored at
construction, never inherited ambiently: toy-shape builders (`targets.testing`, the
placement tests) author the `ragged_dot` oracle instead, because the split arm's
d_weights kernel tiles d_in/d_out at 64 and REFUSES smaller dims — engine-semantics
fixtures stay tiny on the oracle, while the kernel arms' own parity lives in
`tests/routed/test_experts` at 64-multiple shapes."""


_ROW_KINDS = frozenset({"experts_down", "shared_down"})
"""Kinds whose frozen linear consumes the target `row` declaration (fused/hidden input,
model-width output); the rest are `column` (model-width input, fused/hidden output)."""


def _stage_blocked(value: Array, n_stages: int, interval: int) -> Array:
    """A layer-major stack re-laid out stage-blocked, `[n_stages·interval, …] →
    [n_stages, interval, …]` — a pure re-layout (layer-major order makes the reshape a
    view of the resident buffer, no gather) that survives a `reduced` typing. This is
    the stack's whole trip to the stage scan: it enters as xs, so the scan saves a VIEW
    of the resident for its backward, never a fragment copy; `_block_positions` splits
    out the interval positions inside the (checkpointed) body. A `reduced`-tagged stack
    (a materialized resident) takes the provenance-preserving custom VJP, so its dV/dU
    cotangents ride back to the flat stack still unreduced and reduce exactly once, at
    the materialize boundary; an untagged stack (frozen weights, masks, CI values)
    reshapes with ordinary autodiff."""
    off_mesh = value_mesh(value).empty
    tag = frozenset() if off_mesh else frozenset(jax.typeof(value).sharding.spec.reduced)
    if not tag:
        return value.reshape(n_stages, interval, *value.shape[1:])
    return _stage_blocked_provenance(value, n_stages, interval)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _stage_blocked_provenance(value: Array, n_stages: int, interval: int) -> Array:
    # Untag, reshape, re-tag: reshape has no reduced-typing rule of its own, and jax's
    # reshard transpose targets the PLAIN intermediate spec — which would cash the
    # deferred master reduction here as a full ALL-REDUCE per stack — so the backward
    # is owned below.
    tag = frozenset(jax.typeof(value).sharding.spec.reduced)
    blocked = unreduce(value).reshape(n_stages, interval, *value.shape[1:])
    spec = jax.typeof(blocked).sharding.spec
    return jax.sharding.reshard(blocked, P(*spec, reduced=tag))


def _stage_blocked_provenance_fwd(value: Array, n_stages: int, interval: int) -> tuple[Array, None]:
    return _stage_blocked_provenance(value, n_stages, interval), None


def _stage_blocked_provenance_bwd(n_stages: int, interval: int, _: None, ct: Array) -> tuple[Array]:
    # The blocked cotangent arrives `unreduced` over the master provenance axes —
    # per-device partial sums. The re-layout back to the flat layer-major stack is
    # device-LOCAL (one reshape), so it runs under shard_map with the unreduced typing
    # carried through verbatim: zero collectives here; the ONE deferred reduction
    # fires at the materialize boundary's transpose.
    spec = jax.typeof(ct).sharding.spec
    tag = frozenset(spec.unreduced)
    assert tag, ("the provenance staging path exists only for reduced-tagged stacks", spec)
    assert all(axis is None for axis in spec.partitions[:2]), spec
    blocked_spec = P(*spec.partitions, unreduced=tag)
    flat_spec = P(None, *spec.partitions[2:], unreduced=tag)

    def flatten_local(c: Array) -> Array:
        return c.reshape(n_stages * interval, *c.shape[2:])

    flat = jax.shard_map(
        flatten_local,
        mesh=jax.typeof(ct).sharding.mesh,
        in_specs=(blocked_spec,),
        out_specs=flat_spec,
        check_vma=False,
    )(ct)
    return (flat,)


_stage_blocked_provenance.defvjp(_stage_blocked_provenance_fwd, _stage_blocked_provenance_bwd)


def _stage_blocked_tree[T](tree: T, n_stages: int, interval: int) -> T:
    """`_stage_blocked` over a tree's leaves."""
    return jax.tree.map(lambda leaf: _stage_blocked(leaf, n_stages, interval), tree)


def _block_positions(block: Array) -> tuple[Array, ...]:
    """One stage's `[positions, …]` xs slice split into its per-position leaves, INSIDE
    the scan body — under the body's `jax.checkpoint`, so the backward re-slices the
    saved blocked view at zero FLOPs instead of loading a saved fragment copy. Same
    provenance rule as `_stage_blocked`: `x[i]` has no reduced-typing rule, so a tagged
    block takes the custom VJP and its cotangents stay unreduced."""
    off_mesh = value_mesh(block).empty
    tag = frozenset() if off_mesh else frozenset(jax.typeof(block).sharding.spec.reduced)
    if not tag:
        return tuple(block[p] for p in range(block.shape[0]))
    return _block_positions_provenance(block)


@jax.custom_vjp
def _block_positions_provenance(block: Array) -> tuple[Array, ...]:
    tag = frozenset(jax.typeof(block).sharding.spec.reduced)
    plain = unreduce(block)
    frags = tuple(plain[p] for p in range(block.shape[0]))
    spec = jax.typeof(frags[0]).sharding.spec
    return tuple(jax.sharding.reshard(frag, P(*spec, reduced=tag)) for frag in frags)


def _block_positions_provenance_fwd(block: Array) -> tuple[tuple[Array, ...], None]:
    return _block_positions_provenance(block), None


def _block_positions_provenance_bwd(_: None, cts: tuple[Array, ...]) -> tuple[Array]:
    # Position cotangents arrive `unreduced`; restacking them into the block is
    # device-LOCAL (one stack), zero collectives — the deferred reduction stays at
    # the materialize boundary, never inside the while loop.
    spec = jax.typeof(cts[0]).sharding.spec
    tag = frozenset(spec.unreduced)
    assert tag, ("the provenance position split exists only for reduced-tagged blocks", spec)
    frag_spec = P(*spec.partitions, unreduced=tag)
    block_spec = P(None, *spec.partitions, unreduced=tag)

    block = jax.shard_map(
        lambda *frags: jnp.stack(frags, axis=0),
        mesh=jax.typeof(cts[0]).sharding.mesh,
        in_specs=(frag_spec,) * len(cts),
        out_specs=block_spec,
        check_vma=False,
    )(*cts)
    return (block,)


_block_positions_provenance.defvjp(_block_positions_provenance_fwd, _block_positions_provenance_bwd)


def _tree_by_position[T](tree: T, n_positions: int) -> tuple[T, ...]:
    """`_block_positions` over a stage's tree: one tree of per-position leaves per
    interval position (a leafless tree splits into `n_positions` copies of itself)."""
    leaves, treedef = jax.tree.flatten(tree)
    split = [_block_positions(leaf) for leaf in leaves]
    for leaf, frags in zip(leaves, split, strict=True):
        assert len(frags) == n_positions, (leaf.shape, n_positions)
    return tuple(
        jax.tree.unflatten(treedef, [frags[p] for frags in split]) for p in range(n_positions)
    )


def _expert_shard_axis(placement: PlacementRules) -> MeshAxis:
    """The ONE mesh axis the fused expert dimension (and so the expert grid) shards
    over, read off the column operand row — the routed placed arm slices its jobs by
    it. Fail-closed: a multi-axis or absent assignment has no EP spelling here."""
    assignment = placement.target.column.operand.assignment("d_out")
    assert len(assignment) == 1, (
        f"the placed routed expert arm needs the fused expert axis on exactly one mesh "
        f"axis; target/column.operand assigns d_out -> {assignment!r}"
    )
    (axis,) = assignment
    return axis


def _kind_target_linear(
    placement: PlacementRules | None, kind: str
) -> TargetLinearPlacement | None:
    if placement is None:
        return None
    return placement.target.row if kind in _ROW_KINDS else placement.target.column


V_WEIGHT_AXES: tuple[SemanticAxis, SemanticAxis] = ("d_in", "C")
U_WEIGHT_AXES: tuple[SemanticAxis, SemanticAxis] = ("C", "d_out")


SiteEntryTensor = Array | NarrowCI
"""One attached per-site tensor: a bare array, or — for a narrow-emitting expert site's
mask/CI — the `NarrowCI` bundle whose router indices key the routed decomposed arm."""


def _entry_tensor(inputs: Mapping[str, SiteEntryTensor], key: str) -> Array:
    value = inputs[key]
    assert not isinstance(value, NarrowCI), key
    return value


def _entry_masking(
    inputs: Mapping[str, SiteEntryTensor], uses_weight_deltas: bool
) -> tuple[SiteCI, Array | None, Array | None]:
    """One site entry's masking tensors — (mask, per-token delta, per-token route); the
    mask is `[.., C]` for a full site, a `NarrowCI` bundle for a narrow one.
    Materialized entries pass through; the recipe entries carry the shared CI and
    rebuild their masks HERE, inside the checkpointed stage: a stochastic entry
    (`src_key`) draws fresh sources — a narrow site at the narrow shape, its unrouted
    components structurally absent — and a persistent entry (`src`) recomposes
    `ci + (1-ci)·source` from its staged source values."""
    if "src_key" in inputs:
        ci_lower = inputs["ci"]
        src_key = _entry_tensor(inputs, "src_key")
        mask = map_site_ci(lambda v: v + (1.0 - v) * uniform_like(src_key, v), ci_lower)
        delta_mask = (
            uniform_like(
                _entry_tensor(inputs, "delta_key"), site_ci_values(ci_lower), drop_last_axis=True
            )
            if uses_weight_deltas
            else None
        )
    elif "src" in inputs:
        mask = compose_source_mask(inputs["ci"], inputs["src"])
        delta_mask = _entry_tensor(inputs, "delta_src") if uses_weight_deltas else None
    else:
        mask = inputs["mask"]
        delta_mask = inputs.get("delta")
        assert not isinstance(delta_mask, NarrowCI)
    route = inputs.get("route")
    assert not isinstance(route, NarrowCI)
    return mask, delta_mask, route


def _narrow_expert_masking(
    per_kind: Mapping[str, Mapping[str, SiteEntryTensor]],
) -> NarrowCI | None:
    """The expert kinds' narrow mask bundle, when the run emits narrowly — else None.
    One captured routing serves a layer's three expert kinds, so their bundles carry the
    same indices (the CI fn built all three from the same tap); any one keys the arm's
    jobs schedule. Mixed emission across expert kinds is unrepresentable upstream (one
    arch emits every expert site) and refused here."""
    decomposed = sorted(_EXPERT_KINDS & per_kind.keys())
    bundles = {
        kind: value
        for kind in decomposed
        if isinstance(value := per_kind[kind].get("ci", per_kind[kind].get("mask")), NarrowCI)
    }
    if not bundles:
        return None
    assert sorted(bundles) == decomposed, (
        f"mixed mask emission across expert kinds: narrow {sorted(bundles)} vs "
        f"decomposed {decomposed}"
    )
    return next(iter(bundles.values()))


def _decomposed_site_output(
    site_input: Array,
    frozen_weight: Array,
    inputs: Mapping[str, SiteEntryTensor],
    uses_weight_deltas: bool,
    contraction: ExpertContraction | None,
    placement: PlacementRules | None,
    target_linear: TargetLinearPlacement | None,
) -> Array:
    """One site through the core decomposed linear — the dense one for shared sites,
    the expert-blocked one (at the kind's declared orientation) for expert sites, each
    with its plans built from the rules when placed. Full-width masks only: narrow
    bundles belong to the routed decomposed arm (`require_full_emission` refuses)."""
    v, u = _entry_tensor(inputs, "V"), _entry_tensor(inputs, "U")
    full_mask, delta_mask, route = _entry_masking(inputs, uses_weight_deltas)
    mask = require_full_emission(full_mask)
    assert (placement is None) == (target_linear is None)
    frozen_linear = None if target_linear is None else target_linear_plan(site_input, target_linear)
    match contraction:
        case None:
            dense_placement = None
            if placement is not None and target_linear is not None:
                external = activation_axes(site_input.ndim, "feature")
                component = activation_axes(site_input.ndim, "C")
                dense_placement = PlannedComponentLinear(
                    v=placement.target_native_component_linear_plan(
                        target_linear, V_WEIGHT_AXES, external, component
                    ),
                    u=placement.target_native_component_linear_plan(
                        target_linear, U_WEIGHT_AXES, component, external
                    ),
                    component=placement.activations.component,
                    output=target_linear.output,
                )
            return site_out(
                site_input,
                v,
                u,
                frozen_weight,
                mask,
                delta_mask,
                route,
                dense_placement,
                frozen_linear,
            )
        case "fused_output" | "fused_input":
            expert_placement = None
            if placement is not None and target_linear is not None:
                expert_placement = ExpertPlannedComponentLinear(
                    v=placement.expert_component_linear_plan(
                        target_linear, contraction, "V", site_input.ndim
                    ),
                    u=placement.expert_component_linear_plan(
                        target_linear, contraction, "U", site_input.ndim
                    ),
                    component=placement.activations.component,
                    output=target_linear.output,
                )
            return expert_block_site_forward(
                site_input,
                v,
                u,
                frozen_weight,
                mask,
                delta_mask,
                route,
                contraction,
                expert_placement,
                frozen_linear,
            ).output


def _fold_routing_weights(gate: Array, up: Array, weights: Array) -> Array:
    """The down site's input with the routing weight folded in — `silu(gate)·up·weight`
    formed in fp32 and rounded to the activations' dtype ONCE, as the down matmul's
    operand (the grouped matmul takes its operands in one dtype, and the expert weights
    are the big one). The frozen arm weights its down OUTPUT inside the fp32 combine
    instead, so the two arms differ by this single rounding of the fused input; the
    fp32 `weights` broadcast against the trailing feature axis."""
    assert gate.dtype == up.dtype, (gate.dtype, up.dtype)
    assert weights.dtype == jnp.float32, weights.dtype
    fused = jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32) * weights
    return fused.astype(gate.dtype)


@dataclass(frozen=True)
class _UnplacedExpertActivations:
    """One layer's job-space state for expert-site component activations (unplaced):
    the captured routing's jobs schedule and gathered token rows, shared across the
    layer's expert kinds."""

    cfg: Qwen36MoeConfig
    frozen_gate: Array
    frozen_up: Array
    router_indices: Array
    router_weights: Array
    lead: tuple[int, ...]
    jobs: RoutedJobs
    x_jobs: Array
    backend: GroupedMatmulBackend

    def down_input(self) -> Array:
        """The FROZEN-path routed hidden `silu(gate)·up·weight` in job space — the down
        site's clean-forward input."""
        cfg = self.cfg
        d, di, n_experts = cfg.n_embd, cfg.moe_intermediate, cfg.n_experts
        gate = grouped_matmul(
            self.x_jobs,
            self.frozen_gate.reshape(n_experts, di, d).mT,
            self.jobs.group_sizes,
            self.backend,
        )
        up = grouped_matmul(
            self.x_jobs,
            self.frozen_up.reshape(n_experts, di, d).mT,
            self.jobs.group_sizes,
            self.backend,
        )
        weights = sort_jobs(self.router_weights.reshape(-1, cfg.n_experts_per_token), self.jobs)
        return _fold_routing_weights(gate, up, weights[:, None])

    def narrow_values(self, input_jobs: Array, V: Array) -> NarrowCI:
        acts = grouped_matmul(input_jobs, V, self.jobs.group_sizes, self.backend)
        width = self.cfg.n_experts_per_token * V.shape[-1]
        values = unsort_jobs(acts, self.jobs).reshape(*self.lead, width)
        return NarrowCI(values, self.router_indices, self.cfg.n_experts)


@dataclass(frozen=True)
class _PlacedExpertActivations:
    """`_UnplacedExpertActivations`' expert-parallel sibling on the explicit mesh."""

    cfg: Qwen36MoeConfig
    frozen_gate: Array
    frozen_up: Array
    router_indices: Array
    router_weights: Array
    lead: tuple[int, ...]
    shard_axis: str
    jobs: ExpertShardedJobs
    x_jobs: Array
    backend: GroupedMatmulBackend

    def down_input(self) -> Array:
        cfg = self.cfg
        d, di, n_experts = cfg.n_embd, cfg.moe_intermediate, cfg.n_experts
        gate = ep_grouped_matmul(
            self.x_jobs,
            self.frozen_gate.reshape(n_experts, di, d).mT,
            self.jobs,
            self.shard_axis,
            self.backend,
        )
        up = ep_grouped_matmul(
            self.x_jobs,
            self.frozen_up.reshape(n_experts, di, d).mT,
            self.jobs,
            self.shard_axis,
            self.backend,
        )
        weights = ep_sort_jobs(self.router_weights, self.jobs, self.shard_axis)
        return _fold_routing_weights(gate, up, weights[..., None])

    def narrow_values(self, input_jobs: Array, V: Array) -> NarrowCI:
        acts = ep_grouped_matmul(input_jobs, V, self.jobs, self.shard_axis, self.backend)
        width = self.cfg.n_experts_per_token * V.shape[-1]
        values = ep_unsort_jobs(acts, self.jobs, self.shard_axis).reshape(*self.lead, width)
        return NarrowCI(values, self.router_indices, self.cfg.n_experts)


def _expert_activation_context(
    moe_stack: FrozenMoE,
    cfg: Qwen36MoeConfig,
    captures: Mapping[str, Array],
    layer: int,
    placement: PlacementRules | None,
    backend: GroupedMatmulBackend,
) -> _UnplacedExpertActivations | _PlacedExpertActivations:
    router_indices = captures[router_idx_tap_key(layer)]
    router_weights = captures[router_weights_tap_key(layer)]
    h2 = captures[mlp_input_tap_key(layer)]
    frozen_gate = moe_stack.experts_gate[layer]
    frozen_up = moe_stack.experts_up[layer]
    lead = h2.shape[:-1]
    if placement is None:
        jobs = routed_jobs(router_indices.reshape(-1, cfg.n_experts_per_token), cfg.n_experts)
        return _UnplacedExpertActivations(
            cfg=cfg,
            frozen_gate=frozen_gate,
            frozen_up=frozen_up,
            router_indices=router_indices,
            router_weights=router_weights,
            lead=lead,
            jobs=jobs,
            x_jobs=gather_tokens(h2.reshape(-1, cfg.n_embd), jobs),
            backend=backend,
        )
    shard_axis = _expert_shard_axis(placement)
    jobs = expert_sharded_jobs(router_indices, cfg.n_experts, placement.mesh.shape[shard_axis])
    return _PlacedExpertActivations(
        cfg=cfg,
        frozen_gate=frozen_gate,
        frozen_up=frozen_up,
        router_indices=router_indices,
        router_weights=router_weights,
        lead=lead,
        shard_axis=shard_axis,
        jobs=jobs,
        x_jobs=ep_gather_tokens(h2, jobs, shard_axis),
        backend=backend,
    )


# ----------------------------- the DecomposedModel -----------------------------


class Qwen36MoeDecomposedModel(eqx.Module):
    """The qwen36_moe `DecomposedModel` (the `model.py` contract): the FROZEN full model
    as array fields — traced jit args, never HLO constants — with the trainable V/U
    passed to the forwards explicitly.

    The layer stack is stored in its periodic-stage layout: `deltanet` leaves lead with
    `[n_stages, interval−1, …]`, `attn` with `[n_stages, …]` (the stage scan's xs), and
    the uniform per-layer `moe` with `[n_layer, …]` (reshaped to stages per forward;
    read layer-flat by weight_deltas / the norms). Requires each decomposed kind on
    every layer; `shardings` is the Plan-A resident placement."""

    embed: Float[Array, "vocab d"]
    deltanet: DeltaNetSublayer
    attn: AttnSublayer
    moe: FrozenMoE
    norm: Float[Array, " d"]
    lm_head: Float[Array, "vocab d"]
    inv_freq: Float[Array, " r2"]
    cfg: Qwen36MoeConfig = eqx.field(static=True)
    sites: tuple[SiteSpec, ...] = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)
    experts_execution: ExpertsExecution = eqx.field(static=True)
    grouped_matmul_backend: GroupedMatmulBackend = eqx.field(static=True)
    output_edge: OutputEdge = eqx.field(static=True)

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.sites)

    def shardings(self, placement: PlacementRules) -> "Qwen36MoeDecomposedModel":
        """The Plan-A resident placement: every frozen weight PERSISTS at its operand
        layout (÷tp within the node, replicated over data), so no while body ever
        gathers a weight. Expert-major fused axes shard whole experts; mixers shard by
        head (the 2 KV heads replicate); embeddings/head/norms/router replicate per the
        resident rows. `deltanet` leaves lead `[n_stages, interval−1, …]`, `attn`
        `[n_stages, …]`, `moe` `[n_layer, …]` — the leading grid axes are unsharded, so
        they enter the rows as extra (replicated) `layer` names."""
        embedding_axes: Axes = ("vocab", "d_model")
        placement.target.embedding.persist.validate_shape(embedding_axes, self.embed.shape)
        placement.target.output.persist.validate_shape(embedding_axes, self.lm_head.shape)
        placement.target.normalization.validate_shape(("d_model",), self.norm.shape)
        placement.target.position_encoding.validate_shape(("rope_frequency",), self.inv_freq.shape)
        repl = NamedSharding(placement.mesh, P())
        return eqx.tree_at(
            lambda m: (m.embed, m.deltanet, m.attn, m.moe, m.norm, m.lm_head, m.inv_freq),
            self,
            (
                placement.target.embedding.persist.sharding_for(embedding_axes),
                eqx.tree_at(
                    lambda s: (s.ln1, s.mixer),
                    self.deltanet,
                    (repl, self.deltanet.mixer.shardings(placement, ("layer", "layer"))),
                ),
                eqx.tree_at(
                    lambda s: (s.ln1, s.attn),
                    self.attn,
                    (repl, self.attn.attn.shardings(placement, ("layer",))),
                ),
                self.moe.shardings(placement, self.cfg.n_experts),
                placement.target.normalization.sharding_for(("d_model",)),
                placement.target.output.persist.sharding_for(embedding_axes),
                placement.target.position_encoding.sharding_for(("rope_frequency",)),
            ),
        )

    @staticmethod
    def recon_loss_fn(masked_output: LMOutput, clean_output: LMOutput) -> Float[Array, ""]:
        return lm_output_kl_per_position(masked_output, clean_output)

    @staticmethod
    def pin_output_batch(output: LMOutput, mesh: Mesh | None) -> LMOutput:
        return pin_lm_output_batch(output, mesh)

    def site_output_keys(self, sites: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(site_output_tap_key(site) for site in sites)

    def assert_hidden_acts_reconstruction_points(self, keys: tuple[str, ...]) -> None:
        sites_by_layer: dict[int, set[str]] = {}
        for name in self.site_names:
            layer, kind = parse_site_name(name)
            sites_by_layer.setdefault(layer, set()).add(kind)
        dead = []
        for key in keys:
            source = _parse_capture_key(key, self.cfg.n_layer)
            boundary = source.layer + 1 if source.tap is _Tap.RESIDUAL_OUT else source.layer
            changed_before = any(site_layer < boundary for site_layer in sites_by_layer)
            changed_here = (
                bool(sites_by_layer.get(source.layer, set()) & _SAME_LAYER_DEPENDENCIES[source.tap])
                and source.tap is not _Tap.RESIDUAL_IN
            )
            if not changed_before and not changed_here:
                dead.append(key)
        assert not dead, (
            f"hidden_acts_reconstruction points {tuple(dead)} cannot change under masking and "
            "would contribute guaranteed zeros to the point mean"
        )

    # ---------- routing / MoE ----------

    def _router_probs(self, router: Array, h2: Array) -> Array:
        """The frozen router's fp32 softmax over ALL experts."""
        logits = h2 @ router.T
        return jax.nn.softmax(logits.astype(jnp.float32), axis=-1)

    def _topk_routing(self, router: Array, h2: Array) -> tuple[Array, Array]:
        """HF `Qwen3_5MoeTopKRouter`: fp32 softmax over ALL experts, top-k of the probs,
        top-k renormalized to sum 1. Returns (expert ids `[..., k]`, fp32 weights
        `[..., k]`)."""
        probs = self._router_probs(router, h2)
        # A stable sort over the (unsharded) expert axis, not `lax.top_k`:
        # value-identical (same descending order, same lowest-index tie-break), but
        # top_k's explicit-sharding rule gathers the batch axes — an in-loop
        # cross-data collective the census forbids. Spelled via sort_key_val so the
        # index payload can carry the keys' sharding (argsort's internal iota types
        # replicated, which sort refuses against sharded keys).
        # The ordering is piecewise-constant, so the sort runs under stop_gradient
        # (sort's JVP rule builds a replicated iota that explicit sharding refuses);
        # the values re-gather below, which carries exactly top_k's derivative.
        negated = jax.lax.stop_gradient(-probs)
        if value_mesh(negated).empty:
            order = jnp.argsort(negated, axis=-1, stable=True)
        else:
            payload = jax.lax.broadcasted_iota(
                jnp.int32,
                negated.shape,
                negated.ndim - 1,
                out_sharding=jax.typeof(negated).sharding,
            )
            _, order = jax.lax.sort_key_val(negated, payload, is_stable=True)
        top_indices = order[..., : self.cfg.n_experts_per_token]
        top_values = jnp.take_along_axis(probs, top_indices, axis=-1)
        return top_indices, top_values / jnp.sum(top_values, axis=-1, keepdims=True)

    def _routed_frozen_experts(
        self,
        moe: FrozenMoE,
        h2: Array,
        top_indices: Array,
        top_weights: Array,
        captured_taps: frozenset[_Tap],
    ) -> tuple[Array | None, Array | None, Array]:
        """The frozen expert compute, routed: the lead axes flatten to one token axis,
        each token's k assignments become jobs, and the grouped matmuls touch selected
        experts only. Fused-width gate/up taps materialize ONLY when captured
        (`_LayerActs` documents their scattered semantics)."""
        cfg = self.cfg
        backend = self.grouped_matmul_backend
        lead = h2.shape[:-1]
        d, di, n_experts = cfg.n_embd, cfg.moe_intermediate, cfg.n_experts
        jobs = routed_jobs(top_indices.reshape(-1, cfg.n_experts_per_token), n_experts)
        x_jobs = gather_tokens(h2.reshape(-1, d), jobs)
        gate_jobs = grouped_matmul(
            x_jobs,
            moe.experts_gate.reshape(n_experts, di, d).mT,
            jobs.group_sizes,
            backend,
        )
        up_jobs = grouped_matmul(
            x_jobs,
            moe.experts_up.reshape(n_experts, di, d).mT,
            jobs.group_sizes,
            backend,
        )
        down_jobs = grouped_matmul(
            jax.nn.silu(gate_jobs) * up_jobs,
            moe.experts_down.reshape(d, n_experts, di).transpose(1, 2, 0),
            jobs.group_sizes,
            backend,
        )
        routed = combine_jobs(down_jobs, jobs, top_weights.reshape(-1, cfg.n_experts_per_token))

        def fused_tap(tap: _Tap, jobs_out: Array) -> Array | None:
            if tap not in captured_taps:
                return None
            return scatter_jobs(jobs_out, jobs, n_experts).reshape(*lead, n_experts * di)

        return (
            fused_tap(_Tap.EXPERTS_GATE_OUTPUT, gate_jobs),
            fused_tap(_Tap.EXPERTS_UP_OUTPUT, up_jobs),
            routed.reshape(*lead, d),
        )

    def _routed_frozen_experts_placed(
        self,
        moe: FrozenMoE,
        h2: Array,
        top_indices: Array,
        top_weights: Array,
        captured_taps: frozenset[_Tap],
        placement: PlacementRules,
        out_spec: P,
    ) -> tuple[Array | None, Array | None, Array]:
        """The routed frozen arm on the explicit mesh: EP by activation slicing. Each
        rank computes only its expert shard's jobs against its resident expert blocks
        (`ExpertShardedJobs` — one per-batch-row sentinel-sorted schedule per shard,
        replicated integers, data-parallel over the batch rows), and the partial outputs
        reduce across the expert-shard axis inside the fp32 combine, landing at the
        caller's waist `out_spec`. Zero weight movement and zero cross-data collectives
        by construction."""
        cfg = self.cfg
        backend = self.grouped_matmul_backend
        d = h2.shape[-1]
        di, n_experts = cfg.moe_intermediate, cfg.n_experts
        assert not captured_taps & {_Tap.EXPERTS_GATE_OUTPUT, _Tap.EXPERTS_UP_OUTPUT}, (
            "fused expert gate/up taps under the placed routed arms are not built (the "
            "scattered full-width materialization has no placed spelling yet); capture "
            "them from an unplaced forward"
        )
        shard_axis = _expert_shard_axis(placement)
        jobs = expert_sharded_jobs(top_indices, n_experts, placement.mesh.shape[shard_axis])
        x_jobs = ep_gather_tokens(h2, jobs, shard_axis)
        gate_jobs = ep_grouped_matmul(
            x_jobs,
            moe.experts_gate.reshape(n_experts, di, d).mT,
            jobs,
            shard_axis,
            backend,
        )
        up_jobs = ep_grouped_matmul(
            x_jobs,
            moe.experts_up.reshape(n_experts, di, d).mT,
            jobs,
            shard_axis,
            backend,
        )
        down_jobs = ep_grouped_matmul(
            jax.nn.silu(gate_jobs) * up_jobs,
            moe.experts_down.reshape(d, n_experts, di).transpose(1, 2, 0),
            jobs,
            shard_axis,
            backend,
        )
        routed = ep_combine_jobs(down_jobs, jobs, top_weights, shard_axis, out_spec)
        return None, None, routed

    def _routed_decomposed_experts(
        self,
        moe: FrozenMoE,
        h2: Array,
        top_indices: Array,
        top_weights: Array,
        per_kind: dict[str, dict[str, SiteEntryTensor]],
        uses_weight_deltas: bool,
        captured_taps: frozenset[_Tap],
    ) -> tuple[Array | None, Array | None, Array]:
        """The decomposed expert compute on the SAME jobs schedule as the frozen arm:
        one job per (token, selected expert), per-job V_e/U_e grouped matmuls through
        the C_block bottleneck (the expert-blocked stacks are expert-major — exactly
        the grouped-matmul rhs layout), masks/CI gathered to job space (`[J, c]` — the
        k selected blocks per token), the frozen delta/route channels on the same
        grouped matmuls, routing weights folded into the down-site input in fp32
        (`_fold_routing_weights` — the identical math `_dense_experts`
        computes over every expert with the same fold, so those two arms differ by fp32
        reassociation only; against the frozen arm's post-down fp32 combine the fold
        costs one bf16 rounding of the fused input), and an unweighted per-token
        combine. Undecomposed expert kinds run their frozen
        grouped matmuls in the same job space. Fused-width gate/up taps materialize
        only when captured, with the scattered semantics `_LayerActs` documents."""
        cfg = self.cfg
        backend = self.grouped_matmul_backend
        lead = h2.shape[:-1]
        d, di, n_experts = cfg.n_embd, cfg.moe_intermediate, cfg.n_experts
        k = cfg.n_experts_per_token
        jobs = routed_jobs(top_indices.reshape(-1, k), n_experts)
        x_jobs = gather_tokens(h2.reshape(-1, d), jobs)

        def site_jobs(kind: str, input_jobs: Array, frozen_blocks: Array) -> Array:
            entry = per_kind.get(kind)
            if entry is None:
                return grouped_matmul(input_jobs, frozen_blocks, jobs.group_sizes, backend)
            mask, delta_mask, route = _entry_masking(entry, uses_weight_deltas)
            # Delta/route may carry size-1 broadcast lead axes (batch-shared persistent
            # sources). The dense arm broadcasts them through its
            # elementwise ops; the job gathers need them at the full lead first — the
            # broadcast's transpose is the same cross-lead sum dense's autodiff performs.
            if delta_mask is not None:
                delta_mask = jnp.broadcast_to(delta_mask, lead)
            if route is not None:
                route = jnp.broadcast_to(route, lead)
            c = _entry_tensor(entry, "V").shape[-1]
            match mask:
                case NarrowCI():
                    # The bundle's slot m IS the token's m-th routed expert under the
                    # SAME captured routing this arm's jobs schedule was built from
                    # (`_narrow_expert_masking`), so job order is a pure permutation of
                    # the (token, slot) rows — no partial gather, no dense view.
                    assert mask.values.shape[:-1] == lead, (mask.values.shape, lead)
                    coefficients = sort_job_values(mask.values.reshape(-1, k, c), jobs)
                case jax.Array():
                    # Full-width masks ride the CI's full waist shape.
                    assert mask.shape[:-1] == lead, (mask.shape, lead)
                    coefficients = gather_job_blocks(mask.reshape(-1, n_experts, c), jobs)
            delta = None
            if delta_mask is not None:
                delta = gather_tokens(delta_mask.reshape(-1, 1), jobs)
                coefficients = coefficients - delta
            acts = (
                grouped_matmul(input_jobs, _entry_tensor(entry, "V"), jobs.group_sizes, backend)
                * coefficients
            )
            out = grouped_matmul(acts, _entry_tensor(entry, "U"), jobs.group_sizes, backend)
            frozen_out = None
            if delta is not None or route is not None:
                frozen_out = grouped_matmul(input_jobs, frozen_blocks, jobs.group_sizes, backend)
            if delta is not None:
                assert frozen_out is not None
                out = out + delta * frozen_out
            if route is not None:
                assert frozen_out is not None
                out = jnp.where(route.reshape(-1)[jobs.sort_idx // k][:, None], out, frozen_out)
            return out

        gate_jobs = site_jobs("experts_gate", x_jobs, moe.experts_gate.reshape(n_experts, di, d).mT)
        up_jobs = site_jobs("experts_up", x_jobs, moe.experts_up.reshape(n_experts, di, d).mT)
        weight_jobs = sort_jobs(top_weights.reshape(-1, k), jobs)
        down_input = _fold_routing_weights(gate_jobs, up_jobs, weight_jobs[:, None])
        down_jobs = site_jobs(
            "experts_down",
            down_input,
            moe.experts_down.reshape(d, n_experts, di).transpose(1, 2, 0),
        )
        routed = sum_jobs(down_jobs, jobs)

        def fused_tap(tap: _Tap, jobs_out: Array) -> Array | None:
            if tap not in captured_taps:
                return None
            return scatter_jobs(jobs_out, jobs, n_experts).reshape(*lead, n_experts * di)

        return (
            fused_tap(_Tap.EXPERTS_GATE_OUTPUT, gate_jobs),
            fused_tap(_Tap.EXPERTS_UP_OUTPUT, up_jobs),
            routed.reshape(*lead, d),
        )

    def _routed_decomposed_experts_placed(
        self,
        moe: FrozenMoE,
        h2: Array,
        top_indices: Array,
        top_weights: Array,
        per_kind: dict[str, dict[str, SiteEntryTensor]],
        uses_weight_deltas: bool,
        captured_taps: frozenset[_Tap],
        placement: PlacementRules,
        out_spec: P,
    ) -> tuple[Array | None, Array | None, Array]:
        """The routed decomposed arm on the explicit mesh: the expert-sharded schedule
        serves the decomposed compute exactly as it serves the frozen arm — each rank
        computes only its expert shard's jobs against its resident V/U blocks
        (co-located with the frozen experts at `expert: tp`). The component matmuls
        keep the stacks' master provenance (`reduced` typing), so dV/dU cotangents ride
        the loop unreduced and reduce once at the entry boundary — zero in-loop
        cross-data collectives, exactly as the dense placed arm's einsums arrange
        through the reduced-typing rule. Masks/CI reshard their expert axis onto the
        shard axis (a typing move for the expert-major C@tp layout) and gather to job
        space shard-locally. The combine lands at the caller's waist `out_spec`."""
        cfg = self.cfg
        backend = self.grouped_matmul_backend
        d = h2.shape[-1]
        di, n_experts = cfg.moe_intermediate, cfg.n_experts
        k = cfg.n_experts_per_token
        assert not captured_taps & {_Tap.EXPERTS_GATE_OUTPUT, _Tap.EXPERTS_UP_OUTPUT}, (
            "fused expert gate/up taps under the placed routed arms are not built (the "
            "scattered full-width materialization has no placed spelling yet); capture "
            "them from an unplaced forward"
        )
        shard_axis = _expert_shard_axis(placement)
        jobs = expert_sharded_jobs(top_indices, n_experts, placement.mesh.shape[shard_axis])
        x_jobs = ep_gather_tokens(h2, jobs, shard_axis)

        def site_jobs(kind: str, input_jobs: Array, frozen_blocks: Array) -> Array:
            entry = per_kind.get(kind)
            if entry is None:
                return ep_grouped_matmul(input_jobs, frozen_blocks, jobs, shard_axis, backend)
            mask, delta_mask, route = _entry_masking(entry, uses_weight_deltas)
            # Delta/route may carry size-1 broadcast lead axes (batch-shared persistent
            # sources). The dense arm broadcasts them through its
            # elementwise ops; the job gathers need them at the full lead first — the
            # broadcast's transpose is the same cross-lead sum dense's autodiff performs.
            lead = h2.shape[:-1]
            lead_sharding = NamedSharding(placement.mesh, P(*jax.typeof(h2).sharding.spec[:-1]))
            if delta_mask is not None and delta_mask.shape != lead:
                delta_mask = jnp.broadcast_to(delta_mask, lead, out_sharding=lead_sharding)
            if route is not None and route.shape != lead:
                route = jnp.broadcast_to(route, lead, out_sharding=lead_sharding)
            c = _entry_tensor(entry, "V").shape[-1]
            match mask:
                case NarrowCI():
                    # Slot m IS the token's m-th routed expert under the schedule's own
                    # captured routing (`_narrow_expert_masking`): a pure per-cell
                    # permutation of the (token, slot) rows, shard-locally.
                    assert mask.values.shape[:-1] == lead, (mask.values.shape, lead)
                    coefficients = ep_sort_job_values(
                        mask.values.reshape(*lead, k, c), jobs, shard_axis
                    )
                case jax.Array():
                    # Full-width masks ride the CI's full waist shape.
                    assert mask.shape[:-1] == lead, (mask.shape, lead)
                    coefficients = ep_gather_job_blocks(
                        mask.reshape(*mask.shape[:-1], n_experts, c), jobs, shard_axis
                    )
            delta = None
            if delta_mask is not None:
                delta = ep_gather_tokens(delta_mask[..., None], jobs, shard_axis)
                coefficients = coefficients - delta
            acts = (
                ep_grouped_matmul(input_jobs, _entry_tensor(entry, "V"), jobs, shard_axis, backend)
                * coefficients
            )
            out = ep_grouped_matmul(acts, _entry_tensor(entry, "U"), jobs, shard_axis, backend)
            frozen_out = None
            if delta is not None or route is not None:
                frozen_out = ep_grouped_matmul(input_jobs, frozen_blocks, jobs, shard_axis, backend)
            if delta is not None:
                assert frozen_out is not None
                out = out + delta * frozen_out
            if route is not None:
                assert frozen_out is not None
                route_jobs = jnp.take_along_axis(route[:, None, :], jobs.sort_idx // k, axis=-1)
                out = jnp.where(route_jobs[..., None], out, frozen_out)
            return out

        gate_jobs = site_jobs("experts_gate", x_jobs, moe.experts_gate.reshape(n_experts, di, d).mT)
        up_jobs = site_jobs("experts_up", x_jobs, moe.experts_up.reshape(n_experts, di, d).mT)
        weight_jobs = ep_sort_jobs(top_weights, jobs, shard_axis)
        down_input = _fold_routing_weights(gate_jobs, up_jobs, weight_jobs[..., None])
        down_jobs = site_jobs(
            "experts_down",
            down_input,
            moe.experts_down.reshape(d, n_experts, di).transpose(1, 2, 0),
        )
        return None, None, ep_sum_jobs(down_jobs, jobs, shard_axis, out_spec)

    def _dense_experts(
        self,
        moe: FrozenMoE,
        h2: Array,
        top_indices: Array,
        top_weights: Array,
        run: Callable[[str, Array, Array], Array],
    ) -> tuple[Array, Array, Array]:
        """The all-expert dense compute — the `"dense"` `ExpertsExecution`, kept as the
        oracle the routed decomposed arm is verified against. The routing weights fold
        into the fused down INPUT, so the frozen and decomposed matrices share one seam
        and an unrouted expert's component activations are exactly zero."""
        cfg = self.cfg
        lead = h2.shape[:-1]
        dense_routing = jnp.sum(
            jax.nn.one_hot(top_indices, cfg.n_experts, dtype=jnp.float32) * top_weights[..., None],
            axis=-2,
        )
        gate = run("experts_gate", h2, moe.experts_gate)
        up = run("experts_up", h2, moe.experts_up)
        blocked_shape = (*lead, cfg.n_experts, cfg.moe_intermediate)
        gate_blocked, up_blocked = gate.reshape(blocked_shape), up.reshape(blocked_shape)
        if not value_mesh(gate_blocked).empty:
            # the routing vector follows the hidden's expert sharding (a local slice of
            # the replicated softmax) so the weighting multiply stays shard-local.
            dense_routing = jax.sharding.reshard(
                dense_routing,
                NamedSharding(
                    value_mesh(gate_blocked), P(*jax.typeof(gate_blocked).sharding.spec[:-1])
                ),
            )
        weighted = _fold_routing_weights(gate_blocked, up_blocked, dense_routing[..., None])
        return gate, up, run("experts_down", weighted.reshape(*lead, -1), moe.experts_down)

    def _moe_forward(
        self,
        moe: FrozenMoE,
        residual: Array,
        per_kind: dict[str, dict[str, SiteEntryTensor]],
        uses_weight_deltas: bool,
        captured_taps: frozenset[_Tap],
        placement: PlacementRules | None,
        residual_row: PlacedRule | None,
    ) -> _LayerActs:
        """One layer's MoE; kinds in `per_kind` run decomposed, the rest frozen. The
        expert arm dispatches statically: no `experts_*` kind decomposed → routed frozen
        compute; any decomposed → the routed decomposed arm on the same jobs schedule
        (`"dense"` execution is the all-expert oracle). Both routed arms take their
        expert-parallel spelling when placed. `residual_row` is the pass's between-blocks
        residual placement: the MoE interior always runs at the full external width — the
        normed input gathers to it here (the jobs schedule and routing must see every
        token, and one gather serves every consumer, so its transpose is the layer's
        ONE input-cotangent reduction) — and the block-exit combines land back at the
        residual row. `residual_out` is filled by the caller."""
        cfg = self.cfg
        h2 = rms_norm_zero_centered(residual, moe.ln, cfg.rms_norm_eps)
        h2_full = constrain_activation(
            h2, None if placement is None else placement.activations.external
        )

        def run(kind: str, site_input: Array, frozen_weight: Array) -> Array:
            target_linear = _kind_target_linear(placement, kind)
            if kind in per_kind:
                return _decomposed_site_output(
                    site_input,
                    frozen_weight,
                    per_kind[kind],
                    uses_weight_deltas,
                    site_contraction(kind),
                    placement,
                    target_linear,
                )
            return placed_target_linear(site_input, frozen_weight, target_linear)

        narrow = _narrow_expert_masking(per_kind)
        if narrow is None:
            top_indices, top_weights = self._topk_routing(moe.router, h2_full)
        else:
            # Narrow masks carry the CLEAN forward's routing: the decomposed expert arm
            # consumes the bundle's selection — re-deriving top-k from the masked pass's
            # perturbed residual would misalign mask slots with live experts (the §9
            # seam) — and weights it by the masked pass's renormalized probabilities AT
            # those experts, so mask≡1 with unperturbed upstream reproduces the clean
            # forward's routing exactly (the expert compute then differs from the frozen
            # arm by the fused-input rounding `_fold_routing_weights` documents).
            probs = self._router_probs(moe.router, h2_full)
            top_values = jnp.take_along_axis(probs, narrow.router_indices, axis=-1)
            top_indices = narrow.router_indices
            top_weights = top_values / jnp.sum(top_values, axis=-1, keepdims=True)
        if placement is None:
            residual_spec = None
        else:
            assert residual_row is not None
            residual_spec = residual_row.spec_for(activation_axes(h2_full.ndim, "feature"))
        if _EXPERT_KINDS.isdisjoint(per_kind):
            if residual_spec is None:
                gate, up, routed = self._routed_frozen_experts(
                    moe, h2_full, top_indices, top_weights, captured_taps
                )
            else:
                assert placement is not None
                gate, up, routed = self._routed_frozen_experts_placed(
                    moe, h2_full, top_indices, top_weights, captured_taps, placement, residual_spec
                )
        else:
            match self.experts_execution:
                case "routed":
                    if residual_spec is None:
                        gate, up, routed = self._routed_decomposed_experts(
                            moe,
                            h2_full,
                            top_indices,
                            top_weights,
                            per_kind,
                            uses_weight_deltas,
                            captured_taps,
                        )
                    else:
                        assert placement is not None
                        gate, up, routed = self._routed_decomposed_experts_placed(
                            moe,
                            h2_full,
                            top_indices,
                            top_weights,
                            per_kind,
                            uses_weight_deltas,
                            captured_taps,
                            placement,
                            residual_spec,
                        )
                case "dense":
                    assert narrow is None, (
                        "the dense all-expert oracle has no narrow-mask arm (it would "
                        "have to scatter the bundle to the full component axis); narrow "
                        "masks run experts_execution='routed'"
                    )
                    gate, up, routed = self._dense_experts(
                        moe, h2_full, top_indices, top_weights, run
                    )
        shared_gate = run("shared_gate", h2_full, moe.shared_gate)
        shared_up = run("shared_up", h2_full, moe.shared_up)
        shared_down = run("shared_down", jax.nn.silu(shared_gate) * shared_up, moe.shared_down)
        # Block exit: everything lands at the residual row. The placed routed combines
        # emit it directly (`out_spec`); the site linears' outputs cannot — their U
        # contraction carries the masters' chained-reduced typing, and jax's dot rule
        # refuses an output resharded onto the contraction's own tp axis while an
        # unreduced axis is in play — so they emit the external row and reshard here. The
        # scalar gate reads the residual-row-typed h2: a per-token op, so the sharded
        # view is exact.
        routed = constrain_activation(routed, residual_row)
        shared_down = constrain_activation(shared_down, residual_row)
        moe_out = routed + jax.nn.sigmoid(h2 @ moe.shared_expert_gate.T) * shared_down
        return _LayerActs(
            moe_input=h2_full,
            router_indices=top_indices,
            router_weights=top_weights,
            experts_gate_output=gate,
            experts_up_output=up,
            experts_down_output=routed,
            shared_gate_output=shared_gate,
            shared_up_output=shared_up,
            shared_down_output=shared_down,
            residual_out=residual + moe_out,
        )

    # ---------- the stage-scan forward ----------

    def _embed_tokens(self, tokens: Int[Array, "b t"], placement: PlacementRules | None) -> Array:
        assert tokens.shape[1] <= self.cfg.n_ctx, (tokens.shape, self.cfg.n_ctx)
        if placement is None:
            return self.embed[tokens]
        weight = materialize_stored_weight(
            self.embed,
            placement.target.embedding.persist,
            placement.target.embedding.operand,
            axes=("vocab", "d_model"),
        )
        # Type the residual at the external waist directly: the stage scan's carry must
        # enter with the type its body maintains. An off-mesh trace (untyped tokens)
        # takes the plain gather.
        if value_mesh(tokens).empty:
            return constrain_activation(weight[tokens], placement.activations.external)
        external = placement.activations.external
        axes = activation_axes(tokens.ndim + 1, "feature")
        return weight.at[tokens].get(out_sharding=external.sharding_for(axes))

    def _output(self, residual: Array, placement: PlacementRules | None) -> LMOutput:
        """The model-output edge off the final-norm residual: materialized logits, or
        the factored package whose head is the SAME operand-layout unembedding the
        materialized matmul would consume (a traced reference, never a copy)."""
        head = (
            self.lm_head
            if placement is None
            else materialize_stored_weight(
                self.lm_head,
                placement.target.output.persist,
                placement.target.output.operand,
                axes=("vocab", "d_model"),
            )
        )
        match self.output_edge:
            case MaterializedOutputEdge():
                return residual @ head.T
            case StreamedOutputEdge(n_vocab_chunks=n_vocab_chunks):
                assert self.cfg.vocab_size % n_vocab_chunks == 0, (
                    self.cfg.vocab_size,
                    n_vocab_chunks,
                )
                return StreamedLinearOutput(
                    activations=residual, head=head, n_chunks=n_vocab_chunks
                )

    def _forward(
        self,
        tokens: Int[Array, "b t"],
        per_kind: dict[str, dict[str, SiteEntryTensor]],
        ordered_capture_keys: tuple[str, ...],
        uses_weight_deltas: bool,
        checkpoint_policy: Callable[..., bool] | None,
        placement: PlacementRules | None,
        residual_row: PlacedRule | None,
    ) -> ForwardResult[LMOutput]:
        """The one forward engine: a `lax.scan` over stages, each stage body unrolling
        its interval sublayers (mixer + MoE). `per_kind` entries lead with `[n_layer, …]`
        (empty = the all-frozen clean forward); `checkpoint_policy` reruns stage bodies
        in the backward (the masked forwards). `residual_row` is the pass's between-blocks
        residual placement (`activations.external` for clean passes, `masked_external`
        for masked ones): the scan carry rides it, block interiors always run at the full
        external width — each block entry gathers the normed carry, each block exit's
        reduction lands back at the residual row — and the final residual gathers to
        external before the output edge, so a sequence-parallel residual never leaves
        this engine."""
        assert (residual_row is None) == (placement is None), (residual_row, placement)
        cfg = self.cfg
        sequence_parallel = (
            placement is not None and residual_row is not placement.activations.external
        )
        assert not (sequence_parallel and ordered_capture_keys), (
            "activation capture under sequence parallelism is not built (tap buffers are "
            "typed at the external rows); capture from a replicated-residual forward instead"
        )
        # Row kinds' block-exit linears land at the pass's residual row; when that IS the
        # external row this is the declaration itself, object-identically.
        mixer_placement = placement
        if placement is not None and sequence_parallel:
            assert residual_row is not None
            mixer_placement = replace(
                placement,
                target=replace(
                    placement.target, row=replace(placement.target.row, output=residual_row)
                ),
            )
        residual = constrain_activation(self._embed_tokens(tokens, placement), residual_row)

        sources = _capture_sources(ordered_capture_keys, cfg.n_layer)
        captured: dict[_CaptureSource, Array] = {}
        embedding_source = _CaptureSource(layer=0, tap=_Tap.RESIDUAL_IN)
        if embedding_source in sources:
            captured[embedding_source] = residual
        layout = _capture_layout(sources, cfg.n_layer)
        captured_taps = frozenset(_Tap(tap_value) for tap_value in layout)
        widths = self._tap_widths()

        def buffer(tap: str, n_slots: int) -> Array:
            shape = (n_slots, *residual.shape[:-1], widths[tap])
            dtype = _tap_dtype(_Tap(tap), residual.dtype)
            if placement is None:
                return jnp.zeros(shape, dtype)
            # a buffer's feature sharding matches the tap values written into it
            # (fused/hidden taps land tp-sharded at the intermediate row, model-width
            # taps replicated at the external row — the `[.., k]` routing taps ride
            # there too, feature axis unlisted = replicated over tp), so the in-scan
            # dynamic updates are layout-preserving; `from_producer` re-pins at exit.
            row = _tap_feature_row(placement, _Tap(tap))
            spec = row.spec_for(activation_axes(residual.ndim, "feature"))
            return jnp.zeros(
                shape,
                dtype,
                out_sharding=NamedSharding(placement.mesh, P(None, *spec)),
            )

        buffers = {
            tap: buffer(tap, sum(slot >= 0 for slot in slots)) for tap, slots in layout.items()
        }
        interval = cfg.full_attention_interval

        # Per-layer inputs enter the stage scan stage-BLOCKED: each leaf re-laid out
        # `[n_layer, …] → [n_stages, interval, …]`, a pure view of the resident stack,
        # with the interval positions split out INSIDE the body. Scan xs slicing is
        # the one indexing move with a `reduced`-typing rule, and the scan SAVES its
        # xs for the backward, outside every `jax.checkpoint` scope — so xs must be
        # views, never the pre-sliced position fragments (gather copies) that would
        # keep a second resident-stack-sized footprint alive across fwd→bwd. The
        # in-body split runs under the body's checkpoint, re-slicing at zero FLOPs in
        # the backward. `_stage_blocked`/`_block_positions` keep the entry-gather tags
        # forward AND provenance (`unreduced`) backward, so the component stacks'
        # deferred cross-`data` master reduction fires exactly once, at the
        # materialize boundary's transpose — never per fragment, never in the loop.
        moe_blocked = _stage_blocked_tree(self.moe, cfg.n_stages, interval)
        per_kind_blocked = {
            kind: {
                key: _stage_blocked_tree(value, cfg.n_stages, interval)
                for key, value in entry.items()
            }
            for kind, entry in per_kind.items()
        }
        slots_blocked = {
            tap: jnp.asarray(slots, jnp.int32).reshape(cfg.n_stages, interval)
            for tap, slots in layout.items()
        }

        def stage_body(
            state: tuple[Array, dict[str, Array]],
            stage_inputs: tuple[
                DeltaNetSublayer,
                AttnSublayer,
                FrozenMoE,
                dict[str, dict[str, SiteEntryTensor]],
                dict[str, Array],
            ],
        ) -> tuple[tuple[Array, dict[str, Array]], None]:
            x, stage_buffers = state
            deltanet_block, attn_stage, moe_block, per_kind_block, slots_block = stage_inputs
            deltanet_layers = _tree_by_position(deltanet_block, interval - 1)
            moe_layers = _tree_by_position(moe_block, interval)
            per_kind_layers = _tree_by_position(per_kind_block, interval)
            external = None if placement is None else placement.activations.external
            for pos in range(interval):
                # Mixers see the full external width (their sequence-mixing cores and
                # per-head tp splits demand every position); the normed residual gathers
                # here — one collective whose transpose is the mixer input's one
                # cotangent reduction — and the mixer's row output lands at the pass's
                # residual row.
                if pos < interval - 1:
                    sublayer = deltanet_layers[pos]
                    mixer_out = sublayer.mixer(
                        constrain_activation(
                            rms_norm_zero_centered(x, sublayer.ln1, cfg.rms_norm_eps), external
                        ),
                        mixer_placement,
                    )
                else:
                    mixer_out = attn_stage.attn(
                        constrain_activation(
                            rms_norm_zero_centered(x, attn_stage.ln1, cfg.rms_norm_eps), external
                        ),
                        self.inv_freq,
                        mixer_placement,
                    )
                post_mixer = x + mixer_out
                acts = self._moe_forward(
                    moe_layers[pos],
                    post_mixer,
                    per_kind_layers[pos],
                    uses_weight_deltas,
                    captured_taps,
                    placement,
                    residual_row,
                )
                x = acts.residual_out
                if stage_buffers:
                    stage_buffers = _write_captures(
                        stage_buffers,
                        {tap: slots_block[tap][pos] for tap in stage_buffers},
                        acts,
                    )
            return (x, stage_buffers), None

        body = (
            jax.checkpoint(stage_body, policy=checkpoint_policy)
            if checkpoint_policy is not None
            else stage_body
        )
        (residual, buffers), _ = jax.lax.scan(
            body,
            (residual, buffers),
            (self.deltanet, self.attn, moe_blocked, per_kind_blocked, slots_blocked),
        )

        for source in sources:
            if source.tap is not _Tap.RESIDUAL_IN:
                captured[source] = buffers[source.tap.value][layout[source.tap.value][source.layer]]
        residual = constrain_activation(
            rms_norm_zero_centered(residual, self.norm, cfg.rms_norm_eps),
            None if placement is None else placement.activations.external,
        )
        return ForwardResult.from_producer(
            output=self._output(residual, placement),
            capture_keys=ordered_capture_keys,
            capture_values=tuple(captured[source] for source in sources),
        )

    def _tap_widths(self) -> dict[str, int]:
        cfg = self.cfg
        fused = cfg.n_experts * cfg.moe_intermediate
        return {
            _Tap.MOE_INPUT.value: cfg.n_embd,
            _Tap.ROUTER_IDX.value: cfg.n_experts_per_token,
            _Tap.ROUTER_WEIGHTS.value: cfg.n_experts_per_token,
            _Tap.EXPERTS_GATE_OUTPUT.value: fused,
            _Tap.EXPERTS_UP_OUTPUT.value: fused,
            _Tap.EXPERTS_DOWN_OUTPUT.value: cfg.n_embd,
            _Tap.SHARED_GATE_OUTPUT.value: cfg.shared_expert_intermediate,
            _Tap.SHARED_UP_OUTPUT.value: cfg.shared_expert_intermediate,
            _Tap.SHARED_DOWN_OUTPUT.value: cfg.n_embd,
            _Tap.RESIDUAL_OUT.value: cfg.n_embd,
        }

    # ---------- protocol forwards ----------

    def clean_forward(
        self,
        inputs: Int[Array, "b t"],
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        *,
        placement: PlacementRules | None,
    ) -> ForwardResult[LMOutput]:
        return self._forward(
            inputs,
            per_kind={},
            ordered_capture_keys=tuple(sorted(capture_keys)),
            uses_weight_deltas=False,
            checkpoint_policy=None,
            placement=placement,
            residual_row=None if placement is None else placement.activations.external,
        )

    def prepare_compute_weights(
        self, vu: ComponentStacks, placement: PlacementRules | None
    ) -> dict[str, dict[str, Array]]:
        """Placed: the ÷data→resident entry gather runs ONCE per step here (off the hot
        path), landing ÷tp-resident stacks typed `reduced` over the gathered axes. The
        tags ride the scan xs into the stage loop (`_stage_blocked` and the in-body
        `_block_positions` keep them across the entry-side re-layout, provenance-
        preserving in both directions), so component-weight cotangents stay
        replica-local through the whole backward and reduce exactly once, at this
        boundary's transpose."""
        if placement is None:
            return self._stack_per_kind_vu(vu)
        return self._stack_per_kind_vu(
            component_stacks_to_compute_weights(vu, placement.components)
        )

    def _stack_per_kind_vu(self, components: ComponentStacks) -> dict[str, dict[str, Array]]:
        """Per decomposed KIND, the layer-stacked (V, U) — the group stacks directly,
        because canonical order makes each kind's slot axis THE layer axis. Whole-grid
        coverage per kind is this target's masked-forward contract (v1)."""
        assert components.site_names == self.site_names, (
            components.site_names,
            self.site_names,
        )
        for name, group, slot in components.site_slots:
            layer, kind = parse_site_name(name)
            assert group == kind, (group, name)
            assert slot == layer, (
                f"qwen36_moe requires each decomposed kind on EVERY layer (whole-grid "
                f"c-specs); {name} sits at slot {slot}"
            )
        lengths = components.group_lengths()
        for group, length in lengths.items():
            assert length == self.cfg.n_layer, (
                f"qwen36_moe requires each decomposed kind on EVERY layer (whole-grid "
                f"c-specs); {group!r} covers {length} of {self.cfg.n_layer} layers"
            )
        return {kind: {"V": Vs, "U": Us} for kind, (Vs, Us) in components.stacks.items()}

    def stack_ci(self, ci_lower: Mapping[str, SiteCI]) -> dict[str, SiteCI]:
        assert set(ci_lower) == set(self.site_names), (
            sorted(ci_lower),
            sorted(self.site_names),
        )
        kinds = {parse_site_name(name)[1] for name in ci_lower}
        # tree-stacked so a narrow kind's per-layer bundles stack leaf-wise (values AND
        # router indices; the static n_experts must agree across layers).
        return {
            kind: jax.tree.map(
                lambda *layers: jnp.stack(layers),
                *(ci_lower[site_name(layer, kind)] for layer in range(self.cfg.n_layer)),
            )
            for kind in kinds
        }

    def _attach_materialized(
        self,
        prepared_weights: dict[str, dict[str, Array]],
        masking: MaterializedMasking,
    ) -> tuple[dict[str, dict[str, SiteEntryTensor]], bool]:
        site_set = frozenset(self.site_names)
        assert set(masking.component_masks) == site_set, (
            sorted(masking.component_masks),
            sorted(site_set),
        )
        per_kind: dict[str, dict[str, SiteEntryTensor]] = {}
        for kind, entry in prepared_weights.items():
            names = [site_name(layer, kind) for layer in range(self.cfg.n_layer)]
            attached: dict[str, SiteEntryTensor] = {
                **entry,
                "mask": jax.tree.map(
                    lambda *layers: jnp.stack(layers),
                    *(masking.component_masks[name] for name in names),
                ),
            }
            if masking.weight_delta_masks is not None:
                attached["delta"] = jnp.stack([masking.weight_delta_masks[name] for name in names])
            if masking.routes is not None:
                attached["route"] = jnp.stack([masking.routes[name] for name in names])
            per_kind[kind] = attached
        return per_kind, masking.weight_delta_masks is not None

    def _attach_stochastic(
        self,
        prepared_weights: dict[str, dict[str, Array]],
        masking: StochasticMasking,
    ) -> dict[str, dict[str, SiteEntryTensor]]:
        """Attach the SHARED per-kind CI stack plus per-(kind, layer) draw keys; masks
        are rebuilt inside the checkpointed stage bodies (`_decomposed_site_output`)."""
        ci_stacked = masking.ci_stacked
        assert set(ci_stacked) == set(prepared_weights), (
            sorted(ci_stacked),
            sorted(prepared_weights),
        )
        src_base, delta_base = jax.random.split(masking.draw_key)
        per_kind: dict[str, dict[str, SiteEntryTensor]] = {}
        for kind, entry in prepared_weights.items():
            kind_index = KIND_ORDER.index(kind)
            layers = range(self.cfg.n_layer)
            attached: dict[str, SiteEntryTensor] = {
                **entry,
                "ci": ci_stacked[kind],
                "src_key": jnp.stack(
                    [
                        jax.random.fold_in(jax.random.fold_in(src_base, kind_index), layer)
                        for layer in layers
                    ]
                ),
                "delta_key": jnp.stack(
                    [
                        jax.random.fold_in(jax.random.fold_in(delta_base, kind_index), layer)
                        for layer in layers
                    ]
                ),
            }
            if masking.routes is not None:
                names = [site_name(layer, kind) for layer in layers]
                attached["route"] = jnp.stack([masking.routes[name] for name in names])
            per_kind[kind] = attached
        return per_kind

    def _attach_sources(
        self,
        prepared_weights: dict[str, dict[str, Array]],
        masking: SourceMasking,
    ) -> dict[str, dict[str, SiteEntryTensor]]:
        """Attach the SHARED per-kind CI stacks plus the per-kind source-value and
        delta stacks; masks are recomposed inside the checkpointed stage bodies
        (`_entry_masking`), so no per-draw mask stack outlives its block."""
        ci_stacked = masking.ci_stacked
        src_stacked = masking.source_values_stacked
        delta_stacked = masking.delta_values_stacked
        for stacked in (ci_stacked, src_stacked, delta_stacked):
            assert set(stacked) == set(prepared_weights), (
                sorted(stacked),
                sorted(prepared_weights),
            )
        per_kind: dict[str, dict[str, SiteEntryTensor]] = {}
        for kind, entry in prepared_weights.items():
            attached: dict[str, SiteEntryTensor] = {
                **entry,
                "ci": ci_stacked[kind],
                "src": src_stacked[kind],
                "delta_src": delta_stacked[kind],
            }
            if masking.routes is not None:
                names = [site_name(layer, kind) for layer in range(self.cfg.n_layer)]
                attached["route"] = jnp.stack([masking.routes[name] for name in names])
            per_kind[kind] = attached
        return per_kind

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
    ) -> ForwardResult[LMOutput]:
        site_set = frozenset(self.site_names)
        match masking:
            case StochasticMasking(routes=routes):
                assert routes is None or set(routes) == site_set, (
                    sorted(routes or {}),
                    sorted(site_set),
                )
                per_kind = self._attach_stochastic(prepared_weights, masking)
                uses_weight_deltas = True
            case SourceMasking(routes=routes):
                assert routes is None or set(routes) == site_set, (
                    sorted(routes or {}),
                    sorted(site_set),
                )
                per_kind = self._attach_sources(prepared_weights, masking)
                uses_weight_deltas = True
            case MaterializedMasking(routes=routes):
                assert routes is None or set(routes) == site_set, (
                    sorted(routes or {}),
                    sorted(site_set),
                )
                per_kind, uses_weight_deltas = self._attach_materialized(prepared_weights, masking)
        policy = (
            jax.checkpoint_policies.nothing_saveable
            if remat
            else jax.checkpoint_policies.dots_saveable
        )
        return self._forward(
            inputs,
            per_kind=per_kind,
            ordered_capture_keys=tuple(sorted(capture_keys)),
            uses_weight_deltas=uses_weight_deltas,
            checkpoint_policy=policy,
            placement=placement,
            residual_row=None if placement is None else placement.activations.masked_external,
        )

    def component_activation_forward(
        self,
        prepared_weights: dict[str, dict[str, Array]],
        inputs: Int[Array, "b t"],
        /,
        *,
        sites: tuple[str, ...],
        capture_keys: CaptureKeys,
        placement: PlacementRules | None,
    ) -> tuple[ForwardResult[LMOutput], dict[str, SiteCI]]:
        """Run one frozen forward for requested captures and each requested site's
        ``x @ V``: full `[.., C]` for the shared sites; for the expert sites a `NarrowCI`
        bundle of routed slots on the SAME captured routing the CI fn reads (slot m is
        the token's m-th routed expert), computed as grouped matmuls in job space. Down
        sites consume the FROZEN-path routed hidden (recomputed from the captured MoE
        input — the clean forward's own values, matching the GLU pattern of measuring
        activations against frozen-path site inputs)."""
        assert set(sites) <= set(self.site_names), (sorted(sites), self.site_names)
        cfg = self.cfg
        locations = tuple(parse_site_name(site) for site in sites)
        needed: set[str] = set()
        for layer, kind in locations:
            match kind:
                case "experts_gate" | "experts_up" | "experts_down":
                    needed.add(mlp_input_tap_key(layer))
                    needed.add(router_idx_tap_key(layer))
                    needed.add(router_weights_tap_key(layer))
                case "shared_gate" | "shared_up":
                    needed.add(mlp_input_tap_key(layer))
                case "shared_down":
                    needed.add(site_output_tap_key(site_name(layer, "shared_gate")))
                    needed.add(site_output_tap_key(site_name(layer, "shared_up")))
                case _:
                    raise AssertionError(kind)
        full = self.clean_forward(inputs, capture_keys | frozenset(needed), placement=placement)
        captures = full.captures

        @cache
        def expert_context(layer: int) -> _UnplacedExpertActivations | _PlacedExpertActivations:
            return _expert_activation_context(
                self.moe, cfg, captures, layer, placement, self.grouped_matmul_backend
            )

        component_activations: dict[str, SiteCI] = {}
        for site, (layer, kind) in zip(sites, locations, strict=True):
            V = unreduce(prepared_weights[kind]["V"])[layer]
            if is_expert_kind(kind):
                context = expert_context(layer)
                input_jobs = (
                    context.down_input() if kind == "experts_down" else context.x_jobs
                ).astype(V.dtype)
                component_activations[site] = context.narrow_values(input_jobs, V)
                continue
            match kind:
                case "shared_gate" | "shared_up":
                    site_input = captures[mlp_input_tap_key(layer)]
                case "shared_down":
                    gate = captures[site_output_tap_key(site_name(layer, "shared_gate"))]
                    up = captures[site_output_tap_key(site_name(layer, "shared_up"))]
                    site_input = jax.nn.silu(gate) * up
                case _:
                    raise AssertionError(kind)
            site_input = site_input.astype(V.dtype)
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
        requested = ForwardResult(
            output=full.output,
            captures={key: captures[key] for key in sorted(capture_keys)},
        )
        return requested, component_activations

    # ---------- frozen-weight views ----------

    def _frozen_kind_stack(self, kind: str) -> Array:
        """One kind's fused frozen matrices, layer-stacked `[n_layer, d_out, d_in]`."""
        match kind:
            case "experts_gate":
                return self.moe.experts_gate
            case "experts_up":
                return self.moe.experts_up
            case "experts_down":
                return self.moe.experts_down
            case "shared_gate":
                return self.moe.shared_gate
            case "shared_up":
                return self.moe.shared_up
            case "shared_down":
                return self.moe.shared_down
            case _:
                raise AssertionError(f"unknown kind {kind!r}")

    def target_weight_sq_norms(self) -> dict[str, Array]:
        groups = {group for _name, group, _slot in site_slots_for(self.sites)}
        return {
            group: jnp.sum(self._frozen_kind_stack(group).astype(jnp.float32) ** 2, axis=(1, 2))
            for group in sorted(groups)
        }

    def weight_deltas(self, vu: ComponentStacks) -> dict[str, Array]:
        """fp32 `W − V@U` per kind stack (slot axis = layer axis). Shared
        kinds stack fused `[g, d_out, d_in]`; expert kinds stack per-expert blocks
        `[g, expert, d_out, d_in]` — the blocks partition the fused matrix, so per-slot
        Frobenius reductions agree.

        The frozen stack keeps its resident dtype until it has landed on the delta row
        and taken the blocked layout; the fp32 convert commutes exactly with slicing
        and relayout, and happens at the subtract, where it fuses. Converting the
        resident stack first would pin a whole-stack fp32 copy behind the barrier."""
        for name, group, slot in vu.site_slots:
            layer, kind = parse_site_name(name)
            assert group == kind and slot == layer, (name, group, slot)

        def landed(frozen: Array, spec: P) -> Array:
            # Land the frozen stack PIECE-WISE on the delta row, derived from the
            # faithfulness operands' own typing: the sharded contraction reduce-scatters
            # onto d_in instead of forcing an ambiguous (and otherwise
            # full-master-gathering) output. Materialize the stack BEFORE the reshard:
            # without the barrier GSPMD propagates the delta layout backward through
            # the frozen slices and lowers them as cross-node redistribution.
            mesh = value_mesh(frozen)
            if mesh.empty:
                return frozen
            return jax.sharding.reshard(
                jax.lax.optimization_barrier(frozen), NamedSharding(mesh, spec)
            )

        def product(subscripts: str, v32: Array, u32: Array, spec: P) -> Array:
            mesh = value_mesh(v32)
            if mesh.empty:
                return jnp.einsum(subscripts, v32, u32)
            return jnp.einsum(subscripts, v32, u32, out_sharding=NamedSharding(mesh, spec))

        out: dict[str, Array] = {}
        for group, (Vs, Us) in vu.stacks.items():
            frozen = self._frozen_kind_stack(group)
            if pad := vu.pad_of(group):
                # Persist-stack pad slots decompose a ZERO matrix: their deltas ride the
                # faithfulness lane as exact zeros (pad V/U are zero by invariant) and
                # exit at the loss reduction. The pad rows take the frozen stack's own
                # sharding — explicit-mode concatenate demands matching operand specs.
                zeros = jnp.zeros((pad, *frozen.shape[1:]), frozen.dtype)
                if not value_mesh(frozen).empty:
                    zeros = jax.sharding.reshard(zeros, jax.typeof(frozen).sharding)
                frozen = jnp.concatenate([frozen, zeros])
            v32, u32 = Vs.astype(jnp.float32), Us.astype(jnp.float32)
            # Off-mesh operands type as full-rank `P(None, ...)`, so the specs read the
            # same way placed or not; `landed` / `product` are the identity off-mesh.
            v_spec = jax.typeof(Vs).sharding.spec
            u_spec = jax.typeof(Us).sharding.spec
            match site_contraction(group):
                case None:
                    # The C contraction all-reduces rather than scattering onto d_in:
                    # the moe preset's one delta row cannot carry tp on d_in (an expert
                    # delta holds expert+d_in together), so it admits d_in extents the
                    # full-C scatter would refuse. Shared-kind deltas are small and the
                    # engine constrains to the delta row right after.
                    delta_spec = P(v_spec[0], u_spec[2], None)
                    out[group] = landed(frozen, delta_spec).astype(jnp.float32) - product(
                        "gic,gco->goi", v32, u32, delta_spec
                    )
                case "fused_output":
                    n_layer, n_experts = Vs.shape[0], Vs.shape[1]
                    blocks = frozen.reshape(n_layer, n_experts, -1, frozen.shape[-1])
                    delta_spec = P(v_spec[0], v_spec[1], u_spec[3], v_spec[3])
                    out[group] = landed(blocks, delta_spec).astype(jnp.float32) - product(
                        "geic,geco->geoi", v32, u32, delta_spec
                    )
                case "fused_input":
                    n_layer, n_experts = Vs.shape[0], Vs.shape[1]
                    # Splitting the fused input axis into (expert, intermediate) is a
                    # bitcast of the resident stack; the expert-major transpose is the
                    # one relayout, and it runs on the landed piece.
                    split = frozen.reshape(n_layer, frozen.shape[1], n_experts, -1)
                    split_spec = P(v_spec[0], u_spec[3], v_spec[1], v_spec[3])
                    delta_spec = P(v_spec[0], v_spec[1], u_spec[3], v_spec[3])
                    blocks = landed(split, split_spec).transpose(0, 2, 1, 3)
                    out[group] = blocks.astype(jnp.float32) - product(
                        "geic,geco->geoi", v32, u32, delta_spec
                    )
        return out


# ----------------------------- build / HF loading -----------------------------


def build_qwen36_moe_model(
    cfg: Qwen36MoeConfig,
    sites: tuple[SiteSpec, ...],
    *,
    embed: Array,
    deltanet_sublayers: list[DeltaNetSublayer],
    attn_sublayers: list[AttnSublayer],
    moe_layers: list[FrozenMoE],
    norm: Array,
    lm_head: Array,
    stack: Callable[[Sequence[Array]], Array],
    grouped_matmul_backend: GroupedMatmulBackend,
) -> Qwen36MoeDecomposedModel:
    """Assemble the model from per-layer parts (mixer sublayers in layer order within
    their own kinds; `moe_layers` in layer order) into the stage-stacked storage.
    `sites` must be the canonical specs for this config.

    `stack` is the leaf-stacking primitive, and thereby fixes where the assembled
    leaves live: the HF loader passes `np.stack` so the full frozen target stays on
    HOST — `place_via_shardings` serves each device only its shard, and no device ever
    holds a whole-model stopover (a ~70 GiB transit through the process's default GPU
    would fragment that one device's allocator pool out of fitting the step arena its
    siblings fit). Tracing and toy callers pass `jnp.stack`."""
    site_cs = tuple(SiteC(spec.name, spec.C) for spec in sites)
    expected = qwen36_moe_site_specs(cfg, canonical_site_cs(site_cs))
    assert sites == expected, f"sites are not the canonical specs for this config: {sites}"
    interval = cfg.full_attention_interval
    assert len(deltanet_sublayers) == cfg.n_stages * (interval - 1), len(deltanet_sublayers)
    assert len(attn_sublayers) == cfg.n_stages, len(attn_sublayers)
    assert len(moe_layers) == cfg.n_layer, len(moe_layers)

    deltanet_flat = jax.tree.map(lambda *leaves: stack(leaves), *deltanet_sublayers)
    deltanet = jax.tree.map(
        lambda a: a.reshape(cfg.n_stages, interval - 1, *a.shape[1:]), deltanet_flat
    )
    attn = jax.tree.map(lambda *leaves: stack(leaves), *attn_sublayers)
    moe = jax.tree.map(lambda *leaves: stack(leaves), *moe_layers)
    return Qwen36MoeDecomposedModel(
        embed=embed,
        deltanet=deltanet,
        attn=attn,
        moe=moe,
        norm=norm,
        lm_head=lm_head,
        inv_freq=default_inv_freq(cfg.rotary_dim, cfg.rope_theta),
        cfg=cfg,
        sites=sites,
        has_position_axis=True,
        experts_execution="routed",
        grouped_matmul_backend=grouped_matmul_backend,
        output_edge=MaterializedOutputEdge(),
    )


WeightGetter = Callable[[str], Array]
"""`key -> host array` — the HF safetensors reader in production, an in-memory state
dict in the parity tests."""


def _load_moe(get: WeightGetter, prefix: str, cfg: Qwen36MoeConfig) -> FrozenMoE:
    di, d = cfg.moe_intermediate, cfg.n_embd
    fused = cfg.n_experts * di
    # HF stores the routed experts as [E, 2·di, d] (gate rows first) and [E, d, di].
    gate_up = get(f"{prefix}.mlp.experts.gate_up_proj")
    assert gate_up.shape == (cfg.n_experts, 2 * di, d), gate_up.shape
    down = get(f"{prefix}.mlp.experts.down_proj")
    assert down.shape == (cfg.n_experts, d, di), down.shape
    return FrozenMoE(
        ln=get(f"{prefix}.post_attention_layernorm.weight"),
        router=get(f"{prefix}.mlp.gate.weight"),
        experts_gate=gate_up[:, :di, :].reshape(fused, d),
        experts_up=gate_up[:, di:, :].reshape(fused, d),
        experts_down=down.transpose(1, 0, 2).reshape(d, fused),
        shared_gate=get(f"{prefix}.mlp.shared_expert.gate_proj.weight"),
        shared_up=get(f"{prefix}.mlp.shared_expert.up_proj.weight"),
        shared_down=get(f"{prefix}.mlp.shared_expert.down_proj.weight"),
        shared_expert_gate=get(f"{prefix}.mlp.shared_expert_gate.weight"),
    )


def _load_deltanet(get: WeightGetter, prefix: str, cfg: Qwen36MoeConfig) -> DeltaNetSublayer:
    conv_weight = get(f"{prefix}.linear_attn.conv1d.weight")
    kd, vd = cfg.linear_key_dim, cfg.linear_value_dim
    conv_dim = 2 * kd + vd
    assert conv_weight.shape == (conv_dim, 1, cfg.linear_conv_kernel_dim), conv_weight.shape
    conv = conv_weight.reshape(conv_dim, cfg.linear_conv_kernel_dim)
    # HF fuses q|k|v rows in the in_proj and its conv channels; stored split here
    # (bit-identical per-piece compute — `FrozenGatedDeltaNet`).
    w_qkv = get(f"{prefix}.linear_attn.in_proj_qkv.weight")
    assert w_qkv.shape[0] == conv_dim, w_qkv.shape
    return DeltaNetSublayer(
        ln1=get(f"{prefix}.input_layernorm.weight"),
        mixer=FrozenGatedDeltaNet(
            w_q=w_qkv[:kd],
            w_k=w_qkv[kd : 2 * kd],
            w_v=w_qkv[2 * kd :],
            w_z=get(f"{prefix}.linear_attn.in_proj_z.weight"),
            w_b=get(f"{prefix}.linear_attn.in_proj_b.weight"),
            w_a=get(f"{prefix}.linear_attn.in_proj_a.weight"),
            conv_q=conv[:kd],
            conv_k=conv[kd : 2 * kd],
            conv_v=conv[2 * kd :],
            a_log=get(f"{prefix}.linear_attn.A_log"),
            dt_bias=get(f"{prefix}.linear_attn.dt_bias"),
            norm_w=get(f"{prefix}.linear_attn.norm.weight"),
            w_out=get(f"{prefix}.linear_attn.out_proj.weight"),
            n_k_heads=cfg.linear_num_key_heads,
            n_v_heads=cfg.linear_num_value_heads,
            k_head_dim=cfg.linear_key_head_dim,
            v_head_dim=cfg.linear_value_head_dim,
            eps=cfg.rms_norm_eps,
        ),
    )


def _load_attn(
    get: WeightGetter,
    prefix: str,
    cfg: Qwen36MoeConfig,
    implementation: AttentionImplementation,
) -> AttnSublayer:
    return AttnSublayer(
        ln1=get(f"{prefix}.input_layernorm.weight"),
        attn=FrozenGatedAttention(
            wq=get(f"{prefix}.self_attn.q_proj.weight"),
            wk=get(f"{prefix}.self_attn.k_proj.weight"),
            wv=get(f"{prefix}.self_attn.v_proj.weight"),
            wo=get(f"{prefix}.self_attn.o_proj.weight"),
            q_norm=get(f"{prefix}.self_attn.q_norm.weight"),
            k_norm=get(f"{prefix}.self_attn.k_norm.weight"),
            n_head=cfg.n_head,
            n_kv_head=cfg.n_kv_head,
            head_dim=cfg.head_dim,
            eps=cfg.rms_norm_eps,
            implementation=implementation,
        ),
    )


def build_qwen36_moe_from_weights(
    cfg: Qwen36MoeConfig,
    sites: tuple[SiteSpec, ...],
    get: WeightGetter,
    *,
    decoder_prefix: str,
    implementation: AttentionImplementation = "auto",
) -> Qwen36MoeDecomposedModel:
    """Build from HF-keyed weights: `decoder_prefix` is `model.language_model` in the
    real `Qwen3_5MoeForConditionalGeneration` checkpoint and `model` under the text-only
    `Qwen3_5MoeForCausalLM` (the parity fixtures). `get` must return HOST arrays
    (`HFWeights.get` does) and the assembly stacks host-wise, so the built model's
    leaves never touch a device before placement."""
    deltanet_sublayers: list[DeltaNetSublayer] = []
    attn_sublayers: list[AttnSublayer] = []
    moe_layers: list[FrozenMoE] = []
    for layer in range(cfg.n_layer):
        prefix = f"{decoder_prefix}.layers.{layer}"
        if layer_is_full_attention(cfg, layer):
            attn_sublayers.append(_load_attn(get, prefix, cfg, implementation))
        else:
            deltanet_sublayers.append(_load_deltanet(get, prefix, cfg))
        moe_layers.append(_load_moe(get, prefix, cfg))
    return build_qwen36_moe_model(
        cfg,
        sites,
        embed=get(f"{decoder_prefix}.embed_tokens.weight"),
        deltanet_sublayers=deltanet_sublayers,
        attn_sublayers=attn_sublayers,
        moe_layers=moe_layers,
        norm=get(f"{decoder_prefix}.norm.weight"),
        lm_head=get("lm_head.weight"),
        stack=cast(Callable[[Sequence[Array]], Array], np.stack),
        grouped_matmul_backend=GROUPED_MATMUL_BACKEND,
    )


def abstract_qwen36_moe_model(
    cfg: Qwen36MoeConfig, sites: tuple[SiteSpec, ...], weights_dtype: DTypeLike
) -> Qwen36MoeDecomposedModel:
    """The model as SHAPES only — `ShapeDtypeStruct` leaves in the stage-stacked
    storage, via `eval_shape` over the real assembly. For weightless placement and
    trace gates (`fit_check.abstract_placed_model` consumes it); the loaders' shape
    asserts pin it to the checkpoint's layout."""
    d = cfg.n_embd
    kd, vd = cfg.linear_key_dim, cfg.linear_value_dim
    kernel = cfg.linear_conv_kernel_dim
    vh = cfg.linear_num_value_heads
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    fused = cfg.n_experts * cfg.moe_intermediate
    si = cfg.shared_expert_intermediate

    def zeros(*shape: int) -> Array:
        return jnp.zeros(shape, weights_dtype)

    def build() -> Qwen36MoeDecomposedModel:
        deltanet = DeltaNetSublayer(
            ln1=zeros(d),
            mixer=FrozenGatedDeltaNet(
                w_q=zeros(kd, d),
                w_k=zeros(kd, d),
                w_v=zeros(vd, d),
                w_z=zeros(vd, d),
                w_b=zeros(vh, d),
                w_a=zeros(vh, d),
                conv_q=zeros(kd, kernel),
                conv_k=zeros(kd, kernel),
                conv_v=zeros(vd, kernel),
                a_log=zeros(vh),
                dt_bias=zeros(vh),
                norm_w=zeros(cfg.linear_value_head_dim),
                w_out=zeros(d, vd),
                n_k_heads=cfg.linear_num_key_heads,
                n_v_heads=cfg.linear_num_value_heads,
                k_head_dim=cfg.linear_key_head_dim,
                v_head_dim=cfg.linear_value_head_dim,
                eps=cfg.rms_norm_eps,
            ),
        )
        attn = AttnSublayer(
            ln1=zeros(d),
            attn=FrozenGatedAttention(
                wq=zeros(2 * qd, d),
                wk=zeros(kvd, d),
                wv=zeros(kvd, d),
                wo=zeros(d, qd),
                q_norm=zeros(cfg.head_dim),
                k_norm=zeros(cfg.head_dim),
                n_head=cfg.n_head,
                n_kv_head=cfg.n_kv_head,
                head_dim=cfg.head_dim,
                eps=cfg.rms_norm_eps,
                implementation="auto",
            ),
        )
        moe = FrozenMoE(
            ln=zeros(d),
            router=zeros(cfg.n_experts, d),
            experts_gate=zeros(fused, d),
            experts_up=zeros(fused, d),
            experts_down=zeros(d, fused),
            shared_gate=zeros(si, d),
            shared_up=zeros(si, d),
            shared_down=zeros(d, si),
            shared_expert_gate=zeros(1, d),
        )
        n_deltanet = cfg.n_stages * (cfg.full_attention_interval - 1)
        return build_qwen36_moe_model(
            cfg,
            sites,
            embed=zeros(cfg.vocab_size, d),
            deltanet_sublayers=[deltanet] * n_deltanet,
            attn_sublayers=[attn] * cfg.n_stages,
            moe_layers=[moe] * cfg.n_layer,
            norm=zeros(d),
            lm_head=zeros(cfg.vocab_size, d),
            stack=jnp.stack,
            grouped_matmul_backend=GROUPED_MATMUL_BACKEND,
        )

    return eqx.filter_eval_shape(build)


def load_decomposed_qwen36_moe_from_hf(
    model_name: str,
    cfg: Qwen36MoeConfig,
    sites: tuple[SiteSpec, ...],
    weights_dtype: DTypeLike,
    implementation: AttentionImplementation,
) -> Qwen36MoeDecomposedModel:
    """Load from the cached HF snapshot's safetensors (no torch). The vision tower and
    MTP head in the checkpoint are simply never read."""
    weights = HFWeights(hf_snapshot_dir(model_name), weights_dtype)
    return build_qwen36_moe_from_weights(
        cfg,
        sites,
        weights.get,
        decoder_prefix="model.language_model",
        implementation=implementation,
    )
