"""The Qwen3.6-MoE architecture target (HF `qwen3_5_moe`; concrete support:
`Qwen/Qwen3.6-35B-A3B` from its HF snapshot, and the lab-pretrained `Qwen35Moe` toys from
their pretrain-cache entry) — the first MoE and first hybrid-attention slice, on its own
engine rather than `transformer` (the hybrid layer schedule and the MoE waist share
no code shape with the dense transformer stack).

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

SITES. The decomposable projections are the eleven token-mixer matrices and
the six MoE matrices, each on every layer that HAS it — the gated-DeltaNet kinds
(`layers.{i}.linear_attn.{in_proj_qkv.{q,k,v}|in_proj_z|in_proj_b|in_proj_a|out_proj}`,
the q/k/v being the row blocks of HF's fused `in_proj_qkv`) on the linear-attention
layers, the gated-attention kinds (`layers.{i}.self_attn.{q,k,v,o}_proj`) on the
full-attention layers, and the MoE kinds on all. Conv kernels, norms, `A_log`/`dt_bias`, the router and the shared-expert scalar gate
stay frozen. The MoE sites carry the expert axis
STRUCTURALLY: `layers.{i}.mlp.experts.{gate,up,down}_proj` is the FUSED all-expert
matrix (gate/up `[E·di, d]`, down `[d, E·di]` — one honest linear map, the dense
equivalent of the routed MoE), and `layers.{i}.mlp.shared_expert.{gate,up,down}_proj`
the shared expert's. Routing weights fold into the fused down site's INPUT
(`w_e · hidden_e`, exactly `w_e` applied after each expert's down in exact arithmetic),
so an unrouted expert's component activations are exactly zero. The router and the
shared-expert scalar gate stay frozen. A decomposed kind covers every layer that has
it (`layers_of_kind`); a kind's V/U stack slot s is the s-th such layer.

ROUTING. The router's decision is an explicit noun, `components.BlockSelection` (per
layer: which k experts each token ran, and the fp32 mixing weights), built from three
public verbs — `router_probs` (fp32 softmax over all experts), `select_experts` (top-k),
`expert_mixing_weights` (gather at given indices, renormalize). The CLEAN forward
decides it and returns it as the result's `conditioning`; every MASKED forward REQUIRES it:
its experts retain the clean indices at every layer — an expert-blocked site's
mask slot m means expert `indices[l, .., m]`, so masks and live experts cannot
misalign — and its mixing weights are recomputed from its own residual's softmax at
those indices. Nothing re-routes on a masked residual, and routing is never read off
captures, masks, or CI values.

EXECUTION. Expert computation explicitly chooses dense masked matmuls over all
experts or routed grouped matmuls over selected experts only. This choice applies to
frozen, decomposed, and activation-capture forwards; expert CI remains selected.
Wherever no `experts_*` kind is decomposed, the frozen arm weights the down output
in fp32. A forward decomposing any expert kind contracts its V_e/U_e through the
C_block bottleneck with selected masks and the frozen delta/route channels, folding
routing weights into the down-site input. The dense implementation uses token/expert
tables; routed implementations use selected jobs.
Expert masks use token-ordered `SelectedCI` under every placement; shared kinds
take full `[.., C]` arrays. The shared
expert is always dense. The layer stack runs as one `lax.scan` over stages (the
periodic unit), each stage body unrolling its `interval` sublayers. Each decomposed
kind must cover every layer (whole-grid c-specs) — an enumerated gap, asserted loudly.

PLACEMENT (Plan A). On the two-axis `(data, tp)` mesh under the
`zero1-replicated-resident-moe` preset, every frozen weight persists at its operand
layout — expert-major fused axes and mixer heads ÷tp within the node; the router,
embeddings, and norms replicated, and the KV projections replicated wherever tp does
not split their heads whole (the 35B's 2 KV heads at tp=8) — so no while body gathers
a weight. The batch
shards over `data`; activations replicate over `tp` at the residual waist and shard
over it at the fused/hidden widths (Megatron column/row pairs). The routed frozen
expert arm becomes EP by activation slicing (`routed.experts.ExpertShardedJobs`): each
rank computes only its expert shard's jobs, and partial outputs reduce over `tp` in the
fp32 combine. V/U expert blocks co-locate with their frozen experts (`expert: tp`).
"""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import cache, partial
from pathlib import Path
from typing import Literal, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from beartype import beartype
from jax.extend.backend import get_default_device
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped
from safetensors import safe_open

from param_decomp.attention import AttentionImplementation, jax_attention_implementation
from param_decomp.core import family
from param_decomp.core.axes import Axes, MeshAxis, SemanticAxis
from param_decomp.core.components import (
    BlockedFactorization,
    BlockSelection,
    ComponentStacks,
    DenseFactorization,
    Factorization,
    SelectedCI,
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
    BlockedPlannedComponentLinear,
    BlockedSiteWeights,
    PlannedComponentLinear,
    SiteWeights,
    blend_target_output,
    blocked_site_forward,
    component_coefficients,
    site_out,
)
from param_decomp.core.family import ArchFamily
from param_decomp.core.flops.target import (
    GradientTarget,
    TargetPassFlops,
    causal_attention_flops,
    frozen_product_flops,
    linear_flops,
    sum_target_flops,
    validate_batch,
)
from param_decomp.core.flops.types import ForwardBackwardFlops
from param_decomp.core.linear_plan import (
    BlockContraction,
    placed_linear,
    unreduce,
    value_mesh,
)
from param_decomp.core.masking import sample_component_mask, sample_delta_mask
from param_decomp.core.model import (
    EMPTY_CAPTURE_KEYS,
    CaptureKeys,
    ForwardResult,
    Masking,
    MaterializedMasking,
    SiteRoutes,
    SourceMasking,
    StochasticMasking,
    validate_routes,
)
from param_decomp.core.nonlinearity import (
    ComponentSide,
    DeltaNetHeads,
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
from param_decomp.lm.batch import LMBatch, LMBatchWithRouting
from param_decomp.routed.dense import (
    dense_combine_experts,
    dense_expert_matmul,
    dense_select_experts,
)
from param_decomp.routed.experts import (
    ExpertImplementation,
    ExpertShardedJobs,
    GroupedMatmulBackend,
    RoutedJobs,
    combine_jobs,
    ep_combine_jobs,
    ep_gather_tokens,
    ep_grouped_matmul,
    ep_sort_job_values,
    ep_sort_jobs,
    ep_sum_jobs,
    ep_unsort_jobs,
    expert_sharded_jobs,
    gather_tokens,
    grouped_matmul,
    routed_jobs,
    scatter_jobs,
    sort_job_values,
    sort_jobs,
    sum_jobs,
    unsort_jobs,
)
from param_decomp.target_ports.llama import rope_cos_sin
from param_decomp.target_ports.qwen3_5_moe import (
    GatedDeltaKernel,
    apply_partial_rope,
    causal_depthwise_conv1d_silu,
    gated_delta_rule,
    gated_rms_norm,
    rms_norm_zero_centered,
)
from param_decomp.targets.host import cpu_staging
from param_decomp.targets.lm_output import (
    LMOutput,
    OutputEdge,
    linear_output,
    pin_lm_output_batch,
)
from param_decomp.targets.losses import lm_output_kl_per_position
from param_decomp.targets.transformer import (
    HFWeights,
    default_inv_freq,
    hf_snapshot_dir,
    nonlinearity_aligned_factors,
)
from param_decomp.targets.transformer_taps import (
    BlockCaptures,
    BlockTap,
    ResidualBoundary,
    SiteOutput,
    TransformerTapGrammar,
    attention_input_tap_key,
    attention_output_tap_key,
    mlp_input_tap_key,
    site_output_tap_key,
)

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

# The decomposed matrix vocabulary, in a layer's computation order: the two token mixers
# (a layer carries one of them), then the MoE waist. The expert axis is structural
# INSIDE the `experts_*` sites (fused all-expert matrices).
Qwen36MoeMatrix = Literal[
    "gdn_q",
    "gdn_k",
    "gdn_v",
    "gdn_z",
    "gdn_b",
    "gdn_a",
    "gdn_out",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_o",
    "experts_gate",
    "experts_up",
    "experts_down",
    "shared_gate",
    "shared_up",
    "shared_down",
]

KIND_ORDER: tuple[str, ...] = get_args(Qwen36MoeMatrix)
"""Within-layer canonical site order = computation order (mixer, routed experts, then
the shared expert), DERIVED from the `Qwen36MoeMatrix` vocabulary."""

Sublayer = Literal["deltanet", "attn", "moe"]
"""The three sublayer kinds a site can belong to. A layer has an MoE and exactly ONE of
the two mixers (`layer_is_full_attention`), so which layers carry a kind is a fact of
the architecture, never a config choice."""


def sublayer_of(kind: str) -> Sublayer:
    match kind:
        case "gdn_q" | "gdn_k" | "gdn_v" | "gdn_z" | "gdn_b" | "gdn_a" | "gdn_out":
            return "deltanet"
        case "attn_q" | "attn_k" | "attn_v" | "attn_o":
            return "attn"
        case (
            "experts_gate"
            | "experts_up"
            | "experts_down"
            | "shared_gate"
            | "shared_up"
            | "shared_down"
        ):
            return "moe"
        case _:
            raise AssertionError(f"unknown kind {kind!r}")


def layers_of_kind(cfg: Qwen36MoeConfig, kind: str) -> tuple[int, ...]:
    """The layers that HAVE a kind's matrix, ascending — the slot order of the kind's
    V/U stack (slot s is the s-th such layer)."""
    match sublayer_of(kind):
        case "deltanet":
            return tuple(
                layer for layer in range(cfg.n_layer) if not layer_is_full_attention(cfg, layer)
            )
        case "attn":
            return tuple(
                layer for layer in range(cfg.n_layer) if layer_is_full_attention(cfg, layer)
            )
        case "moe":
            return tuple(range(cfg.n_layer))


_GDN_KINDS = frozenset(kind for kind in KIND_ORDER if sublayer_of(kind) == "deltanet")
_ATTN_KINDS = frozenset(kind for kind in KIND_ORDER if sublayer_of(kind) == "attn")

_EXPERT_KINDS = frozenset({"experts_gate", "experts_up", "experts_down"})
"""The kinds sharing one expert arm: decomposing ANY of them switches that arm from
routed-frozen to the routed DECOMPOSED execution (undecomposed expert kinds inside it
run their frozen grouped matmuls on the same jobs schedule)."""


_SITE_SUFFIX: dict[str, str] = {
    "gdn_q": "linear_attn.in_proj_qkv.q",
    "gdn_k": "linear_attn.in_proj_qkv.k",
    "gdn_v": "linear_attn.in_proj_qkv.v",
    "gdn_z": "linear_attn.in_proj_z",
    "gdn_b": "linear_attn.in_proj_b",
    "gdn_a": "linear_attn.in_proj_a",
    "gdn_out": "linear_attn.out_proj",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_o": "self_attn.o_proj",
    "experts_gate": "mlp.experts.gate_proj",
    "experts_up": "mlp.experts.up_proj",
    "experts_down": "mlp.experts.down_proj",
    "shared_gate": "mlp.shared_expert.gate_proj",
    "shared_up": "mlp.shared_expert.up_proj",
    "shared_down": "mlp.shared_expert.down_proj",
}
"""Each kind's HF module path below `layers.{i}.`. The DeltaNet q/k/v are the row
blocks of HF's one fused `in_proj_qkv` (stored split here), spelled as sub-paths of
that module."""
assert tuple(_SITE_SUFFIX) == KIND_ORDER, (tuple(_SITE_SUFFIX), KIND_ORDER)

_KIND_OF_SUFFIX = {suffix: kind for kind, suffix in _SITE_SUFFIX.items()}

_SITE_NAME_PATTERN = re.compile(r"^layers\.(\d+)\.(.+)$")


def site_name(layer: int, kind: str) -> str:
    return f"layers.{layer}.{_SITE_SUFFIX[kind]}"


def router_logits_tap_key(layer: int) -> str:
    """The router scores before softmax, captured in fp32."""
    return _router_tap_key(_Tap.ROUTER_LOGITS, layer)


def router_probs_tap_key(layer: int) -> str:
    """Layer `layer`'s router softmax over ALL experts `[.., E]` (fp32) — the tensor the
    forward routes (clean) or weights (masked) from, so on the masked path it is the
    perturbed residual's router view."""
    return _router_tap_key(_Tap.ROUTER_PROBS, layer)


def router_weights_tap_key(layer: int) -> str:
    """Layer `layer`'s applied mixing weights `[.., k]` (fp32, renormalized) — the one
    per-forward routing quantity the conditioning `BlockSelection` cannot carry: the clean pass's
    are its own, the masked pass's are recomputed at the pinned indices."""
    return _router_tap_key(_Tap.ROUTER_WEIGHTS, layer)


# ----------------------------- the routing verbs -----------------------------
# HF `Qwen3_5MoeTopKRouter` in three pure pieces, so the clean forward (select + weight),
# the masked forward (weight at the pinned indices), the evals, and the tests all spell
# the released router's arithmetic from ONE place.


def router_logits(router: Float[Array, "E d"], normed_input: Float[Array, "*lead d"]) -> Array:
    """Frozen-router scores with the target matmul dtype and fp32 output."""
    return (normed_input @ router.T).astype(jnp.float32)


def router_probs(router: Float[Array, "E d"], normed_input: Float[Array, "*lead d"]) -> Array:
    """The frozen router's fp32 softmax over ALL experts of the normed MoE input."""
    return jax.nn.softmax(router_logits(router, normed_input), axis=-1)


def select_experts(probs: Float[Array, "*lead E"], k: int) -> Int[Array, "*lead k"]:
    """The k most probable experts per position, descending, lowest index on ties."""
    # A stable sort over the (unsharded) expert axis, not `lax.top_k`: value-identical
    # (same descending order, same lowest-index tie-break), but top_k's explicit-sharding
    # rule gathers the batch axes — an in-loop cross-data collective the census forbids.
    # Spelled via sort_key_val so the index payload can carry the keys' sharding
    # (argsort's internal iota types replicated, which sort refuses against sharded
    # keys). The ordering is piecewise-constant, so the sort runs under stop_gradient
    # (sort's JVP rule builds a replicated iota that explicit sharding refuses); the
    # probabilities re-gather in `expert_mixing_weights`, which carries exactly top_k's
    # derivative.
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
    return order[..., :k]


def expert_mixing_weights(
    probs: Float[Array, "*lead E"], indices: Int[Array, "*lead k"]
) -> Float[Array, "*lead k"]:
    """`probs` gathered at `indices`, renormalized to sum 1 over the k. Unconditional:
    we hard-coded the released model's setting (HF `norm_topk_prob=True`)."""
    selected = jnp.take_along_axis(probs, indices, axis=-1)
    return selected / jnp.sum(selected, axis=-1, keepdims=True)


def is_expert_kind(kind: str) -> bool:
    """Whether one matrix kind is expert-blocked (narrow-emitting) vs shared (dense)."""
    assert kind in KIND_ORDER, kind
    return kind in _EXPERT_KINDS


def parse_site_name(name: str) -> tuple[int, str]:
    """`layers.{i}.{module path}` -> (layer, kind) for the paths in `_SITE_SUFFIX`;
    rejects anything else."""
    match = _SITE_NAME_PATTERN.match(name)
    assert match is not None and match.group(2) in _KIND_OF_SUFFIX, (
        f"not a qwen36_moe site: {name!r} (sites are layers.{{i}}.<path> for the paths "
        f"{sorted(_KIND_OF_SUFFIX)})"
    )
    return int(match.group(1)), _KIND_OF_SUFFIX[match.group(2)]


FAMILY = ArchFamily("qwen36_moe", KIND_ORDER, site_name, parse_site_name)
"""This family's matrix grammar as data — the vocabulary + name renderer qwen36_moe
c-specs resolve against."""


def site_dims(cfg: Qwen36MoeConfig, kind: str) -> SiteDims:
    """Dimensions of one per-layer site matrix in right-mult orientation. The `experts_*`
    sites are the FUSED all-expert matrices (expert-major on the `E·di` axis); the
    attention q site is HF's fused query|gate projection (`2·head_dim` rows per head)."""
    d = cfg.n_embd
    match kind:
        case "gdn_q" | "gdn_k":
            return SiteDims(d_in=d, d_out=cfg.linear_key_dim)
        case "gdn_v" | "gdn_z":
            return SiteDims(d_in=d, d_out=cfg.linear_value_dim)
        case "gdn_b" | "gdn_a":
            return SiteDims(d_in=d, d_out=cfg.linear_num_value_heads)
        case "gdn_out":
            return SiteDims(d_in=cfg.linear_value_dim, d_out=d)
        case "attn_q":
            return SiteDims(d_in=d, d_out=2 * cfg.n_head * cfg.head_dim)
        case "attn_k" | "attn_v":
            return SiteDims(d_in=d, d_out=cfg.n_kv_head * cfg.head_dim)
        case "attn_o":
            return SiteDims(d_in=cfg.n_head * cfg.head_dim, d_out=d)
        case "experts_gate" | "experts_up":
            return SiteDims(d_in=d, d_out=cfg.n_experts * cfg.moe_intermediate)
        case "experts_down":
            return SiteDims(d_in=cfg.n_experts * cfg.moe_intermediate, d_out=d)
        case "shared_gate" | "shared_up":
            return SiteDims(d_in=d, d_out=cfg.shared_expert_intermediate)
        case "shared_down":
            return SiteDims(d_in=cfg.shared_expert_intermediate, d_out=d)
        case _:
            raise AssertionError(f"unknown kind {kind!r}")


def capture_grammar(cfg: Qwen36MoeConfig) -> TransformerTapGrammar:
    """The shared transformer vectors this target computes: the normed mixer input, the
    mixer core its out/o projection consumes (DeltaNet's gated-normed core or attention's
    gated SDPA output, by the layer's mixer), and the normed MoE input; and each of the
    layer's own matrices' outputs. The routed expert hidden is not one dense per-block
    vector, and the post-mixer residual is not captured."""

    def block(layer: int) -> BlockCaptures:
        mixer_output = "attn_o" if layer_is_full_attention(cfg, layer) else "gdn_out"
        return BlockCaptures(
            tap_widths={
                "attn_in": cfg.n_embd,
                "attn_out": site_dims(cfg, mixer_output).d_in,
                "mlp_in": cfg.n_embd,
            },
            site_output_widths={
                kind: site_dims(cfg, kind).d_out
                for kind in KIND_ORDER
                if layer in layers_of_kind(cfg, kind)
            },
        )

    return TransformerTapGrammar(
        family=FAMILY,
        d_resid=cfg.n_embd,
        blocks=tuple(block(layer) for layer in range(cfg.n_layer)),
    )


def _delta_rule_flops(
    cfg: Qwen36MoeConfig,
    batch_size: int,
    sequence_length: int,
    *,
    gradients: GradientTarget,
    query_changed: bool,
    key_changed: bool,
    value_changed: bool,
    decay_changed: bool,
    strength_changed: bool,
    auxiliary_gradient: bool,
) -> TargetPassFlops:
    product = (
        2
        * batch_size
        * cfg.linear_num_value_heads
        * cfg.linear_key_head_dim
        * cfg.linear_value_head_dim
    )
    first_state_changed = key_changed or value_changed or strength_changed
    later_state_changed = first_state_changed or decay_changed
    match gradients:
        case "none":
            backward = 0
        case "components" | "sources":
            backward = product * (
                key_changed
                + (value_changed or strength_changed)
                + first_state_changed
                + query_changed
                + (sequence_length - 1)
                * (3 * later_state_changed + 2 * key_changed + query_changed)
            )
    shared = product * (
        (not first_state_changed)
        + (not (first_state_changed or query_changed))
        + (sequence_length - 1)
        * (2 * (not later_state_changed) + (not (later_state_changed or query_changed)))
    )
    return TargetPassFlops(
        ForwardBackwardFlops((3 * sequence_length - 1) * product, backward),
        shared,
        {},
        backward if auxiliary_gradient else 0,
    )


def qwen36_moe_flops(
    cfg: Qwen36MoeConfig,
    sites: tuple[SiteSpec, ...],
    *,
    batch_size: int,
    sequence_length: int,
    gradients: GradientTarget,
    include_frozen_paths: bool,
    capture_keys: frozenset[str],
) -> TargetPassFlops:
    """Count hybrid attention, selected experts, and their required derivatives."""
    validate_batch(batch_size, sequence_length)
    by_name = {site.name: site for site in sites}
    if len(by_name) != len(sites):
        raise ValueError("Decomposed site names must be unique")
    grammar = capture_grammar(cfg)
    captured = frozenset(_parse_capture_key(key, grammar, cfg) for key in capture_keys)

    def tap_captured(layer: int, tap: _Tap) -> bool:
        return _CaptureSource(layer=layer, tap=tap) in captured

    def output_captured(layer: int, kind: str) -> bool:
        return tap_captured(layer, _SITE_OUTPUT_TAP[kind])

    auxiliary: dict[str, bool] = {}
    residual_out_auxiliary: dict[int, bool] = {}
    router_logits_auxiliary: dict[int, bool] = {}
    router_weights_auxiliary: dict[int, bool] = {}
    mixer_core_auxiliary: dict[int, bool] = {}
    valid_sites: set[str] = set()
    downstream = False
    for layer in reversed(range(cfg.n_layer)):
        residual_auxiliary = downstream or tap_captured(layer, _Tap.RESIDUAL_OUT)
        residual_out_auxiliary[layer] = residual_auxiliary
        mlp_auxiliary = residual_auxiliary or tap_captured(layer, _Tap.MOE_INPUT)
        for branch in ("experts", "shared"):
            down_kind = f"{branch}_down"
            down_name = site_name(layer, down_kind)
            down_auxiliary = residual_auxiliary or output_captured(layer, down_kind)
            auxiliary[down_name] = down_auxiliary
            for kind in (f"{branch}_gate", f"{branch}_up"):
                name = site_name(layer, kind)
                auxiliary[name] = down_auxiliary or output_captured(layer, kind)
                mlp_auxiliary = mlp_auxiliary or auxiliary[name]
        router_capture = tap_captured(layer, _Tap.ROUTER_LOGITS) or tap_captured(
            layer, _Tap.ROUTER_PROBS
        )
        router_logits_auxiliary[layer] = router_capture
        router_weights_auxiliary[layer] = router_capture or (
            cfg.n_experts_per_token > 1
            and (
                auxiliary[site_name(layer, "experts_down")]
                or tap_captured(layer, _Tap.ROUTER_WEIGHTS)
            )
        )
        mlp_auxiliary = mlp_auxiliary or router_weights_auxiliary[layer]
        output_kind = "attn_o" if layer_is_full_attention(cfg, layer) else "gdn_out"
        output_name = site_name(layer, output_kind)
        auxiliary[output_name] = mlp_auxiliary or output_captured(layer, output_kind)
        attention_auxiliary = auxiliary[output_name] or tap_captured(
            layer, _mixer_core_tap(cfg, layer)
        )
        mixer_core_auxiliary[layer] = attention_auxiliary
        downstream = mlp_auxiliary or tap_captured(layer, _Tap.MIXER_INPUT)
        kinds = (
            ("attn_q", "attn_k", "attn_v")
            if layer_is_full_attention(cfg, layer)
            else ("gdn_q", "gdn_k", "gdn_v", "gdn_z", "gdn_b", "gdn_a")
        )
        for kind in kinds:
            name = site_name(layer, kind)
            reaches_attention = kind != "gdn_a" or sequence_length > 1
            auxiliary[name] = (attention_auxiliary and reaches_attention) or output_captured(
                layer, kind
            )
            downstream = downstream or auxiliary[name]
        valid_sites.update(
            site_name(layer, kind)
            for kind in (
                *kinds,
                output_kind,
                "experts_gate",
                "experts_up",
                "experts_down",
                "shared_gate",
                "shared_up",
                "shared_down",
            )
        )
    if unknown := by_name.keys() - valid_sites:
        raise ValueError(f"Sites outside the target architecture: {sorted(unknown)}")

    terms: list[TargetPassFlops] = []
    n_rows = batch_size * sequence_length

    def project(layer: int, kind: str, input_changed: bool) -> bool:
        name = site_name(layer, kind)
        dims = site_dims(cfg, kind)
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
                auxiliary_gradient=auxiliary[name],
                component_key=name,
            )
        )
        return input_changed or name in by_name

    residual_changed = False
    for layer in range(cfg.n_layer):
        attention_auxiliary = mixer_core_auxiliary[layer]
        if layer_is_full_attention(cfg, layer):
            query_changed = project(layer, "attn_q", residual_changed)
            key_changed = project(layer, "attn_k", residual_changed)
            value_changed = project(layer, "attn_v", residual_changed)
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
                    auxiliary_gradient=attention_auxiliary,
                )
            )
            mixer_changed = project(layer, "attn_o", query_changed or key_changed or value_changed)
        else:
            query_changed = project(layer, "gdn_q", residual_changed)
            key_changed = project(layer, "gdn_k", residual_changed)
            value_changed = project(layer, "gdn_v", residual_changed)
            gate_changed = project(layer, "gdn_z", residual_changed)
            strength_changed = project(layer, "gdn_b", residual_changed)
            decay_name = site_name(layer, "gdn_a")
            terms.append(
                linear_flops(
                    decay_name,
                    n_rows - batch_size,
                    cfg.n_embd,
                    cfg.linear_num_value_heads,
                    by_name.get(decay_name),
                    gradients=gradients,
                    input_changed=residual_changed,
                    include_frozen_paths=include_frozen_paths,
                    output_gradient=sequence_length > 1,
                    auxiliary_gradient=auxiliary[decay_name],
                    component_key=f"{decay_name}/after_first_token",
                )
            )
            if output_captured(layer, "gdn_a"):
                terms.append(
                    linear_flops(
                        decay_name,
                        batch_size,
                        cfg.n_embd,
                        cfg.linear_num_value_heads,
                        by_name.get(decay_name),
                        gradients=gradients,
                        input_changed=residual_changed,
                        include_frozen_paths=include_frozen_paths,
                        output_gradient=False,
                        auxiliary_gradient=True,
                        component_key=f"{decay_name}/first_token",
                    )
                )
            decay_changed = residual_changed or decay_name in by_name
            n_taps = min(sequence_length, cfg.linear_conv_kernel_dim)
            n_nonzero_taps = n_taps * (n_taps + 1) // 2 + (sequence_length - n_taps) * n_taps
            for width, changed in (
                (cfg.linear_key_dim, query_changed),
                (cfg.linear_key_dim, key_changed),
                (cfg.linear_value_dim, value_changed),
            ):
                terms.append(
                    frozen_product_flops(
                        2 * batch_size * width * n_nonzero_taps,
                        gradients=gradients,
                        left_changed=changed,
                        right_changed=False,
                        output_gradient=True,
                        auxiliary_gradient=attention_auxiliary,
                    )
                )
            terms.append(
                _delta_rule_flops(
                    cfg,
                    batch_size,
                    sequence_length,
                    gradients=gradients,
                    query_changed=query_changed,
                    key_changed=key_changed,
                    value_changed=value_changed,
                    decay_changed=decay_changed,
                    strength_changed=strength_changed,
                    auxiliary_gradient=attention_auxiliary,
                )
            )
            mixed_changed = (
                query_changed
                or key_changed
                or value_changed
                or strength_changed
                or (sequence_length > 1 and decay_changed)
            )
            mixer_changed = project(layer, "gdn_out", mixed_changed or gate_changed)
        residual_changed = residual_changed or mixer_changed
        full_router = (
            not sites
            or tap_captured(layer, _Tap.ROUTER_LOGITS)
            or tap_captured(layer, _Tap.ROUTER_PROBS)
        )
        n_selected = cfg.n_experts_per_token
        if full_router or n_selected > 1:
            terms.append(
                frozen_product_flops(
                    2 * n_rows * cfg.n_embd * n_selected,
                    gradients=gradients,
                    left_changed=residual_changed,
                    right_changed=False,
                    output_gradient=n_selected > 1,
                    auxiliary_gradient=router_weights_auxiliary[layer],
                )
            )
        if full_router:
            terms.append(
                frozen_product_flops(
                    2 * n_rows * cfg.n_embd * (cfg.n_experts - n_selected),
                    gradients=gradients,
                    left_changed=residual_changed,
                    right_changed=False,
                    output_gradient=False,
                    auxiliary_gradient=router_logits_auxiliary[layer],
                )
            )
        expert_hidden_changed = residual_changed
        for kind in ("experts_gate", "experts_up", "experts_down"):
            name = site_name(layer, kind)
            is_down = kind == "experts_down"
            terms.append(
                linear_flops(
                    name,
                    n_rows * n_selected,
                    cfg.moe_intermediate if is_down else cfg.n_embd,
                    cfg.n_embd if is_down else cfg.moe_intermediate,
                    by_name.get(name),
                    gradients=gradients,
                    input_changed=expert_hidden_changed if is_down else residual_changed,
                    include_frozen_paths=include_frozen_paths,
                    output_gradient=True,
                    auxiliary_gradient=auxiliary[name],
                    component_key=name,
                )
            )
            expert_hidden_changed = expert_hidden_changed or name in by_name
        shared_gate_changed = project(layer, "shared_gate", residual_changed)
        shared_up_changed = project(layer, "shared_up", residual_changed)
        shared_changed = project(layer, "shared_down", shared_gate_changed or shared_up_changed)
        terms.append(
            frozen_product_flops(
                2 * n_rows * cfg.n_embd,
                gradients=gradients,
                left_changed=residual_changed,
                right_changed=False,
                output_gradient=True,
                auxiliary_gradient=residual_out_auxiliary[layer],
            )
        )
        residual_changed = residual_changed or expert_hidden_changed or shared_changed
    terms.append(
        frozen_product_flops(
            2 * n_rows * cfg.n_embd * cfg.vocab_size,
            gradients=gradients,
            left_changed=residual_changed,
            right_changed=False,
            output_gradient=True,
            auxiliary_gradient=False,
        )
    )
    return sum_target_flops(terms)


def site_factorization(cfg: Qwen36MoeConfig, kind: str, C: int) -> Factorization:
    """How one kind's V/U factor its matrix: the `experts_*` sites are expert-local
    (`BlockedFactorization`, `c_per_expert = C // n_experts` components confined to each
    expert's block); every other site is dense."""
    d = cfg.n_embd
    di = cfg.moe_intermediate
    match kind:
        case "experts_gate" | "experts_up" | "experts_down":
            assert C % cfg.n_experts == 0, (
                f"{kind}: C={C} must be a multiple of n_experts={cfg.n_experts}"
            )
            c_per_expert = C // cfg.n_experts
            d_in, d_out = (d, di) if kind != "experts_down" else (di, d)
            return BlockedFactorization(
                n_blocks=cfg.n_experts, d_in=d_in, d_out=d_out, c_per_block=c_per_expert
            )
        case (
            "gdn_q"
            | "gdn_k"
            | "gdn_v"
            | "gdn_z"
            | "gdn_b"
            | "gdn_a"
            | "gdn_out"
            | "attn_q"
            | "attn_k"
            | "attn_v"
            | "attn_o"
            | "shared_gate"
            | "shared_up"
            | "shared_down"
        ):
            return site_dims(cfg, kind).dense(C)
        case _:
            raise AssertionError(f"unknown kind {kind!r}")


def site_contraction(kind: str) -> BlockContraction | None:
    """The orientation of one kind's expert-blocked linears (None = a dense site):
    gate/up fuse the experts on their output, down on its input."""
    match kind:
        case "experts_gate" | "experts_up":
            return "fused_output"
        case "experts_down":
            return "fused_input"
        case (
            "gdn_q"
            | "gdn_k"
            | "gdn_v"
            | "gdn_z"
            | "gdn_b"
            | "gdn_a"
            | "gdn_out"
            | "attn_q"
            | "attn_k"
            | "attn_v"
            | "attn_o"
            | "shared_gate"
            | "shared_up"
            | "shared_down"
        ):
            return None
        case _:
            raise AssertionError(f"unknown kind {kind!r}")


def nonlinearity_alignment(cfg: Qwen36MoeConfig, kind: str) -> NonlinearityAlignment:
    """The side and partition of each kind facing a nonlinearity. The
    DeltaNet q/k feed a per-key-head l2norm and delta-rule recurrence — a key head, read
    by the `n_v_heads / n_k_heads` value heads that repeat it; v/z feed the per-value-head
    recurrence and gated RMS norm — a value head. b/a emit one scalar per value head
    (σ(b), softplus(a)), so a coordinate already is a head's chunk. The attention q meets
    its per-head QK-norm and SDPA (the fused gate half rides with its head), k/v their
    per-kv-head norm and the `n_head / n_kv_head` query heads that read them. The MoE
    gate/up feed the silu·up product per hidden unit. GDN out reads value heads,
    attention out reads query heads, and MoE down reads hidden neurons."""
    match kind:
        case "gdn_q" | "gdn_k":
            return NonlinearityAlignment(
                "output",
                DeltaNetHeads(
                    cfg.linear_num_key_heads, cfg.linear_num_value_heads // cfg.linear_num_key_heads
                ),
            )
        case "gdn_v" | "gdn_z":
            return NonlinearityAlignment("output", DeltaNetHeads(cfg.linear_num_value_heads, 1))
        case "gdn_b" | "gdn_a":
            return NonlinearityAlignment("output", Neurons())
        case "attn_q":
            return NonlinearityAlignment("output", QueryHeads(cfg.n_head))
        case "attn_k" | "attn_v":
            return NonlinearityAlignment(
                "output", KVHeads(cfg.n_kv_head, cfg.n_head // cfg.n_kv_head)
            )
        case "experts_gate" | "experts_up" | "shared_gate" | "shared_up":
            return NonlinearityAlignment("output", Neurons())
        case "gdn_out":
            return NonlinearityAlignment("input", DeltaNetHeads(cfg.linear_num_value_heads, 1))
        case "attn_o":
            return NonlinearityAlignment("input", QueryHeads(cfg.n_head))
        case "experts_down" | "shared_down":
            return NonlinearityAlignment("input", Neurons())
        case _:
            raise AssertionError(f"unknown kind {kind!r}")


def canonical_site_cs(site_cs: tuple[SiteC, ...]) -> tuple[SiteC, ...]:
    return family.canonical_site_cs(FAMILY, site_cs)


def qwen36_moe_site_specs(cfg: Qwen36MoeConfig, site_cs: tuple[SiteC, ...]) -> tuple[SiteSpec, ...]:
    for site in site_cs:
        layer, kind = parse_site_name(site.name)
        assert layer in layers_of_kind(cfg, kind), (
            f"{site.name}: layer {layer} has no {kind!r} matrix (the {sublayer_of(kind)} "
            f"sublayer sits on layers {layers_of_kind(cfg, kind)})"
        )
    return family.site_specs(
        FAMILY,
        site_cs,
        lambda kind, c: site_factorization(cfg, kind, c),
        lambda kind: nonlinearity_alignment(cfg, kind),
        cfg.n_layer,
    )


_ROW_KINDS = frozenset({"gdn_out", "attn_o", "experts_down", "shared_down"})
"""Residual writers consume `row` target placement; the remaining kinds use `column`."""


def validate_nonlinearity_aligned_capacity(spec: SiteSpec) -> None:
    """Require every component to receive a nonempty coordinate to align on."""
    match spec.factorization:
        case DenseFactorization(d_in=d_in, d_out=d_out, C=count):
            pass
        case BlockedFactorization(d_in=d_in, d_out=d_out, c_per_block=count):
            pass
    assert count <= d_in * d_out, (
        f"{spec.name}: nonlinearity-aligned init supports at most "
        f"{d_in * d_out} nonempty components per expert block, got {count}"
    )


def _expert_blocks(weight: Array, factorization: BlockedFactorization, kind: str) -> Array:
    """View one fused MoE weight as `[expert, d_out, d_in]` without copying it."""
    match kind:
        case "experts_gate" | "experts_up":
            return weight.reshape(factorization.n_blocks, factorization.d_out, factorization.d_in)
        case "experts_down":
            return weight.reshape(
                factorization.d_out, factorization.n_blocks, factorization.d_in
            ).transpose(1, 0, 2)
        case _:
            raise AssertionError(kind)


def nonlinearity_aligned_component_initializer(
    model: "Qwen36MoeDecomposedModel", key: PRNGKeyArray
) -> ComponentStacks:
    """Initialize every component on its declared matrix side — the
    finest exact factorization (`nonlinearity_aligned_factors`: exact at C = the coordinate
    count, sampled below it, partitioned above it). One coordinate lies inside exactly
    one nonlinearity's chunk (a SwiGLU hidden unit, a DeltaNet channel, an attention
    head — `nonlinearity_alignment`), so at exact width every component starts inside
    its chunk. Expert kinds run the same rule per expert block."""
    cfg = model.cfg
    keys = jax.random.split(key, len(model.sites))
    site_arrays: dict[str, tuple[Array, Array]] = {}
    for spec, site_key in zip(model.sites, keys, strict=True):
        validate_nonlinearity_aligned_capacity(spec)
        layer, kind = parse_site_name(spec.name)
        assert spec.alignment is not None, spec
        side = spec.alignment.side
        weight = model._frozen_kind_stack(kind)[layers_of_kind(cfg, kind).index(layer)]
        match spec.factorization:
            case DenseFactorization() as factorization:
                site_arrays[spec.name] = nonlinearity_aligned_factors(
                    weight, factorization, side, site_key
                )
            case BlockedFactorization(d_in=d_in, d_out=d_out, c_per_block=count) as factorization:
                block_factorization = DenseFactorization(d_in=d_in, d_out=d_out, C=count)

                def init_block(
                    block: Array,
                    f: DenseFactorization = block_factorization,
                    alignment_side: ComponentSide = side,
                    key: PRNGKeyArray = site_key,
                ) -> tuple[Array, Array]:
                    return nonlinearity_aligned_factors(
                        block,
                        f,
                        alignment_side,
                        jax.random.fold_in(key, jax.lax.axis_index("expert")),
                    )

                site_arrays[spec.name] = jax.vmap(init_block, axis_name="expert")(
                    _expert_blocks(weight, factorization, kind)
                )
    return component_stacks_from_site_arrays(model.sites, site_arrays)


def full_site_cs(cfg: Qwen36MoeConfig, c_of: Mapping[str, int]) -> tuple[SiteC, ...]:
    """Every site of the selected kinds at their C, in canonical order — each kind on
    every layer that has it, the coverage this target's masked forward requires."""
    assert set(c_of) <= set(KIND_ORDER), sorted(c_of)
    covered = {kind: frozenset(layers_of_kind(cfg, kind)) for kind in c_of}
    return tuple(
        SiteC(site_name(layer, kind), c_of[kind])
        for layer in range(cfg.n_layer)
        for kind in KIND_ORDER
        if kind in c_of and layer in covered[kind]
    )


# ----------------------------- capture grammar -----------------------------


class _Tap(Enum):
    RESIDUAL_IN = "residual_in"
    MIXER_INPUT = "mixer_input"
    GDN_Q_OUTPUT = "gdn_q_output"
    GDN_K_OUTPUT = "gdn_k_output"
    GDN_V_OUTPUT = "gdn_v_output"
    GDN_Z_OUTPUT = "gdn_z_output"
    GDN_B_OUTPUT = "gdn_b_output"
    GDN_A_OUTPUT = "gdn_a_output"
    GDN_CORE_OUTPUT = "gdn_core_output"
    GDN_OUT_OUTPUT = "gdn_out_output"
    ATTN_Q_OUTPUT = "attn_q_output"
    ATTN_K_OUTPUT = "attn_k_output"
    ATTN_V_OUTPUT = "attn_v_output"
    ATTN_CORE_OUTPUT = "attn_core_output"
    ATTN_O_OUTPUT = "attn_o_output"
    MOE_INPUT = "moe_input"
    ROUTER_LOGITS = "router_logits"
    ROUTER_PROBS = "router_probs"
    ROUTER_WEIGHTS = "router_weights"
    EXPERTS_GATE_OUTPUT = "experts_gate_output"
    EXPERTS_UP_OUTPUT = "experts_up_output"
    EXPERTS_DOWN_OUTPUT = "experts_down_output"
    SHARED_GATE_OUTPUT = "shared_gate_output"
    SHARED_UP_OUTPUT = "shared_up_output"
    SHARED_DOWN_OUTPUT = "shared_down_output"
    RESIDUAL_OUT = "residual_out"


_SITE_OUTPUT_TAP: dict[str, _Tap] = {kind: _Tap(f"{kind}_output") for kind in KIND_ORDER}
_KIND_OF_OUTPUT_TAP = {tap: kind for kind, tap in _SITE_OUTPUT_TAP.items()}

_GDN_TAPS = frozenset({_SITE_OUTPUT_TAP[kind] for kind in _GDN_KINDS} | {_Tap.GDN_CORE_OUTPUT})
_ATTN_TAPS = frozenset({_SITE_OUTPUT_TAP[kind] for kind in _ATTN_KINDS} | {_Tap.ATTN_CORE_OUTPUT})
"""The taps that exist only on one mixer's layers: structurally ABSENT on the other's
(a full-attention layer has no DeltaNet projections and vice versa), so a stage body
writes a layer's buffers for its own mixer's taps alone."""


def _tap_feature_row(placement: PlacementRules, tap: _Tap, cfg: Qwen36MoeConfig) -> PlacedRule:
    """Where a tap's values sit on the mesh: a site output at its kind's linear output
    row (the column kinds' fused/hidden/head widths shard over tp; the residual
    writers' and a replicated K/V's land model-width at the external row); a mixer core
    at the row of the head-sharded projections it is shaped like; the model-width and
    routing-shaped taps at the external row."""
    match tap:
        case _Tap.GDN_CORE_OUTPUT:
            kind = "gdn_v"
        case _Tap.ATTN_CORE_OUTPUT:
            kind = "attn_q"
        case (
            _Tap.GDN_Q_OUTPUT
            | _Tap.GDN_K_OUTPUT
            | _Tap.GDN_V_OUTPUT
            | _Tap.GDN_Z_OUTPUT
            | _Tap.GDN_B_OUTPUT
            | _Tap.GDN_A_OUTPUT
            | _Tap.GDN_OUT_OUTPUT
            | _Tap.ATTN_Q_OUTPUT
            | _Tap.ATTN_K_OUTPUT
            | _Tap.ATTN_V_OUTPUT
            | _Tap.ATTN_O_OUTPUT
            | _Tap.EXPERTS_GATE_OUTPUT
            | _Tap.EXPERTS_UP_OUTPUT
            | _Tap.EXPERTS_DOWN_OUTPUT
            | _Tap.SHARED_GATE_OUTPUT
            | _Tap.SHARED_UP_OUTPUT
            | _Tap.SHARED_DOWN_OUTPUT
        ):
            kind = _KIND_OF_OUTPUT_TAP[tap]
        case (
            _Tap.RESIDUAL_IN
            | _Tap.MIXER_INPUT
            | _Tap.MOE_INPUT
            | _Tap.ROUTER_LOGITS
            | _Tap.ROUTER_PROBS
            | _Tap.ROUTER_WEIGHTS
            | _Tap.RESIDUAL_OUT
        ):
            return placement.activations.external
    target_linear = _kind_target_linear(cfg, placement, kind)
    return placement.activations.external if target_linear is None else target_linear.output


def _tap_dtype(tap: _Tap, residual_dtype: jnp.dtype) -> jnp.dtype:
    """Capture-buffer dtype per tap: activations at the residual dtype; the routing
    taps at the verbs' own fp32."""
    match tap:
        case _Tap.ROUTER_LOGITS | _Tap.ROUTER_PROBS | _Tap.ROUTER_WEIGHTS:
            return jnp.dtype(jnp.float32)
        case (
            _Tap.RESIDUAL_IN
            | _Tap.MIXER_INPUT
            | _Tap.GDN_Q_OUTPUT
            | _Tap.GDN_K_OUTPUT
            | _Tap.GDN_V_OUTPUT
            | _Tap.GDN_Z_OUTPUT
            | _Tap.GDN_B_OUTPUT
            | _Tap.GDN_A_OUTPUT
            | _Tap.GDN_CORE_OUTPUT
            | _Tap.GDN_OUT_OUTPUT
            | _Tap.ATTN_Q_OUTPUT
            | _Tap.ATTN_K_OUTPUT
            | _Tap.ATTN_V_OUTPUT
            | _Tap.ATTN_CORE_OUTPUT
            | _Tap.ATTN_O_OUTPUT
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


@dataclass(frozen=True, kw_only=True)
class _CaptureSource:
    layer: int
    tap: _Tap


_ROUTER_TAP_OF_NAME: dict[str, _Tap] = {
    tap.value: tap for tap in (_Tap.ROUTER_LOGITS, _Tap.ROUTER_PROBS, _Tap.ROUTER_WEIGHTS)
}


def _router_tap_key(tap: _Tap, layer: int) -> str:
    assert tap.value in _ROUTER_TAP_OF_NAME, tap
    return f"{tap.value}.{layer}"


def _mixer_core_tap(cfg: Qwen36MoeConfig, layer: int) -> _Tap:
    return _Tap.ATTN_CORE_OUTPUT if layer_is_full_attention(cfg, layer) else _Tap.GDN_CORE_OUTPUT


def _parse_capture_key(
    key: str, grammar: TransformerTapGrammar, cfg: Qwen36MoeConfig
) -> _CaptureSource:
    """This target's closed activation vocabulary: its own router taps, and the shared
    transformer points `capture_grammar` declares; anything else fails closed."""
    prefix, _, suffix = key.partition(".")
    if prefix in _ROUTER_TAP_OF_NAME:
        assert suffix.isdigit() and int(suffix) < grammar.n_layer, (
            f"router tap {key!r} out of range: layers are 0..{grammar.n_layer - 1}"
        )
        return _CaptureSource(layer=int(suffix), tap=_ROUTER_TAP_OF_NAME[prefix])
    match grammar.parse(key):
        case ResidualBoundary(boundary=0):
            return _CaptureSource(layer=0, tap=_Tap.RESIDUAL_IN)
        case ResidualBoundary(boundary=boundary):
            return _CaptureSource(layer=boundary - 1, tap=_Tap.RESIDUAL_OUT)
        case BlockTap(name=name, block=block):
            match name:
                case "attn_in":
                    return _CaptureSource(layer=block, tap=_Tap.MIXER_INPUT)
                case "attn_out":
                    return _CaptureSource(layer=block, tap=_mixer_core_tap(cfg, block))
                case "mlp_in":
                    return _CaptureSource(layer=block, tap=_Tap.MOE_INPUT)
                case "post_attn" | "mlp_hidden":
                    raise AssertionError(f"{key!r}: `capture_grammar` declares no {name!r} tap")
        case SiteOutput(block=block, kind=kind):
            return _CaptureSource(layer=block, tap=_SITE_OUTPUT_TAP[kind])


def _capture_sources(keys: tuple[str, ...], cfg: Qwen36MoeConfig) -> tuple[_CaptureSource, ...]:
    grammar = capture_grammar(cfg)
    sources = tuple(_parse_capture_key(key, grammar, cfg) for key in keys)
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
class _GdnActs:
    """One DeltaNet layer's mixer activations: the normed input, each projection's
    linear output (before its conv+silu, σ, softplus, or the gated norm), the
    gated-normed core the out projection consumes, and the out projection's output
    (the residual write)."""

    mixer_input: Array
    q_output: Array
    k_output: Array
    v_output: Array
    z_output: Array
    b_output: Array
    a_output: Array
    core_output: Array
    out_output: Array

    def of(self, tap: _Tap) -> Array:
        match tap:
            case _Tap.GDN_Q_OUTPUT:
                return self.q_output
            case _Tap.GDN_K_OUTPUT:
                return self.k_output
            case _Tap.GDN_V_OUTPUT:
                return self.v_output
            case _Tap.GDN_Z_OUTPUT:
                return self.z_output
            case _Tap.GDN_B_OUTPUT:
                return self.b_output
            case _Tap.GDN_A_OUTPUT:
                return self.a_output
            case _Tap.GDN_CORE_OUTPUT:
                return self.core_output
            case _Tap.GDN_OUT_OUTPUT:
                return self.out_output
            case _:
                raise AssertionError(tap)


@dataclass(frozen=True, kw_only=True)
class _AttnActs:
    """One full-attention layer's mixer activations: the normed input, the fused
    query|gate, k and v linear outputs, the gated SDPA output the o projection
    consumes, and the o projection's output (the residual write)."""

    mixer_input: Array
    q_output: Array
    k_output: Array
    v_output: Array
    core_output: Array
    o_output: Array

    def of(self, tap: _Tap) -> Array:
        match tap:
            case _Tap.ATTN_Q_OUTPUT:
                return self.q_output
            case _Tap.ATTN_K_OUTPUT:
                return self.k_output
            case _Tap.ATTN_V_OUTPUT:
                return self.v_output
            case _Tap.ATTN_CORE_OUTPUT:
                return self.core_output
            case _Tap.ATTN_O_OUTPUT:
                return self.o_output
            case _:
                raise AssertionError(tap)


@dataclass(frozen=True, kw_only=True)
class _MoeActs:
    """One layer's MoE activations plus the residual boundary.

    The fused-width gate/up taps and the router probs exist only when captured (None
    otherwise — no full-width materialization). Under every expert arm the gate/up taps
    hold the selected experts' outputs exactly as computed and ZEROS at unselected
    experts — the routed arms never compute those (the taps are their scattered job
    results), the dense arm computes and masks them — so a tap reads alike whatever
    the execution. `router_indices` is the layer's applied routing — returned through
    the forward's `BlockSelection`, never a tap."""

    moe_input: Array
    router_indices: Array
    router_logits: Array | None
    router_probs: Array | None
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
            case _Tap.MOE_INPUT:
                return self.moe_input
            case _Tap.ROUTER_LOGITS:
                assert self.router_logits is not None, "tap not captured"
                return self.router_logits
            case _Tap.ROUTER_PROBS:
                assert self.router_probs is not None, "tap not captured"
                return self.router_probs
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
            case _:
                raise AssertionError(tap)


@dataclass(frozen=True, kw_only=True)
class _LayerActs:
    """One layer's capturable activations: its mixer's bundle (the layer's own mixer —
    the other mixer's taps are structurally absent here) and its MoE's."""

    mixer: _GdnActs | _AttnActs
    moe: _MoeActs

    def of(self, tap: _Tap) -> Array:
        match tap:
            case _Tap.RESIDUAL_IN:
                raise AssertionError("the embedding residual is recorded pre-scan")
            case _Tap.MIXER_INPUT:
                return self.mixer.mixer_input
            case (
                _Tap.GDN_Q_OUTPUT
                | _Tap.GDN_K_OUTPUT
                | _Tap.GDN_V_OUTPUT
                | _Tap.GDN_Z_OUTPUT
                | _Tap.GDN_B_OUTPUT
                | _Tap.GDN_A_OUTPUT
                | _Tap.GDN_CORE_OUTPUT
                | _Tap.GDN_OUT_OUTPUT
            ):
                assert isinstance(self.mixer, _GdnActs), tap
                return self.mixer.of(tap)
            case (
                _Tap.ATTN_Q_OUTPUT
                | _Tap.ATTN_K_OUTPUT
                | _Tap.ATTN_V_OUTPUT
                | _Tap.ATTN_CORE_OUTPUT
                | _Tap.ATTN_O_OUTPUT
            ):
                assert isinstance(self.mixer, _AttnActs), tap
                return self.mixer.of(tap)
            case (
                _Tap.MOE_INPUT
                | _Tap.ROUTER_LOGITS
                | _Tap.ROUTER_PROBS
                | _Tap.ROUTER_WEIGHTS
                | _Tap.EXPERTS_GATE_OUTPUT
                | _Tap.EXPERTS_UP_OUTPUT
                | _Tap.EXPERTS_DOWN_OUTPUT
                | _Tap.SHARED_GATE_OUTPUT
                | _Tap.SHARED_UP_OUTPUT
                | _Tap.SHARED_DOWN_OUTPUT
                | _Tap.RESIDUAL_OUT
            ):
                return self.moe.of(tap)


def _write_captures(
    buffers: dict[str, Array], slots: dict[str, Array], acts: _LayerActs, absent: frozenset[_Tap]
) -> dict[str, Array]:
    """Write this layer's values into the requested taps' buffers at their slots; the
    `absent` taps (the other mixer's) have no value here and an unused slot by
    construction (`_parse_capture_key`), so they are skipped."""
    updated = dict(buffers)
    for buffer_key, buffer in buffers.items():
        tap = _Tap(buffer_key)
        if tap in absent:
            continue
        slot = slots[buffer_key]
        value = acts.of(tap)
        updated[buffer_key] = jax.lax.cond(
            slot != _UNUSED_SLOT,
            lambda buf, v=value, s=slot: jax.lax.dynamic_update_index_in_dim(buf, v, s, axis=0),
            lambda buf: buf,
            buffer,
        )
    return updated


# ----------------------------- frozen modules -----------------------------


_GATED_DELTA_KERNEL: GatedDeltaKernel = "chunkwise"
"""The wired gated-delta-rule arm; `sequential` is the parity oracle the tests hold it
against (`GatedDeltaKernel` enumerates both)."""


class GatedDeltaNet(eqx.Module):
    """The `qwen3_5_moe` gated-DeltaNet token mixer (split in_proj variant): depthwise
    causal conv + silu on q/k/v, per-v-head decay `−exp(A_log)·softplus(a + dt_bias)`
    and write strength `σ(b)`, l2-normed q/k, the fp32 delta-rule scan, then the
    plain-weight gated output norm and out projection. The seven projections are sites,
    run through the layer's `_SiteExecutor`; the conv kernels, the decay/write vectors
    and the gated-norm weight are frozen.

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

    def shardings(self, placement: PlacementRules, lead: Axes) -> ShardingTree:
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

    def __call__(self, x: Float[Array, "b t d"], run: "_SiteExecutor") -> _GdnActs:
        b, t, _ = x.shape
        q_output = run("gdn_q", x, self.w_q)
        k_output = run("gdn_k", x, self.w_k)
        v_output = run("gdn_v", x, self.w_v)
        z_output = run("gdn_z", x, self.w_z)
        b_output = run("gdn_b", x, self.w_b)
        a_output = run("gdn_a", x, self.w_a)
        q = causal_depthwise_conv1d_silu(q_output, self.conv_q).reshape(
            b, t, self.n_k_heads, self.k_head_dim
        )
        k = causal_depthwise_conv1d_silu(k_output, self.conv_k).reshape(
            b, t, self.n_k_heads, self.k_head_dim
        )
        v = causal_depthwise_conv1d_silu(v_output, self.conv_v).reshape(
            b, t, self.n_v_heads, self.v_head_dim
        )
        z = z_output.reshape(b, t, self.n_v_heads, self.v_head_dim)
        beta = jax.nn.sigmoid(b_output)
        # fp32 before the exp/softplus: a bf16 A_log exponentiates to ±inf (HF's note).
        g = -jnp.exp(self.a_log.astype(jnp.float32)) * jax.nn.softplus(
            a_output.astype(jnp.float32) + self.dt_bias.astype(jnp.float32)
        )
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
        core = gated_rms_norm(core, self.norm_w, z, self.eps).reshape(
            b, t, self.n_v_heads * self.v_head_dim
        )
        return _GdnActs(
            mixer_input=x,
            q_output=q_output,
            k_output=k_output,
            v_output=v_output,
            z_output=z_output,
            b_output=b_output,
            a_output=a_output,
            core_output=core,
            out_output=run("gdn_out", core, self.w_out),
        )


def _kv_target_linear(placement: PlacementRules, n_kv_head: int) -> TargetLinearPlacement | None:
    """The K/V projections' Megatron layout: the column row when its tp degree splits
    WHOLE kv heads (persist, operand, and the output activation alike), else None — the
    weight replicated and the matmul replicated (the 35B's 2 KV heads sit below tp=8,
    and a row split inside one head would break the SDPA head unit; the placed forward
    then repeats K/V to the query-head count and shards the copies). A None-layout K/V
    site has no placed decomposed spelling (`_SiteExecutor`)."""
    column = placement.target.column
    rows: tuple[tuple[PlacedRule, SemanticAxis], ...] = (
        (column.persist, "d_out"),
        (column.operand, "d_out"),
        (column.output, "feature"),
    )
    tiles = all(n_kv_head % row.shard_count(axis) == 0 for row, axis in rows)
    return column if tiles else None


class GatedAttention(eqx.Module):
    """The `qwen3_5_moe` full-attention token mixer: the q projection emits query and a
    per-head sigmoid output gate (2·head_dim per head, query first), zero-centered
    per-head QK-norm before partial RoPE, GQA causal SDPA, gate, o projection. The four
    projections are sites, run through the layer's `_SiteExecutor`; the per-head norms
    are frozen."""

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

    def shardings(self, placement: PlacementRules, lead: Axes) -> ShardingTree:
        """q/o shard by query head (`wq` rows are head-major query|gate pairs); K/V
        shard by kv head where the column row splits whole kv heads and REPLICATE
        otherwise (`_kv_target_linear`); the per-head norms replicate."""
        column = placement.target.column.persist
        row = placement.target.row.persist
        matrix: Axes = (*lead, "d_out", "d_in")
        column.validate_shape(matrix, self.wq.shape)
        row.validate_shape(matrix, self.wo.shape)
        repl = NamedSharding(placement.mesh, P())
        kv = _kv_target_linear(placement, self.n_kv_head)
        kv_sharding = repl if kv is None else kv.persist.sharding_for(matrix)
        return eqx.tree_at(
            lambda m: (m.wq, m.wk, m.wv, m.wo, m.q_norm, m.k_norm),
            self,
            (
                column.sharding_for(matrix),
                kv_sharding,
                kv_sharding,
                row.sharding_for(matrix),
                repl,
                repl,
            ),
        )

    def __call__(
        self, x: Float[Array, "b t d"], inv_freq: Array, run: "_SiteExecutor"
    ) -> _AttnActs:
        b, t, _ = x.shape
        q_output = run("attn_q", x, self.wq)
        k_output = run("attn_k", x, self.wk)
        v_output = run("attn_v", x, self.wv)
        query_and_gate = q_output.reshape(b, t, self.n_head, 2 * self.head_dim)
        q = query_and_gate[..., : self.head_dim]
        gate = query_and_gate[..., self.head_dim :]
        q = rms_norm_zero_centered(q, self.q_norm, self.eps)
        k = rms_norm_zero_centered(
            k_output.reshape(b, t, self.n_kv_head, self.head_dim), self.k_norm, self.eps
        )
        v = v_output.reshape(b, t, self.n_kv_head, self.head_dim)
        cos, sin = rope_cos_sin(inv_freq, jnp.arange(t, dtype=jnp.int32)[None, :], q.dtype)
        q, k = apply_partial_rope(q, k, cos[0], sin[0])
        if run.placement is not None:
            # The internal GQA head-group reshape cannot split a q-head axis sharded
            # finer than the KV head count, and cuDNN SDPA wants q/k/v identically
            # sharded — so under placement K/V repeat to the query-head count (out-head
            # j reads kv-head j // rep, the GQA grouping: rank-local when the kv heads
            # are sharded, a broadcast when replicated) and take q's head sharding; the
            # unplaced arm keeps the grouped-KV fast path.
            rep = self.n_head // self.n_kv_head
            k = jax.sharding.reshard(
                jnp.repeat(k, rep, axis=2, out_sharding=jax.typeof(k).sharding),
                jax.typeof(q).sharding,
            )
            v = jax.sharding.reshard(
                jnp.repeat(v, rep, axis=2, out_sharding=jax.typeof(v).sharding),
                jax.typeof(q).sharding,
            )
        # Pin the SDPA operand shapes: a mis-split query/gate or a bad head reshape
        # reaches cuDNN as an unequal-head-dim (MLA) graph and dies at device-side graph
        # validation — make it a Python error with the shapes in hand instead.
        expected_kv_heads = self.n_head if run.placement is not None else self.n_kv_head
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
            implementation=jax_attention_implementation(
                self.implementation, get_default_device().platform, q.dtype
            ),
        )
        core = (out * jax.nn.sigmoid(gate)).reshape(b, t, self.n_head * self.head_dim)
        return _AttnActs(
            mixer_input=x,
            q_output=q_output,
            k_output=k_output,
            v_output=v_output,
            core_output=core,
            o_output=run("attn_o", core, self.wo),
        )


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

    def shardings(self, placement: PlacementRules, n_experts: int) -> ShardingTree:
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
    mixer: GatedDeltaNet


class AttnSublayer(eqx.Module):
    ln1: Float[Array, " d"]
    attn: GatedAttention


# ----------------------------- decomposed-site execution -----------------------------


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
    cfg: Qwen36MoeConfig, placement: PlacementRules | None, kind: str
) -> TargetLinearPlacement | None:
    """The Megatron layout one kind's frozen matrix takes: the residual writers consume
    the column shard and reduce onto the waist (`row`), the K/V projections their
    head-tiling-conditional row (`_kv_target_linear`), every other kind shards its
    output (`column`)."""
    if placement is None:
        return None
    if kind in ("attn_k", "attn_v"):
        return _kv_target_linear(placement, cfg.n_kv_head)
    return placement.target.row if kind in _ROW_KINDS else placement.target.column


V_WEIGHT_AXES: tuple[SemanticAxis, SemanticAxis] = ("d_in", "C")
U_WEIGHT_AXES: tuple[SemanticAxis, SemanticAxis] = ("C", "d_out")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _StagedMaterialized:
    mask: SiteCI
    delta: Array | None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _StagedStochastic:
    ci: SiteCI
    src_key: Array
    delta_key: Array


type _StagedMasking = _StagedMaterialized | _StagedStochastic | SourceMaskIngredients


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _StagedKind:
    """One kind's weights and mask recipe, sliced together inside checkpointed stages."""

    v: Array
    u: Array
    masking: _StagedMasking
    route: Array | None


def _require_kind_emission(kind: str, value: SiteCI) -> SiteCI:
    """Expert kinds require selected bundles; shared kinds require full-C arrays."""
    match value:
        case SelectedCI():
            assert is_expert_kind(kind), (
                f"shared kind {kind!r} takes full [.., C] masks/CI, got {type(value).__name__}"
            )
        case jax.Array():
            assert not is_expert_kind(kind), (
                f"expert kind {kind!r} takes selected masks/CI, got a full [.., C] array"
            )
    return value


@jax.tree_util.register_static
@dataclass(frozen=True)
class _CleanLayers:
    """The clean forward's per-layer inputs: none — every kind runs frozen and the
    router decides the routing."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _MaskedLayers:
    """A masked forward's per-layer inputs at entry, stacked per SUBLAYER: each decomposed
    kind's attached entry (V/U plus the masking mode's tensors) in its kind's stack
    order (`layers_of_kind`) — `deltanet` leaves lead `[n_stages·(interval−1), …]`,
    `attn` `[n_stages, …]`, `moe` `[n_layer, …]` — plus the pinned expert indices
    `[n_layer, …]` every layer reproduces."""

    deltanet: dict[str, _StagedKind]
    attn: dict[str, _StagedKind]
    moe: dict[str, _StagedKind]
    pinned_indices: Array


@dataclass(frozen=True)
class _MaskedLayer:
    """One layer's slice of `_MaskedLayers` inside the stage body: its own mixer's and
    its MoE's entries in one map (disjoint kinds) and its pinned indices."""

    sites: dict[str, _StagedKind]
    pinned_indices: Array


def _stage_layer_inputs(
    stage: _CleanLayers | _MaskedLayers, interval: int
) -> tuple[_CleanLayers | _MaskedLayer, ...]:
    """One stage's per-position layer inputs out of its blocked xs slice: the
    `interval−1` DeltaNet positions take their mixer entries by position, the closing
    attention position the stage's attention entries; each merges with the MoE entries
    and pinned indices of its position (`_block_positions`, inside the body's
    checkpoint)."""
    match stage:
        case _CleanLayers():
            return (stage,) * interval
        case _MaskedLayers(deltanet=deltanet, attn=attn, moe=moe):
            deltanet_by_position = _tree_by_position(deltanet, interval - 1)
            moe_by_position = _tree_by_position(moe, interval)
            pinned_by_position = _block_positions(stage.pinned_indices)
            return tuple(
                _MaskedLayer(
                    sites={
                        **(deltanet_by_position[pos] if pos < interval - 1 else attn),
                        **moe_by_position[pos],
                    },
                    pinned_indices=pinned_by_position[pos],
                )
                for pos in range(interval)
            )


def _entry_masking(inputs: _StagedKind) -> tuple[SiteCI, Array | None, Array | None]:
    """Resolve the recipe inside the checkpoint so backward can redraw or recompose."""
    match inputs.masking:
        case _StagedMaterialized(mask=mask, delta=delta):
            return mask, delta, inputs.route
        case _StagedStochastic(ci=ci, src_key=src_key, delta_key=delta_key):
            return (
                sample_component_mask(ci, src_key),
                sample_delta_mask(ci, delta_key),
                inputs.route,
            )
        case SourceMaskIngredients() as ingredients:
            return ingredients.compose(), ingredients.delta, inputs.route


def _decomposed_site_output(
    site_input: Array,
    frozen_weight: Array,
    inputs: _StagedKind,
    contraction: BlockContraction | None,
    placement: PlacementRules | None,
    target_linear: TargetLinearPlacement | None,
) -> Array:
    """One site through the core decomposed linear — the dense one for shared sites,
    the expert-blocked one (at the kind's declared orientation) for expert sites, each
    with its plans built from the rules when placed. Full-width masks only: narrow
    bundles belong to the routed decomposed arm (`require_full_emission` refuses)."""
    v, u = inputs.v, inputs.u
    full_mask, delta_mask, route = _entry_masking(inputs)
    mask = require_full_emission(full_mask)
    assert (placement is None) == (target_linear is None)
    W_plan = None if target_linear is None else target_linear_plan(site_input, target_linear)
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
                SiteWeights(frozen_weight, v, u, W_plan, dense_placement),
                mask,
                delta_mask,
                route,
            )
        case "fused_output" | "fused_input":
            expert_placement = None
            if placement is not None and target_linear is not None:
                expert_placement = BlockedPlannedComponentLinear(
                    v=placement.blocked_component_linear_plan(
                        target_linear, contraction, "V", site_input.ndim
                    ),
                    u=placement.blocked_component_linear_plan(
                        target_linear, contraction, "U", site_input.ndim
                    ),
                    component=placement.activations.component,
                    output=target_linear.output,
                )
            return blocked_site_forward(
                site_input,
                BlockedSiteWeights(frozen_weight, v, u, W_plan, expert_placement),
                mask,
                delta_mask,
                route,
                contraction,
            ).output


@dataclass(frozen=True)
class _SiteExecutor:
    """How one layer turns a matrix site into its output — the mixers' and the MoE's one
    injected concern. A kind with an entry runs decomposed (`_decomposed_site_output` on
    the entry's V/U and masking tensors); every other kind applies exactly its frozen
    `W` — not the `V@U + Δ` identity, so an undecomposed site carries no V/U gradient
    and no decomposition rounding. Both are placed by the kind's
    target-linear row (`_kind_target_linear`)."""

    cfg: Qwen36MoeConfig
    per_kind: dict[str, _StagedKind]
    placement: PlacementRules | None

    def __call__(self, kind: str, site_input: Array, frozen_weight: Array) -> Array:
        target_linear = _kind_target_linear(self.cfg, self.placement, kind)
        entry = self.per_kind.get(kind)
        if entry is None:
            return placed_target_linear(site_input, frozen_weight, target_linear)
        assert self.placement is None or target_linear is not None, (
            f"{kind!r} has no placed linear under this placement (the column row's tp "
            f"degree does not split the {self.cfg.n_kv_head} KV heads whole), so it "
            f"cannot be decomposed here"
        )
        return _decomposed_site_output(
            site_input,
            frozen_weight,
            entry,
            site_contraction(kind),
            self.placement,
            target_linear,
        )


def _layer_site_executor(
    cfg: Qwen36MoeConfig, layer: _CleanLayers | _MaskedLayer, placement: PlacementRules | None
) -> _SiteExecutor:
    match layer:
        case _CleanLayers():
            return _SiteExecutor(cfg, {}, placement)
        case _MaskedLayer(sites=sites):
            return _SiteExecutor(cfg, sites, placement)


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


def _dense_selected_values(value: SelectedCI, expert_weights: Array) -> Array:
    """Expand selected coefficients onto the computation's expert-sharded axis."""
    sharding = jax.typeof(expert_weights).sharding
    leading = jax.typeof(value.values).sharding.spec.partitions[:-1]
    blocks = value.values.reshape(*value.block_indices.shape, value.c_per_block)
    return jnp.einsum(
        "...kc,...ke->...ec",
        blocks,
        jax.nn.one_hot(value.block_indices, value.n_blocks, dtype=value.values.dtype),
        precision=jax.lax.Precision.HIGHEST,
        out_sharding=None
        if sharding.mesh.empty
        else NamedSharding(sharding.mesh, P(*leading, sharding.spec.partitions[0], None)),
    )


def _dense_routing_weights(indices: Array, weights: Array, n_experts: int) -> Array:
    return jnp.sum(
        jax.nn.one_hot(indices, n_experts, dtype=weights.dtype) * weights[..., None], axis=-2
    )


@dataclass(frozen=True)
class _DenseExpertActivations:
    cfg: Qwen36MoeConfig
    frozen_gate: Array
    frozen_up: Array
    router_indices: Array
    router_weights: Array
    input_values: Array

    def down_input(self) -> Array:
        cfg = self.cfg
        gate = dense_expert_matmul(
            self.input_values,
            self.frozen_gate.reshape(cfg.n_experts, cfg.moe_intermediate, cfg.n_embd).mT,
        )
        up = dense_expert_matmul(
            self.input_values,
            self.frozen_up.reshape(cfg.n_experts, cfg.moe_intermediate, cfg.n_embd).mT,
        )
        weights = _dense_routing_weights(self.router_indices, self.router_weights, cfg.n_experts)
        return _fold_routing_weights(gate, up, weights[..., None])

    def selected_values(self, inputs: Array, V: Array) -> SelectedCI:
        values = dense_select_experts(dense_expert_matmul(inputs, V), self.router_indices)
        return SelectedCI(
            values.reshape(*self.router_indices.shape[:-1], -1),
            self.router_indices,
            self.cfg.n_experts,
        )


@dataclass(frozen=True)
class _UnplacedExpertActivations:
    """One layer's job-space state for expert-site component activations (unplaced):
    the pinned routing's jobs schedule and gathered token rows, shared across the
    layer's expert kinds."""

    cfg: Qwen36MoeConfig
    frozen_gate: Array
    frozen_up: Array
    router_indices: Array
    router_weights: Array
    lead: tuple[int, ...]
    jobs: RoutedJobs
    input_values: Array
    backend: GroupedMatmulBackend

    def down_input(self) -> Array:
        """The FROZEN-path routed hidden `silu(gate)·up·weight` in job space — the down
        site's clean-forward input."""
        cfg = self.cfg
        d, di, n_experts = cfg.n_embd, cfg.moe_intermediate, cfg.n_experts
        gate = grouped_matmul(
            self.input_values,
            self.frozen_gate.reshape(n_experts, di, d).mT,
            self.jobs.group_sizes,
            self.backend,
        )
        up = grouped_matmul(
            self.input_values,
            self.frozen_up.reshape(n_experts, di, d).mT,
            self.jobs.group_sizes,
            self.backend,
        )
        weights = sort_jobs(self.router_weights.reshape(-1, cfg.n_experts_per_token), self.jobs)
        return _fold_routing_weights(gate, up, weights[:, None])

    def selected_values(self, input_jobs: Array, V: Array) -> SelectedCI:
        acts = grouped_matmul(input_jobs, V, self.jobs.group_sizes, self.backend)
        width = self.cfg.n_experts_per_token * V.shape[-1]
        values = unsort_jobs(acts, self.jobs).reshape(*self.lead, width)
        return SelectedCI(values, self.router_indices, self.cfg.n_experts)


@dataclass(frozen=True)
class _PlacedExpertActivations:
    """`_UnplacedExpertActivations`' expert-parallel sibling on the explicit mesh."""

    cfg: Qwen36MoeConfig
    frozen_gate: Array
    frozen_up: Array
    router_weights: Array
    shard_axis: str
    jobs: ExpertShardedJobs
    input_values: Array
    backend: GroupedMatmulBackend

    def down_input(self) -> Array:
        cfg = self.cfg
        d, di, n_experts = cfg.n_embd, cfg.moe_intermediate, cfg.n_experts
        gate = ep_grouped_matmul(
            self.input_values,
            self.frozen_gate.reshape(n_experts, di, d).mT,
            self.jobs,
            self.shard_axis,
            self.backend,
        )
        up = ep_grouped_matmul(
            self.input_values,
            self.frozen_up.reshape(n_experts, di, d).mT,
            self.jobs,
            self.shard_axis,
            self.backend,
        )
        weights = ep_sort_jobs(self.router_weights, self.jobs, self.shard_axis)
        return _fold_routing_weights(gate, up, weights[..., None])

    def selected_values(self, input_jobs: Array, V: Array) -> SelectedCI:
        acts = ep_grouped_matmul(input_jobs, V, self.jobs, self.shard_axis, self.backend)
        values = ep_unsort_jobs(acts, self.jobs, self.shard_axis)
        b, t, k, c = values.shape
        return SelectedCI(values.reshape(b, t, k * c), self.jobs.top_idx, self.cfg.n_experts)


def _expert_activation_context(
    moe_stack: FrozenMoE,
    cfg: Qwen36MoeConfig,
    captures: Mapping[str, Array],
    routing: BlockSelection,
    layer: int,
    placement: PlacementRules | None,
    backend: ExpertImplementation,
) -> _UnplacedExpertActivations | _PlacedExpertActivations | _DenseExpertActivations:
    router_indices = routing.indices[layer]
    router_weights = routing.weights[layer]
    h2 = captures[mlp_input_tap_key(layer)]
    frozen_gate = moe_stack.experts_gate[layer]
    frozen_up = moe_stack.experts_up[layer]
    lead = h2.shape[:-1]
    match backend:
        case "dense_masked":
            return _DenseExpertActivations(
                cfg,
                frozen_gate,
                frozen_up,
                router_indices,
                router_weights,
                h2[..., None, :],
            )
        case "ragged_dot" | "tokamax" | "tokamax_split_vjp":
            pass
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
            input_values=gather_tokens(h2.reshape(-1, cfg.n_embd), jobs),
            backend=backend,
        )
    shard_axis = _expert_shard_axis(placement)
    jobs = expert_sharded_jobs(router_indices, cfg.n_experts, placement.mesh.shape[shard_axis])
    return _PlacedExpertActivations(
        cfg=cfg,
        frozen_gate=frozen_gate,
        frozen_up=frozen_up,
        router_weights=router_weights,
        shard_axis=shard_axis,
        jobs=jobs,
        input_values=ep_gather_tokens(h2, jobs, shard_axis),
        backend=backend,
    )


# ----------------------------- the DecomposedModel -----------------------------


class QwenPreparedMasking(eqx.Module):
    """Per-kind layer stacks of mask recipes."""

    per_kind: dict[str, _StagedMasking]


class QwenPreparedWeights(eqx.Module):
    """Per decomposed kind, the stacked compute-layout `V` and `U`."""

    per_kind: dict[str, dict[str, Array]]

    def component_activations(
        self, site: str, x: Float[Array, "*leading d_in"]
    ) -> Float[Array, "*leading C"]:
        del x
        raise NotImplementedError(f"qwen36_moe does not serve component activations to CI: {site}")


class Qwen36MoeDecomposedModel(eqx.Module):
    """The qwen36_moe `DecomposedModel` (the `model.py` contract): the FROZEN full model
    as array fields — traced jit args, never HLO constants — with the trainable V/U
    passed to the forwards explicitly.

    The layer stack is stored in its periodic-stage layout: `deltanet` leaves lead with
    `[n_stages, interval−1, …]`, `attn` with `[n_stages, …]` (the stage scan's xs), and
    the uniform per-layer `moe` with `[n_layer, …]` (reshaped to stages per forward).
    A kind's frozen matrices read site-flat in its `layers_of_kind` order
    (`_frozen_kind_stack`) — the same order as its V/U stack. Requires each decomposed
    kind on every layer that has it; `shardings` is the Plan-A resident placement."""

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
    expert_implementation: ExpertImplementation = eqx.field(static=True)
    output_edge: OutputEdge = eqx.field(static=True)

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.sites)

    def shardings(self, placement: PlacementRules) -> ShardingTree:
        """The Plan-A resident placement: every frozen weight PERSISTS at its operand
        layout (÷tp within the node, replicated over data), so no while body ever
        gathers a weight. Expert-major fused axes shard whole experts; mixers shard by
        head (the KV projections replicate where tp does not split their heads whole,
        `_kv_target_linear`); embeddings/head/norms/router replicate per the resident
        rows. `deltanet` leaves lead `[n_stages, interval−1, …]`, `attn`
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

    # ---------- the expert-router readout (`router_divergence_eval.ExpertRouterModel`) ----------

    @property
    def expert_router_layers(self) -> tuple[int, ...]:
        """Every layer carries an MoE, so the pinned `BlockSelection`'s leading axis IS
        the layer axis."""
        return tuple(range(self.cfg.n_layer))

    @property
    def router_logits_capture_keys(self) -> tuple[str, ...]:
        return tuple(router_logits_tap_key(layer) for layer in range(self.cfg.n_layer))

    def router_probs_capture_key(self, layer: int) -> str:
        return router_probs_tap_key(layer)

    def router_weights_capture_key(self, layer: int) -> str:
        return router_weights_tap_key(layer)

    def select_experts(self, probs: Float[Array, "*lead E"]) -> Int[Array, "*lead k"]:
        return select_experts(probs, self.cfg.n_experts_per_token)

    # ---------- routing / MoE ----------

    def _dense_experts(
        self,
        moe: FrozenMoE,
        h2: Array,
        top_indices: Array,
        top_weights: Array,
        per_kind: dict[str, _StagedKind],
        captured_taps: frozenset[_Tap],
    ) -> tuple[Array | None, Array | None, Array]:
        cfg = self.cfg
        d, di, n_experts = cfg.n_embd, cfg.moe_intermediate, cfg.n_experts

        def site(kind: str, inputs: Array, frozen: Array) -> Array:
            entry = per_kind.get(kind)
            if entry is None:
                return dense_expert_matmul(inputs, frozen)
            mask, delta, route = _entry_masking(entry)
            assert isinstance(mask, SelectedCI), kind
            delta_e = None if delta is None else delta[..., None, None]
            coefficients = component_coefficients(_dense_selected_values(mask, entry.v), delta_e)
            acts = dense_expert_matmul(inputs, entry.v) * coefficients
            return blend_target_output(
                dense_expert_matmul(acts, entry.u),
                delta_e,
                None if route is None else route[..., None, None],
                lambda: dense_expert_matmul(inputs, frozen),
            )

        gate = site("experts_gate", h2[..., None, :], moe.experts_gate.reshape(n_experts, di, d).mT)
        up = site("experts_up", h2[..., None, :], moe.experts_up.reshape(n_experts, di, d).mT)
        down_blocks = moe.experts_down.reshape(d, n_experts, di).transpose(1, 2, 0)
        if _EXPERT_KINDS.isdisjoint(per_kind):
            down = dense_expert_matmul(jax.nn.silu(gate) * up, down_blocks)
            output = dense_combine_experts(down, top_indices, top_weights)
        else:
            weights = _dense_routing_weights(top_indices, top_weights, n_experts)
            hidden = _fold_routing_weights(gate, up, weights[..., None])
            down = site("experts_down", hidden, down_blocks)
            output = dense_combine_experts(down, top_indices, jnp.ones_like(top_weights))

        def fused_tap(tap: _Tap, values: Array) -> Array | None:
            if tap not in captured_taps:
                return None
            selected = _dense_routing_weights(top_indices, jnp.ones_like(top_weights), n_experts)
            return jnp.where(selected[..., None] != 0, values, 0).reshape(
                *h2.shape[:-1], n_experts * di
            )

        return (
            fused_tap(_Tap.EXPERTS_GATE_OUTPUT, gate),
            fused_tap(_Tap.EXPERTS_UP_OUTPUT, up),
            output,
        )

    def _routed_frozen_experts(
        self,
        moe: FrozenMoE,
        h2: Array,
        top_indices: Array,
        top_weights: Array,
        captured_taps: frozenset[_Tap],
        backend: GroupedMatmulBackend,
    ) -> tuple[Array | None, Array | None, Array]:
        """The frozen expert compute, routed: the lead axes flatten to one token axis,
        each token's k assignments become jobs, and the grouped matmuls touch selected
        experts only. Fused-width gate/up taps materialize ONLY when captured
        (`_LayerActs` documents their scattered semantics)."""
        cfg = self.cfg
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
        backend: GroupedMatmulBackend,
    ) -> tuple[Array | None, Array | None, Array]:
        """The routed frozen arm on the explicit mesh: EP by activation slicing. Each
        rank computes only its expert shard's jobs against its resident expert blocks
        (`ExpertShardedJobs` — one per-batch-row sentinel-sorted schedule per shard,
        replicated integers, data-parallel over the batch rows), and the partial outputs
        reduce across the expert-shard axis inside the fp32 combine, landing at the
        caller's waist `out_spec`. Zero weight movement and zero cross-data collectives
        by construction."""
        cfg = self.cfg
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
        per_kind: dict[str, _StagedKind],
        captured_taps: frozenset[_Tap],
        backend: GroupedMatmulBackend,
    ) -> tuple[Array | None, Array | None, Array]:
        """The decomposed expert compute on the SAME jobs schedule as the frozen arm:
        one job per (token, selected expert), per-job V_e/U_e grouped matmuls through
        the C_block bottleneck (the expert-blocked stacks are expert-major — exactly
        the grouped-matmul rhs layout), masks/CI gathered to job space (`[J, c]` — the
        k selected blocks per token), the frozen delta/route channels on the same
        grouped matmuls, routing weights folded into the down-site input in fp32
        (`_fold_routing_weights` — the identical math the all-expert dense
        spelling computes with the same fold, so the two differ by fp32 reassociation
        only; against the frozen arm's post-down fp32 combine the fold costs one bf16
        rounding of the fused input), and an unweighted per-token combine.
        Undecomposed expert kinds run their frozen grouped matmuls in the same job
        space. Fused-width gate/up taps materialize only when captured, with the
        scattered semantics `_LayerActs` documents."""
        cfg = self.cfg
        lead = h2.shape[:-1]
        d, di, n_experts = cfg.n_embd, cfg.moe_intermediate, cfg.n_experts
        k = cfg.n_experts_per_token
        jobs = routed_jobs(top_indices.reshape(-1, k), n_experts)
        x_jobs = gather_tokens(h2.reshape(-1, d), jobs)

        def site_jobs(kind: str, input_jobs: Array, frozen_blocks: Array) -> Array:
            entry = per_kind.get(kind)
            if entry is None:
                return grouped_matmul(input_jobs, frozen_blocks, jobs.group_sizes, backend)
            mask, delta_mask, route = _entry_masking(entry)
            # Delta/route may carry size-1 broadcast lead axes (batch-shared persistent
            # sources). The dense arm broadcasts them through its
            # elementwise ops; the job gathers need them at the full lead first — the
            # broadcast's transpose is the same cross-lead sum dense's autodiff performs.
            if delta_mask is not None:
                delta_mask = jnp.broadcast_to(delta_mask, lead)
            if route is not None:
                route = jnp.broadcast_to(route, lead)
            c = entry.v.shape[-1]
            # Slot m IS the token's m-th expert under the pinned routing this arm's jobs
            # schedule was built from, so job order is a pure permutation of the
            # (token, slot) rows — no partial gather, no dense view.
            assert isinstance(mask, SelectedCI), kind
            assert mask.values.shape[:-1] == lead, (mask.values.shape, lead)
            assert mask.block_indices.shape == top_indices.shape, (
                mask.block_indices.shape,
                top_indices.shape,
            )
            delta = None if delta_mask is None else gather_tokens(delta_mask.reshape(-1, 1), jobs)
            coefficients = component_coefficients(
                sort_job_values(mask.values.reshape(-1, k, c), jobs), delta
            )
            acts = grouped_matmul(input_jobs, entry.v, jobs.group_sizes, backend) * coefficients
            return blend_target_output(
                grouped_matmul(acts, entry.u, jobs.group_sizes, backend),
                delta,
                None if route is None else route.reshape(-1)[jobs.sort_idx // k][:, None],
                lambda: grouped_matmul(input_jobs, frozen_blocks, jobs.group_sizes, backend),
            )

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
        per_kind: dict[str, _StagedKind],
        captured_taps: frozenset[_Tap],
        placement: PlacementRules,
        out_spec: P,
        backend: GroupedMatmulBackend,
    ) -> tuple[Array | None, Array | None, Array]:
        """The routed decomposed arm on the explicit mesh: the expert-sharded schedule
        serves the decomposed compute exactly as it serves the frozen arm — each rank
        computes only its expert shard's jobs against its resident V/U blocks
        (co-located with the frozen experts at `expert: tp`). The component matmuls
        keep the stacks' master provenance (`reduced` typing), so dV/dU cotangents ride
        the loop unreduced and reduce once at the entry boundary — zero in-loop
        cross-data collectives, exactly as the dense placed arm's einsums arrange
        through the reduced-typing rule. Token-ordered selected masks are sorted into
        this forward's expert jobs using its pinned selection. The combine
        lands at the caller's waist `out_spec`."""
        cfg = self.cfg
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
            mask, delta_mask, route = _entry_masking(entry)
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
            c = entry.v.shape[-1]
            assert isinstance(mask, SelectedCI), kind
            assert mask.values.shape == (*lead, k * c), mask.values.shape
            assert mask.block_indices.shape == top_indices.shape
            assert mask.n_blocks == n_experts and mask.c_per_block == c
            delta = (
                None
                if delta_mask is None
                else ep_gather_tokens(delta_mask[..., None], jobs, shard_axis)
            )
            coefficients = component_coefficients(
                ep_sort_job_values(mask.values.reshape(*lead, k, c), jobs, shard_axis), delta
            )
            acts = ep_grouped_matmul(input_jobs, entry.v, jobs, shard_axis, backend) * coefficients
            return blend_target_output(
                ep_grouped_matmul(acts, entry.u, jobs, shard_axis, backend),
                delta,
                None
                if route is None
                else jnp.take_along_axis(route[:, None, :], jobs.sort_idx // k, axis=-1)[..., None],
                lambda: ep_grouped_matmul(input_jobs, frozen_blocks, jobs, shard_axis, backend),
            )

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

    def _moe_forward(
        self,
        moe: FrozenMoE,
        residual: Array,
        layer: _CleanLayers | _MaskedLayer,
        execute: _SiteExecutor,
        captured_taps: frozenset[_Tap],
        placement: PlacementRules | None,
        residual_row: PlacedRule | None,
    ) -> _MoeActs:
        """One layer's MoE. The router's fp32 softmax is computed once; a clean layer
        selects its experts from it, a masked layer is pinned to the clean forward's
        indices, and both weight the selected experts by that softmax gathered at
        those indices and renormalized — so mask≡1 with unperturbed upstream reproduces
        the clean routing exactly, and a perturbed upstream changes only the weights.
        Kinds in a masked layer's `per_kind` run decomposed, the rest frozen; the expert
        arm uses the explicitly selected dense or routed implementation. Frozen experts
        weight the down output; decomposed experts fold weights into the down input. `residual_row` is the pass's
        between-blocks residual placement: the MoE interior always runs at the full
        external width — the normed input gathers to it here (the jobs schedule and
        routing must see every token, and one gather serves every consumer, so its
        transpose is the layer's ONE input-cotangent reduction) — and the block-exit
        combines land back at the residual row."""
        cfg = self.cfg
        per_kind = execute.per_kind
        h2 = rms_norm_zero_centered(residual, moe.ln, cfg.rms_norm_eps)
        h2_full = constrain_activation(
            h2, None if placement is None else placement.activations.external
        )
        logits = router_logits(moe.router, h2_full)
        probs = jax.nn.softmax(logits, axis=-1)
        match layer:
            case _CleanLayers():
                top_indices = select_experts(probs, cfg.n_experts_per_token)
            case _MaskedLayer(pinned_indices=top_indices):
                pass
        top_weights = expert_mixing_weights(probs, top_indices)

        if placement is None:
            residual_spec = None
        else:
            assert residual_row is not None
            residual_spec = residual_row.spec_for(activation_axes(h2_full.ndim, "feature"))
        match self.expert_implementation:
            case "dense_masked":
                gate, up, routed = self._dense_experts(
                    moe,
                    h2_full,
                    top_indices,
                    top_weights,
                    per_kind,
                    captured_taps,
                )
            case ("ragged_dot" | "tokamax" | "tokamax_split_vjp") as backend:
                if _EXPERT_KINDS.isdisjoint(per_kind):
                    if residual_spec is None:
                        gate, up, routed = self._routed_frozen_experts(
                            moe, h2_full, top_indices, top_weights, captured_taps, backend
                        )
                    else:
                        assert placement is not None
                        gate, up, routed = self._routed_frozen_experts_placed(
                            moe,
                            h2_full,
                            top_indices,
                            top_weights,
                            captured_taps,
                            placement,
                            residual_spec,
                            backend,
                        )
                elif residual_spec is None:
                    gate, up, routed = self._routed_decomposed_experts(
                        moe,
                        h2_full,
                        top_indices,
                        top_weights,
                        per_kind,
                        captured_taps,
                        backend,
                    )
                else:
                    assert placement is not None
                    gate, up, routed = self._routed_decomposed_experts_placed(
                        moe,
                        h2_full,
                        top_indices,
                        top_weights,
                        per_kind,
                        captured_taps,
                        placement,
                        residual_spec,
                        backend,
                    )
        shared_gate = execute("shared_gate", h2_full, moe.shared_gate)
        shared_up = execute("shared_up", h2_full, moe.shared_up)
        shared_down = execute("shared_down", jax.nn.silu(shared_gate) * shared_up, moe.shared_down)
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
        return _MoeActs(
            moe_input=h2_full,
            router_indices=top_indices,
            router_logits=logits if _Tap.ROUTER_LOGITS in captured_taps else None,
            router_probs=probs if _Tap.ROUTER_PROBS in captured_taps else None,
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

    @jaxtyped(typechecker=beartype)
    def _output(
        self, residual: Float[Array, "batch seq d_model"], placement: PlacementRules | None
    ) -> LMOutput:
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
        return linear_output(residual, head, self.output_edge)

    def _forward(
        self,
        tokens: Int[Array, "b t"],
        layers: _CleanLayers | _MaskedLayers,
        ordered_capture_keys: tuple[str, ...],
        checkpoint_policy: Callable[..., bool] | None,
        placement: PlacementRules | None,
        residual_row: PlacedRule | None,
    ) -> tuple[LMOutput, tuple[Array, ...], BlockSelection]:
        """The one forward engine: a `lax.scan` over stages, each stage body unrolling
        its interval sublayers (mixer + MoE). `layers` is the pass's per-layer inputs,
        layer-stacked `[n_layer, …]` — `_CleanLayers` (all frozen, the router decides) or
        `_MaskedLayers` (the decomposed kinds' entries plus the conditioning expert indices);
        `checkpoint_policy` reruns stage bodies in the backward (the masked forwards).
        Returns the output edge, the requested captures in request order, and the
        routing every layer APPLIED (the clean forward's decision; a masked forward's
        pinned indices with its own recomputed weights). `residual_row` is the pass's
        between-blocks residual placement (`activations.external` for clean passes,
        `masked_external` for masked ones): the scan carry rides it, block interiors
        always run at the full external width — each block entry gathers the normed
        carry, each block exit's reduction lands back at the residual row — and the final
        residual gathers to external before the output edge, so a sequence-parallel
        residual never leaves this engine."""
        assert (residual_row is None) == (placement is None), (residual_row, placement)
        cfg = self.cfg
        sequence_parallel = (
            placement is not None and residual_row is not placement.activations.external
        )
        assert not (sequence_parallel and ordered_capture_keys), (
            "activation capture under sequence parallelism is not built (tap buffers are "
            "typed at the external rows); capture from a replicated-residual forward instead"
        )
        residual = constrain_activation(self._embed_tokens(tokens, placement), residual_row)

        sources = _capture_sources(ordered_capture_keys, cfg)
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
            # taps replicated at the external row — the `[.., E]` / `[.., k]` routing
            # taps ride there too, feature axis unlisted = replicated over tp), so the
            # in-scan dynamic updates are layout-preserving; `from_producer` re-pins at
            # exit.
            row = _tap_feature_row(placement, _Tap(tap), cfg)
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
        # `[n_layer, …] → [n_stages, interval, …]` (a DeltaNet kind's `[n_stages·
        # (interval−1), …] → [n_stages, interval−1, …]`; an attention kind's
        # `[n_stages, …]` is already per stage), a pure view of the resident stack, with
        # the positions split out INSIDE the body. Scan xs slicing is the one indexing
        # move with a `reduced`-typing rule, and the scan SAVES its xs for the
        # backward, outside every `jax.checkpoint` scope — so xs must be views, never
        # the pre-sliced position fragments (gather copies) that would keep a second
        # resident-stack-sized footprint alive across fwd→bwd. The in-body split runs
        # under the body's checkpoint, re-slicing at zero FLOPs in the backward.
        # `_stage_blocked`/`_block_positions` keep the entry-gather tags forward AND
        # provenance (`unreduced`) backward, so the component stacks' deferred
        # cross-`data` master reduction fires exactly once, at the materialize
        # boundary's transpose — never per fragment, never in the loop.
        moe_blocked = _stage_blocked_tree(self.moe, cfg.n_stages, interval)
        match layers:
            case _CleanLayers():
                layers_blocked: _CleanLayers | _MaskedLayers = layers
            case _MaskedLayers():
                layers_blocked = _MaskedLayers(
                    deltanet=_stage_blocked_tree(layers.deltanet, cfg.n_stages, interval - 1),
                    attn=layers.attn,
                    moe=_stage_blocked_tree(layers.moe, cfg.n_stages, interval),
                    pinned_indices=_stage_blocked(layers.pinned_indices, cfg.n_stages, interval),
                )
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
                _CleanLayers | _MaskedLayers,
                dict[str, Array],
            ],
        ) -> tuple[tuple[Array, dict[str, Array]], tuple[Array, Array]]:
            x, stage_buffers = state
            deltanet_block, attn_stage, moe_block, layers_block, slots_block = stage_inputs
            deltanet_layers = _tree_by_position(deltanet_block, interval - 1)
            moe_layers = _tree_by_position(moe_block, interval)
            layer_inputs = _stage_layer_inputs(layers_block, interval)
            external = None if placement is None else placement.activations.external
            routing_indices: list[Array] = []
            routing_weights: list[Array] = []
            for pos in range(interval):
                execute = _layer_site_executor(cfg, layer_inputs[pos], placement)
                # Mixers see the full external width (their sequence-mixing cores and
                # per-head tp splits demand every position); the normed residual gathers
                # here — one collective whose transpose is the mixer input's one
                # cotangent reduction — and the mixer's residual write lands back at the
                # pass's residual row (like the MoE's block exit: a site linear emits
                # the external row, never the residual row directly).
                if pos < interval - 1:
                    sublayer = deltanet_layers[pos]
                    mixer: _GdnActs | _AttnActs = sublayer.mixer(
                        constrain_activation(
                            rms_norm_zero_centered(x, sublayer.ln1, cfg.rms_norm_eps), external
                        ),
                        execute,
                    )
                    mixer_out, absent = mixer.out_output, _ATTN_TAPS
                else:
                    mixer = attn_stage.attn(
                        constrain_activation(
                            rms_norm_zero_centered(x, attn_stage.ln1, cfg.rms_norm_eps), external
                        ),
                        self.inv_freq,
                        execute,
                    )
                    mixer_out, absent = mixer.o_output, _GDN_TAPS
                post_mixer = x + constrain_activation(mixer_out, residual_row)
                moe_acts = self._moe_forward(
                    moe_layers[pos],
                    post_mixer,
                    layer_inputs[pos],
                    execute,
                    captured_taps,
                    placement,
                    residual_row,
                )
                x = moe_acts.residual_out
                routing_indices.append(moe_acts.router_indices)
                routing_weights.append(moe_acts.router_weights)
                if stage_buffers:
                    stage_buffers = _write_captures(
                        stage_buffers,
                        {tap: slots_block[tap][pos] for tap in stage_buffers},
                        _LayerActs(mixer=mixer, moe=moe_acts),
                        absent,
                    )
            return (x, stage_buffers), (jnp.stack(routing_indices), jnp.stack(routing_weights))

        body = (
            jax.checkpoint(stage_body, policy=checkpoint_policy)
            if checkpoint_policy is not None
            else stage_body
        )
        (residual, buffers), (indices_by_stage, weights_by_stage) = jax.lax.scan(
            body,
            (residual, buffers),
            (self.deltanet, self.attn, moe_blocked, layers_blocked, slots_blocked),
        )
        routing = BlockSelection(
            indices=indices_by_stage.reshape(cfg.n_layer, *indices_by_stage.shape[2:]),
            weights=weights_by_stage.reshape(cfg.n_layer, *weights_by_stage.shape[2:]),
        )

        for source in sources:
            if source.tap is not _Tap.RESIDUAL_IN:
                captured[source] = buffers[source.tap.value][layout[source.tap.value][source.layer]]
        residual = constrain_activation(
            rms_norm_zero_centered(residual, self.norm, cfg.rms_norm_eps),
            None if placement is None else placement.activations.external,
        )
        return (
            self._output(residual, placement),
            tuple(captured[source] for source in sources),
            routing,
        )

    def _tap_widths(self) -> dict[str, int]:
        cfg = self.cfg
        widths = {
            _Tap.MIXER_INPUT.value: cfg.n_embd,
            _Tap.GDN_CORE_OUTPUT.value: cfg.linear_value_dim,
            _Tap.ATTN_CORE_OUTPUT.value: cfg.n_head * cfg.head_dim,
            _Tap.MOE_INPUT.value: cfg.n_embd,
            _Tap.ROUTER_LOGITS.value: cfg.n_experts,
            _Tap.ROUTER_PROBS.value: cfg.n_experts,
            _Tap.ROUTER_WEIGHTS.value: cfg.n_experts_per_token,
            _Tap.RESIDUAL_OUT.value: cfg.n_embd,
        }
        for kind, tap in _SITE_OUTPUT_TAP.items():
            widths[tap.value] = site_dims(cfg, kind).d_out
        return widths

    # ---------- protocol forwards ----------

    def clean_forward(
        self,
        inputs: LMBatch,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        *,
        placement: PlacementRules | None,
    ) -> ForwardResult[LMOutput, LMBatchWithRouting[LMBatch]]:
        ordered_capture_keys = tuple(sorted(capture_keys))
        output, capture_values, routing = self._forward(
            inputs.token_ids,
            _CleanLayers(),
            ordered_capture_keys,
            checkpoint_policy=None,
            placement=placement,
            residual_row=None if placement is None else placement.activations.external,
        )
        return ForwardResult.from_producer(
            leading_shape=inputs.token_ids.shape,
            output=output,
            capture_keys=ordered_capture_keys,
            capture_values=capture_values,
            conditioning=LMBatchWithRouting(inputs, routing),
            sequence=None,
        )

    def prepare_compute_weights(
        self, vu: ComponentStacks, placement: PlacementRules | None
    ) -> QwenPreparedWeights:
        """Placed: the ÷data→resident entry gather runs ONCE per step here (off the hot
        path), landing ÷tp-resident stacks typed `reduced` over the gathered axes. The
        tags ride the scan xs into the stage loop (`_stage_blocked` and the in-body
        `_block_positions` keep them across the entry-side re-layout, provenance-
        preserving in both directions), so component-weight cotangents stay
        replica-local through the whole backward and reduce exactly once, at this
        boundary's transpose."""
        if placement is None:
            return QwenPreparedWeights(self._stack_per_kind_vu(vu))
        return QwenPreparedWeights(
            self._stack_per_kind_vu(component_stacks_to_compute_weights(vu, placement.components))
        )

    def _stack_per_kind_vu(self, components: ComponentStacks) -> dict[str, dict[str, Array]]:
        """Per decomposed KIND, the stacked (V, U) — the group stacks directly, because
        canonical order makes each kind's slot axis its `layers_of_kind` order. Full
        coverage per kind (every layer that has the kind) is this target's
        masked-forward contract."""
        assert components.site_names == self.site_names, (
            components.site_names,
            self.site_names,
        )
        for name, group, slot in components.site_stack_indices:
            layer, kind = parse_site_name(name)
            covered = layers_of_kind(self.cfg, kind)
            assert group == kind and slot < len(covered) and covered[slot] == layer, (
                f"qwen36_moe requires each decomposed kind on EVERY layer that has it "
                f"({kind!r}: layers {covered}); {name} sits at slot {slot}"
            )
        for group, length in components.group_lengths().items():
            covered = layers_of_kind(self.cfg, group)
            assert length == len(covered), (
                f"qwen36_moe requires each decomposed kind on EVERY layer that has it; "
                f"{group!r} covers {length} of its {len(covered)} layers {covered}"
            )
        return {kind: {"V": Vs, "U": Us} for kind, (Vs, Us) in components.stacks.items()}

    def _stack_sites[T: SiteCI | _StagedMasking | Array](
        self, values: Mapping[str, T]
    ) -> dict[str, T]:
        assert set(values) == set(self.site_names), (sorted(values), sorted(self.site_names))
        kinds = {parse_site_name(name)[1] for name in values}
        stacked: dict[str, T] = {}
        for kind in kinds:
            layers = [values[site_name(layer, kind)] for layer in layers_of_kind(self.cfg, kind)]
            assert len({jax.tree.structure(value) for value in layers}) == 1, (
                f"kind {kind!r} mixes payload structures"
            )
            stacked[kind] = jax.tree.map(lambda *leaves: jnp.stack(leaves), *layers)
        return stacked

    @staticmethod
    def _prepare_stochastic(
        ci_stacked: Mapping[str, SiteCI],
        draw_key: Array,
        layers_by_kind: Mapping[str, tuple[int, ...]],
    ) -> dict[str, _StagedMasking]:
        """Stage per-layer keys; checkpointed bodies draw and redraw the same masks."""
        src_base, delta_base = jax.random.split(draw_key)
        per_kind: dict[str, _StagedMasking] = {}
        for kind, ci in ci_stacked.items():
            kind_index = KIND_ORDER.index(kind)
            layers = layers_by_kind[kind]
            recipe = _StagedStochastic(
                ci=_require_kind_emission(kind, ci),
                src_key=jnp.stack(
                    [
                        jax.random.fold_in(jax.random.fold_in(src_base, kind_index), layer)
                        for layer in layers
                    ]
                ),
                delta_key=jnp.stack(
                    [
                        jax.random.fold_in(jax.random.fold_in(delta_base, kind_index), layer)
                        for layer in layers
                    ]
                ),
            )
            per_kind[kind] = recipe
        return per_kind

    def prepare_masking(self, masking: Masking) -> QwenPreparedMasking:
        per_site: dict[str, _StagedMasking] = {}
        match masking:
            case StochasticMasking(ci=ci, draw_key=draw_key):
                return self.prepare_stochastic_masking(ci)(draw_key)
            case SourceMasking(ingredients=ingredients):
                for name, pair in ingredients.items():
                    _require_kind_emission(parse_site_name(name)[1], pair.ci)
                    per_site[name] = pair
            case MaterializedMasking(component_masks=masks, weight_delta_masks=deltas):
                per_site = {
                    name: _StagedMaterialized(
                        _require_kind_emission(parse_site_name(name)[1], mask),
                        None if deltas is None else deltas[name],
                    )
                    for name, mask in masks.items()
                }
        return QwenPreparedMasking(self._stack_sites(per_site))

    def prepare_stochastic_masking(
        self, ci: Mapping[str, SiteCI]
    ) -> Callable[[Array], QwenPreparedMasking]:
        stacked = self._stack_sites(ci)
        layers_by_kind = {kind: layers_of_kind(self.cfg, kind) for kind in stacked}

        def draw(draw_key: Array) -> QwenPreparedMasking:
            return QwenPreparedMasking(
                Qwen36MoeDecomposedModel._prepare_stochastic(stacked, draw_key, layers_by_kind)
            )

        return draw

    def _masked_layers(
        self,
        per_kind: dict[str, _StagedKind],
        pinned_indices: Array,
    ) -> _MaskedLayers:
        """The attached per-kind entries sorted onto their sublayer's stack."""
        by_sublayer: dict[Sublayer, dict[str, _StagedKind]] = {
            "deltanet": {},
            "attn": {},
            "moe": {},
        }
        for kind, entry in per_kind.items():
            by_sublayer[sublayer_of(kind)][kind] = entry
        return _MaskedLayers(
            deltanet=by_sublayer["deltanet"],
            attn=by_sublayer["attn"],
            moe=by_sublayer["moe"],
            pinned_indices=pinned_indices,
        )

    def masked_forward(
        self,
        prepared_weights: QwenPreparedWeights,
        conditioning: LMBatchWithRouting[LMBatch],
        /,
        *,
        masking: QwenPreparedMasking,
        routes: SiteRoutes | None,
        placement: PlacementRules | None,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        remat: bool,
    ) -> ForwardResult[LMOutput, LMBatchWithRouting[LMBatch]]:
        inputs = conditioning.batch.token_ids
        selection = conditioning.selection
        cfg = self.cfg
        routing_shape = (cfg.n_layer, *inputs.shape, cfg.n_experts_per_token)
        assert selection.indices.shape == routing_shape and selection.indices.dtype == jnp.int32, (
            "pinned routing indices must be int32 [n_layer, *batch, k] for this batch",
            selection.indices.shape,
            selection.indices.dtype,
            routing_shape,
        )
        assert (
            selection.weights.shape == routing_shape and selection.weights.dtype == jnp.float32
        ), (
            "pinned routing weights must be fp32 [n_layer, *batch, k] for this batch",
            selection.weights.shape,
            selection.weights.dtype,
            routing_shape,
        )
        stacks = prepared_weights.per_kind
        assert set(masking.per_kind) == set(stacks)
        validate_routes(routes, self.site_names)
        route_stacks = None if routes is None else self._stack_sites(routes)
        per_kind = {
            kind: _StagedKind(
                stacks[kind]["V"],
                stacks[kind]["U"],
                entry,
                None if route_stacks is None else route_stacks[kind],
            )
            for kind, entry in masking.per_kind.items()
        }
        policy = (
            jax.checkpoint_policies.nothing_saveable
            if remat
            else jax.checkpoint_policies.dots_saveable
        )
        ordered_capture_keys = tuple(sorted(capture_keys))
        output, capture_values, _applied = self._forward(
            inputs,
            self._masked_layers(per_kind, selection.indices),
            ordered_capture_keys,
            checkpoint_policy=policy,
            placement=placement,
            residual_row=None if placement is None else placement.activations.masked_external,
        )
        return ForwardResult.from_producer(
            leading_shape=inputs.shape,
            output=output,
            capture_keys=ordered_capture_keys,
            capture_values=capture_values,
            conditioning=conditioning,
            sequence=None,
        )

    def component_activation_forward(
        self,
        prepared_weights: QwenPreparedWeights,
        inputs: LMBatch,
        /,
        *,
        sites: tuple[str, ...],
        capture_keys: CaptureKeys,
        placement: PlacementRules | None,
    ) -> tuple[ForwardResult[LMOutput, LMBatchWithRouting[LMBatch]], dict[str, SiteCI]]:
        """Run one frozen forward for requested captures and each requested site's
        ``x @ V``: full `[.., C]` for shared sites; selected bundles for expert sites,
        retaining token/selected-slot order under every placement. Down
        sites consume the FROZEN-path routed hidden (recomputed from the captured MoE
        input — the clean forward's own values, matching the dense transformer pattern of measuring
        activations against frozen-path site inputs)."""
        assert set(sites) <= set(self.site_names), (sorted(sites), self.site_names)
        cfg = self.cfg
        locations = tuple(parse_site_name(site) for site in sites)
        needed: set[str] = set()
        for layer, kind in locations:
            match kind:
                case "gdn_q" | "gdn_k" | "gdn_v" | "gdn_z" | "gdn_b" | "gdn_a":
                    needed.add(attention_input_tap_key(layer))
                case "attn_q" | "attn_k" | "attn_v":
                    needed.add(attention_input_tap_key(layer))
                case "gdn_out" | "attn_o":
                    needed.add(attention_output_tap_key(layer))
                case "experts_gate" | "experts_up" | "experts_down":
                    needed.add(mlp_input_tap_key(layer))
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
        def expert_context(
            layer: int,
        ) -> _UnplacedExpertActivations | _PlacedExpertActivations | _DenseExpertActivations:
            return _expert_activation_context(
                self.moe,
                cfg,
                captures,
                full.conditioning.selection,
                layer,
                placement,
                self.expert_implementation,
            )

        component_activations: dict[str, SiteCI] = {}
        for site, (layer, kind) in zip(sites, locations, strict=True):
            V = unreduce(prepared_weights.per_kind[kind]["V"])[
                layers_of_kind(cfg, kind).index(layer)
            ]
            if is_expert_kind(kind):
                context = expert_context(layer)
                expert_input = (
                    context.down_input() if kind == "experts_down" else context.input_values
                ).astype(V.dtype)
                component_activations[site] = context.selected_values(expert_input, V)
                continue
            match kind:
                case "gdn_q" | "gdn_k" | "gdn_v" | "gdn_z" | "gdn_b" | "gdn_a":
                    site_input = captures[attention_input_tap_key(layer)]
                case "attn_q" | "attn_k" | "attn_v":
                    site_input = captures[attention_input_tap_key(layer)]
                case "gdn_out" | "attn_o":
                    site_input = captures[attention_output_tap_key(layer)]
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
            leading_shape=full.leading_shape,
            output=full.output,
            captures={key: captures[key] for key in sorted(capture_keys)},
            conditioning=full.conditioning,
            sequence=full.sequence,
        )
        return requested, component_activations

    # ---------- frozen-weight views ----------

    def _frozen_kind_stack(self, kind: str) -> Array:
        """One kind's frozen matrices stacked in its `layers_of_kind` order,
        `[n_sites, d_out, d_in]`: the MoE stacks as stored; a DeltaNet kind's
        `[n_stages, interval−1, …]` storage flattened layer-major (a view); an
        attention kind's `[n_stages, …]` as stored."""
        cfg = self.cfg
        deltanet = self.deltanet.mixer
        attn = self.attn.attn

        def flat(leaf: Array) -> Array:
            return leaf.reshape(cfg.n_stages * (cfg.full_attention_interval - 1), *leaf.shape[2:])

        match kind:
            case "gdn_q":
                return flat(deltanet.w_q)
            case "gdn_k":
                return flat(deltanet.w_k)
            case "gdn_v":
                return flat(deltanet.w_v)
            case "gdn_z":
                return flat(deltanet.w_z)
            case "gdn_b":
                return flat(deltanet.w_b)
            case "gdn_a":
                return flat(deltanet.w_a)
            case "gdn_out":
                return flat(deltanet.w_out)
            case "attn_q":
                return attn.wq
            case "attn_k":
                return attn.wk
            case "attn_v":
                return attn.wv
            case "attn_o":
                return attn.wo
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
        groups = {group for _name, group, _slot in site_stack_indices_for(self.sites)}
        return {
            group: jnp.sum(self._frozen_kind_stack(group).astype(jnp.float32) ** 2, axis=(1, 2))
            for group in sorted(groups)
        }

    def weight_deltas(self, vu: ComponentStacks) -> dict[str, Array]:
        """fp32 `W − V@U` per kind stack (slot axis = the kind's `layers_of_kind` order).
        DenseFactorization kinds stack whole matrices `[g, d_out, d_in]`; expert kinds
        stack per-expert blocks `[g, expert, d_out, d_in]` — the blocks partition the
        fused matrix, so per-slot Frobenius reductions agree.

        The frozen stack keeps its resident dtype until it has landed on the delta row
        and taken the blocked layout; the fp32 convert commutes exactly with slicing
        and relayout, and happens at the subtract, where it fuses. Converting the
        resident stack first would pin a whole-stack fp32 copy behind the barrier."""
        for name, group, slot in vu.site_stack_indices:
            layer, kind = parse_site_name(name)
            assert group == kind and layers_of_kind(self.cfg, kind)[slot] == layer, (
                name,
                group,
                slot,
            )

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
    expert_implementation: ExpertImplementation,
    output_edge: OutputEdge,
) -> Qwen36MoeDecomposedModel:
    """Assemble JAX per-layer parts into the runtime's stage-stacked storage."""
    site_cs = tuple(SiteC(spec.name, spec.C) for spec in sites)
    expected = qwen36_moe_site_specs(cfg, canonical_site_cs(site_cs))
    assert sites == expected, f"sites are not the canonical specs for this config: {sites}"
    interval = cfg.full_attention_interval
    assert len(deltanet_sublayers) == cfg.n_stages * (interval - 1), len(deltanet_sublayers)
    assert len(attn_sublayers) == cfg.n_stages, len(attn_sublayers)
    assert len(moe_layers) == cfg.n_layer, len(moe_layers)

    deltanet_flat = jax.tree.map(lambda *leaves: jnp.stack(leaves), *deltanet_sublayers)
    deltanet = jax.tree.map(
        lambda a: a.reshape(cfg.n_stages, interval - 1, *a.shape[1:]), deltanet_flat
    )
    attn = jax.tree.map(lambda *leaves: jnp.stack(leaves), *attn_sublayers)
    moe = jax.tree.map(lambda *leaves: jnp.stack(leaves), *moe_layers)
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
        expert_implementation=expert_implementation,
        output_edge=output_edge,
    )


def _load_moe(get: Callable[[str], Array], prefix: str, cfg: Qwen36MoeConfig) -> FrozenMoE:
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


def _load_deltanet(
    get: Callable[[str], Array], prefix: str, cfg: Qwen36MoeConfig
) -> DeltaNetSublayer:
    conv_weight = get(f"{prefix}.linear_attn.conv1d.weight")
    kd, vd = cfg.linear_key_dim, cfg.linear_value_dim
    conv_dim = 2 * kd + vd
    assert conv_weight.shape == (conv_dim, 1, cfg.linear_conv_kernel_dim), conv_weight.shape
    conv = conv_weight.reshape(conv_dim, cfg.linear_conv_kernel_dim)
    # HF fuses q|k|v rows in the in_proj and its conv channels; stored split here
    # (bit-identical per-piece compute — `GatedDeltaNet`).
    w_qkv = get(f"{prefix}.linear_attn.in_proj_qkv.weight")
    assert w_qkv.shape[0] == conv_dim, w_qkv.shape
    return DeltaNetSublayer(
        ln1=get(f"{prefix}.input_layernorm.weight"),
        mixer=GatedDeltaNet(
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
    get: Callable[[str], Array],
    prefix: str,
    cfg: Qwen36MoeConfig,
    implementation: AttentionImplementation,
) -> AttnSublayer:
    return AttnSublayer(
        ln1=get(f"{prefix}.input_layernorm.weight"),
        attn=GatedAttention(
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


@cpu_staging()
def build_qwen36_moe_from_weights(
    cfg: Qwen36MoeConfig,
    sites: tuple[SiteSpec, ...],
    get: Callable[[str], Array],
    *,
    decoder_prefix: str,
    implementation: AttentionImplementation,
    expert_implementation: ExpertImplementation,
    output_edge: OutputEdge,
) -> Qwen36MoeDecomposedModel:
    """Build from HF-keyed weights: `decoder_prefix` is `model.language_model` in the
    real `Qwen3_5MoeForConditionalGeneration` checkpoint and `model` under the text-only
    `Qwen3_5MoeForCausalLM` (the parity fixtures and the lab-pretrained toys). The reader
    returns CPU JAX arrays; assembly and generated constants remain on CPU."""
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
        expert_implementation=expert_implementation,
        output_edge=output_edge,
    )


def abstract_qwen36_moe_model(
    cfg: Qwen36MoeConfig,
    sites: tuple[SiteSpec, ...],
    weights_dtype: DTypeLike,
    expert_implementation: ExpertImplementation,
    output_edge: OutputEdge,
    attention_implementation: AttentionImplementation,
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
            mixer=GatedDeltaNet(
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
            attn=GatedAttention(
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
                implementation=attention_implementation,
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
            expert_implementation=expert_implementation,
            output_edge=output_edge,
        )

    return eqx.filter_eval_shape(build)


@cpu_staging()
def load_decomposed_qwen36_moe_from_hf(
    model_name: str,
    cfg: Qwen36MoeConfig,
    sites: tuple[SiteSpec, ...],
    weights_dtype: DTypeLike,
    implementation: AttentionImplementation,
    expert_implementation: ExpertImplementation,
    output_edge: OutputEdge,
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
        expert_implementation=expert_implementation,
        output_edge=output_edge,
    )


def qwen36_moe_config_from_pretrain_cache(cache_dir: Path) -> Qwen36MoeConfig:
    """The arch of a lab-pretrained `Qwen35Moe` toy, from its pretrain-cache entry's
    `model_config.yaml` (the pretrainer's own config, dumped verbatim). Only the
    architecture fields are read: `block_size`, the router-loss coefficients and
    `tied_head` are training facts (a tied toy still exports `lm_head.weight`)."""
    raw = yaml.safe_load((cache_dir / "model_config.yaml").read_text())
    assert raw["model_type"] == "Qwen35Moe", raw["model_type"]
    return Qwen36MoeConfig(
        vocab_size=raw["vocab_size"],
        n_layer=raw["n_layer"],
        full_attention_interval=raw["full_attention_interval"],
        n_embd=raw["n_embd"],
        n_head=raw["n_head"],
        n_kv_head=raw["n_kv_head"],
        head_dim=raw["head_dim"],
        partial_rotary_factor=raw["partial_rotary_factor"],
        rope_theta=raw["rope_theta"],
        linear_num_key_heads=raw["linear_num_key_heads"],
        linear_key_head_dim=raw["linear_key_head_dim"],
        linear_num_value_heads=raw["linear_num_value_heads"],
        linear_value_head_dim=raw["linear_value_head_dim"],
        linear_conv_kernel_dim=raw["linear_conv_kernel_dim"],
        n_experts=raw["n_experts"],
        n_experts_per_token=raw["n_experts_per_tok"],
        moe_intermediate=raw["moe_intermediate"],
        shared_expert_intermediate=raw["shared_expert_intermediate"],
        rms_norm_eps=raw["rms_norm_eps"],
        max_position_embeddings=raw["n_ctx"],
    )


@cpu_staging()
def load_decomposed_qwen36_moe_from_pretrain_cache(
    cache_dir: Path,
    cfg: Qwen36MoeConfig,
    sites: tuple[SiteSpec, ...],
    weights_dtype: DTypeLike,
    implementation: AttentionImplementation,
    expert_implementation: ExpertImplementation,
    output_edge: OutputEdge,
) -> Qwen36MoeDecomposedModel:
    """Load a lab-pretrained `Qwen35Moe` toy from its pretrain-cache entry's one
    `model_step_*.safetensors`: HF-keyed under the text-only `model` prefix, with
    `lm_head.weight` always present (a tied toy exports its embedding copy)."""
    [checkpoint] = cache_dir.glob("model_step_*.safetensors")
    handle = safe_open(str(checkpoint), framework="numpy")

    def get(key: str) -> Array:
        with cpu_staging():
            return jax.device_put(
                np.asarray(handle.get_tensor(key), dtype=weights_dtype),
                jax.local_devices(backend="cpu")[0],
            )

    return build_qwen36_moe_from_weights(
        cfg,
        sites,
        get,
        decoder_prefix="model",
        implementation=implementation,
        expert_implementation=expert_implementation,
        output_edge=output_edge,
    )
