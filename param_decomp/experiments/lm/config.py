"""LM experiment config schema (target spec, data settings, tiled site specs, full YAML
tree) PLUS the LM YAML→`BuiltRun` conversion.

This module reads the canonical `LMExperimentConfig` schema directly and builds the engine's
`BuiltRun` bundle (`param_decomp.core.built_run`) — the pydantic `pd` / `cadence`
verbatim plus the resolved target / data / CI-fn arch / eval — asserting loudly on anything
the JAX trainer doesn't implement. The composition entry (`run.py`) calls `load_config` /
`build_from_schema`; stored-run consumers rebuild the same canonical schema.

The authored `decomposition.sites` c-specs (`GluTransformerCSpec` / `SimpleMlpCSpec`, keys
typed by each target family's own matrix vocabulary) resolve here into the block-structured
`SiteTree` (`resolve_site_tree`) — the layer index carried as DATA, never parsed back out
of a site name — which the CI-arch resolvers (`resolve_lm_ci_fn_arch`) consume directly.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import equinox as eqx
import jax
import yaml
from jax.typing import DTypeLike
from pydantic import (
    Discriminator,
    Field,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

from param_decomp.attention import AttentionImplementation
from param_decomp.core import placement
from param_decomp.core.base_config import BaseConfig
from param_decomp.core.built_run import BuiltRun
from param_decomp.core.ci_fn.implementations.block_selected.arch import (
    BlockSelectedChunk,
    BlockSelectedChunkSlot,
    BlockSelectedChunkwiseTransformerCIFnArch,
    FullSlot,
    SelectedSlot,
)
from param_decomp.core.ci_fn.implementations.chunkwise.arch import (
    Chunk,
    ChunkwiseTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.global_mlp import GlobalMLPCIFnArch
from param_decomp.core.ci_fn.implementations.global_transformer.arch import (
    GlobalTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.transformer.component_conditioning import (
    ConditionedCIFnArch,
    InputScaleCalibration,
    SiteInput,
)
from param_decomp.core.ci_fn.implementations.transformer.layers import (
    CIFnAttentionMask,
    GQACIFnAttention,
    MHACIFnAttention,
)
from param_decomp.core.ci_fn.interface import TapSpec
from param_decomp.core.ci_fn.optimizer import assert_ci_fn_muon_staging_tiles
from param_decomp.core.components import SiteC, SiteSpec
from param_decomp.core.configs import (
    AdamWOptimizerConfig,
    AnyLossMetricConfig,
    AnyPDConfig,
    MuonOptimizerConfig,
    NontargetConfig,
    ResumeProvenance,
    TargetedLossMetricConfig,
    TargetedPDConfig,
)
from param_decomp.core.family import ArchFamily
from param_decomp.core.objective import (
    PDObjective,
    TargetedPDObjective,
    build_objective,
    build_targeted_objective,
)
from param_decomp.core.sharding import abstract_mesh_for_shape
from param_decomp.experiments.config import (
    ExperimentConfig,
    ExperimentConfigBase,
    run_instance,
)
from param_decomp.experiments.eval_config import EvalConfig
from param_decomp.experiments.lm.eval_config import (
    RouterDivergenceConfig,
    assert_router_divergence_persistent_terms_exist,
)
from param_decomp.experiments.lm.resolved import (
    AnyLMTargetConfig,
    HFSnapshotWeights,
    LlamaSimpleMLPTargetConfig,
    LMCIFnArch,
    LMRun,
    LMTargetedRun,
    PretrainCacheWeights,
    Qwen36MoeTargetConfig,
    Qwen36MoeWeights,
    ResolvedLMData,
    TargetConfig,
    WeightsDtype,
    dense_transformer_anatomy,
)
from param_decomp.experiments.lm.run_data import LMDataConfig
from param_decomp.experiments.lm.runtime import RuntimeConfig
from param_decomp.experiments.lm.targeted_data import LMPromptPoolConfig
from param_decomp.infra import pretrain_cache
from param_decomp.infra.dataset_store import resolve_dataset_ref
from param_decomp.migrations.schedule_knots import migrate_raw as migrate_schedule_knots
from param_decomp.routed.experts import ExpertImplementation
from param_decomp.target_ports.llama import LlamaConfig
from param_decomp.targets import llama31, llama_simple_mlp, qwen3, qwen36_moe, transformer
from param_decomp.targets.llama_simple_mlp import SimpleMlpMatrix
from param_decomp.targets.lm_output import MaterializedOutputEdge, OutputEdge, StreamedOutputEdge
from param_decomp.targets.qwen36_moe import Qwen36MoeConfig, Qwen36MoeMatrix
from param_decomp.targets.transformer import (
    GluMatrix,
    HFTransformerArch,
    TransformerConfig,
    TransformerDecomposedModel,
)
from param_decomp.targets.transformer_taps import TransformerTapGrammar, resid_tap_key


class HFTarget(BaseConfig):
    """Load a HuggingFace model via `<model_class>.from_pretrained(<model_name>)`."""

    kind: Literal["hf"] = "hf"
    model_class: str
    model_name: str


class PretrainedTarget(BaseConfig):
    """Load an in-repo lab-pretrained model (`param_decomp.pretrain.train`'s output)."""

    kind: Literal["pretrained"] = "pretrained"
    model_class: str
    run_path: str
    """`entity/project[/runs]/run_id` — the W&B pretrain run whose checkpoint is the
    target's weights. A name, never a location: it resolves to the local store entry
    `<data_root>/pretrain_cache/<project>-<run_id>`, fetched from W&B on first use if
    not already there (`infra.pretrain_cache`), read from disk ever after."""


class HFWeightsInVendored(BaseConfig):
    """Load HF pretrained weights into the vendored `VendoredLlama` architecture.

    Llama-3.1-8B only — `resolve_decomposition` asserts the class and model name;
    other HF families go through `kind: hf`.
    """

    kind: Literal["hf_weights_in_vendored"] = "hf_weights_in_vendored"
    model_class: str  # must be `VendoredLlama`
    model_name: str  # HF hub id


class PretrainedQwen35MoeTarget(BaseConfig):
    """Load a lab-pretrained `Qwen35Moe` toy into the qwen36_moe target engine: its arch
    from the pretrain-cache entry's `model_config.yaml`, its weights from the entry's
    safetensors. Takes a `qwen36_moe` c-spec."""

    kind: Literal["pretrained_qwen35_moe"] = "pretrained_qwen35_moe"
    run_path: str
    """As `PretrainedTarget.run_path`: the W&B pretrain run, resolved to
    `<data_root>/pretrain_cache/<project>-<run_id>`."""


LMTargetSpec = Annotated[
    HFTarget | PretrainedTarget | HFWeightsInVendored | PretrainedQwen35MoeTarget,
    Discriminator("kind"),
]


class MaterializedOutputEdgeConfig(BaseConfig):
    """The materialized model-output edge: every forward forms its full `[B, S, vocab]`
    logits in the target's native dtype (bf16 on a bf16 target), cast to fp32 at the
    comparison kernels."""

    kind: Literal["materialized"] = "materialized"


class StreamedOutputEdgeConfig(BaseConfig):
    """The streamed model-output edge: forwards return the
    factored {final activations, unembedding} package and every output comparison — the
    recon KL and the eval CE/KL variants — streams over vocab chunks with fp32 online
    accumulators, so no `[B, S, vocab]` buffer ever materializes. Each chunk's logits are
    fp32-accumulated from the native-dtype operands, so this edge differs from
    `materialized` by the one bf16 rounding of the logits the materialized edge carries
    (`targets.losses` documents the recurrences and the seam)."""

    kind: Literal["streamed"] = "streamed"
    n_vocab_chunks: PositiveInt
    """Chunks the vocab axis streams in; must divide the target's vocab size
    (248320 = 2^9·5·97 — e.g. 32 chunks of 7760)."""


LMOutputEdgeConfig = Annotated[
    MaterializedOutputEdgeConfig | StreamedOutputEdgeConfig, Discriminator("kind")
]


DEFAULT_EXPERT_IMPLEMENTATION: ExpertImplementation = "tokamax_split_vjp"


class LMTargetConfig(BaseConfig):
    """Config for the LM target model."""

    spec: LMTargetSpec
    expert_implementation: ExpertImplementation = DEFAULT_EXPERT_IMPLEMENTATION
    """Execution strategy for the Qwen target expert banks.

    The CI expert implementation is authored independently on its CI config.

    `dense_masked` uses ordinary token/expert matmuls; routed implementations compute
    selected jobs. CI emission remains an independent interface choice. Other target
    families have no experts and accept only the default value.
    """
    attention_implementation: AttentionImplementation
    output_edge: LMOutputEdgeConfig
    """The model-output edge, authored on every LM config — a stored config that omitted
    it could not say which logits its run compared. The `streamed` edge avoids the
    full-vocab logits buffer and accumulates each chunk's logits in fp32."""
    weights_dtype: WeightsDtype
    """dtype for the FROZEN target weights. Only the frozen target is cast; trained V/U
    components keep their fp32 AdamW master.

    `bfloat16` halves the target's resident footprint on every pool — the dominant resident
    term for an 8B target — and for a natively-bf16 checkpoint costs nothing beyond
    residual/norm accumulation precision.

    `float32` is a genuine option but not a free one: the masked forward promotes where
    fp32 frozen weights meet the bf16 compute V/U, so the whole recon forward runs fp32.
    That is ~2x the activation memory. Float32 requires explicitly selecting `xla`
    attention, which materializes the [B, H, T, T] scores.

    Deliberately has no default: it is the largest single memory decision in the config,
    and a stored config that omitted it could not say how its run was trained."""

    @model_validator(mode="after")
    def validate_attention_dtype(self) -> Self:
        match self.attention_implementation:
            case "flash":
                if self.weights_dtype == "float32":
                    raise ValueError(
                        "flash attention requires bfloat16 target weights; choose "
                        "weights_dtype: bfloat16 or explicitly select attention_implementation: xla"
                    )
            case "xla":
                pass
        return self


class AllLayers(BaseConfig):
    kind: Literal["all"] = "all"


class LayerRange(BaseConfig):
    """Half-open `[start, end)` — matches `range()` / slice semantics."""

    kind: Literal["range"] = "range"
    start: NonNegativeInt
    end: PositiveInt

    @model_validator(mode="after")
    def _nonempty(self) -> Self:
        assert self.start < self.end, (self.start, self.end)
        return self


class LayerList(BaseConfig):
    kind: Literal["list"] = "list"
    indices: list[NonNegativeInt] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _unique_sorted(self) -> Self:
        assert self.indices == sorted(set(self.indices)), self.indices
        return self


LayerSelection = Annotated[AllLayers | LayerRange | LayerList, Field(discriminator="kind")]


class GluTransformerCSpec(BaseConfig):
    """Per-matrix-type C tiled across the selected layers (GLU family, e.g. Llama-3.1). Every
    selected layer is decomposed at the same `cs` matrices and C; a matrix absent from `cs`
    is not decomposed on any layer. Tiled ⇒ every block is structurally identical, so the
    chunkwise CI fn's chunks are homogeneous by construction."""

    kind: Literal["glu_transformer"] = "glu_transformer"
    layers: LayerSelection
    cs: dict[GluMatrix, PositiveInt] = Field(..., min_length=1)
    initialization: Literal["random", "nonlinearity_aligned"] = "random"


class SimpleMlpCSpec(BaseConfig):
    """Per-matrix-type C tiled across the selected layers (plain-GELU family, LlamaSimpleMLP)."""

    kind: Literal["simple_mlp"] = "simple_mlp"
    layers: LayerSelection
    cs: dict[SimpleMlpMatrix, PositiveInt] = Field(..., min_length=1)
    initialization: Literal["random", "nonlinearity_aligned"] = "random"


class Qwen36MoeCSpec(BaseConfig):
    """Per-matrix-type C for the qwen36_moe family, each kind on every layer that HAS it —
    coverage derived from the architecture (`qwen36_moe.layers_of_kind`: the DeltaNet kinds
    on the non-full-attention layers, the attention kinds on the full-attention layers, the
    MoE kinds everywhere), never selected, so `layers` must be `all`. An `experts_*` C must
    be a multiple of `n_experts` (components are expert-local), which `site_factorization`
    asserts at resolution. `nonlinearity_aligned` puts each component on one matrix coordinate —
    a chunk of the matrix attached to one nonlinearity: a SwiGLU hidden unit (per routed
    expert or the shared expert), a DeltaNet channel; attention components sit inside one
    head."""

    kind: Literal["qwen36_moe"] = "qwen36_moe"
    layers: LayerSelection
    cs: dict[Qwen36MoeMatrix, PositiveInt] = Field(..., min_length=1)
    initialization: Literal["random", "nonlinearity_aligned"] = "random"


@dataclass(frozen=True)
class BlockSites:
    """One transformer block's decomposed matrices, in canonical within-block (family) order.
    `layer_idx` is a field — the whole point is that structure is never thrown into a string."""

    layer_idx: int
    slots: tuple[tuple[str, int], ...]  # (matrix_type, C)


@dataclass(frozen=True)
class SiteTree:
    """A decomposition as blocks, strictly layer-ascending — the layer index is carried as
    DATA, never parsed back out of a site name. The tiled site specs resolve INTO it, and
    the chunkwise CI resolver consumes it directly (a chunk = a slice of consecutive
    `BlockSites`), so nothing downstream recovers block structure by regex-ing site-name
    strings. The flat `SiteC` view is DERIVED via the family's name grammar —
    construction only, no inverse parse."""

    blocks: tuple[BlockSites, ...]

    def site_cs(self, name_of: Callable[[int, str], str]) -> tuple[SiteC, ...]:
        return tuple(
            SiteC(name_of(b.layer_idx, kind), c) for b in self.blocks for kind, c in b.slots
        )


def _select_layers(sel: LayerSelection, n_layer: int) -> tuple[int, ...]:
    match sel:
        case AllLayers():
            return tuple(range(n_layer))
        case LayerRange(start=start, end=end):
            assert end <= n_layer, f"layer range end {end} exceeds n_layer {n_layer}"
            return tuple(range(start, end))
        case LayerList(indices=indices):
            assert indices[-1] < n_layer, f"layer {indices[-1]} exceeds n_layer {n_layer}"
            return tuple(indices)


def _canonical_slots(
    cs: Iterable[tuple[str, int]], family: ArchFamily
) -> tuple[tuple[str, int], ...]:
    # cs keys are Literal-typed by the family vocabulary `matrices` derives from, so every
    # key is a family matrix by construction — ordering is the only work left.
    rank = {matrix: i for i, matrix in enumerate(family.matrices)}
    return tuple(sorted(cs, key=lambda slot: rank[slot[0]]))


def resolve_site_tree(
    sites: "GluTransformerCSpec | SimpleMlpCSpec", family: ArchFamily, n_layer: int
) -> SiteTree:
    """Tile the per-matrix-type `cs` across the selected layers into a `SiteTree`. Every block
    shares ONE `slots` tuple (canonical family order, only the requested matrices), so the tree
    is homogeneous by construction — which is exactly what makes the chunkwise CI fn's chunks
    homogeneous. Asserts the spec's declared family matches the target's."""
    assert sites.kind == family.key, f"c-spec family {sites.kind!r} != target family {family.key!r}"
    slots = _canonical_slots(sites.cs.items(), family)
    layers = _select_layers(sites.layers, n_layer)
    return SiteTree(tuple(BlockSites(layer, slots) for layer in layers))


def _resolve_qwen36_site_tree(sites: Qwen36MoeCSpec, arch: qwen36_moe.Qwen36MoeConfig) -> SiteTree:
    """Every layer's block carries the authored kinds that layer HAS (`layers_of_kind`), in
    canonical family order — so blocks differ by the mixer their layer seats, and a block
    is empty when no authored kind sits on its layer (one mixer's kinds alone leave the
    other mixer's layers empty). A chunkwise CI fn over this tree is homogeneous only when
    its chunk length keeps every chunk's blocks structurally identical (`_resolved_chunks`
    asserts it)."""
    slots = _canonical_slots(sites.cs.items(), qwen36_moe.FAMILY)
    covered = {kind: frozenset(qwen36_moe.layers_of_kind(arch, kind)) for kind, _ in slots}
    return SiteTree(
        tuple(
            BlockSites(layer, tuple(slot for slot in slots if layer in covered[slot[0]]))
            for layer in range(arch.n_layer)
        )
    )


class ResidualBoundaryTapSelection(BaseConfig):
    """Residual boundaries at `offsets` target layers past the selected group's first
    block. Offsets count undecomposed layers between selected blocks too, and reach
    through the last block's output."""

    kind: Literal["residual_boundaries"] = "residual_boundaries"
    offsets: tuple[NonNegativeInt, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_increasing(self) -> Self:
        assert self.offsets == tuple(sorted(set(self.offsets))), (
            f"residual boundary offsets must be unique and increasing: {self.offsets}"
        )
        return self


CIFnInputTapSelection = (
    Literal["first_block_resid", "all_block_resids", "all_block_taps"]
    | ResidualBoundaryTapSelection
)
"""Which activations a CI network reads from its group of blocks (one chunk, or every
decomposed block). Extend here + add a match arm in `_resolve_input_taps` below; the
concrete tap keys and their widths are the family tap grammar's
(`param_decomp.targets.transformer_taps`) — opaque strings everywhere generic."""


class MHACIFnAttentionConfig(BaseConfig):
    """Every query head carries its own K/V head."""

    kind: Literal["mha"] = "mha"
    n_heads: PositiveInt
    implementation: AttentionImplementation
    mask: CIFnAttentionMask


class GQACIFnAttentionConfig(BaseConfig):
    """Grouped-query attention: `n_heads // n_kv_heads` query heads share each K/V head, so
    `wk`/`wv` narrow to `n_kv_heads * head_dim`. head_dim, the RoPE tables, `wq`/`wo` and
    every sharding are identical to MHA — only the K/V projections change."""

    kind: Literal["gqa"] = "gqa"
    n_heads: PositiveInt
    n_kv_heads: PositiveInt
    implementation: AttentionImplementation
    mask: CIFnAttentionMask

    @model_validator(mode="after")
    def validate_grouping(self) -> Self:
        assert self.n_heads % self.n_kv_heads == 0, (
            "n_heads must be divisible by n_kv_heads (each K/V head serves an equal group "
            f"of query heads): {self.n_heads} % {self.n_kv_heads}"
        )
        assert self.n_kv_heads < self.n_heads, (
            f"n_kv_heads == n_heads ({self.n_heads}) is MHA — use `kind: mha` rather than "
            "spelling it as a degenerate gqa"
        )
        return self


class GeluCIFnFfnConfig(BaseConfig):
    """`Linear+b -> GELU -> Linear+b` — two matrices."""

    kind: Literal["gelu"] = "gelu"
    hidden: PositiveInt


class SwigluCIFnFfnConfig(BaseConfig):
    """`silu(h@w_gate + b_gate) * (h@w1 + b1) -> Linear+b` — THREE matrices, so at a given
    `hidden` this is ~1.5x the GELU FFN's params. Iso-param is `hidden` at 2/3 (Shazeer's GLU
    variants: "decrease d_ff by a factor of 2/3"); nothing here rescales it, because a width
    that silently differs from the one you wrote is worse than doing the arithmetic."""

    kind: Literal["swiglu"] = "swiglu"
    hidden: PositiveInt


CIFnFfnConfig = Annotated[GeluCIFnFfnConfig | SwigluCIFnFfnConfig, Field(discriminator="kind")]
"""The CI transformer's feed-forward sublayer. Named FFN, not MLP: `swiglu` is a gated
linear unit, not a multi-layer perceptron — FFN is the name that stays honest across both
arms. `hidden` belongs to the FFN, not to the transformer around it."""


CIFnAttentionConfig = Annotated[
    MHACIFnAttentionConfig | GQACIFnAttentionConfig, Field(discriminator="kind")
]
"""The CI transformer's attention, keyed by CLASS rather than an optional `n_kv_heads` —
so a K/V head count cannot exist without meaning, and the grouping invariant lives on the
arm that has both fields instead of being a runtime check on a shape that shouldn't parse.
Mirrors how the target's attention variants are keyed (see the Qwen3 family split)."""


class InputScaleCalibrationConfig(BaseConfig):
    """The whole step-0 training batch, which must hold no padding, at least `min_n_tokens`
    tokens, and a whole number of the calibration's fixed-size chunks
    (`CALIBRATION_CHUNK_TOKENS`)."""

    min_n_tokens: PositiveInt


class AffineClippedComponentActivationConfig(BaseConfig):
    """Condition each site's CI on its component activations `h = x @ V` of the clean site
    input: add `s₊ * clamp(a₊ * h + b₊) + s₋ * clamp(a₋ * (-h) + b₋) + bias` to its
    preactivation, per component, where the clamp holds [0, 1] and outside it leaks only
    gradients pointing back inside. At init every output scale is `output_scale_init`, the
    input biases and the bias are 0, and every input scale is `1 / q`: `q` is the
    component's exact 99.9th percentile of `|h|` over the `calibration` batch. Targeted
    training calibrates on its non-target stream. V's gradient joins the component update.
    Requires a dense transformer target (GLU or SimpleMLP), whose anatomy names each site's
    input."""

    kind: Literal["affine_clipped_component_activation"]
    output_scale_init: float = Field(allow_inf_nan=False)
    calibration: InputScaleCalibrationConfig


class TransformerCIFnConfig(BaseConfig):
    """The CI transformer backbone every transformer CI arch sizes the same way
    (`d_model % n_heads == 0`; head_dim even for RoPE)."""

    d_model: PositiveInt
    n_blocks: PositiveInt
    attention: CIFnAttentionConfig
    learned_norm_scale: bool = Field(
        default=False,
        description="Learned per-channel scale on the block RMSNorms (the per-tap input norms "
        "stay weightless). Inits to ones, so step 0 is identical to weightless.",
    )

    @model_validator(mode="after")
    def validate_head_dim(self) -> Self:
        n_heads = self.attention.n_heads
        assert self.d_model % n_heads == 0, (self.d_model, n_heads)
        assert (self.d_model // n_heads) % 2 == 0, "head_dim must be even for RoPE"
        return self


class ChunkwiseTransformerCIFnConfig(TransformerCIFnConfig):
    """Chunkwise-transformer CI fn (LMs). Each chunk is `blocks_per_chunk` consecutive
    transformer blocks; `input_tap` names which activations the chunk reads and its output
    is CI for every matrix site in those blocks. The backbone and `ffn` size the per-chunk
    CI transformer."""

    type: Literal["chunkwise_transformer"] = "chunkwise_transformer"
    blocks_per_chunk: PositiveInt
    input_tap: CIFnInputTapSelection = "first_block_resid"
    """`first_block_resid`: the residual stream entering the chunk's first block — one tap.
    `all_block_resids`: the residual entering EVERY block the chunk runs over, RMS-normed
    per tap and concatenated (`ci_fn.Chunk.input_taps` is generic over tap count) —
    `blocks_per_chunk`x the per-chunk CI transformer's input width.
    `all_block_taps`: every site-input vector the target computes in each block of the
    chunk (`transformer_taps.SiteInputTapName`; qwen36_moe has no dense MLP hidden).
    Each physical vector appears once.
    `residual_boundaries`: the residual boundaries at the given layer offsets from the
    chunk's first block."""
    ffn: CIFnFfnConfig


class GlobalTransformerCIFnConfig(TransformerCIFnConfig):
    """Global-transformer CI fn (LMs): ONE transformer over `input_tap`'s taps across ALL
    decomposed blocks, with an output head for every decomposed site."""

    type: Literal["global_transformer"] = "global_transformer"
    input_tap: CIFnInputTapSelection
    ffn: CIFnFfnConfig
    component_conditioning: AffineClippedComponentActivationConfig | None = None
    """`None` leaves the CI transformer unconditioned."""


class MoEChunkwiseTransformerCIFnConfig(TransformerCIFnConfig):
    """MoE chunkwise-transformer CI fn (the qwen36_moe family): each chunk covers
    `blocks_per_chunk` consecutive target layers (one stage) and runs `n_blocks` blocks
    of non-causal attention + a CONCAT-WIDE routed MoE FFN — one expert bank per covered
    layer, dispatched by that layer's CAPTURED routing — plus a dense swiglu shared
    expert. Expert-blocked sites emit NARROW `SelectedCI` bundles via per-expert heads
    fused into the last block's expert slots; shared sites emit full-width. The chunk
    input concatenates `input_tap`'s RMS-normed taps with one dense routing-weight
    vector per covered layer. `n_experts`/`experts_per_token` come from the target;
    `expert_ffn_hidden` sizes one CI expert (the target's `moe_intermediate` mirrors
    the stage's expert parameters exactly), `shared_ffn_hidden` the shared expert."""

    type: Literal["moe_chunkwise_transformer"] = "moe_chunkwise_transformer"
    blocks_per_chunk: PositiveInt
    input_tap: CIFnInputTapSelection = "first_block_resid"
    expert_ffn_hidden: PositiveInt
    shared_ffn_hidden: PositiveInt
    expert_implementation: ExpertImplementation


class GlobalMlpCIFnConfig(BaseConfig):
    """Global-MLP CI fn (the tPD paper's LM CI net, arXiv 2607.13047): ONE shared MLP over
    the concatenation of `input_tap`'s taps across ALL decomposed blocks, applied pointwise
    per token, split back per site. Conceptually one chunk spanning every block — the same
    tap vocabulary, no attention, so a position's CI reads only that position."""

    type: Literal["global_mlp"] = "global_mlp"
    hidden_dims: tuple[PositiveInt, ...] = Field(..., min_length=1)
    input_tap: CIFnInputTapSelection


LMCIFnConfig = Annotated[
    ChunkwiseTransformerCIFnConfig
    | GlobalTransformerCIFnConfig
    | MoEChunkwiseTransformerCIFnConfig
    | GlobalMlpCIFnConfig,
    Field(discriminator="type"),
]
"""The CI-fn arches an LM run can author, all positioned: the chunkwise transformer
(cross-position CI within a chunk), the global transformer (cross-position CI over every
decomposed block), the chunkwise MoE sibling (routed banks + narrow emission), and the
global MLP (pointwise per token)."""


class LMDecompositionConfig(BaseConfig):
    """The LM decomposition apparatus: a per-matrix-type site-spec + the CI-fn arch. The
    GLU / simple-MLP specs tile one slot tuple over a layer selection, so their blocks are
    structurally identical and chunkwise chunks homogeneous by construction; the qwen36_moe
    spec derives each kind's layers from the architecture, and the chunkwise resolver
    asserts its chunks homogeneous. No `explicit` per-site variant exists here, so a
    heterogeneous decomposition is unrepresentable rather than refused late. The
    `sites.kind` family is checked against the target family at resolve."""

    sites: Annotated[GluTransformerCSpec | SimpleMlpCSpec | Qwen36MoeCSpec, Discriminator("kind")]
    ci: LMCIFnConfig


def _assert_eval_reads_this_run(
    eval: EvalConfig | None, loss_metrics: Sequence[AnyLossMetricConfig | TargetedLossMetricConfig]
) -> None:
    """The eval metrics that read the run's TRAINING state resolve against it here, where
    the two sections first coexist."""
    if eval is None:
        return
    for metric in eval.metrics:
        if isinstance(metric, RouterDivergenceConfig):
            assert_router_divergence_persistent_terms_exist(metric, loss_metrics)


class LMExperimentConfig(ExperimentConfig):
    runtime: RuntimeConfig
    """The LM's compute substrate. Declared HERE, not on the shared base: an LM run is the
    only domain that spans devices and nodes, so it is the only one with a
    world size, a placement policy, remat trades and an XLA-flag surface to author."""

    eval: EvalConfig | None = None
    resume_provenance: ResumeProvenance | None = None
    target: LMTargetConfig
    decomposition: LMDecompositionConfig
    data: LMDataConfig

    @model_validator(mode="after")
    def validate_eval_reads_this_run(self) -> Self:
        _assert_eval_reads_this_run(self.eval, self.pd.loss_metrics)
        return self


class LMTargetedExperimentConfig(ExperimentConfigBase):
    """The targeted (tPD) LM run shape — its own top-level schema, not a mode
    flag on the plain one: choosing the root (`experiments.lm.run_targeted` vs
    `experiments.lm.run`) chooses the algorithm, and each shape refuses the other's
    sections at parse. `pd.loss_metrics` authors the TARGET pass at `pd.batch_size`
    over the `prompts:` pool; `data:` is the broad NON-TARGET stream at
    `nontarget.batch_size`. No `resume_provenance`: fine-tune semantics for a
    targeted run (whose parent may be plain OR targeted) are undefined, so the field is
    unrepresentable rather than accepted and wrong.

    `eval:` stays available, unlike the toy targeted shape: the LM operations are
    forward-only diagnostics on the broad eval split, and an unobservable 8B run is worse
    than one whose probes need reading with tPD in mind."""

    pd: TargetedPDConfig
    runtime: RuntimeConfig
    eval: EvalConfig | None = None
    target: LMTargetConfig
    decomposition: LMDecompositionConfig
    data: LMDataConfig
    prompts: LMPromptPoolConfig
    nontarget: NontargetConfig

    @model_validator(mode="after")
    def validate_eval_reads_this_run(self) -> Self:
        _assert_eval_reads_this_run(self.eval, self.pd.loss_metrics)
        return self


@dataclass(frozen=True)
class HFModelVariant[ArchT: HFTransformerArch]:
    """An HF architecture paired with the loader that accepts that exact config type.
    Site shapes and loaded weights derive from the same immutable architecture."""

    arch: ArchT
    load_weights: Callable[
        [str, ArchT, tuple[SiteSpec, ...], DTypeLike, OutputEdge, AttentionImplementation],
        TransformerDecomposedModel,
    ]
    model_type: str
    model_class: str
    """The `target.spec.model_class` identifier; never imported."""

    def load(
        self,
        model_name: str,
        sites: tuple[SiteC, ...],
        weights_dtype: DTypeLike,
        output_edge: OutputEdge,
        attention_implementation: AttentionImplementation,
    ) -> TransformerDecomposedModel:
        return self.load_weights(
            model_name,
            self.arch,
            transformer.glu_site_specs(self.arch, sites),
            weights_dtype,
            output_edge,
            attention_implementation,
        )


type SupportedHFModelVariant = HFModelVariant[LlamaConfig] | HFModelVariant[TransformerConfig]


def _qwen3_variant(arch: TransformerConfig) -> HFModelVariant[TransformerConfig]:
    return HFModelVariant(
        arch=arch,
        load_weights=qwen3.load_decomposed_qwen3_from_hf,
        model_type="Qwen3",
        model_class="transformers.Qwen3ForCausalLM",
    )


HF_MODEL_VARIANTS: dict[str, SupportedHFModelVariant] = {
    "meta-llama/Llama-3.1-8B": HFModelVariant(
        arch=llama31.llama31_8b_config(),
        load_weights=llama31.load_decomposed_llama31_from_hf,
        model_type="Llama",
        model_class="transformers.LlamaForCausalLM",
    ),
    "Qwen/Qwen3-0.6B-Base": _qwen3_variant(qwen3.qwen3_0_6b_base_config()),
    "Qwen/Qwen3-0.6B": _qwen3_variant(qwen3.qwen3_0_6b_config()),
    "Qwen/Qwen3-1.7B-Base": _qwen3_variant(qwen3.qwen3_1_7b_base_config()),
    "Qwen/Qwen3-1.7B": _qwen3_variant(qwen3.qwen3_1_7b_config()),
    "Qwen/Qwen3-4B-Base": _qwen3_variant(qwen3.qwen3_4b_base_config()),
    "Qwen/Qwen3-4B": _qwen3_variant(qwen3.qwen3_4b_config()),
    "Qwen/Qwen3-8B-Base": _qwen3_variant(qwen3.qwen3_8b_base_config()),
    "Qwen/Qwen3-8B": _qwen3_variant(qwen3.qwen3_8b_config()),
    "Qwen/Qwen3-14B-Base": _qwen3_variant(qwen3.qwen3_14b_base_config()),
    "Qwen/Qwen3-14B": _qwen3_variant(qwen3.qwen3_14b_config()),
}
"""The HF model names the LM composition implements. Anything else refuses loudly at
convert time — a new model gets an explicit variant entry (config checked against its HF
config.json), never a silent guess."""


QWEN36_MOE_MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
QWEN36_MOE_MODEL_CLASS = "transformers.Qwen3_5MoeForCausalLM"
"""The one MoE checkpoint the composition implements. It is not an `HFModelVariant`:
that registry is typed over the GLU-transformer machinery, while this target has its own
engine (`param_decomp.targets.qwen36_moe`) — a second MoE checkpoint would generalize
this pair into a registry of its own."""


def _assert_qwen36_ci_fn_chunks_per_stage(ci: "LMCIFnConfig", arch: Qwen36MoeConfig) -> None:
    if isinstance(ci, MoEChunkwiseTransformerCIFnConfig):
        assert ci.blocks_per_chunk == arch.full_attention_interval, (
            f"the MoE chunkwise arch chunks per STAGE ({arch.full_attention_interval} "
            f"layers — the concat-wide banks mirror one stage's routers); got "
            f"blocks_per_chunk={ci.blocks_per_chunk}"
        )


def hf_model_variant(model_name: str) -> SupportedHFModelVariant:
    assert model_name in HF_MODEL_VARIANTS, (
        f"no vendored model variant for {model_name!r}; supported: {sorted(HF_MODEL_VARIANTS)}"
    )
    return HF_MODEL_VARIANTS[model_name]


@dataclass(frozen=True)
class ResolvedDecomposition:
    """Target config + its block-structured `SiteTree` + arch family, resolved once and shared
    by the target's flat `.sites`, the chunkwise chunk generator, and validation. `grammar`
    is the target module's own capture grammar for this shape — what CI tap keys and
    widths resolve against. `site_specs` are the
    shape-carrying specs (built by the same per-family builder the composition root's target
    load uses), the placement gate's input."""

    target: AnyLMTargetConfig
    tree: SiteTree
    grammar: TransformerTapGrammar
    site_specs: tuple[SiteSpec, ...]


def _resolve_output_edge(edge: LMOutputEdgeConfig, vocab_size: int) -> OutputEdge:
    match edge:
        case MaterializedOutputEdgeConfig():
            return MaterializedOutputEdge()
        case StreamedOutputEdgeConfig(n_vocab_chunks=n_vocab_chunks):
            assert vocab_size % n_vocab_chunks == 0, (
                f"target.output_edge.n_vocab_chunks={n_vocab_chunks} must divide the "
                f"vocab size {vocab_size}"
            )
            return StreamedOutputEdge(n_vocab_chunks=n_vocab_chunks)


def _resolve_qwen36_moe(
    target_config: LMTargetConfig,
    sites: Qwen36MoeCSpec,
    ci: "LMCIFnConfig",
    arch: Qwen36MoeConfig,
    weights: Qwen36MoeWeights,
) -> ResolvedDecomposition:
    """The qwen36_moe family's resolution, shared by its weight sources: whole-grid
    coverage per decomposed kind, per-stage MoE CI chunks, the output edge against the
    arch's vocab."""
    assert isinstance(sites.layers, AllLayers), (
        "qwen36_moe decomposes each kind on every layer that has it — coverage the "
        "architecture derives (`layers_of_kind`), never a selection: author "
        "`layers: {kind: all}`"
    )
    _assert_qwen36_ci_fn_chunks_per_stage(ci, arch)
    tree = _resolve_qwen36_site_tree(sites, arch)
    target = Qwen36MoeTargetConfig(
        arch=arch,
        weights=weights,
        sites=tree.site_cs(qwen36_moe.FAMILY.name_of),
        weights_dtype=target_config.weights_dtype,
        attention_implementation=target_config.attention_implementation,
        output_edge=_resolve_output_edge(target_config.output_edge, arch.vocab_size),
        expert_implementation=target_config.expert_implementation,
        component_initialization=sites.initialization,
    )
    grammar = qwen36_moe.capture_grammar(arch)
    site_specs = qwen36_moe.qwen36_moe_site_specs(arch, target.sites)
    if sites.initialization == "nonlinearity_aligned":
        for spec in site_specs:
            qwen36_moe.validate_nonlinearity_aligned_capacity(spec)
    return ResolvedDecomposition(target, tree, grammar, site_specs)


def _assert_default_expert_implementation(target_config: LMTargetConfig) -> None:
    assert target_config.expert_implementation == DEFAULT_EXPERT_IMPLEMENTATION, (
        "target.expert_implementation serves the qwen36_moe family only; this target has "
        "no expert banks"
    )


def resolve_decomposition(
    target_config: LMTargetConfig, decomposition: LMDecompositionConfig, data_root: Path
) -> ResolvedDecomposition:
    """Target spec + tiled `decomposition.sites` -> target config + `SiteTree`.

    The schema admits every (target spec, c-spec) product; each target decomposes exactly
    one c-spec family, so the pair resolves through one match that enumerates the legal
    combinations and refuses the rest by name. HF specs resolve their variant from
    `HF_MODEL_VARIANTS` (all GLU-transformer targets), or — with a `qwen36_moe` c-spec — the
    one qwen36_moe HF checkpoint; the `pretrained*` kinds read their arch from the
    pretrain-cache entry (`pretrained` LlamaSimpleMLP, plain-MLP family;
    `pretrained_qwen35_moe`, the qwen36_moe family). The tree is tiled from the
    per-matrix-type `cs` over the selected layers."""
    spec, sites = target_config.spec, decomposition.sites
    match spec, sites:
        case HFTarget() as spec, Qwen36MoeCSpec() as sites:
            assert spec.model_name == QWEN36_MOE_MODEL_NAME, (
                f"the qwen36_moe HF target serves only {QWEN36_MOE_MODEL_NAME!r}, got "
                f"{spec.model_name!r}"
            )
            assert spec.model_class == QWEN36_MOE_MODEL_CLASS, spec.model_class
            return _resolve_qwen36_moe(
                target_config,
                sites,
                decomposition.ci,
                arch=qwen36_moe.qwen36_35b_a3b_config(),
                weights=HFSnapshotWeights(spec.model_name),
            )
        case PretrainedQwen35MoeTarget() as spec, Qwen36MoeCSpec() as sites:
            cache_dir = pretrain_cache.resolved_model_config_dir(data_root, spec.run_path)
            return _resolve_qwen36_moe(
                target_config,
                sites,
                decomposition.ci,
                arch=qwen36_moe.qwen36_moe_config_from_pretrain_cache(cache_dir),
                weights=PretrainCacheWeights(spec.run_path),
            )
        case (
            (HFWeightsInVendored() | HFTarget()),
            GluTransformerCSpec(),
        ) | (PretrainedTarget(), SimpleMlpCSpec()):
            return resolve_dense_decomposition(target_config, sites, data_root)
        case _:
            raise ValueError(f"{type(spec).__name__} can't decompose {type(sites).__name__} sites")


type DenseCSpec = Annotated[GluTransformerCSpec | SimpleMlpCSpec, Discriminator("kind")]
"""The site specs of the dense families: the ones that resolve without a CI config."""


def resolve_dense_decomposition(
    target_config: LMTargetConfig, sites: DenseCSpec, data_root: Path
) -> ResolvedDecomposition:
    """`resolve_decomposition` for the dense families, whose resolution reads the sites
    alone — never the CI config (the qwen36_moe arms need it, so they stay there)."""
    spec = target_config.spec
    match spec, sites:
        case (HFWeightsInVendored() | HFTarget()) as spec, GluTransformerCSpec() as sites:
            _assert_default_expert_implementation(target_config)
            match spec:
                case HFWeightsInVendored():
                    assert spec.model_class.rsplit(".", 1)[-1] == "VendoredLlama", spec.model_class
                    assert "Llama-3.1-8B" in spec.model_name, spec.model_name
                case HFTarget():
                    known_classes = {variant.model_class for variant in HF_MODEL_VARIANTS.values()}
                    assert spec.model_class in known_classes, spec.model_class
                    assert spec.model_class == hf_model_variant(spec.model_name).model_class, (
                        f"{spec.model_class!r} is not {spec.model_name!r}'s registered variant"
                    )
            hf_variant = hf_model_variant(spec.model_name)  # refuses unknown model names
            arch = hf_variant.arch
            tree = resolve_site_tree(sites, transformer.FAMILY, arch.n_layer)
            target = TargetConfig(
                model_name=spec.model_name,
                sites=tree.site_cs(transformer.FAMILY.name_of),
                weights_dtype=target_config.weights_dtype,
                attention_implementation=target_config.attention_implementation,
                output_edge=_resolve_output_edge(target_config.output_edge, arch.vocab_size),
                component_initialization=sites.initialization,
            )
            grammar = transformer.capture_grammar(
                transformer.GLU_ANATOMY,
                arch.n_layer,
                arch.n_embd,
                lambda kind: transformer.site_dims(arch, kind),
            )
            site_specs = transformer.glu_site_specs(arch, target.sites)
            if sites.initialization == "nonlinearity_aligned":
                for site_spec in site_specs:
                    transformer.validate_nonlinearity_aligned_capacity(site_spec)
            return ResolvedDecomposition(target, tree, grammar, site_specs)
        case PretrainedTarget() as spec, SimpleMlpCSpec() as sites:
            _assert_default_expert_implementation(target_config)
            assert spec.model_class.rsplit(".", 1)[-1] == "LlamaSimpleMLP", spec.model_class
            cache_dir = pretrain_cache.resolved_model_config_dir(data_root, spec.run_path)
            arch = llama_simple_mlp.load_model_config(cache_dir)
            tree = resolve_site_tree(sites, llama_simple_mlp.FAMILY, arch.n_layer)
            target = LlamaSimpleMLPTargetConfig(
                pretrain_run_path=spec.run_path,
                sites=tree.site_cs(llama_simple_mlp.FAMILY.name_of),
                weights_dtype=target_config.weights_dtype,
                attention_implementation=target_config.attention_implementation,
                output_edge=_resolve_output_edge(target_config.output_edge, arch.vocab_size),
                component_initialization=sites.initialization,
            )
            grammar = transformer.capture_grammar(
                llama_simple_mlp.SIMPLE_MLP_ANATOMY,
                arch.n_layer,
                arch.n_embd,
                lambda kind: llama_simple_mlp.site_dims(arch, kind),
            )
            site_specs = llama_simple_mlp.site_specs(arch, target.sites)
            if sites.initialization == "nonlinearity_aligned":
                for site_spec in site_specs:
                    transformer.validate_nonlinearity_aligned_capacity(site_spec)
            return ResolvedDecomposition(target, tree, grammar, site_specs)
        case _:
            raise ValueError(f"{type(spec).__name__} can't decompose {type(sites).__name__} sites")


def _resolve_input_taps(
    input_tap: CIFnInputTapSelection, blocks: tuple[BlockSites, ...], grammar: TransformerTapGrammar
) -> tuple[str, ...]:
    """The tap keys a CI network reads, from its config source + the blocks it spans."""
    match input_tap:
        case "first_block_resid":
            return (resid_tap_key(blocks[0].layer_idx),)
        case "all_block_resids":
            return tuple(resid_tap_key(b.layer_idx) for b in blocks)
        case "all_block_taps":
            return grammar.site_input_tap_keys(tuple(block.layer_idx for block in blocks))
        case ResidualBoundaryTapSelection(offsets=offsets):
            start = blocks[0].layer_idx
            span = blocks[-1].layer_idx + 1 - start
            assert offsets[-1] <= span, (
                f"residual boundary offsets {offsets} exceed the selected boundary range "
                f"0..{span} (target layers {start}..{blocks[-1].layer_idx})"
            )
            return tuple(resid_tap_key(start + offset) for offset in offsets)


def _resolved_chunks(
    tree: SiteTree,
    blocks_per_chunk: int,
    input_tap: CIFnInputTapSelection,
    grammar: TransformerTapGrammar,
) -> tuple[Chunk, ...]:
    """Partition the site tree's blocks into consecutive `blocks_per_chunk`-block chunks. The
    tree IS the block grouping (layer-ascending, already grouped), so there is no name parsing
    and no groupby: a chunk reads the taps `input_tap` selects and emits CI for every slot in
    its blocks, in tree order."""
    family = grammar.family
    blocks = tree.blocks
    assert len(blocks) % blocks_per_chunk == 0, (
        f"{len(blocks)} decomposed blocks not divisible by blocks_per_chunk={blocks_per_chunk}"
    )
    groups = tuple(
        blocks[start : start + blocks_per_chunk]
        for start in range(0, len(blocks), blocks_per_chunk)
    )
    # The CI transformer stacks its per-slot heads chunk-by-chunk, so every chunk must
    # carry the same (kind, C) slots at the same in-chunk block positions. A tiled tree
    # guarantees it; a tree whose blocks differ by layer (qwen36_moe's mixer kinds) does
    # only when the chunk length matches the pattern's period.
    first_signature = tuple(block.slots for block in groups[0])
    for index, group in enumerate(groups):
        signature = tuple(block.slots for block in group)
        assert signature == first_signature, (
            f"chunk {index} (layers {[block.layer_idx for block in group]}) carries slots "
            f"{signature} but chunk 0 (layers {[block.layer_idx for block in groups[0]]}) "
            f"carries {first_signature}: chunks must be structurally identical — choose a "
            f"blocks_per_chunk that spans the tree's block pattern"
        )
    return tuple(
        Chunk(
            input_taps=_resolve_input_taps(input_tap, group, grammar),
            output_sites=tuple(
                family.name_of(block.layer_idx, kind) for block in group for kind, _ in block.slots
            ),
        )
        for group in groups
    )


def _resolve_chunkwise_ci_fn_arch(
    tree: SiteTree,
    ci: ChunkwiseTransformerCIFnConfig,
    grammar: TransformerTapGrammar,
) -> ChunkwiseTransformerCIFnArch:
    """Resolve the chunkwise-transformer arch from the site tree: the chunk generator
    (`_resolved_chunks`, which asserts the chunks structurally identical) + the per-chunk
    input width (the sum of the chunk's tap widths — `grammar.width_of`), asserted equal
    across chunks because a block tap's width may differ by block. The `attention` union
    collapses here to two concrete head counts — MHA is `n_kv_heads == n_heads` — so
    nothing downstream re-derives the grouping, and the fine-tune
    `parent.ci_fn == built.ci_fn` compare sees concrete values on both sides."""
    chunks = _resolved_chunks(tree, ci.blocks_per_chunk, ci.input_tap, grammar)
    input_dims = {sum(grammar.width_of(key) for key in chunk.input_taps) for chunk in chunks}
    assert len(input_dims) == 1, (
        f"chunks read taps of different total widths {sorted(input_dims)}; the CI transformer's "
        "in_proj is one width across chunks"
    )
    (input_dim,) = input_dims
    return ChunkwiseTransformerCIFnArch(
        chunks=chunks,
        input_dim=input_dim,
        d_model=ci.d_model,
        n_blocks=ci.n_blocks,
        attention=_resolve_ci_fn_attention(ci.attention),
        ffn_hidden=ci.ffn.hidden,
        ffn_kind=ci.ffn.kind,
        learned_norm_scale=ci.learned_norm_scale,
    )


def _resolve_global_mlp_ci_fn_arch(
    tree: SiteTree,
    ci: GlobalMlpCIFnConfig,
    grammar: TransformerTapGrammar,
) -> GlobalMLPCIFnArch:
    """Build one pointwise MLP CI predictor spanning all decomposed blocks.

    Input taps are deduplicated physical vectors with grammar-declared widths.
    `has_position_axis=True` preserves the LM's token axis; the MLP does not mix
    positions, so it imposes no attention-boundary requirement."""
    tap_keys = _resolve_input_taps(ci.input_tap, tree.blocks, grammar)
    return GlobalMLPCIFnArch(
        hidden_dims=ci.hidden_dims,
        has_position_axis=True,
        input_taps=tuple(TapSpec(key=key, width=grammar.width_of(key)) for key in tap_keys),
    )


def _resolve_global_transformer_ci_fn_arch(
    resolved: ResolvedDecomposition, ci: GlobalTransformerCIFnConfig
) -> GlobalTransformerCIFnArch | ConditionedCIFnArch[GlobalTransformerCIFnArch]:
    """Under component conditioning, every decomposed site reads its clean input at the
    capture the target's anatomy names for it."""
    grammar = resolved.grammar
    tap_keys = _resolve_input_taps(ci.input_tap, resolved.tree.blocks, grammar)
    arch = GlobalTransformerCIFnArch(
        input_taps=tuple(TapSpec(key=key, width=grammar.width_of(key)) for key in tap_keys),
        d_model=ci.d_model,
        n_blocks=ci.n_blocks,
        attention=_resolve_ci_fn_attention(ci.attention),
        ffn_hidden=ci.ffn.hidden,
        ffn_kind=ci.ffn.kind,
        learned_norm_scale=ci.learned_norm_scale,
    )
    match ci.component_conditioning:
        case None:
            return arch
        case AffineClippedComponentActivationConfig(
            output_scale_init=output_scale_init,
            calibration=calibration,
        ):
            match resolved.target:
                case TargetConfig() | LlamaSimpleMLPTargetConfig() as target:
                    anatomy = dense_transformer_anatomy(target)
                case Qwen36MoeTargetConfig():
                    raise ValueError("component conditioning requires a dense transformer target")
            return ConditionedCIFnArch(
                arch,
                tuple(
                    SiteInput(site.name, anatomy.site_input_key(site.name))
                    for site in resolved.site_specs
                ),
                output_scale_init=output_scale_init,
                calibration=InputScaleCalibration(min_n_tokens=calibration.min_n_tokens),
            )


def _resolve_ci_fn_attention(
    ci_fn_attention: CIFnAttentionConfig,
) -> MHACIFnAttention | GQACIFnAttention:
    match ci_fn_attention:
        case MHACIFnAttentionConfig():
            return MHACIFnAttention(
                n_heads=ci_fn_attention.n_heads,
                implementation=ci_fn_attention.implementation,
                mask=ci_fn_attention.mask,
            )
        case GQACIFnAttentionConfig():
            return GQACIFnAttention(
                n_heads=ci_fn_attention.n_heads,
                n_kv_heads=ci_fn_attention.n_kv_heads,
                implementation=ci_fn_attention.implementation,
                mask=ci_fn_attention.mask,
            )


def _resolve_moe_chunkwise_ci_fn_arch(
    resolved: ResolvedDecomposition,
    ci: MoEChunkwiseTransformerCIFnConfig,
) -> BlockSelectedChunkwiseTransformerCIFnArch:
    """Resolve the MoE chunkwise arch: each chunk covers `blocks_per_chunk` consecutive
    target layers, names every covered layer (bank r on layer r's pinned routing), and
    emits expert kinds as `SelectedSlot`s keyed by their in-chunk router index. `n_experts`
    and the grouped-matmul arm are the resolved target's: the CI experts mirror the
    target's MoE grid and run its kernel arm."""
    target = resolved.target
    assert isinstance(target, Qwen36MoeTargetConfig), (
        f"the MoE chunkwise CI arch serves the qwen36_moe family, got {type(target).__name__}"
    )
    grammar = resolved.grammar
    blocks = resolved.tree.blocks
    assert len(blocks) % ci.blocks_per_chunk == 0, (
        f"{len(blocks)} decomposed blocks not divisible by blocks_per_chunk={ci.blocks_per_chunk}"
    )
    chunks: list[BlockSelectedChunk] = []
    for start in range(0, len(blocks), ci.blocks_per_chunk):
        group = blocks[start : start + ci.blocks_per_chunk]
        slots: list[BlockSelectedChunkSlot] = []
        for router, block in enumerate(group):
            for kind, _c in block.slots:
                site = grammar.family.name_of(block.layer_idx, kind)
                if qwen36_moe.is_expert_kind(kind):
                    slots.append(SelectedSlot(site=site, selection=router))
                else:
                    slots.append(FullSlot(site=site))
        chunks.append(
            BlockSelectedChunk(
                input_taps=_resolve_input_taps(ci.input_tap, group, grammar),
                layers=tuple(block.layer_idx for block in group),
                slots=tuple(slots),
            )
        )
    first = chunks[0]
    input_dim = sum(grammar.width_of(key) for key in first.input_taps)
    return BlockSelectedChunkwiseTransformerCIFnArch(
        chunks=tuple(chunks),
        input_dim=input_dim,
        d_model=ci.d_model,
        n_blocks=ci.n_blocks,
        attention=_resolve_ci_fn_attention(ci.attention),
        table_size=target.arch.n_experts,
        selected_ffn_hidden=ci.expert_ffn_hidden,
        shared_ffn_hidden=ci.shared_ffn_hidden,
        learned_norm_scale=ci.learned_norm_scale,
        expert_implementation=ci.expert_implementation,
    )


def resolve_lm_ci_fn_arch(resolved: ResolvedDecomposition, ci: LMCIFnConfig) -> LMCIFnArch:
    """The one authored-CI → resolved-arch seam: every LM build route (train, targeted,
    deliverable restore) dispatches here, so a new schema arm cannot reach training
    without also reaching the consumers."""
    match ci:
        case ChunkwiseTransformerCIFnConfig():
            return _resolve_chunkwise_ci_fn_arch(resolved.tree, ci, resolved.grammar)
        case GlobalTransformerCIFnConfig():
            return _resolve_global_transformer_ci_fn_arch(resolved, ci)
        case MoEChunkwiseTransformerCIFnConfig():
            return _resolve_moe_chunkwise_ci_fn_arch(resolved, ci)
        case GlobalMlpCIFnConfig():
            return _resolve_global_mlp_ci_fn_arch(resolved.tree, ci, resolved.grammar)


def _assert_objective_builds(build: Callable[[], PDObjective | TargetedPDObjective]) -> None:
    """Trace the objective's construction so unsupported loss configs refuse at convert
    time rather than on the GPUs. Traced, not built: config resolution precedes
    `initialize_topology`, and a concrete objective's schedule magnitudes would
    initialize a backend, which forbids the multi-node `jax.distributed.initialize`.
    Nothing is returned; the engine reads `pd.loss_metrics` verbatim (yaml order is
    RNG-load-bearing)."""
    eqx.filter_eval_shape(build)


def _data(data: LMDataConfig, data_root: Path) -> ResolvedLMData:
    shard_dir = resolve_dataset_ref(data.train, data_root)
    eval_dir = resolve_dataset_ref(data.eval, data_root)
    assert eval_dir != shard_dir, (
        f"data.eval resolves to the training shard dir ({shard_dir}) — not a holdout"
    )
    return ResolvedLMData(dir=shard_dir, eval_dir=eval_dir)


def _assert_supported_weights_dtype(target: AnyLMTargetConfig) -> None:
    """Refuse a frozen-target dtype the loader cannot honour. Every build route passes
    through this check, so no accepted config can be silently loaded at another dtype."""
    assert target.weights_dtype in target.supported_weights_dtypes, (
        f"target {type(target).__name__} supports frozen-target weights_dtype "
        f"{sorted(target.supported_weights_dtypes)}, config asks for "
        f"{target.weights_dtype!r}. No silent downgrade: declare a "
        f"supported dtype in the yaml."
    )


def _assert_placement_claims(
    resolved: ResolvedDecomposition,
    runtime: RuntimeConfig,
    ci_fn: LMCIFnArch,
    pd: AnyPDConfig,
) -> None:
    """Build the run's placement rules on an abstract mesh of the declared shape and
    construct the CI fn abstractly on them, so every placement refusal the run would hit
    fires at config build, before execution."""
    rules = placement.from_config(
        runtime.sharding,
        abstract_mesh_for_shape(runtime.mesh),
        resolved.site_specs,
        sequence_sharding=runtime.sequence_sharding,
    )
    # The Muon staging claims fire before execution ONLY for an optimizer that will consume the
    # ns_compute rows — a non-muon run keeps any-stack-length placement
    # (the optimizers re-check the same tiling at the consumer boundary).
    match pd.components_optimizer:
        case MuonOptimizerConfig():
            placement.assert_stacked_muon_component_staging(rules)
        case AdamWOptimizerConfig():
            pass
    # Abstract construction runs the architecture's own placement resolution, exactly as
    # the run will.
    abstract_ci_fn = eqx.filter_eval_shape(
        lambda: ci_fn.initialize(resolved.site_specs, rules, jax.random.PRNGKey(0))
    )
    match pd.ci_fn_optimizer:
        case MuonOptimizerConfig():
            assert_ci_fn_muon_staging_tiles(abstract_ci_fn)
        case AdamWOptimizerConfig():
            pass


def _assert_batch_size(name_for_err: str, batch_size: int, runtime: RuntimeConfig) -> None:
    n_data = runtime.data_parallel_size
    assert batch_size >= n_data and batch_size % n_data == 0, (
        f"{name_for_err}={batch_size} must be a positive multiple of effective data-parallel size "
        f"{n_data} (mesh={runtime.mesh})"
    )


def assert_placement_claims(
    cfg: "LMExperimentConfig | LMTargetedExperimentConfig", data_root: Path
) -> None:
    """Standalone placement gate for callers that validate before starting a run and for
    the repository config parse gate; both build routes run it on every build."""
    resolved = resolve_decomposition(cfg.target, cfg.decomposition, data_root)
    _assert_placement_claims(
        resolved,
        cfg.runtime,
        resolve_lm_ci_fn_arch(resolved, cfg.decomposition.ci),
        cfg.pd,
    )


def build_experiment_config(cfg: LMExperimentConfig, run_id: str, data_root: Path) -> LMRun:
    resolved = resolve_decomposition(cfg.target, cfg.decomposition, data_root)
    target = resolved.target
    _assert_objective_builds(lambda: build_objective(cfg.pd.loss_metrics, resolved.site_specs))
    _assert_supported_weights_dtype(target)
    ci_fn = resolve_lm_ci_fn_arch(resolved, cfg.decomposition.ci)
    _assert_placement_claims(resolved, cfg.runtime, ci_fn, cfg.pd)
    _assert_batch_size("pd.batch_size", cfg.pd.batch_size, cfg.runtime)
    data = _data(cfg.data, data_root)

    return BuiltRun(
        pd=cfg.pd,
        cadence=cfg.cadence,
        run=run_instance(cfg, run_id, data_root, cfg.resume_provenance),
        target=target,
        data=data,
        ci_fn=ci_fn,
    )


def build_targeted_experiment_config(
    cfg: LMTargetedExperimentConfig, run_id: str, data_root: Path
) -> LMTargetedRun:
    """The targeted build route: identical resolution to `build_experiment_config`, with
    the objective validated as the two-pass tPD surface (faithfulness refused, the
    non-target list checked). The targeted sections (`prompts`, `nontarget`)
    ride the authored config into the composition root, like `runtime` — the engine
    bundle stays the shared `BuiltRun`."""
    resolved = resolve_decomposition(cfg.target, cfg.decomposition, data_root)
    target = resolved.target
    _assert_objective_builds(
        lambda: build_targeted_objective(cfg.pd.loss_metrics, cfg.nontarget, resolved.site_specs)
    )
    _assert_supported_weights_dtype(target)
    ci_fn = resolve_lm_ci_fn_arch(resolved, cfg.decomposition.ci)
    _assert_placement_claims(resolved, cfg.runtime, ci_fn, cfg.pd)
    _assert_batch_size("pd.batch_size", cfg.pd.batch_size, cfg.runtime)
    _assert_batch_size("nontarget.batch_size", cfg.nontarget.batch_size, cfg.runtime)
    data = _data(cfg.data, data_root)

    return BuiltRun(
        pd=cfg.pd,
        cadence=cfg.cadence,
        run=run_instance(cfg, run_id, data_root, None),
        target=target,
        data=data,
        ci_fn=ci_fn,
    )


def build_from_schema(
    schema_raw: dict[str, Any],
    run_id: str,
    data_root: Path,
) -> tuple[LMRun, LMExperimentConfig]:
    """Validate a single self-contained LM run config (the canonical `LMExperimentConfig`
    schema) and convert it to the engine's `BuiltRun` bundle. `run_id` is the minted run
    identity (the entry point's CLI arg, or the run-dir name when reloading a finished run).

    The authored config comes back alongside the bundle: `runtime` lives there, and the
    composition root threads it into the engine explicitly (the bundle is core's, and core
    reads no substrate).

    The LM composition entry (`run.py`) is LM-only. The toy domains (TMS, ResidMLP) build
    their `BuiltRun` in their own `run.py` via the public shared helpers
    (`run_instance`, `ci_fn_arch`)."""
    cfg = LMExperimentConfig.model_validate(schema_raw)
    return build_experiment_config(cfg, run_id, data_root), cfg


def load_config(
    config_path: Path, run_id: str, data_root: Path
) -> tuple[LMRun, LMExperimentConfig]:
    """Parse one pinned LM run YAML into its built run and authored config.

    The stored-config boundary converts the retired warmup/decay schedule shape in
    memory before canonical validation. The pin stays byte-immutable, while pre-knot
    runs remain loadable, resumable, and usable as fine-tune parents.
    """
    raw = yaml.safe_load(config_path.read_text())
    return build_from_schema(migrate_schedule_knots(raw), run_id, data_root)
