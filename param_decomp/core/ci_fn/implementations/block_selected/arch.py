"""Chunkwise transformers whose expert FFNs and heads follow target routing."""

from dataclasses import dataclass, replace
from typing import Literal, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int, PRNGKeyArray

from param_decomp.core.axes import Axes
from param_decomp.core.ci_fn.architecture import sequence_length_for_flops
from param_decomp.core.ci_fn.implementations.block_selected.placement import (
    CI_FN_BLOCKED_HEAD_AXES,
    CI_FN_SELECTED_FFN_IN_AXES,
    CI_FN_SELECTED_FFN_OUT_AXES,
    BlockSelectedMuonStaging,
    PlacedBlockSelectedCIFnRows,
    bind_block_selected_rows,
    preset_rows,
)
from param_decomp.core.ci_fn.implementations.block_selected.routing import (
    Selection,
    block_sharded_selection,
    token_selection,
)
from param_decomp.core.ci_fn.implementations.chunkwise.chunk_stack import (
    pad_chunk_stack,
    real_chunks,
    unpadded_census,
    validate_chunk_leaves,
    validated_census,
)
from param_decomp.core.ci_fn.implementations.chunkwise.placement import (
    STACKED_ATTN_KV_AXES,
    STACKED_ATTN_OUT_AXES,
    STACKED_ATTN_Q_AXES,
    STACKED_FFN_IN_AXES,
    STACKED_FFN_OUT_AXES,
    STACKED_INPUT_AXES,
    STACKED_OUTPUT_AXES,
    AttentionHeadCounts,
    ChunkedCIFnPlacementRequirements,
    CIFnMatrixBatchShape,
    CIFnTensorShape,
    CIFnVectorBatchShape,
    chunk_persist_rows,
    chunk_placement_for_log,
    resolve_chunk_census,
    vector_sharding,
)
from param_decomp.core.ci_fn.implementations.transformer.layers import (
    CI_FN_FFN_IN_AXES,
    CI_FN_FFN_OUT_AXES,
    CI_FN_INPUT_AXES,
    CI_FN_OUTPUT_AXES,
    CI_FN_RMS_EPS,
    CI_FN_TRANSFORMER_STAGING_IN_PLACE,
    LOCAL_CI_FN_TRANSFORMER_PLACEMENT,
    CIFnAttention,
    CIFnTransformerPlacement,
    attention_flops,
    attention_half,
    ci_fn_linear,
    input_attention_norm_parameter_census,
    normalized_tap_concatenation,
    rms_norm_maybe_scaled,
    swiglu_ffn_flops,
)
from param_decomp.core.ci_fn.interface import CI, CIFnMatrix, SiteDict
from param_decomp.core.components import (
    BlockedFactorization,
    BlockSelection,
    SelectedCI,
    SiteCI,
    SiteSpec,
)
from param_decomp.core.flops.types import ForwardBackwardFlops, MatrixParameters, ParameterCensus
from param_decomp.core.model import CaptureKeys, ComponentActivations, PositionAxis
from param_decomp.core.muon_stacked import StageInPlace
from param_decomp.core.placement import (
    CIFnWeightPlacement,
    PlacementRules,
    StackCensus,
    gather_reduced_weights,
    reachable_placed_rules,
)
from param_decomp.core.precision import COMPUTE_DT, cast_floating
from param_decomp.routed.dense import (
    dense_expert_matmul,
    dense_project_and_combine_experts,
    dense_select_experts,
)
from param_decomp.routed.experts import (
    ExpertImplementation,
    GroupedMatmulBackend,
)
from param_decomp.sequence import SequenceLayout


@dataclass(frozen=True)
class BlockSelectedTransformerCIFnPlacement:
    """The block-selected rows bound to the run's mesh and the chunk-stack census they
    resolve for this arch."""

    rows: PlacedBlockSelectedCIFnRows
    chunks: StackCensus

    def layers(self) -> CIFnTransformerPlacement:
        return self.rows.chunkwise.layers()

    def chunk_census(self, n_chunks: int) -> StackCensus:
        return validated_census(self.chunks, n_chunks)

    def muon_staging(self) -> BlockSelectedMuonStaging:
        return self.rows.muon_staging()

    def constrain_activation(self, x: Array) -> Array:
        return self.rows.chunkwise.constrain_activation(x)

    def description_for_log(self) -> str:
        return chunk_placement_for_log(reachable_placed_rules(self.rows), self.chunks)

    def chunk_shardings(
        self, chunks: "BlockSelectedChunkTransformer", n_chunks: int
    ) -> "BlockSelectedChunkTransformer":
        validate_chunk_leaves(chunks, self.chunk_census(n_chunks))
        return chunks.shardings(self.rows)

    def enter_compute_layout(
        self, chunks: "BlockSelectedChunkTransformer"
    ) -> "BlockSelectedChunkTransformer":
        """Stored chunks in their compute layout; off-mesh they are already there."""
        if jax.sharding.get_abstract_mesh().empty:
            return chunks
        return reconstruct_block_selected_ci_fn_compute_weights(chunks, self.rows)


@dataclass(frozen=True)
class UnplacedBlockSelectedCIFn:
    """Off-mesh execution: plain matmuls, the real chunks unpadded, Muon in place, and
    routed experts dispatched in token order."""

    def layers(self) -> CIFnTransformerPlacement:
        return LOCAL_CI_FN_TRANSFORMER_PLACEMENT

    def chunk_census(self, n_chunks: int) -> StackCensus:
        return unpadded_census(n_chunks)

    def muon_staging(self) -> BlockSelectedMuonStaging:
        return BlockSelectedMuonStaging(
            chunkwise=CI_FN_TRANSFORMER_STAGING_IN_PLACE,
            expert_ffn=StageInPlace(),
            expert_head=StageInPlace(),
        )

    def constrain_activation(self, x: Array) -> Array:
        return x

    def description_for_log(self) -> str:
        return "ci_fn placement: unplaced"

    def chunk_shardings(
        self, chunks: "BlockSelectedChunkTransformer", n_chunks: int
    ) -> "BlockSelectedChunkTransformer":
        del chunks, n_chunks
        raise AssertionError("an unplaced CI fn has no mesh to shard its parameters over")

    def enter_compute_layout(
        self, chunks: "BlockSelectedChunkTransformer"
    ) -> "BlockSelectedChunkTransformer":
        assert jax.sharding.get_abstract_mesh().empty, (
            "on-mesh CI compute-weight materialization requires placement"
        )
        return chunks


BlockSelectedCIFnPlacement = BlockSelectedTransformerCIFnPlacement | UnplacedBlockSelectedCIFn


@dataclass(frozen=True)
class FullSlot:
    """A chunk output site scored by a dense `[d_model, C]` head — full emission."""

    site: str


@dataclass(frozen=True)
class SelectedSlot:
    """A block-factored output site, scored over its selected blocks under selection
    `selection` (an index into the chunk's `layers`): its head's table dispatches on
    exactly that target layer's picks — the per-(layer, block) parameter identity."""

    site: str
    selection: int


BlockSelectedChunkSlot = FullSlot | SelectedSlot


@dataclass(frozen=True)
class BlockSelectedChunk:
    """One resolved block-selected chunk: the input taps to concatenate, EVERY target
    layer the chunk covers (indices into the pinned `BlockSelection`; each transformer
    block's concat-wide FFN banks dispatch on all of those layers' selections), and the
    output slots in emission order. Authored lab-side; core treats every tap key as
    opaque."""

    input_taps: tuple[str, ...]
    layers: tuple[int, ...]
    slots: tuple[BlockSelectedChunkSlot, ...]

    @property
    def output_sites(self) -> tuple[str, ...]:
        return tuple(slot.site for slot in self.slots)


@dataclass(frozen=True)
class BlockSelectedChunkwiseTransformerCIFnArch:
    """Resolved block-selected chunkwise-transformer arch. Each chunk covers one target
    stage and runs `n_blocks` transformer blocks of configured RoPE attention + a
    CONCAT-WIDE block-selected FFN: one `table_size`-entry swiglu bank per covered
    target layer (`len(chunk.layers)` banks, each entry `d_model -> selected_ffn_hidden`),
    every bank dispatched by ITS layer's pinned selection in the same transformer block —
    so bank entry (layer, e) holds parameters that activate exactly when the target
    picked block (layer, e) — plus an always-on dense swiglu (`shared_ffn_hidden`). There
    is no learned selection and no gelu arm: the FFN mirrors the target's table shape by
    construction. `n_blocks >= 1` is the size lever: each transformer block's banks cost
    one target stage's table parameters.

    `input_dim`, `d_model`, `attention`, and `learned_norm_scale` mean exactly what they
    mean on `ChunkwiseTransformerCIFnArch`: the chunk input is the RMS-normed activation
    taps alone — the selection reaches the chunk as dispatch, not as an input feature."""

    chunks: tuple[BlockSelectedChunk, ...]
    input_dim: int
    d_model: int
    n_blocks: int
    attention: CIFnAttention
    table_size: int
    selected_ffn_hidden: int
    shared_ffn_hidden: int
    learned_norm_scale: bool
    expert_implementation: ExpertImplementation
    """Expert computation strategy; selected output slots retain their declared interface."""

    @property
    def capture_keys(self) -> CaptureKeys:
        """Every chunk's activation taps (the selection arrives with conditioning, not captured)."""
        return frozenset(key for chunk in self.chunks for key in chunk.input_taps)

    def initialize(
        self, sites: tuple[SiteSpec, ...], rules: PlacementRules | None, key: PRNGKeyArray
    ) -> "BlockSelectedChunkwiseTransformerCIFn":
        placement = UnplacedBlockSelectedCIFn() if rules is None else self.resolve_placement(rules)
        return init_block_selected_chunkwise_transformer_ci_fn(self, sites, placement, key)

    def parameter_census(self, sites: tuple[SiteSpec, ...]) -> ParameterCensus:
        return block_selected_chunkwise_transformer_ci_fn_parameters(self, sites)

    def resolve_placement(self, rules: PlacementRules) -> BlockSelectedTransformerCIFnPlacement:
        rows = bind_block_selected_rows(preset_rows(rules.ci_fn), rules.mesh)
        return BlockSelectedTransformerCIFnPlacement(
            rows, resolve_chunk_census(self.placement_requirements(rows), rows.chunkwise)
        )

    def placement_requirements(
        self, rows: PlacedBlockSelectedCIFnRows
    ) -> ChunkedCIFnPlacementRequirements:
        chunkwise = rows.chunkwise.weights
        d, n = self.d_model, len(self.chunks)
        tensors: list[CIFnTensorShape] = [
            CIFnMatrixBatchShape(chunkwise.input, STACKED_INPUT_AXES, (n, self.input_dim, d)),
            CIFnVectorBatchShape(("stack", "d_model"), (n, d)),
        ]
        persist_rows = chunk_persist_rows(rows.chunkwise)
        if self.n_blocks:
            kv_width = d // self.attention.n_heads * self.attention.n_kv_heads
            selected = self.selected_ffn_hidden
            shared = self.shared_ffn_hidden
            tensors.extend(
                (
                    CIFnMatrixBatchShape(chunkwise.attention, STACKED_ATTN_Q_AXES, (n, d, d)),
                    CIFnMatrixBatchShape(
                        chunkwise.attention, STACKED_ATTN_KV_AXES, (n, kv_width, d)
                    ),
                    CIFnMatrixBatchShape(chunkwise.attention, STACKED_ATTN_OUT_AXES, (n, d, d)),
                    CIFnMatrixBatchShape(
                        rows.expert_ffn,
                        CI_FN_SELECTED_FFN_IN_AXES,
                        (n, self.table_size, d, selected),
                    ),
                    CIFnMatrixBatchShape(
                        rows.expert_ffn,
                        CI_FN_SELECTED_FFN_OUT_AXES,
                        (n, self.table_size, selected, d),
                    ),
                    CIFnMatrixBatchShape(chunkwise.ffn, STACKED_FFN_IN_AXES, (n, d, shared)),
                    CIFnMatrixBatchShape(chunkwise.ffn, STACKED_FFN_OUT_AXES, (n, shared, d)),
                )
            )
            # The routed banks and heads dispatch whole experts to their operand shards.
            rows.expert_ffn.operands.validate_shape(("expert",), (self.table_size,))
            rows.expert_head.operands.validate_shape(("expert",), (self.table_size,))
            persist_rows = rows.persist_rows
        return ChunkedCIFnPlacementRequirements(
            n,
            tuple(tensors),
            AttentionHeadCounts(self.attention.n_heads, self.attention.n_kv_heads),
            persist_rows,
        )

    def useful_flops(
        self,
        sites: tuple[SiteSpec, ...],
        batch_size: int,
        positions: PositionAxis,
        *,
        n_selected_blocks_per_token: int | None,
    ) -> ForwardBackwardFlops:
        length = sequence_length_for_flops(batch_size, positions, has_position_axis=True)
        if (
            n_selected_blocks_per_token is None
            or not 0 < n_selected_blocks_per_token <= self.table_size
        ):
            raise ValueError("Block-selected CI requires a target top-k within its table size")
        return block_selected_chunkwise_transformer_ci_fn_flops(
            self, sites, batch_size, length, n_selected_blocks_per_token
        )


def block_selected_chunkwise_transformer_ci_fn_flops(
    arch: BlockSelectedChunkwiseTransformerCIFnArch,
    sites: tuple[SiteSpec, ...],
    batch_size: int,
    sequence_length: int,
    n_selected_blocks_per_token: int,
) -> ForwardBackwardFlops:
    n_tokens = batch_size * sequence_length
    site_by_name = {site.name: site for site in sites}
    input_projection = 2 * n_tokens * arch.input_dim * arch.d_model
    attention = attention_flops(arch.attention, arch.d_model, batch_size, sequence_length)
    shared_ffn = swiglu_ffn_flops(arch.d_model, arch.shared_ffn_hidden, n_tokens)
    forward = 0
    for chunk in arch.chunks:
        n_selected_tokens = n_tokens * len(chunk.layers) * n_selected_blocks_per_token
        selected_ffn = swiglu_ffn_flops(arch.d_model, arch.selected_ffn_hidden, n_selected_tokens)
        forward += input_projection + arch.n_blocks * (attention + selected_ffn + shared_ffn)
        has_full_head = False
        selected_head_selections: set[int] = set()
        for slot in chunk.slots:
            site = site_by_name[slot.site]
            match slot:
                case FullSlot():
                    has_full_head = True
                    forward += 2 * n_tokens * arch.d_model * site.C
                case SelectedSlot(selection=selection):
                    selected_head_selections.add(selection)
                    factorization = site.factorization
                    if not isinstance(factorization, BlockedFactorization):
                        raise ValueError(f"Selected CI head {site.name!r} needs blocked components")
                    forward += (
                        2
                        * n_tokens
                        * n_selected_blocks_per_token
                        * arch.selected_ffn_hidden
                        * factorization.c_per_block
                    )
        if not has_full_head:
            # Selected heads read bank hiddens, so the final residual has no consumer.
            forward -= selected_ffn + shared_ffn
            forward += (
                4
                * n_tokens
                * n_selected_blocks_per_token
                * len(selected_head_selections)
                * arch.d_model
                * arch.selected_ffn_hidden
            )
    return ForwardBackwardFlops(forward, 2 * forward - len(arch.chunks) * input_projection)


def block_selected_chunkwise_transformer_ci_fn_parameters(
    arch: BlockSelectedChunkwiseTransformerCIFnArch, sites: tuple[SiteSpec, ...]
) -> ParameterCensus:
    input_attention_norms = input_attention_norm_parameter_census(
        input_dim=arch.input_dim,
        width=arch.d_model,
        n_blocks=arch.n_blocks,
        attention=arch.attention,
        learned_norm_scale=arch.learned_norm_scale,
        n_chunks=len(arch.chunks),
    )
    matrices = list(input_attention_norms.matrices)
    n_vector_parameters = input_attention_norms.n_vector_parameters
    site_by_name = {site.name: site for site in sites}
    for chunk in arch.chunks:
        has_full_head = any(isinstance(slot, FullSlot) for slot in chunk.slots)
        n_residual_blocks = arch.n_blocks if has_full_head else arch.n_blocks - 1
        if n_residual_blocks:
            matrices.extend(
                (
                    MatrixParameters(
                        arch.d_model,
                        arch.selected_ffn_hidden,
                        3 * n_residual_blocks * len(chunk.layers) * arch.table_size,
                    ),
                    MatrixParameters(arch.d_model, arch.shared_ffn_hidden, 3 * n_residual_blocks),
                )
            )
        if not has_full_head:
            selections = {slot.selection for slot in chunk.slots if isinstance(slot, SelectedSlot)}
            matrices.append(
                MatrixParameters(
                    arch.d_model, arch.selected_ffn_hidden, 2 * len(selections) * arch.table_size
                )
            )
        for slot in chunk.slots:
            site = site_by_name[slot.site]
            match slot:
                case FullSlot():
                    matrices.append(MatrixParameters(arch.d_model, site.C, 1))
                    n_vector_parameters += site.C
                case SelectedSlot():
                    if not isinstance(site.factorization, BlockedFactorization):
                        raise ValueError("Selected CI outputs require blocked components")
                    matrices.append(
                        MatrixParameters(
                            arch.selected_ffn_hidden,
                            site.factorization.c_per_block,
                            arch.table_size,
                        )
                    )
    return ParameterCensus(tuple(matrices), n_vector_parameters)


@dataclass(frozen=True)
class _RoutedDispatch:
    selection: Selection
    backend: GroupedMatmulBackend

    def gather(self, selection: int, h: Array) -> Array:
        return self.selection.gather(selection, h)

    def block_matmul(self, selection: int, values: Array, table: Array) -> Array:
        return self.selection.block_matmul(selection, values, table, self.backend)

    def project_and_combine(self, selection: int, hidden: Array, projection: Array) -> Array:
        projected = self.selection.block_matmul(selection, hidden, projection, self.backend)
        return self.selection.combine(selection, projected)

    def selected_ci(self, selection: int, values: Array, n_blocks: int) -> SiteCI:
        return self.selection.selected_ci(selection, values, n_blocks)


@dataclass(frozen=True)
class _DenseDispatch:
    indices: Int[Array, "R b t k"]
    weights: Float[Array, "R b t k"]

    def gather(self, _selection: int, h: Array) -> Array:
        return h[..., None, :]

    def block_matmul(self, _selection: int, values: Array, table: Array) -> Array:
        return dense_expert_matmul(values, table)

    def project_and_combine(
        self,
        selection: int,
        hidden: Float[Array, "b t E h"],
        projection: Float[Array, "E h d"],
    ) -> Float[Array, "b t d"]:
        return dense_project_and_combine_experts(
            hidden, projection, self.indices[selection], self.weights[selection]
        )

    def selected_ci(self, selection: int, values: Array, n_blocks: int) -> SelectedCI:
        ids = self.indices[selection]
        selected = dense_select_experts(values, ids)
        return SelectedCI(selected.reshape(*ids.shape[:-1], -1), ids, n_blocks)


SelectionDispatch = _RoutedDispatch | _DenseDispatch
"""Expert computation is independent of the selected output interface."""


def _selection_dispatch(
    ids: Int[Array, "R b t k"],
    weights: Float[Array, "R b t k"],
    table_size: int,
    placement: BlockSelectedCIFnPlacement,
    implementation: ExpertImplementation,
) -> SelectionDispatch:
    match implementation:
        case "dense_masked":
            return _DenseDispatch(ids, weights)
        case "ragged_dot" | "tokamax" | "tokamax_split_vjp":
            match placement:
                case UnplacedBlockSelectedCIFn():
                    selection = token_selection(ids, weights, table_size)
                case BlockSelectedTransformerCIFnPlacement(rows=rows):
                    selection = block_sharded_selection(
                        ids, weights, table_size, rows.expert_ffn.operands
                    )
            return _RoutedDispatch(selection, implementation)


class BlockSelectedCIFnBlock(eqx.Module):
    """Pre-norm transformer block: RMSNorm → configured RoPE attention → residual;
    RMSNorm → concat-wide block-selected FFN → residual. The attention half is
    `CIFnBlock`'s exactly (`attention_half`). The FFN is the target STAGE's table shape:
    one `E`-entry swiglu bank per covered target layer, bank r dispatched by selection
    r's pinned picks — every token activates its `R·k` picked entries — fp32
    selection-weighted combines scaled 1/R (each selection's renormalized weights sum
    to 1, so R banks would write residual mass R where a target block writes 1), plus
    the always-on dense swiglu. No FFN biases — the bank leaves are exactly the
    grouped-matmul operand shapes. The banks are LENGTH-R TUPLES of `[E, ., .]` leaves, not one
    `[R, E, ., .]` axis: the stacked bundle's leaves stay rank-4 — the grouped-matmul
    rhs, the muon 4D canonical fold, and the placement rows all consume that rank
    directly. `norm_scales` as on `CIFnBlock`."""

    wq: Array
    wk: Array
    wv: Array
    wo: Array
    selected_gate: tuple[Float[Array, "E d_model selected_ffn"], ...]
    selected_up: tuple[Float[Array, "E d_model selected_ffn"], ...]
    selected_down: tuple[Float[Array, "E selected_ffn d_model"], ...]
    shared_gate: Float[Array, "d_model shared_ffn"]
    shared_up: Float[Array, "d_model shared_ffn"]
    shared_down: Float[Array, "shared_ffn d_model"]
    norm_scales: tuple[Array, Array] | None
    attention: CIFnAttention = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def shardings(self, placement: PlacedBlockSelectedCIFnRows) -> "BlockSelectedCIFnBlock":
        """Attention at the ci_fn/attention rows, selected banks at the expert_ffn rows
        (block axis co-located with the target's block shard), the shared swiglu at the
        ci_fn/ffn rows, norm scales at the vectors row."""
        attention = placement.chunkwise.weights.attention.optimizer_state
        ffn = placement.chunkwise.weights.ffn.optimizer_state
        bank_row = placement.expert_ffn.optimizer_state
        attention.validate_shape(STACKED_ATTN_Q_AXES, self.wq.shape)
        attention.validate_shape(STACKED_ATTN_KV_AXES, self.wk.shape)
        attention.validate_shape(STACKED_ATTN_KV_AXES, self.wv.shape)
        attention.validate_shape(STACKED_ATTN_OUT_AXES, self.wo.shape)
        for gate, up, down in zip(
            self.selected_gate, self.selected_up, self.selected_down, strict=True
        ):
            bank_row.validate_shape(CI_FN_SELECTED_FFN_IN_AXES, gate.shape)
            bank_row.validate_shape(CI_FN_SELECTED_FFN_IN_AXES, up.shape)
            bank_row.validate_shape(CI_FN_SELECTED_FFN_OUT_AXES, down.shape)
        ffn.validate_shape(STACKED_FFN_IN_AXES, self.shared_gate.shape)
        ffn.validate_shape(STACKED_FFN_IN_AXES, self.shared_up.shape)
        ffn.validate_shape(STACKED_FFN_OUT_AXES, self.shared_down.shape)
        bank_in = bank_row.sharding_for(CI_FN_SELECTED_FFN_IN_AXES)
        bank_out = bank_row.sharding_for(CI_FN_SELECTED_FFN_OUT_AXES)
        placed = eqx.tree_at(
            lambda b: (
                b.wq,
                b.wk,
                b.wv,
                b.wo,
                b.selected_gate,
                b.selected_up,
                b.selected_down,
                b.shared_gate,
                b.shared_up,
                b.shared_down,
            ),
            self,
            (
                attention.sharding_for(STACKED_ATTN_Q_AXES),
                attention.sharding_for(STACKED_ATTN_KV_AXES),
                attention.sharding_for(STACKED_ATTN_KV_AXES),
                attention.sharding_for(STACKED_ATTN_OUT_AXES),
                tuple(bank_in for _ in self.selected_gate),
                tuple(bank_in for _ in self.selected_up),
                tuple(bank_out for _ in self.selected_down),
                ffn.sharding_for(STACKED_FFN_IN_AXES),
                ffn.sharding_for(STACKED_FFN_IN_AXES),
                ffn.sharding_for(STACKED_FFN_OUT_AXES),
            ),
        )
        if self.norm_scales is not None:
            vectors = placement.chunkwise.vectors
            norm = vector_sharding(vectors, ("stack", "d_model"), self.norm_scales[0].shape)
            placed = eqx.tree_at(lambda b: b.norm_scales, placed, (norm, norm))
        return placed

    def __call__(
        self,
        x: Float[Array, "b t d"],
        inv_freq: Array,
        dispatch: SelectionDispatch,
        *,
        sequence: SequenceLayout,
        placement: CIFnTransformerPlacement,
    ) -> tuple[Float[Array, "b t d"], tuple[Array, ...]]:
        """Returns the residual and bank hidden states in the chosen computation layout —
        the last transformer block's feed the selected heads before interface conversion."""
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
        shared_ffn = placement.ffn
        n_selections = len(self.selected_gate)
        selected = jnp.zeros_like(x)
        hidden_values: list[Array] = []
        for selection in range(n_selections):
            expert_inputs = dispatch.gather(selection, h)
            gate = dispatch.block_matmul(selection, expert_inputs, self.selected_gate[selection])
            up = dispatch.block_matmul(selection, expert_inputs, self.selected_up[selection])
            hidden = jax.nn.silu(gate) * up
            selected = selected + dispatch.project_and_combine(
                selection, hidden, self.selected_down[selection]
            )
            hidden_values.append(hidden)
        shared_gate = ci_fn_linear(
            h, self.shared_gate, shared_ffn, CI_FN_FFN_IN_AXES, transposed=False
        )
        shared_up = ci_fn_linear(h, self.shared_up, shared_ffn, CI_FN_FFN_IN_AXES, transposed=False)
        shared = ci_fn_linear(
            jax.nn.silu(shared_gate) * shared_up,
            self.shared_down,
            shared_ffn,
            CI_FN_FFN_OUT_AXES,
            transposed=False,
        )
        return x + selected / n_selections + shared, tuple(hidden_values)


class DenseCIFnHead(eqx.Module):
    """One full-emission site head: `x_final @ w + b -> [*leading, C]`."""

    w: Float[Array, "d_model C"]
    b: Float[Array, " C"]


class BlockedCIFnHead(eqx.Module):
    """One selected-emission site head, FUSED into the bank's table: entry (e, :) reads
    the LAST transformer block's (selection, e) hidden state on that pick's jobs — the
    same computation layout as its swiglu — and emits the pick's `c` preactivations
    in token/selected-slot order. The head's parameters activate
    exactly when its (layer, block) is picked. Biasless like the banks: the leaf is
    exactly the grouped-matmul operand shape."""

    w: Float[Array, "E selected_ffn c"]
    selection: int = eqx.field(static=True)


CIFnHead = DenseCIFnHead | BlockedCIFnHead
"""Per-slot head union: the static discriminator rides the treedef, so the stacked
bundle stays rectangular per slot while slots differ in emission."""


class BlockSelectedChunkTransformer(eqx.Module):
    """ONE block-selected chunk: its (already assembled, concatenated) input
    `[b, t, total_d_in]` → in_proj → `n_blocks` `BlockSelectedCIFnBlock`s (every one
    dispatching on all `R` pinned selections) → one head per output slot:
    `DenseCIFnHead`s read the final residual full-width; `BlockedCIFnHead`s read the last
    transformer block's job-space hidden states and emit selected bundles carrying
    the dispatch's routing. In the bundle every array leaf carries a leading
    `n_chunks` axis and the module runs under a `jax.lax.scan` over that axis, exactly
    as `CIFnTransformer` does. Its blocks are individually checkpointed so the
    differentiated chunk does not retain every expert bank's internal activations."""

    in_proj_w: Float[Array, "total_d_in d_model"]
    in_proj_b: Float[Array, " d_model"]
    blocks: list[BlockSelectedCIFnBlock]
    heads: tuple[CIFnHead, ...]

    def shardings(self, placement: PlacedBlockSelectedCIFnRows) -> "BlockSelectedChunkTransformer":
        """in_proj at ci_fn/input, dense heads at ci_fn/output, blocked heads at the
        expert_head rows; biases at the vectors row."""
        input_row = placement.chunkwise.weights.input.optimizer_state
        output_row = placement.chunkwise.weights.output.optimizer_state
        head_row = placement.expert_head.optimizer_state
        input_row.validate_shape(STACKED_INPUT_AXES, self.in_proj_w.shape)
        vectors = placement.chunkwise.vectors
        placed_heads: list[CIFnHead] = []
        for head in self.heads:
            match head:
                case DenseCIFnHead():
                    output_row.validate_shape(STACKED_OUTPUT_AXES, head.w.shape)
                    placed_heads.append(
                        DenseCIFnHead(
                            w=output_row.sharding_for(STACKED_OUTPUT_AXES),  # pyright: ignore[reportArgumentType]
                            b=vector_sharding(vectors, ("stack", "C"), head.b.shape),  # pyright: ignore[reportArgumentType]
                        )
                    )
                case BlockedCIFnHead():
                    head_row.validate_shape(CI_FN_BLOCKED_HEAD_AXES, head.w.shape)
                    placed_heads.append(
                        BlockedCIFnHead(
                            w=head_row.sharding_for(CI_FN_BLOCKED_HEAD_AXES),  # pyright: ignore[reportArgumentType]
                            selection=head.selection,
                        )
                    )
        return eqx.tree_at(
            lambda ct: (ct.in_proj_w, ct.in_proj_b, ct.blocks, ct.heads),
            self,
            (
                input_row.sharding_for(STACKED_INPUT_AXES),
                vector_sharding(vectors, ("stack", "d_model"), self.in_proj_b.shape),
                [b.shardings(placement) for b in self.blocks],
                tuple(placed_heads),
            ),
        )

    def __call__(
        self,
        x: Float[Array, "b t total_d_in"],
        inv_freq: Array,
        dispatch: SelectionDispatch,
        *,
        sequence: SequenceLayout,
        placement: CIFnTransformerPlacement,
    ) -> tuple[SiteCI, ...]:
        x = ci_fn_linear(x, self.in_proj_w, placement.input, CI_FN_INPUT_AXES, transposed=False)
        x = x + self.in_proj_b
        hidden_values: tuple[Array, ...] = ()
        for block in self.blocks:
            x, hidden_values = eqx.filter_checkpoint(block)(
                x, inv_freq, dispatch, sequence=sequence, placement=placement
            )
        outputs: list[SiteCI] = []
        for head in self.heads:
            match head:
                case DenseCIFnHead(w=w, b=b):
                    outputs.append(
                        ci_fn_linear(x, w, placement.output, CI_FN_OUTPUT_AXES, transposed=False)
                        + b
                    )
                case BlockedCIFnHead(w=w, selection=selection):
                    values = dispatch.block_matmul(selection, hidden_values[selection], w)
                    outputs.append(dispatch.selected_ci(selection, values, w.shape[0]))
        return tuple(outputs)


class RoutingConditioning(Protocol):
    """The clean block selection consumed by a routed CI function."""

    @property
    def selection(self) -> BlockSelection: ...


@dataclass(frozen=True)
class _BlockSelectedChunkMeta:
    """Per-chunk static routing, index-aligned with the stacked `chunks` leading axis."""

    input_taps: tuple[str, ...]
    layers: tuple[int, ...]
    slots: tuple[BlockSelectedChunkSlot, ...]


class BlockSelectedChunkwiseTransformerCIFn(eqx.Module):
    """`ChunkwiseTransformerBackbone`'s block-selected sibling: stacked
    `BlockSelectedChunkTransformer`s under a `jax.lax.scan` with per-chunk remat.
    Block-factored sites emit `SelectedCI` bundles — each selected site's values leave
    this fn already married to the block indices that key them (the slot's
    `selection`); dense sites emit full-width arrays. Each chunk's input is its
    RMS-normed activation taps concatenated, exactly as on the dense chunkwise fn: the
    selection enters as dispatch alone, never as an input feature."""

    chunks: BlockSelectedChunkTransformer  # arrays stacked along leading n_chunks (+ stack_pad)
    inv_freq: Array  # shared across chunks (RoPE buffer); NOT mapped

    capture_keys: CaptureKeys = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)
    chunk_meta: tuple[_BlockSelectedChunkMeta, ...] = eqx.field(static=True)
    placement: BlockSelectedCIFnPlacement = eqx.field(static=True)
    table_size: int = eqx.field(static=True)
    expert_implementation: ExpertImplementation = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)

    @property
    def census(self) -> StackCensus:
        """As on `ChunkwiseTransformerBackbone`: the real chunks and the zero chunks trailing
        them on every `chunks` leaf."""
        return self.placement.chunk_census(len(self.chunk_meta))

    def placement_for_log(self) -> str:
        return self.placement.description_for_log()

    def matrix_parameters(self) -> tuple[CIFnMatrix, ...]:
        staging = self.placement.muon_staging()
        chunkwise = staging.chunkwise
        chunks = self.chunks
        matrices = [CIFnMatrix(chunks.in_proj_w, chunkwise.input)]
        for head in chunks.heads:
            match head:
                case DenseCIFnHead():
                    matrices.append(CIFnMatrix(head.w, chunkwise.output))
                case BlockedCIFnHead():
                    matrices.append(CIFnMatrix(head.w, staging.expert_head))
        for block in chunks.blocks:
            matrices.extend(
                (
                    CIFnMatrix(block.wq, chunkwise.attention),
                    CIFnMatrix(block.wk, chunkwise.attention),
                    CIFnMatrix(block.wv, chunkwise.attention),
                    CIFnMatrix(block.wo, chunkwise.attention),
                    *(CIFnMatrix(w, staging.expert_ffn) for w in block.selected_gate),
                    *(CIFnMatrix(w, staging.expert_ffn) for w in block.selected_up),
                    *(CIFnMatrix(w, staging.expert_ffn) for w in block.selected_down),
                    CIFnMatrix(block.shared_gate, chunkwise.ffn),
                    CIFnMatrix(block.shared_up, chunkwise.ffn),
                    CIFnMatrix(block.shared_down, chunkwise.ffn),
                )
            )
        return tuple(matrices)

    def prepare(self) -> "BlockSelectedChunkwiseTransformerCIFn":
        validate_chunk_leaves(self.chunks, self.census)
        compute = cast_floating(self, COMPUTE_DT)
        return replace(compute, chunks=self.placement.enter_compute_layout(compute.chunks))

    def shardings(self, mesh: Mesh) -> "BlockSelectedChunkwiseTransformerCIFn":
        """The stacked per-chunk transformer's persist layout (`BlockSelectedChunkTransformer.
        shardings` at the padded chunk extent); `inv_freq` replicates."""
        return eqx.tree_at(
            lambda f: (f.chunks, f.inv_freq),
            self,
            (
                self.placement.chunk_shardings(self.chunks, len(self.chunk_meta)),
                NamedSharding(mesh, P()),
            ),
        )

    def _chunk_input(self, meta: _BlockSelectedChunkMeta, taps: dict[str, Array]) -> Array:
        return normalized_tap_concatenation(
            [taps[key] for key in meta.input_taps], self.placement.constrain_activation, self.eps
        )

    def __call__(
        self,
        taps: dict[str, Array],
        conditioning: RoutingConditioning,
        components: ComponentActivations,
        *,
        sequence: SequenceLayout | None,
        remat: bool,
    ) -> CI:
        del components
        layers = self.placement.layers()
        taps = cast_floating(taps, COMPUTE_DT)
        # the selection enters at compute precision like every other CI input (the ids stay integer)
        selection = BlockSelection(
            conditioning.selection.indices, conditioning.selection.weights.astype(COMPUTE_DT)
        )
        per_chunk_in = [self._chunk_input(meta, taps) for meta in self.chunk_meta]
        if sequence is None:
            sequence = SequenceLayout.unsegmented_sequences_like(per_chunk_in[0][..., 0])
        stacked_in = jnp.stack(per_chunk_in, axis=0)  # [n_chunks, b, t, total_d_in]
        stacked_ids = jnp.stack(
            [jnp.stack([selection.indices[layer] for layer in m.layers]) for m in self.chunk_meta]
        )  # [n_chunks, R, b, t, k]
        stacked_weights = jnp.stack(
            [jnp.stack([selection.weights[layer] for layer in m.layers]) for m in self.chunk_meta]
        )
        inv_freq = jax.lax.stop_gradient(self.inv_freq)
        chunks = real_chunks(self.chunks, self.census)
        chunk_arrays, chunk_static = eqx.partition(chunks, eqx.is_array)

        def run_chunk(
            _: None, scanned: tuple[BlockSelectedChunkTransformer, Array, Array, Array]
        ) -> tuple[None, tuple[SiteCI, ...]]:
            chunk_array, chunk_input, ids, weights = scanned
            chunk = eqx.combine(chunk_array, chunk_static)
            dispatch = _selection_dispatch(
                ids, weights, self.table_size, self.placement, self.expert_implementation
            )
            return None, chunk(chunk_input, inv_freq, dispatch, sequence=sequence, placement=layers)

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
            for slot, chunk_slot in enumerate(meta.slots):
                # The scan stacks routing metadata alongside selected values.
                preactivations[chunk_slot.site] = jax.tree.map(
                    lambda a, i=chunk_idx: a[i], stacked_per_slot[slot]
                )
        return CI.from_preactivations(preactivations)


def reconstruct_block_selected_ci_fn_compute_weights(
    chunks: BlockSelectedChunkTransformer, placement: PlacedBlockSelectedCIFnRows
) -> BlockSelectedChunkTransformer:
    """`reconstruct_ci_fn_compute_weights` over the block-selected chunk module: the
    selected banks and fused heads enter through the expert rows, pads included."""
    rows = placement.chunkwise.weights

    def enter(x: Array, weights: CIFnWeightPlacement, axes: Axes) -> Array:
        return gather_reduced_weights(
            x, source=weights.optimizer_state, destination=weights.compute_weights, axes=axes
        )

    def head(h: CIFnHead) -> CIFnHead:
        match h:
            case DenseCIFnHead():
                return replace(h, w=enter(h.w, rows.output, STACKED_OUTPUT_AXES))
            case BlockedCIFnHead():
                return replace(h, w=enter(h.w, placement.expert_head, CI_FN_BLOCKED_HEAD_AXES))

    def block(blk: BlockSelectedCIFnBlock) -> BlockSelectedCIFnBlock:
        return replace(
            blk,
            wq=enter(blk.wq, rows.attention, STACKED_ATTN_Q_AXES),
            wk=enter(blk.wk, rows.attention, STACKED_ATTN_KV_AXES),
            wv=enter(blk.wv, rows.attention, STACKED_ATTN_KV_AXES),
            wo=enter(blk.wo, rows.attention, STACKED_ATTN_OUT_AXES),
            selected_gate=tuple(
                enter(w, placement.expert_ffn, CI_FN_SELECTED_FFN_IN_AXES)
                for w in blk.selected_gate
            ),
            selected_up=tuple(
                enter(w, placement.expert_ffn, CI_FN_SELECTED_FFN_IN_AXES) for w in blk.selected_up
            ),
            selected_down=tuple(
                enter(w, placement.expert_ffn, CI_FN_SELECTED_FFN_OUT_AXES)
                for w in blk.selected_down
            ),
            shared_gate=enter(blk.shared_gate, rows.ffn, STACKED_FFN_IN_AXES),
            shared_up=enter(blk.shared_up, rows.ffn, STACKED_FFN_IN_AXES),
            shared_down=enter(blk.shared_down, rows.ffn, STACKED_FFN_OUT_AXES),
        )

    return replace(
        chunks,
        in_proj_w=enter(chunks.in_proj_w, rows.input, STACKED_INPUT_AXES),
        blocks=[block(blk) for blk in chunks.blocks],
        heads=tuple(head(h) for h in chunks.heads),
    )


type _SlotSignature = tuple[Literal["full"], int] | tuple[Literal["selected"], int, int]


def _block_selected_slot_signature(
    chunk: BlockSelectedChunk, site_spec: dict[str, SiteSpec], table_size: int
) -> tuple[_SlotSignature, ...]:
    """One chunk's per-slot (emission, shape) signature — equal across chunks ⟺ the
    stacked bundle is rectangular and every slot means the same thing in every chunk."""
    signature: list[_SlotSignature] = []
    for slot in chunk.slots:
        spec = site_spec[slot.site]
        match slot:
            case FullSlot():
                signature.append(("full", spec.C))
            case SelectedSlot(selection=selection):
                factorization = spec.factorization
                assert isinstance(factorization, BlockedFactorization), (
                    f"selected slot {slot.site!r} needs a block-factored site, "
                    f"got {type(factorization).__name__}"
                )
                assert factorization.n_blocks == table_size, (
                    slot.site,
                    factorization.n_blocks,
                    table_size,
                )
                assert 0 <= selection < len(chunk.layers), (slot.site, selection, len(chunk.layers))
                signature.append(("selected", selection, factorization.c_per_block))
    return tuple(signature)


def _init_block_selected_chunk_transformer(
    arch: BlockSelectedChunkwiseTransformerCIFnArch,
    slot_signature: tuple[_SlotSignature, ...],
    n_selections: int,
    key: PRNGKeyArray,
) -> BlockSelectedChunkTransformer:
    """One block-selected chunk's params under the chunkwise Kaiming scheme: relu-gain (√2) on
    in_proj / gate / up projections, linear gain (1) on down projections and heads,
    PyTorch-default `U(±1/√fan_in)` on the attention projections, zero biases. Each
    consumer takes its OWN explicit key — the split counts live next to their use."""
    relu_gain = 2.0**0.5
    d, di, ds = arch.d_model, arch.selected_ffn_hidden, arch.shared_ffn_hidden
    table_size = arch.table_size
    d_kv = (d // arch.attention.n_heads) * arch.attention.n_kv_heads

    def kaiming(k: PRNGKeyArray, shape: tuple[int, ...], fan_in: int, gain: float) -> Array:
        return jax.random.normal(k, shape) * (gain / fan_in**0.5)

    def attn_default(k: PRNGKeyArray, shape: tuple[int, ...], fan_in: int) -> Array:
        bound = 1.0 / fan_in**0.5
        return jax.random.uniform(k, shape, minval=-bound, maxval=bound)

    def block(bkey: PRNGKeyArray) -> BlockSelectedCIFnBlock:
        # 4 attention + 3 per selection bank + 3 shared draws; the split count derives
        # every key, so it lives here, next to the draws.
        kq, kk, kv, ko, *rest = jax.random.split(bkey, 4 + 3 * n_selections + 3)
        gate_keys, up_keys = rest[:n_selections], rest[n_selections : 2 * n_selections]
        down_keys = rest[2 * n_selections : 3 * n_selections]
        ksg, ksu, ksd = rest[3 * n_selections :]
        norm_scales = (jnp.ones((d,)), jnp.ones((d,))) if arch.learned_norm_scale else None
        return BlockSelectedCIFnBlock(
            wq=attn_default(kq, (d, d), d),
            wk=attn_default(kk, (d_kv, d), d),
            wv=attn_default(kv, (d_kv, d), d),
            wo=attn_default(ko, (d, d), d),
            selected_gate=tuple(kaiming(k, (table_size, d, di), d, relu_gain) for k in gate_keys),
            selected_up=tuple(kaiming(k, (table_size, d, di), d, relu_gain) for k in up_keys),
            selected_down=tuple(kaiming(k, (table_size, di, d), di, 1.0) for k in down_keys),
            shared_gate=kaiming(ksg, (d, ds), d, relu_gain),
            shared_up=kaiming(ksu, (d, ds), d, relu_gain),
            shared_down=kaiming(ksd, (ds, d), ds, 1.0),
            norm_scales=norm_scales,
            attention=arch.attention,
            eps=CI_FN_RMS_EPS,
        )

    in_key, heads_key, *block_keys = jax.random.split(key, arch.n_blocks + 2)
    head_keys = jax.random.split(heads_key, len(slot_signature))
    heads: list[CIFnHead] = []
    for slot_sig, head_key in zip(slot_signature, head_keys, strict=True):
        match slot_sig:
            case ("full", c_full):
                heads.append(
                    DenseCIFnHead(w=kaiming(head_key, (d, c_full), d, 1.0), b=jnp.zeros((c_full,)))
                )
            case ("selected", selection, c):
                heads.append(
                    BlockedCIFnHead(
                        w=kaiming(head_key, (table_size, di, c), di, 1.0), selection=selection
                    )
                )
    return BlockSelectedChunkTransformer(
        in_proj_w=kaiming(in_key, (arch.input_dim, d), arch.input_dim, relu_gain),
        in_proj_b=jnp.zeros((d,)),
        blocks=[block(bk) for bk in block_keys],
        heads=tuple(heads),
    )


def init_block_selected_chunkwise_transformer_ci_fn(
    arch: BlockSelectedChunkwiseTransformerCIFnArch,
    sites: tuple[SiteSpec, ...],
    placement: BlockSelectedCIFnPlacement,
    key: PRNGKeyArray,
) -> BlockSelectedChunkwiseTransformerCIFn:
    """Validate the output partition and chunk homogeneity as the dense chunkwise init
    does — plus: every `SelectedSlot` names a block-factored site whose `n_blocks` is
    the arch's `table_size`, every `FullSlot` a dense site, and every chunk
    covers one shared selection count — then build stacked chunk params under the same
    Kaiming scheme, padded to the placement's chunk-stack census."""
    site_spec = {s.name: s for s in sites}
    covered = [slot.site for chunk in arch.chunks for slot in chunk.slots]
    assert sorted(covered) == sorted(site_spec), "chunks must partition sites"
    assert len(covered) == len(set(covered)), "chunks overlap on an output site"
    assert arch.n_blocks >= 1, (
        f"the block-selected chunkwise arch needs n_blocks >= 1 ({arch.n_blocks}): the fused "
        "selected heads read the last transformer block's bank hiddens"
    )
    selection_counts = {len(chunk.layers) for chunk in arch.chunks}
    assert len(selection_counts) == 1, (
        f"chunks not homogeneous in selection count: {selection_counts}"
    )
    (n_selections,) = selection_counts
    signatures = {
        _block_selected_slot_signature(chunk, site_spec, arch.table_size) for chunk in arch.chunks
    }
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
    stacked: BlockSelectedChunkTransformer = eqx.filter_vmap(
        lambda k: _init_block_selected_chunk_transformer(arch, slot_signature, n_selections, k)
    )(chunk_keys)
    chunks = pad_chunk_stack(stacked, placement.chunk_census(len(arch.chunks)))

    return BlockSelectedChunkwiseTransformerCIFn(
        chunks=chunks,
        inv_freq=inv_freq,
        capture_keys=arch.capture_keys,
        output_names=tuple(name for chunk in arch.chunks for name in chunk.output_sites),
        chunk_meta=tuple(
            _BlockSelectedChunkMeta(chunk.input_taps, chunk.layers, chunk.slots)
            for chunk in arch.chunks
        ),
        placement=placement,
        table_size=arch.table_size,
        expert_implementation=arch.expert_implementation,
        eps=CI_FN_RMS_EPS,
        has_position_axis=True,
    )
