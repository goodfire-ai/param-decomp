"""Independent transformers over explicit groups of input taps and output sites."""

from dataclasses import dataclass, replace

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, PRNGKeyArray

from param_decomp.core.axes import Axes
from param_decomp.core.ci_fn.architecture import sequence_length_for_flops
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
    bind_chunkwise_rows,
    chunk_persist_rows,
    chunk_placement_for_log,
    preset_rows,
    resolve_chunk_census,
    vector_sharding,
)
from param_decomp.core.ci_fn.implementations.transformer.backbone import BackboneCIFn
from param_decomp.core.ci_fn.implementations.transformer.layers import (
    CI_FN_INPUT_AXES,
    CI_FN_OUTPUT_AXES,
    CI_FN_RMS_EPS,
    CI_FN_TRANSFORMER_STAGING_IN_PLACE,
    LOCAL_CI_FN_TRANSFORMER_PLACEMENT,
    CIFnAttention,
    CIFnBlock,
    CIFnFfnKind,
    CIFnTransformerMuonStaging,
    CIFnTransformerPlacement,
    ci_fn_linear,
    dense_transformer_flops,
    dense_transformer_parameters,
    init_transformer_parameters,
    normalized_tap_concatenation,
)
from param_decomp.core.ci_fn.implementations.transformer.placement import (
    PlacedTransformerCIFnRows,
)
from param_decomp.core.ci_fn.interface import CIFnMatrix, SiteDict
from param_decomp.core.components import SiteSpec
from param_decomp.core.flops.types import ForwardBackwardFlops, ParameterCensus
from param_decomp.core.model import CaptureKeys, ComponentActivations, PositionAxis
from param_decomp.core.placement import (
    CIFnWeightPlacement,
    PlacementRules,
    StackCensus,
    gather_reduced_weights,
    reachable_placed_rules,
)
from param_decomp.core.precision import COMPUTE_DT, cast_floating
from param_decomp.sequence import SequenceLayout


@dataclass(frozen=True)
class ChunkwiseTransformerCIFnPlacement:
    """The chunkwise rows bound to the run's mesh and the chunk-stack census they
    resolve for this arch."""

    rows: PlacedTransformerCIFnRows
    chunks: StackCensus

    def layers(self) -> CIFnTransformerPlacement:
        return self.rows.layers()

    def chunk_census(self, n_chunks: int) -> StackCensus:
        return validated_census(self.chunks, n_chunks)

    def muon_staging(self) -> CIFnTransformerMuonStaging:
        return self.rows.muon_staging()

    def constrain_activation(self, x: Array) -> Array:
        return self.rows.constrain_activation(x)

    def description_for_log(self) -> str:
        return chunk_placement_for_log(reachable_placed_rules(self.rows), self.chunks)

    def chunk_shardings(self, chunks: "CIFnTransformer", n_chunks: int) -> "CIFnTransformer":
        validate_chunk_leaves(chunks, self.chunk_census(n_chunks))
        return chunks.shardings(self.rows)

    def enter_compute_layout(self, chunks: "CIFnTransformer") -> "CIFnTransformer":
        """Stored chunks in their compute layout; off-mesh they are already there."""
        if jax.sharding.get_abstract_mesh().empty:
            return chunks
        return reconstruct_ci_fn_compute_weights(chunks, self.rows)


@dataclass(frozen=True)
class UnplacedChunkwiseCIFn:
    """Off-mesh execution: plain matmuls, the real chunks unpadded, Muon in place."""

    def layers(self) -> CIFnTransformerPlacement:
        return LOCAL_CI_FN_TRANSFORMER_PLACEMENT

    def chunk_census(self, n_chunks: int) -> StackCensus:
        return unpadded_census(n_chunks)

    def muon_staging(self) -> CIFnTransformerMuonStaging:
        return CI_FN_TRANSFORMER_STAGING_IN_PLACE

    def constrain_activation(self, x: Array) -> Array:
        return x

    def description_for_log(self) -> str:
        return "ci_fn placement: unplaced"

    def chunk_shardings(self, chunks: "CIFnTransformer", n_chunks: int) -> "CIFnTransformer":
        del chunks, n_chunks
        raise AssertionError("an unplaced CI fn has no mesh to shard its parameters over")

    def enter_compute_layout(self, chunks: "CIFnTransformer") -> "CIFnTransformer":
        assert jax.sharding.get_abstract_mesh().empty, (
            "on-mesh CI compute-weight materialization requires placement"
        )
        return chunks


ChunkwiseCIFnPlacement = ChunkwiseTransformerCIFnPlacement | UnplacedChunkwiseCIFn


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
class ChunkwiseTransformerCIFnArch:
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
    target that wants a real per-position CI fn wants `LayerwiseMLPCIFnArch(has_position_axis=True)`.
    Pinned by `param_decomp/tests/core/test_ci_fn_zero_blocks.py` (locality) and
    `param_decomp/tests/core/test_ci_fn_positioned_mlp.py` (the magnitude contrast)."""

    chunks: tuple[Chunk, ...]
    input_dim: int
    d_model: int
    n_blocks: int
    attention: CIFnAttention
    ffn_hidden: int
    ffn_kind: CIFnFfnKind
    learned_norm_scale: bool

    @property
    def capture_keys(self) -> CaptureKeys:
        return frozenset(tap for chunk in self.chunks for tap in chunk.input_taps)

    def initialize(
        self, sites: tuple[SiteSpec, ...], rules: PlacementRules | None, key: PRNGKeyArray
    ) -> BackboneCIFn:
        placement = UnplacedChunkwiseCIFn() if rules is None else self.resolve_placement(rules)
        return BackboneCIFn(init_chunkwise_transformer_backbone(self, sites, placement, key))

    def parameter_census(self, sites: tuple[SiteSpec, ...]) -> ParameterCensus:
        return dense_transformer_parameters(
            self,
            input_dim=self.input_dim,
            n_stacks=len(self.chunks),
            output_widths=tuple(site.C for site in sites),
        )

    def resolve_placement(self, rules: PlacementRules) -> ChunkwiseTransformerCIFnPlacement:
        rows = bind_chunkwise_rows(preset_rows(rules.ci_fn), rules.mesh)
        return ChunkwiseTransformerCIFnPlacement(
            rows, resolve_chunk_census(self.placement_requirements(rows), rows)
        )

    def placement_requirements(
        self, rows: PlacedTransformerCIFnRows
    ) -> ChunkedCIFnPlacementRequirements:
        d, n, weights = self.d_model, len(self.chunks), rows.weights
        tensors: list[CIFnTensorShape] = [
            CIFnMatrixBatchShape(weights.input, STACKED_INPUT_AXES, (n, self.input_dim, d)),
            CIFnVectorBatchShape(("stack", "d_model"), (n, d)),
        ]
        if self.n_blocks:
            kv_width = d // self.attention.n_heads * self.attention.n_kv_heads
            tensors.extend(
                (
                    CIFnMatrixBatchShape(weights.attention, STACKED_ATTN_Q_AXES, (n, d, d)),
                    CIFnMatrixBatchShape(weights.attention, STACKED_ATTN_KV_AXES, (n, kv_width, d)),
                    CIFnMatrixBatchShape(weights.attention, STACKED_ATTN_OUT_AXES, (n, d, d)),
                    CIFnMatrixBatchShape(weights.ffn, STACKED_FFN_IN_AXES, (n, d, self.ffn_hidden)),
                    CIFnMatrixBatchShape(
                        weights.ffn, STACKED_FFN_OUT_AXES, (n, self.ffn_hidden, d)
                    ),
                    CIFnVectorBatchShape(("stack", "ffn_hidden"), (n, self.ffn_hidden)),
                )
            )
        return ChunkedCIFnPlacementRequirements(
            n,
            tuple(tensors),
            AttentionHeadCounts(self.attention.n_heads, self.attention.n_kv_heads),
            chunk_persist_rows(rows),
        )

    def useful_flops(
        self,
        sites: tuple[SiteSpec, ...],
        batch_size: int,
        positions: PositionAxis,
        *,
        n_selected_blocks_per_token: int | None,
    ) -> ForwardBackwardFlops:
        del n_selected_blocks_per_token
        length = sequence_length_for_flops(batch_size, positions, has_position_axis=True)
        return dense_transformer_flops(
            self,
            input_dim=self.input_dim,
            n_stacks=len(self.chunks),
            output_width=sum(site.C for site in sites),
            batch_size=batch_size,
            sequence_length=length,
        )


def _chunk_block_shardings(block: CIFnBlock, placement: PlacedTransformerCIFnRows) -> CIFnBlock:
    """Place stacked attention and FFN parameters at their persistence rows.

    Every large weight's sharding derives from its semantic axes and the placement
    table."""
    attention = placement.weights.attention.optimizer_state
    ffn = placement.weights.ffn.optimizer_state
    attention.validate_shape(STACKED_ATTN_Q_AXES, block.wq.shape)
    attention.validate_shape(STACKED_ATTN_KV_AXES, block.wk.shape)
    attention.validate_shape(STACKED_ATTN_KV_AXES, block.wv.shape)
    attention.validate_shape(STACKED_ATTN_OUT_AXES, block.wo.shape)
    ffn.validate_shape(STACKED_FFN_IN_AXES, block.w1.shape)
    ffn.validate_shape(STACKED_FFN_OUT_AXES, block.w2.shape)
    attn_q = attention.sharding_for(STACKED_ATTN_Q_AXES)
    attn_kv = attention.sharding_for(STACKED_ATTN_KV_AXES)
    attn_out = attention.sharding_for(STACKED_ATTN_OUT_AXES)
    ffn_in = ffn.sharding_for(STACKED_FFN_IN_AXES)
    ffn_out = ffn.sharding_for(STACKED_FFN_OUT_AXES)
    vectors = placement.vectors
    b1 = vector_sharding(vectors, ("stack", "ffn_hidden"), block.b1.shape)
    b2 = vector_sharding(vectors, ("stack", "d_model"), block.b2.shape)
    placed = eqx.tree_at(
        lambda b: (b.wq, b.wk, b.wv, b.wo, b.w1, b.b1, b.w2, b.b2),
        block,
        (attn_q, attn_kv, attn_kv, attn_out, ffn_in, b1, ffn_out, b2),
    )
    if block.gate is not None:
        # The swiglu gate is a second `[nc, d_model, ffn_hidden]` up-proj: same
        # Megatron-on-ffn_hidden placement as w1, same ÷N divisibility requirement.
        ffn.validate_shape(STACKED_FFN_IN_AXES, block.gate[0].shape)
        gate_bias = vector_sharding(vectors, ("stack", "ffn_hidden"), block.gate[1].shape)
        placed = eqx.tree_at(lambda b: b.gate, placed, (ffn_in, gate_bias))
    if block.norm_scales is not None:
        norm = vector_sharding(vectors, ("stack", "d_model"), block.norm_scales[0].shape)
        placed = eqx.tree_at(lambda b: b.norm_scales, placed, (norm, norm))
    return placed


class CIFnTransformer(eqx.Module):
    """Transformer CI predictor for one chunk of decomposed sites.

    RMS-normalized, concatenated inputs `[*leading, total_d_in]` pass through an
    input projection and RoPE blocks to per-site preactivations `[*leading, C_j]`.
    Each site has its own output head, matching its component and mask width.

    Chunkwise CI stacks independent instances of this body.
    Each block is checkpointed separately to bound its backward intermediates."""

    in_proj_w: Float[Array, "total_d_in d_model"]
    in_proj_b: Float[Array, " d_model"]
    blocks: list[CIFnBlock]
    out_ws: tuple[Float[Array, "d_model _C"], ...]
    out_bs: tuple[Float[Array, " _C"], ...]

    def shardings(self, placement: PlacedTransformerCIFnRows) -> "CIFnTransformer":
        """Place the complete stacked CI transformer from its semantic placement rows."""
        input_row = placement.weights.input.optimizer_state
        output_row = placement.weights.output.optimizer_state
        input_row.validate_shape(STACKED_INPUT_AXES, self.in_proj_w.shape)
        for w in self.out_ws:
            output_row.validate_shape(STACKED_OUTPUT_AXES, w.shape)
        in_proj_sh = input_row.sharding_for(STACKED_INPUT_AXES)
        out_ws_sh = output_row.sharding_for(STACKED_OUTPUT_AXES)
        vectors = placement.vectors
        in_proj_b = vector_sharding(vectors, ("stack", "d_model"), self.in_proj_b.shape)
        out_bs = tuple(vector_sharding(vectors, ("stack", "C"), bias.shape) for bias in self.out_bs)
        return eqx.tree_at(
            lambda ct: (ct.in_proj_w, ct.in_proj_b, ct.blocks, ct.out_ws, ct.out_bs),
            self,
            (
                in_proj_sh,
                in_proj_b,
                [_chunk_block_shardings(b, placement) for b in self.blocks],
                tuple(out_ws_sh for _ in self.out_ws),
                out_bs,
            ),
        )

    def __call__(
        self,
        x: Float[Array, "*leading total_d_in"],
        inv_freq: Array,
        *,
        placement: CIFnTransformerPlacement,
        sequence: SequenceLayout,
    ) -> tuple[Float[Array, "*leading _C"], ...]:
        x = ci_fn_linear(x, self.in_proj_w, placement.input, CI_FN_INPUT_AXES, transposed=False)
        x = x + self.in_proj_b
        for block in self.blocks:
            x = eqx.filter_checkpoint(block)(
                x,
                inv_freq,
                placement=placement,
                sequence=sequence,
            )
        return tuple(
            ci_fn_linear(x, w, placement.output, CI_FN_OUTPUT_AXES, transposed=False) + b
            for w, b in zip(self.out_ws, self.out_bs, strict=True)
        )


def reconstruct_ci_fn_compute_weights(
    chunks: CIFnTransformer, placement: PlacedTransformerCIFnRows
) -> CIFnTransformer:
    """The stacked chunk module's entry: every weight leaf gathered from its persist row
    to its compute row, pads included (`gather_reduced_weights`; the pad policy is in
    `chunk_stack`). The leaves arrive compute-dtype, since the whole fn is cast first. The
    vector leaves' row is their persist and compute layout, so they stay put."""

    def enter(x: Array, weights: CIFnWeightPlacement, axes: Axes) -> Array:
        return gather_reduced_weights(
            x, source=weights.optimizer_state, destination=weights.compute_weights, axes=axes
        )

    rows = placement.weights

    def block(blk: CIFnBlock) -> CIFnBlock:
        return replace(
            blk,
            wq=enter(blk.wq, rows.attention, STACKED_ATTN_Q_AXES),
            wk=enter(blk.wk, rows.attention, STACKED_ATTN_KV_AXES),
            wv=enter(blk.wv, rows.attention, STACKED_ATTN_KV_AXES),
            wo=enter(blk.wo, rows.attention, STACKED_ATTN_OUT_AXES),
            w1=enter(blk.w1, rows.ffn, STACKED_FFN_IN_AXES),
            w2=enter(blk.w2, rows.ffn, STACKED_FFN_OUT_AXES),
            gate=None
            if blk.gate is None
            else (enter(blk.gate[0], rows.ffn, STACKED_FFN_IN_AXES), blk.gate[1]),
        )

    return replace(
        chunks,
        in_proj_w=enter(chunks.in_proj_w, rows.input, STACKED_INPUT_AXES),
        blocks=[block(blk) for blk in chunks.blocks],
        out_ws=tuple(enter(w, rows.output, STACKED_OUTPUT_AXES) for w in chunks.out_ws),
    )


class ChunkwiseTransformerBackbone(eqx.Module):
    """Per-chunk `CIFnTransformer`s stacked along a leading `n_chunks` axis, iterated by a
    `jax.lax.scan` over that axis (lowers as a loop so one chunk's FSDP weight gather is live
    at a time, not all `n_chunks` at once). Each chunk's input is its `chunk_input_taps`
    RMS-normed per tap and concatenated. Requires homogeneous chunks (equal total input width
    and an identical per-slot C tuple — same C-per-output-site ORDER) so the stack, including
    the per-slot output heads, is rectangular — asserted at init."""

    chunks: CIFnTransformer  # arrays stacked along leading n_chunks (+ stack_pad)
    inv_freq: Array  # shared across chunks (RoPE buffer); NOT mapped

    capture_keys: CaptureKeys = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)  # all sites, flat
    chunk_meta: tuple[_ChunkMeta, ...] = eqx.field(static=True)  # per-chunk routing
    placement: ChunkwiseCIFnPlacement = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    @property
    def census(self) -> StackCensus:
        """The real chunks and the zero chunks trailing them on every stored and compute
        `chunks` leaf; the forward slices the pad off before its scan."""
        return self.placement.chunk_census(len(self.chunk_meta))

    def placement_for_log(self) -> str:
        return self.placement.description_for_log()

    def matrix_parameters(self) -> tuple[CIFnMatrix, ...]:
        staging = self.placement.muon_staging()
        chunks = self.chunks
        matrices = [
            CIFnMatrix(chunks.in_proj_w, staging.input),
            *(CIFnMatrix(w, staging.output) for w in chunks.out_ws),
        ]
        for block in chunks.blocks:
            matrices.extend(
                (
                    CIFnMatrix(block.wq, staging.attention),
                    CIFnMatrix(block.wk, staging.attention),
                    CIFnMatrix(block.wv, staging.attention),
                    CIFnMatrix(block.wo, staging.attention),
                    CIFnMatrix(block.w1, staging.ffn),
                    CIFnMatrix(block.w2, staging.ffn),
                )
            )
            if block.gate is not None:
                matrices.append(CIFnMatrix(block.gate[0], staging.ffn))
        return tuple(matrices)

    def prepare(self) -> "ChunkwiseTransformerBackbone":
        """Compute-dtype residents, every matrix gathered to its compute row."""
        validate_chunk_leaves(self.chunks, self.census)
        compute = cast_floating(self, COMPUTE_DT)
        return replace(compute, chunks=self.placement.enter_compute_layout(compute.chunks))

    def shardings(self, mesh: Mesh) -> "ChunkwiseTransformerBackbone":
        """The stacked per-chunk transformer's persist layout (`CIFnTransformer.shardings`
        at the padded chunk extent); `inv_freq` (a 1-D RoPE buffer) replicates."""
        return eqx.tree_at(
            lambda f: (f.chunks, f.inv_freq),
            self,
            (
                self.placement.chunk_shardings(self.chunks, len(self.chunk_meta)),
                NamedSharding(mesh, P()),
            ),
        )

    def preactivations(
        self,
        taps: dict[str, Array],
        conditioning: object,
        components: ComponentActivations,
        *,
        sequence: SequenceLayout | None,
        remat: bool,
    ) -> SiteDict:
        del conditioning, components
        placement = self.placement
        layers = placement.layers()
        per_chunk_in = [
            normalized_tap_concatenation(
                [taps[key] for key in m.input_taps], placement.constrain_activation, self.eps
            )
            for m in self.chunk_meta
        ]
        if sequence is None:
            sequence = SequenceLayout.unsegmented_sequences_like(per_chunk_in[0][..., 0])
        stacked_in = jnp.stack(per_chunk_in, axis=0)  # [n_chunks, *leading, total_d_in]
        inv_freq = jax.lax.stop_gradient(self.inv_freq)
        # `lax.scan` (not `filter_vmap`) over the leading `n_chunks` axis so XLA lowers the
        # chunk iteration as a loop: one chunk's FSDP weight all-gather (∝ ΣC/tp) is live at
        # a time, then freed, instead of every chunk's gathered weights materialized at once
        # (the vmap unrolls, hoisting all n_chunks gathers into the flat entry computation).
        # Same math as the vmap — scan stacks per-iteration outputs exactly as vmap maps
        # them; results match up to fp32 reassociation (XLA picks different matmul layouts).
        chunks = real_chunks(self.chunks, self.census)
        chunk_arrays, chunk_static = eqx.partition(chunks, eqx.is_array)

        def run_chunk(
            _: None, scanned: tuple[CIFnTransformer, Array]
        ) -> tuple[None, tuple[Array, ...]]:
            chunk_array, chunk_input = scanned
            chunk = eqx.combine(chunk_array, chunk_static)
            return None, chunk(
                chunk_input,
                inv_freq,
                placement=layers,
                sequence=sequence,
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
        return preactivations


def init_chunk_transformer(
    arch: ChunkwiseTransformerCIFnArch,
    total_d_in: int,
    slot_cs: tuple[int, ...],
    key: PRNGKeyArray,
) -> CIFnTransformer:
    params = init_transformer_parameters(arch, total_d_in, slot_cs, key)
    return CIFnTransformer(
        in_proj_w=params.in_proj_w,
        in_proj_b=params.in_proj_b,
        blocks=params.blocks,
        out_ws=params.out_ws,
        out_bs=params.out_bs,
    )


def init_chunkwise_transformer_backbone(
    arch: ChunkwiseTransformerCIFnArch,
    sites: tuple[SiteSpec, ...],
    placement: ChunkwiseCIFnPlacement,
    key: PRNGKeyArray,
) -> ChunkwiseTransformerBackbone:
    """Validate the output partition + chunk homogeneity, then build STACKED chunk params
    padded to the placement's chunk-stack census.

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
        "in_proj + output heads, no attention — see ChunkwiseTransformerCIFnArch"
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
    stacked: CIFnTransformer = eqx.filter_vmap(
        lambda k: init_chunk_transformer(arch, arch.input_dim, slot_cs, k)
    )(chunk_keys)
    chunks = pad_chunk_stack(stacked, placement.chunk_census(len(arch.chunks)))

    return ChunkwiseTransformerBackbone(
        chunks=chunks,
        inv_freq=inv_freq,
        capture_keys=arch.capture_keys,
        output_names=tuple(name for ch in arch.chunks for name in ch.output_sites),
        chunk_meta=tuple(_ChunkMeta(ch.input_taps, ch.output_sites) for ch in arch.chunks),
        placement=placement,
        eps=CI_FN_RMS_EPS,
    )
