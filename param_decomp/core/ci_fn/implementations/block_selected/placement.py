"""The block-selected transformer's placement: its expert-family axes, the preset rows
(the chunkwise rows plus the expert families), and their binding to the mesh."""

from dataclasses import dataclass, replace

from jax.sharding import AbstractMesh, Mesh, NamedSharding

from param_decomp.core.axes import Axes
from param_decomp.core.ci_fn.implementations.chunkwise.placement import (
    CHUNK_STACK_AXES,
    ZERO1_CHUNKWISE_ROWS,
    bind_chunkwise_rows,
    chunk_persist_rows,
)
from param_decomp.core.ci_fn.implementations.transformer.layers import CIFnTransformerMuonStaging
from param_decomp.core.ci_fn.implementations.transformer.placement import (
    PlacedTransformerCIFnRows,
    TransformerCIFnRows,
    resident_transformer_ci_fn_rows,
)
from param_decomp.core.configs import PlacementPresetName
from param_decomp.core.muon_stacked import StageInPlace
from param_decomp.core.placement import (
    EXPERT_PARTITION,
    REPLICATED,
    CIFnWeightPlacement,
    CIFnWeightTable,
    PlacedRule,
    bind_ci_fn_weight_rows,
)

CI_FN_SELECTED_FFN_IN_AXES: Axes = ("stack", "expert", "d_model", "ffn_hidden")
CI_FN_SELECTED_FFN_OUT_AXES: Axes = ("stack", "expert", "ffn_hidden", "d_model")
CI_FN_BLOCKED_HEAD_AXES: Axes = ("stack", "expert", "ffn_hidden", "C_block")


@dataclass(frozen=True)
class BlockSelectedCIFnRows:
    """The block-selected transformer's placement rules as a preset declares them: the
    chunkwise rows, plus the per-transformer-block concat-wide selected FFN banks
    (`expert_ffn`; leaves `[stack, expert, d_model, ffn_hidden]` /
    `[stack, expert, ffn_hidden, d_model]`) and the per-slot fused blocked heads
    (`expert_head`; `[stack, expert, ffn_hidden, C_block]`). The block axis (spelled
    `expert`) describes these internal expert operands independently of public selected
    values and per-component statistics."""

    chunkwise: TransformerCIFnRows
    expert_ffn: CIFnWeightTable
    expert_head: CIFnWeightTable


# The expert families mirror the V/U blocks: residents/operands rest whole per block
# shard, CO-LOCATED with the target's frozen blocks and the V/U blocks (`expert: tp` —
# the banks' selected compute and the fused heads ride `ExpertShardedJobs` on the same
# axis with zero weight movement); masters ÷(tp·data) in the zero1 spirit, so entry is a
# pure all-gather over `data` with `expert` staying put. NS staging folds the block axis
# into the canonical stack ({stack: data}, whole matrices per device). Both `-moe`
# presets carry these same rows over the zero1 resident chunkwise rows: owner-flavored
# stack-cut masters would pad the chunk stack to tile `data` (n_chunks = 10 pads to 16
# at data=8), and the intra-matrix cut costs owner nothing it claims — CI weights have
# no faithfulness row, and the owner flavor's ruling is the components' muon pairing.
_RESIDENT_ROWS = BlockSelectedCIFnRows(
    chunkwise=resident_transformer_ci_fn_rows(ZERO1_CHUNKWISE_ROWS),
    expert_ffn=CIFnWeightTable(
        optimizer_state={"expert": ("tp",), "ffn_hidden": ("data",)},
        compute_weights=EXPERT_PARTITION,
        operands=EXPERT_PARTITION,
        ns_compute={"stack": ("data",)},
    ),
    expert_head=CIFnWeightTable(
        optimizer_state={"expert": ("tp",), "C_block": ("data",)},
        compute_weights=EXPERT_PARTITION,
        operands=EXPERT_PARTITION,
        ns_compute={"stack": ("data",)},
    ),
)


def _replicated_ns(table: CIFnWeightTable) -> CIFnWeightTable:
    return replace(table, ns_compute=REPLICATED)


_RESIDENT_REPLICATED_NS_ROWS = BlockSelectedCIFnRows(
    chunkwise=replace(
        _RESIDENT_ROWS.chunkwise, weights=_RESIDENT_ROWS.chunkwise.weights.map(_replicated_ns)
    ),
    expert_ffn=_replicated_ns(_RESIDENT_ROWS.expert_ffn),
    expert_head=_replicated_ns(_RESIDENT_ROWS.expert_head),
)


def preset_rows(name: PlacementPresetName) -> BlockSelectedCIFnRows:
    """The block-selected transformer's rows under the run's placement preset."""
    match name:
        case "zero1-replicated-resident-moe" | "owner-replicated-resident-moe":
            return _RESIDENT_ROWS
        case "zero1-replicated-resident-moe-replicated-ns":
            return _RESIDENT_REPLICATED_NS_ROWS
        case "owner" | "zero1" | "zero1-replicated-resident" | "owner-replicated-resident" | "ddp":
            raise NotImplementedError(
                f"the block-selected transformer CI has no rows for placement preset {name!r}"
            )


@dataclass(frozen=True)
class BlockSelectedMuonStaging:
    """Where stacked Muon stages the chunkwise layers' parts and the expert families."""

    chunkwise: CIFnTransformerMuonStaging
    expert_ffn: NamedSharding | StageInPlace
    expert_head: NamedSharding | StageInPlace


@dataclass(frozen=True)
class PlacedBlockSelectedCIFnRows:
    """`BlockSelectedCIFnRows` bound to the run's mesh (`bind_block_selected_rows`)."""

    chunkwise: PlacedTransformerCIFnRows
    expert_ffn: CIFnWeightPlacement
    expert_head: CIFnWeightPlacement

    @property
    def persist_rows(self) -> tuple[PlacedRule, ...]:
        """Every row a stacked chunk leaf rests at."""
        return (
            *chunk_persist_rows(self.chunkwise),
            self.expert_ffn.optimizer_state,
            self.expert_head.optimizer_state,
        )

    def muon_staging(self) -> BlockSelectedMuonStaging:
        return BlockSelectedMuonStaging(
            chunkwise=self.chunkwise.muon_staging(),
            expert_ffn=self.expert_ffn.muon_staging(),
            expert_head=self.expert_head.muon_staging(),
        )


def bind_block_selected_rows(
    rows: BlockSelectedCIFnRows, mesh: Mesh | AbstractMesh
) -> PlacedBlockSelectedCIFnRows:
    return PlacedBlockSelectedCIFnRows(
        chunkwise=bind_chunkwise_rows(rows.chunkwise, mesh),
        expert_ffn=bind_ci_fn_weight_rows(
            "ci_fn/expert_ffn",
            rows.expert_ffn,
            mesh,
            (CI_FN_SELECTED_FFN_IN_AXES, CI_FN_SELECTED_FFN_OUT_AXES),
            CHUNK_STACK_AXES,
        ),
        expert_head=bind_ci_fn_weight_rows(
            "ci_fn/expert_head",
            rows.expert_head,
            mesh,
            (CI_FN_BLOCKED_HEAD_AXES,),
            CHUNK_STACK_AXES,
        ),
    )
