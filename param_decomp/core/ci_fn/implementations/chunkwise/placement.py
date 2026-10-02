"""The chunkwise transformer CI's placement: its chunk-stacked parameter axes and row
binding, its preset rows (the zero1 rows shared with block-selected), and the chunk-stack
census."""

from dataclasses import dataclass, replace

from jax.sharding import AbstractMesh, Mesh, NamedSharding

from param_decomp.core.axes import Axes
from param_decomp.core.ci_fn.implementations.transformer.layers import (
    CI_FN_ATTN_KV_AXES,
    CI_FN_ATTN_OUT_AXES,
    CI_FN_ATTN_Q_AXES,
    CI_FN_FFN_IN_AXES,
    CI_FN_FFN_OUT_AXES,
    CI_FN_INPUT_AXES,
    CI_FN_OUTPUT_AXES,
    CIFnTransformerStructure,
    MatrixAxes,
)
from param_decomp.core.ci_fn.implementations.transformer.placement import (
    ZERO1_DATA,
    PlacedTransformerCIFnRows,
    TransformerCIFnRows,
    bind_transformer_ci_fn_rows,
    resident_transformer_ci_fn_rows,
)
from param_decomp.core.configs import PlacementPresetName
from param_decomp.core.placement import (
    BATCH,
    COMPONENT_PARTITION,
    NS_STACK_SPLIT,
    REPLICATED,
    CIFnWeightPlacement,
    CIFnWeightTable,
    PlacedRule,
    StackCensus,
    placed_rule_for_log,
    resolve_stack_census,
)

CHUNK_STACK_AXES: Axes = ("stack",)
"""The chunk axis every stored chunk leaf leads with: it indexes independent matrices."""


def _stacked(axes: MatrixAxes) -> Axes:
    return (*CHUNK_STACK_AXES, *axes)


STACKED_ATTN_Q_AXES = _stacked(CI_FN_ATTN_Q_AXES)
STACKED_ATTN_KV_AXES = _stacked(CI_FN_ATTN_KV_AXES)
STACKED_ATTN_OUT_AXES = _stacked(CI_FN_ATTN_OUT_AXES)
STACKED_FFN_IN_AXES = _stacked(CI_FN_FFN_IN_AXES)
STACKED_FFN_OUT_AXES = _stacked(CI_FN_FFN_OUT_AXES)
STACKED_INPUT_AXES = _stacked(CI_FN_INPUT_AXES)
STACKED_OUTPUT_AXES = _stacked(CI_FN_OUTPUT_AXES)


def bind_chunkwise_rows(
    rows: TransformerCIFnRows, mesh: Mesh | AbstractMesh
) -> PlacedTransformerCIFnRows:
    """Every chunk leaf, bias and norm scale included, leads with the chunk stack."""
    return bind_transformer_ci_fn_rows(
        rows, mesh, CIFnTransformerStructure.uniform(CHUNK_STACK_AXES), CHUNK_STACK_AXES
    )


def chunk_persist_rows(rows: PlacedTransformerCIFnRows) -> tuple[PlacedRule, ...]:
    """Every row a stacked chunk leaf rests at."""
    weights = rows.weights
    return (
        weights.attention.optimizer_state,
        weights.ffn.optimizer_state,
        weights.input.optimizer_state,
        weights.output.optimizer_state,
        rows.vectors,
    )


def chunk_placement_for_log(rows: tuple[PlacedRule, ...], chunks: StackCensus) -> str:
    """A chunkwise CI fn's placement as the startup audit prints it."""
    return "\n".join(
        (
            "ci_fn placement:",
            f"  chunk stack: {chunks.stack_len} chunks + {chunks.stack_pad} pad",
            *(placed_rule_for_log(row) for row in rows),
        )
    )


ZERO1_CHUNKWISE_ROWS = TransformerCIFnRows(
    weights=CIFnTransformerStructure(
        input=CIFnWeightTable(
            optimizer_state={"input": ("tp",), "d_model": ZERO1_DATA},
            compute_weights={"input": ("tp",), "d_model": ("fsdp",)},
            operands={"input": ("tp",)},
            ns_compute=NS_STACK_SPLIT,
        ),
        attention=CIFnWeightTable(
            optimizer_state={"d_model": ZERO1_DATA, "q_head": ("tp",), "kv_head": ("tp",)},
            compute_weights={"d_model": ("fsdp",), "q_head": ("tp",), "kv_head": ("tp",)},
            operands={"q_head": ("tp",), "kv_head": ("tp",)},
            ns_compute=NS_STACK_SPLIT,
        ),
        ffn=CIFnWeightTable(
            optimizer_state={"ffn_hidden": ("tp", "fsdp", "replicate")},
            compute_weights={"ffn_hidden": ("tp", "fsdp")},
            operands={"ffn_hidden": ("tp",)},
            ns_compute=NS_STACK_SPLIT,
        ),
        output=CIFnWeightTable(
            optimizer_state={"d_model": ZERO1_DATA, "C": ("tp",)},
            compute_weights={"d_model": ("fsdp",), **COMPONENT_PARTITION},
            operands=COMPONENT_PARTITION,
            ns_compute=NS_STACK_SPLIT,
        ),
    ),
    vectors={"ffn_hidden": ("tp",), **COMPONENT_PARTITION},
    activations={
        "batch": BATCH,
        "input": ("tp",),
        "q_head": ("tp",),
        "kv_head": ("tp",),
        "ffn_hidden": ("tp",),
        **COMPONENT_PARTITION,
    },
)


# Owner stack-cuts every master like the V/U masters (stack ÷replicate, d_model ÷fsdp),
# so a chunk count that does not tile `replicate` pads the chunk stack.
_OWNER_ROWS = replace(
    ZERO1_CHUNKWISE_ROWS,
    weights=CIFnTransformerStructure(
        input=CIFnWeightTable(
            optimizer_state={"stack": ("replicate",), "input": ("tp",), "d_model": ("fsdp",)},
            compute_weights={"input": ("tp",), "d_model": ("fsdp",)},
            operands={"input": ("tp",)},
            ns_compute=NS_STACK_SPLIT,
        ),
        attention=CIFnWeightTable(
            optimizer_state={
                "stack": ("replicate",),
                "d_model": ("fsdp",),
                "q_head": ("tp",),
                "kv_head": ("tp",),
            },
            compute_weights={"d_model": ("fsdp",), "q_head": ("tp",), "kv_head": ("tp",)},
            operands={"q_head": ("tp",), "kv_head": ("tp",)},
            ns_compute=NS_STACK_SPLIT,
        ),
        ffn=CIFnWeightTable(
            optimizer_state={"stack": ("replicate",), "ffn_hidden": ("tp", "fsdp")},
            compute_weights={"ffn_hidden": ("tp", "fsdp")},
            operands={"ffn_hidden": ("tp",)},
            ns_compute=NS_STACK_SPLIT,
        ),
        output=CIFnWeightTable(
            optimizer_state={"stack": ("replicate",), "d_model": ("fsdp",), "C": ("tp",)},
            compute_weights={"d_model": ("fsdp",), **COMPONENT_PARTITION},
            operands=COMPONENT_PARTITION,
            ns_compute=NS_STACK_SPLIT,
        ),
    ),
)
_DDP_WEIGHTS = CIFnWeightTable(
    optimizer_state=REPLICATED,
    compute_weights=REPLICATED,
    operands=REPLICATED,
    ns_compute=NS_STACK_SPLIT,
)
_DDP_ROWS = TransformerCIFnRows(
    weights=CIFnTransformerStructure.uniform(_DDP_WEIGHTS),
    vectors=REPLICATED,
    activations={"batch": BATCH},
)


def preset_rows(name: PlacementPresetName) -> TransformerCIFnRows:
    """The chunkwise transformer's rows under the run's placement preset."""
    match name:
        case "owner":
            return _OWNER_ROWS
        case "zero1":
            return ZERO1_CHUNKWISE_ROWS
        case "zero1-replicated-resident" | "zero1-replicated-resident-moe":
            return resident_transformer_ci_fn_rows(ZERO1_CHUNKWISE_ROWS)
        case "owner-replicated-resident":
            return resident_transformer_ci_fn_rows(_OWNER_ROWS)
        case "ddp":
            return _DDP_ROWS
        case "owner-replicated-resident-moe" | "zero1-replicated-resident-moe-replicated-ns":
            raise NotImplementedError(
                f"the chunkwise transformer CI has no rows for placement preset {name!r}"
            )


@dataclass(frozen=True)
class CIFnMatrixBatchShape:
    """Matrix axes follow zero or more independent batch axes; `weight` holds the rows
    the matrix rests, computes, and stages at."""

    weight: CIFnWeightPlacement
    axes: Axes
    shape: tuple[int, ...]


@dataclass(frozen=True)
class CIFnVectorBatchShape:
    """The vector axis follows zero or more independent batch axes."""

    axes: Axes
    shape: tuple[int, ...]


CIFnTensorShape = CIFnMatrixBatchShape | CIFnVectorBatchShape


@dataclass(frozen=True)
class AttentionHeadCounts:
    query_heads: int
    kv_heads: int


@dataclass(frozen=True)
class ChunkedCIFnPlacementRequirements:
    """Logical tensor extents whose leading stack permits inert trailing padding.
    `persist_rows` are every row the stacked leaves rest at; each stack cut must tile the
    padded stack."""

    n_chunks: int
    tensors: tuple[CIFnTensorShape, ...]
    heads: AttentionHeadCounts
    persist_rows: tuple[PlacedRule, ...]


def resolve_chunk_census(
    requirements: ChunkedCIFnPlacementRequirements, rows: PlacedTransformerCIFnRows
) -> StackCensus:
    """The chunk-stack census every persist row tiles, with each stacked tensor validated
    against its rows at the padded extent."""
    rows.activations.validate_shape(("q_head",), (requirements.heads.query_heads,))
    rows.activations.validate_shape(("kv_head",), (requirements.heads.kv_heads,))
    census = resolve_stack_census(
        requirements.n_chunks, requirements.persist_rows, CHUNK_STACK_AXES
    )
    for tensor in requirements.tensors:
        shape = (census.padded_stack_len, *tensor.shape[1:])
        match tensor:
            case CIFnMatrixBatchShape(weight=weight, axes=axes):
                weight.optimizer_state.validate_shape(axes, shape)
                weight.compute_weights.validate_shape(axes, shape)
            case CIFnVectorBatchShape(axes=axes):
                rows.vectors.validate_shape(axes, shape)
    return census


def vector_sharding(row: PlacedRule, axes: Axes, shape: tuple[int, ...]) -> NamedSharding:
    row.validate_shape(axes, shape)
    return row.sharding_for(axes)
