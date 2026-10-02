"""Placement for the global transformer CI: its leaves' independent axes, row binding,
and the rows each placement preset names."""

from jax.sharding import AbstractMesh, Mesh

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
)
from param_decomp.core.ci_fn.implementations.transformer.placement import (
    ZERO1_DATA,
    PlacedTransformerCIFnRows,
    TransformerCIFnRows,
    bind_transformer_ci_fn_rows,
)
from param_decomp.core.configs import PlacementPresetName
from param_decomp.core.placement import (
    BATCH,
    COMPONENT_PARTITION,
    REPLICATED,
    CIFnWeightTable,
)

DEPTH_AXES: Axes = ("depth",)
"""The axis every block leaf leads with: one entry per transformer block, scanned."""
SITE_AXES: Axes = ("site",)
"""The axis one semantic group's output heads stack along, in the group's site order."""

GLOBAL_INDEPENDENT_AXES: CIFnTransformerStructure[Axes] = CIFnTransformerStructure(
    input=(), attention=DEPTH_AXES, ffn=DEPTH_AXES, output=SITE_AXES
)
"""The independent axes each part's matrices lead with: blocks along `depth`, heads
along `site`, and none for the one input projection."""

ATTN_Q_AXES: Axes = (*DEPTH_AXES, *CI_FN_ATTN_Q_AXES)
ATTN_KV_AXES: Axes = (*DEPTH_AXES, *CI_FN_ATTN_KV_AXES)
ATTN_OUT_AXES: Axes = (*DEPTH_AXES, *CI_FN_ATTN_OUT_AXES)
FFN_IN_AXES: Axes = (*DEPTH_AXES, *CI_FN_FFN_IN_AXES)
FFN_OUT_AXES: Axes = (*DEPTH_AXES, *CI_FN_FFN_OUT_AXES)
INPUT_AXES: Axes = CI_FN_INPUT_AXES
OUTPUT_AXES: Axes = (*SITE_AXES, *CI_FN_OUTPUT_AXES)


def bind_global_rows(
    rows: TransformerCIFnRows, mesh: Mesh | AbstractMesh
) -> PlacedTransformerCIFnRows:
    """The input bias leads with no axis, every block bias and norm scale with `depth`,
    and every head stack's biases with `site`."""
    return bind_transformer_ci_fn_rows(
        rows, mesh, GLOBAL_INDEPENDENT_AXES, (*DEPTH_AXES, *SITE_AXES)
    )


# `replicate` owns whole blocks and whole output heads, and Newton-Schulz splits them
# the same way. The input projection has no independent axis, so its masters rest
# intra-matrix and Newton-Schulz stages it whole on every device.
_OWNER_ROWS = TransformerCIFnRows(
    weights=CIFnTransformerStructure(
        input=CIFnWeightTable(
            optimizer_state={"input": ("tp",), "d_model": ZERO1_DATA},
            compute_weights={"input": ("tp",), "d_model": ("fsdp",)},
            operands={"input": ("tp",)},
            ns_compute=REPLICATED,
        ),
        attention=CIFnWeightTable(
            optimizer_state={
                "depth": ("replicate",),
                "d_model": ("fsdp",),
                "q_head": ("tp",),
                "kv_head": ("tp",),
            },
            compute_weights={"d_model": ("fsdp",), "q_head": ("tp",), "kv_head": ("tp",)},
            operands={"q_head": ("tp",), "kv_head": ("tp",)},
            ns_compute={"depth": ("replicate",)},
        ),
        ffn=CIFnWeightTable(
            optimizer_state={"depth": ("replicate",), "ffn_hidden": ("tp", "fsdp")},
            compute_weights={"ffn_hidden": ("tp", "fsdp")},
            operands={"ffn_hidden": ("tp",)},
            ns_compute={"depth": ("replicate",)},
        ),
        output=CIFnWeightTable(
            optimizer_state={"site": ("replicate",), "d_model": ("fsdp",), "C": ("tp",)},
            compute_weights={"d_model": ("fsdp",), **COMPONENT_PARTITION},
            operands=COMPONENT_PARTITION,
            ns_compute={"site": ("replicate",)},
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


def preset_rows(name: PlacementPresetName) -> TransformerCIFnRows:
    """The global transformer's rows under the run's placement preset."""
    match name:
        case "owner":
            return _OWNER_ROWS
        case (
            "zero1"
            | "zero1-replicated-resident"
            | "owner-replicated-resident"
            | "ddp"
            | "zero1-replicated-resident-moe"
            | "owner-replicated-resident-moe"
            | "zero1-replicated-resident-moe-replicated-ns"
        ):
            raise NotImplementedError(
                f"the global transformer CI has no rows for placement preset {name!r}"
            )
