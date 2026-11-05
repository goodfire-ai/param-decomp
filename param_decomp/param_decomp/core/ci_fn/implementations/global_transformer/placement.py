"""Placement for the global transformer CI: its leaves' independent axes, row binding,
the rows each placement preset names, and the persist-stack census those rows resolve."""

from dataclasses import dataclass

from jax.sharding import AbstractMesh, Mesh
from jaxtyping import Array

from param_decomp.core.axes import Axes
from param_decomp.core.ci_fn.implementations.transformer.layers import (
    CI_FN_ATTN_KV_AXES,
    CI_FN_ATTN_OUT_AXES,
    CI_FN_ATTN_Q_AXES,
    CI_FN_FFN_IN_AXES,
    CI_FN_FFN_OUT_AXES,
    CI_FN_INPUT_AXES,
    CI_FN_OUTPUT_AXES,
    CIFnTransformerMuonStaging,
    CIFnTransformerPlacement,
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
    StackCensus,
    resolve_stack_census,
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


@dataclass(frozen=True)
class GlobalCIStacks:
    """The global CI's persist stacks: the blocks along `depth` and each semantic group's
    output heads along `site`, in head order. Each census holds the real length and the
    trailing all-zero slots that make the persist stack tile the rows that cut it
    (`StackCensus`). Pads live only in the fp32 masters and their optimizer moments; the
    compute entry strips them, so the forward never sees one."""

    depth: StackCensus
    sites: tuple[StackCensus, ...]

    @staticmethod
    def unpadded(n_blocks: int, head_lengths: tuple[int, ...]) -> "GlobalCIStacks":
        return GlobalCIStacks(
            depth=StackCensus(stack_len=n_blocks, stack_pad=0),
            sites=tuple(StackCensus(stack_len=n, stack_pad=0) for n in head_lengths),
        )

    @property
    def real(self) -> "GlobalCIStacks":
        """The same stacks without their pads: what the compute residents hold."""
        return GlobalCIStacks.unpadded(
            self.depth.stack_len, tuple(site.stack_len for site in self.sites)
        )

    def description_for_log(self) -> str:
        sites = ", ".join(f"{site.stack_len} + {site.stack_pad}" for site in self.sites)
        return (
            f"  persist stacks: depth {self.depth.stack_len} + {self.depth.stack_pad} pad; "
            f"site stacks (real + pad) {sites}"
        )


def resolve_global_stacks(
    rows: PlacedTransformerCIFnRows, n_blocks: int, head_lengths: tuple[int, ...]
) -> GlobalCIStacks:
    """Each persist stack padded to the next multiple of its optimizer rows' cut: blocks
    by the attention and FFN rows' `depth` split, heads by the output row's `site` split."""
    weights = rows.weights
    return GlobalCIStacks(
        depth=resolve_stack_census(
            n_blocks, (weights.attention.optimizer_state, weights.ffn.optimizer_state), DEPTH_AXES
        ),
        sites=tuple(
            resolve_stack_census(n, (weights.output.optimizer_state,), SITE_AXES)
            for n in head_lengths
        ),
    )


@dataclass(frozen=True)
class PlacedGlobalCIFn:
    """The global CI's rows bound to the run's mesh, and the persist-stack census they
    resolve for this CI's block and head counts."""

    rows: PlacedTransformerCIFnRows
    stacks: GlobalCIStacks

    def layers(self) -> CIFnTransformerPlacement:
        return self.rows.layers()

    def muon_staging(self) -> CIFnTransformerMuonStaging:
        return self.rows.muon_staging()

    def constrain_activation(self, x: Array) -> Array:
        return self.rows.constrain_activation(x)

    def description_for_log(self) -> str:
        return "\n".join((self.rows.description_for_log(), self.stacks.description_for_log()))


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
