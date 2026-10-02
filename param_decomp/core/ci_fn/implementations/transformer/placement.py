"""Placement shared by the transformer CI architectures: one lifecycle table per part of
the layers plus the vector and activation rules, bound against the independent axes a
variant stacks its matrices on."""

from dataclasses import dataclass

import jax
from jax.sharding import AbstractMesh, Mesh
from jaxtyping import Array

from param_decomp.core.axes import Axes, MeshAxis, SemanticAxis
from param_decomp.core.ci_fn.implementations.transformer.layers import (
    CI_FN_TRANSFORMER_MATRIX_AXES,
    CIFnTransformerMuonStaging,
    CIFnTransformerPlacement,
    CIFnTransformerStructure,
    MatrixAxes,
    PlacedProjection,
)
from param_decomp.core.components import activation_axes
from param_decomp.core.placement import (
    CIFnWeightPlacement,
    CIFnWeightTable,
    PlacedRule,
    Rule,
    bind_ci_fn_row,
    bind_ci_fn_weight_rows,
    bind_scanned_ci_fn_row,
    placed_rule_for_log,
    reachable_placed_rules,
    resident_ci_fn_weight_table,
    resident_rule,
)

# The activation row also covers the attention head-split views.
_ACTIVATION_AXES: frozenset[SemanticAxis] = frozenset(
    {"batch", "position", "feature", "input", "q_head", "kv_head", "ffn_hidden", "C", "d_model"}
)
_VECTOR_FEATURE_AXES: frozenset[SemanticAxis] = frozenset({"ffn_hidden", "d_model", "C"})

# ÷N master rows keep `replicate` MINOR on whichever dim carries it, so each compute
# shard is the contiguous concatenation of its own replicate group's optimizer shards:
# optimizer-state→compute-weights is then an all-gather over `replicate` alone, and its
# transpose a reduce-scatter. Nested-axis order is semantics (PLACEMENT_DESIGN.md
# invariant 5).
ZERO1_DATA: tuple[MeshAxis, ...] = ("fsdp", "replicate")


@dataclass(frozen=True)
class TransformerCIFnRows:
    """A transformer CI's placement rules as a preset declares them, before a mesh is
    known: one lifecycle table per part of the layers, the rule for every bias and norm scale,
    and the activation rule."""

    weights: CIFnTransformerStructure[CIFnWeightTable]
    vectors: Rule
    activations: Rule


@dataclass(frozen=True)
class PlacedTransformerCIFnRows:
    """`TransformerCIFnRows` bound to the run's mesh (`bind_transformer_ci_fn_rows`)."""

    weights: CIFnTransformerStructure[CIFnWeightPlacement]
    vectors: PlacedRule
    activations: PlacedRule

    def layers(self) -> CIFnTransformerPlacement:
        """The rows as the shared transformer layers consume them."""
        return self.weights.map(lambda weights: PlacedProjection(weights, self.activations))

    def muon_staging(self) -> CIFnTransformerMuonStaging:
        return self.weights.map(CIFnWeightPlacement.muon_staging)

    def constrain_activation(self, x: Array) -> Array:
        axes = activation_axes(x.ndim, "feature")
        self.activations.validate_shape(axes, x.shape)
        return jax.sharding.reshard(x, self.activations.sharding_for(axes))

    def description_for_log(self) -> str:
        """The rows as the startup audit prints them."""
        return "\n".join(
            (
                "ci_fn placement:",
                *(placed_rule_for_log(row) for row in reachable_placed_rules(self)),
            )
        )


def bind_transformer_ci_fn_rows(
    rows: TransformerCIFnRows,
    mesh: Mesh | AbstractMesh,
    independent_axes: CIFnTransformerStructure[Axes],
    vector_independent_axes: Axes,
) -> PlacedTransformerCIFnRows:
    """`independent_axes` lead each part's matrices; `vector_independent_axes` lead
    every bias and norm scale."""

    def bind_part(
        name: str, table: CIFnWeightTable, leading: Axes, matrices: tuple[MatrixAxes, ...]
    ) -> CIFnWeightPlacement:
        leaf_axes = tuple((*leading, *matrix) for matrix in matrices)
        return bind_ci_fn_weight_rows(f"ci_fn/{name}", table, mesh, leaf_axes, leading)

    tables, matrices = rows.weights, CI_FN_TRANSFORMER_MATRIX_AXES
    return PlacedTransformerCIFnRows(
        weights=CIFnTransformerStructure(
            input=bind_part("input", tables.input, independent_axes.input, matrices.input),
            attention=bind_part(
                "attention", tables.attention, independent_axes.attention, matrices.attention
            ),
            ffn=bind_part("ffn", tables.ffn, independent_axes.ffn, matrices.ffn),
            output=bind_part("output", tables.output, independent_axes.output, matrices.output),
        ),
        vectors=bind_scanned_ci_fn_row(
            "ci_fn/vectors",
            rows.vectors,
            mesh,
            _VECTOR_FEATURE_AXES | set(vector_independent_axes),
            vector_independent_axes,
        ),
        activations=bind_ci_fn_row("ci_fn/activations", rows.activations, mesh, _ACTIVATION_AXES),
    )


def resident_transformer_ci_fn_rows(rows: TransformerCIFnRows) -> TransformerCIFnRows:
    """The rows re-spelled for the `(data, tp)` mesh with the working copy resident
    whole (`placement.resident_rule`)."""
    return TransformerCIFnRows(
        weights=rows.weights.map(resident_ci_fn_weight_table),
        vectors=resident_rule(rows.vectors),
        activations=resident_rule(rows.activations),
    )
