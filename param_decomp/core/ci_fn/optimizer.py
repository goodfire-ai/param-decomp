"""Adapt declared CI matrices and their staging to the Muon implementation."""

from math import prod

import equinox as eqx
import jax
import optax
from jax.sharding import NamedSharding
from jaxtyping import Array

from param_decomp.core.ci_fn.interface import CIFn
from param_decomp.core.linear_plan import spec_axes
from param_decomp.core.muon_stacked import NSWaypoints, StageInPlace


def _matrix_leaves[Conditioning](fn: CIFn[Conditioning]) -> tuple[Array, ...]:
    return tuple(matrix.value for matrix in fn.matrix_parameters())


def ci_fn_muon_dimension_numbers(params: optax.Params) -> optax.Params:
    """Select declared matrices, keeping stacked biases and scales elementwise."""
    dims = optax.contrib.MuonDimensionNumbers(reduction_axis=-2, output_axis=-1)
    marked = eqx.tree_at(_matrix_leaves, params, replace_fn=lambda _: dims)
    return jax.tree.map(
        lambda leaf: leaf if isinstance(leaf, optax.contrib.MuonDimensionNumbers) else None,
        marked,
        is_leaf=lambda leaf: isinstance(leaf, optax.contrib.MuonDimensionNumbers),
    )


def _assert_staging_tiles(matrix: Array, staging: NamedSharding) -> None:
    """Stacked Muon folds a matrix's independent leading axes into one batch and splits
    that batch at its staging; the split must tile it."""
    n_matrices = prod(matrix.shape[:-2])
    n_owners = prod(staging.mesh.shape[axis] for axis in spec_axes(staging.spec))
    assert n_matrices % n_owners == 0, (
        f"stacked Muon stages {n_matrices} independent CI matrices of shape "
        f"{matrix.shape} at {staging.spec}, which splits them {n_owners} ways and does "
        "not tile. NS stages the persist stacks as they rest (persist pads included). Use a "
        "mesh the stacks tile, an explicit table whose ns_compute rows they do tile, or an "
        "Adam-family optimizer."
    )


def assert_ci_fn_muon_staging_tiles[Conditioning](fn: CIFn[Conditioning]) -> None:
    """The stacked-Muon staging claim over every declared matrix; only a Muon CI
    optimizer stages, so only it makes this claim."""
    for matrix in fn.matrix_parameters():
        match matrix.staging:
            case NamedSharding() as staging:
                _assert_staging_tiles(matrix.value, staging)
            case StageInPlace():
                pass


def ci_fn_muon_waypoints() -> NSWaypoints:
    """Stage each declared matrix at its declared staging."""

    def waypoints(tree: optax.Params) -> optax.Params:
        assert isinstance(tree, CIFn), type(tree)
        assert_ci_fn_muon_staging_tiles(tree)
        stagings = tuple(matrix.staging for matrix in tree.matrix_parameters())
        return eqx.tree_at(_matrix_leaves, tree, stagings)

    return waypoints
