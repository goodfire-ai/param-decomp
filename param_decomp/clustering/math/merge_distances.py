from collections.abc import Callable, Iterable
from multiprocessing import Pool
from typing import TypeVar

import numpy as np
from jaxtyping import Float, Int

from param_decomp.clustering.consts import (
    DistancesArray,
    DistancesMethod,
    MergesArray,
    MergesAtIterArray,
)
from param_decomp.clustering.math.matching_dist import matching_dist_np, matching_dist_vec_np
from param_decomp.clustering.math.perm_invariant_hamming import perm_invariant_hamming_matrix

_T = TypeVar("_T")
_R = TypeVar("_R")


def _run_parallel(func: Callable[[_T], _R], iterable: Iterable[_T]) -> list[_R]:
    """Run a function in parallel over an iterable using multiprocessing."""
    items = list(iterable)
    with Pool() as pool:
        return pool.map(func, items)


DISTANCES_METHODS: dict[DistancesMethod, Callable[[MergesAtIterArray], DistancesArray]] = {
    "perm_invariant_hamming": perm_invariant_hamming_matrix,
    "matching_dist": matching_dist_np,
}

# pyright: reportUnnecessaryComparison=false, reportUnreachable=false


def compute_distances(
    normalized_merge_array: MergesArray,
    method: DistancesMethod = "perm_invariant_hamming",
) -> DistancesArray:
    n_iters: int = normalized_merge_array.shape[1]
    merges_array_list: list[Int[np.ndarray, "n_ens n_components"]]
    distances_list: list[Float[np.ndarray, "n_ens n_ens"]]
    match method:
        case "perm_invariant_hamming":
            merges_array_list = [normalized_merge_array[:, i, :] for i in range(n_iters)]
            distances_list = _run_parallel(perm_invariant_hamming_matrix, merges_array_list)
            return np.stack(distances_list, axis=0)
        case "matching_dist":
            merges_array_list = [normalized_merge_array[:, i, :] for i in range(n_iters)]
            distances_list = _run_parallel(matching_dist_np, merges_array_list)
            return np.stack(distances_list, axis=0)
        case "matching_dist_vec":
            merges_array_list = [normalized_merge_array[:, i, :] for i in range(n_iters)]
            distances_list = _run_parallel(matching_dist_vec_np, merges_array_list)
            return np.stack(distances_list, axis=0)
        case _:
            raise ValueError(f"Unknown distance method: {method}")
