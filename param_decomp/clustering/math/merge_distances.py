import multiprocessing
import os
import sys
from collections.abc import Callable, Sequence

import numpy as np

from param_decomp.clustering.math.matching_dist import matching_dist, matching_dist_vec
from param_decomp.clustering.math.perm_invariant_hamming import (
    perm_invariant_hamming_matrix,
)
from param_decomp.clustering.types import (
    DistancesArray,
    DistancesMethod,
    MergesArray,
)

_WORKER_CONTEXT = multiprocessing.get_context("forkserver")
"""NOT the platform default (`fork` on Linux). Every caller reaches here with JAX
imported, and JAX runs ~100 threads: `fork(2)` hands the child a copy of a mutex whose
owning thread does not exist on the other side, so the child blocks in futex forever and
the parent blocks in `wait(2)`. `forkserver` forks from a thread-free server process."""


def _run_parallel[T, R](func: Callable[[T], R], items: Sequence[T]) -> list[R]:
    assert items, "nothing to distribute"
    match sys.platform:
        case "linux":
            worker_count = len(os.sched_getaffinity(0))
        case "darwin":
            worker_count = os.cpu_count()
            assert worker_count is not None, "macOS did not report a CPU count"
        case platform:
            raise RuntimeError(f"unsupported multiprocessing platform: {platform}")
    with _WORKER_CONTEXT.Pool(min(len(items), worker_count)) as pool:
        return pool.map(func, items)


def compute_distances(
    normalized_merge_array: MergesArray,
    method: DistancesMethod,
) -> DistancesArray:
    match method:
        case "perm_invariant_hamming":
            pairwise_distances = perm_invariant_hamming_matrix
        case "matching_dist":
            pairwise_distances = matching_dist
        case "matching_dist_vec":
            pairwise_distances = matching_dist_vec
    n_iters = normalized_merge_array.shape[1]
    labels_per_iteration = [normalized_merge_array[:, i, :] for i in range(n_iters)]
    return np.stack(_run_parallel(pairwise_distances, labels_per_iteration), axis=0)
