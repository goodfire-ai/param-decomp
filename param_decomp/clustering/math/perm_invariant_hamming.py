import warnings

import numpy as np
from jaxtyping import Float, Int
from scipy.optimize import linear_sum_assignment


def perm_invariant_hamming_matrix(
    X: Int[np.ndarray, "n_ens n_components"],
) -> Float[np.ndarray, "n_ens n_ens"]:
    """Compute all pairwise permutation-invariant Hamming distances.

    The strictly lower-triangular entries are filled with distances;
    the diagonal and upper triangle are left as `np.nan`.

    # Parameters:
     - `X : Int[np.ndarray, "n_ens n_components"]`
       Matrix where each of the `n_ens` rows is a label vector of length `n_components`.

    # Returns:
     - `Float[np.ndarray, "n_ens n_ens"]`
       Distance matrix `D` with `D[i, j]` defined only for `i > j`;
       all other positions are `np.nan`.

    # Usage:
    ```python
    >>> X = np.array([[0, 0, 1],
    ...               [1, 1, 0],
    ...               [0, 1, 0]])
    >>> D = perm_invariant_hamming_matrix(X)
    >>> D
    array([[nan, nan, nan],
           [ 0., nan, nan],
           [ 2., 2., nan]])
    ```
    """
    n_ens: int
    n_components: int
    n_ens, n_components = X.shape
    D: Float[np.ndarray, "n_ens n_ens"] = np.full((n_ens, n_ens), np.nan, dtype=float)

    # Pre-compute max label in each row once.
    row_max: Int[np.ndarray, " n_ens"] = X.max(axis=1)

    for i in range(1, n_ens):
        a: Int[np.ndarray, " n_components"] = X[i]
        for j in range(i):
            b: Int[np.ndarray, " n_components"] = X[j]

            k_lbls: int = int(max(row_max[i], row_max[j]) + 1)

            # Handle case where all labels are -1 (no valid clustering)
            if k_lbls <= 0:
                warnings.warn(
                    f"All labels are -1 at rows {i} and {j}. Setting distance to 0.",
                    UserWarning,
                    stacklevel=2,
                )
                D[i, j] = 0.0
                continue

            C: Int[np.ndarray, "k_lbls k_lbls"] = np.zeros((k_lbls, k_lbls), dtype=int)
            np.add.at(C, (a, b), 1)

            row_ind, col_ind = linear_sum_assignment(-C)
            matches: int = int(C[row_ind, col_ind].sum())

            D[i, j] = n_components - matches  # int is fine; array is float because of NaN

    return D
