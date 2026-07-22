import numpy as np
from jaxtyping import Bool, Float, Int


def matching_dist(
    X: Int[np.ndarray, "s n"],
) -> Float[np.ndarray, "s s"]:
    s_ensemble, _n_components = X.shape
    matches: Bool[np.ndarray, "s n n"] = X[:, :, None] == X[:, None, :]

    dists: Float[np.ndarray, "s s"] = np.full((s_ensemble, s_ensemble), np.nan, dtype=np.float32)

    for i in range(s_ensemble):
        for j in range(i + 1, s_ensemble):
            dist_mat = matches[i].astype(np.float32) - matches[j].astype(np.float32)
            dists[i, j] = np.abs(np.tril(dist_mat, k=-1)).sum()

    return dists


def matching_dist_vec(
    X: Int[np.ndarray, "s n"],
) -> Float[np.ndarray, "s s"]:
    matches: Bool[np.ndarray, "s n n"] = X[:, :, None] == X[:, None, :]
    diffs: Bool[np.ndarray, "s s n n"] = matches[:, None, :, :] ^ matches[None, :, :, :]
    return diffs.sum(axis=(-1, -2)).astype(np.float32)


def matching_dist_np(X: Int[np.ndarray, "s n"]) -> Float[np.ndarray, "s s"]:
    return matching_dist(X)


def matching_dist_vec_np(X: Int[np.ndarray, "s n"]) -> Float[np.ndarray, "s s"]:
    return matching_dist_vec(X)
