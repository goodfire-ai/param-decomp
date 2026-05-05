import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

_DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def matching_dist(
    X: Int[Tensor, "s n"],
) -> Float[Tensor, "s s"]:
    s_ensemble, _n_components = X.shape
    matches: Bool[Tensor, "s n n"] = X[:, :, None] == X[:, None, :]

    dists: Float[Tensor, "s s"] = torch.full((s_ensemble, s_ensemble), torch.nan)

    for i in range(s_ensemble):
        for j in range(i + 1, s_ensemble):
            dist_mat = matches[i].float() - matches[j].float()
            dists[i, j] = torch.tril(dist_mat, diagonal=-1).abs().sum()

    return dists


def matching_dist_vec(
    X: Int[Tensor, "s n"],
) -> Float[Tensor, "s s"]:
    matches: Bool[Tensor, "s n n"] = X[:, :, None] == X[:, None, :]
    diffs: Bool[Tensor, "s s n n"] = matches[:, None, :, :] ^ matches[None, :, :, :]

    dists_int: torch.Tensor = diffs.sum(dim=(-1, -2))
    dists: Float[Tensor, "s s"] = dists_int.to(torch.float32)
    return dists


def matching_dist_np(
    X: Int[np.ndarray, "s n"],
    device: torch.device = _DEVICE,
) -> Float[np.ndarray, "s s"]:
    return matching_dist(torch.tensor(X, device=device)).cpu().numpy()


def matching_dist_vec_np(
    X: Int[np.ndarray, "s n"],
    device: torch.device = _DEVICE,
) -> Float[np.ndarray, "s s"]:
    return matching_dist_vec(torch.tensor(X, device=device)).cpu().numpy()
