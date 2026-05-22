import einops
import torch
from jaxtyping import Float
from torch import Tensor


def compute_feature_importances(
    batch_size: int,
    n_features: int,
    importance_val: float | None,
    device: str,
) -> Float[Tensor, "batch_size n_features"]:
    """Per-feature importance weights for the resid-MLP target loss.

    Feature `i` gets importance `importance_val ** i`. `None` or `1.0` returns ones.
    """
    if importance_val is None or importance_val == 1.0:
        return torch.ones(batch_size, n_features, device=device)
    powers = torch.arange(n_features, device=device)
    importances = torch.pow(importance_val, powers)
    return einops.repeat(importances, "n_features -> batch_size n_features", batch_size=batch_size)
