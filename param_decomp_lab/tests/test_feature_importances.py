import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from param_decomp_lab.experiments.resid_mlp.feature_importances import compute_feature_importances


@pytest.mark.parametrize(
    "importance_val, expected_tensor",
    [
        (1.0, torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])),
        (0.5, torch.tensor([[1.0, 0.5, 0.25], [1.0, 0.5, 0.25]])),
        (0.0, torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])),
    ],
)
def test_compute_feature_importances(
    importance_val: float, expected_tensor: Float[Tensor, "batch_size n_features"]
):
    importances = compute_feature_importances(
        batch_size=2, n_features=3, importance_val=importance_val, device="cpu"
    )
    torch.testing.assert_close(importances, expected_tensor)
