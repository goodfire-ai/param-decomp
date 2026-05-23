"""Sparse-feature dataset for the Residual MLP experiment.

Inherits the basic batch-generation logic from `SparseFeatureDataset` in the TMS
experiment, and optionally computes labels of the form `act_fn(coeffs*x) + x` or
`abs(coeffs*x)`.
"""

from typing import Literal, override

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from param_decomp_lab.experiments.tms.data import SparseFeatureDataset


class ResidMLPDataset(SparseFeatureDataset):
    def __init__(
        self,
        n_features: int,
        feature_probability: float,
        device: str,
        batch_size: int,
        calc_labels: bool = True,
        label_type: Literal["act_plus_resid", "abs"] | None = None,
        act_fn_name: Literal["relu", "gelu"] | None = None,
        label_fn_seed: int | None = None,
        label_coeffs: Float[Tensor, " n_features"] | None = None,
        data_generation_type: Literal[
            "exactly_one_active", "exactly_two_active", "at_least_zero_active"
        ] = "at_least_zero_active",
        synced_inputs: list[list[int]] | None = None,
    ):
        super().__init__(
            n_features=n_features,
            feature_probability=feature_probability,
            device=device,
            batch_size=batch_size,
            data_generation_type=data_generation_type,
            value_range=(-1.0, 1.0),
            synced_inputs=synced_inputs,
        )

        self.label_fn = None
        self.label_coeffs = None

        if calc_labels:
            self.label_coeffs = (
                self.calc_label_coeffs(label_fn_seed) if label_coeffs is None else label_coeffs
            ).to(self.device)

            assert label_type is not None, "Must provide label_type if calc_labels is True"
            if label_type == "act_plus_resid":
                assert act_fn_name in ["relu", "gelu"], "act_fn_name must be 'relu' or 'gelu'"
                self.label_fn = lambda batch: self.calc_act_plus_resid_labels(
                    batch=batch, act_fn_name=act_fn_name
                )
            elif label_type == "abs":
                self.label_fn = lambda batch: self.calc_abs_labels(batch)

    @override
    def generate_batch(
        self, batch_size: int
    ) -> tuple[Float[Tensor, "batch n_functions"], Float[Tensor, "batch n_functions"]]:
        batch, parent_labels = super().generate_batch(batch_size)
        labels = self.label_fn(batch) if self.label_fn is not None else parent_labels
        return batch, labels

    def calc_act_plus_resid_labels(
        self,
        batch: Float[Tensor, "batch n_functions"],
        act_fn_name: Literal["relu", "gelu"],
    ) -> Float[Tensor, "batch n_functions"]:
        """Calculate the corresponding labels for the batch using `act_fn(coeffs*x) + x`."""
        assert self.label_coeffs is not None
        weighted_inputs = einops.einsum(
            batch,
            self.label_coeffs,
            "batch n_functions, n_functions -> batch n_functions",
        )
        assert act_fn_name in ["relu", "gelu"], "act_fn_name must be 'relu' or 'gelu'"
        act_fn = F.relu if act_fn_name == "relu" else F.gelu
        labels = act_fn(weighted_inputs) + batch
        return labels

    def calc_abs_labels(
        self, batch: Float[Tensor, "batch n_functions"]
    ) -> Float[Tensor, "batch n_functions"]:
        assert self.label_coeffs is not None
        weighted_inputs = einops.einsum(
            batch,
            self.label_coeffs,
            "batch n_functions, n_functions -> batch n_functions",
        )
        return torch.abs(weighted_inputs)

    def calc_label_coeffs(self, label_fn_seed: int | None = None) -> Float[Tensor, " n_features"]:
        """Create random coeffs between [1, 2] using label_fn_seed if provided."""
        gen = torch.Generator(device=self.device)
        if label_fn_seed is not None:
            gen.manual_seed(label_fn_seed)
        return torch.rand(self.n_features, generator=gen, device=self.device) + 1
