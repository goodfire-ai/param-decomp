from typing import Literal

import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from param_decomp.utils.data_utils import SparseFeatureDataset
from param_decomp.utils.general_utils import compute_feature_importances, resolve_class


def test_dataset_at_least_zero_active():
    n_features = 5
    feature_probability = 0.5
    device = "cpu"
    batch_size = 200

    dataset = SparseFeatureDataset(
        n_features=n_features,
        feature_probability=feature_probability,
        device=device,
        data_generation_type="at_least_zero_active",
        value_range=(0.0, 1.0),
    )

    batch, _ = dataset.generate_batch(batch_size)

    # Check shape
    assert batch.shape == (batch_size, n_features), "Incorrect batch shape"

    # Check that the values are between 0 and 1
    assert torch.all((batch >= 0) & (batch <= 1)), "Values should be between 0 and 1"

    # Check that the proportion of non-zero elements is close to feature_probability
    non_zero_proportion = torch.count_nonzero(batch) / batch.numel()
    assert abs(non_zero_proportion - feature_probability) < 0.05, (
        f"Expected proportion {feature_probability}, but got {non_zero_proportion}"
    )


def test_generate_multi_feature_batch_no_zero_samples():
    n_features = 5
    feature_probability = 0.05  # Low probability to increase chance of zero samples
    device = "cpu"
    batch_size = 100
    buffer_ratio = 1.5

    dataset = SparseFeatureDataset(
        n_features=n_features,
        feature_probability=feature_probability,
        device=device,
        data_generation_type="at_least_zero_active",
        value_range=(0.0, 1.0),
    )

    batch = dataset._generate_multi_feature_batch_no_zero_samples(batch_size, buffer_ratio)

    # Check shape
    assert batch.shape == (batch_size, n_features), "Incorrect batch shape"

    # Check that the values are between 0 and 1
    assert torch.all((batch >= 0) & (batch <= 1)), "Values should be between 0 and 1"

    # Check that there are no all-zero samples
    zero_samples = (batch.sum(dim=-1) == 0).sum()
    assert zero_samples == 0, f"Found {zero_samples} samples with all zeros"


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_dataset_exactly_n_active(n: int):
    n_features = 10
    feature_probability = 0.5  # This won't be used when data_generation_type="exactly_one_active"
    device = "cpu"
    batch_size = 10
    value_range = (0.0, 1.0)

    n_map: dict[
        int,
        Literal[
            "exactly_one_active",
            "exactly_two_active",
            "exactly_three_active",
            "exactly_four_active",
            "exactly_five_active",
        ],
    ] = {
        1: "exactly_one_active",
        2: "exactly_two_active",
        3: "exactly_three_active",
        4: "exactly_four_active",
        5: "exactly_five_active",
    }
    dataset = SparseFeatureDataset(
        n_features=n_features,
        feature_probability=feature_probability,
        device=device,
        data_generation_type=n_map[n],
        value_range=value_range,
    )

    batch, _ = dataset.generate_batch(batch_size)

    # Check shape
    assert batch.shape == (batch_size, n_features), "Incorrect batch shape"

    # Check that there's exactly one non-zero value per sample
    for sample in batch:
        non_zero_count = torch.count_nonzero(sample)
        assert non_zero_count == n, f"Expected {n} non-zero values, but found {non_zero_count}"

    # Check that the non-zero values are in the value_range
    non_zero_values = batch[batch != 0]
    assert torch.all((non_zero_values >= value_range[0]) & (non_zero_values <= value_range[1])), (
        f"Non-zero values should be between {value_range[0]} and {value_range[1]}"
    )


@pytest.mark.parametrize(
    "importance_val, expected_tensor",
    [
        (
            1.0,
            torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        ),
        (
            0.5,
            torch.tensor([[1.0, 0.5, 0.25], [1.0, 0.5, 0.25]]),
        ),
        (
            0.0,
            torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        ),
    ],
)
def test_compute_feature_importances(
    importance_val: float, expected_tensor: Float[Tensor, "batch_size n_features"]
):
    importances = compute_feature_importances(
        batch_size=2, n_features=3, importance_val=importance_val, device="cpu"
    )
    torch.testing.assert_close(importances, expected_tensor)


def test_sync_inputs_non_overlapping():
    dataset = SparseFeatureDataset(
        n_features=6,
        feature_probability=0.5,
        device="cpu",
        data_generation_type="at_least_zero_active",
        value_range=(0.0, 1.0),
        synced_inputs=[[0, 1], [2, 3, 4]],
    )

    batch, _ = dataset.generate_batch(5)

    for sample in batch:
        # If there is a value in 0 or 1, there should be a value in 1 or
        if sample[0] != 0.0:
            assert sample[1] != 0.0
        if sample[1] != 0.0:
            assert sample[0] != 0.0
        if sample[2] != 0.0:
            assert sample[3] != 0.0 and sample[4] != 0.0
        if sample[3] != 0.0:
            assert sample[2] != 0.0 and sample[4] != 0.0
        if sample[4] != 0.0:
            assert sample[2] != 0.0 and sample[3] != 0.0


def test_sync_inputs_overlapping():
    dataset = SparseFeatureDataset(
        n_features=6,
        feature_probability=0.5,
        device="cpu",
        data_generation_type="at_least_zero_active",
        value_range=(0.0, 1.0),
        synced_inputs=[[0, 1], [1, 2, 3]],
    )
    # Should raise an assertion error with the word "overlapping"
    with pytest.raises(AssertionError, match="overlapping"):
        dataset.generate_batch(5)


def test_resolve_class():
    assert resolve_class("torch.nn.Linear") == torch.nn.Linear
    from transformers import LlamaForCausalLM

    assert resolve_class("transformers.LlamaForCausalLM") == LlamaForCausalLM
    with pytest.raises(ImportError):
        resolve_class("fakepackage.fakemodule.FakeClass")
