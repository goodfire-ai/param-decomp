from collections import OrderedDict
from typing import Any

import numpy as np
import pytest
import torch

from param_decomp_lab.clustering.activations import (
    ProcessedActivations,
    process_activations,
)
from param_decomp_lab.clustering.formatting import DeadComponentFilterStat
from param_decomp_lab.clustering.harvest_config import HarvestConfig
from param_decomp_lab.clustering.memberships import (
    MembershipBuilder,
    ProcessedMemberships,
    collect_memberships,
)


def _assert_processed_memberships_match_dense(
    *,
    processed_memberships: ProcessedMemberships,
    processed_dense: ProcessedActivations,
    activation_threshold: float,
    expected_preview_rows: int | None = None,
) -> None:
    assert processed_memberships.module_component_counts == processed_dense.module_component_counts
    assert processed_memberships.module_alive_counts == processed_dense.module_alive_counts
    assert processed_memberships.labels == processed_dense.labels
    assert processed_memberships.dead_components_lst == processed_dense.dead_components_lst
    assert processed_memberships.n_components_alive == processed_dense.n_components_alive
    assert processed_memberships.n_components_dead == processed_dense.n_components_dead

    for membership, dense_column in zip(
        processed_memberships.memberships,
        processed_dense.activations.T,
        strict=True,
    ):
        expected_indices = torch.nonzero(dense_column > activation_threshold, as_tuple=False).view(
            -1
        )
        np.testing.assert_array_equal(
            membership.to_sample_indices(),
            expected_indices.numpy(),
        )

    if expected_preview_rows is None:
        return

    assert processed_memberships.preview is not None
    assert processed_memberships.preview.labels == processed_dense.labels
    assert processed_memberships.preview.dead_components_lst == processed_dense.dead_components_lst
    assert torch.allclose(
        processed_memberships.preview.activations,
        processed_dense.activations[:expected_preview_rows],
    )


@pytest.mark.parametrize("filter_dead_stat", ["max", "mean"])
def test_membership_builder_matches_dense_thresholded_path(
    filter_dead_stat: DeadComponentFilterStat,
) -> None:
    activation_threshold = 0.1
    filter_dead_threshold = 0.1

    batch_1 = OrderedDict(
        {
            "module_a": torch.tensor(
                [
                    [0.20, 0.11, 0.00],
                    [0.00, 0.12, 0.00],
                ]
            ),
            "module_b": torch.tensor(
                [
                    [0.09, 0.20],
                    [0.09, 0.20],
                ]
            ),
        }
    )
    batch_2 = OrderedDict(
        {
            "module_a": torch.tensor(
                [
                    [0.00, 0.13, 0.00],
                    [0.00, 0.14, 0.00],
                ]
            ),
            "module_b": torch.tensor(
                [
                    [0.09, 0.20],
                    [0.09, 0.20],
                ]
            ),
        }
    )

    builder = MembershipBuilder(
        activation_threshold=activation_threshold,
        filter_dead_threshold=filter_dead_threshold,
        filter_dead_stat=filter_dead_stat,
        filter_modules=None,
        preview_n_samples=3,
    )
    builder.add_batch(batch_1)
    builder.add_batch(batch_2)
    processed_memberships = builder.finalize()

    dense_activations = {key: torch.cat([batch_1[key], batch_2[key]], dim=0) for key in batch_1}
    processed_dense = process_activations(
        activations=dense_activations,
        filter_dead_threshold=filter_dead_threshold,
        filter_dead_stat=filter_dead_stat,
        filter_modules=None,
    )

    assert processed_memberships.n_samples == 4
    _assert_processed_memberships_match_dense(
        processed_memberships=processed_memberships,
        processed_dense=processed_dense,
        activation_threshold=activation_threshold,
        expected_preview_rows=3,
    )


@pytest.mark.parametrize("batch_as_dict", [True, False])
def test_collect_memberships_all_tokens_matches_dense(
    monkeypatch: Any, batch_as_dict: bool
) -> None:
    activation_threshold = 0.1
    filter_dead_threshold = 0.1

    def fake_component_activations(model: Any, device: torch.device | str, batch: torch.Tensor):  # pyright: ignore[reportUnusedParameter]
        vals = batch.to(torch.float32)
        return OrderedDict(
            {
                "module_a": torch.stack(
                    [
                        vals / 10.0,
                        (vals.remainder(3) == 0).to(torch.float32) * 0.2,
                        torch.zeros_like(vals),
                    ],
                    dim=-1,
                ),
                "module_b": torch.stack(
                    [
                        (vals >= 4).to(torch.float32) * 0.11,
                        ((vals + 1).remainder(2) == 0).to(torch.float32) * 0.3,
                    ],
                    dim=-1,
                ),
            }
        )

    monkeypatch.setattr(
        "param_decomp_lab.clustering.memberships.component_activations",
        fake_component_activations,
    )

    input_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    batch = {"input_ids": input_ids} if batch_as_dict else input_ids
    config = HarvestConfig(
        model_path="entity/project/runs/p-deadbeef",  # unused: collect_memberships only reads sampling fields
        batch_size=2,
        n_tokens=6,
        n_tokens_per_seq=None,
        use_all_tokens_per_seq=True,
        dataset_seed=0,
        activation_threshold=activation_threshold,
        filter_dead_threshold=filter_dead_threshold,
        filter_dead_stat="max",
    )
    processed_memberships = collect_memberships(
        model=None,  # pyright: ignore[reportArgumentType]
        dataloader=[batch],  # pyright: ignore[reportArgumentType]
        device="cpu",
        config=config,
    )

    raw_activations = fake_component_activations(None, "cpu", input_ids)
    dense_activations = {
        key: tensor.reshape(-1, tensor.shape[-1])[:6] for key, tensor in raw_activations.items()
    }
    processed_dense = process_activations(
        activations=dense_activations,
        filter_dead_threshold=filter_dead_threshold,
        filter_dead_stat="max",
        filter_modules=None,
    )

    assert processed_memberships.n_samples == 6
    _assert_processed_memberships_match_dense(
        processed_memberships=processed_memberships,
        processed_dense=processed_dense,
        activation_threshold=activation_threshold,
        expected_preview_rows=6,
    )
