from collections import OrderedDict

import numpy as np
import pytest

from param_decomp_lab.clustering.activations import (
    ProcessedActivations,
    process_activations,
)
from param_decomp_lab.clustering.formatting import DeadComponentFilterStat
from param_decomp_lab.clustering.memberships import (
    MembershipBuilder,
    ProcessedMemberships,
)


def _assert_processed_memberships_match_dense(
    *,
    processed_memberships: ProcessedMemberships,
    processed_dense: ProcessedActivations,
    activation_threshold: float,
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
        expected_indices = np.flatnonzero(dense_column > activation_threshold)
        np.testing.assert_array_equal(
            membership.to_sample_indices(),
            expected_indices,
        )


@pytest.mark.parametrize("filter_dead_stat", ["max", "mean"])
def test_membership_builder_matches_dense_thresholded_path(
    filter_dead_stat: DeadComponentFilterStat,
) -> None:
    activation_threshold = 0.1
    filter_dead_threshold = 0.1

    batch_1 = OrderedDict(
        {
            "module_a": np.array(
                [
                    [0.20, 0.11, 0.00],
                    [0.00, 0.12, 0.00],
                ],
                dtype=np.float32,
            ),
            "module_b": np.array(
                [
                    [0.09, 0.20],
                    [0.09, 0.20],
                ],
                dtype=np.float32,
            ),
        }
    )
    batch_2 = OrderedDict(
        {
            "module_a": np.array(
                [
                    [0.00, 0.13, 0.00],
                    [0.00, 0.14, 0.00],
                ],
                dtype=np.float32,
            ),
            "module_b": np.array(
                [
                    [0.09, 0.20],
                    [0.09, 0.20],
                ],
                dtype=np.float32,
            ),
        }
    )

    builder = MembershipBuilder(
        activation_threshold=activation_threshold,
        filter_dead_threshold=filter_dead_threshold,
        filter_dead_stat=filter_dead_stat,
        filter_modules=None,
    )
    builder.add_batch(batch_1)
    builder.add_batch(batch_2)
    processed_memberships = builder.finalize()

    dense_activations = {
        key: np.concatenate([batch_1[key], batch_2[key]], axis=0) for key in batch_1
    }
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
    )
