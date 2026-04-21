"""Integration tests for the merge system with different samplers."""

import torch

from spd.clustering.compute_costs import recompute_coacts_merge_pair_memberships
from spd.clustering.consts import ComponentLabels, MergePair
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.merge import merge_iteration_memberships
from spd.clustering.merge_config import MergeConfig
from spd.clustering.sample_membership import (
    CompressedMembership,
    compute_coactivation_matrix,
    memberships_to_sample_component_csr,
)


def _activations_to_memberships(
    activations: torch.Tensor, threshold: float
) -> list[CompressedMembership]:
    """Convert dense activations to compressed memberships for testing."""
    n_samples = activations.shape[0]
    memberships = []
    for comp_idx in range(activations.shape[1]):
        active = (activations[:, comp_idx] > threshold).nonzero(as_tuple=True)[0]
        memberships.append(
            CompressedMembership.from_sample_indices(active.numpy(), n_samples=n_samples)
        )
    return memberships


class TestMergeIntegration:
    """Test the full merge iteration with different samplers."""

    def test_merge_with_range_sampler(self):
        """Test merge iteration with range sampler."""
        n_samples = 100
        n_components = 10
        threshold = 0.1
        activations = torch.rand(n_samples, n_components)
        component_labels = ComponentLabels([f"comp_{i}" for i in range(n_components)])

        config = MergeConfig(
            alpha=1.0,
            iters=5,
            merge_pair_sampling_method="range",
            merge_pair_sampling_kwargs={"threshold": 0.1},
        )

        memberships = _activations_to_memberships(activations, threshold)
        history = merge_iteration_memberships(
            merge_config=config,
            memberships=memberships,
            n_samples=n_samples,
            component_labels=component_labels,
        )

        assert history is not None
        assert len(history.merges.k_groups) > 0
        assert history.merges.k_groups[0].item() == n_components - 1
        assert history.merges.k_groups[-1].item() < n_components
        assert history.merges.k_groups[-1].item() >= 2

    def test_merge_with_mcmc_sampler(self):
        """Test merge iteration with MCMC sampler."""
        n_samples = 100
        n_components = 10
        threshold = 0.1
        activations = torch.rand(n_samples, n_components)
        component_labels = ComponentLabels([f"comp_{i}" for i in range(n_components)])

        config = MergeConfig(
            alpha=1.0,
            iters=5,
            merge_pair_sampling_method="mcmc",
            merge_pair_sampling_kwargs={"temperature": 1.0},
        )

        memberships = _activations_to_memberships(activations, threshold)
        history = merge_iteration_memberships(
            merge_config=config,
            memberships=memberships,
            n_samples=n_samples,
            component_labels=component_labels,
        )

        assert history is not None
        assert len(history.merges.k_groups) > 0
        assert history.merges.k_groups[0].item() == n_components - 1
        assert history.merges.k_groups[-1].item() < n_components
        assert history.merges.k_groups[-1].item() >= 2

    def test_merge_comparison_samplers(self):
        """Compare behavior of different samplers with same data."""
        n_samples = 100
        n_components = 8
        activations = torch.rand(n_samples, n_components)
        activations[:, 0] *= 2
        activations[:, 1] *= 0.1

        component_labels = ComponentLabels([f"comp_{i}" for i in range(n_components)])
        threshold = 0.1

        config_range = MergeConfig(
            alpha=1.0,
            iters=3,
            merge_pair_sampling_method="range",
            merge_pair_sampling_kwargs={"threshold": 0.0},
        )

        memberships_range = _activations_to_memberships(activations.clone(), threshold)
        history_range = merge_iteration_memberships(
            merge_config=config_range,
            memberships=memberships_range,
            n_samples=n_samples,
            component_labels=ComponentLabels(component_labels.copy()),
        )

        config_mcmc = MergeConfig(
            alpha=1.0,
            iters=3,
            merge_pair_sampling_method="mcmc",
            merge_pair_sampling_kwargs={"temperature": 0.01},
        )

        memberships_mcmc = _activations_to_memberships(activations.clone(), threshold)
        history_mcmc = merge_iteration_memberships(
            merge_config=config_mcmc,
            memberships=memberships_mcmc,
            n_samples=n_samples,
            component_labels=ComponentLabels(component_labels.copy()),
        )

        assert history_range.merges.k_groups[-1].item() < n_components
        assert history_mcmc.merges.k_groups[-1].item() < n_components
        assert history_range.merges.k_groups[-1].item() >= 2
        assert history_mcmc.merges.k_groups[-1].item() >= 2

    def test_merge_with_small_components(self):
        """Test merge with very few components."""
        n_samples = 50
        n_components = 3
        threshold = 0.1
        activations = torch.rand(n_samples, n_components)
        component_labels = ComponentLabels([f"comp_{i}" for i in range(n_components)])

        config = MergeConfig(
            alpha=1.0,
            iters=1,
            merge_pair_sampling_method="mcmc",
            merge_pair_sampling_kwargs={"temperature": 2.0},
        )

        memberships = _activations_to_memberships(activations, threshold)
        history = merge_iteration_memberships(
            merge_config=config,
            memberships=memberships,
            n_samples=n_samples,
            component_labels=component_labels,
        )

        assert history.merges.k_groups[0].item() == 2
        assert history.merges.k_groups[-1].item() >= 2
        assert history.merges.k_groups[-1].item() <= 3

    def test_membership_recompute_matches_row_oriented_path(self):
        """Row-oriented overlap recompute should match the direct membership path exactly."""
        memberships = [
            CompressedMembership.from_sample_indices(torch.tensor(indices).numpy(), n_samples=8)
            for indices in ([0, 2, 5], [1, 2], [0, 3], [4, 5, 6])
        ]
        coact = compute_coactivation_matrix(memberships)
        merges = GroupMerge.identity(n_components=len(memberships))
        component_activity_csr = memberships_to_sample_component_csr(memberships)

        merge_row, coact_row, memberships_row = recompute_coacts_merge_pair_memberships(
            coact=coact,
            merges=merges,
            merge_pair=MergePair((0, 1)),
            memberships=memberships,
            component_activity_csr=component_activity_csr,
        )

        expected_group_idxs = torch.tensor([0, 0, 1, 2], dtype=torch.int64)
        expected_coact = torch.tensor(
            [
                [4.0, 1.0, 1.0],
                [1.0, 2.0, 0.0],
                [1.0, 0.0, 3.0],
            ]
        )
        assert torch.equal(merge_row.group_idxs, expected_group_idxs)
        assert torch.equal(coact_row, expected_coact)
        assert [membership.count() for membership in memberships_row] == [4, 2, 3]
