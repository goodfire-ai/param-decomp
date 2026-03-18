"""Integration tests for the merge system with new samplers."""

import torch

from spd.clustering.compute_costs import recompute_coacts_merge_pair_memberships
from spd.clustering.consts import ComponentLabels
from spd.clustering.merge import merge_iteration
from spd.clustering.merge_config import MergeConfig
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.sample_membership import (
    CompressedMembership,
    compute_coactivation_matrix,
    memberships_to_sample_component_csr,
)


class TestMergeIntegration:
    """Test the full merge iteration with different samplers."""

    def test_merge_with_range_sampler(self):
        """Test merge iteration with range sampler."""
        # Create test data
        n_samples = 100
        n_components = 10
        activations = torch.rand(n_samples, n_components)
        component_labels = ComponentLabels([f"comp_{i}" for i in range(n_components)])

        # Configure with range sampler
        config = MergeConfig(
            activation_threshold=0.1,
            alpha=1.0,
            iters=5,
            merge_pair_sampling_method="range",
            merge_pair_sampling_kwargs={"threshold": 0.1},
            filter_dead_threshold=0.001,
        )

        # Run merge iteration
        history = merge_iteration(
            activations=activations, merge_config=config, component_labels=component_labels
        )

        # Check results
        assert history is not None
        assert len(history.merges.k_groups) > 0
        # First entry is after first merge, so should be n_components - 1
        assert history.merges.k_groups[0].item() == n_components - 1
        # After iterations, should have fewer groups (merges reduce count)
        # Exact count depends on early stopping conditions
        assert history.merges.k_groups[-1].item() < n_components
        assert history.merges.k_groups[-1].item() >= 2  # Should stop before going below 2

    def test_merge_with_mcmc_sampler(self):
        """Test merge iteration with MCMC sampler."""
        # Create test data
        n_samples = 100
        n_components = 10
        activations = torch.rand(n_samples, n_components)
        component_labels = ComponentLabels([f"comp_{i}" for i in range(n_components)])

        # Configure with MCMC sampler
        config = MergeConfig(
            activation_threshold=0.1,
            alpha=1.0,
            iters=5,
            merge_pair_sampling_method="mcmc",
            merge_pair_sampling_kwargs={"temperature": 1.0},
            filter_dead_threshold=0.001,
        )

        # Run merge iteration
        history = merge_iteration(
            activations=activations, merge_config=config, component_labels=component_labels
        )

        # Check results
        assert history is not None
        assert len(history.merges.k_groups) > 0
        # First entry is after first merge, so should be n_components - 1
        assert history.merges.k_groups[0].item() == n_components - 1
        # Should have fewer groups after iterations
        assert history.merges.k_groups[-1].item() < n_components
        assert history.merges.k_groups[-1].item() >= 2

    def test_merge_comparison_samplers(self):
        """Compare behavior of different samplers with same data."""
        # Create test data with clear structure
        n_samples = 100
        n_components = 8
        activations = torch.rand(n_samples, n_components)

        # Make some components more active to create cost structure
        activations[:, 0] *= 2  # Component 0 is very active
        activations[:, 1] *= 0.1  # Component 1 is rarely active

        component_labels = ComponentLabels([f"comp_{i}" for i in range(n_components)])

        # Run with range sampler (threshold=0 for deterministic minimum selection)
        config_range = MergeConfig(
            activation_threshold=0.1,
            alpha=1.0,
            iters=3,
            merge_pair_sampling_method="range",
            merge_pair_sampling_kwargs={"threshold": 0.0},  # Always select minimum
        )

        history_range = merge_iteration(
            activations=activations.clone(),
            merge_config=config_range,
            component_labels=ComponentLabels(component_labels.copy()),
        )

        # Run with MCMC sampler (low temperature for near-deterministic)
        config_mcmc = MergeConfig(
            activation_threshold=0.1,
            alpha=1.0,
            iters=3,
            merge_pair_sampling_method="mcmc",
            merge_pair_sampling_kwargs={"temperature": 0.01},  # Very low temp
        )

        history_mcmc = merge_iteration(
            activations=activations.clone(),
            merge_config=config_mcmc,
            component_labels=ComponentLabels(component_labels.copy()),
        )

        # Both should reduce groups from initial count
        assert history_range.merges.k_groups[-1].item() < n_components
        assert history_mcmc.merges.k_groups[-1].item() < n_components
        assert history_range.merges.k_groups[-1].item() >= 2
        assert history_mcmc.merges.k_groups[-1].item() >= 2

    def test_merge_with_small_components(self):
        """Test merge with very few components."""
        # Edge case: only 3 components
        n_samples = 50
        n_components = 3
        activations = torch.rand(n_samples, n_components)
        component_labels = ComponentLabels([f"comp_{i}" for i in range(n_components)])

        config = MergeConfig(
            activation_threshold=0.1,
            alpha=1.0,
            iters=1,  # Just one merge
            merge_pair_sampling_method="mcmc",
            merge_pair_sampling_kwargs={"temperature": 2.0},
        )

        history = merge_iteration(
            activations=activations, merge_config=config, component_labels=component_labels
        )

        # First entry is after first merge, so should be 3 - 1 = 2
        assert history.merges.k_groups[0].item() == 2
        # Early stopping may occur at 2 groups, so final count could be 2 or 3
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
            merge_pair=(0, 1),
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
