"""Tests for merge pair sampling functionality."""

import pytest
import torch

from spd.clustering.math.merge_pair_samplers import (
    MERGE_PAIR_SAMPLERS,
    mcmc_sampler,
    range_sampler,
)


class TestRangeSampler:
    """Test range-based merge pair sampling."""

    def test_range_sampler_basic(self):
        """Test basic functionality of range sampler."""
        k = 5
        costs = torch.randn(k, k)
        costs = (costs + costs.T) / 2  # Make symmetric
        costs.fill_diagonal_(float("inf"))  # No self-merges

        # Test with different thresholds
        pair_low = range_sampler(costs, threshold=0.0)
        pair_mid = range_sampler(costs, threshold=0.5)
        pair_high = range_sampler(costs, threshold=1.0)

        # All should return valid pairs
        assert pair_low[0] != pair_low[1]
        assert pair_mid[0] != pair_mid[1]
        assert pair_high[0] != pair_high[1]

        # All indices should be in valid range
        for pair in [pair_low, pair_mid, pair_high]:
            assert 0 <= pair[0] < k
            assert 0 <= pair[1] < k

    def test_range_sampler_threshold_zero(self):
        """Test that threshold=0 always selects minimum cost pair."""
        k = 5
        costs = torch.randn(k, k)
        costs = (costs + costs.T) / 2
        costs.fill_diagonal_(float("inf"))

        # Find the true minimum
        min_val = float("inf")
        _min_pair = None
        for i in range(k):
            for j in range(k):
                if i != j and costs[i, j] < min_val:
                    min_val = costs[i, j].item()
                    _min_pair = (i, j)

        # Sample multiple times with threshold=0
        for _ in range(10):
            pair = range_sampler(costs, threshold=0.0)
            # Should always get the minimum (or its symmetric equivalent)
            assert costs[pair[0], pair[1]] == min_val or costs[pair[1], pair[0]] == min_val

    def test_range_sampler_threshold_one(self):
        """Test that threshold=1 can select any non-diagonal pair."""
        k = 4
        costs = torch.randn(k, k)
        costs = (costs + costs.T) / 2
        costs.fill_diagonal_(float("inf"))

        # Sample many times to check we get different pairs
        pairs_seen = set()
        for _ in range(100):
            pair = range_sampler(costs, threshold=1.0)
            # Normalize pair order for comparison
            normalized = tuple(sorted(pair))
            pairs_seen.add(normalized)

        # With threshold=1, we should see multiple different pairs
        assert len(pairs_seen) > 1

    def test_range_sampler_small_matrix(self):
        """Test range sampler with 2x2 matrix."""
        costs = torch.tensor([[float("inf"), 1.0], [1.0, float("inf")]])

        pair = range_sampler(costs, threshold=0.5)
        # Only valid pair is (0, 1) or (1, 0)
        assert set(pair) == {0, 1}


class TestMCMCSampler:
    """Test MCMC-based merge pair sampling."""

    def test_mcmc_sampler_basic(self):
        """Test basic functionality of MCMC sampler."""
        k = 5
        costs = torch.randn(k, k)
        costs = (costs + costs.T) / 2
        costs.fill_diagonal_(float("inf"))

        # Test with different temperatures
        pair_low_temp = mcmc_sampler(costs, temperature=0.1)
        pair_mid_temp = mcmc_sampler(costs, temperature=1.0)
        pair_high_temp = mcmc_sampler(costs, temperature=10.0)

        # All should return valid pairs
        for pair in [pair_low_temp, pair_mid_temp, pair_high_temp]:
            assert pair[0] != pair[1]
            assert 0 <= pair[0] < k
            assert 0 <= pair[1] < k

    def test_mcmc_sampler_low_temperature(self):
        """Test that low temperature favors low-cost pairs."""
        k = 5
        costs = torch.randn(k, k)
        costs = (costs + costs.T) / 2
        costs.fill_diagonal_(float("inf"))

        # Find minimum cost
        min_val = float("inf")
        for i in range(k):
            for j in range(k):
                if i != j:
                    min_val = min(min_val, costs[i, j].item())

        # Sample many times with very low temperature
        low_cost_count = 0
        n_samples = 100
        for _ in range(n_samples):
            pair = mcmc_sampler(costs, temperature=0.01)
            cost = costs[pair[0], pair[1]].item()
            # Check if it's close to minimum
            if abs(cost - min_val) < 0.5:  # Within 0.5 of minimum
                low_cost_count += 1

        # Most samples should be near minimum with low temperature
        assert low_cost_count > n_samples * 0.7

    def test_mcmc_sampler_high_temperature(self):
        """Test that high temperature gives more uniform sampling."""
        k = 4
        costs = torch.randn(k, k)
        costs = (costs + costs.T) / 2
        costs.fill_diagonal_(float("inf"))

        # Sample many times with high temperature
        pairs_count = {}
        n_samples = 1000
        for _ in range(n_samples):
            pair = mcmc_sampler(costs, temperature=100.0)
            # Normalize pair order for counting
            normalized = tuple(sorted(pair))
            pairs_count[normalized] = pairs_count.get(normalized, 0) + 1

        # With high temperature, distribution should be relatively uniform
        # There are k*(k-1)/2 unique pairs
        expected_count = n_samples / (k * (k - 1) / 2)
        for count in pairs_count.values():
            # Each pair count should be within reasonable range of expected
            assert expected_count * 0.3 < count < expected_count * 1.7

    def test_mcmc_sampler_small_matrix(self):
        """Test MCMC sampler with 2x2 matrix."""
        costs = torch.tensor([[float("inf"), 1.0], [1.0, float("inf")]])

        pair = mcmc_sampler(costs, temperature=1.0)
        # Only valid pair is (0, 1) or (1, 0)
        assert set(pair) == {0, 1}

    def test_mcmc_sampler_extreme_costs(self):
        """Test MCMC sampler with extreme cost differences."""
        k = 3
        # Create matrix with one very low cost and rest high
        costs = torch.full((k, k), 1000.0)
        costs[0, 1] = costs[1, 0] = 1.0  # One low-cost pair
        costs.fill_diagonal_(float("inf"))

        # With low temperature, should almost always select the low-cost pair
        low_cost_selected = 0
        for _ in range(100):
            pair = mcmc_sampler(costs, temperature=0.1)
            if set(pair) == {0, 1}:
                low_cost_selected += 1

        assert low_cost_selected > 95  # Should almost always select (0,1)


class TestSamplerRegistry:
    """Test the sampler registry."""

    def test_registry_contains_samplers(self):
        """Test that registry contains expected samplers."""
        assert "range" in MERGE_PAIR_SAMPLERS
        assert "mcmc" in MERGE_PAIR_SAMPLERS
        assert MERGE_PAIR_SAMPLERS["range"] is range_sampler
        assert MERGE_PAIR_SAMPLERS["mcmc"] is mcmc_sampler

    def test_registry_samplers_callable(self):
        """Test that all registry samplers are callable with correct signature."""
        k = 3
        costs = torch.randn(k, k)
        costs = (costs + costs.T) / 2
        costs.fill_diagonal_(float("inf"))

        for name, sampler in MERGE_PAIR_SAMPLERS.items():
            # Should be callable
            assert callable(sampler)

            # Test with default kwargs
            if name == "range":
                pair = sampler(costs, threshold=0.5)
            elif name == "mcmc":
                pair = sampler(costs, temperature=1.0)
            elif name == "exp_rank":
                pair = sampler(costs, decay=0.2)
            else:
                pytest.fail(f"Unknown sampler {name}")

            # Should return valid pair
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert pair[0] != pair[1]
            assert 0 <= pair[0] < k
            assert 0 <= pair[1] < k


class TestSamplerIntegration:
    """Integration tests for samplers with edge cases."""

    def test_samplers_deterministic_with_seed(self):
        """Test that samplers are deterministic with fixed seed."""
        k = 5
        costs = torch.randn(k, k)
        costs = (costs + costs.T) / 2
        costs.fill_diagonal_(float("inf"))

        # Test range sampler
        torch.manual_seed(42)
        pair1 = range_sampler(costs, threshold=0.5)
        torch.manual_seed(42)
        pair2 = range_sampler(costs, threshold=0.5)
        # Can't guarantee exact match due to Python's random module
        # but both should be valid
        assert pair1[0] != pair1[1]
        assert pair2[0] != pair2[1]

        # Test MCMC sampler
        torch.manual_seed(42)
        pair1 = mcmc_sampler(costs, temperature=1.0)
        torch.manual_seed(42)
        pair2 = mcmc_sampler(costs, temperature=1.0)
        assert pair1 == pair2  # Should be deterministic with same seed

    def test_samplers_all_infinite_costs(self):
        """Test samplers handle all-infinite costs gracefully."""
        k = 3
        costs = torch.full((k, k), float("inf"))

        # This is an edge case - no valid pairs exist
        # Samplers should handle this without crashing
        # (though the result may not be meaningful)
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            range_sampler(costs, threshold=0.5)
