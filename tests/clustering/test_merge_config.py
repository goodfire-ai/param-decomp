"""Tests for MergeConfig with new sampling system."""

import pytest
import torch

from spd.clustering.merge_config import MergeConfig


class TestMergeConfigSampling:
    """Test MergeConfig integration with sampling system."""

    def test_default_config(self):
        """Test default MergeConfig uses range sampler."""
        config = MergeConfig()

        assert config.merge_pair_sampling_method == "range"
        assert config.merge_pair_sampling_kwargs == {"threshold": 0.05}
        assert config.alpha == 1.0

    def test_range_sampler_config(self):
        """Test MergeConfig with range sampler."""
        config = MergeConfig(
            merge_pair_sampling_method="range", merge_pair_sampling_kwargs={"threshold": 0.1}
        )

        assert config.merge_pair_sampling_method == "range"
        assert config.merge_pair_sampling_kwargs == {"threshold": 0.1}

        # Test that sampler works
        k = 4
        costs = torch.randn(k, k)
        costs = (costs + costs.T) / 2
        costs.fill_diagonal_(float("inf"))

        pair = config.merge_pair_sample(costs)

        assert isinstance(pair, tuple)
        assert len(pair) == 2
        assert pair[0] != pair[1]

    def test_mcmc_sampler_config(self):
        """Test MergeConfig with MCMC sampler."""
        config = MergeConfig(
            merge_pair_sampling_method="mcmc", merge_pair_sampling_kwargs={"temperature": 2.0}
        )

        assert config.merge_pair_sampling_method == "mcmc"
        assert config.merge_pair_sampling_kwargs == {"temperature": 2.0}

        # Test that sampler works
        k = 4
        costs = torch.randn(k, k)
        costs = (costs + costs.T) / 2
        costs.fill_diagonal_(float("inf"))

        pair = config.merge_pair_sample(costs)

        assert isinstance(pair, tuple)
        assert len(pair) == 2
        assert pair[0] != pair[1]

    def test_invalid_sampler_method(self):
        """Test that invalid sampler method raises error."""
        from pydantic import ValidationError

        # Pydantic validates at construction time
        with pytest.raises(ValidationError):
            _config = MergeConfig(merge_pair_sampling_method="invalid")  # pyright: ignore[reportArgumentType]

    def test_config_with_all_parameters(self):
        """Test MergeConfig with all parameters set."""
        config = MergeConfig(
            alpha=1.5,
            iters=200,
            merge_pair_sampling_method="mcmc",
            merge_pair_sampling_kwargs={"temperature": 0.5},
        )

        assert config.alpha == 1.5
        assert config.iters == 200
        assert config.merge_pair_sampling_method == "mcmc"
        assert config.merge_pair_sampling_kwargs == {"temperature": 0.5}

    def test_config_serialization(self):
        """Test that config can be serialized and deserialized."""
        config = MergeConfig(
            merge_pair_sampling_method="mcmc", merge_pair_sampling_kwargs={"temperature": 1.5}
        )

        # Serialize to dict
        config_dict = config.model_dump()
        assert config_dict["merge_pair_sampling_method"] == "mcmc"
        assert config_dict["merge_pair_sampling_kwargs"] == {"temperature": 1.5}

        # Deserialize from dict
        config2 = MergeConfig(**config_dict)
        assert config2.merge_pair_sampling_method == "mcmc"
        assert config2.merge_pair_sampling_kwargs == {"temperature": 1.5}

    def test_config_json_serialization(self):
        """Test JSON serialization of config."""
        config = MergeConfig(
            merge_pair_sampling_method="range", merge_pair_sampling_kwargs={"threshold": 0.2}
        )

        # Serialize to JSON string
        json_str = config.model_dump_json()
        assert "range" in json_str
        assert "0.2" in json_str

        # Parse back from JSON
        import json

        config_dict = json.loads(json_str)
        config2 = MergeConfig(**config_dict)

        assert config2.merge_pair_sampling_method == "range"
        assert config2.merge_pair_sampling_kwargs == {"threshold": 0.2}

    def test_stable_hash_changes_with_sampling_params(self):
        """Test that stable_hash changes when sampling parameters change."""
        config1 = MergeConfig(
            merge_pair_sampling_method="range", merge_pair_sampling_kwargs={"threshold": 0.1}
        )
        config2 = MergeConfig(
            merge_pair_sampling_method="range", merge_pair_sampling_kwargs={"threshold": 0.2}
        )
        config3 = MergeConfig(
            merge_pair_sampling_method="mcmc", merge_pair_sampling_kwargs={"temperature": 1.0}
        )

        # Different configs should have different hashes
        assert config1.stable_hash != config2.stable_hash
        assert config1.stable_hash != config3.stable_hash
        assert config2.stable_hash != config3.stable_hash

        # Same config should have same hash
        config4 = MergeConfig(
            merge_pair_sampling_method="range", merge_pair_sampling_kwargs={"threshold": 0.1}
        )
        assert config1.stable_hash == config4.stable_hash

    def test_empty_kwargs(self):
        """Test that empty kwargs dict works."""
        config = MergeConfig(merge_pair_sampling_method="range", merge_pair_sampling_kwargs={})

        # Should work with default parameters of the sampler
        k = 3
        costs = torch.randn(k, k)
        costs = (costs + costs.T) / 2
        costs.fill_diagonal_(float("inf"))

        # Range sampler has default threshold=0.05
        pair = config.merge_pair_sample(costs)

        assert isinstance(pair, tuple)
        assert pair[0] != pair[1]

    def test_extra_kwargs_filtered(self):
        """Test that only valid kwargs are used by sampler."""
        config = MergeConfig(
            merge_pair_sampling_method="range", merge_pair_sampling_kwargs={"threshold": 0.3}
        )

        k = 3
        costs = torch.randn(k, k)
        costs = (costs + costs.T) / 2
        costs.fill_diagonal_(float("inf"))

        # Should work with config's method
        pair = config.merge_pair_sample(costs)

        assert isinstance(pair, tuple)
        assert pair[0] != pair[1]
