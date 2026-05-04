"""Tests for sweep parameter loading and merging in param_decomp/scripts/run.py."""

import pytest
import yaml

from param_decomp.scripts.run import _get_experiment_sweep_params, _merge_sweep_params


class TestMergeSweepParams:
    """Test the _merge_sweep_params function."""

    def test_merge_simple_values(self):
        """Test merging simple values."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        _merge_sweep_params(base, override)
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested_dicts(self):
        """Test merging nested dictionaries."""
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 3, "z": 4}, "c": 5}
        _merge_sweep_params(base, override)
        assert base == {"a": {"x": 1, "y": 3, "z": 4}, "b": 3, "c": 5}

    def test_merge_nested_sweep_values(self):
        """Test merging with nested sweep values."""
        base = {
            "loss": {
                "faithfulness_weight": {"values": [0.1, 0.5]},
                "reconstruction_weight": {"values": [1.0]},
            }
        }
        override = {"loss": {"faithfulness_weight": {"values": [1.0, 2.0]}}}
        _merge_sweep_params(base, override)
        assert base == {
            "loss": {
                "faithfulness_weight": {"values": [1.0, 2.0]},
                "reconstruction_weight": {"values": [1.0]},
            }
        }

    def test_override_dict_with_non_dict(self):
        """Test overriding a dict value with a non-dict value."""
        base = {"a": {"x": 1}}
        override = {"a": "new_value"}
        _merge_sweep_params(base, override)
        assert base == {"a": "new_value"}

    def test_override_non_dict_with_dict(self):
        """Test overriding a non-dict value with a dict."""
        base = {"a": "old_value"}
        override = {"a": {"x": 1}}
        _merge_sweep_params(base, override)
        assert base == {"a": {"x": 1}}


class TestLoadSweepParams:
    """Test the load_sweep_params function."""

    def test_global_params_only(self):
        """Test loading with only global parameters."""
        params_yaml = """\
global:
  seed:
    values: [0, 1, 2]
  learning_rate:
    values: [0.001, 0.01]
"""
        result = _get_experiment_sweep_params("tms_5-2", yaml.safe_load(params_yaml))
        assert result == {
            "seed": {"values": [0, 1, 2]},
            "learning_rate": {"values": [0.001, 0.01]},
        }

    def test_experiment_specific_override(self):
        """Test experiment-specific parameters overriding global ones."""
        params_yaml = """
global:
  seed:
    values: [0, 1, 2]
  learning_rate:
    values: [0.001, 0.01]

tms_5-2:
  seed:
    values: [100, 200]
  n_components:
    values: [5, 10]
"""
        result = _get_experiment_sweep_params("tms_5-2", yaml.safe_load(params_yaml))
        assert result == {
            "seed": {"values": [100, 200]},  # Overridden
            "learning_rate": {"values": [0.001, 0.01]},  # From global
            "n_components": {"values": [5, 10]},  # Added by experiment
        }

    def test_nested_parameter_override(self):
        """Test overriding nested parameters."""
        params_yaml = """\
global:
  loss:
    faithfulness_weight:
      values: [0.1, 0.5]
    reconstruction_weight:
      values: [1.0, 2.0]

resid_mlp1:
  loss:
    faithfulness_weight:
      values: [1.0, 2.0]
"""

        result = _get_experiment_sweep_params("resid_mlp1", yaml.safe_load(params_yaml))
        assert result == {
            "loss": {
                "faithfulness_weight": {"values": [1.0, 2.0]},  # Overridden
                "reconstruction_weight": {"values": [1.0, 2.0]},  # From global
            }
        }

    def test_no_parameters_error(self):
        """Test error when no parameters are found."""
        params_yaml = """
some_other_key:
  foo: bar
"""
        with pytest.raises(ValueError, match="No sweep parameters found"):
            _get_experiment_sweep_params("tms_5-2", yaml.safe_load(params_yaml))

    def test_experiment_without_global(self):
        """Test loading experiment-specific params without global params."""
        params_yaml = """
tms_5-2:
  seed:
    values: [100, 200]
  n_components:
    values: [5, 10]
"""
        result = _get_experiment_sweep_params("tms_5-2", yaml.safe_load(params_yaml))
        assert result == {
            "seed": {"values": [100, 200]},
            "n_components": {"values": [5, 10]},
        }

    def test_complex_merge_scenario(self):
        """Test a complex scenario with multiple levels of nesting and overrides."""
        params_yaml = """
global:
  seed:
    values: [0, 1, 2]
  optimizer:
    learning_rate:
      values: [0.001, 0.01]
    momentum:
      values: [0.9]
  loss:
    faithfulness_weight:
      values: [0.1, 0.5]
    reconstruction_weight:
      values: [1.0]
    sparsity:
      lambda:
        values: [0.01]

tms_5-2:
  seed:
    values: [100]
  optimizer:
    learning_rate:
      values: [0.1]
    weight_decay:
      values: [0.0001]
  loss:
    sparsity:
      lambda:
        values: [0.1, 0.2]
"""
        result = _get_experiment_sweep_params("tms_5-2", yaml.safe_load(params_yaml))
        expected = {
            "seed": {"values": [100]},
            "optimizer": {
                "learning_rate": {"values": [0.1]},
                "momentum": {"values": [0.9]},
                "weight_decay": {"values": [0.0001]},
            },
            "loss": {
                "faithfulness_weight": {"values": [0.1, 0.5]},
                "reconstruction_weight": {"values": [1.0]},
                "sparsity": {"lambda": {"values": [0.1, 0.2]}},
            },
        }
        assert result == expected
