"""Tests for sweep functionality with the named-field metric configs."""

from param_decomp.utils.run_utils import apply_nested_updates, generate_grid_combinations


class TestGenerateGridCombinations:
    def test_simple_sweep_single_loss(self):
        parameters = {
            "seed": {"values": [0, 1]},
            "loss_metrics": {
                "ImportanceMinimalityLoss": {"coeff": {"values": [0.1, 0.2]}},
            },
        }

        combinations = generate_grid_combinations(parameters)

        assert len(combinations) == 4  # 2 seeds × 2 coeffs
        assert {
            "seed": 0,
            "loss_metrics.ImportanceMinimalityLoss.coeff": 0.1,
        } in combinations
        assert {
            "seed": 1,
            "loss_metrics.ImportanceMinimalityLoss.coeff": 0.2,
        } in combinations

    def test_sweep_multiple_losses(self):
        parameters = {
            "seed": {"values": [0]},
            "loss_metrics": {
                "ImportanceMinimalityLoss": {"coeff": {"values": [0.1, 0.2]}},
                "FaithfulnessLoss": {"coeff": {"values": [0.5]}},
            },
        }

        combinations = generate_grid_combinations(parameters)

        assert len(combinations) == 2  # 2 × 1
        assert {
            "seed": 0,
            "loss_metrics.ImportanceMinimalityLoss.coeff": 0.1,
            "loss_metrics.FaithfulnessLoss.coeff": 0.5,
        } in combinations
        assert {
            "seed": 0,
            "loss_metrics.ImportanceMinimalityLoss.coeff": 0.2,
            "loss_metrics.FaithfulnessLoss.coeff": 0.5,
        } in combinations

    def test_sweep_multiple_params_per_loss(self):
        parameters = {
            "seed": {"values": [0]},
            "loss_metrics": {
                "ImportanceMinimalityLoss": {
                    "coeff": {"values": [0.1, 0.2]},
                    "pnorm": {"values": [1.0, 2.0]},
                },
            },
        }

        combinations = generate_grid_combinations(parameters)

        assert len(combinations) == 4  # 2 × 2
        expected = [
            {
                "seed": 0,
                "loss_metrics.ImportanceMinimalityLoss.coeff": 0.1,
                "loss_metrics.ImportanceMinimalityLoss.pnorm": 1.0,
            },
            {
                "seed": 0,
                "loss_metrics.ImportanceMinimalityLoss.coeff": 0.1,
                "loss_metrics.ImportanceMinimalityLoss.pnorm": 2.0,
            },
            {
                "seed": 0,
                "loss_metrics.ImportanceMinimalityLoss.coeff": 0.2,
                "loss_metrics.ImportanceMinimalityLoss.pnorm": 1.0,
            },
            {
                "seed": 0,
                "loss_metrics.ImportanceMinimalityLoss.coeff": 0.2,
                "loss_metrics.ImportanceMinimalityLoss.pnorm": 2.0,
            },
        ]
        for exp in expected:
            assert exp in combinations

    def test_mixed_regular_and_metric_sweeps(self):
        parameters = {
            "seed": {"values": [0, 1]},
            "lr": {"values": [0.001, 0.01]},
            "task_config": {"feature_probability": {"values": [0.05, 0.1]}},
            "loss_metrics": {
                "ImportanceMinimalityLoss": {"coeff": {"values": [0.1, 0.2]}},
            },
        }

        combinations = generate_grid_combinations(parameters)

        # 2 seeds × 2 lrs × 2 feature_probs × 2 coeffs = 16
        assert len(combinations) == 16
        assert {
            "seed": 0,
            "lr": 0.001,
            "task_config.feature_probability": 0.05,
            "loss_metrics.ImportanceMinimalityLoss.coeff": 0.1,
        } in combinations
        assert {
            "seed": 1,
            "lr": 0.01,
            "task_config.feature_probability": 0.1,
            "loss_metrics.ImportanceMinimalityLoss.coeff": 0.2,
        } in combinations

    def test_sweep_over_list_values(self):
        """Test sweeping a parameter whose values are themselves lists."""
        parameters = {
            "ci_fn_hidden_dims": {"values": [[8], [4, 3]]},
            "loss_metrics": {
                "ImportanceMinimalityLoss": {"coeff": {"values": [0.1, 0.2]}},
            },
        }
        combinations = generate_grid_combinations(parameters)
        assert len(combinations) == 4  # 2 × 2
        assert {
            "ci_fn_hidden_dims": [8],
            "loss_metrics.ImportanceMinimalityLoss.coeff": 0.1,
        } in combinations
        assert {
            "ci_fn_hidden_dims": [4, 3],
            "loss_metrics.ImportanceMinimalityLoss.coeff": 0.2,
        } in combinations


class TestApplyNestedUpdates:
    def test_update_existing_loss_config(self):
        base = {
            "seed": 0,
            "lr": 0.001,
            "loss_metrics": {
                "ImportanceMinimalityLoss": {"coeff": 0.5, "pnorm": 1.0, "eps": 1e-12},
                "FaithfulnessLoss": {"coeff": 1.0},
            },
        }

        updates = {
            "seed": 42,
            "loss_metrics.ImportanceMinimalityLoss.coeff": 0.1,
            "loss_metrics.ImportanceMinimalityLoss.pnorm": 2.0,
        }

        result = apply_nested_updates(base, updates)

        assert result == {
            "seed": 42,
            "lr": 0.001,
            "loss_metrics": {
                "ImportanceMinimalityLoss": {
                    "coeff": 0.1,  # Updated
                    "pnorm": 2.0,  # Updated
                    "eps": 1e-12,  # Preserved
                },
                "FaithfulnessLoss": {"coeff": 1.0},  # Preserved
            },
        }

    def test_add_new_loss_config(self):
        base = {
            "seed": 0,
            "loss_metrics": {"FaithfulnessLoss": {"coeff": 1.0}},
        }

        updates = {
            "loss_metrics.ImportanceMinimalityLoss.coeff": 0.1,
            "loss_metrics.ImportanceMinimalityLoss.pnorm": 1.0,
        }

        result = apply_nested_updates(base, updates)

        assert result == {
            "seed": 0,
            "loss_metrics": {
                "FaithfulnessLoss": {"coeff": 1.0},  # Preserved
                "ImportanceMinimalityLoss": {"coeff": 0.1, "pnorm": 1.0},  # Added
            },
        }

    def test_multiple_losses_overlap(self):
        base = {
            "seed": 0,
            "loss_metrics": {
                "ImportanceMinimalityLoss": {"coeff": 0.5, "pnorm": 1.0, "eps": 1e-12},
                "FaithfulnessLoss": {"coeff": 1.0},
                "StochasticReconLoss": {"coeff": 0.2},
            },
        }

        updates = {
            "seed": 42,
            "loss_metrics.ImportanceMinimalityLoss.coeff": 0.1,
            "loss_metrics.ImportanceMinimalityLoss.pnorm": 2.0,
            "loss_metrics.CIMaskedReconLoss.coeff": 0.3,
        }

        result = apply_nested_updates(base, updates)

        assert result == {
            "seed": 42,
            "loss_metrics": {
                "ImportanceMinimalityLoss": {"coeff": 0.1, "pnorm": 2.0, "eps": 1e-12},
                "FaithfulnessLoss": {"coeff": 1.0},
                "StochasticReconLoss": {"coeff": 0.2},
                "CIMaskedReconLoss": {"coeff": 0.3},
            },
        }

    def test_regular_nested_updates(self):
        base = {"config": {"param1": 1, "param2": 2}, "other": 3}
        updates = {"config.param1": 10, "config.param3": 30}

        result = apply_nested_updates(base, updates)

        assert result == {"config": {"param1": 10, "param2": 2, "param3": 30}, "other": 3}

    def test_create_nested_structures(self):
        base = {"existing": 1}
        updates = {"new.nested.value": 42}

        result = apply_nested_updates(base, updates)

        assert result == {"existing": 1, "new": {"nested": {"value": 42}}}

    def test_update_nested_optimizer_config(self):
        base = {
            "pd": {
                "components_optimizer": {
                    "lr_schedule": {"start_val": 1e-3, "fn_type": "constant"},
                },
                "ci_fn_optimizer": {
                    "lr_schedule": {"start_val": 1e-4, "fn_type": "constant"},
                },
            }
        }
        updates = {
            "pd.components_optimizer.lr_schedule.start_val": 2e-3,
            "pd.ci_fn_optimizer.weight_decay": 0.01,
        }

        result = apply_nested_updates(base, updates)

        assert result == {
            "pd": {
                "components_optimizer": {
                    "lr_schedule": {"start_val": 2e-3, "fn_type": "constant"},
                },
                "ci_fn_optimizer": {
                    "lr_schedule": {"start_val": 1e-4, "fn_type": "constant"},
                    "weight_decay": 0.01,
                },
            }
        }


class TestInvalidConfigurations:
    def test_leaf_without_values_dict(self):
        parameters = {
            "seed": {"values": [0, 1]},
            "lr": 0.001,  # Should be {"values": [0.001]}
        }

        try:
            generate_grid_combinations(parameters)
            raise AssertionError("Expected ValueError for leaf without values dict")
        except ValueError as e:
            assert 'must be {"values": [...]}' in str(e)

    def test_nested_leaf_without_values_dict(self):
        parameters = {
            "seed": {"values": [0]},
            "task_config": {
                "feature_probability": 0.05,  # Should be {"values": [0.05]}
            },
        }

        try:
            generate_grid_combinations(parameters)
            raise AssertionError("Expected ValueError for nested leaf without values dict")
        except ValueError as e:
            assert 'must be {"values": [...]}' in str(e)

    def test_metric_field_without_values_dict(self):
        parameters = {
            "seed": {"values": [0]},
            "loss_metrics": {
                "ImportanceMinimalityLoss": {
                    "coeff": 0.1,  # Should be {"values": [0.1]}
                },
            },
        }

        try:
            generate_grid_combinations(parameters)
            raise AssertionError("Expected ValueError for field without values dict")
        except ValueError as e:
            assert 'must be {"values": [...]}' in str(e)

    def test_empty_values_list(self):
        parameters = {
            "seed": {"values": []},
        }

        combinations = generate_grid_combinations(parameters)
        assert len(combinations) == 0

    def test_empty_parameters_dict(self):
        parameters = {}

        combinations = generate_grid_combinations(parameters)
        assert len(combinations) == 1
        assert combinations[0] == {}
