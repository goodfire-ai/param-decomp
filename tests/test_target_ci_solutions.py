import torch

from param_decomp.utils.target_ci_solutions import (
    DenseCIPattern,
    IdentityCIPattern,
    TargetCISolution,
    compute_target_metrics,
)


class TestIdentityCIPattern:
    def test_perfect_identity_distance_zero(self):
        """Perfect identity matrix should have distance 0."""
        pattern = IdentityCIPattern(n_features=3)
        ci_array = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )
        assert pattern.distance_from(ci_array, tolerance=0.1) == 0

    def test_within_tolerance_identity(self):
        """Single off-diagonal element above tolerance."""
        pattern = IdentityCIPattern(n_features=3)
        ci_array = torch.tensor(
            [
                [0.95, 0.01, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.05],
                [0.0, 0.0, 0.99, 0.0],
            ]
        )
        assert pattern.distance_from(ci_array, tolerance=0.1) == 0

    def test_one_off_diagonal_error(self):
        """Single off-diagonal element above tolerance."""
        pattern = IdentityCIPattern(n_features=3)
        ci_array = torch.tensor(
            [
                [1.0, 0.2, 0.0, 0.0],  # 0.2 > 0.1 tolerance
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        assert pattern.distance_from(ci_array, tolerance=0.1) == 1

    def test_multiple_errors(self):
        """Multiple off-diagonal and diagonal errors."""
        pattern = IdentityCIPattern(n_features=3)
        ci_array = torch.tensor(
            [
                [0.7, 0.3, 0.0, 0.2],
                [0.2, 0.95, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],  # perfect row
            ]
        )
        assert pattern.distance_from(ci_array, tolerance=0.1) == 4


class TestDenseCIPattern:
    def test_exactly_k_columns_distance_zero(self):
        """When exactly k columns are active, distance is 0."""
        pattern = DenseCIPattern(k=2)
        ci_array = torch.tensor(
            [
                [0.9, 0.0, 0.3, 0.0],
                [0.6, 0.0, 0.4, 0.0],
                [0.7, 0.0, 1.0, 0.0],
            ]
        )
        # Columns 0 and 2 active, columns 1 and 3 inactive
        assert pattern.distance_from(ci_array, tolerance=0.1) == 0

    def test_one_excess_column(self):
        """One column beyond k has entries."""
        pattern = DenseCIPattern(k=1)
        ci_array = torch.tensor(
            [
                [0.9, 0.2, 0.0, 0.0],
                [0.6, 0.95, 0.0, 0.0],
            ]
        )
        # Columns 0 and 1 active, but k=1, so column 1 has 2 excess entries
        assert pattern.distance_from(ci_array, tolerance=0.1) == 2

    def test_multiple_excess_columns(self):
        """Multiple columns beyond k have entries."""
        pattern = DenseCIPattern(k=2)
        ci_array = torch.tensor(
            [
                [0.5, 0.4, 1.0, 0.2, 0.1],
                [0.6, 0.9, 0.4, 0.3, 0.2],
                [0.9, 0.6, 0.5, 0.4, 0.3],
            ]
        )
        # All 5 columns active, but k=2
        # Column 4 has values [0.1, 0.2, 0.3], only 2 are > 0.1
        # So excess entries: col 2 (3) + col 3 (3) + col 4 (2) = 8
        assert pattern.distance_from(ci_array, tolerance=0.1) == 8

    def test_no_columns_active(self):
        """When no columns have sufficient activations."""
        pattern = DenseCIPattern(k=2, min_entries=1)
        ci_array = torch.tensor([[0.5, 0.5, 0.0], [0.7, 0.8, 0.0]])
        # Both first k=2 columns missing 1 activation each = 2 errors
        assert pattern.distance_from(ci_array, tolerance=0.1) == 2

    def test_mixed_active_and_inactive_errors(self):
        """Mix of insufficient active columns and excess inactive columns."""
        pattern = DenseCIPattern(k=2, min_entries=1)
        ci_array = torch.tensor(
            [
                [0.95, 0.05, 0.15, 0.0],  # Col 0: sufficient, Col 2: violation
                [0.05, 0.5, 0.05, 0.0],  # Col 1: insufficient
            ]
        )
        # Col 1 missing 1 activation + Col 2 has 1 violation = 2 errors
        assert pattern.distance_from(ci_array, tolerance=0.1) == 2


class TestTargetCISolution:
    def test_combined_errors_from_modules(self):
        """Errors from multiple modules should sum."""
        solution = TargetCISolution(
            {"module1": IdentityCIPattern(n_features=2), "module2": DenseCIPattern(k=1)}
        )
        ci_arrays = {
            "module1": torch.tensor(
                [
                    [0.8, 0.2, 0.0],  # diagonal low, off-diag high
                    [0.0, 1.0, 0.0],
                ]
            ),
            "module2": torch.tensor(
                [
                    [0.9, 0.3, 0.0, 0.0],  # 2 columns active but k=1
                    [0.0, 0.1, 0.0, 0.0],
                ]
            ),
        }
        # module1: 1 diagonal + 1 off-diagonal = 2 errors
        # module2: column 0 has 1 entry (0.9), column 1 has 1 entry (0.4)
        # With k=1, we keep column 0, so column 1's 1 entry is excess
        # Total: 2 + 1 = 3 errors
        assert solution.distance_from(ci_arrays, tolerance=0.1) == 3
        # when we increase the tolerance, the distance should now be only 1
        assert solution.distance_from(ci_arrays, tolerance=0.2) == 1

    def test_expand_module_targets(self):
        """Test that expand_module_targets correctly matches patterns."""
        solution = TargetCISolution(
            {
                "layers.*.mlp_in": IdentityCIPattern(n_features=2),
                "layers.*.mlp_out": DenseCIPattern(k=1),
            }
        )

        module_names = ["layers.0.mlp_in", "layers.1.mlp_out", "other.module"]
        expanded = solution.expand_module_targets(module_names)

        assert len(expanded) == 2
        assert isinstance(expanded["layers.0.mlp_in"], IdentityCIPattern)
        assert isinstance(expanded["layers.1.mlp_out"], DenseCIPattern)
        assert "other.module" not in expanded

    def test_distance_from_with_patterns(self):
        """Test that distance_from works with pattern expansion on multiple modules."""
        solution = TargetCISolution(
            {
                "layers.*.mlp_in": IdentityCIPattern(n_features=2),
                "layers.*.mlp_out": DenseCIPattern(k=1),
            }
        )

        ci_arrays = {
            "layers.0.mlp_in": torch.tensor(
                [
                    [0.8, 0.2, 0.0],  # diagonal low, off-diag high
                    [0.0, 1.0, 0.0],
                ]
            ),
            "layers.0.mlp_out": torch.tensor(
                [
                    [0.5, 0.3, 0.0],  # 2 columns active but k=1
                    [0.0, 0.9, 0.0],
                ]
            ),
            "layers.1.mlp_in": torch.tensor(
                [
                    [1.0, 0.0],  # perfect identity
                    [0.0, 1.0],
                ]
            ),
            "layers.1.mlp_out": torch.tensor(
                [
                    [0.9, 0.0],  # only 1 column active, perfect for k=1
                    [0.7, 0.0],
                ]
            ),
        }

        # layers.0.mlp_in: 1 diagonal + 1 off-diagonal = 2 errors
        # layers.0.mlp_out: column 1 has 1 excess entry = 1 error
        # layers.1.mlp_in: perfect identity = 0 errors
        # layers.1.mlp_out: perfect dense = 0 errors
        # Total: 2 + 1 + 0 + 0 = 3 errors
        assert solution.distance_from(ci_arrays, tolerance=0.1) == 3

        metrics = compute_target_metrics(ci_arrays, solution, tolerance=0.1)

        # Check total errors
        assert metrics["total"] == 3
        # Check per-module errors
        assert metrics["layers.0.mlp_in"] == 2
        assert metrics["layers.0.mlp_out"] == 1
        assert metrics["layers.1.mlp_in"] == 0
        assert metrics["layers.1.mlp_out"] == 0
