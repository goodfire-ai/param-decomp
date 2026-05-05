import torch

from param_decomp.utils.target_ci_solutions import (
    permute_to_dense,
    permute_to_identity_greedy,
    permute_to_identity_hungarian,
)


class TestPermutationFunctions:
    def test_simple_swap(self):
        """Test simple column swap."""
        # Columns 0 and 1 are swapped
        tensor = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        # Test Hungarian
        permuted_hungarian, indices_hungarian = permute_to_identity_hungarian(tensor)
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        assert torch.allclose(permuted_hungarian, expected)
        assert torch.equal(indices_hungarian, torch.tensor([1, 0, 2]))

        # Test Greedy
        permuted_greedy, indices_greedy = permute_to_identity_greedy(tensor)
        assert torch.allclose(permuted_greedy, expected)
        assert torch.equal(indices_greedy, torch.tensor([1, 0, 2]))

    def test_suboptimal_greedy(self):
        """Test case where greedy is suboptimal."""
        tensor = torch.tensor([[0.0, 1.0, 0.9], [0.0, 0.9, 0.5]])

        greedy_expected = torch.tensor([[1.0, 0.9, 0.0], [0.9, 0.5, 0.0]])

        hungarian_expected = torch.tensor([[0.9, 1.0, 0.0], [0.5, 0.9, 0.0]])

        # Test greedy
        permuted_greedy, _ = permute_to_identity_greedy(tensor)
        assert torch.allclose(permuted_greedy, greedy_expected)

        # Test Hungarian
        permuted_hungarian, _ = permute_to_identity_hungarian(tensor)
        assert torch.allclose(permuted_hungarian, hungarian_expected)

    def test_permute_to_dense(self):
        """Test permute_to_dense sorts columns by total mass."""
        tensor = torch.tensor(
            [
                [0.1, 0.8, 0.3],
                [0.2, 0.9, 0.4],
                [0.0, 0.7, 0.5],
            ]
        )
        # Column sums: [0.3, 2.4, 1.2]
        # Expected order: column 1, column 2, column 0

        permuted, indices = permute_to_dense(tensor)

        expected = torch.tensor(
            [
                [0.8, 0.3, 0.1],
                [0.9, 0.4, 0.2],
                [0.7, 0.5, 0.0],
            ]
        )
        assert torch.allclose(permuted, expected)
        assert torch.equal(indices, torch.tensor([1, 2, 0]))
