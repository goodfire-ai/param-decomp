import torch

from spd.metrics import importance_minimality_loss


class TestImportanceMinimalityLoss:
    def test_basic_l1_norm(self: object) -> None:
        # L1 norm: sum of absolute values (already positive with upper_leaky)
        ci_upper_leaky = {
            "layer1": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            "layer2": torch.tensor([[0.5, 1.5]], dtype=torch.float32),
        }
        # With eps=0, p=1, no annealing:
        # layer1: per_component_mean = [1, 2, 3], sum = 6
        # layer2: per_component_mean = [0.5, 1.5], sum = 2
        # total = 8
        result = importance_minimality_loss(
            ci_upper_leaky=ci_upper_leaky,
            current_frac_of_training=0.0,
            pnorm=1.0,
            beta=0.0,
            eps=0.0,
            p_anneal_start_frac=1.0,
            p_anneal_final_p=None,
            p_anneal_end_frac=1.0,
            coeff_warmup_frac=0.0,
            coeff_peak_multiplier=1.0,
            coeff_anneal_start_frac=1.0,
            coeff_anneal_end_frac=1.0,
        )
        expected = torch.tensor(8.0)
        assert torch.allclose(result, expected)

    def test_basic_l2_norm(self: object) -> None:
        ci_upper_leaky = {
            "layer1": torch.tensor([[2.0, 3.0]], dtype=torch.float32),
        }
        # L2: per_component_mean = [4, 9], sum = 13
        result = importance_minimality_loss(
            ci_upper_leaky=ci_upper_leaky,
            current_frac_of_training=0.0,
            pnorm=2.0,
            beta=0.0,
            eps=0.0,
            p_anneal_start_frac=1.0,
            p_anneal_final_p=None,
            p_anneal_end_frac=1.0,
            coeff_warmup_frac=0.0,
            coeff_peak_multiplier=1.0,
            coeff_anneal_start_frac=1.0,
            coeff_anneal_end_frac=1.0,
        )
        expected = torch.tensor(13.0)
        assert torch.allclose(result, expected)

    def test_epsilon_stability(self: object) -> None:
        # Verify epsilon prevents issues with zero values
        ci_upper_leaky = {
            "layer1": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        }
        eps = 1e-6
        # With p=0.5: per_component_mean = [(0+eps)^0.5, (1+eps)^0.5]
        result = importance_minimality_loss(
            ci_upper_leaky=ci_upper_leaky,
            current_frac_of_training=0.0,
            pnorm=0.5,
            beta=0.0,
            eps=eps,
            p_anneal_start_frac=1.0,
            p_anneal_final_p=None,
            p_anneal_end_frac=1.0,
            coeff_warmup_frac=0.0,
            coeff_peak_multiplier=1.0,
            coeff_anneal_start_frac=1.0,
            coeff_anneal_end_frac=1.0,
        )
        expected = (0.0 + eps) ** 0.5 + (1.0 + eps) ** 0.5
        assert torch.allclose(result, torch.tensor(expected))

    def test_p_annealing_before_start(self: object) -> None:
        # Before annealing starts, should use initial p
        ci_upper_leaky = {"layer1": torch.tensor([[2.0]], dtype=torch.float32)}
        result = importance_minimality_loss(
            ci_upper_leaky=ci_upper_leaky,
            current_frac_of_training=0.3,
            pnorm=2.0,
            beta=0.0,
            eps=0.0,
            p_anneal_start_frac=0.5,
            p_anneal_final_p=1.0,
            p_anneal_end_frac=1.0,
            coeff_warmup_frac=0.0,
            coeff_peak_multiplier=1.0,
            coeff_anneal_start_frac=1.0,
            coeff_anneal_end_frac=1.0,
        )
        # Should use p=2: 2^2 = 4
        expected = torch.tensor(4.0)
        assert torch.allclose(result, expected)

    def test_p_annealing_during(self: object) -> None:
        # During annealing, should interpolate
        ci_upper_leaky = {"layer1": torch.tensor([[2.0]], dtype=torch.float32)}
        # At 50% through annealing (0.25 between 0.0 and 0.5)
        # p should be: 2.0 + (1.0 - 2.0) * 0.5 = 1.5
        result = importance_minimality_loss(
            ci_upper_leaky=ci_upper_leaky,
            current_frac_of_training=0.25,
            pnorm=2.0,
            beta=0.0,
            eps=0.0,
            p_anneal_start_frac=0.0,
            p_anneal_final_p=1.0,
            p_anneal_end_frac=0.5,
            coeff_warmup_frac=0.0,
            coeff_peak_multiplier=1.0,
            coeff_anneal_start_frac=1.0,
            coeff_anneal_end_frac=1.0,
        )
        # 2^1.5 = 2.828...
        expected = torch.tensor(2.0**1.5)
        assert torch.allclose(result, expected)

    def test_p_annealing_after_end(self: object) -> None:
        # After annealing ends, should use final p
        ci_upper_leaky = {"layer1": torch.tensor([[2.0]], dtype=torch.float32)}
        result = importance_minimality_loss(
            ci_upper_leaky=ci_upper_leaky,
            current_frac_of_training=0.9,
            pnorm=2.0,
            beta=0.0,
            eps=0.0,
            p_anneal_start_frac=0.0,
            p_anneal_final_p=1.0,
            p_anneal_end_frac=0.5,
            coeff_warmup_frac=0.0,
            coeff_peak_multiplier=1.0,
            coeff_anneal_start_frac=1.0,
            coeff_anneal_end_frac=1.0,
        )
        # Should use p=1: 2^1 = 2
        expected = torch.tensor(2.0)
        assert torch.allclose(result, expected)

    def test_no_annealing_when_final_p_none(self: object) -> None:
        # When p_anneal_final_p is None, should always use initial p
        ci_upper_leaky = {"layer1": torch.tensor([[2.0]], dtype=torch.float32)}
        result = importance_minimality_loss(
            ci_upper_leaky=ci_upper_leaky,
            current_frac_of_training=0.9,
            pnorm=2.0,
            beta=0.0,
            eps=0.0,
            p_anneal_start_frac=0.0,
            p_anneal_final_p=None,
            p_anneal_end_frac=0.5,
            coeff_warmup_frac=0.0,
            coeff_peak_multiplier=1.0,
            coeff_anneal_start_frac=1.0,
            coeff_anneal_end_frac=1.0,
        )
        # Should use p=2: 2^2 = 4
        expected = torch.tensor(4.0)
        assert torch.allclose(result, expected)

    def test_multiple_layers_aggregation(self: object) -> None:
        # Test that losses from multiple layers are correctly summed
        ci_upper_leaky = {
            "layer1": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
            "layer2": torch.tensor([[2.0, 2.0]], dtype=torch.float32),
        }
        result = importance_minimality_loss(
            ci_upper_leaky=ci_upper_leaky,
            current_frac_of_training=0.0,
            pnorm=1.0,
            beta=0.0,
            eps=0.0,
            p_anneal_start_frac=1.0,
            p_anneal_final_p=None,
            p_anneal_end_frac=1.0,
            coeff_warmup_frac=0.0,
            coeff_peak_multiplier=1.0,
            coeff_anneal_start_frac=1.0,
            coeff_anneal_end_frac=1.0,
        )
        # layer1: per_component_mean = [1, 1], sum = 2
        # layer2: per_component_mean = [2, 2], sum = 4
        # total = 6
        expected = torch.tensor(6.0)
        assert torch.allclose(result, expected)

    def test_beta_zero_simple_sum(self: object) -> None:
        ci_upper_leaky = {
            "layer1": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        }
        # With pnorm=1 and eps=0:
        # per_component_sums = [1+3, 2+4] = [4, 6]
        # n_examples = 2
        # per_component_mean = [2, 3]
        # beta=0 => layer_loss = sum(per_component_mean) = 5
        result = importance_minimality_loss(
            ci_upper_leaky=ci_upper_leaky,
            current_frac_of_training=0.0,
            pnorm=1.0,
            beta=0.0,
            eps=0.0,
            p_anneal_start_frac=1.0,
            p_anneal_final_p=None,
            p_anneal_end_frac=1.0,
            coeff_warmup_frac=0.0,
            coeff_peak_multiplier=1.0,
            coeff_anneal_start_frac=1.0,
            coeff_anneal_end_frac=1.0,
        )
        expected = torch.tensor(5.0)
        assert torch.allclose(result, expected)

    def test_beta_logarithmic_penalty(self: object) -> None:
        """Verify the logarithmic penalty with beta > 0 works correctly.

        Tests:
        1. Manual calculation verification
        2. beta > 0 produces larger loss than beta = 0
        3. Penalty is finite for edge cases (small/large values)
        """
        import math

        ci_upper_leaky = {
            "layer1": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        }
        # With pnorm=1, eps=0, beta=1.0:
        # per_component_sums = [1+3, 2+4] = [4, 6]
        # n_examples = 2
        # per_component_mean = [2, 3]
        # layer_loss = sum(per_component_mean * (1 + beta * log2(1 + layer_sums)))
        #            = 2 * (1 + log2(5)) + 3 * (1 + log2(7))
        expected_beta_1 = 2.0 * (1 + math.log2(5)) + 3.0 * (1 + math.log2(7))
        # beta=0 => layer_loss = sum(per_component_mean) = 5
        expected_beta_0 = 5.0

        loss_beta_0 = importance_minimality_loss(
            ci_upper_leaky=ci_upper_leaky,
            current_frac_of_training=0.0,
            pnorm=1.0,
            beta=0.0,
            eps=0.0,
            p_anneal_start_frac=1.0,
            p_anneal_final_p=None,
            p_anneal_end_frac=1.0,
            coeff_warmup_frac=0.0,
            coeff_peak_multiplier=1.0,
            coeff_anneal_start_frac=1.0,
            coeff_anneal_end_frac=1.0,
        )
        loss_beta_1 = importance_minimality_loss(
            ci_upper_leaky=ci_upper_leaky,
            current_frac_of_training=0.0,
            pnorm=1.0,
            beta=1.0,
            eps=0.0,
            p_anneal_start_frac=1.0,
            p_anneal_final_p=None,
            p_anneal_end_frac=1.0,
            coeff_warmup_frac=0.0,
            coeff_peak_multiplier=1.0,
            coeff_anneal_start_frac=1.0,
            coeff_anneal_end_frac=1.0,
        )

        assert torch.allclose(loss_beta_0, torch.tensor(expected_beta_0))
        assert torch.allclose(loss_beta_1, torch.tensor(expected_beta_1))
        assert loss_beta_1 > loss_beta_0

    def test_beta_edge_cases(self: object) -> None:
        """Verify the penalty is finite for edge cases."""
        # Very small values
        ci_small = {"layer1": torch.tensor([[1e-10, 1e-10]], dtype=torch.float32)}
        result_small = importance_minimality_loss(
            ci_upper_leaky=ci_small,
            current_frac_of_training=0.0,
            pnorm=1.0,
            beta=1.0,
            eps=0.0,
            p_anneal_start_frac=1.0,
            p_anneal_final_p=None,
            p_anneal_end_frac=1.0,
            coeff_warmup_frac=0.0,
            coeff_peak_multiplier=1.0,
            coeff_anneal_start_frac=1.0,
            coeff_anneal_end_frac=1.0,
        )
        assert torch.isfinite(result_small)
        assert result_small >= 0

        # Very large values
        ci_large = {"layer1": torch.tensor([[1e6, 1e6]], dtype=torch.float32)}
        result_large = importance_minimality_loss(
            ci_upper_leaky=ci_large,
            current_frac_of_training=0.0,
            pnorm=1.0,
            beta=1.0,
            eps=0.0,
            p_anneal_start_frac=1.0,
            p_anneal_final_p=None,
            p_anneal_end_frac=1.0,
            coeff_warmup_frac=0.0,
            coeff_peak_multiplier=1.0,
            coeff_anneal_start_frac=1.0,
            coeff_anneal_end_frac=1.0,
        )
        assert torch.isfinite(result_large)
