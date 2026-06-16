import math
from dataclasses import dataclass
from typing import Any

import torch

from param_decomp.metrics.importance_minimality import (
    ImportanceMinimalityLoss,
    importance_minimality_loss,
)
from param_decomp_config.losses import ImportanceMinimalityLossConfig


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
        )
        assert torch.isfinite(result_large)


@dataclass
class _FakeCI:
    upper_leaky: dict[str, torch.Tensor]
    lower_leaky: dict[str, torch.Tensor]
    pre_sigmoid: dict[str, torch.Tensor]


@dataclass
class _FakeCtx:
    ci: _FakeCI
    current_frac_of_training: float


def _make_bound_metric(
    *, pnorm: float, beta: float, eps: float, device: str = "cpu"
) -> ImportanceMinimalityLoss:
    cfg = ImportanceMinimalityLossConfig(coeff=1.0, pnorm=pnorm, beta=beta, eps=eps)
    m = ImportanceMinimalityLoss(cfg)
    # Bypass Metric.bind (which wants a real ComponentModel). update/compute only
    # touch self.device and self.per_component_sums / self.n_examples (set by reset()).
    m.model = None  # pyright: ignore[reportAttributeAccessIssue]
    m.device = device
    m._bound = True
    m.reset()
    return m


class TestImportanceMinimalityLossUpdate:
    """Verify the single-rank Metric.update() path matches the closed-form formula.

    The distributed-mode equivalence is covered by
    `param_decomp_lab/tests/test_importance_minimality_distributed.py`.
    """

    def test_update_matches_closed_form(self: object) -> None:
        ci_upper_leaky = {
            "layer1": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        }
        # pnorm=1, eps=0, beta=1.0, n=2:
        # per_component_sums = [4, 6]; per_component_mean = [2, 3]
        # layer_loss = 2*(1 + log2(5)) + 3*(1 + log2(7))
        expected = 2.0 * (1 + math.log2(5)) + 3.0 * (1 + math.log2(7))

        m = _make_bound_metric(pnorm=1.0, beta=1.0, eps=0.0)
        ctx: Any = _FakeCtx(
            ci=_FakeCI(upper_leaky=ci_upper_leaky, lower_leaky={}, pre_sigmoid={}),
            current_frac_of_training=0.0,
        )
        live = m.update(ctx)
        assert torch.allclose(live, torch.tensor(expected, dtype=torch.float32))

    def test_update_matches_compute_single_rank(self: object) -> None:
        """In non-distributed runs, update() returns the same value as compute()."""
        ci_upper_leaky = {
            "layer1": torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32),
            "layer2": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32),
        }
        m = _make_bound_metric(pnorm=1.5, beta=0.3, eps=1e-6)
        ctx: Any = _FakeCtx(
            ci=_FakeCI(upper_leaky=ci_upper_leaky, lower_leaky={}, pre_sigmoid={}),
            current_frac_of_training=0.0,
        )
        live = m.update(ctx)
        evaluated = m.compute()
        assert isinstance(evaluated, dict)
        assert torch.allclose(live, evaluated["imp_min/ImportanceMinimalityLoss"])

    def test_update_returns_grad_tracking_scalar(self: object) -> None:
        """The live scalar must keep autograd connected to its CI inputs."""
        ci = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
        m = _make_bound_metric(pnorm=2.0, beta=0.0, eps=0.0)
        ctx: Any = _FakeCtx(
            ci=_FakeCI(upper_leaky={"layer1": ci}, lower_leaky={}, pre_sigmoid={}),
            current_frac_of_training=0.0,
        )
        live = m.update(ctx)
        live.backward()
        # beta=0, p=2: per_component_mean = sum_b ci[b]^2 / n; layer_loss summed over c
        # gradient wrt ci[b,c] = 2 * ci[b,c] / n
        n = 2
        expected_grad = 2.0 * ci.detach() / n
        assert ci.grad is not None
        assert torch.allclose(ci.grad, expected_grad)

    def test_compute_logs_beta_and_no_beta(self: object) -> None:
        """`compute()` emits the headline (beta-weighted) loss and the pure L_p value — a
        beta-independent sparsity proxy — both under fully-qualified `imp_min/` keys, so the
        pair groups together off the loss panel and the proxy doesn't read as a loss term."""
        cfg = ImportanceMinimalityLossConfig(coeff=1.0, pnorm=1.0, beta=1.0, eps=0.0)
        metric = ImportanceMinimalityLoss(cfg)
        # Bypass `bind` (no ComponentModel needed) — set the accumulator state directly.
        metric.device = "cpu"
        metric.per_component_sums = {"layer1": torch.tensor([4.0, 6.0])}
        metric.n_examples = torch.tensor(2, dtype=torch.long)

        out = metric.compute()
        assert isinstance(out, dict)
        assert set(out) == {
            "imp_min/ImportanceMinimalityLoss",
            "imp_min/ImportanceMinimalityLoss_no_beta",
        }

        # per_component_mean = [2, 3]; no_beta = sum = 5; beta=1 adds log2 term => larger.
        expected_no_beta = 5.0
        expected_with_beta = 2.0 * (1 + math.log2(5)) + 3.0 * (1 + math.log2(7))
        assert torch.allclose(
            out["imp_min/ImportanceMinimalityLoss_no_beta"], torch.tensor(expected_no_beta)
        )
        assert torch.allclose(
            out["imp_min/ImportanceMinimalityLoss"], torch.tensor(expected_with_beta)
        )
        assert (
            out["imp_min/ImportanceMinimalityLoss"]
            > out["imp_min/ImportanceMinimalityLoss_no_beta"]
        )
