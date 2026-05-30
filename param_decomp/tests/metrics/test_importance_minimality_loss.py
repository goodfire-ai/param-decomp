import math
from dataclasses import dataclass
from typing import Any

import torch

from param_decomp.metrics.importance_minimality import (
    FrequencyMinimalityLoss,
    FrequencyMinimalityLossConfig,
    ImportanceMinimalityLoss,
    ImportanceMinimalityLossConfig,
    finalize_freq_min,
    importance_minimality_loss,
    per_component_lp_sums,
)


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

    def test_bare_mean_over_examples(self: object) -> None:
        ci_upper_leaky = {
            "layer1": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        }
        # With pnorm=1 and eps=0:
        # per_component_sums = [1+3, 2+4] = [4, 6]; n_examples = 2
        # per_component_mean = [2, 3]; loss = sum = 5
        result = importance_minimality_loss(
            ci_upper_leaky=ci_upper_leaky,
            current_frac_of_training=0.0,
            pnorm=1.0,
            eps=0.0,
            p_anneal_start_frac=1.0,
            p_anneal_final_p=None,
            p_anneal_end_frac=1.0,
        )
        expected = torch.tensor(5.0)
        assert torch.allclose(result, expected)


def _freq(per_component_sums: dict[str, torch.Tensor], n: int, a_prime: int) -> torch.Tensor:
    return finalize_freq_min(
        per_component_sums=per_component_sums, n_examples=n, reference_token_count=a_prime
    )


class TestFrequencyMinimalityLoss:
    def test_closed_form(self: object) -> None:
        # sums = [4, 6], n = 2, a' = 8.  f = [2, 3]
        # loss = sum_c f_c * log2(1 + 8 * f_c)
        sums = {"layer1": torch.tensor([4.0, 6.0])}
        n, a_prime = 2, 8
        expected = 2.0 * math.log2(1 + 8 * 2.0) + 3.0 * math.log2(1 + 8 * 3.0)
        assert torch.allclose(_freq(sums, n, a_prime), torch.tensor(expected))

    def test_zero_frequency_zero_contribution(self: object) -> None:
        # A component that never fires (f=0) contributes exactly 0.
        sums = {"layer1": torch.tensor([0.0, 5.0])}
        n, a_prime = 4, 16
        contributions = (torch.tensor([0.0, 5.0]) / n) * torch.log2(
            1 + a_prime * (torch.tensor([0.0, 5.0]) / n)
        )
        assert contributions[0].item() == 0.0
        assert torch.allclose(_freq(sums, n, a_prime), contributions.sum())

    def test_batch_invariance(self: object) -> None:
        # Same per-token frequency at two different batch sizes => same L_freq.
        # Build sums so f_c is identical: doubling n doubles the sum.
        f = torch.tensor([0.1, 0.4, 0.7])
        a_prime = 1024
        small_n, large_n = 256, 4096
        sums_small = {"layer1": f * small_n}
        sums_large = {"layer1": f * large_n}
        loss_small = _freq(sums_small, small_n, a_prime)
        loss_large = _freq(sums_large, large_n, a_prime)
        assert torch.allclose(loss_small, loss_large)

    def test_a_prime_reproduces_old_rolled_log_term(self: object) -> None:
        # Old rolled imp-min was Σ_c mean_c + beta * mean_c * log2(1 + sum_c), with the
        # B*T implicit inside the log (sum_c = f_c * B*T). The split:
        #   imp = Σ_c mean_c
        #   freq = Σ_c f_c * log2(1 + a' * f_c)   with a' = B*T
        # so beta * freq == the old log term exactly. Here B*T = n_examples.
        torch.manual_seed(0)
        ci = {
            "layer1": torch.rand(8, 5),  # [B*T, C], n_examples = 8
            "layer2": torch.rand(8, 3),
        }
        pnorm, eps, beta = 2.0, 1e-12, 0.7
        sums, n = per_component_lp_sums(ci_upper_leaky=ci, pnorm=pnorm, eps=eps)

        # Old rolled value (mean + beta * mean * log2(1 + sum)).
        old = torch.zeros(())
        for layer_sums in sums.values():
            mean = layer_sums / n
            old = old + (mean + beta * mean * torch.log2(1 + layer_sums)).sum()

        imp = importance_minimality_loss(
            ci_upper_leaky=ci,
            current_frac_of_training=0.0,
            pnorm=pnorm,
            eps=eps,
            p_anneal_start_frac=1.0,
            p_anneal_final_p=None,
            p_anneal_end_frac=1.0,
        )
        freq = _freq(sums, n, a_prime=n)  # a' = B*T = n_examples
        assert torch.allclose(imp + beta * freq, old)


@dataclass
class _FakeCI:
    upper_leaky: dict[str, torch.Tensor]
    lower_leaky: dict[str, torch.Tensor]
    pre_sigmoid: dict[str, torch.Tensor]


@dataclass
class _FakeCtx:
    ci: _FakeCI
    current_frac_of_training: float


def _make_bound_imp_metric(*, pnorm: float, eps: float) -> ImportanceMinimalityLoss:
    cfg = ImportanceMinimalityLossConfig(coeff=1.0, pnorm=pnorm, eps=eps)
    m = ImportanceMinimalityLoss(cfg)
    # Bypass Metric.bind (which wants a real ComponentModel). update/compute only
    # touch self.device and self.per_component_sums / self.n_examples (set by reset()).
    m.model = None  # pyright: ignore[reportAttributeAccessIssue]
    m.device = "cpu"
    m._bound = True
    m.reset()
    return m


def _make_bound_freq_metric(
    *, pnorm: float, eps: float, reference_token_count: int
) -> FrequencyMinimalityLoss:
    cfg = FrequencyMinimalityLossConfig(
        coeff=1.0, pnorm=pnorm, eps=eps, reference_token_count=reference_token_count
    )
    m = FrequencyMinimalityLoss(cfg)
    m.model = None  # pyright: ignore[reportAttributeAccessIssue]
    m.device = "cpu"
    m._bound = True
    m.reset()
    return m


class TestMetricUpdate:
    """Verify the single-rank Metric.update() path matches the closed-form formula.

    The distributed-mode equivalence is covered by
    `param_decomp_lab/tests/test_importance_minimality_distributed.py`.
    """

    def test_imp_update_matches_closed_form(self: object) -> None:
        ci_upper_leaky = {
            "layer1": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        }
        # pnorm=1, eps=0, n=2: per_component_sums = [4, 6]; mean = [2, 3]; loss = 5
        m = _make_bound_imp_metric(pnorm=1.0, eps=0.0)
        ctx: Any = _FakeCtx(
            ci=_FakeCI(upper_leaky=ci_upper_leaky, lower_leaky={}, pre_sigmoid={}),
            current_frac_of_training=0.0,
        )
        live = m.update(ctx)
        assert torch.allclose(live, torch.tensor(5.0))

    def test_imp_update_matches_compute_single_rank(self: object) -> None:
        ci_upper_leaky = {
            "layer1": torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32),
            "layer2": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32),
        }
        m = _make_bound_imp_metric(pnorm=1.5, eps=1e-6)
        ctx: Any = _FakeCtx(
            ci=_FakeCI(upper_leaky=ci_upper_leaky, lower_leaky={}, pre_sigmoid={}),
            current_frac_of_training=0.0,
        )
        live = m.update(ctx)
        evaluated = m.compute()
        assert isinstance(evaluated, torch.Tensor)
        assert torch.allclose(live, evaluated)

    def test_imp_update_returns_grad_tracking_scalar(self: object) -> None:
        ci = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
        m = _make_bound_imp_metric(pnorm=2.0, eps=0.0)
        ctx: Any = _FakeCtx(
            ci=_FakeCI(upper_leaky={"layer1": ci}, lower_leaky={}, pre_sigmoid={}),
            current_frac_of_training=0.0,
        )
        live = m.update(ctx)
        live.backward()
        # p=2: mean_c = sum_b ci[b]^2 / n; gradient wrt ci[b,c] = 2 * ci[b,c] / n
        n = 2
        expected_grad = 2.0 * ci.detach() / n
        assert ci.grad is not None
        assert torch.allclose(ci.grad, expected_grad)

    def test_freq_update_matches_closed_form(self: object) -> None:
        ci_upper_leaky = {
            "layer1": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        }
        # pnorm=1, eps=0, n=2, a'=8: sums = [4, 6]; f = [2, 3]
        m = _make_bound_freq_metric(pnorm=1.0, eps=0.0, reference_token_count=8)
        ctx: Any = _FakeCtx(
            ci=_FakeCI(upper_leaky=ci_upper_leaky, lower_leaky={}, pre_sigmoid={}),
            current_frac_of_training=0.0,
        )
        live = m.update(ctx)
        expected = 2.0 * math.log2(1 + 8 * 2.0) + 3.0 * math.log2(1 + 8 * 3.0)
        assert torch.allclose(live, torch.tensor(expected, dtype=torch.float32))

    def test_freq_update_matches_compute_single_rank(self: object) -> None:
        ci_upper_leaky = {
            "layer1": torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32),
        }
        m = _make_bound_freq_metric(pnorm=1.0, eps=0.0, reference_token_count=16)
        ctx: Any = _FakeCtx(
            ci=_FakeCI(upper_leaky=ci_upper_leaky, lower_leaky={}, pre_sigmoid={}),
            current_frac_of_training=0.0,
        )
        live = m.update(ctx)
        evaluated = m.compute()
        assert isinstance(evaluated, torch.Tensor)
        assert torch.allclose(live, evaluated)
