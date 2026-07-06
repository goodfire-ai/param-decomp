import math
from dataclasses import dataclass
from typing import Any

import torch

from param_decomp.metrics.smooth_l0_importance_minimality import (
    SmoothL0ImportanceMinimalityLoss,
    annealed_gamma,
    smooth_l0_importance_minimality_loss,
)
from param_decomp_config.losses import SmoothL0ImportanceMinimalityLossConfig


def _loss(
    ci_upper_leaky: dict[str, torch.Tensor],
    *,
    gamma: float,
    beta: float = 0.0,
    current_frac_of_training: float = 0.0,
    gamma_anneal_start_frac: float = 1.0,
    gamma_final: float | None = None,
    gamma_anneal_end_frac: float = 1.0,
) -> torch.Tensor:
    return smooth_l0_importance_minimality_loss(
        ci_upper_leaky=ci_upper_leaky,
        current_frac_of_training=current_frac_of_training,
        gamma=gamma,
        beta=beta,
        gamma_anneal_start_frac=gamma_anneal_start_frac,
        gamma_final=gamma_final,
        gamma_anneal_end_frac=gamma_anneal_end_frac,
    )


class TestSmoothL0Penalty:
    def test_phi_values(self: object) -> None:
        # phi(c) = c^2 / (c^2 + gamma^2): phi(0)=0, phi(gamma)=0.5, phi(c>>gamma)~1.
        ci = {"layer1": torch.tensor([[0.0, 1.0, 1e6]], dtype=torch.float32)}
        # gamma=1: [0, 1/2, ~1] -> sum 1.5
        result = _loss(ci, gamma=1.0)
        assert torch.allclose(result, torch.tensor(1.5), atol=1e-5)

    def test_saturates_to_active_count(self: object) -> None:
        # Large activations saturate phi to ~1, so the sum counts active components.
        ci = {"layer1": torch.tensor([[5.0, 5.0, 5.0, 5.0]], dtype=torch.float32)}
        result = _loss(ci, gamma=0.1)
        assert torch.allclose(result, torch.tensor(4.0), atol=1e-2)

    def test_flat_finite_gradient_at_zero(self: object) -> None:
        """The defining property: gradient is finite and ~0 at c=0 (no L_p cliff)."""
        ci = torch.zeros((1, 4), dtype=torch.float32, requires_grad=True)
        result = _loss({"layer1": ci}, gamma=0.5)
        result.backward()
        assert ci.grad is not None
        assert torch.isfinite(ci.grad).all()
        assert torch.allclose(ci.grad, torch.zeros_like(ci.grad))

    def test_gradient_bounded_and_peaks_near_gamma(self: object) -> None:
        # dphi/dc = 2 c gamma^2 / (c^2+gamma^2)^2, max 0.65/gamma at c=gamma/sqrt(3).
        gamma = 0.1
        # one position, 200 components, so finalize divides by n_examples=1 and each
        # component's grad is the raw dphi/dc.
        cs = torch.linspace(0.0, 1.0, 200, dtype=torch.float64).reshape(1, -1)
        cs.requires_grad_(True)
        out = _loss({"layer1": cs}, gamma=gamma)
        out.backward()
        grad = cs.grad
        assert grad is not None
        max_grad = grad.abs().max().item()
        expected_max = (3.0**1.5) / (8.0 * gamma)  # ~0.6495/gamma
        assert max_grad <= expected_max + 1e-3
        assert math.isclose(max_grad, expected_max, rel_tol=2e-2)
        argmax_c = cs.flatten()[grad.abs().argmax()].item()
        assert math.isclose(argmax_c, gamma / math.sqrt(3.0), abs_tol=0.01)

    def test_multiple_layers_aggregation(self: object) -> None:
        ci = {
            "layer1": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
            "layer2": torch.tensor([[1.0]], dtype=torch.float32),
        }
        # gamma=1: every phi=0.5 -> 0.5+0.5 + 0.5 = 1.5
        result = _loss(ci, gamma=1.0)
        assert torch.allclose(result, torch.tensor(1.5))

    def test_beta_logarithmic_penalty(self: object) -> None:
        ci = {"layer1": torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)}
        # gamma=1: phi=0.5 everywhere. per_component_sums=[1,1], n=2, mean=[0.5,0.5].
        # no_beta = sum(mean) = 1.0
        # beta term per comp = mean * log2(1 + sum) = 0.5 * log2(2) = 0.5 -> total 1.0
        loss_b0 = _loss(ci, gamma=1.0, beta=0.0)
        loss_b1 = _loss(ci, gamma=1.0, beta=1.0)
        assert torch.allclose(loss_b0, torch.tensor(1.0))
        assert torch.allclose(loss_b1, torch.tensor(1.0 + 1.0))
        assert loss_b1 > loss_b0

    def test_finite_for_extreme_values(self: object) -> None:
        for vals in ([[1e-12, 1e-12]], [[1e8, 1e8]]):
            ci = {"layer1": torch.tensor(vals, dtype=torch.float32)}
            result = _loss(ci, gamma=0.3, beta=1.0)
            assert torch.isfinite(result)
            assert result >= 0


class TestAnnealedGamma:
    def test_no_anneal_when_final_none(self: object) -> None:
        assert annealed_gamma(0.9, 1.0, 0.0, None, 1.0) == 1.0

    def test_before_start(self: object) -> None:
        assert annealed_gamma(0.1, 1.0, 0.5, 0.1, 1.0) == 1.0

    def test_during(self: object) -> None:
        # halfway through [0.0, 1.0]: 1.0 + (0.1 - 1.0) * 0.5 = 0.55
        assert math.isclose(annealed_gamma(0.5, 1.0, 0.0, 0.1, 1.0), 0.55)

    def test_after_end(self: object) -> None:
        assert annealed_gamma(0.9, 1.0, 0.0, 0.1, 0.5) == 0.1


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
    *, gamma: float, beta: float, device: str = "cpu"
) -> SmoothL0ImportanceMinimalityLoss:
    cfg = SmoothL0ImportanceMinimalityLossConfig(coeff=1.0, gamma=gamma, beta=beta)
    m = SmoothL0ImportanceMinimalityLoss(cfg)
    m.model = None  # pyright: ignore[reportAttributeAccessIssue]
    m.device = device
    m._bound = True
    m.reset()
    return m


class TestSmoothL0Update:
    def test_update_matches_closed_form(self: object) -> None:
        ci = {"layer1": torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)}
        m = _make_bound_metric(gamma=1.0, beta=1.0)
        ctx: Any = _FakeCtx(
            ci=_FakeCI(upper_leaky=ci, lower_leaky={}, pre_sigmoid={}),
            current_frac_of_training=0.0,
        )
        live = m.update(ctx)
        assert torch.allclose(live, torch.tensor(2.0, dtype=torch.float32))

    def test_update_matches_compute_single_rank(self: object) -> None:
        ci = {"layer1": torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32)}
        m = _make_bound_metric(gamma=0.7, beta=0.3)
        ctx: Any = _FakeCtx(
            ci=_FakeCI(upper_leaky=ci, lower_leaky={}, pre_sigmoid={}),
            current_frac_of_training=0.0,
        )
        live = m.update(ctx)
        evaluated = m.compute()
        assert isinstance(evaluated, dict)
        assert torch.allclose(live, evaluated["imp_min/SmoothL0ImportanceMinimalityLoss"])

    def test_update_returns_grad_tracking_scalar(self: object) -> None:
        ci = torch.tensor([[0.2, 0.4]], dtype=torch.float32, requires_grad=True)
        m = _make_bound_metric(gamma=0.5, beta=0.0)
        ctx: Any = _FakeCtx(
            ci=_FakeCI(upper_leaky={"layer1": ci}, lower_leaky={}, pre_sigmoid={}),
            current_frac_of_training=0.0,
        )
        live = m.update(ctx)
        live.backward()
        assert ci.grad is not None
        # beta=0, n=1: grad wrt c = dphi/dc = 2 c gamma^2 / (c^2 + gamma^2)^2
        g2 = 0.5**2
        expected = 2.0 * ci.detach() * g2 / (ci.detach() ** 2 + g2) ** 2
        assert torch.allclose(ci.grad, expected, atol=1e-6)

    def test_compute_logs_beta_and_no_beta(self: object) -> None:
        m = _make_bound_metric(gamma=1.0, beta=1.0)
        # phi=0.5 sums: per_component_sums=[1,1], n=2 -> mean=[0.5,0.5]
        m.per_component_sums = {"layer1": torch.tensor([1.0, 1.0])}
        m.n_examples = torch.tensor(2, dtype=torch.long)
        out = m.compute()
        assert isinstance(out, dict)
        assert set(out) == {
            "imp_min/SmoothL0ImportanceMinimalityLoss",
            "imp_min/SmoothL0ImportanceMinimalityLoss_no_beta",
        }
        expected_no_beta = 1.0
        expected_with_beta = 1.0 + (0.5 * math.log2(2) + 0.5 * math.log2(2))
        assert torch.allclose(
            out["imp_min/SmoothL0ImportanceMinimalityLoss_no_beta"],
            torch.tensor(expected_no_beta),
        )
        assert torch.allclose(
            out["imp_min/SmoothL0ImportanceMinimalityLoss"], torch.tensor(expected_with_beta)
        )
