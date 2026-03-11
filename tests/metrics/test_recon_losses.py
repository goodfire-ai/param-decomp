"""Sanity checks for stochastic, CI, PGD, and persistent PGD reconstruction losses."""

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from spd.configs import PGDConfig
from spd.metrics import ci_masked_recon_loss, pgd_recon_loss, stochastic_recon_loss
from spd.metrics.hidden_acts_recon_loss import (
    CIHiddenActsReconLoss,
    _sum_per_module_mse,
    calc_hidden_acts_mse,
)
from spd.metrics.ppgd_eval_losses import PPGDReconEval
from spd.models.component_model import CIOutputs, ComponentModel
from spd.models.components import make_mask_infos
from spd.persistent_pgd import PPGDSources, get_ppgd_mask_infos
from tests.metrics.fixtures import (
    OneLayerLinearModel,
    TwoLayerLinearModel,
    make_one_layer_component_model,
    make_two_layer_component_model,
)

ReconLossFn = Callable[[ComponentModel, Tensor, Tensor, dict[str, Tensor]], Tensor]


def _stochastic(
    model: ComponentModel,
    batch: Tensor,
    target_out: Tensor,
    ci: dict[str, Tensor],
) -> Tensor:
    return stochastic_recon_loss(
        model=model,
        sampling="continuous",
        n_mask_samples=4,
        output_loss_type="mse",
        batch=batch,
        target_out=target_out,
        ci=ci,
        weight_deltas=None,
    )


def _ci(
    model: ComponentModel,
    batch: Tensor,
    target_out: Tensor,
    ci: dict[str, Tensor],
) -> Tensor:
    return ci_masked_recon_loss(
        model=model,
        output_loss_type="mse",
        batch=batch,
        target_out=target_out,
        ci=ci,
    )


def _pgd(
    model: ComponentModel,
    batch: Tensor,
    target_out: Tensor,
    ci: dict[str, Tensor],
) -> Tensor:
    return pgd_recon_loss(
        model=model,
        batch=batch,
        target_out=target_out,
        output_loss_type="mse",
        ci=ci,
        weight_deltas=None,
        pgd_config=PGDConfig(
            init="random",
            step_size=0.1,
            n_steps=5,
            mask_scope="unique_per_datapoint",
        ),
    )


LOSS_FNS = [_stochastic, _ci, _pgd]


@pytest.mark.parametrize("loss_fn", LOSS_FNS, ids=["stochastic", "ci", "pgd"])
class TestOutputReconLoss:
    def test_perfect_init_zero_recon(self, loss_fn: ReconLossFn) -> None:
        """V=W.T, U=I with CI=1 → all masks are 1 → output matches target → loss ≈ 0."""
        torch.manual_seed(42)
        d = 3
        weight = torch.randn(d, d)
        model = make_one_layer_component_model(weight=weight, C=d)

        assert isinstance(model.target_model, OneLayerLinearModel)
        target_weight = model.target_model.fc.weight.data
        with torch.no_grad():
            model.components["fc"].V.copy_(target_weight.T)
            model.components["fc"].U.copy_(torch.eye(d))

        batch = torch.randn(4, d)
        target_out = model.target_model(batch)
        ci = {"fc": torch.ones(4, d)}

        loss = loss_fn(model, batch, target_out, ci)
        assert loss < 1e-5, f"Expected ~0 loss with perfect init, got {loss}"

    def test_random_init_high_recon(self, loss_fn: ReconLossFn) -> None:
        """Random V and U should give substantially nonzero recon loss."""
        torch.manual_seed(42)
        d = 3
        weight = torch.randn(d, d)
        model = make_one_layer_component_model(weight=weight, C=d)

        batch = torch.randn(4, d)
        target_out = model.target_model(batch)
        ci = {"fc": torch.ones(4, d)}

        loss = loss_fn(model, batch, target_out, ci)
        assert loss > 0.01, f"Expected high loss with random init, got {loss}"


def test_output_recon_manual_calculation() -> None:
    """Verify CI-masked output recon loss matches a manual forward pass computation."""
    torch.manual_seed(42)

    fc_weight = torch.randn(2, 2)
    model = make_one_layer_component_model(weight=fc_weight)

    V = model.components["fc"].V
    U = model.components["fc"].U

    batch = torch.randn(1, 2)
    target_out = torch.randn(1, 2)
    ci = {"fc": torch.tensor([[0.8]])}

    # Manual: component_acts = batch @ V, masked = acts * ci, out = masked @ U
    out = (batch @ V * ci["fc"]) @ U
    expected_loss = torch.nn.functional.mse_loss(out, target_out, reduction="sum") / out.numel()

    actual_loss = ci_masked_recon_loss(
        model=model,
        output_loss_type="mse",
        batch=batch,
        target_out=target_out,
        ci=ci,
    )

    assert torch.allclose(actual_loss, expected_loss, rtol=1e-5), (
        f"Expected {expected_loss}, got {actual_loss}"
    )


def test_per_module_recon_manual_calculation() -> None:
    """Verify per-module recon loss matches manual computation for a two-layer model."""
    torch.manual_seed(42)

    fc1_weight = torch.randn(3, 2)
    fc2_weight = torch.randn(2, 3)
    model = make_two_layer_component_model(weight1=fc1_weight, weight2=fc2_weight)

    V1, U1 = model.components["fc1"].V, model.components["fc1"].U
    V2, U2 = model.components["fc2"].V, model.components["fc2"].U

    batch = torch.randn(1, 2)
    ci = {"fc1": torch.tensor([[0.8]]), "fc2": torch.tensor([[0.7]])}

    # Target activations (output of each layer through the target model)
    assert isinstance(model.target_model, TwoLayerLinearModel)
    target_fc1 = batch @ model.target_model.fc1.weight.data.T
    target_fc2 = target_fc1 @ model.target_model.fc2.weight.data.T

    # Component activations with CI as masks (fc2 input is fc1's component output, not target)
    comp_fc1 = batch @ (V1 * ci["fc1"]) @ U1
    comp_fc2 = comp_fc1 @ (V2 * ci["fc2"]) @ U2

    expected_fc1_mse = F.mse_loss(comp_fc1, target_fc1, reduction="sum")
    expected_fc2_mse = F.mse_loss(comp_fc2, target_fc2, reduction="sum")
    expected_total = (expected_fc1_mse + expected_fc2_mse) / (
        target_fc1.numel() + target_fc2.numel()
    )

    # Actual computation
    target_acts = model(batch, cache_type="output").cache
    mask_infos = make_mask_infos(ci, weight_deltas_and_masks=None)
    per_module, _ = calc_hidden_acts_mse(model, batch, mask_infos, target_acts)
    sum_mse, n_examples = _sum_per_module_mse(per_module)
    actual_total = sum_mse / n_examples

    assert torch.allclose(actual_total, expected_total, rtol=1e-5)
    fc1_mse, _ = per_module["fc1"]
    assert torch.allclose(fc1_mse, expected_fc1_mse, rtol=1e-5)
    fc2_mse, _ = per_module["fc2"]
    assert torch.allclose(fc2_mse, expected_fc2_mse, rtol=1e-5)


def test_per_module_recon_metric_keys() -> None:
    """CIHiddenActsReconLoss.compute() returns per-module + total keys."""
    torch.manual_seed(42)

    model = make_two_layer_component_model(weight1=torch.randn(3, 2), weight2=torch.randn(2, 3))
    batch = torch.randn(2, 2)

    target_output = model(batch, cache_type="input")
    ci = model.calc_causal_importances(pre_weight_acts=target_output.cache, sampling="continuous")

    metric = CIHiddenActsReconLoss(model=model, device="cpu")
    metric.update(batch=batch, ci=ci)
    result = metric.compute()

    assert set(result.keys()) == {
        "CIHiddenActsReconLoss",
        "CIHiddenActsReconLoss/fc1",
        "CIHiddenActsReconLoss/fc2",
    }
    for v in result.values():
        assert v.item() >= 0


def _make_ci_outputs(ci: dict[str, Tensor]) -> CIOutputs:
    return CIOutputs(
        lower_leaky=ci,
        upper_leaky=ci,
        pre_sigmoid={k: torch.ones_like(v) * 10 for k, v in ci.items()},
    )


def test_ppgd_recon_eval_metric_keys() -> None:
    """PPGDReconEval.compute() returns hidden_acts (total + per-module) and output_recon keys."""
    torch.manual_seed(42)

    model = make_two_layer_component_model(weight1=torch.randn(3, 2), weight2=torch.randn(2, 3))
    batch = torch.randn(2, 2)
    target_out = model.target_model(batch)
    ci = {"fc1": torch.ones(2, 1), "fc2": torch.ones(2, 1)}
    sources: PPGDSources = {k: torch.zeros(1, v.shape[-1]) for k, v in ci.items()}

    metric = PPGDReconEval(
        model=model,
        device="cpu",
        effective_sources=sources,
        use_delta_component=False,
        output_loss_type="mse",
        metric_name="my_ppgd",
    )
    metric.update(
        batch=batch,
        ci=_make_ci_outputs(ci),
        weight_deltas={},
        target_out=target_out,
    )
    result = metric.compute()

    assert set(result.keys()) == {
        "my_ppgd/hidden_acts",
        "my_ppgd/hidden_acts/fc1",
        "my_ppgd/hidden_acts/fc2",
        "my_ppgd/output_recon",
    }
    for v in result.values():
        assert v.item() >= 0


def test_ppgd_recon_eval_manual_calculation() -> None:
    """Verify PPGD hidden-acts MSE and output recon match hand-computed values."""
    torch.manual_seed(42)

    fc1_weight = torch.randn(3, 2)
    fc2_weight = torch.randn(2, 3)
    model = make_two_layer_component_model(weight1=fc1_weight, weight2=fc2_weight)

    V1, U1 = model.components["fc1"].V, model.components["fc1"].U
    V2, U2 = model.components["fc2"].V, model.components["fc2"].U

    batch = torch.randn(1, 2)
    ci = {"fc1": torch.tensor([[0.8]]), "fc2": torch.tensor([[0.7]])}
    adv_sources: PPGDSources = {"fc1": torch.tensor([[0.3]]), "fc2": torch.tensor([[0.5]])}

    # mask = ci + (1 - ci) * source
    mask_fc1 = ci["fc1"] + (1 - ci["fc1"]) * adv_sources["fc1"]  # 0.8 + 0.2*0.3 = 0.86
    mask_fc2 = ci["fc2"] + (1 - ci["fc2"]) * adv_sources["fc2"]  # 0.7 + 0.3*0.5 = 0.85

    # Target activations
    assert isinstance(model.target_model, TwoLayerLinearModel)
    target_fc1 = batch @ model.target_model.fc1.weight.data.T
    target_fc2 = target_fc1 @ model.target_model.fc2.weight.data.T

    # Component activations with PPGD masks
    comp_fc1 = batch @ (V1 * mask_fc1) @ U1
    comp_fc2 = comp_fc1 @ (V2 * mask_fc2) @ U2

    expected_fc1_mse = F.mse_loss(comp_fc1, target_fc1, reduction="sum")
    expected_fc2_mse = F.mse_loss(comp_fc2, target_fc2, reduction="sum")
    expected_hidden_total = (expected_fc1_mse + expected_fc2_mse) / (
        target_fc1.numel() + target_fc2.numel()
    )
    expected_output_recon = ((comp_fc2 - target_fc2) ** 2).sum() / comp_fc2.numel()

    # Actual computation via get_ppgd_mask_infos + calc_hidden_acts_mse
    target_acts = model(batch, cache_type="output").cache
    mask_infos = get_ppgd_mask_infos(
        ci=ci,
        weight_deltas=None,
        ppgd_sources=adv_sources,
        routing_masks="all",
        batch_dims=(1,),
    )
    per_module, comp_output = calc_hidden_acts_mse(model, batch, mask_infos, target_acts)
    sum_mse, n_examples = _sum_per_module_mse(per_module)
    actual_hidden_total = sum_mse / n_examples
    actual_output_recon = ((comp_output - target_fc2) ** 2).sum() / comp_output.numel()

    assert torch.allclose(actual_hidden_total, expected_hidden_total, rtol=1e-5)
    assert torch.allclose(actual_output_recon, expected_output_recon, rtol=1e-5)
    fc1_mse, _ = per_module["fc1"]
    assert torch.allclose(fc1_mse, expected_fc1_mse, rtol=1e-5)
    fc2_mse, _ = per_module["fc2"]
    assert torch.allclose(fc2_mse, expected_fc2_mse, rtol=1e-5)
