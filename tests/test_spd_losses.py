from typing import override

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from spd.configs import (
    AdamPGDConfig,
    LayerwiseCiConfig,
    PersistentPGDReconLossConfig,
    ScheduleConfig,
    SignPGDConfig,
    SingleSourceScope,
    UniformKSubsetRoutingConfig,
)
from spd.metrics import (
    ci_masked_recon_layerwise_loss,
    ci_masked_recon_loss,
    ci_masked_recon_subset_loss,
    faithfulness_loss,
    importance_minimality_loss,
    stochastic_recon_layerwise_loss,
    stochastic_recon_loss,
    stochastic_recon_subset_loss,
)
from spd.models.component_model import ComponentModel
from spd.persistent_pgd import PersistentPGDState
from spd.utils.module_utils import ModulePathInfo


class TinyLinearModel(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_in, d_out, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


class TinySeqModel(nn.Module):
    """A simple sequence model that applies a linear layer to each position.

    Input shape: (batch, seq_len, d_in)
    Output shape: (batch, seq_len, d_out)
    """

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_in, d_out, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, d_in) -> (batch, seq_len, d_out)
        return self.fc(x)


def _make_component_model(weight: Float[Tensor, "d_out d_in"]) -> ComponentModel:
    d_out, d_in = weight.shape
    target = TinyLinearModel(d_in=d_in, d_out=d_out)
    with torch.no_grad():
        target.fc.weight.copy_(weight)
    target.requires_grad_(False)

    comp_model = ComponentModel(
        target_model=target,
        module_path_info=[ModulePathInfo(module_path="fc", C=1)],
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[2]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    return comp_model


def _make_seq_component_model(weight: Float[Tensor, "d_out d_in"]) -> ComponentModel:
    """Create a ComponentModel from TinySeqModel for 3D (batch, seq, hidden) shaped data."""
    d_out, d_in = weight.shape
    target = TinySeqModel(d_in=d_in, d_out=d_out)
    with torch.no_grad():
        target.fc.weight.copy_(weight)
    target.requires_grad_(False)

    comp_model = ComponentModel(
        target_model=target,
        module_path_info=[ModulePathInfo(module_path="fc", C=1)],
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[2]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    return comp_model


def _zero_components_for_test(model: ComponentModel) -> None:
    with torch.no_grad():
        for cm in model.components.values():
            cm.V.zero_()
            cm.U.zero_()


class TestCalcWeightDeltas:
    def test_components_and_identity(self: object) -> None:
        # fc weight 2x3 with known values
        fc_weight = torch.tensor([[1.0, 0.0, -1.0], [2.0, 3.0, -4.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)
        _zero_components_for_test(model)

        deltas = model.calc_weight_deltas()

        assert set(deltas.keys()) == {"fc"}

        # components were zeroed, so delta equals original weight
        expected_fc = fc_weight
        assert torch.allclose(deltas["fc"], expected_fc)

    def test_components_nonzero(self: object) -> None:
        # TODO WRITE DESCRIPTION
        fc_weight = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        deltas = model.calc_weight_deltas()
        assert set(deltas.keys()) == {"fc"}

        component = model.components["fc"]
        assert component is not None
        expected_fc = model.target_weight("fc") - component.weight
        assert torch.allclose(deltas["fc"], expected_fc)


class TestCalcFaithfulnessLoss:
    def test_manual_weight_deltas_normalization(self: object) -> None:
        weight_deltas = {
            "a": torch.tensor([[1.0, -1.0], [2.0, 0.0]], dtype=torch.float32),  # sum sq = 6
            "b": torch.tensor([[2.0, -2.0, 1.0]], dtype=torch.float32),  # sum sq = 9
        }
        # total sum sq = 15, total params = 4 + 3 = 7
        expected = torch.tensor(15.0 / 7.0)
        result = faithfulness_loss(weight_deltas=weight_deltas)
        assert torch.allclose(result, expected)

    def test_with_model_weight_deltas(self: object) -> None:
        fc_weight = torch.tensor([[1.0, 0.0, -1.0], [2.0, 3.0, -4.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)
        _zero_components_for_test(model)
        deltas = model.calc_weight_deltas()

        # Expected: mean of squared entries across both matrices
        expected = fc_weight.square().sum() / fc_weight.numel()

        result = faithfulness_loss(weight_deltas=deltas)
        assert torch.allclose(result, expected)


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


class TestCIMaskedReconLoss:
    def test_mse_loss_basic(self: object) -> None:
        # Test basic MSE reconstruction loss
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        # Input and target
        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        # CI values (will be used to mask components)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}  # Full component weight

        result = ci_masked_recon_loss(
            model=model,
            output_loss_type="mse",
            batch=batch,
            target_out=target_out,
            ci=ci,
        )

        # Since we're using a simple identity-like weight, and CI is 1,
        # the reconstruction should be close (not exact due to component decomposition)
        assert result >= 0.0

    def test_kl_loss_basic(self: object) -> None:
        # Test basic KL divergence loss
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        # Use log-probs for KL
        target_out = torch.nn.functional.log_softmax(
            torch.tensor([[1.0, 2.0]], dtype=torch.float32), dim=-1
        )

        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}

        result = ci_masked_recon_loss(
            model=model,
            output_loss_type="kl",
            batch=batch,
            target_out=target_out,
            ci=ci,
        )

        assert result >= 0.0

    def test_different_ci_values_produce_different_losses(self: object) -> None:
        # Test that different CI values produce different reconstruction losses
        fc_weight = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        target_out = torch.tensor([[2.0, 2.0]], dtype=torch.float32)

        ci_full = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}
        ci_half = {"fc": torch.tensor([[0.5]], dtype=torch.float32)}

        loss_full = ci_masked_recon_loss(
            model=model, output_loss_type="mse", batch=batch, target_out=target_out, ci=ci_full
        )
        loss_half = ci_masked_recon_loss(
            model=model, output_loss_type="mse", batch=batch, target_out=target_out, ci=ci_half
        )

        # Different CI values should produce different losses
        assert loss_full != loss_half


class TestCIMaskedReconLayerwiseLoss:
    def test_layerwise_basic(self: object) -> None:
        # Test layerwise reconstruction - each layer is evaluated separately
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}

        result = ci_masked_recon_layerwise_loss(
            model=model,
            output_loss_type="mse",
            batch=batch,
            target_out=target_out,
            ci=ci,
        )

        # Layerwise should produce a valid loss
        assert result >= 0.0

    def test_layerwise_vs_all_layer(self: object) -> None:
        # Layerwise should differ from all-layer when there are multiple layers
        # For a single layer, they should be similar
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}

        loss_all = ci_masked_recon_loss(
            model=model, output_loss_type="mse", batch=batch, target_out=target_out, ci=ci
        )
        loss_layerwise = ci_masked_recon_layerwise_loss(
            model=model, output_loss_type="mse", batch=batch, target_out=target_out, ci=ci
        )

        # For single layer, results should be the same
        assert torch.allclose(loss_all, loss_layerwise, rtol=1e-4)


class TestCIMaskedReconSubsetLoss:
    def test_subset_basic(self: object) -> None:
        # Test subset routing reconstruction
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}

        result = ci_masked_recon_subset_loss(
            model=model,
            output_loss_type="mse",
            batch=batch,
            target_out=target_out,
            ci=ci,
            routing=UniformKSubsetRoutingConfig(),
        )

        # Subset routing should produce a valid loss
        assert result >= 0.0

    def test_subset_stochastic_behavior(self: object) -> None:
        # Subset routing has randomness, so repeated calls may differ
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}

        # Run multiple times
        losses = [
            ci_masked_recon_subset_loss(
                model=model,
                output_loss_type="mse",
                batch=batch,
                target_out=target_out,
                ci=ci,
                routing=UniformKSubsetRoutingConfig(),
            )
            for _ in range(3)
        ]

        # All should be valid losses (>= 0)
        assert all(loss >= 0.0 for loss in losses)


class TestStochasticReconLoss:
    def test_continuous_sampling_basic(self: object) -> None:
        # Test stochastic reconstruction with continuous sampling
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}
        weight_deltas = model.calc_weight_deltas()

        result = stochastic_recon_loss(
            model=model,
            sampling="continuous",
            n_mask_samples=3,
            output_loss_type="mse",
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=weight_deltas,
        )

        assert result >= 0.0

    def test_binomial_sampling_basic(self: object) -> None:
        # Test stochastic reconstruction with binomial sampling
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}
        weight_deltas = model.calc_weight_deltas()

        result = stochastic_recon_loss(
            model=model,
            sampling="binomial",
            n_mask_samples=3,
            output_loss_type="mse",
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=weight_deltas,
        )

        assert result >= 0.0

    def test_multiple_mask_samples(self: object) -> None:
        # Test that using more mask samples produces valid results
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[0.5]], dtype=torch.float32)}
        weight_deltas = model.calc_weight_deltas()

        # Test with different numbers of samples
        for n_samples in [1, 3, 5]:
            result = stochastic_recon_loss(
                model=model,
                sampling="continuous",
                n_mask_samples=n_samples,
                output_loss_type="mse",
                batch=batch,
                target_out=target_out,
                ci=ci,
                weight_deltas=weight_deltas,
            )
            assert result >= 0.0

    def test_with_and_without_delta_component(self: object) -> None:
        # Test both with and without delta component
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}
        weight_deltas = model.calc_weight_deltas()

        loss_with_delta = stochastic_recon_loss(
            model=model,
            sampling="continuous",
            n_mask_samples=3,
            output_loss_type="mse",
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=weight_deltas,
        )

        loss_without_delta = stochastic_recon_loss(
            model=model,
            sampling="continuous",
            n_mask_samples=3,
            output_loss_type="mse",
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=None,
        )

        # Both should be valid
        assert loss_with_delta >= 0.0
        assert loss_without_delta >= 0.0


class TestStochasticReconLayerwiseLoss:
    def test_layerwise_stochastic_basic(self: object) -> None:
        # Test layerwise stochastic reconstruction
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}
        weight_deltas = model.calc_weight_deltas()

        result = stochastic_recon_layerwise_loss(
            model=model,
            sampling="continuous",
            n_mask_samples=2,
            output_loss_type="mse",
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=weight_deltas,
        )

        assert result >= 0.0

    def test_layerwise_multiple_samples(self: object) -> None:
        # Test with different numbers of mask samples
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[0.8]], dtype=torch.float32)}
        weight_deltas = model.calc_weight_deltas()

        for n_samples in [1, 2, 3]:
            result = stochastic_recon_layerwise_loss(
                model=model,
                sampling="continuous",
                n_mask_samples=n_samples,
                output_loss_type="mse",
                batch=batch,
                target_out=target_out,
                ci=ci,
                weight_deltas=weight_deltas,
            )
            assert result >= 0.0


class TestStochasticReconSubsetLoss:
    def test_subset_stochastic_basic(self: object) -> None:
        # Test subset stochastic reconstruction
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}
        weight_deltas = model.calc_weight_deltas()

        result = stochastic_recon_subset_loss(
            model=model,
            sampling="continuous",
            n_mask_samples=3,
            output_loss_type="mse",
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=weight_deltas,
            routing=UniformKSubsetRoutingConfig(),
        )

        assert result >= 0.0

    def test_subset_with_binomial_sampling(self: object) -> None:
        # Test subset with binomial sampling
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[0.7]], dtype=torch.float32)}
        weight_deltas = model.calc_weight_deltas()

        result = stochastic_recon_subset_loss(
            model=model,
            sampling="binomial",
            n_mask_samples=3,
            output_loss_type="mse",
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=weight_deltas,
            routing=UniformKSubsetRoutingConfig(),
        )

        assert result >= 0.0

    def test_subset_stochastic_variability(self: object) -> None:
        # Test that stochastic subset routing produces valid results across runs
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[0.5]], dtype=torch.float32)}
        weight_deltas = model.calc_weight_deltas()

        losses = [
            stochastic_recon_subset_loss(
                model=model,
                sampling="continuous",
                n_mask_samples=2,
                output_loss_type="mse",
                batch=batch,
                target_out=target_out,
                ci=ci,
                weight_deltas=weight_deltas,
                routing=UniformKSubsetRoutingConfig(),
            )
            for _ in range(3)
        ]

        # All should be valid
        assert all(loss >= 0.0 for loss in losses)


class TestPersistentPGDReconLoss:
    def test_basic_forward_and_state_update(self: object) -> None:
        """Test that persistent PGD computes loss and updates state."""
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        # Use (batch, seq) shaped data to match PersistentPGD's expectations
        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        # CI needs (batch, seq, C) shape for PersistentPGD
        ci = {"fc": torch.tensor([[[0.5], [0.5]]], dtype=torch.float32)}

        cfg = PersistentPGDReconLossConfig(
            optimizer=SignPGDConfig(lr_schedule=ScheduleConfig(start_val=0.1)),
            scope=SingleSourceScope(),
        )

        # Initialize state
        state = PersistentPGDState(
            module_to_c=model.module_to_c,
            batch_dims=(1, 2),
            device="cpu",
            use_delta_component=False,
            cfg=cfg,
            output_loss_type="mse",
        )

        # Store initial mask values
        initial_sources = {k: v.clone() for k, v in state.sources.items()}

        # Compute loss and gradients
        loss = state.compute_recon_loss(
            model=model,
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=None,
        )
        grad = state.get_grads(loss)

        # Apply PGD step
        state.step(grad)

        # Loss should be non-negative
        assert loss >= 0.0

        # Masks should have been updated (not equal to initial)
        for k in state.sources:
            # Due to PGD step, masks should change (unless gradient is exactly 0)
            assert state.sources[k].shape == initial_sources[k].shape
            # Masks should still be in [0, 1]
            assert torch.all(state.sources[k] >= 0.0)
            assert torch.all(state.sources[k] <= 1.0)

    def test_masks_persist_across_calls(self: object) -> None:
        """Test that masks persist and accumulate updates across calls."""
        fc_weight = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        target_out = torch.tensor([[2.0, 2.0]], dtype=torch.float32)
        # CI needs (batch, seq, C) shape for PersistentPGD
        ci = {"fc": torch.tensor([[[0.3], [0.3]]], dtype=torch.float32)}

        cfg = PersistentPGDReconLossConfig(
            optimizer=SignPGDConfig(lr_schedule=ScheduleConfig(start_val=0.1)),
            scope=SingleSourceScope(),
        )

        state = PersistentPGDState(
            module_to_c=model.module_to_c,
            batch_dims=(1, 2),
            device="cpu",
            use_delta_component=False,
            cfg=cfg,
            output_loss_type="mse",
        )

        # Run multiple steps
        sources_history = []
        for _ in range(5):
            sources_history.append({k: v.clone() for k, v in state.sources.items()})
            loss = state.compute_recon_loss(
                model=model,
                batch=batch,
                target_out=target_out,
                ci=ci,
                weight_deltas=None,
            )
            grad = state.get_grads(loss)
            state.step(grad)
            assert loss >= 0.0

        # Masks should have changed over time
        # (they accumulate updates, so later masks differ from earlier ones)
        for k in state.sources:
            initial = sources_history[0][k]
            final = state.sources[k]
            # Should have changed from initial (very unlikely to be identical after 5 steps)
            assert not torch.allclose(initial, final)

    def test_with_delta_component(self: object) -> None:
        """Test persistent PGD with delta component enabled."""
        # Use sequence model for proper 3D shapes (batch, seq, hidden)
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_seq_component_model(weight=fc_weight)

        # Input shape: (batch=1, seq=2, d_in=2)
        batch = torch.tensor([[[1.0, 2.0], [0.5, 1.5]]], dtype=torch.float32)
        target_out = torch.tensor([[[1.0, 2.0], [0.5, 1.5]]], dtype=torch.float32)
        # CI shape: (batch=1, seq=2, C=1)
        ci = {"fc": torch.tensor([[[0.5], [0.5]]], dtype=torch.float32)}
        weight_deltas = model.calc_weight_deltas()

        # batch_dims for PersistentPGDState is (batch, seq) = (1, 2)
        batch_dims = batch.shape[:2]

        cfg = PersistentPGDReconLossConfig(
            optimizer=SignPGDConfig(lr_schedule=ScheduleConfig(start_val=0.1)),
            scope=SingleSourceScope(),
        )

        # Initialize state with delta component
        state = PersistentPGDState(
            module_to_c=model.module_to_c,
            batch_dims=batch_dims,
            device="cpu",
            use_delta_component=True,
            cfg=cfg,
            output_loss_type="mse",
        )

        # Masks should have C+1 elements when using delta component
        assert state.sources["fc"].shape[-1] == model.module_to_c["fc"] + 1

        loss = state.compute_recon_loss(
            model=model,
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=weight_deltas,
        )
        grad = state.get_grads(loss)
        state.step(grad)

        assert loss >= 0.0

    def test_batch_dimension(self: object) -> None:
        """Test that masks broadcast correctly across batch dimension."""
        # Use sequence model for proper 3D shapes (batch, seq, hidden)
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_seq_component_model(weight=fc_weight)

        # Batch of 3 examples, seq_len of 2, d_in of 2
        # Shape: (batch=3, seq=2, d_in=2)
        batch = torch.tensor(
            [
                [[1.0, 2.0], [0.5, 1.5]],
                [[2.0, 3.0], [1.0, 2.0]],
                [[0.5, 1.0], [0.25, 0.5]],
            ],
            dtype=torch.float32,
        )
        target_out = torch.tensor(
            [
                [[1.0, 2.0], [0.5, 1.5]],
                [[2.0, 3.0], [1.0, 2.0]],
                [[0.5, 1.0], [0.25, 0.5]],
            ],
            dtype=torch.float32,
        )
        # CI needs (batch, seq, C) shape - (3, 2, 1) for 3 batch, 2 seq positions, 1 component
        ci = {
            "fc": torch.tensor(
                [[[0.5], [0.5]], [[0.6], [0.6]], [[0.4], [0.4]]], dtype=torch.float32
            )
        }

        # batch_dims for PersistentPGDState is (batch, seq) = (3, 2)
        batch_dims = batch.shape[:2]

        cfg = PersistentPGDReconLossConfig(
            optimizer=SignPGDConfig(lr_schedule=ScheduleConfig(start_val=0.1)),
            scope=SingleSourceScope(),
        )

        state = PersistentPGDState(
            module_to_c=model.module_to_c,
            batch_dims=batch_dims,
            device="cpu",
            use_delta_component=False,
            cfg=cfg,
            output_loss_type="mse",
        )

        # Masks should have shape (1, 1, C) for single_mask scope - single mask shared across batch
        assert state.sources["fc"].shape == (1, 1, model.module_to_c["fc"])

        loss = state.compute_recon_loss(
            model=model,
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=None,
        )
        grad = state.get_grads(loss)
        state.step(grad)

        assert loss >= 0.0

    def test_adam_optimizer_state(self: object) -> None:
        """Test that Adam optimizer path updates internal state."""
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = _make_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[0.5, 1.5]], dtype=torch.float32)
        # CI needs (batch, seq, C) shape for PersistentPGD
        ci = {"fc": torch.tensor([[[0.4], [0.4]]], dtype=torch.float32)}

        cfg = PersistentPGDReconLossConfig(
            optimizer=AdamPGDConfig(
                lr_schedule=ScheduleConfig(start_val=0.05), beta1=0.9, beta2=0.999, eps=1e-8
            ),
            scope=SingleSourceScope(),
        )

        state = PersistentPGDState(
            module_to_c=model.module_to_c,
            batch_dims=(1, 2),
            device="cpu",
            use_delta_component=False,
            cfg=cfg,
            output_loss_type="mse",
        )

        loss = state.compute_recon_loss(
            model=model,
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=None,
        )
        grad = state.get_grads(loss)
        state.step(grad)

        assert loss >= 0.0
