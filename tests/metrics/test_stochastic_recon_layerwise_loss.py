from unittest.mock import patch

import torch
from torch import Tensor

from spd.configs import SamplingType
from spd.metrics import stochastic_recon_layerwise_loss, stochastic_recon_loss
from spd.models.batch_and_loss_fns import recon_loss_mse
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.routing import Router
from tests.metrics.fixtures import make_one_layer_component_model, make_two_layer_component_model


class TestStochasticReconLayerwiseLoss:
    def test_two_layer_manual_calculation(self: object) -> None:
        """Test layerwise stochastic reconstruction with manual calculation.

        Mocks calc_stochastic_component_mask_info to use deterministic masks.
        """
        torch.manual_seed(42)

        fc1_weight = torch.randn(4, 3, dtype=torch.float32)
        fc2_weight = torch.randn(2, 4, dtype=torch.float32)

        model = make_two_layer_component_model(weight1=fc1_weight, weight2=fc2_weight)

        V1 = model.components["fc1"].V
        U1 = model.components["fc1"].U
        V2 = model.components["fc2"].V
        U2 = model.components["fc2"].U

        batch = torch.randn(1, 3, dtype=torch.float32)
        target_out = torch.randn(1, 2, dtype=torch.float32)

        ci = {
            "fc1": torch.tensor([[0.8]], dtype=torch.float32),
            "fc2": torch.tensor([[0.6]], dtype=torch.float32),
        }

        # Define deterministic masks for our samples
        # n_mask_samples=2, so we'll have 2 samples
        # Each sample will have one mask_info per layer
        sample_masks = [
            # Sample 1: fc1 has mask 0.9, fc2 has mask 0.7
            {
                "fc1": torch.tensor([[0.9]], dtype=torch.float32),
                "fc2": torch.tensor([[0.7]], dtype=torch.float32),
            },
            # Sample 2: fc1 has mask 0.85, fc2 has mask 0.65
            {
                "fc1": torch.tensor([[0.85]], dtype=torch.float32),
                "fc2": torch.tensor([[0.65]], dtype=torch.float32),
            },
        ]

        # Mock calc_stochastic_component_mask_info to return our deterministic masks
        call_count = [0]

        def mock_calc_stochastic_component_mask_info(
            causal_importances: dict[str, Tensor],  # pyright: ignore[reportUnusedParameter]
            component_mask_sampling: SamplingType,  # pyright: ignore[reportUnusedParameter]
            router: Router,  # pyright: ignore[reportUnusedParameter]
            weight_deltas: dict[str, Tensor] | None,  # pyright: ignore[reportUnusedParameter]
        ) -> dict[str, ComponentsMaskInfo]:
            # Get the current call index (we'll cycle through sample_masks)
            idx = call_count[0] % len(sample_masks)
            call_count[0] += 1
            masks = sample_masks[idx]

            return make_mask_infos(
                component_masks=masks,
                routing_masks="all",
                weight_deltas_and_masks=None,
            )

        with patch(
            "spd.metrics.stochastic_recon_layerwise_loss.calc_stochastic_component_mask_info",
            side_effect=mock_calc_stochastic_component_mask_info,
        ):
            # Calculate expected loss manually
            sum_loss = 0.0
            n_examples = 0

            for masks in sample_masks:
                # For each sample, we evaluate each layer separately
                # Layer fc1: out = batch @ (V1 * mask_fc1 @ U1) @ fc2_weight.T
                masked_component_fc1 = V1 * masks["fc1"] @ U1
                hidden_fc1 = batch @ masked_component_fc1
                out_fc1 = hidden_fc1 @ fc2_weight.T
                loss_fc1 = torch.nn.functional.mse_loss(out_fc1, target_out, reduction="sum")
                sum_loss += loss_fc1.item()
                n_examples += out_fc1.numel()

                # Layer fc2: out = batch @ fc1_weight.T @ (V2 * mask_fc2 @ U2)
                hidden_fc2 = batch @ fc1_weight.T
                masked_component_fc2 = V2 * masks["fc2"] @ U2
                out_fc2 = hidden_fc2 @ masked_component_fc2
                loss_fc2 = torch.nn.functional.mse_loss(out_fc2, target_out, reduction="sum")
                sum_loss += loss_fc2.item()
                n_examples += out_fc2.numel()

            expected_loss = sum_loss / n_examples

            # Calculate actual loss
            actual_loss = stochastic_recon_layerwise_loss(
                model=model,
                sampling="continuous",
                n_mask_samples=2,
                batch=batch,
                target_out=target_out,
                ci=ci,
                weight_deltas=None,
                reconstruction_loss=recon_loss_mse,
            )

            assert torch.allclose(actual_loss, torch.tensor(expected_loss), rtol=1e-5), (
                f"Expected {expected_loss}, got {actual_loss}"
            )

    def test_layerwise_vs_full_loss(self: object) -> None:
        """For a single layer, layerwise and full loss should be the same."""
        torch.manual_seed(42)
        fc_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)

        batch = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        target_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}

        loss_full = stochastic_recon_loss(
            model=model,
            sampling="continuous",
            n_mask_samples=5,
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=None,
            reconstruction_loss=recon_loss_mse,
        )
        loss_layerwise = stochastic_recon_layerwise_loss(
            model=model,
            sampling="continuous",
            n_mask_samples=5,
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=None,
            reconstruction_loss=recon_loss_mse,
        )

        # For single layer, results should be the same
        assert torch.allclose(loss_full, loss_layerwise, rtol=1e-4)
