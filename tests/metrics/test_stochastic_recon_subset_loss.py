from unittest.mock import patch

import torch
from torch import Tensor

from spd.configs import SamplingType, UniformKSubsetRoutingConfig
from spd.metrics import stochastic_recon_subset_loss
from spd.models.batch_and_loss_fns import recon_loss_mse
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.routing import Router
from tests.metrics.fixtures import make_one_layer_component_model


class TestStochasticReconSubsetLoss:
    def test_manual_calculation_with_routing(self: object) -> None:
        """Test stochastic reconstruction with routing to layer subsets."""
        torch.manual_seed(42)

        # Setup: 2D input -> 2D output
        fc_weight = torch.randn(2, 2, dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)

        V = model.components["fc"].V
        U = model.components["fc"].U

        batch = torch.randn(1, 2, dtype=torch.float32)
        target_out = model(batch)

        ci = {"fc": torch.tensor([[0.8]], dtype=torch.float32)}

        # Define deterministic masks and routing masks for our samples
        # n_mask_samples=2, so we'll have 2 samples
        # routing_mask=True means route to this layer, False means skip
        sample_data = [
            {
                "component_mask": torch.tensor([[0.9]], dtype=torch.float32),
                "routing_mask": torch.tensor([True], dtype=torch.bool),  # Route to layer
            },
            {
                "component_mask": torch.tensor([[0.7]], dtype=torch.float32),
                "routing_mask": torch.tensor([False], dtype=torch.bool),  # Skip layer
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
            idx = call_count[0] % len(sample_data)
            call_count[0] += 1
            data = sample_data[idx]

            return make_mask_infos(
                component_masks={"fc": data["component_mask"]},
                routing_masks={"fc": data["routing_mask"]},
                weight_deltas_and_masks=None,
            )

        with patch(
            "spd.metrics.stochastic_recon_subset_loss.calc_stochastic_component_mask_info",
            side_effect=mock_calc_stochastic_component_mask_info,
        ):
            # Calculate expected loss manually
            sum_loss = 0.0
            n_examples = 0

            for data in sample_data:
                component_mask = data["component_mask"]
                routing_mask = data["routing_mask"]
                if not routing_mask:
                    # If not routed, the output should be the same as the target output and have
                    # no loss
                    n_examples += target_out.numel()
                    continue
                # Manually calculate forward pass with routing:
                masked_component = V * component_mask @ U
                components_out = batch @ masked_component
                # routing_mask is shape [1, 1], need to broadcast with [..., None] like the code does
                out = torch.where(routing_mask[..., None], components_out, target_out)
                loss = torch.nn.functional.mse_loss(out, target_out, reduction="sum")
                sum_loss += loss.item()
                n_examples += out.numel()

            expected_loss = sum_loss / n_examples

            # Calculate actual loss
            actual_loss = stochastic_recon_subset_loss(
                model=model,
                sampling="continuous",
                n_mask_samples=2,
                batch=batch,
                target_out=target_out,
                ci=ci,
                weight_deltas=None,
                routing=UniformKSubsetRoutingConfig(),
                reconstruction_loss=recon_loss_mse,
            )

            assert torch.allclose(actual_loss, torch.tensor(expected_loss), rtol=1e-5), (
                f"Expected {expected_loss}, got {actual_loss}"
            )
