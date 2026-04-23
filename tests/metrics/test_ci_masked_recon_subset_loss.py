from unittest.mock import patch

import torch
from torch import Tensor

from spd.configs import UniformKSubsetRoutingConfig
from spd.metrics import ci_masked_recon_subset_loss
from spd.models.batch_and_loss_fns import recon_loss_mse
from tests.metrics.fixtures import make_one_layer_component_model


class TestCIMaskedReconSubsetLoss:
    def test_manual_calculation_with_routing(self: object) -> None:
        """Test CI masked reconstruction with routing to layer subsets."""
        torch.manual_seed(42)

        # Setup: 2D input -> 2D output
        fc_weight = torch.randn(2, 2, dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)

        V = model.components["fc"].V
        U = model.components["fc"].U

        batch = torch.randn(1, 2, dtype=torch.float32)
        target_out = model(batch)

        ci = {"fc": torch.tensor([[0.8]], dtype=torch.float32)}

        # Define deterministic routing masks for our test
        # routing_mask=True means route to this layer, False means skip
        routing_masks = [
            {"fc": torch.tensor([True], dtype=torch.bool)},  # Route to layer
            {"fc": torch.tensor([False], dtype=torch.bool)},  # Skip layer
        ]

        # Mock sample_uniform_k_subset_routing_masks to return our deterministic masks
        call_count = [0]

        def mock_sample_uniform_k_subset_routing_masks(
            mask_shape: tuple[int, ...],  # pyright: ignore[reportUnusedParameter]
            module_names: list[str],  # pyright: ignore[reportUnusedParameter]
            device: torch.device | str = "cpu",  # pyright: ignore[reportUnusedParameter]
            generator: torch.Generator | None = None,  # pyright: ignore[reportUnusedParameter]
        ) -> dict[str, Tensor]:
            idx = call_count[0] % len(routing_masks)
            call_count[0] += 1
            return routing_masks[idx]

        with patch(
            "spd.routing.sample_uniform_k_subset_routing_masks",
            side_effect=mock_sample_uniform_k_subset_routing_masks,
        ):
            # Calculate expected loss manually
            sum_loss = 0.0
            n_examples = 0

            for routing_mask_dict in routing_masks:
                routing_mask = routing_mask_dict["fc"]
                if not routing_mask:
                    # If not routed, the output should be the same as the target output and have
                    # no loss
                    n_examples += target_out.numel()
                    continue
                # Manually calculate forward pass with routing:
                # Use CI as the component mask directly (no stochastic sampling)
                masked_component = V * ci["fc"] @ U
                components_out = batch @ masked_component
                # routing_mask is shape [1], need to broadcast with [..., None] like the code does
                out = torch.where(routing_mask[..., None], components_out, target_out)
                loss = torch.nn.functional.mse_loss(out, target_out, reduction="sum")
                sum_loss += loss.item()
                n_examples += out.numel()

            expected_loss = sum_loss / n_examples

            # Calculate actual loss - run twice since we mocked two routing masks
            actual_losses = []
            for _ in range(2):
                actual_loss = ci_masked_recon_subset_loss(
                    model=model,
                    batch=batch,
                    target_out=target_out,
                    ci=ci,
                    routing=UniformKSubsetRoutingConfig(),
                    reconstruction_loss=recon_loss_mse,
                )
                actual_losses.append(actual_loss.item())

            # Average the losses from both runs
            actual_loss_avg = sum(actual_losses) / len(actual_losses)

            assert torch.allclose(
                torch.tensor(actual_loss_avg), torch.tensor(expected_loss), rtol=1e-5
            ), f"Expected {expected_loss}, got {actual_loss_avg}"
