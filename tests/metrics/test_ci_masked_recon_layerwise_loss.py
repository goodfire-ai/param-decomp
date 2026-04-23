import torch

from spd.metrics import ci_masked_recon_layerwise_loss, ci_masked_recon_loss
from spd.models.batch_and_loss_fns import recon_loss_mse
from tests.metrics.fixtures import make_one_layer_component_model, make_two_layer_component_model


class TestCIMaskedReconLayerwiseLoss:
    def test_two_layer_manual_calculation(self: object) -> None:
        """Test layerwise reconstruction with manual calculation on two layers."""
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
            "fc1": torch.tensor([[1.0]], dtype=torch.float32),
            "fc2": torch.tensor([[1.0]], dtype=torch.float32),
        }

        # Calculate expected loss manually
        # 1. Loss from fc1 component active: x -> V1 @ U1 (replacing fc1) -> fc2_weight
        h_comp1 = (batch @ V1) @ U1
        out_fc1_active = h_comp1 @ fc2_weight.T
        loss_fc1 = torch.nn.functional.mse_loss(out_fc1_active, target_out, reduction="sum")

        # 2. Loss from fc2 component active: x -> fc1_weight -> V2 @ U2 (replacing fc2)
        h = batch @ fc1_weight.T
        out_fc2_active = (h @ V2) @ U2
        loss_fc2 = torch.nn.functional.mse_loss(out_fc2_active, target_out, reduction="sum")

        n_examples = out_fc1_active.numel() + out_fc2_active.numel()
        expected_loss = (loss_fc1 + loss_fc2) / n_examples

        # Calculate actual loss
        actual_loss = ci_masked_recon_layerwise_loss(
            model=model,
            batch=batch,
            target_out=target_out,
            ci=ci,
            reconstruction_loss=recon_loss_mse,
        )

        assert torch.allclose(actual_loss, expected_loss, rtol=1e-5), (
            f"Expected {expected_loss}, got {actual_loss}"
        )

    def test_layerwise_vs_all_layer(self: object) -> None:
        """For a single layer, layerwise and all-layer should be the same."""
        fc_weight = torch.randn(2, 2, dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)

        batch = torch.randn(1, 2, dtype=torch.float32)
        target_out = torch.randn(1, 2, dtype=torch.float32)
        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}

        loss_all = ci_masked_recon_loss(
            model=model,
            batch=batch,
            target_out=target_out,
            ci=ci,
            reconstruction_loss=recon_loss_mse,
        )
        loss_layerwise = ci_masked_recon_layerwise_loss(
            model=model,
            batch=batch,
            target_out=target_out,
            ci=ci,
            reconstruction_loss=recon_loss_mse,
        )

        # For single layer, results should be the same
        assert torch.allclose(loss_all, loss_layerwise, rtol=1e-5)
