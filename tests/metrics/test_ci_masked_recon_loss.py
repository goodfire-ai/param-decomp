import torch

from spd.metrics import ci_masked_recon_loss
from spd.models.batch_and_loss_fns import recon_loss_mse
from tests.metrics.fixtures import make_one_layer_component_model


class TestCIMaskedReconLoss:
    def test_manual_calculation(self: object) -> None:
        """Test all-layer reconstruction with manual calculation."""
        torch.manual_seed(42)

        fc_weight = torch.randn(2, 3, dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)

        V = model.components["fc"].V
        U = model.components["fc"].U

        batch = torch.randn(1, 3, dtype=torch.float32)
        target_out = torch.randn(1, 2, dtype=torch.float32)

        ci = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}

        # Calculate expected loss manually
        out = (batch @ V) @ U
        expected_loss = torch.nn.functional.mse_loss(out, target_out)

        # Calculate actual loss
        actual_loss = ci_masked_recon_loss(
            model=model,
            batch=batch,
            target_out=target_out,
            ci=ci,
            reconstruction_loss=recon_loss_mse,
        )

        assert torch.allclose(actual_loss, expected_loss, rtol=1e-5), (
            f"Expected {expected_loss}, got {actual_loss}"
        )

    def test_different_ci_values_produce_different_losses(self: object) -> None:
        # Test that different CI values produce different reconstruction losses
        fc_weight = torch.randn(2, 2, dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)

        batch = torch.randn(1, 2, dtype=torch.float32)
        target_out = torch.randn(1, 2, dtype=torch.float32)

        ci_full = {"fc": torch.tensor([[1.0]], dtype=torch.float32)}
        ci_half = {"fc": torch.tensor([[0.5]], dtype=torch.float32)}

        loss_full = ci_masked_recon_loss(
            model=model,
            batch=batch,
            target_out=target_out,
            ci=ci_full,
            reconstruction_loss=recon_loss_mse,
        )
        loss_half = ci_masked_recon_loss(
            model=model,
            batch=batch,
            target_out=target_out,
            ci=ci_half,
            reconstruction_loss=recon_loss_mse,
        )

        # Different CI values should produce different losses
        assert loss_full != loss_half
