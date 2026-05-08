"""Build a `PDTarget` for a Residual MLP experiment."""

from param_decomp.experiments.resid_mlp.configs import ResidMLPTargetConfig
from param_decomp.experiments.resid_mlp.models import ResidMLP, ResidMLPTargetRunInfo
from param_decomp.models.batch_and_loss_fns import PDTarget, recon_loss_mse, run_batch_first_element


def load_resid_mlp_target(
    target_cfg: ResidMLPTargetConfig,
) -> tuple[PDTarget, ResidMLPTargetRunInfo]:
    """Load ResidMLP target weights, build `PDTarget`, return both target and run_info."""
    run_info = ResidMLPTargetRunInfo.from_path(target_cfg.run_path)
    target_model = ResidMLP.from_run_info(run_info)
    target_model.eval()

    target = PDTarget(
        model=target_model,
        run_batch=run_batch_first_element,
        reconstruction_loss=recon_loss_mse,
        name="resid_mlp",
    )
    return target, run_info
