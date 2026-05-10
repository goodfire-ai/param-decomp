"""Build a `PDTarget` for a TMS experiment."""

from param_decomp.experiments.tms.configs import TMSTargetConfig
from param_decomp.experiments.tms.models import TMSModel, TMSTargetRunInfo
from param_decomp.models.batch_and_loss_fns import PDTarget, recon_loss_mse, run_batch_first_element


def load_tms_target(target_cfg: TMSTargetConfig) -> tuple[PDTarget, TMSTargetRunInfo]:
    """Load TMS target weights, build `PDTarget`, return both target and run_info.

    The `TMSTargetRunInfo` is returned so the TMS driver can build dataloaders and persist
    self-contained target artifacts beside the PD checkpoint.
    """
    run_info = TMSTargetRunInfo.from_path(target_cfg.run_path)
    target_model = TMSModel.from_run_info(run_info)
    target_model.eval()

    tied_weights: list[tuple[str, str]] | None = None
    if target_model.config.tied_weights:
        tied_weights = [("linear1", "linear2")]

    target = PDTarget(
        model=target_model,
        run_batch=run_batch_first_element,
        reconstruction_loss=recon_loss_mse,
        tied_weights=tied_weights,
        name="tms",
    )
    return target, run_info
