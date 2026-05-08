"""Build a `PDTarget` for an induction-head experiment."""

from param_decomp.experiments.ih.configs import IHTargetConfig
from param_decomp.experiments.ih.model import InductionModelTargetRunInfo, InductionTransformer
from param_decomp.models.batch_and_loss_fns import PDTarget, recon_loss_kl, run_batch_first_element


def load_ih_target(
    target_cfg: IHTargetConfig,
) -> tuple[PDTarget, InductionModelTargetRunInfo]:
    """Load IH target weights, build `PDTarget`, return both target and run_info."""
    run_info = InductionModelTargetRunInfo.from_path(target_cfg.run_path)
    target_model = InductionTransformer.from_run_info(run_info)
    target_model.eval()

    target = PDTarget(
        model=target_model,
        run_batch=run_batch_first_element,
        reconstruction_loss=recon_loss_kl,
        name="ih",
    )
    return target, run_info
