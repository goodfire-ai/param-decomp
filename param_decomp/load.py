"""Public load API for PD runs."""

from pathlib import Path

from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.models.component_model import ComponentModel, PDRunInfo
from param_decomp.param_decomp_types import ModelPath
from param_decomp.utils.general_utils import fetch_latest_local_checkpoint


def load_pd(
    path: ModelPath,
    *,
    target: PDTarget,
    checkpoint: str | Path | None = None,
) -> ComponentModel:
    """Load a `ComponentModel` from a saved PD run.

    Args:
        path: Run directory, wandb path (`wandb:entity/project/runs/id`), or checkpoint file.
        target: User-supplied target. The caller is responsible for instantiating the target
            model (e.g. via the experiment config's `load_target()` method).
        checkpoint: Optional override for which `model_*.pth` to load. Defaults to the latest.
    """
    # If `path` is a local directory, pick the latest checkpoint and pass that to PDRunInfo so it
    # can locate the config alongside it.
    resolved: ModelPath = path
    if not str(path).startswith("wandb:"):
        path_obj = Path(path)
        if path_obj.is_dir():
            resolved = fetch_latest_local_checkpoint(path_obj, prefix="model")

    run_info = PDRunInfo.from_path(resolved)
    checkpoint_path = Path(checkpoint) if checkpoint is not None else run_info.checkpoint_path
    return ComponentModel.from_checkpoint(
        config=run_info.pd_config,
        checkpoint_path=checkpoint_path,
        target_model=target.model,
        run_batch=target.run_batch,
        tied_weights=target.tied_weights,
    )
