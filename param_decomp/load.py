"""Public load API for PD runs."""

from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.models.component_model import ComponentModel
from param_decomp.pd_run import PDRun
from param_decomp.types import ModelPath


def load_pd(
    path: ModelPath,
    *,
    target: PDTarget | None = None,
) -> ComponentModel:
    """Load a `ComponentModel` from a saved PD run.

    Args:
        path: Run directory, wandb path (`wandb:entity/project/runs/id`), or checkpoint file.
        target: Optional override. When ``None``, the run's driver reconstructs the target
            from the saved metadata. For manual/notebook runs (no driver), ``target`` is required.
    """
    return PDRun.from_path(path).load_model(target=target)
