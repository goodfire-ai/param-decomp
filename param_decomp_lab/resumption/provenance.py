"""Resume provenance: which run a resumed run was forked from.

Resumed runs get their own `run_id` and own `run_meta.yaml`. Provenance is what makes
them traceable back to the parent: the resume composition root sets
`ExperimentConfig.resume_provenance` on the effective config it hands to `init_pd_run`,
so the lineage is dumped into `run_meta.yaml` and surfaced in `wandb.config` (and thus
the wandb UI). A run with `resume_provenance is None` is a fresh run.
"""

from pathlib import Path

from param_decomp.base_config import BaseConfig


class ResumeProvenance(BaseConfig):
    """Records where a resumed run came from. Lives on `ExperimentConfig`."""

    parent_run_dir: Path
    """Path to the parent run's directory."""

    parent_step: int
    """The step at which we resumed (i.e. the step number in the parent's
    `training_<step>.pth` snapshot we loaded from)."""
