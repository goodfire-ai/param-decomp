"""CLI entry point for autointerp SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.

Usage:
    spd-autointerp <wandb_path>
    spd-autointerp <wandb_path> --config autointerp_config.yaml
"""

import fire


def main(
    decomposition_id: str,
    config: str,
    harvest_subrun_id: str,
    snapshot_branch: str | None = None,
) -> None:
    """Submit autointerp pipeline (interpret + evals) to SLURM.

    Args:
        decomposition_id: ID of the target decomposition run.
        config: Path to AutointerpSlurmConfig YAML/JSON.
        harvest_subrun_id: Harvest subrun to use (e.g. "h-20260306_120000").
        snapshot_branch: Git branch to run from (default: current REPO_ROOT checkout).
    """
    from spd.autointerp.config import AutointerpSlurmConfig
    from spd.autointerp.scripts.run_slurm import submit_autointerp

    slurm_config = AutointerpSlurmConfig.from_file(config)
    submit_autointerp(
        decomposition_id,
        slurm_config,
        harvest_subrun_id=harvest_subrun_id,
        snapshot_branch=snapshot_branch,
    )


def cli() -> None:
    fire.Fire(main)
