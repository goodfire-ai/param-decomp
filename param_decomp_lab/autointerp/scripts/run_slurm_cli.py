"""CLI entry point for autointerp SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.

Usage:
    pd-autointerp <decomposition_id> --config autointerp_slurm_config.yaml --harvest_subrun_id h-YYYYMMDD_HHMMSS
"""

import fire


def main(
    decomposition_id: str,
    config: str,
    harvest_subrun_id: str,
    snapshot_ref: str | None = None,
) -> None:
    """Submit autointerp pipeline (interpret + evals) to SLURM.

    `harvest_subrun_id` like `"h-20260306_120000"`. `snapshot_ref` defaults to the
    current REPO_ROOT checkout.
    """
    from param_decomp_lab.autointerp.config import AutointerpSlurmConfig
    from param_decomp_lab.autointerp.scripts.run_slurm import submit_autointerp

    slurm_config = AutointerpSlurmConfig.from_file(config)
    submit_autointerp(
        decomposition_id,
        slurm_config,
        harvest_subrun_id=harvest_subrun_id,
        snapshot_ref=snapshot_ref,
    )


def cli() -> None:
    fire.Fire(main)
