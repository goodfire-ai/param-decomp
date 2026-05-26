"""CLI entry point for harvest SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.

Usage:
    pd-harvest <harvest_slurm_config.yaml>
    pd-harvest <harvest_slurm_config.yaml> --job_suffix v2
"""

import fire


def harvest(
    config: str,
    job_suffix: str | None = None,
) -> None:
    """Submit multi-GPU harvest job to SLURM.

    `job_suffix` is appended to SLURM job names (e.g. `"v2"` → `"pd-harvest-v2"`).
    """
    from param_decomp_lab.harvest.config import HarvestSlurmConfig
    from param_decomp_lab.harvest.scripts.run_slurm import submit_harvest

    slurm_config = HarvestSlurmConfig.from_file(config)
    submit_harvest(config=slurm_config, job_suffix=job_suffix)


def cli() -> None:
    fire.Fire(harvest)
