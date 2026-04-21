"""CLI entry point for harvest SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.

Usage:
    spd-harvest <wandb_path> --n_gpus 8
    spd-harvest <wandb_path> --config harvest_config.yaml
"""

import fire


def harvest(
    config: str,
    job_suffix: str | None = None,
) -> None:
    """Submit multi-GPU harvest job to SLURM.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        config: Path to HarvestSlurmConfig YAML/JSON. Uses built-in defaults if omitted.
        job_suffix: Optional suffix for SLURM job names (e.g., "v2" -> "spd-harvest-v2").
    """
    from spd.harvest.config import HarvestSlurmConfig
    from spd.harvest.scripts.run_slurm import submit_harvest

    slurm_config = HarvestSlurmConfig.from_file(config)
    submit_harvest(config=slurm_config, job_suffix=job_suffix)


def cli() -> None:
    fire.Fire(harvest)
