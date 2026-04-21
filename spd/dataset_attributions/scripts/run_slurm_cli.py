"""CLI entry point for dataset attribution SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.

Usage:
    spd-attributions <wandb_path> --n_gpus 8
    spd-attributions <wandb_path> --config attr_config.yaml
"""

import fire


def submit_attributions(
    wandb_path: str,
    config: str,
    harvest_subrun_id: str,
    job_suffix: str | None = None,
) -> None:
    """Submit multi-GPU dataset attribution harvesting to SLURM.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        config: Path to AttributionsSlurmConfig YAML/JSON. Uses built-in defaults if omitted.
        harvest_subrun_id: Harvest subrun to use for alive masks (e.g. "h-20260306_120000").
        job_suffix: Optional suffix for SLURM job names (e.g., "v2" -> "spd-attr-v2").
    """
    from spd.dataset_attributions.config import AttributionsSlurmConfig
    from spd.dataset_attributions.scripts.run_slurm import submit_attributions as impl
    from spd.utils.wandb_utils import parse_wandb_run_path

    parse_wandb_run_path(wandb_path)

    slurm_config = AttributionsSlurmConfig.from_file(config)
    impl(
        wandb_path=wandb_path,
        config=slurm_config,
        harvest_subrun_id=harvest_subrun_id,
        job_suffix=job_suffix,
    )


def cli() -> None:
    fire.Fire(submit_attributions)
