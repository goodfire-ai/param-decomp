"""CLI entry point for dataset attribution SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.

Usage:
    pd-attributions <wandb_path> --config attr_slurm_config.yaml --harvest_subrun_id h-YYYYMMDD_HHMMSS
    pd-attributions <wandb_path> --config attr_slurm_config.yaml --harvest_subrun_id h-... --job_suffix v2
"""

import fire


def submit_attributions(
    wandb_path: str,
    config: str,
    harvest_subrun_id: str,
    job_suffix: str | None = None,
) -> None:
    """Submit multi-GPU dataset-attribution harvesting to SLURM.

    `harvest_subrun_id` (like `"h-20260306_120000"`) supplies the alive-mask set;
    `job_suffix` is appended to SLURM job names (e.g. `"v2"` → `"pd-attr-v2"`).
    """
    from param_decomp_lab.dataset_attributions.config import AttributionsSlurmConfig
    from param_decomp_lab.dataset_attributions.scripts.run_slurm import (
        submit_attributions as impl,
    )
    from param_decomp_lab.infra.wandb import parse_wandb_run_path

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
