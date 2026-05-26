"""CLI entry point for intruder eval SLURM launcher.

Usage:
    pd-intruder <decomposition_id> <harvest_subrun_id>
    pd-intruder <decomposition_id> <harvest_subrun_id> --config intruder_config.yaml
"""

import fire


def main(
    decomposition_id: str,
    harvest_subrun_id: str,
    config: str | None = None,
) -> None:
    """Submit intruder eval to SLURM.

    `decomposition_id` looks like `"clt-1d4752ea"`; `harvest_subrun_id` like
    `"h-20260323_163726"`.
    """
    from param_decomp_lab.harvest.config import IntruderSlurmConfig
    from param_decomp_lab.harvest.scripts.run_intruder_slurm import submit_intruder

    slurm_config = IntruderSlurmConfig.from_file(config) if config else IntruderSlurmConfig()
    submit_intruder(decomposition_id, slurm_config, harvest_subrun_id)


def cli() -> None:
    fire.Fire(main)
