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

    Args:
        decomposition_id: ID of the target decomposition (e.g. "clt-1d4752ea").
        harvest_subrun_id: Harvest subrun to use (e.g. "h-20260323_163726").
        config: Path to IntruderSlurmConfig YAML/JSON. Uses defaults if omitted.
    """
    from param_decomp.harvest.config import IntruderSlurmConfig
    from param_decomp.harvest.scripts.run_intruder_slurm import submit_intruder

    slurm_config = IntruderSlurmConfig.from_file(config) if config else IntruderSlurmConfig()
    submit_intruder(decomposition_id, slurm_config, harvest_subrun_id)


def cli() -> None:
    fire.Fire(main)
