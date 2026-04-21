"""CLI entry point for graph interp SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.

Usage:
    spd-graph-interp <decomposition_id> --config graph_interp_config.yaml
"""

import fire


def main(decomposition_id: str, config: str, harvest_subrun_id: str) -> None:
    """Submit graph interpretation pipeline to SLURM.

    Args:
        decomposition_id: ID of the target decomposition run.
        config: Path to GraphInterpSlurmConfig YAML/JSON.
        harvest_subrun_id: Harvest subrun to use (e.g. "h-20260306_120000").
    """
    from spd.graph_interp.config import GraphInterpSlurmConfig
    from spd.graph_interp.scripts.run_slurm import submit_graph_interp

    slurm_config = GraphInterpSlurmConfig.from_file(config)
    submit_graph_interp(
        decomposition_id, slurm_config, dependency_job_ids=[], harvest_subrun_id=harvest_subrun_id
    )


def cli() -> None:
    fire.Fire(main)
