"""CLI entry point for graph interp SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.

Usage:
    spd-graph-interp <decomposition_id> --config graph_interp_config.yaml
"""

import fire


def main(decomposition_id: str, config: str) -> None:
    """Submit graph interpretation pipeline to SLURM.

    Args:
        decomposition_id: ID of the target decomposition run.
        config: Path to GraphInterpSlurmConfig YAML/JSON.
    """
    from spd.graph_interp.config import GraphInterpSlurmConfig
    from spd.graph_interp.scripts.run_slurm import submit_graph_interp

    slurm_config = GraphInterpSlurmConfig.from_file(config)
    submit_graph_interp(decomposition_id, slurm_config, dependency_job_ids=[])


def cli() -> None:
    fire.Fire(main)
