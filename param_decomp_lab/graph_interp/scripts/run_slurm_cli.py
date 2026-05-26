"""CLI entry point for graph interp SLURM launcher.

Thin wrapper for fast --help. Heavy imports deferred to run_slurm.py.

Usage:
    pd-graph-interp <decomposition_id> --config graph_interp_config.yaml
"""

import fire


def main(decomposition_id: str, config: str, harvest_subrun_id: str) -> None:
    """Submit graph interpretation pipeline to SLURM.

    `harvest_subrun_id` looks like `"h-20260306_120000"`.
    """
    from param_decomp_lab.graph_interp.config import GraphInterpSlurmConfig
    from param_decomp_lab.graph_interp.scripts.run_slurm import submit_graph_interp

    slurm_config = GraphInterpSlurmConfig.from_file(config)
    submit_graph_interp(
        decomposition_id, slurm_config, dependency_job_ids=[], harvest_subrun_id=harvest_subrun_id
    )


def cli() -> None:
    fire.Fire(main)
