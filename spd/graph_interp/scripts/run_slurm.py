"""SLURM launcher for graph interpretation.

Submits a single CPU job that runs the three-phase interpretation pipeline.
Depends on both harvest merge and attribution merge jobs.
"""

from dataclasses import dataclass
from datetime import datetime

from spd.graph_interp.config import GraphInterpSlurmConfig
from spd.graph_interp.scripts import run
from spd.log import logger
from spd.utils.slurm import SlurmConfig, SubmitResult, generate_script, submit_slurm_job


@dataclass
class GraphInterpSubmitResult:
    result: SubmitResult


def submit_graph_interp(
    decomposition_id: str,
    config: GraphInterpSlurmConfig,
    dependency_job_ids: list[str],
    harvest_subrun_id: str,
    snapshot_branch: str | None = None,
) -> GraphInterpSubmitResult:
    """Submit graph interpretation to SLURM."""
    subrun_id = "ti-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    cmd = run.get_command(
        decomposition_id=decomposition_id,
        config=config.config,
        subrun_id=subrun_id,
        harvest_subrun_id=harvest_subrun_id,
    )

    dependency_str = ":".join(dependency_job_ids) if dependency_job_ids else None

    slurm_config = SlurmConfig(
        job_name="spd-graph-interp",
        partition=config.partition,
        n_gpus=0,
        cpus_per_task=16,
        mem="240G",
        time=config.time,
        snapshot_branch=snapshot_branch,
        dependency_job_id=dependency_str,
        comment=decomposition_id,
    )
    script_content = generate_script(slurm_config, cmd)
    result = submit_slurm_job(script_content, "spd-graph-interp")

    logger.section("Graph interp job submitted")
    logger.values(
        {
            "Job ID": result.job_id,
            "Decomposition ID": decomposition_id,
            "Model": config.config.llm.model,
            "Depends on": ", ".join(dependency_job_ids),
            "Log": result.log_pattern,
        }
    )

    return GraphInterpSubmitResult(result=result)
