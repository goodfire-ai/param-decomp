"""SLURM submission for intruder eval jobs."""

from param_decomp.harvest.config import IntruderSlurmConfig
from param_decomp.harvest.scripts.run_intruder import get_command
from param_decomp.log import logger
from param_decomp.utils.slurm import SlurmConfig, SubmitResult, generate_script, submit_slurm_job


def submit_intruder(
    decomposition_id: str,
    slurm_config: IntruderSlurmConfig,
    harvest_subrun_id: str,
    snapshot_branch: str | None = None,
    dependency_job_id: str | None = None,
) -> SubmitResult:
    cmd = get_command(decomposition_id, slurm_config.config, harvest_subrun_id)

    slurm = SlurmConfig(
        job_name=f"pd-intruder-{decomposition_id}",
        partition=slurm_config.partition,
        n_gpus=0,
        time=slurm_config.time,
        cpus_per_task=4,
        mem="64G",
        snapshot_branch=snapshot_branch,
        dependency_job_id=dependency_job_id,
        comment=f"intruder {decomposition_id}/{harvest_subrun_id}",
    )
    script = generate_script(slurm, cmd)
    result = submit_slurm_job(script, f"intruder_{decomposition_id}")

    logger.section("Intruder eval job submitted")
    logger.values(
        {
            "Decomposition": decomposition_id,
            "Harvest subrun": harvest_subrun_id,
            "Job ID": result.job_id,
            "Log": result.log_pattern,
        }
    )
    return result
