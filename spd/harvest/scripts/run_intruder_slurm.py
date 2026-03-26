"""SLURM submission for intruder eval jobs."""

import secrets

from spd.harvest.config import IntruderSlurmConfig
from spd.harvest.scripts.run_intruder import get_command
from spd.log import logger
from spd.utils.git_utils import create_git_snapshot
from spd.utils.slurm import SlurmConfig, SubmitResult, generate_script, submit_slurm_job


def submit_intruder(
    decomposition_id: str,
    slurm_config: IntruderSlurmConfig,
    harvest_subrun_id: str,
    snapshot_branch: str | None = None,
    dependency_job_id: str | None = None,
) -> SubmitResult:
    if snapshot_branch is None:
        run_id = f"intruder-{secrets.token_hex(4)}"
        snapshot_branch, commit_hash = create_git_snapshot(snapshot_id=run_id)
        logger.info(f"Created git snapshot: {snapshot_branch} ({commit_hash[:8]})")

    cmd = get_command(decomposition_id, slurm_config.config, harvest_subrun_id)

    slurm = SlurmConfig(
        job_name=f"spd-intruder-{decomposition_id}",
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
            "Snapshot": snapshot_branch,
            "Job ID": result.job_id,
            "Log": result.log_pattern,
        }
    )
    return result
