"""SLURM launcher for harvest pipeline.

Harvest is a functional unit: GPU workers -> merge. This module submits all
jobs in the unit with proper dependency chaining.

Usage:
    spd-harvest <wandb_path> --n_gpus 24
    spd-harvest <wandb_path> --n_batches 1000 --n_gpus 8  # Only process 1000 batches
"""

import secrets
from dataclasses import dataclass
from datetime import datetime

from spd.harvest.config import HarvestSlurmConfig
from spd.harvest.scripts import run_merge as harvest_merge
from spd.harvest.scripts import run_worker as harvest_worker
from spd.log import logger
from spd.utils.git_utils import create_git_snapshot
from spd.utils.slurm import (
    SlurmArrayConfig,
    SlurmConfig,
    SubmitResult,
    generate_array_script,
    generate_script,
    submit_slurm_job,
)


@dataclass
class HarvestSubmitResult:
    array_result: SubmitResult
    merge_result: SubmitResult
    subrun_id: str


def submit_harvest(
    config: HarvestSlurmConfig,
    job_suffix: str | None = None,
    snapshot_branch: str | None = None,
    dependency_job_id: str | None = None,
) -> HarvestSubmitResult:
    """Submit multi-GPU harvest job to SLURM.

    Submits a job array where each task processes a subset of batches, then
    submits a merge job that depends on all workers completing.
    """
    n_gpus = config.n_gpus
    partition = config.partition
    time = config.time

    if snapshot_branch is None:
        run_id = f"harvest-{secrets.token_hex(4)}"
        snapshot_branch, commit_hash = create_git_snapshot(snapshot_id=run_id)
        logger.info(f"Created git snapshot: {snapshot_branch} ({commit_hash[:8]})")
    else:
        commit_hash = "shared"

    subrun_id = "h-" + datetime.now().strftime("%Y%m%d_%H%M%S")

    suffix = f"-{job_suffix}" if job_suffix else ""
    array_job_name = f"spd-harvest{suffix}"

    worker_commands = []
    for rank in range(n_gpus):
        cmd = harvest_worker.get_command(
            config.config,
            rank=rank,
            world_size=n_gpus,
            subrun_id=subrun_id,
        )
        worker_commands.append(cmd)

    array_config = SlurmArrayConfig(
        job_name=array_job_name,
        partition=partition,
        n_gpus=1,
        time=time,
        snapshot_branch=snapshot_branch,
        dependency_job_id=dependency_job_id,
        comment=config.config.method_config.id,
    )
    array_script = generate_array_script(array_config, worker_commands)
    array_result = submit_slurm_job(
        array_script,
        "harvest_worker",
        is_array=True,
        n_array_tasks=n_gpus,
    )

    merge_command = harvest_merge.get_command(
        subrun_id,
        config.config,
    )
    merge_config = SlurmConfig(
        job_name="spd-harvest-merge",
        partition=partition,
        n_gpus=0,
        time=config.merge_time,
        mem=config.merge_mem,
        snapshot_branch=snapshot_branch,
        dependency_job_id=array_result.job_id,
        comment=config.config.method_config.id,
    )
    merge_script = generate_script(merge_config, merge_command)
    merge_result = submit_slurm_job(merge_script, "harvest_merge")

    logger.section("Harvest jobs submitted!")
    logger.values(
        {
            "Sub-run ID": subrun_id,
            "N batches": config.config.n_batches,
            "N GPUs": n_gpus,
            "Batch size": config.config.batch_size,
            "Snapshot": f"{snapshot_branch} ({commit_hash[:8]})",
            "Array Job ID": array_result.job_id,
            "Merge Job ID": merge_result.job_id,
            "Worker logs": array_result.log_pattern,
            "Merge log": merge_result.log_pattern,
        }
    )

    return HarvestSubmitResult(
        array_result=array_result,
        merge_result=merge_result,
        subrun_id=subrun_id,
    )
