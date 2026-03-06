"""SLURM launcher for dataset attribution harvesting.

Submits multi-GPU attribution jobs as a SLURM array, with a dependent merge job
that runs after all workers complete. Creates a git snapshot to ensure consistent
code across all workers even if jobs are queued.

Usage:
    spd-attributions <wandb_path> --n_gpus 24
    spd-attributions <wandb_path> --n_batches 1000 --n_gpus 8
"""

import secrets
from dataclasses import dataclass
from datetime import datetime

from spd.dataset_attributions.config import AttributionsSlurmConfig
from spd.dataset_attributions.scripts import run_merge, run_worker
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
from spd.utils.wandb_utils import wandb_path_to_url


@dataclass
class AttributionsSubmitResult:
    array_result: SubmitResult
    merge_result: SubmitResult
    subrun_id: str

    @property
    def job_id(self) -> str:
        return self.merge_result.job_id


def submit_attributions(
    wandb_path: str,
    config: AttributionsSlurmConfig,
    job_suffix: str | None = None,
    snapshot_branch: str | None = None,
    dependency_job_id: str | None = None,
    harvest_subrun_id: str | None = None,
) -> AttributionsSubmitResult:
    """Submit multi-GPU attribution harvesting job to SLURM.

    Submits a job array where each task processes a subset of batches, then
    submits a merge job that depends on all workers completing. Creates a git
    snapshot to ensure consistent code across all workers.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        config: Attribution SLURM configuration.
        job_suffix: Optional suffix for SLURM job names (e.g., "1h" -> "spd-attr-1h").
        snapshot_branch: Git snapshot branch to use. If None, creates a new snapshot.
        dependency_job_id: SLURM job to wait for before starting (e.g. harvest merge).
        harvest_subrun_id: Harvest subrun for alive masks. If None, uses most recent.

    Returns:
        AttributionsSubmitResult with array, merge results and subrun ID.
    """
    n_gpus = config.n_gpus
    partition = config.partition
    time = config.time

    if snapshot_branch is None:
        run_id = f"attr-{secrets.token_hex(4)}"
        snapshot_branch, commit_hash = create_git_snapshot(snapshot_id=run_id)
        logger.info(f"Created git snapshot: {snapshot_branch} ({commit_hash[:8]})")
    else:
        commit_hash = "shared"

    subrun_id = "da-" + datetime.now().strftime("%Y%m%d_%H%M%S")

    suffix = f"-{job_suffix}" if job_suffix else ""
    array_job_name = f"spd-attr{suffix}"

    config_json = config.config.model_dump_json(exclude_none=True)

    # SLURM arrays are 1-indexed, so task ID 1 -> rank 0, etc.
    worker_commands = []
    for rank in range(n_gpus):
        cmd = run_worker.get_command(
            wandb_path,
            config_json,
            rank=rank,
            world_size=n_gpus,
            subrun_id=subrun_id,
            harvest_subrun_id=harvest_subrun_id,
        )
        worker_commands.append(cmd)

    wandb_url = wandb_path_to_url(wandb_path)

    array_config = SlurmArrayConfig(
        job_name=array_job_name,
        partition=partition,
        n_gpus=1,  # 1 GPU per worker
        time=time,
        snapshot_branch=snapshot_branch,
        dependency_job_id=dependency_job_id,
        comment=wandb_url,
    )
    array_script = generate_array_script(array_config, worker_commands)
    array_result = submit_slurm_job(
        array_script,
        "attr_harvest",
        is_array=True,
        n_array_tasks=n_gpus,
    )

    # Submit merge job with dependency on array completion
    merge_cmd = run_merge.get_command(wandb_path, subrun_id)
    merge_config = SlurmConfig(
        job_name="spd-attr-merge",
        partition=partition,
        n_gpus=0,
        time=config.merge_time,
        mem=config.merge_mem,
        snapshot_branch=snapshot_branch,
        dependency_job_id=array_result.job_id,
        comment=wandb_url,
    )
    merge_script = generate_script(merge_config, merge_cmd)
    merge_result = submit_slurm_job(merge_script, "attr_merge")

    logger.section("Dataset attribution jobs submitted!")
    logger.values(
        {
            "WandB path": wandb_path,
            "Sub-run ID": subrun_id,
            "N batches": config.config.n_batches,
            "N GPUs": n_gpus,
            "Batch size": config.config.batch_size,
            "Snapshot": f"{snapshot_branch} ({commit_hash[:8]})",
            "Array Job ID": array_result.job_id,
            "Merge Job ID": merge_result.job_id,
            "Worker logs": array_result.log_pattern,
            "Merge log": merge_result.log_pattern,
            "Array script": str(array_result.script_path),
            "Merge script": str(merge_result.script_path),
        }
    )

    return AttributionsSubmitResult(
        array_result=array_result,
        merge_result=merge_result,
        subrun_id=subrun_id,
    )
