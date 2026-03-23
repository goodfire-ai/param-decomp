"""Submit all 28 unharvested 4k TC/CLT runs as a single compact SLURM allocation.

Requests N nodes exclusively, then fans out 28 harvest workers via srun (1 GPU each).
No merge step needed since each worker is single-GPU.

Usage:
    python scripts/submit_batch_harvest.py
    python scripts/submit_batch_harvest.py --n_nodes 4 --time 12:00:00
    python scripts/submit_batch_harvest.py --dry_run  # Print script without submitting
"""

import secrets
from datetime import datetime
from pathlib import Path

import fire
import yaml

from spd.harvest.config import HarvestConfig
from spd.harvest.scripts.run_worker import get_command
from spd.log import logger
from spd.utils.git_utils import create_git_snapshot
from spd.utils.slurm import (
    SLURM_LOGS_DIR,
    SlurmConfig,
    generate_git_snapshot_setup,
    submit_slurm_job,
)

CONFIG_DIR = Path(__file__).parent / "harvest_configs"


def main(n_nodes: int = 4, time: str = "14:00:00", dry_run: bool = False) -> None:
    config_paths = sorted(CONFIG_DIR.glob("*.yaml"))
    assert len(config_paths) > 0, f"No configs found in {CONFIG_DIR}"

    n_gpus_needed = len(config_paths)
    n_gpus_available = n_nodes * 8
    assert n_gpus_available >= n_gpus_needed, (
        f"Need {n_gpus_needed} GPUs but {n_nodes} nodes only provide {n_gpus_available}"
    )

    logger.info(f"Found {len(config_paths)} harvest configs")
    logger.info(f"Requesting {n_nodes} nodes ({n_gpus_available} GPUs) for {n_gpus_needed} workers")

    snapshot_id = f"batch-harvest-{secrets.token_hex(4)}"
    if not dry_run:
        snapshot_branch, commit_hash = create_git_snapshot(snapshot_id=snapshot_id)
        logger.info(f"Created git snapshot: {snapshot_branch} ({commit_hash[:8]})")
    else:
        snapshot_branch = "DRY_RUN"

    subrun_id = "h-" + datetime.now().strftime("%Y%m%d_%H%M%S")

    srun_commands = []
    for cfg_path in config_paths:
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        config = HarvestConfig.model_validate(raw["config"])
        cmd = get_command(config, rank=0, world_size=1, subrun_id=subrun_id)
        label = cfg_path.stem
        srun_line = (
            f'srun --exact -N1 -n1 --gpus-per-task=1 '
            f'bash -c "{cmd}" &'
        )
        srun_commands.append(f"echo 'Starting {label}...'\n{srun_line}")

    workers_block = "\n\n".join(srun_commands)

    setup = generate_git_snapshot_setup(
        work_dir="/tmp/spd/workspace-batch-harvest-$SLURM_JOB_ID",
        snapshot_branch=snapshot_branch,
    )

    script = f"""\
#!/bin/bash
#SBATCH --job-name=spd-batch-harvest
#SBATCH --nodes={n_nodes}
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time={time}
#SBATCH --output={SLURM_LOGS_DIR}/slurm-%j.out
#SBATCH --comment=batch-harvest-28x-4k-tc-clt

set -euo pipefail
umask 002

{setup}

echo "Allocation: $SLURM_JOB_NODELIST ({n_nodes} nodes, {n_gpus_available} GPUs)"
echo "Launching {n_gpus_needed} harvest workers (subrun: {subrun_id})"
echo "---"

{workers_block}

echo "All {n_gpus_needed} workers launched, waiting..."
wait
echo "All workers finished!"
"""

    if dry_run:
        print(script)
        return

    result = submit_slurm_job(script, "batch_harvest")
    logger.section("Batch harvest submitted!")
    logger.values({
        "Job ID": result.job_id,
        "Nodes": n_nodes,
        "Workers": n_gpus_needed,
        "Sub-run ID": subrun_id,
        "Snapshot": snapshot_branch,
        "Log": result.log_pattern,
    })


if __name__ == "__main__":
    fire.Fire(main)
