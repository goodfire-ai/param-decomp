"""SLURM submission logic for investigation jobs."""

import json
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path

from spd.log import logger
from spd.settings import SPD_OUT_DIR
from spd.utils.git_utils import create_git_snapshot
from spd.utils.slurm import SlurmConfig, generate_script, submit_slurm_job
from spd.utils.wandb_utils import parse_wandb_run_path


@dataclass
class InvestigationResult:
    inv_id: str
    job_id: str
    output_dir: Path


def get_investigation_output_dir(inv_id: str) -> Path:
    return SPD_OUT_DIR / "investigations" / inv_id


def launch_investigation(
    wandb_path: str,
    prompt: str,
    context_length: int,
    max_turns: int,
    partition: str,
    time: str,
    job_suffix: str | None,
) -> InvestigationResult:
    """Launch a single investigation agent via SLURM.

    Creates a SLURM job that starts an isolated app backend, loads the SPD run,
    and launches a Claude Code agent with the given research question.
    """
    # Normalize wandb_path to canonical form (entity/project/run_id)
    entity, project, run_id = parse_wandb_run_path(wandb_path)
    canonical_wandb_path = f"{entity}/{project}/{run_id}"

    inv_id = f"inv-{secrets.token_hex(4)}"
    output_dir = get_investigation_output_dir(inv_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_branch, commit_hash = create_git_snapshot(inv_id)

    suffix = f"-{job_suffix}" if job_suffix else ""
    job_name = f"spd-investigate{suffix}"

    metadata = {
        "inv_id": inv_id,
        "wandb_path": canonical_wandb_path,
        "prompt": prompt,
        "context_length": context_length,
        "max_turns": max_turns,
        "snapshot_branch": snapshot_branch,
        "commit_hash": commit_hash,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    cmd = (
        f"{sys.executable} -m spd.investigate.scripts.run_agent "
        f'"{wandb_path}" '
        f"--inv_id {inv_id} "
        f"--context_length {context_length} "
        f"--max_turns {max_turns}"
    )

    slurm_config = SlurmConfig(
        job_name=job_name,
        partition=partition,
        n_gpus=1,
        time=time,
        snapshot_branch=snapshot_branch,
    )
    script = generate_script(slurm_config, cmd)
    result = submit_slurm_job(script, "investigate")

    logger.section("Investigation submitted")
    logger.values(
        {
            "Investigation ID": inv_id,
            "Job ID": result.job_id,
            "WandB path": canonical_wandb_path,
            "Prompt": prompt[:100] + ("..." if len(prompt) > 100 else ""),
            "Output directory": str(output_dir),
            "Logs": result.log_pattern,
        }
    )

    return InvestigationResult(inv_id=inv_id, job_id=result.job_id, output_dir=output_dir)
