"""SLURM launcher for autointerp pipeline.

Autointerp is a functional unit: interpret + label-dependent evals. This module
submits all jobs in the unit with proper dependency chaining.

Dependency graph (depends on a prior harvest merge):
    interpret         (depends on harvest merge)
    ├── detection     (depends on interpret)
    └── fuzzing       (depends on interpret)

(Intruder eval is label-free and belongs to the harvest functional unit.)
"""

import re
from dataclasses import dataclass
from datetime import datetime

from spd.autointerp.config import AutointerpSlurmConfig
from spd.autointerp.scoring.scripts import run_label_scoring
from spd.autointerp.scripts import run_interpret
from spd.log import logger
from spd.utils.slurm import SlurmConfig, SubmitResult, generate_script, submit_slurm_job


@dataclass
class AutointerpSubmitResult:
    autointerp_subrun_id: str
    interpret_result: SubmitResult
    detection_result: SubmitResult | None
    fuzzing_result: SubmitResult | None


def _make_autointerp_subrun_id(snapshot_branch: str | None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if snapshot_branch is None:
        return f"a-{timestamp}"
    branch_tag = re.sub(r"[^a-zA-Z0-9]+", "-", snapshot_branch).strip("-").lower()
    return f"a-{timestamp}-{branch_tag}"[:120]


def submit_autointerp(
    decomposition_id: str,
    config: AutointerpSlurmConfig,
    harvest_subrun_id: str,
    dependency_job_id: str | None = None,
    snapshot_branch: str | None = None,
) -> AutointerpSubmitResult:
    """Submit the autointerp pipeline to SLURM.

    Submits interpret + eval jobs as a functional unit. All jobs depend on a
    prior harvest merge (passed as dependency_job_id).

    Args:
        wandb_path: WandB run path for the target decomposition run.
        config: Autointerp SLURM configuration.
        dependency_job_id: Job to wait for before starting (e.g. harvest merge).
        snapshot_branch: Git snapshot branch to use.

    Returns:
        AutointerpSubmitResult with interpret, detection, and fuzzing results.
    """
    autointerp_subrun_id = _make_autointerp_subrun_id(snapshot_branch)

    # === 1. Interpret job ===
    interpret_cmd = run_interpret.get_command(
        decomposition_id=decomposition_id,
        config=config.config,
        harvest_subrun_id=harvest_subrun_id,
        autointerp_subrun_id=autointerp_subrun_id,
    )

    interpret_slurm = SlurmConfig(
        job_name="spd-interpret",
        partition=config.partition,
        n_gpus=2,
        time=config.time,
        snapshot_branch=snapshot_branch,
        dependency_job_id=dependency_job_id,
        comment=decomposition_id,
    )
    script_content = generate_script(interpret_slurm, interpret_cmd)
    interpret_result = submit_slurm_job(script_content, "spd-interpret")

    logger.section("Interpret job submitted")
    logger.values(
        {
            "Job ID": interpret_result.job_id,
            "Decomposition ID": decomposition_id,
            "Autointerp Subrun": autointerp_subrun_id,
            "Model": config.config.llm.model,
            "Log": interpret_result.log_pattern,
        }
    )

    if config.evals is None:
        return AutointerpSubmitResult(
            autointerp_subrun_id=autointerp_subrun_id,
            interpret_result=interpret_result,
            detection_result=None,
            fuzzing_result=None,
        )

    # === 2. Detection + fuzzing scoring (depend on interpret) ===
    scoring_results: dict[str, SubmitResult] = {}
    for scorer in ("detection", "fuzzing"):
        scoring_cmd = run_label_scoring.get_command(
            decomposition_id,
            scorer_type=scorer,
            config=config.evals,
            harvest_subrun_id=harvest_subrun_id,
            autointerp_subrun_id=autointerp_subrun_id,
        )
        eval_slurm = SlurmConfig(
            job_name=f"spd-{scorer}",
            partition=config.partition,
            n_gpus=2,
            time=config.evals_time,
            snapshot_branch=snapshot_branch,
            dependency_job_id=interpret_result.job_id,
        )
        eval_script = generate_script(eval_slurm, scoring_cmd)
        scoring_result = submit_slurm_job(eval_script, f"spd-{scorer}")
        scoring_results[scorer] = scoring_result

        logger.section(f"{scorer.capitalize()} scoring job submitted")
        logger.values(
            {
                "Job ID": scoring_result.job_id,
                "Depends on": f"interpret ({interpret_result.job_id})",
                "Autointerp Subrun": autointerp_subrun_id,
                "Log": scoring_result.log_pattern,
            }
        )

    return AutointerpSubmitResult(
        autointerp_subrun_id=autointerp_subrun_id,
        interpret_result=interpret_result,
        detection_result=scoring_results["detection"],
        fuzzing_result=scoring_results["fuzzing"],
    )
