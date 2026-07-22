"""Unified postprocessing pipeline for decomposition runs.

Submits all postprocessing steps to SLURM with proper dependency chaining.
All steps always run — data accumulates (harvest upserts, autointerp resumes).

Dependency graph:
    harvest             (GPU array -> merge, GPU, PD-only)
    ├── intruder eval   (CPU, depends on harvest merge, label-free)
    └── autointerp      (CPU, LLM calls, resumes via completed keys)
        ├── detection   (CPU, label-dependent)
        └── fuzzing     (CPU, label-dependent)
"""

import secrets
from datetime import datetime
from pathlib import Path

import yaml

from param_decomp.log import logger
from param_decomp_lab.autointerp.scripts.run_slurm import AutointerpSubmitResult, submit_autointerp
from param_decomp_lab.harvest.scripts import run_intruder
from param_decomp_lab.harvest.scripts.run_slurm import submit_harvest
from param_decomp_lab.infra.git import create_git_snapshot
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.infra.slurm import (
    SlurmConfig,
    SubmitResult,
    generate_script,
    submit_slurm_job,
)
from param_decomp_lab.postprocess.config import PostprocessConfig


def postprocess(config: PostprocessConfig, dependency_job_id: str | None = None) -> Path:
    """Submit all postprocessing jobs with SLURM dependency chaining.

    Args:
        config: Postprocessing configuration.
        dependency_job_id: SLURM job to wait for before starting harvest
            (e.g. a training job that must complete first).

    Returns:
        Path to the metadata YAML file.
    """

    snapshot_ref, commit_hash = create_git_snapshot(f"postprocess-{secrets.token_hex(4)}")
    logger.info(f"Created git snapshot: {snapshot_ref} ({commit_hash[:8]})")

    decomp_cfg = config.harvest.config.method_config

    # === 1. Harvest (always runs, upserts into harvest.db) ===
    harvest_result = submit_harvest(
        config.harvest,
        snapshot_ref=snapshot_ref,
        dependency_job_id=dependency_job_id,
    )

    # === 2. Autointerp (depends on harvest, resumes via completed keys) ===
    autointerp_result: AutointerpSubmitResult | None = None
    if config.autointerp is not None:
        autointerp_result = submit_autointerp(
            decomposition_id=decomp_cfg.id,
            config=config.autointerp,
            dependency_job_id=harvest_result.merge_result.job_id,
            snapshot_ref=snapshot_ref,
            harvest_subrun_id=harvest_result.subrun_id,
        )

    # === 3. Intruder eval (depends on harvest merge, label-free) ===
    intruder_result: SubmitResult | None = None
    if config.intruder is not None:
        intruder_cmd = run_intruder.get_command(
            decomposition_id=decomp_cfg.id,
            config=config.intruder.config,
            harvest_subrun_id=harvest_result.subrun_id,
        )

        intruder_slurm = SlurmConfig(
            job_name="pd-intruder-eval",
            partition=config.intruder.partition,
            n_gpus=2,
            time=config.intruder.time,
            snapshot_ref=snapshot_ref,
            dependency_job_id=harvest_result.merge_result.job_id,
        )
        intruder_script = generate_script(intruder_slurm, intruder_cmd)
        intruder_result = submit_slurm_job(intruder_script, "intruder_eval")

        logger.section("Intruder eval job submitted")
        logger.values(
            {
                "Job ID": intruder_result.job_id,
                "Depends on": f"harvest merge ({harvest_result.merge_result.job_id})",
                "Log": intruder_result.log_pattern,
            }
        )

    # === Write metadata ===
    metadata_id = "pp-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_dir = PARAM_DECOMP_OUT_DIR / "postprocess" / metadata_id
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / "metadata.yaml"

    jobs: dict[str, str] = {
        "harvest_array": harvest_result.array_result.job_id,
        "harvest_merge": harvest_result.merge_result.job_id,
        "harvest_subrun": harvest_result.subrun_id,
    }
    if intruder_result is not None:
        jobs["intruder_eval"] = intruder_result.job_id
    if autointerp_result is not None:
        jobs["interpret"] = autointerp_result.interpret_result.job_id
        if autointerp_result.detection_result is not None:
            jobs["detection"] = autointerp_result.detection_result.job_id
        if autointerp_result.fuzzing_result is not None:
            jobs["fuzzing"] = autointerp_result.fuzzing_result.job_id

    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "decomposition": config.harvest.config.method_config.model_dump(),
        "snapshot_ref": snapshot_ref,
        "commit_hash": commit_hash,
        "config": config.model_dump(),
        "jobs": jobs,
    }

    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    logger.section("Postprocess metadata saved")
    logger.info(str(metadata_path))

    return metadata_path
