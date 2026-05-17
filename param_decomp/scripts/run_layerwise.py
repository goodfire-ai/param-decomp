"""Launcher for layerwise-split PD training.

Given an orchestrator YAML — a fully normalised `Config` whose `module_info` enumerates
every target module to decompose — fan out one independent training run per module.

Each per-module run gets a copy of the orchestrator `Config` with `module_info` filtered
to a single entry. Runs share a `wandb_group` so they cluster in the WandB UI.

The orchestrator must:
  - Set `wandb_group` (required so the runs are grouped).
  - List every module concretely in `module_info` (no `*` wildcards).
  - Use only layerwise-compatible loss configs: faithfulness, importance minimality, and
    `StochasticReconLayerwiseLoss`. Whole-model losses (`StochasticReconLoss`,
    `StochasticReconSubsetLoss`, `PersistentPGDReconLoss*`) are rejected — they defeat
    the point of the split.
"""

from datetime import datetime
from pathlib import Path

from param_decomp.configs import Config
from param_decomp.log import logger
from param_decomp.settings import PARAM_DECOMP_OUT_DIR
from param_decomp.utils.compute_utils import TrainingJob, create_slurm_script
from param_decomp.utils.git_utils import create_git_snapshot
from param_decomp.utils.run_utils import generate_run_id
from param_decomp.utils.slurm import submit_slurm_job
from param_decomp.utils.wandb_utils import get_wandb_run_url

LM_DECOMPOSITION_SCRIPT = Path("param_decomp/experiments/lm/lm_decomposition.py")

FORBIDDEN_LOSS_CLASSNAMES = {
    "StochasticReconLoss",
    "StochasticReconSubsetLoss",
    "PersistentPGDReconSubsetLoss",
}


def launch_layerwise_run(
    orchestrator_path: Path,
    partition: str,
    max_concurrent_tasks: int,
) -> None:
    """Fan out one single-GPU training run per module in the orchestrator config."""

    orchestrator = Config.from_file(orchestrator_path)
    _validate_orchestrator(orchestrator)
    base_group = orchestrator.wandb_group
    assert base_group is not None  # narrows for type checker; enforced by validator

    launch_id = f"layerwise-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    group_id = launch_id.removeprefix("layerwise-")
    wandb_group = f"{base_group}-{group_id}"

    logger.info(f"Launch ID: {launch_id}")
    logger.info(f"WandB group: {wandb_group}  (base: {base_group})")
    logger.info(f"Modules: {len(orchestrator.module_info)}")

    per_module_configs = _split_per_module(orchestrator, wandb_group)

    out_dir = PARAM_DECOMP_OUT_DIR / "layerwise_configs" / wandb_group
    out_dir.mkdir(parents=True, exist_ok=True)
    for cfg in per_module_configs:
        module_name = cfg.module_info[0].module_pattern
        cfg.to_file(out_dir / f"{module_name}.yaml")
    logger.info(f"Wrote {len(per_module_configs)} per-module configs to {out_dir}")

    training_jobs = [
        TrainingJob(
            experiment="layerwise",
            script_path=LM_DECOMPOSITION_SCRIPT,
            config=cfg,
            run_id=generate_run_id("param_decomp"),
        )
        for cfg in per_module_configs
    ]

    snapshot_ref, commit_hash = create_git_snapshot(snapshot_id=launch_id)
    logger.info(f"Created git snapshot ref: {snapshot_ref} ({commit_hash[:8]})")

    wandb_urls = [
        get_wandb_run_url(orchestrator.wandb_project, job.run_id)
        if orchestrator.wandb_project is not None
        else ""
        for job in training_jobs
    ]

    script_content = create_slurm_script(
        slurm_job_name=f"pd-layerwise-{wandb_group}",
        launch_id=launch_id,
        training_jobs=training_jobs,
        sweep_params=None,
        snapshot_ref=snapshot_ref,
        n_gpus=None,
        partition=partition,
        max_concurrent_tasks=max_concurrent_tasks,
        per_task_comments=wandb_urls,
    )

    result = submit_slurm_job(
        script_content,
        f"layerwise_{launch_id}",
        is_array=True,
        n_array_tasks=len(training_jobs),
    )

    logger.section("Job submitted successfully!")
    summary: dict[str, str | int | None] = {
        "Array Job ID": result.job_id,
        "Total training jobs": len(training_jobs),
        "Max concurrent tasks": max_concurrent_tasks,
        "View logs in": result.log_pattern,
        "Script": str(result.script_path),
        "Per-module configs": str(out_dir),
    }
    if len(wandb_urls) <= 10:
        summary["WandB run URLs"] = "\n" + "\n".join(f"  - {u}" for u in wandb_urls)
    logger.values(summary)


def _validate_orchestrator(config: Config) -> None:
    assert config.wandb_group is not None, "orchestrator config must set wandb_group"
    assert len(config.module_info) >= 1, "orchestrator must enumerate at least one module"

    for info in config.module_info:
        assert "*" not in info.module_pattern, (
            f"orchestrator module patterns must be concrete (no wildcards), "
            f"got {info.module_pattern!r}"
        )

    seen: set[str] = set()
    for info in config.module_info:
        assert info.module_pattern not in seen, (
            f"duplicate module pattern in orchestrator: {info.module_pattern!r}"
        )
        seen.add(info.module_pattern)

    for cfg in config.loss_metric_configs:
        assert cfg.classname not in FORBIDDEN_LOSS_CLASSNAMES, (
            f"loss {cfg.classname!r} cannot appear in a layerwise orchestrator — "
            f"it requires whole-model joint training and defeats the per-module split"
        )


def _split_per_module(orchestrator: Config, wandb_group: str) -> list[Config]:
    assert orchestrator.wandb_group is not None
    base_group = orchestrator.wandb_group
    return [
        orchestrator.model_copy(
            update={
                "module_info": [info],
                "wandb_group": wandb_group,
                "wandb_run_name": info.module_pattern,
                "wandb_tags": [
                    *orchestrator.wandb_tags,
                    *_module_dimensions(info.module_pattern).values(),
                ],
                "extra_wandb_config": {
                    **orchestrator.extra_wandb_config,
                    **_module_dimensions(info.module_pattern),
                    "group_base": base_group,
                },
            }
        )
        for info in orchestrator.module_info
    ]


def _module_dimensions(module_pattern: str) -> dict[str, str]:
    """Parse a concrete module path like `h.0.mlp.c_fc` into group-by dimensions."""
    parts = module_pattern.split(".")
    assert parts[0] == "h" and len(parts) >= 3, (
        f"unexpected module pattern (need h.<block>.<type>...): {module_pattern!r}"
    )
    return {"block": f"L{parts[1]}", "module_type": ".".join(parts[2:])}
