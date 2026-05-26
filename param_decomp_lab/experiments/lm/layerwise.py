"""Layerwise per-matrix LM PD launcher (MVP).

Takes a single LM experiment YAML and emits one config per (layer, module_pattern) pair —
each output config has exactly one entry in `pd.decomposition_targets`, with the pattern's
``*`` replaced by a concrete layer index. Submits a SLURM array of `pd-lm` jobs over the
generated configs.

Usage:
    pd-lm-layerwise <base_config.yaml> --n_layers 4
    pd-lm-layerwise <base_config.yaml> --n_layers 4 --include q_proj,k_proj
    pd-lm-layerwise <base_config.yaml> --n_layers 4 --layers 0,2 --no_snapshot
"""

import secrets
from datetime import datetime
from pathlib import Path

import fire
import wandb_workspaces.workspaces as ws

from param_decomp.decomposition_targets import DecompositionTargetConfig
from param_decomp.log import logger
from param_decomp_lab.experiments.lm.run import LMExperimentConfig
from param_decomp_lab.infra.git import create_git_snapshot
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME, PARAM_DECOMP_OUT_DIR
from param_decomp_lab.infra.slurm import (
    SlurmArrayConfig,
    generate_array_script,
    submit_slurm_job,
)
from param_decomp_lab.infra.wandb import get_wandb_entity


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [s.strip() for s in value.split(",") if s.strip()]


def _parse_int_csv(value: str | None) -> list[int] | None:
    parts = _parse_csv(value)
    if parts is None:
        return None
    return [int(s) for s in parts]


def _substitute_pattern(pattern: str, layer_idx: int) -> str:
    """Replace the single ``*`` in a glob with a concrete layer index.

    MVP assumes one wildcard per pattern (the layer slot). Patterns with zero or multiple
    wildcards are not supported here — split those by hand.
    """
    assert pattern.count("*") == 1, (
        f"pd-lm-layerwise expects exactly one '*' per module_pattern, got: {pattern!r}"
    )
    return pattern.replace("*", str(layer_idx))


def _build_configs(
    base_cfg: LMExperimentConfig,
    *,
    n_layers: int,
    include: list[str] | None,
    layers: list[int] | None,
) -> list[tuple[str, LMExperimentConfig]]:
    """Cross base `decomposition_targets` with the requested layer indices.

    Returns a list of (job_tag, per-matrix config) pairs. The tag is used for filenames and
    SLURM comments. `include` filters patterns by substring (e.g. ``["q_proj", "k_proj"]``);
    `layers` restricts to specific layer indices.
    """
    base_targets = base_cfg.pd.decomposition_targets
    assert base_targets, "base config has no decomposition_targets to split"
    if base_cfg.pd.identity_decomposition_targets:
        # Identity targets attach hooks to extra modules and don't fit the per-matrix MVP cleanly.
        # Bail out instead of silently dropping them.
        raise ValueError(
            "pd-lm-layerwise does not support identity_decomposition_targets; "
            "drop them from the base config first"
        )

    selected_targets = (
        base_targets
        if include is None
        else [t for t in base_targets if any(s in t.module_pattern for s in include)]
    )
    assert selected_targets, (
        f"--include={include!r} matched no patterns in base config "
        f"(have: {[t.module_pattern for t in base_targets]})"
    )

    selected_layers = list(range(n_layers)) if layers is None else layers
    for li in selected_layers:
        assert 0 <= li < n_layers, f"layer index {li} out of range for n_layers={n_layers}"

    out: list[tuple[str, LMExperimentConfig]] = []
    for layer_idx in selected_layers:
        for target in selected_targets:
            resolved = _substitute_pattern(target.module_pattern, layer_idx)
            new_target = DecompositionTargetConfig(module_pattern=resolved, C=target.C)
            new_pd = base_cfg.pd.model_copy(update={"decomposition_targets": [new_target]})
            new_cfg = base_cfg.model_copy(update={"pd": new_pd})
            out.append((resolved, new_cfg))
    return out


def submit_lm_layerwise(
    base_config: str | Path,
    *,
    n_layers: int,
    include: list[str] | None,
    layers: list[int] | None,
    tags: list[str] | None,
    partition: str | None,
    time: str,
    max_concurrent: int | None,
    no_snapshot: bool,
) -> None:
    """Generate per-matrix configs and submit them as a SLURM array of pd-lm jobs."""
    base_cfg = LMExperimentConfig.from_file(base_config)
    per_matrix = _build_configs(
        base_cfg,
        n_layers=n_layers,
        include=include,
        layers=layers,
    )

    run_id = "lw-" + datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + secrets.token_hex(2)
    run_dir = PARAM_DECOMP_OUT_DIR / "layerwise" / run_id
    configs_dir = run_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    tags_csv = ",".join(tags) if tags else None
    commands: list[str] = []
    per_task_comments: list[str] = []
    for tag, cfg in per_matrix:
        cfg_path = configs_dir / f"{tag}.yaml"
        cfg.to_file(cfg_path)
        cmd = f"pd-lm {cfg_path} --group={run_id}"
        if tags_csv:
            cmd += f" --tags={tags_csv}"
        commands.append(cmd)
        per_task_comments.append(tag)

    snapshot_ref: str | None = None
    commit_hash = "no-snapshot"
    if not no_snapshot:
        snapshot_ref, commit_hash = create_git_snapshot(snapshot_id=run_id)
        logger.info(f"Created git snapshot: {snapshot_ref} ({commit_hash[:8]})")

    array_config = SlurmArrayConfig(
        job_name="pd-lm-layerwise",
        partition=partition,
        n_gpus=1,
        time=time,
        snapshot_ref=snapshot_ref,
        max_concurrent_tasks=max_concurrent,
        comment=run_id,
    )
    array_script = generate_array_script(
        array_config,
        commands,
        per_task_comments=per_task_comments,
    )
    result = submit_slurm_job(array_script, "lm_layerwise", n_array_tasks=len(commands))

    workspace_url = (
        _create_layerwise_workspace_view(run_id, base_cfg.wandb.project)
        if base_cfg.wandb is not None
        else "(none — base config has no wandb block)"
    )

    logger.section("Layerwise PD jobs submitted!")
    logger.values(
        {
            "Run ID": run_id,
            "Run dir": str(run_dir),
            "N configs": len(commands),
            "Snapshot": f"{snapshot_ref} ({commit_hash[:8]})" if snapshot_ref else "(none)",
            "Array Job ID": result.job_id,
            "Logs": result.log_pattern,
            "W&B workspace": workspace_url,
        }
    )


def _create_layerwise_workspace_view(run_id: str, project: str) -> str:
    """Create a W&B workspace view that collects the layerwise array's per-matrix runs.

    Each subjob is invoked with ``--group=<run_id>``; this workspace filters on that
    field so the whole sweep is browsable in one place.
    """
    workspace = ws.Workspace(entity=get_wandb_entity(), project=project)
    workspace.name = f"Layerwise - {run_id}"
    workspace.runset_settings.filters = [ws.Metric("Group").isin([run_id])]
    workspace.save_as_new_view()
    return workspace.url


def main(
    base_config: str,
    n_layers: int,
    include: str | None = None,
    layers: str | None = None,
    tags: str | None = None,
    partition: str | None = DEFAULT_PARTITION_NAME,
    time: str = "12:00:00",
    max_concurrent: int | None = None,
    no_snapshot: bool = False,
) -> None:
    """CLI shim — Fire-friendly types, then delegate to `submit_lm_layerwise`.

    Args:
        base_config: Path to an LM experiment YAML to split.
        n_layers: Number of layers in the target model (used to expand `*` in patterns).
        include: Comma-separated substrings; keep only base patterns containing one of them
            (e.g. "q_proj,k_proj"). Default: keep all base patterns.
        layers: Comma-separated layer indices to include (e.g. "0,2,3"). Default: all layers
            in [0, n_layers).
        tags: Comma-separated wandb tags propagated to every child run (in addition to
            the auto-generated launch-id `--group`).
        partition: SLURM partition for the array job.
        time: SLURM time limit per task (HH:MM:SS).
        max_concurrent: Cap on concurrent array tasks. Default: no cap.
        no_snapshot: Skip git snapshot; SLURM jobs will cd into the live worktree instead.
    """
    submit_lm_layerwise(
        base_config=base_config,
        n_layers=n_layers,
        include=_parse_csv(include),
        layers=_parse_int_csv(layers),
        tags=_parse_csv(tags),
        partition=partition,
        time=time,
        max_concurrent=max_concurrent,
        no_snapshot=no_snapshot,
    )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
