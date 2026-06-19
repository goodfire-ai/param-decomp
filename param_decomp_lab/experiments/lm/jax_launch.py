"""Submit a JAX single-pool (`jsp-train`) run to SLURM with torch-launch parity.

Mints the `p-<8hex>` run id, snapshots the working tree to `refs/runs/snapshot/<id>`,
materializes the snapshot as a shared-FS workspace (clone + both venvs, built at
submit time on the login node — `jsp-train` runs 8 srun tasks per node, so in-job
per-node cloning would race), stamps the run id (+ out_dir / wandb group / tags) into
the workspace's single config yaml, and sbatches. Requeues re-enter the same immutable
workspace.

The submit side needs no JAX imports, so it lives with the other `pd-*` scripts and
runs from the torch venv.
"""

import os
import shlex
import subprocess
from pathlib import Path

import fire
import yaml

from param_decomp.log import logger
from param_decomp_config.lm import LMExperimentConfig
from param_decomp_config.resid_mlp import ResidMLPExperimentConfig
from param_decomp_config.tms import TMSExperimentConfig
from param_decomp_lab.infra.git import create_git_snapshot
from param_decomp_lab.infra.run_files import generate_run_id
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR, REPO_ROOT
from param_decomp_lab.infra.slurm import SlurmConfig, generate_script, submit_slurm_job
from param_decomp_lab.infra.wandb import get_wandb_entity

AnyRunConfig = LMExperimentConfig | TMSExperimentConfig | ResidMLPExperimentConfig

GPUS_PER_NODE = 8
WORKSPACES_DIR = PARAM_DECOMP_OUT_DIR / "workspaces"

# uv materializes a venv by hardlinking wheels from its cache when cache and venv share
# one filesystem, else copying. The default per-user cache (~/.cache/uv) sits on the home
# mount while workspaces live on the data mount — different PVCs — so every build copied
# multi-GB CUDA wheels. Pinning the cache beside the workspaces (same PVC) makes the
# venv materialization metadata-only hardlinks. Scoped to the build subprocesses below so
# other uv usage on the cluster keeps its default cache. uv still auto-falls-back to copy
# if this ever resolves cross-filesystem, so the change is safe everywhere.
UV_CACHE_DIR = PARAM_DECOMP_OUT_DIR / "uv_cache"

# Mirrors the validated llama8b.sbatch srun line: one task per GPU, block placement.
_SRUN_FLAGS = (
    "--kill-on-bad-exit=1 --ntasks-per-node=8 --cpus-per-task=8 --distribution=block:block"
)

# Default 0.75 caps the XLA pool too low for production steps (OOM, job 50644);
# CUDA-graph capture (XLA command buffers) intermittently dies with
# CUDA_ERROR_STREAM_CAPTURE_INVALIDATED on disjoint allocations — disabling
# measured ~0% cost (8,007 vs 8,015 tok/s/GPU).
_RANK_ENV = """\
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export XLA_FLAGS="--xla_gpu_enable_command_buffer=\""""


def main(
    config_path: str,
    *,
    nodes: int,
    time: str = "72:00:00",
    qos: str | None = None,
    run_id: str | None = None,
    group: str | None = None,
    tags: str | None = None,
    comment: str | None = None,
    allocator: str | None = None,
) -> None:
    """Submit a jsp-train run.

    Args:
        config_path: Single self-contained run yaml (the canonical schema + top-level
            `run_name`, optional `out_dir`), inside the repo. `run_id` and `out_dir`
            are minted here and stamped into the workspace copy; `out_dir` defaults to
            `PARAM_DECOMP_OUT_DIR/runs` (the current cluster) when absent.
        nodes: Node count (8 GPUs each).
        time: SLURM time limit.
        qos: SLURM QoS (e.g. `opportunistic`); None is the normal QoS.
        run_id: Resubmit an existing launch — reuses its workspace (and identity)
            instead of building a new one. `group`/`tags` are ignored on resubmit
            (the original workspace config already carries them).
        group: wandb UI group (no-op when the config omits `wandb:`).
        tags: Comma-separated wandb tags (no-op when `wandb:` is omitted).
        comment: SLURM `--comment`; defaults to the wandb run URL (or run id).
        allocator: `XLA_PYTHON_CLIENT_ALLOCATOR` override (e.g. `platform` for the
            on-demand cudaMalloc allocator, which avoids BFC fragmentation OOMs on
            runs near the HBM cap, at some per-alloc cost). None leaves the default BFC.
    """
    config_rel = _config_path_relative_to_repo(config_path)
    cfg, run_name = _validate_config(REPO_ROOT / config_rel)
    tag_list = [s.strip() for s in tags.split(",")] if tags is not None else []

    if run_id is None:
        run_id = generate_run_id("param_decomp")
        snapshot_ref, commit_hash = create_git_snapshot(snapshot_id=run_id)
        logger.info(f"Created git snapshot: {snapshot_ref} ({commit_hash[:8]})")
        workspace = WORKSPACES_DIR / run_id
        _build_workspace(workspace, snapshot_ref, run_id, config_rel, group, tag_list)
    else:
        snapshot_ref = f"refs/runs/snapshot/{run_id}"
        workspace = WORKSPACES_DIR / run_id
        assert workspace.exists(), f"no workspace to resubmit: {workspace}"

    wandb_url = _wandb_url(cfg, run_id)
    job_name = f"jsp-{run_name}"
    slurm_config = SlurmConfig(
        job_name=job_name,
        partition=None,
        qos=qos,
        n_gpus=GPUS_PER_NODE,
        n_nodes=nodes,
        ntasks_per_node=GPUS_PER_NODE,
        cpus_per_task=8,
        time=time,
        signal="TERM@300",
        requeue=True,
        comment=comment if comment is not None else (wandb_url or run_id),
    )
    jax_dir = workspace / "param_decomp_jax"
    rank_env = _RANK_ENV
    if allocator is not None:
        rank_env = f"{rank_env}\nexport XLA_PYTHON_CLIENT_ALLOCATOR={allocator}"
    rank_command = f"source .venv-cuda/bin/activate\n{rank_env}\nexec jsp-train {config_rel.relative_to('param_decomp_jax')}"
    command = f"srun {_SRUN_FLAGS} bash -c {shlex.quote(rank_command)}"
    script = generate_script(slurm_config, command, setup=f'cd "{jax_dir}"')
    result = submit_slurm_job(script, "jax-lm")

    logger.section("JAX single-pool job submitted!")
    summary: dict[str, str | None] = {
        "Run ID": run_id,
        "Run name": run_name,
        "Job ID": result.job_id,
        "Log file": result.log_pattern,
        "Script": str(result.script_path),
        "Snapshot": snapshot_ref,
        "Workspace": str(workspace),
    }
    if wandb_url is not None:
        summary["WandB run URL"] = wandb_url
    logger.values(summary)


def _config_path_relative_to_repo(config_path: str) -> Path:
    path = Path(config_path).resolve()
    assert path.exists(), f"config not found: {path}"
    assert path.is_relative_to(REPO_ROOT), (
        f"config must live inside the repo so the snapshot carries it: {path}"
    )
    rel = path.relative_to(REPO_ROOT)
    assert rel.parts[0] == "param_decomp_jax", (
        f"config must live under param_decomp_jax/ (jsp-train runs from there): {rel}"
    )
    return rel


def _validate_config(config_path: Path) -> tuple[AnyRunConfig, str]:
    """Validate the not-yet-stamped single run config against the shared torch-free
    schema, dispatching on the structural target marker (`n_hidden` → TMS, `d_embed` →
    ResidMLP, else LM) — the same dispatch the runtime loader uses. The loader's module
    pulls jax and can't be imported in this venv, but both read the same
    `param_decomp_config` schema. A hand-authored config must NOT carry `run_id` (minted
    at submit)."""
    raw = yaml.safe_load(config_path.read_text())
    assert "run_id" not in raw, f"{config_path}: run_id is minted at submit, omit it"
    target = raw.get("target", {})
    assert isinstance(target, dict), target
    if "n_hidden" in target:
        cfg: AnyRunConfig = TMSExperimentConfig(**raw)
    elif "d_embed" in target:
        cfg = ResidMLPExperimentConfig(**raw)
    else:
        cfg = LMExperimentConfig(**raw)
    return cfg, cfg.run_name


def _build_workspace(
    workspace: Path,
    snapshot_ref: str,
    run_id: str,
    config_rel: Path,
    group: str | None,
    tags: list[str],
) -> None:
    """Materialize the snapshot as an immutable shared-FS checkout with both venvs, then
    stamp the run identity (run_id, out_dir-if-absent, wandb group/tags) into the
    workspace's single config yaml."""
    assert not workspace.exists(), f"workspace already exists: {workspace}"
    workspace.parent.mkdir(parents=True, exist_ok=True)
    UV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    build_env = {**os.environ, "UV_CACHE_DIR": str(UV_CACHE_DIR)}

    def run(args: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
        subprocess.run(args, cwd=cwd, check=True, env=env)

    logger.info(f"Building workspace {workspace} ...")
    run(["git", "clone", "--quiet", str(REPO_ROOT), str(workspace)], cwd=REPO_ROOT)
    run(
        ["git", "fetch", "--quiet", str(REPO_ROOT), f"{snapshot_ref}:{snapshot_ref}"], cwd=workspace
    )
    run(["git", "checkout", "--quiet", snapshot_ref], cwd=workspace)
    env_file = REPO_ROOT / ".env"
    assert env_file.exists(), f".env with wandb credentials required: {env_file}"
    (workspace / ".env").write_bytes(env_file.read_bytes())

    logger.info("torch venv: uv sync --all-packages --no-dev ...")
    run(["uv", "sync", "--all-packages", "--no-dev", "-q"], cwd=workspace, env=build_env)
    logger.info("jax venv: make install-jax-cuda ...")
    run(["make", "install-jax-cuda"], cwd=workspace, env=build_env)

    _stamp_config(workspace / config_rel, run_id, group, tags)


def _stamp_config(config: Path, run_id: str, group: str | None, tags: list[str]) -> None:
    """Stamp the minted run identity into the workspace's single config yaml: top-level
    `run_id`, `out_dir` (minted when the author left it absent), and the wandb UI knobs
    onto the `wandb:` block (no-op when `wandb:` is omitted)."""
    raw = yaml.safe_load(config.read_text())
    assert "run_id" not in raw, "run_id already stamped"
    raw["run_id"] = run_id
    if raw.get("out_dir") is None:
        raw["out_dir"] = str(PARAM_DECOMP_OUT_DIR / "runs")
    if group is not None or tags:
        assert raw.get("wandb") is not None, "wandb group/tags need a wandb: block in the config"
        if group is not None:
            raw["wandb"]["group"] = group
        if tags:
            raw["wandb"]["tags"] = tags
    config.write_text(yaml.safe_dump(raw, sort_keys=False))


def _wandb_url(cfg: AnyRunConfig, run_id: str) -> str | None:
    if cfg.wandb is None:
        return None
    entity = cfg.wandb.entity or get_wandb_entity()
    return f"https://wandb.ai/{entity}/{cfg.wandb.project}/runs/{run_id}"


def cli() -> None:
    fire.Fire(main)
