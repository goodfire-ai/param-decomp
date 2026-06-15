"""Submit a JAX single-pool (`jsp-train`) run to SLURM with torch-launch parity.

Mints the `p-<8hex>` run id, snapshots the working tree to `refs/runs/snapshot/<id>`,
materializes the snapshot as a shared-FS workspace (clone + both venvs, built at
submit time on the login node — `jsp-train` runs 8 srun tasks per node, so in-job
per-node cloning would race), stamps the run id into the workspace's wrapper yaml,
and sbatches. Requeues re-enter the same immutable workspace.

The submit side needs no JAX imports, so it lives with the other `pd-*` scripts and
runs from the torch venv.
"""

import shlex
import subprocess
from pathlib import Path

import fire
import yaml

from param_decomp.log import logger
from param_decomp_config.lm import LMExperimentConfig
from param_decomp_lab.infra.git import create_git_snapshot
from param_decomp_lab.infra.run_files import generate_run_id
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR, REPO_ROOT
from param_decomp_lab.infra.slurm import (
    RESET_CPU_AFFINITY,
    SlurmConfig,
    generate_script,
    submit_slurm_job,
)
from param_decomp_lab.infra.wandb import get_wandb_entity

GPUS_PER_NODE = 8
WORKSPACES_DIR = PARAM_DECOMP_OUT_DIR / "workspaces"

# Mirrors the validated llama8b.sbatch srun line: one task per GPU, block placement.
# --cpu-bind=none + the in-task RESET_CPU_AFFINITY work around the 2026-06-11 slurm
# redeploy's broken step affinity (ranks float over the job cpuset instead of per-rank
# 8-CPU blocks until the cluster-side fix lands).
_SRUN_FLAGS = (
    "--kill-on-bad-exit=1 --ntasks-per-node=8 --cpus-per-task=8 --distribution=block:block"
    " --cpu-bind=none"
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
) -> None:
    """Submit a jsp-train run.

    Args:
        config_path: Wrapper yaml (`{torch_config, run_name, out_dir,
            remat_recon_forwards}`), inside the repo. `run_id` must be absent —
            this launcher mints it.
        nodes: Node count (8 GPUs each).
        time: SLURM time limit.
        qos: SLURM QoS (e.g. `opportunistic`); None is the normal QoS.
        run_id: Resubmit an existing launch — reuses its workspace (and identity)
            instead of building a new one.
    """
    wrapper_rel = _wrapper_path_relative_to_repo(config_path)
    torch_cfg, run_name = _validate_wrapper(REPO_ROOT / wrapper_rel)

    if run_id is None:
        run_id = generate_run_id("param_decomp")
        snapshot_ref, commit_hash = create_git_snapshot(snapshot_id=run_id)
        logger.info(f"Created git snapshot: {snapshot_ref} ({commit_hash[:8]})")
        workspace = WORKSPACES_DIR / run_id
        _build_workspace(workspace, snapshot_ref, run_id, wrapper_rel)
    else:
        snapshot_ref = f"refs/runs/snapshot/{run_id}"
        workspace = WORKSPACES_DIR / run_id
        assert workspace.exists(), f"no workspace to resubmit: {workspace}"

    wandb_url = _wandb_url(torch_cfg, run_id)
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
        comment=wandb_url or run_id,
    )
    jax_dir = workspace / "param_decomp_jax"
    rank_command = f"{RESET_CPU_AFFINITY}\nsource .venv-cuda/bin/activate\n{_RANK_ENV}\nexec jsp-train {wrapper_rel.relative_to('param_decomp_jax')}"
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


def _wrapper_path_relative_to_repo(config_path: str) -> Path:
    path = Path(config_path).resolve()
    assert path.exists(), f"config not found: {path}"
    assert path.is_relative_to(REPO_ROOT), (
        f"wrapper must live inside the repo so the snapshot carries it: {path}"
    )
    rel = path.relative_to(REPO_ROOT)
    assert rel.parts[0] == "param_decomp_jax", (
        f"wrapper must live under param_decomp_jax/ (jsp-train runs from there): {rel}"
    )
    return rel


def _validate_wrapper(wrapper_path: Path) -> tuple[LMExperimentConfig, str]:
    """Lab-side mirror of `jax_single_pool.torch_config.load_torch_wrapper`'s checks
    (which can't be imported here — that module pulls jax)."""
    raw = yaml.safe_load(wrapper_path.read_text())
    expected = {"torch_config", "run_name", "out_dir", "remat_recon_forwards"}
    assert set(raw) == expected, (
        f"{wrapper_path}: keys must be {sorted(expected)} (run_id is minted at submit), "
        f"got {sorted(raw)}"
    )
    torch_yaml_path = (wrapper_path.parent / raw["torch_config"]).resolve()
    assert torch_yaml_path.exists(), f"torch config not found: {torch_yaml_path}"
    assert torch_yaml_path.is_relative_to(REPO_ROOT), (
        f"torch config must live inside the repo so the snapshot carries it: {torch_yaml_path}"
    )
    torch_cfg = LMExperimentConfig(**yaml.safe_load(torch_yaml_path.read_text()))
    return torch_cfg, raw["run_name"]


def _build_workspace(workspace: Path, snapshot_ref: str, run_id: str, wrapper_rel: Path) -> None:
    """Materialize the snapshot as an immutable shared-FS checkout with both venvs."""
    assert not workspace.exists(), f"workspace already exists: {workspace}"
    workspace.parent.mkdir(parents=True, exist_ok=True)

    def run(args: list[str], cwd: Path) -> None:
        subprocess.run(args, cwd=cwd, check=True)

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
    run(["uv", "sync", "--all-packages", "--no-dev", "--link-mode", "copy", "-q"], cwd=workspace)
    logger.info("jax venv: make install-jax-cuda ...")
    run(["make", "install-jax-cuda"], cwd=workspace)

    wrapper = workspace / wrapper_rel
    assert "run_id" not in yaml.safe_load(wrapper.read_text())
    with wrapper.open("a") as f:
        f.write(f"\nrun_id: {run_id}\n")


def _wandb_url(torch_cfg: LMExperimentConfig, run_id: str) -> str | None:
    if torch_cfg.wandb is None:
        return None
    entity = torch_cfg.wandb.entity or get_wandb_entity()
    return f"https://wandb.ai/{entity}/{torch_cfg.wandb.project}/runs/{run_id}"


def cli() -> None:
    fire.Fire(main)
