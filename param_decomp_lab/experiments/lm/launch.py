"""Launch a JAX decomposition run (`python -m param_decomp_lab.experiments.lm.run`).

CONFIG-DRIVEN: the launch mode is a pure function of `runtime.dp` in the run config — no
`--nodes` / `--local` flags. `dp is None` → run the trainer INLINE in the current process
(single device, no SLURM, no workspace; smoke / debug). `dp is not None` → submit to SLURM
across `nodes = dp // 8` nodes (8 GPUs each, one task per GPU).

The SLURM path mints the `p-<8hex>` run id, snapshots the working tree to
`refs/runs/snapshot/<id>`, materializes the snapshot as a shared-FS workspace (clone + the
one CUDA venv, built at submit time on the login node — the trainer runs 8 srun tasks per
node, so in-job per-node cloning would race), stamps the run id (+ out_dir / wandb group /
tags) into the workspace's single config yaml, and sbatches. Requeues re-enter the same
immutable workspace.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import fire
import yaml

from param_decomp.log import logger
from param_decomp_lab.experiments.lm.config import LMExperimentConfig
from param_decomp_lab.infra.git import create_git_snapshot
from param_decomp_lab.infra.run_files import generate_run_id
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR, REPO_ROOT
from param_decomp_lab.infra.slurm import SlurmConfig, generate_script, submit_slurm_job
from param_decomp_lab.infra.wandb import get_wandb_entity

GPUS_PER_NODE = 8
WORKSPACES_DIR = PARAM_DECOMP_OUT_DIR / "workspaces"

# uv hardlinks wheels from its cache into a venv when both share a filesystem, else
# copies. The default cache (~/.cache/uv, home mount) and the workspaces (data mount) are
# different PVCs, so every build copied multi-GB CUDA wheels; co-locating the cache here
# makes it hardlink instead. Set per-build (below), not globally, so other cluster uv
# usage keeps its default cache. uv falls back to copy if ever cross-FS — safe anywhere.
UV_CACHE_DIR = PARAM_DECOMP_OUT_DIR / "uv_cache"

# Mirrors the validated llama8b.sbatch srun line: one task per GPU, block placement.
_SRUN_FLAGS = (
    "--kill-on-bad-exit=1 --ntasks-per-node=8 --cpus-per-task=8 --distribution=block:block"
)

# btdr NCCL InfiniBand env. login.sh wires these into interactive shells only, so a
# non-interactive sbatch submission (e.g. from an agent / a 1-CPU job) inherits the baked
# NCCL_IB_HCA=ibp, which matches no device here (HCAs are mlx5_*) and silently drops NCCL
# to TCP/eth0 (~10x slower inter-node). We set them per-rank so multi-node IB works
# regardless of submit context. TEMPORARY: andromeda-scripts #414 makes this cluster-wide
# via /etc/environment; drop this block once that lands. Values captured live from btdr.
_NCCL_IB_ENV = """\
export NCCL_IB_DISABLE=0
export NCCL_IB_GDR_SUPPORT=1
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_10,mlx5_11,mlx5_2,mlx5_7,mlx5_8,mlx5_9
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_TIMEOUT=22
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME=eth0"""

# Default 0.75 caps the XLA pool too low for production steps (OOM, job 50644);
# CUDA-graph capture (XLA command buffers) intermittently dies with
# CUDA_ERROR_STREAM_CAPTURE_INVALIDATED on disjoint allocations — disabling
# measured ~0% cost (8,007 vs 8,015 tok/s/GPU).
#
# `--xla_gpu_force_compilation_parallelism=1`: all 8 per-GPU ranks on a node compile the
# SAME executable simultaneously on a cold cache; each one's multi-threaded LLVM codegen of
# the big chunkwise-step graph spikes host RAM, and 8x that OOMs the node even at --mem=0
# (~2TB). Serializing each rank's codegen caps peak host RAM (slower one-time compile).
_RANK_ENV = f"""\
{_NCCL_IB_ENV}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_force_compilation_parallelism=1\""""


def _rank_command(config_rel: Path, rank_env: str) -> str:
    return (
        f"source .venv/bin/activate\n{rank_env}\n"
        f"exec python -m param_decomp_lab.experiments.lm.run {shlex.quote(str(config_rel))}"
    )


def main(
    config_path: str,
    *,
    time: str = "72:00:00",
    qos: str | None = None,
    run_id: str | None = None,
    group: str | None = None,
    tags: str | None = None,
    comment: str | None = None,
    allocator: str | None = None,
) -> None:
    """Launch a decomposition trainer (`param_decomp_lab.experiments.lm.run`) run. The mode
    (inline vs SLURM) is a pure function of the config's `runtime.dp`.

    Args:
        config_path: Single self-contained run yaml (the canonical schema + top-level
            `run_name`, optional `out_dir`), inside the repo. `runtime.dp` declares the
            world size: `None` → run inline (single device); `N` (a multiple of 8) →
            submit across `N // 8` nodes. `run_id` and `out_dir` are minted here and
            stamped into the workspace copy; `out_dir` defaults to
            `PARAM_DECOMP_OUT_DIR/runs` (the current cluster) when absent.
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

    dp = cfg.runtime.dp
    if dp is None:
        _run_local(config_rel, run_name, group, tag_list)
        return
    assert dp % GPUS_PER_NODE == 0, f"runtime.dp={dp} must be a multiple of {GPUS_PER_NODE}"
    nodes = dp // GPUS_PER_NODE

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
    job_name = f"pd-{run_name}"
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
        exclusive=True,
        comment=comment if comment is not None else (wandb_url or run_id),
        # --mem=0 = all node RAM. The default (DefMemPerCPU*cpus = 640G/node) suffices for
        # a warm XLA compile cache, but a cache MISS recompiles the full graph in 8 parallel
        # per-GPU processes and spikes host RAM past 640G -> OOM (e.g. the first run of a
        # new loss op). Nodes have ~2TB; grabbing it all is safe since these are whole-node
        # (8-GPU) jobs anyway.
        mem="0",
    )
    rank_env = _RANK_ENV
    if allocator is not None:
        rank_env = f"{rank_env}\nexport XLA_PYTHON_CLIENT_ALLOCATOR={allocator}"
    command = f"srun {_SRUN_FLAGS} bash -c {shlex.quote(_rank_command(config_rel, rank_env))}"
    script = generate_script(slurm_config, command, setup=f'cd "{workspace}"')
    result = submit_slurm_job(script, "pd-lm")

    logger.section("pd-lm job submitted!")
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


def _run_local(config_rel: Path, run_name: str, group: str | None, tags: list[str]) -> None:
    """Mint a run id, stamp the live config in place, and run the trainer inline."""
    run_id = generate_run_id("param_decomp")
    config = REPO_ROOT / config_rel
    _stamp_config(config, run_id, group, tags)
    logger.section(f"pd-lm local: {run_name} ({run_id})")
    subprocess.run(
        [sys.executable, "-m", "param_decomp_lab.experiments.lm.run", str(config_rel)],
        cwd=REPO_ROOT,
        check=True,
        env=os.environ.copy(),
    )


def _config_path_relative_to_repo(config_path: str) -> Path:
    path = Path(config_path).resolve()
    assert path.exists(), f"config not found: {path}"
    assert path.is_relative_to(REPO_ROOT), (
        f"config must live inside the repo so the snapshot carries it: {path}"
    )
    return path.relative_to(REPO_ROOT)


def _validate_config(config_path: Path) -> tuple[LMExperimentConfig, str]:
    """Validate the not-yet-stamped single run config against the shared torch-free LM
    schema. `pd-lm` is LM-ONLY; the toy domains (TMS, ResidMLP) run on CPU in-process
    via `pd-tms` / `pd-resid-mlp`, never here. A hand-authored config must NOT carry
    `run_id` (minted at submit)."""
    raw = yaml.safe_load(config_path.read_text())
    assert "run_id" not in raw, f"{config_path}: run_id is minted at submit, omit it"
    target = raw.get("target", {})
    assert isinstance(target, dict), target
    assert "n_hidden" not in target and "d_embed" not in target, (
        f"{config_path}: pd-lm is LM-only; run TMS/ResidMLP toys on CPU via pd-tms / pd-resid-mlp"
    )
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
    """Materialize the snapshot as an immutable shared-FS checkout with the one CUDA
    venv, then stamp the run identity (run_id, out_dir-if-absent, wandb group/tags) into
    the workspace's single config yaml."""
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

    logger.info("venv: uv sync --all-packages --no-dev --extra cuda (hardlink from uv_cache) ...")
    run(
        ["uv", "sync", "--all-packages", "--no-dev", "--extra", "cuda", "-q"],
        cwd=workspace,
        env=build_env,
    )

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


def _wandb_url(cfg: LMExperimentConfig, run_id: str) -> str | None:
    if cfg.wandb is None:
        return None
    entity = cfg.wandb.entity or get_wandb_entity()
    return f"https://wandb.ai/{entity}/{cfg.wandb.project}/runs/{run_id}"


def cli() -> None:
    fire.Fire(main)
