"""Launch a JAX decomposition run (`python -m param_decomp_lab.experiments.lm.run`).

CONFIG-DRIVEN: the launch mode is a pure function of `runtime.dp` in the run config — no
`--nodes` / `--local` flags. `dp is None` → run the trainer INLINE in the current process
(single device, no SLURM; smoke / debug). `dp is not None` → submit to SLURM across
`nodes = dp // 8` nodes (8 GPUs each, one srun task per node claiming all 8 GPUs).

The SLURM path mints the `p-<8hex>` run id, snapshots the working tree to
`refs/runs/snapshot/<id>` (pushed to origin best-effort, as a provenance backup), stages
the run dir with the pinned (group/tags-stamped) `launch_config.yaml` + `.env` (the
copies the job — and every requeue — reads), and sbatches. Each node then builds its own
workspace at job start: shallow-fetch the snapshot from the submitting checkout's
shared-FS git dir into node-local `/tmp`, `uv sync` with the driver-gated CUDA extra,
exec the trainer. Nothing is built at submit time and no shared-FS workspace exists to
leak or clean up; requeues depend only on shared FS, never on GitHub reachability.

The rank env (XLA flags, NCCL/host-memory knobs, `PD_*` profiling toggles) is rendered from
the config's `runtime.launch_env` (single source of truth, defaults in `LaunchEnv`), plus
`LD_LIBRARY_PATH` computed in-job from the freshly-built venv. So a run's
`launch_config.yaml` fully captures the environment it ran with, and A/B-ing a flag is a
config edit, not a launcher edit.
"""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import fire
import yaml

from param_decomp.built_run import LAUNCH_CONFIG_FILENAME
from param_decomp.configs import LaunchEnv
from param_decomp.log import logger
from param_decomp_lab.experiments.lm.config import LMExperimentConfig
from param_decomp_lab.infra.git import create_git_snapshot
from param_decomp_lab.infra.run_files import generate_run_id
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR, REPO_ROOT
from param_decomp_lab.infra.slurm import SlurmConfig, generate_script, submit_slurm_job
from param_decomp_lab.infra.wandb import get_wandb_entity

GPUS_PER_NODE = 8

# Node-local. Trainer jobs hold whole nodes (8/8 GPUs), so no concurrent job shares a
# node with one — the start-of-task sweep below cannot race another run's workspace.
_NODE_WORKSPACES_DIR = "/tmp/$USER/param-decomp/node-workspaces"

# ONE srun task per node (the torch torchrun model): each task runs the trainer once and
# `sharding.init_distributed` claims all local GPUs for that one process. `--ntasks=N
# --ntasks-per-node=1` makes the step unpackable on this cluster's CR_Pack_Nodes selection
# (N whole-node tasks can't collapse onto one node), so no --distribution / --cpu-bind /
# --cpus-per-task games are needed. (Earlier one-task-per-GPU attempts packed onto one node
# or hit "Unable to satisfy cpu bind request".)
_SRUN_FLAGS = "--kill-on-bad-exit=1 --ntasks-per-node=1"

# jax[cuda12]'s version check dlopens cuSPARSE et al. by soname and on this cluster doesn't
# find the pip-installed nvidia libs ("Unable to load cuSPARSE") — point the loader at the
# venv's nvidia/*/lib dirs. Evaluated in-job, after the node's venv is built and activated.
_LD_LIBRARY_PATH_EXPORT = (
    "export LD_LIBRARY_PATH=\"$(python -c 'import nvidia, os, glob; "
    'print(":".join(sorted(glob.glob(os.path.join(list(nvidia.__path__)[0], "*", "lib")))))\')'
    '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
)


def _render_rank_env(launch_env: LaunchEnv) -> str:
    """The bash `export` block for a rank, from the config's `runtime.launch_env` plus the
    in-job-computed `LD_LIBRARY_PATH`. The typed knobs are the single source of truth
    (defaults in `LaunchEnv`); shell-quote each value so spaces (e.g. multi-flag `XLA_FLAGS`)
    survive."""
    exports = [f"export {k}={shlex.quote(v)}" for k, v in launch_env.as_env().items()]
    exports.append(_LD_LIBRARY_PATH_EXPORT)
    return "\n".join(exports)


def _snapshot_source_repo() -> Path:
    """The local git state the nodes fetch the snapshot from: the submitting checkout's
    COMMON git dir (the main checkout's `.git`, even when submitting from a worktree —
    worktrees share refs but may be deleted before a requeue re-fetches). Shared FS, so
    neither submits nor requeues depend on GitHub reachability; the best-effort origin
    push in `create_git_snapshot` is a provenance backup, not the job's fetch source."""
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def _node_workspace_setup(run_id: str, snapshot_ref: str, source_repo: Path, run_dir: Path) -> str:
    """The start of each node's srun task: sweep stale workspaces, shallow-fetch the
    snapshot from the shared-FS git dir into node-local /tmp, build the CUDA venv,
    activate. (`file://` rather than the bare path — git silently ignores `--depth` on
    the local transport.) The `.env` (secrets, not in git) comes from the run dir, where
    submit staged it. The task `exec`s the trainer afterwards (bash is replaced, so no
    EXIT trap can run here) — normal-end cleanup is the batch script's trap
    (`_node_workspace_cleanup_trap`), and the up-front sweep self-heals leftovers from
    jobs that died before their trap fired.

    The CUDA extra is driver-gated per node (buildable here, unlike at submit on a
    GPU-less login node): cuda13's runtime libs include the Blackwell-TMEM-fixed
    cuBLAS >= 13.2 (cuBLAS 12.9 warns of silent corruption on B200) but need driver
    >= r580, below which (and-gmi's r570) the node falls back to cu12."""
    return f"""\
set -euo pipefail
WORK_DIR="{_NODE_WORKSPACES_DIR}/{run_id}"
rm -rf "{_NODE_WORKSPACES_DIR}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
git init --quiet
git fetch --quiet --depth 1 "file://{source_repo}" "{snapshot_ref}"
git checkout --quiet FETCH_HEAD
cp "{run_dir}/.env" .env
unset VIRTUAL_ENV
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | cut -d. -f1)
if [ "$DRIVER_MAJOR" -ge 580 ]; then CUDA_EXTRA=cuda13; else CUDA_EXTRA=cuda; fi
echo "cuda extra: $CUDA_EXTRA (driver major $DRIVER_MAJOR)"
uv sync --all-packages --no-dev --extra "$CUDA_EXTRA" --link-mode copy -q
source .venv/bin/activate"""


def _node_workspace_cleanup_trap(nodes: int) -> str:
    """Batch-script EXIT trap: sweep every node's /tmp workspace after the run. Best
    effort — a hard kill skips it; the next job's start-of-task sweep is the backstop."""
    return (
        f"trap 'srun --nodes={nodes} --ntasks={nodes} --ntasks-per-node=1 "
        f'rm -rf "{_NODE_WORKSPACES_DIR}"\' EXIT'
    )


def _rank_command(
    run_id: str, snapshot_ref: str, source_repo: Path, run_dir: Path, rank_env: str
) -> str:
    launch_config = run_dir / LAUNCH_CONFIG_FILENAME
    return (
        f"{_node_workspace_setup(run_id, snapshot_ref, source_repo, run_dir)}\n{rank_env}\n"
        f"exec python -m param_decomp_lab.experiments.lm.run "
        f"{shlex.quote(str(launch_config))} --run-id {shlex.quote(run_id)}"
    )


def main(
    config_path: str,
    *,
    time: str = "72:00:00",
    qos: str | None = None,
    run_id: str | None = None,
    group: str | None = None,
    tags: str | tuple[str, ...] | None = None,
    comment: str | None = None,
) -> None:
    """Launch a decomposition trainer (`param_decomp_lab.experiments.lm.run`) run. The mode
    (inline vs SLURM) is a pure function of the config's `runtime.dp`.

    Args:
        config_path: Single self-contained run yaml (the canonical schema + top-level
            `run_name`). `runtime.dp` declares the world size: `None` → run inline
            (single device); `N` (a multiple of 8) → submit across `N // 8` nodes.
            The `run_id` is minted here; the config is pinned as
            `PARAM_DECOMP_OUT_DIR/runs/<run_id>/launch_config.yaml` and the job reads
            THAT copy, so this file is free to change after submit.
        time: SLURM time limit.
        qos: SLURM QoS (e.g. `opportunistic`); None is the normal QoS.
        run_id: Resubmit an existing launch — reuses its pinned launch config (and
            identity). `group`/`tags` are ignored on resubmit (the pinned config
            already carries them).
        group: wandb UI group (no-op when the config omits `wandb:`).
        tags: Comma-separated wandb tags (no-op when `wandb:` is omitted).
        comment: SLURM `--comment`; defaults to the wandb run URL (or run id).

    The rank env (XLA flags, NCCL/host-memory knobs, profiling toggles) is config-driven
    via `runtime.launch_env` — set it in the YAML, not here (so `launch_config.yaml` records it).
    """
    config = Path(config_path).resolve()
    assert config.exists(), f"config not found: {config}"
    cfg, run_name = _validate_config(config)
    # Python Fire parses a comma-separated `--tags a,b,c` into a tuple, but keeps a value with a
    # hyphen (e.g. `a,b,c-d`) as a string — normalize both (and the single-token case) to a list.
    if tags is None:
        tag_list = []
    elif isinstance(tags, str):
        tag_list = [s.strip() for s in tags.split(",") if s.strip()]
    else:
        tag_list = [str(t).strip() for t in tags]

    dp = cfg.runtime.dp
    if dp is None:
        _run_local(config, run_name, group, tag_list)
        return
    assert dp % GPUS_PER_NODE == 0, f"runtime.dp={dp} must be a multiple of {GPUS_PER_NODE}"
    nodes = dp // GPUS_PER_NODE

    source_repo = _snapshot_source_repo()
    if run_id is None:
        run_id = generate_run_id("param_decomp")
        snapshot_ref, commit_hash = create_git_snapshot(snapshot_id=run_id)
        logger.info(f"Created git snapshot: {snapshot_ref} ({commit_hash[:8]})")
        run_dir = _write_run_dir(config, run_id, group, tag_list)
        _stage_env_file(run_dir)
    else:
        snapshot_ref = f"refs/runs/snapshot/{run_id}"
        _assert_snapshot_ref_exists(source_repo, snapshot_ref)
        run_dir = PARAM_DECOMP_OUT_DIR / "runs" / run_id
        for staged in (LAUNCH_CONFIG_FILENAME, ".env"):
            assert (run_dir / staged).exists(), f"no {staged} to resubmit: {run_dir / staged}"

    wandb_url = _wandb_url(cfg, run_id)
    job_name = f"pd-{run_name}"
    slurm_config = SlurmConfig(
        job_name=job_name,
        partition=None,
        qos=qos,
        n_gpus=GPUS_PER_NODE,
        n_nodes=nodes,
        ntasks_per_node=1,
        time=time,
        signal="TERM@300",
        requeue=True,
        comment=comment if comment is not None else (wandb_url or run_id),
    )
    rank_env = _render_rank_env(cfg.runtime.launch_env)
    # One task per node: `--nodes=N --ntasks=N` (N whole-node tasks) can't pack onto one
    # node, so the trainer's single process per node claims all local GPUs.
    srun = f"srun --nodes={nodes} --ntasks={nodes} {_SRUN_FLAGS}"
    rank_command = _rank_command(run_id, snapshot_ref, source_repo, run_dir, rank_env)
    command = f"{srun} bash -c {shlex.quote(rank_command)}"
    script = generate_script(slurm_config, command, setup=_node_workspace_cleanup_trap(nodes))
    result = submit_slurm_job(script, "pd-lm")

    logger.section("pd-lm job submitted!")
    summary: dict[str, str | None] = {
        "Run ID": run_id,
        "Run name": run_name,
        "Job ID": result.job_id,
        "Log file": result.log_pattern,
        "Script": str(result.script_path),
        "Snapshot": snapshot_ref,
        "Launch config": str(run_dir / LAUNCH_CONFIG_FILENAME),
    }
    if wandb_url is not None:
        summary["WandB run URL"] = wandb_url
    logger.values(summary)


def _run_local(config: Path, run_name: str, group: str | None, tags: list[str]) -> None:
    """Mint a run id, pin the launch config into the run dir, and run the trainer inline."""
    run_id = generate_run_id("param_decomp")
    launch_config = _write_run_dir(config, run_id, group, tags) / LAUNCH_CONFIG_FILENAME
    logger.section(f"pd-lm local: {run_name} ({run_id})")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "param_decomp_lab.experiments.lm.run",
            str(launch_config),
            "--run-id",
            run_id,
        ],
        cwd=REPO_ROOT,
        check=True,
        env=os.environ.copy(),
    )


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


def _write_run_dir(config: Path, run_id: str, group: str | None, tags: list[str]) -> Path:
    """Mint the run dir and pin the config (wandb group/tags stamped in) as its
    `launch_config.yaml` — the one copy the job and every requeue read, and the same file
    the trainer pins (`run.py::_pin_config_copy` no-ops on it)."""
    run_dir = PARAM_DECOMP_OUT_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    launch_config = run_dir / LAUNCH_CONFIG_FILENAME
    launch_config.write_text(config.read_text())
    _stamp_config(launch_config, group, tags)
    return run_dir


def _stage_env_file(run_dir: Path) -> None:
    """Stage `.env` (secrets, not in git — they can't ride the snapshot) into the run dir
    for the node setup to copy into its workspace. Mode-preserving copy (0o660 secrets
    stay group-only under the runs dir's group-writable umask)."""
    env_file = REPO_ROOT / ".env"
    assert env_file.exists(), f".env with wandb credentials required: {env_file}"
    shutil.copy(env_file, run_dir / ".env")


def _assert_snapshot_ref_exists(source_repo: Path, snapshot_ref: str) -> None:
    """Load-bearing on `--run-id` resubmits (a fresh submit just created the ref in a
    checkout sharing `source_repo`'s refs)."""
    result = subprocess.run(
        ["git", "-C", str(source_repo), "rev-parse", "--verify", "--quiet", snapshot_ref],
        capture_output=True,
    )
    assert result.returncode == 0, (
        f"{snapshot_ref} not in {source_repo} — nodes fetch the snapshot from there at job start"
    )


def _stamp_config(config: Path, group: str | None, tags: list[str]) -> None:
    """Stamp the wandb UI knobs onto the pinned launch config's `wandb:` block
    (no-op when `wandb:` is omitted). The run id is NOT stamped — it is passed to the
    trainer as a `--run-id` CLI arg, and the run dir derives from it."""
    if group is None and not tags:
        return
    raw = yaml.safe_load(config.read_text())
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
