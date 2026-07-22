"""Launch a JAX target-pretraining run (`python -m pretrain.train`) — `pd-pretrain`.

The in-house target LMs (`gpt2_simple` / `llama_simple` / `llama_simple_mlp`) that the
decomposition trainer then decomposes are pretrained by `pretrain.train`. CONFIG-DRIVEN, a
slimmed mirror of `pd-lm` (`experiments/lm/launch.py`): the mode is a pure function of the
config's `dp`. `dp is None` → run the pretrainer INLINE in the current shell (single
process, CPU / 1 GPU; smoke). `dp is not None` → mint a `t-<hex>` run id, snapshot the
tree, materialize an immutable shared-FS workspace (clone + the one CUDA venv), stamp the
id (+ out_dir / wandb group / tags) into the workspace's config, and sbatch across
`dp // 8` nodes.
"""

import shlex
import subprocess
import sys
from pathlib import Path

import fire
import yaml

from param_decomp.log import logger
from param_decomp_lab.infra.git import create_git_snapshot
from param_decomp_lab.infra.run_files import generate_run_id
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR, REPO_ROOT
from param_decomp_lab.infra.slurm import SlurmConfig, generate_script, submit_slurm_job

GPUS_PER_NODE = 8
WORKSPACES_DIR = PARAM_DECOMP_OUT_DIR / "workspaces"

_SRUN_FLAGS = (
    "--kill-on-bad-exit=1 --ntasks-per-node=8 --cpus-per-task=8 --distribution=block:block"
)

_RANK_ENV = """\
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export XLA_FLAGS="--xla_gpu_enable_command_buffer=\""""


def main(
    config_path: str,
    *,
    time: str = "72:00:00",
    qos: str | None = None,
    run_id: str | None = None,
    group: str | None = None,
    tags: str | None = None,
    comment: str | None = None,
) -> None:
    """Launch a target-pretraining job (`pretrain.train`). The mode (inline vs SLURM) is a
    pure function of the config's `dp`.

    Args:
        config_path: Single self-contained run yaml inside the repo, with a `run_name` and
            NO `run_id` (minted here). `dp` declares the world size: `None` → run inline
            (single process); `N` (a multiple of 8) → submit across `N // 8` nodes.
        time: SLURM time limit.
        qos: SLURM QoS (e.g. `opportunistic`); None is the normal QoS.
        run_id: Resubmit an existing launch — reuses its workspace and identity.
        group: wandb UI group (no-op when the config omits `wandb:`).
        tags: Comma-separated wandb tags (no-op when `wandb:` is omitted).
        comment: SLURM `--comment`; defaults to the run id.
    """
    config_rel = _config_path_relative_to_repo(config_path)
    run_name, dp = _read_run_name_and_dp(REPO_ROOT / config_rel)
    tag_list = [s.strip() for s in tags.split(",")] if tags is not None else []

    if dp is None:
        _run_local(REPO_ROOT / config_rel)
        return
    assert dp % GPUS_PER_NODE == 0, f"dp={dp} must be a multiple of {GPUS_PER_NODE}"
    nodes = dp // GPUS_PER_NODE

    if run_id is None:
        run_id = generate_run_id("train")
        snapshot_ref, commit_hash = create_git_snapshot(snapshot_id=run_id)
        logger.info(f"Created git snapshot: {snapshot_ref} ({commit_hash[:8]})")
        workspace = WORKSPACES_DIR / run_id
        _build_workspace(workspace, snapshot_ref, run_id, config_rel, group, tag_list)
    else:
        snapshot_ref = f"refs/runs/snapshot/{run_id}"
        workspace = WORKSPACES_DIR / run_id
        assert workspace.exists(), f"no workspace to resubmit: {workspace}"

    job_name = f"pd-pretrain-{run_name}"
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
        comment=comment if comment is not None else run_id,
    )
    rank_command = (
        f"source .venv/bin/activate\n{_RANK_ENV}\n"
        f"exec python -m pretrain.train {shlex.quote(str(config_rel))}"
    )
    command = f"srun {_SRUN_FLAGS} bash -c {shlex.quote(rank_command)}"
    script = generate_script(slurm_config, command, setup=f'cd "{workspace}"')
    result = submit_slurm_job(script, "pd-pretrain")

    logger.section("Target-pretraining job submitted!")
    logger.values(
        {
            "Run ID": run_id,
            "Run name": run_name,
            "Job ID": result.job_id,
            "Log file": result.log_pattern,
            "Snapshot": snapshot_ref,
            "Workspace": str(workspace),
        }
    )


def _run_local(config_path: Path) -> None:
    assert config_path.exists(), f"config not found: {config_path}"
    cmd = [sys.executable, "-m", "pretrain.train", str(config_path.resolve())]
    logger.info(f"Running locally: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _config_path_relative_to_repo(config_path: str) -> Path:
    path = Path(config_path).resolve()
    assert path.exists(), f"config not found: {path}"
    assert path.is_relative_to(REPO_ROOT), (
        f"config must live inside the repo so the snapshot carries it: {path}"
    )
    return path.relative_to(REPO_ROOT)


def _read_run_name_and_dp(config_path: Path) -> tuple[str, int | None]:
    raw = yaml.safe_load(config_path.read_text())
    assert "run_id" not in raw, f"{config_path}: run_id is minted at submit, omit it"
    run_name = raw.get("run_name")
    assert isinstance(run_name, str) and run_name, f"{config_path}: run_name required"
    dp = raw.get("dp")
    assert dp is None or (isinstance(dp, int) and dp > 0), f"{config_path}: bad dp {dp!r}"
    return run_name, dp


def _build_workspace(
    workspace: Path,
    snapshot_ref: str,
    run_id: str,
    config_rel: Path,
    group: str | None,
    tags: list[str],
) -> None:
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

    logger.info("venv: uv sync --all-packages --no-dev --extra cuda ...")
    run(
        [
            "uv",
            "sync",
            "--all-packages",
            "--no-dev",
            "--extra",
            "cuda",
            "--link-mode",
            "copy",
            "-q",
        ],
        cwd=workspace,
    )

    _stamp_config(workspace / config_rel, run_id, group, tags)


def _stamp_config(config: Path, run_id: str, group: str | None, tags: list[str]) -> None:
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


def cli() -> None:
    fire.Fire(main)
