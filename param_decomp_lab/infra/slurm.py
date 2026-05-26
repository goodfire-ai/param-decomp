"""Unified SLURM job submission utilities.

This module provides a single source of truth for generating and submitting SLURM jobs.
It handles:
- SBATCH header generation
- Workspace creation with cleanup
- Git snapshot checkout (optional)
- Virtual environment activation
- Job submission with script renaming and log file creation
"""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from param_decomp_lab.infra.settings import REPO_ROOT, SBATCH_SCRIPTS_DIR, SLURM_LOGS_DIR

# Bash expressions that uniquely identify a job invocation, used to name per-job /tmp
# workspaces. Exposed so other modules building SLURM commands (e.g. multi-node DDP
# srun wrappers) don't have to re-spell the same magic strings.
SINGLETON_JOB_ID_BASH = "$SLURM_JOB_ID"
ARRAY_JOB_ID_BASH = "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"


@dataclass
class SlurmConfig:
    """SLURM job config.

    `n_gpus=0` is CPU-only. `snapshot_ref` is a fully-qualified git ref (e.g.
    `refs/runs/snapshot/<id>`) to fetch and checkout in the SLURM job; `None` means
    just `cd` to `REPO_ROOT`. `dependency_job_id` adds an `afterok` dependency.
    """

    job_name: str
    partition: str | None
    n_gpus: int = 1
    n_nodes: int = 1
    time: str = "72:00:00"
    mem: str | None = None  # Memory limit (e.g., "64G", "128G")
    cpus_per_task: int | None = None
    snapshot_ref: str | None = None
    dependency_job_id: str | None = None
    comment: str | None = None


@dataclass
class SlurmArrayConfig(SlurmConfig):
    """SLURM job-array config. `max_concurrent_tasks=None` means no limit."""

    max_concurrent_tasks: int | None = None


@dataclass
class SubmitResult:
    """Result of submitting a SLURM job."""

    job_id: str
    script_path: Path
    log_pattern: str


def generate_script(config: SlurmConfig, command: str, env: dict[str, str] | None = None) -> str:
    """Generate a single SLURM job script. `env` is exported at the start of the script."""
    header = _sbatch_header_singleton(config)
    if config.n_nodes == 1:
        setup = _setup_section_singleton(config)
    else:
        setup = "# Multi-node job: each node sets up its own workspace in the srun command"
    env_exports = _env_exports(env)

    return f"""\
#!/bin/bash
{header}

set -euo pipefail
umask 002  # Ensure files are group-writable
{env_exports}
{setup}

{command}
"""


def generate_array_script(
    config: SlurmArrayConfig,
    commands: list[str],
    env: dict[str, str] | None = None,
    per_task_comments: list[str] | None = None,
) -> str:
    """Generate a SLURM job-array script.

    Each command becomes one array task, dispatched via a case statement on
    `SLURM_ARRAY_TASK_ID`. `per_task_comments` (must match `len(commands)`) sets each
    task's SLURM comment via `scontrol` at execution start.
    """
    if not commands:
        raise ValueError("Cannot generate array script with empty commands list")

    if per_task_comments is not None:
        assert len(per_task_comments) == len(commands)

    n_jobs = len(commands)

    # Build array range (SLURM arrays are 1-indexed)
    if config.max_concurrent_tasks is not None:
        array_range = f"1-{n_jobs}%{config.max_concurrent_tasks}"
    else:
        array_range = f"1-{n_jobs}"

    header = _sbatch_header_array(config, array_range=array_range)
    # Multi-node: each node sets up its own workspace in the srun command (can't share /tmp)
    setup = "" if config.n_nodes > 1 else _setup_section_array(config)
    env_exports = _env_exports(env)
    case_block = _case_block(commands)

    # Set per-task comment from inside the running job
    if per_task_comments is not None:
        comment_case_block = _case_block(
            [
                f'scontrol update job="${{SLURM_ARRAY_JOB_ID}}_{i}" comment="{comment}"'
                for i, comment in enumerate(per_task_comments, start=1)
            ]
        )
        comment_section = f"""
# Set per-task SLURM comment
case $SLURM_ARRAY_TASK_ID in
{comment_case_block}
esac
"""
    else:
        comment_section = ""

    return f"""\
#!/bin/bash
{header}

set -euo pipefail
umask 002  # Ensure files are group-writable
{env_exports}
{comment_section}
{setup}

# Execute the appropriate command based on array task ID
case $SLURM_ARRAY_TASK_ID in
{case_block}
esac
"""


def submit_slurm_job(
    script_content: str,
    script_name_prefix: str,
    n_array_tasks: int | None = None,
) -> SubmitResult:
    """Submit `script_content` via `sbatch` and prepare its logs.

    Writes the script to `SBATCH_SCRIPTS_DIR`, submits via `sbatch`, renames the script
    to include the job ID, and `touch`es empty log file(s) for tailing.
    `n_array_tasks=None` is a singleton job.
    """
    SBATCH_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    SLURM_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Write script to a unique temporary file (safe for concurrent submissions)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=SBATCH_SCRIPTS_DIR,
        prefix=f"{script_name_prefix}_",
        suffix=".sh",
        delete=False,
    ) as f:
        f.write(script_content)
        temp_script_path = Path(f.name)
    temp_script_path.chmod(0o755)

    # Submit via sbatch
    job_id = _submit_script(temp_script_path)

    # Rename script to include job ID
    final_script_path = SBATCH_SCRIPTS_DIR / f"{script_name_prefix}_{job_id}.sh"
    temp_script_path.rename(final_script_path)

    # Create empty log file(s) for tailing
    if n_array_tasks is not None:
        for i in range(1, n_array_tasks + 1):
            (SLURM_LOGS_DIR / f"slurm-{job_id}_{i}.out").touch()
        log_pattern = str(SLURM_LOGS_DIR / f"slurm-{job_id}_*.out")
    else:
        (SLURM_LOGS_DIR / f"slurm-{job_id}.out").touch()
        log_pattern = str(SLURM_LOGS_DIR / f"slurm-{job_id}.out")

    return SubmitResult(
        job_id=job_id,
        script_path=final_script_path,
        log_pattern=log_pattern,
    )


# =============================================================================
# Internal helpers
# =============================================================================


def _common_sbatch_lines(config: SlurmConfig, log_pattern: str) -> list[str]:
    """Shared #SBATCH directives between singleton and array jobs.

    `log_pattern` is the SLURM filename pattern: `%j` for singletons, `%A_%a` for arrays.
    """
    lines = [
        f"#SBATCH --job-name={config.job_name}",
        f"#SBATCH --nodes={config.n_nodes}",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --gpus-per-node={config.n_gpus}",
        f"#SBATCH --time={config.time}",
        f"#SBATCH --output={SLURM_LOGS_DIR}/slurm-{log_pattern}.out",
    ]
    if config.partition is not None:
        lines.append(f"#SBATCH --partition={config.partition}")
    if config.cpus_per_task is not None:
        lines.append(f"#SBATCH --cpus-per-task={config.cpus_per_task}")
    if config.mem is not None:
        lines.append(f"#SBATCH --mem={config.mem}")
    if config.dependency_job_id:
        lines.append(f"#SBATCH --dependency=afterok:{config.dependency_job_id}")
    if config.comment:
        lines.append(f'#SBATCH --comment="{config.comment}"')
    return lines


def _sbatch_header_singleton(config: SlurmConfig) -> str:
    """Generate the #SBATCH directive block for a non-array job."""
    return "\n".join(_common_sbatch_lines(config, log_pattern="%j"))


def _sbatch_header_array(config: SlurmArrayConfig, array_range: str) -> str:
    """Generate the #SBATCH directive block for an array job."""
    lines = _common_sbatch_lines(config, log_pattern="%A_%a")
    lines.append(f"#SBATCH --array={array_range}")
    return "\n".join(lines)


def generate_git_snapshot_setup(work_dir: str, snapshot_ref: str) -> str:
    """Bash fragment to clone the repo, fetch `snapshot_ref`, check it out, and set up env.

    `git clone` only fetches `refs/heads/*` + tags, so custom namespaces like
    `refs/runs/snapshot/*` need an explicit fetch. Also copies `.env` and activates the
    venv. `work_dir` is a bash expression and can include `$SLURM_*` vars.
    """
    return f"""\
WORK_DIR="{work_dir}"
mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT
git clone "{REPO_ROOT}" "$WORK_DIR"
cd "$WORK_DIR"
[ -f "{REPO_ROOT}/.env" ] && cp "{REPO_ROOT}/.env" .env
git fetch "{REPO_ROOT}" "{snapshot_ref}:{snapshot_ref}"
git checkout "{snapshot_ref}"
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
uv sync --all-packages --no-dev --link-mode copy -q
source .venv/bin/activate"""


def _workspace_setup(config: SlurmConfig, workspace_suffix: str) -> str:
    """Generate workspace creation and git/venv setup, parameterized by the bash
    expression that uniquely identifies this job invocation."""
    if config.snapshot_ref is not None:
        work_dir = f"/tmp/$USER/param-decomp/workspace-{config.job_name}-{workspace_suffix}"
        return generate_git_snapshot_setup(work_dir, config.snapshot_ref)
    else:
        return f"""\
cd "{REPO_ROOT}"
source .venv/bin/activate"""


def _setup_section_singleton(config: SlurmConfig) -> str:
    return _workspace_setup(config, SINGLETON_JOB_ID_BASH)


def _setup_section_array(config: SlurmConfig) -> str:
    return _workspace_setup(config, ARRAY_JOB_ID_BASH)


def _env_exports(env: dict[str, str] | None) -> str:
    """Generate export statements for environment variables.

    Returns empty string if env is None or empty, otherwise returns
    export statements with a leading newline for proper formatting.
    """
    if not env:
        return ""
    exports = "\n".join(f"export {k}={v}" for k, v in env.items())
    return f"\n{exports}"


def _case_block(commands: list[str]) -> str:
    """Generate bash case statement for array jobs.

    SLURM arrays are 1-indexed, so command[0] goes in case 1).
    """
    lines = []
    for i, cmd in enumerate(commands):
        lines.append(f"    {i + 1})")
        lines.append(f"        {cmd}")
        lines.append("        ;;")
    return "\n".join(lines)


def _submit_script(script_path: Path) -> str:
    """Submit script via sbatch and return job ID.

    Raises RuntimeError if sbatch fails.
    """
    result = subprocess.run(
        ["sbatch", str(script_path)], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to submit SLURM job: {result.stderr}")
    # Extract job ID from sbatch output (format: "Submitted batch job 12345")
    job_id = result.stdout.strip().split()[-1]
    return job_id
