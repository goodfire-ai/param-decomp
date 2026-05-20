"""Utilities for managing experiment run directories and IDs."""

import json
import os
import secrets
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple

import torch
import yaml

from param_decomp.log import logger
from param_decomp.settings import PARAM_DECOMP_OUT_DIR
from param_decomp.utils.git_utils import (
    create_git_snapshot,
    repo_current_branch,
    repo_current_commit_hash,
    repo_is_clean,
)


def _save_json(data: Any, path: Path | str, **kwargs: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, **kwargs)


def _save_yaml(data: Any, path: Path | str, **kwargs: Any) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False, **kwargs)


def _save_torch(data: Any, path: Path | str, **kwargs: Any) -> None:
    torch.save(data, path, **kwargs)


def _save_text(data: str, path: Path | str, encoding: str = "utf-8") -> None:
    with open(path, "w", encoding=encoding) as f:
        f.write(data)


def save_file(data: dict[str, Any] | Any, path: Path | str, **kwargs: Any) -> None:
    """Write ``data`` to ``path``, dispatching on the file extension.

    Ensures the parent directory exists. Format:

    - ``.json`` → JSON via ``json.dump``
    - ``.yaml`` / ``.yml`` → YAML via ``yaml.dump`` (sort_keys=False)
    - ``.pth`` / ``.pt`` → ``torch.save``
    - anything else → plain text (``data`` must be a string)
    """
    path = Path(path)
    suffix = path.suffix.lower()

    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".json":
        _save_json(data, path, **kwargs)
    elif suffix in [".yaml", ".yml"]:
        _save_yaml(data, path, **kwargs)
    elif suffix in [".pth", ".pt"]:
        _save_torch(data, path, **kwargs)
    else:
        # Default to text file
        assert isinstance(data, str), f"For {suffix} files, data must be a string, got {type(data)}"
        _save_text(data, path, encoding=kwargs.get("encoding", "utf-8"))


RunType = Literal[
    "param_decomp", "train", "clustering/runs", "clustering/ensembles", "clustering/harvests"
]

RUN_TYPE_ABBREVIATIONS: Final[dict[RunType, str]] = {
    "param_decomp": "p",
    "train": "t",
    "clustering/runs": "c",
    "clustering/ensembles": "e",
    "clustering/harvests": "ch",
}


def generate_run_id(run_type: RunType) -> str:
    """Generate a unique run identifier.

    Format: `{type_abbr}-{random_hex}`
    """
    type_abbr = RUN_TYPE_ABBREVIATIONS[run_type]
    return f"{type_abbr}-{secrets.token_hex(4)}"


class ExecutionStamp(NamedTuple):
    run_id: str
    snapshot_ref: str
    commit_hash: str
    run_type: RunType

    @classmethod
    def create(
        cls,
        run_type: RunType,
        create_snapshot: bool,
    ) -> "ExecutionStamp":
        """Create an execution stamp, possibly including a git snapshot ref."""
        run_id = generate_run_id(run_type)
        snapshot_ref: str
        commit_hash: str

        if create_snapshot:
            snapshot_ref, commit_hash = create_git_snapshot(snapshot_id=run_id)
            logger.info(f"Created git snapshot ref: {snapshot_ref} ({commit_hash[:8]})")
        else:
            snapshot_ref = repo_current_branch()
            if repo_is_clean():
                commit_hash = repo_current_commit_hash()
                logger.info(f"Using current branch: {snapshot_ref} ({commit_hash[:8]})")
            else:
                commit_hash = "none"
                logger.info(
                    f"Using current branch: {snapshot_ref} (uncommitted changes, no commit hash)"
                )

        return ExecutionStamp(
            run_id=run_id,
            snapshot_ref=snapshot_ref,
            commit_hash=commit_hash,
            run_type=run_type,
        )

    @property
    def out_dir(self) -> Path:
        """Get the output directory for this execution stamp."""
        run_dir = PARAM_DECOMP_OUT_DIR / self.run_type / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir


_NO_ARG_PARSSED_SENTINEL = object()


def read_noneable_str(value: str) -> str | None:
    """Read a string that may be 'None' and convert to None."""
    if value == "None":
        return None
    return value


def run_locally(
    commands: list[str],
    parallel: bool = False,
    track_resources: bool = False,
) -> dict[str, dict[str, float]] | None:
    """Run commands locally instead of via SLURM.

    Useful for testing and for --local mode in clustering pipeline.

    Args:
        commands: List of shell commands to run
        parallel: If True, run all commands in parallel. If False, run sequentially.
        track_resources: If True, track and return resource usage via /usr/bin/time

    Returns:
        If track_resources is True, dict mapping commands to resource metrics.
        Metrics include: K (avg memory KB), M (max memory KB), P (CPU %),
        S (system CPU sec), U (user CPU sec), e (wall time sec).
        Otherwise None.
    """
    n_commands = len(commands)
    resources: dict[str, dict[str, float]] = {}
    resource_files: list[Path] = []

    # Wrap commands with /usr/bin/time if resource tracking is requested
    if track_resources:
        wrapped_commands: list[str] = []
        for cmd in commands:
            # Create a unique temp file for resource tracking output
            fd, resource_file_path = tempfile.mkstemp(suffix=".resources")
            os.close(fd)  # Close fd, we just need the path for /usr/bin/time -o
            resource_file = Path(resource_file_path)
            resource_files.append(resource_file)
            # Use /usr/bin/time to track comprehensive resource usage
            # K=avg total mem, M=max resident, P=CPU%, S=system time, U=user time, e=wall time
            wrapped_cmd = (
                f'/usr/bin/time -f "K:%K M:%M P:%P S:%S U:%U e:%e" -o {resource_file} {cmd}'
            )
            wrapped_commands.append(wrapped_cmd)
        commands_to_run = wrapped_commands
    else:
        commands_to_run = commands

    try:
        if not parallel:
            logger.section(f"LOCAL EXECUTION: Running {n_commands} tasks serially")
            for i, cmd in enumerate(commands_to_run, 1):
                logger.info(f"[{i}/{n_commands}] Running: {commands[i - 1]}")
                subprocess.run(cmd, shell=True, check=True)
            logger.section("LOCAL EXECUTION COMPLETE")
        else:
            logger.section(f"LOCAL EXECUTION: Starting {n_commands} tasks in parallel")
            procs: list[subprocess.Popen[bytes]] = []

            for i, cmd in enumerate(commands_to_run, 1):
                logger.info(f"[{i}/{n_commands}] Starting: {commands[i - 1]}")
                proc = subprocess.Popen(cmd, shell=True)
                procs.append(proc)

            logger.section("WAITING FOR ALL TASKS TO COMPLETE")
            for proc, cmd in zip(procs, commands, strict=True):  # noqa: B007
                proc.wait()
                if proc.returncode != 0:
                    logger.error(f"Process {proc.pid} failed with exit code {proc.returncode}")
            logger.section("LOCAL EXECUTION COMPLETE")

        # Read resource usage results
        if track_resources:
            for cmd, resource_file in zip(commands, resource_files, strict=True):
                if resource_file.exists():
                    # Parse format: "K:123 M:456 P:78% S:1.23 U:4.56 e:7.89"
                    output = resource_file.read_text().strip()
                    metrics: dict[str, float] = {}

                    for part in output.split():
                        if ":" in part:
                            key, value = part.split(":", 1)
                            # Remove % sign from CPU percentage
                            value = value.rstrip("%")
                            try:
                                metrics[key] = float(value)
                            except ValueError:
                                logger.warning(f"Could not parse {key}:{value} for command: {cmd}")

                    resources[cmd] = metrics
                else:
                    logger.warning(f"Resource file not found for: {cmd}")

            # Log comprehensive resource usage table
            logger.section("RESOURCE USAGE RESULTS")
            for cmd, metrics in resources.items():
                logger.info(f"Command: {cmd}")
                logger.info(
                    f"  Time: {metrics.get('e', 0):.2f}s wall, "
                    f"{metrics.get('U', 0):.2f}s user, "
                    f"{metrics.get('S', 0):.2f}s system"
                )
                logger.info(
                    f"  Memory: {metrics.get('M', 0) / 1024:.1f} MB peak, "
                    f"{metrics.get('K', 0) / 1024:.1f} MB avg"
                )
                logger.info(f"  CPU: {metrics.get('P', 0):.1f}%")

    finally:
        # Clean up temp files
        if track_resources:
            for resource_file in resource_files:
                if resource_file.exists():
                    resource_file.unlink()

    return resources if track_resources else None
