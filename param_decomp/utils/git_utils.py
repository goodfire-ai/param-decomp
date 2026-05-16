"""Git utilities for creating code snapshots."""

import subprocess
import tempfile
from pathlib import Path

from param_decomp.log import logger
from param_decomp.settings import REPO_ROOT


def repo_current_branch() -> str:
    """Return the active Git branch by invoking the `git` CLI.

    Uses `git rev-parse --abbrev-ref HEAD`, which prints either the branch
    name (e.g. `main`) or `HEAD` if the repo is in a detached-HEAD state.

    Returns:
        The name of the current branch, or `HEAD` if in detached state.

    Raises:
        subprocess.CalledProcessError: If the `git` command fails.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def repo_is_clean() -> bool:
    """Return True if the current git repository has no uncommitted or untracked changes."""
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    return status == ""


def repo_current_commit_hash() -> str:
    """Return the current commit hash of the active HEAD."""
    commit_hash: str = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return commit_hash


def create_git_snapshot(snapshot_id: str) -> tuple[str, str]:
    """Create a git snapshot ref with current changes.

    Creates a ref under `refs/runs/snapshot/<snapshot_id>` containing all current changes (staged
    and unstaged). Uses a temporary detached worktree to avoid affecting the current working
    directory. Will push the snapshot ref to origin if possible, but will continue without error
    if push permissions are lacking.

    The ref lives outside `refs/heads/*` and `refs/tags/*`, so it is invisible to a default
    `git fetch` — clients only pull it down if they ask for it explicitly. This keeps the set of
    branches teammates sync small even as we accumulate many snapshots.

    Args:
        snapshot_id: Identifier used in the ref name and commit message (e.g. a launch_id
            or run_id).

    Returns:
        (ref_name, commit_hash) where ref_name is the fully-qualified ref (e.g.
        `refs/runs/snapshot/<id>`) and commit_hash is the commit it points at (a new snapshot
        commit if changes existed, otherwise the base commit).

    Raises:
        subprocess.CalledProcessError: If git commands fail (except for push)
    """
    snapshot_ref: str = f"refs/runs/snapshot/{snapshot_id}"

    with tempfile.TemporaryDirectory() as temp_dir:
        worktree_path = Path(temp_dir) / f"pd-snapshot-{snapshot_id}"

        try:
            # Detached worktree at HEAD — we will move the snapshot ref to the resulting commit
            # rather than creating a branch.
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree_path), "HEAD"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
            )

            # Copy current working tree to worktree (including untracked files)
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--delete",
                    "--exclude=.git",
                    "--filter=:- .gitignore",
                    f"{REPO_ROOT}/",
                    f"{worktree_path}/",
                ],
                check=True,
                capture_output=True,
            )

            # Stage all changes in the worktree
            subprocess.run(["git", "add", "-A"], cwd=worktree_path, check=True, capture_output=True)

            # Check if there are changes to commit
            diff_result = subprocess.run(
                ["git", "diff", "--cached", "--quiet"], cwd=worktree_path, capture_output=True
            )

            # Commit changes if any exist
            if diff_result.returncode != 0:  # Non-zero means there are changes
                subprocess.run(
                    ["git", "commit", "-m", f"snapshot {snapshot_id}", "--no-verify"],
                    cwd=worktree_path,
                    check=True,
                    capture_output=True,
                )

            # Get the commit hash of HEAD (either new commit or base commit if nothing changed)
            rev_parse = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=worktree_path,
                check=True,
                capture_output=True,
                text=True,
            )
            commit_hash = rev_parse.stdout.strip()

            # Point the snapshot ref at the commit, in the main repo's ref db.
            subprocess.run(
                ["git", "update-ref", snapshot_ref, commit_hash],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
            )

            # Try push (non-fatal if fails)
            try:
                subprocess.run(
                    ["git", "push", "origin", f"{snapshot_ref}:{snapshot_ref}"],
                    cwd=REPO_ROOT,
                    check=True,
                    capture_output=True,
                )
                logger.info(f"Successfully pushed snapshot ref '{snapshot_ref}' to origin")
            except subprocess.CalledProcessError as e:
                logger.warning(
                    f"Could not push snapshot ref '{snapshot_ref}' to origin. "
                    f"The ref was created locally but won't be accessible to other users. "
                    f"Error: {e.stderr.decode().strip() if e.stderr else 'Unknown error'}"
                )

        finally:
            # Clean up worktree (the snapshot ref in the main repo remains)
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_path)],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
            )

    return snapshot_ref, commit_hash
