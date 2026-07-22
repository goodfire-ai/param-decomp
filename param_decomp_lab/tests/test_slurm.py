"""The SLURM job-script generator (`param_decomp_lab.infra.slurm`). Exercises the
git-snapshot setup fragment: shallow worktree-safe fetch from the submitting checkout's
common git dir, `.env` from the durable main checkout, no `git clone`."""

from pathlib import Path

import pytest

from param_decomp_lab.infra import slurm


def test_snapshot_setup_shallow_fetches_from_common_dir(monkeypatch: pytest.MonkeyPatch):
    # submit resolves the common git dir (main checkout's `.git`, worktree-safe); shared
    # helper `git.snapshot_source_repo`, imported into slurm's namespace
    monkeypatch.setattr(slurm, "snapshot_source_repo", lambda: Path("/home/u/param-decomp/.git"))
    fragment = slurm.generate_git_snapshot_setup(
        "/tmp/$USER/param-decomp/workspace-harvest-$SLURM_JOB_ID",
        "refs/runs/snapshot/p-abcd1234",
    )
    # snapshot comes from the shared-FS common git dir; file:// so --depth works (git
    # ignores it on the bare-path local transport)
    assert (
        'git fetch --quiet --depth 1 "file:///home/u/param-decomp/.git" '
        '"refs/runs/snapshot/p-abcd1234"'
    ) in fragment
    assert "git init --quiet" in fragment
    assert "git checkout --quiet FETCH_HEAD" in fragment
    # no full clone of the worktree, and no hardcoded $HOME/param-decomp
    assert "git clone" not in fragment
    assert "$HOME/param-decomp" not in fragment
    # .env (untracked secrets) copied from the durable main checkout — the common git
    # dir's parent worktree — not the possibly-ephemeral submitting worktree
    assert '[ -f "/home/u/param-decomp/.env" ] && cp "/home/u/param-decomp/.env" .env' in fragment
    # workspace is created, trapped for cleanup, and the venv is built + activated
    assert "trap 'rm -rf \"$WORK_DIR\"' EXIT" in fragment
    assert "uv sync --all-packages --no-dev --link-mode copy -q" in fragment
    assert fragment.rstrip().endswith("source .venv/bin/activate")
