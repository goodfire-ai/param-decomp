"""The SLURM snapshot-setup fragment (`harvest` / `autointerp` / `investigate` jobs).

Mirrors the `pd-lm` node-workspace pattern: shallow-fetch the snapshot from the
submitting checkout's shared-FS common git dir (no GitHub dependency, worktree-safe),
copy the durable `.env`, build + activate the venv. Guards against the old
`$HOME/param-decomp` shell-time clone.
"""

from pathlib import Path

from param_decomp_lab.infra import slurm


def test_git_snapshot_setup_fetches_from_common_dir(monkeypatch):
    monkeypatch.setattr(slurm, "_snapshot_source_repo", lambda: Path("/home/u/param-decomp/.git"))
    setup = slurm.generate_git_snapshot_setup(
        "/tmp/$USER/param-decomp/workspace-harvest-$SLURM_JOB_ID",
        "refs/runs/snapshot/h-20260708",
    )
    # snapshot comes from the shared-FS COMMON git dir via file:// so `--depth` is honoured
    # (git silently ignores it on the bare-path local transport)
    assert (
        'git fetch --quiet --depth 1 "file:///home/u/param-decomp/.git" '
        '"refs/runs/snapshot/h-20260708"'
    ) in setup
    assert "git checkout --quiet FETCH_HEAD" in setup
    # the dead `$HOME/param-decomp` clone hack is gone
    assert "git clone" not in setup
    assert "$HOME/param-decomp" not in setup
    # `.env` (secrets, not in git) comes from the durable main checkout that owns the
    # common dir, not the possibly-ephemeral submitting worktree
    assert '[ -f "/home/u/param-decomp/.env" ] && cp "/home/u/param-decomp/.env" .env' in setup
    # venv build + activation
    assert "uv sync --all-packages --no-dev --link-mode copy -q" in setup
    assert setup.rstrip().endswith("source .venv/bin/activate")
    # the node-local work dir is created and cleaned up
    assert 'WORK_DIR="/tmp/$USER/param-decomp/workspace-harvest-$SLURM_JOB_ID"' in setup
    assert "trap 'rm -rf \"$WORK_DIR\"' EXIT" in setup


def test_snapshot_source_repo_resolves_common_git_dir():
    # REPO_ROOT is a real checkout in the test env; the common dir is an absolute path
    # ending in `.git` (or a linked-worktree's shared git dir).
    source_repo = slurm._snapshot_source_repo()
    assert source_repo.is_absolute()
    assert source_repo.exists()
