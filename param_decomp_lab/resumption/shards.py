"""On-disk layout for per-rank resume shards.

Each periodic checkpoint produces one ``shard_rank<R>.pth`` per rank under
``<run_dir>/resume/step_<step>/``. The shard payload is whatever
``trainer.state_blob()`` returned — atomic cfg + state for that rank.

Resume reads the shard for the current rank only; layout-fingerprint checks
inside the trainer's ``from_blob`` catch world-shape drift.
"""

from pathlib import Path
from typing import Any, Literal

import torch


def shard_path(run_dir: Path, step: int, rank: int) -> Path:
    """Canonical path for rank ``rank``'s resume shard at ``step``."""
    return run_dir / "resume" / f"step_{step}" / f"shard_rank{rank}.pth"


def save_shard(blob: dict[str, Any], run_dir: Path, step: int, rank: int) -> None:
    """Persist a rank's :meth:`state_blob` dict to its canonical shard path.

    Embeds ``run_id = run_dir.name`` in the blob so ``load_shard`` can assert
    on retrieval that the shard came from the run the caller pointed at.
    Catches "I loaded the wrong parent" cleanly before any tensors get copied
    into the new model — much better than relying on a downstream
    topology-fingerprint mismatch (which only fires if shapes happen to differ).
    """
    blob = {**blob, "run_id": run_dir.name}
    path = shard_path(run_dir, step, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)


def load_shard(run_dir: Path, step: int, rank: int) -> dict[str, Any]:
    """Read a rank's resume shard back from disk.

    ``weights_only=False`` because the blob contains arbitrary cfg dicts
    (model_dump output) alongside the tensors.

    Asserts the shard's saved ``run_id`` matches ``run_dir.name``. If they
    don't match, the caller pointed at a directory whose contents came from
    a different run — fail loud before model state gets corrupted.
    """
    path = shard_path(run_dir, step, rank)
    assert path.is_file(), f"resume shard not found: {path}"
    blob: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    expected_run_id = run_dir.name
    actual_run_id = blob.get("run_id")
    assert actual_run_id == expected_run_id, (
        f"resume shard identity mismatch at {path}:\n"
        f"  expected run_id (= dir name): {expected_run_id!r}\n"
        f"  shard's saved run_id:         {actual_run_id!r}\n"
        f"the shard came from a different run than the dir it lives in — "
        f"someone copied / moved shards across runs, or you pointed at the "
        f"wrong parent."
    )
    return blob


def list_resume_steps(run_dir: Path) -> list[int]:
    """Enumerate the step numbers with a resume snapshot under ``run_dir/resume/``."""
    resume_dir = run_dir / "resume"
    if not resume_dir.is_dir():
        return []
    steps: list[int] = []
    for d in resume_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                steps.append(int(d.name.removeprefix("step_")))
            except ValueError:
                continue
    steps.sort()
    return steps


def resolve_step(run_dir: Path, step: int | Literal["latest"]) -> int:
    """Resolve ``"latest"`` to the highest-numbered snapshot under ``run_dir/resume/``.

    Errors loudly if no snapshots exist, or if a specific step was requested
    that isn't present on disk.
    """
    available = list_resume_steps(run_dir)
    assert available, f"no resume snapshots under {run_dir / 'resume'}"
    if step == "latest":
        return available[-1]
    assert step in available, (
        f"resume step {step} not on disk under {run_dir / 'resume'}; available: {available}"
    )
    return step
