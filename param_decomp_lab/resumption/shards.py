"""On-disk layout for per-rank resume shards.

Each periodic checkpoint produces one ``shard_rank<R>.pth`` per rank under
``<run_dir>/resume/step_<step>/``. The shard's payload is a
:class:`ShardEnvelope` — the trainer's rank-local resume blob plus the
``run_id`` of the run that owns it, so a "loaded the wrong parent" mistake
fails before any tensors get copied into a new model.

Each shard write also lays down a sibling ``state_hash_rank<R>.txt`` with
a stable SHA256 of the envelope contents — used by
``check_snapshot_fidelity`` to assert that ``Trainer.from_snapshot``
reconstructs a trainer whose own ``snapshot()`` produces the same hash.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch


@dataclass(frozen=True)
class ShardEnvelope:
    """What lives on disk as one rank's shard.

    Attributes:
        run_id: Identity tag of the parent run (= ``run_dir.name``). Verified
            on load so a shard moved/copied to the wrong run directory errors
            cleanly.
        resume: The trainer's :attr:`TrainerSnapshot.resume` dict — atomic
            cfg + state for this rank, opaque to the shard layer.
    """

    run_id: str
    resume: dict[str, Any]


def shard_path(run_dir: Path, step: int, rank: int) -> Path:
    """Canonical path for rank ``rank``'s resume shard at ``step``."""
    return run_dir / "resume" / f"step_{step}" / f"shard_rank{rank}.pth"


def state_hash_path(run_dir: Path, step: int, rank: int) -> Path:
    """Sibling SHA256 file written next to each shard for fidelity checks."""
    return run_dir / "resume" / f"step_{step}" / f"state_hash_rank{rank}.txt"


def compute_state_hash(envelope: ShardEnvelope) -> str:
    """Stable SHA256 over the envelope's structural contents.

    Walks dict/list/tuple structure in deterministic order, hashes tensor
    byte contents (device-independent — copied to CPU and read via numpy),
    and ``repr()``s scalar values. Two envelopes with bit-equal tensor data
    and equal scalar values produce the same digest regardless of which
    device the tensors lived on.
    """
    h = hashlib.sha256()

    def walk(obj: Any) -> None:
        if isinstance(obj, torch.Tensor):
            h.update(b"T")
            h.update(repr(tuple(obj.shape)).encode())
            h.update(repr(obj.dtype).encode())
            h.update(obj.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(obj, dict):
            h.update(b"{")
            for k in sorted(obj.keys(), key=repr):
                h.update(b"K")
                h.update(repr(k).encode())
                walk(obj[k])
            h.update(b"}")
        elif isinstance(obj, (list, tuple)):
            h.update(b"[" if isinstance(obj, list) else b"(")
            for item in obj:
                walk(item)
            h.update(b"]" if isinstance(obj, list) else b")")
        else:
            h.update(b"V")
            h.update(repr(obj).encode())

    walk({"run_id": envelope.run_id, "resume": envelope.resume})
    return h.hexdigest()


def save_shard(envelope: ShardEnvelope, run_dir: Path, step: int, rank: int) -> None:
    """Persist ``envelope`` to its canonical shard path, plus the sibling hash file."""
    assert envelope.run_id == run_dir.name, (
        f"envelope run_id={envelope.run_id!r} doesn't match run_dir.name={run_dir.name!r} — "
        f"refusing to write a shard whose identity tag disagrees with its on-disk location."
    )
    payload = {"run_id": envelope.run_id, "resume": envelope.resume}
    path = shard_path(run_dir, step, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    state_hash_path(run_dir, step, rank).write_text(compute_state_hash(envelope))


def load_shard(run_dir: Path, step: int, rank: int) -> ShardEnvelope:
    """Read a rank's shard back into a :class:`ShardEnvelope`.

    ``weights_only=False`` because the resume blob contains arbitrary cfg
    dicts (model_dump output) alongside tensors.

    Asserts the saved ``run_id`` matches ``run_dir.name`` — if they don't,
    the caller pointed at a directory whose contents came from a different
    run, and we fail loud before any model state gets corrupted.
    """
    path = shard_path(run_dir, step, rank)
    assert path.is_file(), f"resume shard not found: {path}"
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    saved_run_id = payload["run_id"]
    assert saved_run_id == run_dir.name, (
        f"resume shard identity mismatch at {path}:\n"
        f"  expected run_id (= dir name): {run_dir.name!r}\n"
        f"  shard's saved run_id:         {saved_run_id!r}\n"
        f"the shard came from a different run than the dir it lives in — "
        f"someone copied / moved shards across runs, or you pointed at the "
        f"wrong parent."
    )
    return ShardEnvelope(run_id=saved_run_id, resume=payload["resume"])


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
