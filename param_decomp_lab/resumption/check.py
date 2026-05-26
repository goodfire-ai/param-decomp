"""RNG-independent snapshot fidelity check.

Verifies that ``Trainer.from_snapshot`` reconstructs a trainer whose own
``snapshot()`` produces a resume blob bit-identical to what was originally
saved. Catches any drift in the save→load→re-snapshot path without any
data, RNG, or dataloader involvement.

The check pairs a freshly-taken :class:`TrainerSnapshot` with the on-disk
``state_hash_rank<R>.txt`` written next to the shard at save time; the
caller is responsible for building the trainer (since the concrete trainer
class depends on the experiment's pool configuration).
"""

from pathlib import Path

from param_decomp.trainer_snapshot import TrainerSnapshot
from param_decomp_lab.resumption.shards import (
    ShardEnvelope,
    compute_state_hash,
    state_hash_path,
)


def check_snapshot_fidelity(
    *,
    fresh: TrainerSnapshot,
    parent_run_dir: Path,
    step: int,
    rank: int,
) -> str:
    """Hash ``fresh`` and assert it matches the on-disk sidecar at save time.

    Returns the matched SHA256 hex digest on success; raises on mismatch.
    """
    envelope = ShardEnvelope(run_id=parent_run_dir.name, resume=fresh.resume)
    fresh_hash = compute_state_hash(envelope)
    saved_hash_path = state_hash_path(parent_run_dir, step, rank)
    assert saved_hash_path.is_file(), f"no saved state hash at {saved_hash_path}"
    saved_hash = saved_hash_path.read_text().strip()
    assert fresh_hash == saved_hash, (
        f"state hash mismatch at rank {rank}:\n"
        f"  saved:   {saved_hash}\n"
        f"  fresh:   {fresh_hash}\n"
        f"  path:    {saved_hash_path}\n"
        f"save→load→re-snapshot is not byte-exact — resumption fidelity broken."
    )
    return saved_hash
