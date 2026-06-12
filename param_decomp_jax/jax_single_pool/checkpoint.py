"""Checkpoint / resume of the generic trainer's `TrainState` via orbax (SPEC S22).

The whole trajectory — V/U + CI masters, both optimizer states, the persistent
adversary (sources + its Adam moments), and the step counter — lives in `TrainState`
as one pytree; orbax saves it **sharded** (every process writes its own shards, no
full-gather on the training loop) and restores it onto the reference state's
shardings. The frozen target is NOT saved (SPEC §3): resume rebuilds it from HF and
loads only the trajectory.

Synchronous saves (no async): a SIGTERM-triggered save must be on disk before the
process exits for SLURM requeue-resume.
"""

from pathlib import Path
from typing import cast

import jax
import orbax.checkpoint as ocp
from orbax.checkpoint.type_handlers import ArrayHandler, register_type_handler

from jax_single_pool.train import TrainState

# Replica-parallel writes (multiple hosts cooperatively writing a REPLICATED array)
# hit a Shard-internals incompatibility on multi-controller jax 0.10 and buy nothing
# here: the big leaves (V/U + moments) are C-sharded, the replicated leaves (sources,
# scalars) are small. Single-replica writes are correct and simple.
register_type_handler(jax.Array, ArrayHandler(use_replica_parallel=False), override=True)


def make_checkpoint_manager(ckpt_dir: Path, keep_last: int) -> ocp.CheckpointManager:
    return ocp.CheckpointManager(
        ckpt_dir.resolve(),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=keep_last,
            enable_async_checkpointing=False,
        ),
    )


def save_state(mgr: ocp.CheckpointManager, step: int, state: TrainState) -> None:
    mgr.save(step, args=ocp.args.StandardSave(state))  # pyright: ignore[reportCallIssue]
    mgr.wait_until_finished()


def restore_step(mgr: ocp.CheckpointManager, reference: TrainState, step: int) -> TrainState:
    """Restore checkpoint `step` onto `reference`'s shapes/dtypes/shardings
    (a freshly-initialised, correctly-placed `TrainState`)."""
    abstract = jax.tree.map(ocp.utils.to_shape_dtype_struct, reference)
    restored = mgr.restore(step, args=ocp.args.StandardRestore(abstract))  # pyright: ignore[reportCallIssue]
    return cast(TrainState, restored)


def restore_latest(
    mgr: ocp.CheckpointManager, reference: TrainState
) -> tuple[TrainState, int] | None:
    """`restore_step` at the newest checkpoint; None if no checkpoint."""
    step = mgr.latest_step()
    if step is None:
        return None
    return restore_step(mgr, reference, step), step
