"""Checkpoint / resume of the generic trainer's `TrainState` via orbax (SPEC S22).

The whole trajectory — V/U + CI masters, both optimizer states, the persistent
adversary (sources + its Adam moments), and the step counter — lives in `TrainState`
as one pytree; orbax saves it **sharded** (every process writes its own shards, no
full-gather on the training loop) and restores it onto the reference state's
shardings. The frozen target is NOT saved (SPEC §3): resume rebuilds it from HF and
loads only the trajectory.

Synchronous saves (no async): a SIGTERM-triggered save must be on disk before the
process exits for SLURM requeue-resume.

`init_from_parent` is the fine-tune entry (SPEC S33): a fresh run loads a PARENT
checkpoint's V/U + ci_fn (the trained decomposition) but starts a clean schedule —
fresh optimizer / sources, `step=0` — under a NEW config (changed LR / coeffs / steps,
same component & ci-fn structure).
"""

import dataclasses
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


def init_from_parent(parent_ckpt_dir: Path, parent_step: int, reference: TrainState) -> TrainState:
    """Fine-tune init (SPEC S33): load the parent checkpoint's trained V/U + ci_fn ONTO
    `reference` (a fresh-from-init `TrainState` built from the NEW config), and keep the
    fresh reference's optimizer states, persistent sources, and `step=0`.

    Only the components and ci_fn carry over — the new schedule wants a clean Adam (no
    stale momentum scale) and a fresh adversary; the schedule recomputes from step 0 over
    the new `cfg.steps`. The full parent state is restored onto `reference`, which orbax
    requires to be shape/dtype/sharding-identical to the parent's saved state: a mismatch
    in component/ci_fn structure (different sites / C / ci-fn arch) fails the restore. The
    config-level structural guard in `run.py` is the readable pre-check before this point."""
    parent_mgr = make_checkpoint_manager(parent_ckpt_dir, keep_last=1)
    assert parent_step in parent_mgr.all_steps(), (
        f"parent step {parent_step} not in {parent_ckpt_dir} (have {sorted(parent_mgr.all_steps())})"
    )
    parent = restore_step(parent_mgr, reference, parent_step)
    # Keep reference.step (already a GLOBAL replicated zero: init_train_state builds step=0,
    # _ensure_global re-materializes it as a well-formed global array). Re-creating it as
    # jnp.zeros((), jnp.int32) yields a HOST-LOCAL SingleDeviceSharding array that orbax
    # refuses to serialize in a multi-host save.
    return dataclasses.replace(
        reference,
        components=parent.components,
        ci_fn=parent.ci_fn,
    )
