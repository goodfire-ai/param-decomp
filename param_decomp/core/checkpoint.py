"""Checkpoint / resume of the generic trainer's `TrainState` via orbax (SPEC S22).

Each checkpoint step holds TWO orbax items, splitting the product from the process:

- `decomposition` — `train.Decomposition` (V/U components + ci_fn), the trained product.
  Every consumer (clustering, fine-tune initialization, and others) restores ONLY this
  item, with zero knowledge of how training initializes its optimizers or adversaries.
- `training` — `train.TrainingItem` (both optimizer states, the persistent adversaries,
  the step counter), the trainer-only trajectory tail. Only trainer resume touches it.

`TrainState` composes exactly these two — one representation — so save/restore map onto its
own `.decomposition` / `.training` fields with no regrouping.

Both items save **sharded** (every process writes its own shards, no full-gather on the
training loop) and restore onto the reference state's shardings. The frozen target is
NOT saved (SPEC §3): resume rebuilds it from HF and loads only the trajectory.

Checkpoints are therefore topology-free: orbax saves the LOGICAL array, and restore
places values by the abstract reference's shardings — a reference rebuilt from config
on the restoring side's OWN mesh. Any checkpoint restores onto any mesh whose placement
constructs, train mesh to train mesh included; consumers re-place finished runs the
same way (`zero1` on `hsdp_mesh(1, device_count, 1)` — every stack length tiles).
Pinned by the cross-topology restore tests in `tests/test_checkpoint.py`.

Synchronous saves (no async): a SIGTERM-triggered save must be on disk before the
process exits for SLURM requeue-resume.

`init_from_parent` is the fine-tune entry (SPEC S33): a fresh run loads a PARENT
checkpoint's `decomposition` (the trained product) but starts a clean schedule —
fresh optimizer / sources, `step=0` — under a NEW config (changed LR / coeffs / steps,
same component & ci-fn structure).
"""

import dataclasses
from pathlib import Path
from typing import cast

import jax
import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_managers import PreservationPolicy, preservation_policy
from orbax.checkpoint.type_handlers import ArrayHandler, register_type_handler

from param_decomp.core.configs import (
    CheckpointRetention,
    KeepAllCheckpoints,
    KeepLastNCheckpoints,
)
from param_decomp.core.train import Decomposition, TrainState

# Replica-parallel writes (multiple hosts cooperatively writing a REPLICATED array)
# hit a Shard-internals incompatibility on multi-controller jax 0.10 and buy nothing
# here: the big leaves (V/U + moments) are C-sharded, the replicated leaves (sources,
# scalars) are small. Single-replica writes are correct and simple.
register_type_handler(jax.Array, ArrayHandler(use_replica_parallel=False), override=True)


def _preservation_policy(retention: CheckpointRetention) -> PreservationPolicy:
    match retention:
        case KeepLastNCheckpoints(n=n):
            return preservation_policy.LatestN(n=n)
        case KeepAllCheckpoints():
            return preservation_policy.PreserveAll()


def make_checkpoint_manager(
    ckpt_dir: Path, retention: CheckpointRetention
) -> ocp.CheckpointManager:
    """The WRITING manager (the trainer's). Every save prunes `ckpt_dir` down to
    `retention`'s surviving set."""
    return ocp.CheckpointManager(
        ckpt_dir.resolve(),
        options=ocp.CheckpointManagerOptions(
            preservation_policy=_preservation_policy(retention),
            enable_async_checkpointing=False,
        ),
    )


def make_read_only_checkpoint_manager(ckpt_dir: Path) -> ocp.CheckpointManager:
    """A manager for consumers that only restore (fine-tune parent init, `open_jax_run`,
    clustering, and others). `read_only` makes orbax refuse both saves and deletes, so no
    retention question arises: a reader of someone else's run has no say in what that run
    keeps on disk."""
    return ocp.CheckpointManager(
        ckpt_dir.resolve(), options=ocp.CheckpointManagerOptions(read_only=True)
    )


def save_state(mgr: ocp.CheckpointManager, step: int, state: TrainState) -> None:
    mgr.save(
        step,
        args=ocp.args.Composite(
            decomposition=ocp.args.StandardSave(state.decomposition),
            training=ocp.args.StandardSave(state.training),
        ),
    )
    mgr.wait_until_finished()


def restore_step(mgr: ocp.CheckpointManager, reference: TrainState, step: int) -> TrainState:
    """Restore checkpoint `step` onto `reference`'s shapes/dtypes/shardings
    (a freshly-initialised, correctly-placed `TrainState`)."""
    abstract = jax.tree.map(ocp.utils.to_shape_dtype_struct, reference)
    composite = mgr.restore(
        step,
        args=ocp.args.Composite(
            decomposition=ocp.args.StandardRestore(abstract.decomposition),
            training=ocp.args.StandardRestore(abstract.training),
        ),
    )
    restored = TrainState(decomposition=composite["decomposition"], training=composite["training"])
    # Coerce the restored tree onto the reference's exact FORMAT (layout + sharding), not just its
    # sharding. StandardRestore already honors the sharding SPEC (verified), so a device_put onto
    # sharding alone is a no-op — but orbax-restored arrays carry a default memory LAYOUT that
    # differs from what the jitted step was compiled for. The reference is a fresh-init state built
    # by the same XLA layout assignment as the step, so its `.format` IS the step's expected input
    # layout; matching it avoids a ÷1-scale entry relayout on the first resumed step (the resume OOM).
    restored = jax.device_put(restored, jax.tree.map(lambda r: r.format, reference))
    return cast(TrainState, restored)


def restore_latest(
    mgr: ocp.CheckpointManager, reference: TrainState
) -> tuple[TrainState, int] | None:
    """`restore_step` at the newest checkpoint; None if no checkpoint."""
    step = mgr.latest_step()
    if step is None:
        return None
    return restore_step(mgr, reference, step), step


def restore_decomposition(
    mgr: ocp.CheckpointManager, step: int, abstract: Decomposition
) -> Decomposition:
    """Restore ONLY the trained decomposition of checkpoint `step` onto `abstract`'s
    shapes/dtypes/shardings (`to_shape_dtype_struct` of a correctly-placed reference)."""
    composite = mgr.restore(
        step, args=ocp.args.Composite(decomposition=ocp.args.StandardRestore(abstract))
    )
    return cast(Decomposition, composite["decomposition"])


def init_from_parent(parent_ckpt_dir: Path, parent_step: int, reference: TrainState) -> TrainState:
    """Fine-tune init (SPEC S33): load the parent checkpoint's trained decomposition ONTO
    `reference` (a fresh-from-init `TrainState` built from the NEW config), and keep the
    fresh reference's optimizer states, persistent sources, and `step=0`.

    Only the decomposition carries over — the new schedule wants a clean Adam (no stale
    momentum scale) and a fresh adversary; the schedule recomputes from step 0 over the
    new `cfg.steps`. Orbax requires the reference's decomposition to be
    shape/dtype/sharding-identical to the parent's saved one: a mismatch in
    component/ci_fn structure (different sites / C / ci-fn arch) fails the restore. The
    config-level structural guard in `run.py` is the readable pre-check before this
    point. The parent's optimizer/adversary state is never read, so it may differ freely.
    `reference.step` is kept as-is: it is already a GLOBAL replicated zero
    (`_ensure_global`); re-creating it host-local would break the multi-host save."""
    parent_mgr = make_read_only_checkpoint_manager(parent_ckpt_dir)
    assert parent_step in parent_mgr.all_steps(), (
        f"parent step {parent_step} not in {parent_ckpt_dir} (have {sorted(parent_mgr.all_steps())})"
    )
    abstract = jax.tree.map(ocp.utils.to_shape_dtype_struct, reference.decomposition)
    parent = restore_decomposition(parent_mgr, parent_step, abstract)
    return dataclasses.replace(reference, decomposition=parent)
