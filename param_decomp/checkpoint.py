"""Checkpoint / resume of the generic trainer's `TrainState` via orbax (SPEC S22).

Each checkpoint step holds TWO orbax items, splitting the product from the process:

- `decomposition` — `train.Decomposition` (V/U components + ci_fn), the trained product.
  Every consumer (harvest/autointerp/clustering/app/fine-tune init) restores ONLY this
  item, with zero knowledge of how training initializes its optimizers or adversaries.
- `training` — `TrainingItem` (both optimizer states, the persistent adversaries, the
  step counter), the trainer-only trajectory tail. Only trainer resume touches it.

Both items save **sharded** (every process writes its own shards, no full-gather on the
training loop) and restore onto the reference state's shardings. The frozen target is
NOT saved (SPEC §3): resume rebuilds it from HF and loads only the trajectory.

Synchronous saves (no async): a SIGTERM-triggered save must be on disk before the
process exits for SLURM requeue-resume.

`init_from_parent` is the fine-tune entry (SPEC S33): a fresh run loads a PARENT
checkpoint's `decomposition` (the trained product) but starts a clean schedule —
fresh optimizer / sources, `step=0` — under a NEW config (changed LR / coeffs / steps,
same component & ci-fn structure).
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
from jaxtyping import Array
from orbax.checkpoint.type_handlers import ArrayHandler, register_type_handler

from param_decomp.adversary import PersistentAdversary
from param_decomp.train import Decomposition, TrainState

# Replica-parallel writes (multiple hosts cooperatively writing a REPLICATED array)
# hit a Shard-internals incompatibility on multi-controller jax 0.10 and buy nothing
# here: the big leaves (V/U + moments) are C-sharded, the replicated leaves (sources,
# scalars) are small. Single-replica writes are correct and simple.
register_type_handler(jax.Array, ArrayHandler(use_replica_parallel=False), override=True)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainingItem:
    """The `training` checkpoint item: the trajectory state that is training's business
    only — no consumer restores it."""

    components_opt_state: optax.OptState
    ci_fn_opt_state: optax.OptState
    adversaries: dict[str, PersistentAdversary]
    step: Array


def _split(state: TrainState) -> tuple[Decomposition, TrainingItem]:
    return (
        Decomposition(components=state.components, ci_fn=state.ci_fn),
        TrainingItem(
            components_opt_state=state.components_opt_state,
            ci_fn_opt_state=state.ci_fn_opt_state,
            adversaries=state.adversaries,
            step=state.step,
        ),
    )


def _join(decomposition: Decomposition, training: TrainingItem) -> TrainState:
    return TrainState(
        components=decomposition.components,
        ci_fn=decomposition.ci_fn,
        components_opt_state=training.components_opt_state,
        ci_fn_opt_state=training.ci_fn_opt_state,
        adversaries=training.adversaries,
        step=training.step,
    )


def make_checkpoint_manager(ckpt_dir: Path, keep_last: int) -> ocp.CheckpointManager:
    return ocp.CheckpointManager(
        ckpt_dir.resolve(),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=keep_last,
            enable_async_checkpointing=False,
        ),
    )


def save_state(mgr: ocp.CheckpointManager, step: int, state: TrainState) -> None:
    decomposition, training = _split(state)
    mgr.save(
        step,
        args=ocp.args.Composite(
            decomposition=ocp.args.StandardSave(decomposition),
            training=ocp.args.StandardSave(training),
        ),
    )
    mgr.wait_until_finished()


def restore_step(mgr: ocp.CheckpointManager, reference: TrainState, step: int) -> TrainState:
    """Restore checkpoint `step` onto `reference`'s shapes/dtypes/shardings
    (a freshly-initialised, correctly-placed `TrainState`)."""
    abstract_decomp, abstract_training = jax.tree.map(
        ocp.utils.to_shape_dtype_struct, _split(reference)
    )
    composite = mgr.restore(
        step,
        args=ocp.args.Composite(
            decomposition=ocp.args.StandardRestore(abstract_decomp),
            training=ocp.args.StandardRestore(abstract_training),
        ),
    )
    restored = _join(composite["decomposition"], composite["training"])
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


def restore_decomposition_to_host(
    mgr: ocp.CheckpointManager, step: int, abstract: Decomposition
) -> Decomposition:
    """`restore_decomposition` for a consumer without the run's device topology: leaves
    restore as HOST numpy (`abstract` needs no shardings — `jax.eval_shape` over
    `run_state.init_decomposition` yields it with zero allocation), and the caller
    `jax.device_put`s the result wherever it computes."""
    host_args_for_leaf = lambda _: ocp.RestoreArgs(restore_type=np.ndarray)  # noqa: E731
    restore_args = jax.tree.map(host_args_for_leaf, abstract)
    composite = mgr.restore(
        step,
        args=ocp.args.Composite(
            decomposition=ocp.args.PyTreeRestore(item=abstract, restore_args=restore_args)
        ),
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
    parent_mgr = make_checkpoint_manager(parent_ckpt_dir, keep_last=1)
    assert parent_step in parent_mgr.all_steps(), (
        f"parent step {parent_step} not in {parent_ckpt_dir} (have {sorted(parent_mgr.all_steps())})"
    )
    abstract = jax.tree.map(ocp.utils.to_shape_dtype_struct, _split(reference)[0])
    parent = restore_decomposition(parent_mgr, parent_step, abstract)
    return dataclasses.replace(
        reference,
        components=parent.components,
        ci_fn=parent.ci_fn,
    )
