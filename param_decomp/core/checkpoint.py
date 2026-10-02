"""Checkpoint and resume for plain and targeted PD via orbax.

Each checkpoint step holds TWO orbax items, splitting the product from the process:

- `decomposition` — `train.Decomposition` (V/U components + ci_fn), the trained product.
  Every downstream consumer (including fine-tune initialization) restores ONLY this
  item, with zero knowledge of how training initializes its optimizers or adversaries.
- `training` — the algorithm-specific objective and trajectory, including optimizer
  states, persistent adversaries, and the step counter. Only trainer resume touches it.

Both algorithms compose these two items, so save/restore map onto their
own `.decomposition` / `.training` fields with no regrouping.

Both items save **sharded** (every process writes its own shards, no full-gather on the
training loop) and restore directly into the consumer's declared array formats.
The frozen target is NOT saved: resume rebuilds it from HF and loads only the trajectory.

Checkpoints are therefore topology-free: orbax saves the LOGICAL array, and restore
places values using shapes from the initializer and layouts from the compiled consumer
on the restoring side's OWN mesh. Any checkpoint restores onto any mesh whose placement
constructs, train mesh to train mesh included; consumers re-place finished runs the
same way, on whatever layout their caller names.
Pinned by the cross-topology restore tests in `param_decomp/tests/core/test_checkpoint.py`.

Synchronous saves (no async): a SIGTERM-triggered save must be on disk before the
process exits for restart and resume.

Fine-tuning restores only the decomposition, then initializes fresh training state
under the new config. The parent optimizer and adversary history are never read.
"""

from pathlib import Path
from typing import cast

import jax
import orbax.checkpoint as ocp
from beartype import beartype
from jax.experimental.layout import Format
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import jaxtyped
from orbax.checkpoint.checkpoint_managers import PreservationPolicy, preservation_policy
from orbax.checkpoint.type_handlers import ArrayHandler, register_type_handler

from param_decomp.core.components import ComponentStacks
from param_decomp.core.configs import (
    CheckpointRetention,
    KeepAllCheckpoints,
    KeepLastNCheckpoints,
)
from param_decomp.core.pytree import FormatTree, ShapeTree
from param_decomp.core.train import TrainingProgress, TrainState

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
    """A manager for consumers that only restore (fine-tune parent init, `open_run`, and
    other downstream readers). `read_only` makes orbax refuse both saves and deletes, so no
    retention question arises: a reader of someone else's run has no say in what that run
    keeps on disk."""
    return ocp.CheckpointManager(
        ckpt_dir.resolve(), options=ocp.CheckpointManagerOptions(read_only=True)
    )


def save_state[Conditioning, Training: TrainingProgress](
    mgr: ocp.CheckpointManager, step: int, state: TrainState[Conditioning, Training]
) -> None:
    mgr.save(
        step,
        args=ocp.args.Composite(
            decomposition=ocp.args.StandardSave(state.decomposition),
            training=ocp.args.StandardSave(state.training),
        ),
    )
    mgr.wait_until_finished()


@jaxtyped(typechecker=beartype)
def restore_destination[Tree](
    abstract: ShapeTree[Tree], formats: FormatTree, mesh: Mesh
) -> ShapeTree[Tree]:
    """Pair initializer shapes with the compiled consumer's physical input formats."""

    def destination(shape: jax.ShapeDtypeStruct, fmt: Format) -> jax.ShapeDtypeStruct:
        if fmt.sharding is None:
            # Pruned inputs have no consumer layout; retain their declared placement.
            assert fmt.layout is None
            match shape.sharding:
                case None:
                    sharding = NamedSharding(mesh, P())
                case NamedSharding(spec=spec):
                    sharding = NamedSharding(mesh, spec)
                case unexpected:
                    raise TypeError(f"unexpected initializer sharding: {unexpected}")
            fmt = Format(None, sharding)
        return jax.ShapeDtypeStruct(shape.shape, shape.dtype, sharding=fmt)

    return jax.tree.map(destination, abstract, formats)


@jaxtyped(typechecker=beartype)
def restore_step[Conditioning, Training: TrainingProgress](
    mgr: ocp.CheckpointManager,
    destination: ShapeTree[TrainState[Conditioning, Training]],
    step: int,
) -> TrainState[Conditioning, Training]:
    """Load checkpoint values directly into the declared shapes and physical layouts."""
    # Orbax uploads local shards before assembling global arrays.
    with jax.set_mesh(None):
        composite = mgr.restore(
            step,
            args=ocp.args.Composite(
                decomposition=ocp.args.StandardRestore(
                    destination.decomposition, support_layout=True
                ),
                training=ocp.args.StandardRestore(destination.training, support_layout=True),
            ),
        )
    return TrainState(decomposition=composite["decomposition"], training=composite["training"])


def restore_latest[Conditioning, Training: TrainingProgress](
    mgr: ocp.CheckpointManager, destination: ShapeTree[TrainState[Conditioning, Training]]
) -> tuple[TrainState[Conditioning, Training], int] | None:
    """`restore_step` at the newest checkpoint; None if no checkpoint."""
    step = mgr.latest_step()
    if step is None:
        return None
    return restore_step(mgr, destination, step), step


@jaxtyped(typechecker=beartype)
def restore_decomposition[DecompositionTree](
    mgr: ocp.CheckpointManager, step: int, abstract: ShapeTree[DecompositionTree]
) -> DecompositionTree:
    """Restore ONLY the trained decomposition of checkpoint `step` onto `abstract`'s
    shapes, dtypes, and optional physical layouts."""
    with jax.set_mesh(None):
        composite = mgr.restore(
            step,
            args=ocp.args.Composite(
                decomposition=ocp.args.StandardRestore(abstract, support_layout=True)
            ),
        )
    return cast(DecompositionTree, composite["decomposition"])


def restore_components(
    mgr: ocp.CheckpointManager, step: int, abstract: ShapeTree[ComponentStacks]
) -> ComponentStacks:
    """Restore ONLY the component stacks of checkpoint `step`'s decomposition, leaving its
    CI fn unread — a consumer that never evaluates CI needs no CI architecture."""
    item = {"components": abstract}
    with jax.set_mesh(None):
        composite = mgr.restore(
            step,
            args=ocp.args.Composite(
                decomposition=ocp.args.PyTreeRestore(
                    item=item,
                    restore_args=ocp.checkpoint_utils.construct_restore_args(item),
                    partial_restore=True,
                )
            ),
        )
    return composite["decomposition"]["components"]
