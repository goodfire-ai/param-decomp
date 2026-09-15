"""Sharding helpers for the logical device meshes: the 3-D `(replicate, fsdp, tp)`
HSDP mesh and the 2-D `(data, tp)` mesh of the `*-replicated-resident` placements.

`fsdp` shards parameter dimensions used by HSDP, while `tp` tensor-parallelizes
declared target dimensions; residency leaves fsdp nothing to shard, so the resident
mesh spells only the merged data axis. Batches shard over every non-`tp` axis
(`placement.batch_axes`) and replicate over `tp`. Process and node boundaries are
allocation facts, not mesh-shape constraints.

No mesh axis is REQUIRED to coincide with a hardware boundary: `initialize_topology`
checks that the realized device count equals the authored world size (and that the fleet
is a device kind the trainer knows), nothing about which devices share a node. Alignment
is an AUTHORED property of a config. The maintained seats put `replicate` on node
boundaries and the `(fsdp, tp)` plane on NVLink, and the owner preset's headline
properties — zero cross-node weight collectives per step, node-local muon
Newton-Schulz — hold exactly when a config is authored that way. A convention-breaking
mesh is valid and numerically identical; it forfeits only that locality.

The mesh axes are EXPLICIT (`jax.sharding.AxisType.Explicit`): every traced array
carries its sharding in its type, layout transitions are `jax.sharding.reshard`, and
the compute-weight residents are typed `reduced` over their gathered master axes
(`replicate` on the HSDP mesh, `data` on the resident mesh) so their backward
reduction is deferred to the master boundary (one reduce-scatter at exit, zero
cross-replica collectives inside the layer scan).
"""

import math

import jax
import numpy as np
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core.axes import MeshAxis
from param_decomp.core.configs import HsdpMeshShape, MeshShape, ResidentMeshShape
from param_decomp.core.hardware_utilization import checked_device_kind
from param_decomp.core.model import DecomposedModel, PlacedModel
from param_decomp.core.placement import PlacementRules, batch_axes

HSDP_MESH_AXES: tuple[MeshAxis, MeshAxis, MeshAxis] = ("replicate", "fsdp", "tp")
"""The 3-D mesh's axis names, in the device-grid order the constructors reshape to."""

RESIDENT_MESH_AXES: tuple[MeshAxis, MeshAxis] = ("data", "tp")
"""The 2-D resident mesh's axis names, in device-grid order."""


def initialize_topology(world_size: int, local_device_count: int) -> None:
    """Bring up exactly the process topology declared by the launch boundary, on a fleet
    whose device kind the trainer knows (`hardware_utilization.DeviceKind`) — the
    refusal fires here, before the target loads and the step compiles."""
    assert world_size % local_device_count == 0, (world_size, local_device_count)
    if world_size > local_device_count:
        jax.distributed.initialize(local_device_ids=list(range(local_device_count)))
    else:
        assert jax.process_count() == 1, jax.process_count()
    assert jax.device_count() == world_size, (
        f"declared world size {world_size} != realized device count {jax.device_count()} "
        f"({jax.process_count()} processes × {jax.local_device_count()} local devices)"
    )
    checked_device_kind(jax.devices())


def mesh_over_devices(shape: MeshShape, devices: np.ndarray) -> Mesh:
    """The declared mesh shape realized over an explicit device array — THE one
    shape→mesh reshape (the fit check hands it a fabricated topology;
    `mesh_for_shape` the process's real devices)."""
    match shape:
        case HsdpMeshShape(replicate=replicate, fsdp=fsdp, tp=tp):
            assert devices.size == shape.world_size, (devices.size, shape)
            return Mesh(
                devices.reshape(replicate, fsdp, tp),
                axis_names=HSDP_MESH_AXES,
                axis_types=(AxisType.Explicit,) * 3,
            )
        case ResidentMeshShape(data=data, tp=tp):
            assert devices.size == shape.world_size, (devices.size, shape)
            return Mesh(
                devices.reshape(data, tp),
                axis_names=RESIDENT_MESH_AXES,
                axis_types=(AxisType.Explicit,) * 2,
            )


def mesh_for_shape(shape: MeshShape) -> Mesh:
    """The declared `runtime.mesh` shape, realized over this process's devices."""
    return mesh_over_devices(shape, np.array(jax.devices()))


def hsdp_mesh(replicate: int, fsdp: int, tp: int) -> Mesh:
    """The explicitly declared `(replicate, fsdp, tp)` device mesh."""
    return mesh_for_shape(HsdpMeshShape(replicate=replicate, fsdp=fsdp, tp=tp))


def resident_mesh(data: int, tp: int) -> Mesh:
    """The two-axis `(data, tp)` device mesh of the `*-replicated-resident` placements —
    the whole ÷tp working copy is resident, so no parameter-sharding axis exists."""
    return mesh_for_shape(ResidentMeshShape(data=data, tp=tp))


def single_device_mesh() -> Mesh:
    """The degenerate `(1, 1, 1)` mesh for a domain that is single-device BY CONSTRUCTION
    (the toys: one CPU process, seconds of training, so no `dp` to author). Asserts the
    world it assumes rather than absorbing whatever devices happen to be visible — a toy
    started inside someone's 8-GPU allocation is a mis-targeted job, not a free speedup."""
    assert jax.process_count() == 1 and jax.device_count() == 1, (
        f"single-device by construction, found {jax.process_count()} processes × "
        f"{jax.local_device_count()} local devices"
    )
    return hsdp_mesh(1, 1, 1)


def hsdp_abstract_mesh(replicate: int, fsdp: int, tp: int) -> AbstractMesh:
    return AbstractMesh(
        (replicate, fsdp, tp),
        HSDP_MESH_AXES,
        axis_types=(AxisType.Explicit,) * 3,
    )


def resident_abstract_mesh(data: int, tp: int) -> AbstractMesh:
    return AbstractMesh((data, tp), RESIDENT_MESH_AXES, axis_types=(AxisType.Explicit,) * 2)


def abstract_mesh_for_shape(shape: MeshShape) -> AbstractMesh:
    """`mesh_for_shape` without devices — for pre-submission placement claims."""
    match shape:
        case HsdpMeshShape(replicate=replicate, fsdp=fsdp, tp=tp):
            return hsdp_abstract_mesh(replicate, fsdp, tp)
        case ResidentMeshShape(data=data, tp=tp):
            return resident_abstract_mesh(data, tp)


def data_parallel_size(mesh: Mesh | AbstractMesh) -> int:
    """Number of distinct batch shards; `tp` holds replicas of each shard."""
    return math.prod(mesh.shape[axis] for axis in batch_axes(mesh))


def local_data_parallel_size(mesh: Mesh | None) -> int:
    """Distinct batch shards addressable by this process."""
    if mesh is None:
        return 1
    local_shape = mesh.local_mesh.shape
    return math.prod(local_shape[axis] for axis in batch_axes(mesh))


def place_via_shardings[T](tree: T, shardings: T) -> T:
    """Place each array leaf of `tree` onto the matching `NamedSharding` leaf of `shardings`
    (a same-structure pytree, e.g. from a model's `.shardings(placement)`). Static / non-array
    leaves pass through. The apply path for an already-loaded frozen model (vs the jitted
    `out_shardings` init path for freshly-seeded params).

    Every leaf goes through `make_array_from_callback`, never `device_put`. Each process
    loaded the same complete frozen leaf, so the callback can serve its addressable shards
    directly. `device_put(local, multi-process sharding)` first checks cross-host replica
    equality; for an FSDP target leaf replicated across the process axis that check gathers
    the full leaf on every process before slicing it (56 GiB for full32L's stacked MLP
    weights)."""
    is_array = lambda x: hasattr(x, "shape") and hasattr(x, "dtype")  # noqa: E731
    place = lambda a, s: jax.make_array_from_callback(  # noqa: E731
        a.shape, s, lambda index: a[index]
    )
    return jax.tree.map(
        lambda a, s: place(a, s) if is_array(a) else a,
        tree,
        shardings,
        is_leaf=lambda x: isinstance(x, NamedSharding),
    )


def place_target[Out, PreparedT](
    tgt: DecomposedModel[Out, PreparedT], placement: PlacementRules
) -> PlacedModel[Out, PreparedT]:
    """Eagerly place an already-loaded frozen target using the run's resolved rules, and
    bundle it with them — THE one placed-model assembly point."""
    placed = place_via_shardings(tgt, tgt.shardings(placement))
    return PlacedModel(model=placed, placement=placement)


def target_shardings_audit[Out](
    placed: PlacedModel[Out],
) -> dict[str, tuple[NamedSharding, tuple[int, ...]]]:
    """Audit every dynamic frozen-target leaf against its declared sharding tree."""
    target, placement = placed.model, placed.placement
    assert placement is not None, "an unplaced model has no sharding tree to audit"
    shardings = target.shardings(placement)
    target_with_paths, target_tree = jax.tree_util.tree_flatten_with_path(target)
    sharding_leaves, sharding_tree = jax.tree.flatten(
        shardings, is_leaf=lambda value: isinstance(value, NamedSharding)
    )
    assert target_tree == sharding_tree, "target and target sharding trees differ"
    out: dict[str, tuple[NamedSharding, tuple[int, ...]]] = {}
    for (path, value), sharding in zip(target_with_paths, sharding_leaves, strict=True):
        assert hasattr(value, "shape"), (jax.tree_util.keystr(path), type(value))
        assert isinstance(sharding, NamedSharding), (
            jax.tree_util.keystr(path),
            type(sharding),
        )
        out[f"target{jax.tree_util.keystr(path)}"] = (sharding, value.shape)
    return out


def batch_shard_leading(x: jax.Array, mesh: Mesh | None) -> jax.Array:
    """In-jit reshard pinning the LEADING (batch) axis over the full data mesh
    (`placement.batch_axes`), the rest replicated. `mesh is None` (single device) is a
    passthrough. Keeps the masked re-forwards on per-rank sub-batches (activation memory
    1/N)."""
    if mesh is None:
        return x
    spec = [batch_axes(mesh)] + [None] * (x.ndim - 1)
    return jax.sharding.reshard(x, NamedSharding(mesh, P(*spec)))


def shard_batch(full_global: jax.Array, mesh: Mesh, batch_axis: int) -> jax.Array:
    """Shard `full_global` over the data axes and replicate each shard over `tp`.
    Generated identically on every process (same seed), so each process slices out its
    process-local sub-batch and `make_array_from_process_local_data` does the device
    placement.

    Works for both topologies the spike uses: single-process / many-devices (CPU
    sim, or 1 process with N local GPUs — the process owns the whole batch and it
    splits across the local devices) and multi-process / 1-device-each (SLURM —
    each process owns its 1/n_processes slice). `batch_axis` is axis 1 for the
    stacked-site `[S, B, ..., d]` layout.
    """
    n_proc = jax.process_count()
    B = full_global.shape[batch_axis]
    n_data = data_parallel_size(mesh)
    assert B % n_data == 0, (
        f"batch {B} (axis {batch_axis}) not divisible by data-parallel size {n_data}"
    )
    spec: list[str | tuple[str, ...] | None] = [None] * full_global.ndim
    spec[batch_axis] = batch_axes(mesh)
    sharding = NamedSharding(mesh, P(*spec))

    per_proc = B // n_proc
    idx = jax.process_index()
    sl = [slice(None)] * full_global.ndim
    sl[batch_axis] = slice(idx * per_proc, (idx + 1) * per_proc)
    local = full_global[tuple(sl)]
    return jax.make_array_from_process_local_data(sharding, local, full_global.shape)
