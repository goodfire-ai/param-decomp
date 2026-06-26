"""GSPMD sharding helpers — the JAX analog of HSDP (hierarchical FSDP).

The HSDP single-pool SPMD design: a 2-D `(replicate, fsdp)` device mesh. `fsdp` is the
8 intra-node NVLink GPUs (the weight all-gather / grad reduce-scatter axis — kept ON-CHIP);
`replicate` is the across-node axis (one cross-node grad all-reduce per step, plus the
pure data-parallel replicas). There is NO tensor-parallel / Megatron-C axis: V/U + the CI
fn are FSDP-sharded over `fsdp` only, gathered per-layer on NVLink, with C never sharded.

Data shards over the FULL mesh (both axes) so per-rank batch = B/N. Params + PGD sources
are placed with an explicit sharding, and `jax.jit` inserts every collective (the FSDP
weight gather on `fsdp`, the grad reduce-scatter on `fsdp` + the cross-node all-reduce on
`replicate`, the source-grad reduction) automatically because the mean-losses reduce over
the batch axis. No manual NCCL, no pool-coordination code.

Placement is expressed as `NamedSharding`: target-specific plans (like
`llama8b_sharding.py`) FSDP-shard params on `fsdp`; `shard_batch` shards the data axis over
the full mesh.
"""

import os

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

_GPUS_PER_NODE = 8


def init_distributed(dp: int | None) -> bool:
    """Bring up `jax.distributed` iff `dp` is set. Distributedness is config-driven
    (`runtime.dp`), NEVER inferred from ambient SLURM env — `SLURM_PROCID` is present in
    every process on a SLURM box (incl. a pytest worker), so sniffing it would wrongly
    fire `jax.distributed.initialize` mid-test.

    `dp is None` → single device, no-op (return False). Otherwise the cluster recipe:
    ONE process per node, each owning all its local GPUs (mirrors the torch torchrun
    model — the launcher runs srun `--ntasks-per-node=1`). jax auto-detects the SLURM
    topology (process_id = node rank, num_processes = node count) but its SLURM cluster
    env claims only ONE device per process by default, so we pass the full local device
    list explicitly (`CUDA_VISIBLE_DEVICES`, set to all 8 by `--gpus-per-node=8`). The
    realized total device count must equal `dp`. This avoids the 8-tasks-per-node srun
    placement that the cluster's `CR_Pack_Nodes` selection packs onto one node. `dp`
    (config) decides distributedness; SLURM env only supplies the topology.
    """
    if dp is None:
        return False
    assert dp % _GPUS_PER_NODE == 0, f"dp={dp} must be a multiple of {_GPUS_PER_NODE} (GPUs/node)"
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_local = len([d for d in cuda_visible.split(",") if d]) or _GPUS_PER_NODE
    jax.distributed.initialize(local_device_ids=list(range(n_local)))
    assert jax.device_count() == dp, (
        f"runtime.dp={dp} != realized device count {jax.device_count()} "
        f"({jax.process_count()} procs × {jax.local_device_count()} local GPUs; "
        f"CUDA_VISIBLE_DEVICES={cuda_visible!r}) — the config's declared world size must "
        f"match the launch topology (nodes × {_GPUS_PER_NODE})"
    )
    return True


BATCH_AXES = ("replicate", "fsdp")
"""The full-mesh batch sharding: data shards over BOTH axes (per-rank batch = B/N)."""


def hsdp_mesh() -> Mesh:
    """The 2-D HSDP device mesh `(replicate, fsdp)`. `fsdp` is the intra-node NVLink axis
    (the FSDP weight-gather / grad-reduce axis), so it is the FAST-VARYING / minor axis of
    the reshape — `jax.devices()` lists a node's GPUs contiguously, so a row of the
    `(n_nodes, GPUS_PER_NODE)` reshape is exactly one node. `replicate` (= n_devices // 8)
    is the across-node axis. At a single node (8 devices) `replicate` is size 1; on CPU sim
    with a non-multiple-of-8 device count the `fsdp` axis takes the full count and
    `replicate` is 1 (so the divisibility asserts still bite on the real shard dims)."""
    devices = np.array(jax.devices())
    n = devices.size
    fsdp = _GPUS_PER_NODE if n % _GPUS_PER_NODE == 0 else n
    return Mesh(devices.reshape(n // fsdp, fsdp), axis_names=BATCH_AXES)


def place_via_shardings[T](tree: T, shardings: T) -> T:
    """Eager `device_put` of each array leaf of `tree` onto the matching `NamedSharding`
    leaf of `shardings` (a same-structure pytree, e.g. from a model's `.shardings(mesh)`).
    Static / non-array leaves pass through. The apply path for an already-loaded frozen
    model (vs the jitted `out_shardings` init path for freshly-seeded params)."""
    is_array = lambda x: hasattr(x, "shape") and hasattr(x, "dtype")  # noqa: E731
    return jax.tree.map(
        lambda a, s: jax.device_put(a, s) if is_array(a) else a,
        tree,
        shardings,
        is_leaf=lambda x: isinstance(x, NamedSharding),
    )


def assert_divisible(dim: int, mesh: Mesh, axis: str, what: str) -> None:
    """Fail loud if a dim sharded on mesh `axis` cannot tile that axis. Uniform across mesh
    sizes — at axis size 1 it is trivially true, so there is no single-device special case.
    `what` names the model / field / axis so a non-dividing dim crashes with a clear
    message rather than silently replicating."""
    n = mesh.shape[axis]
    assert dim % n == 0, f"{what}: dim {dim} not divisible by mesh axis '{axis}' size {n}"


def batch_shard_leading(x: jax.Array, mesh: Mesh | None) -> jax.Array:
    """In-jit `with_sharding_constraint` pinning the LEADING (batch) axis over the FULL mesh
    (`('replicate', 'fsdp')`), the rest replicated. `mesh is None` (single device) is a
    passthrough. Keeps the masked re-forwards on per-rank sub-batches (activation memory
    1/N)."""
    if mesh is None:
        return x
    spec = [BATCH_AXES] + [None] * (x.ndim - 1)
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(*spec)))


def shard_batch(full_global: jax.Array, mesh: Mesh, batch_axis: int) -> jax.Array:
    """Shard `full_global` over the FULL mesh (`('replicate', 'fsdp')`) along `batch_axis`.
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
    assert B % mesh.devices.size == 0, (
        f"batch {B} (axis {batch_axis}) not divisible by mesh size {mesh.devices.size}"
    )
    spec: list[str | tuple[str, ...] | None] = [None] * full_global.ndim
    spec[batch_axis] = BATCH_AXES
    sharding = NamedSharding(mesh, P(*spec))

    per_proc = B // n_proc
    idx = jax.process_index()
    sl = [slice(None)] * full_global.ndim
    sl[batch_axis] = slice(idx * per_proc, (idx + 1) * per_proc)
    local = full_global[tuple(sl)]
    return jax.make_array_from_process_local_data(sharding, local, full_global.shape)
