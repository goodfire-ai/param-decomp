"""GSPMD sharding helpers — the JAX analog of HSDP×TP (FSDP2 + Megatron + replicate-DP).

The single-pool SPMD design: a 3-D device mesh `(replicate, dp, tp)` (`dp_mesh`), where
`(dp, tp)` are the two INTRA-node axes (`dp · tp = node size = 8`) and `replicate` is the
CROSS-node data-parallel axis. Data is sharded over BOTH data-parallel axes
(`P(DATA_AXES, ...)` = `replicate × dp`); params are placed with explicit `NamedSharding`s
that name only `dp` (FSDP) + `tp` (Megatron) — never `replicate`, so weights replicate
cross-node and NO weight collective crosses IB. `jax.jit` inserts every collective: the
weight all-gather lands on the intra-node `dp` (NVLink), the C/Megatron reductions on `tp`
(NVLink), and the grad all-reduce on `replicate × dp` (the mean loss reduces over the
sharded batch). No manual NCCL, no pool-coordination code.

Placement is MODEL-OWNED: target-specific plans (like `llama8b_sharding.py`) place params
with d_in/d_out FSDP-on-`dp` + C-on-`tp`; `shard_batch` / `batch_shard_leading` shard the
data axis over `DATA_AXES`. `tp = 1` ⇒ pure HSDP (`dp = 8`); `tp = 8` ⇒ pure intra-node TP
+ cross-node replicate-DP (`dp = 1`); a single node ⇒ `replicate = 1` (the old 2-D
`(dp, tp)` mesh, weights placed identically).
"""

import os

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

_GPUS_PER_NODE = 8

DATA_AXES = ("replicate", "dp")
"""The two data-parallel mesh axes the batch shards over (cross-node `replicate` ×
intra-node FSDP `dp`). A `PartitionSpec` entry of this tuple shards one tensor axis over
BOTH — `P(DATA_AXES, ...)`. The grad all-reduce falls out over both (the mean loss reduces
the batch); weights, sharded only on `dp`+`tp`, are replicated over `replicate`."""


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


def dp_mesh(tp: int = 1) -> Mesh:
    """The 3-D HSDP×TP device mesh `(replicate, dp, tp)` — the two intra-node axes
    `(dp, tp)` partition each node's `_GPUS_PER_NODE` GPUs, `replicate` is the cross-node
    data-parallel axis.

      * `tp` — intra-node Megatron tensor-parallel degree (C-on-`tp`; on NVLink).
      * `dp` — intra-node FSDP degree (`= _GPUS_PER_NODE // tp`); shards V's d_in / U's
        d_out, gathered on NVLink. This is the FSDP axis the weight `.shardings` specs
        name (`P("dp", "tp")` etc.) — UNCHANGED by the 3-D split.
      * `replicate` — cross-node DP (`= n_devices // _GPUS_PER_NODE`); weights are
        REPLICATED here (no weight collective crosses IB), it carries only the grad
        all-reduce. Sized from the realized device count.

    Data shards over BOTH data-parallel axes `(replicate, dp)`; the masked re-forwards run
    on B/(replicate·dp) per rank. `tp = _GPUS_PER_NODE` ⇒ `dp = 1` (pure TP intra-node +
    cross-node replicate-DP); `tp = 1` ⇒ `dp = _GPUS_PER_NODE` (pure HSDP). With a single
    node (`n == _GPUS_PER_NODE`) `replicate` is size 1 — a degenerate axis, identical to the
    old 2-D `(dp, tp)` mesh for any sharding that doesn't name `replicate` (so weights,
    which never name it, are placed identically; the equivalence goldens are unchanged).

    A CPU sim (`--xla_force_host_platform_device_count=N`) presents N devices in ONE
    process, so the whole N tiles `_GPUS_PER_NODE`-worth intra-node and `replicate =
    N // _GPUS_PER_NODE`; at `N < _GPUS_PER_NODE` the node size collapses to N (no
    cross-node axis) so the 3-D mesh stays well-formed at any sim count."""
    devices = np.array(jax.devices())
    n = devices.size
    node = min(n, _GPUS_PER_NODE)
    assert n % node == 0, f"device count {n} not divisible by node size {node}"
    assert node % tp == 0, f"node size {node} not divisible by tp={tp}"
    fsdp = node // tp
    replicate = n // node
    return Mesh(devices.reshape(replicate, fsdp, tp), axis_names=("replicate", "dp", "tp"))


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
    """In-jit `with_sharding_constraint` pinning the LEADING (batch) axis to the two
    data-parallel axes `DATA_AXES` (`replicate × dp`), the rest replicated. `mesh is None`
    (single device) is a passthrough. Keeps the masked re-forwards on per-rank sub-batches
    (activation memory 1/(replicate·dp))."""
    if mesh is None:
        return x
    rest: list[None] = [None] * (x.ndim - 1)
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(DATA_AXES, *rest)))


def shard_batch(full_global: jax.Array, mesh: Mesh, batch_axis: int) -> jax.Array:
    """Shard `full_global` over 'dp' along `batch_axis`. Generated identically on
    every process (same seed), so each process slices out its process-local
    sub-batch and `make_array_from_process_local_data` does the device placement.

    Works for both topologies the spike uses: single-process / many-devices (CPU
    sim, or 1 process with N local GPUs — the process owns the whole batch and it
    splits across the local devices) and multi-process / 1-device-each (SLURM —
    each process owns its 1/n_processes slice). `batch_axis` is axis 1 for the
    stacked-site `[S, B, ..., d]` layout.
    """
    n_proc = jax.process_count()
    B = full_global.shape[batch_axis]
    n_data = mesh.shape["replicate"] * mesh.shape["dp"]
    assert B % n_data == 0, (
        f"batch {B} (axis {batch_axis}) not divisible by data-parallel size {n_data} "
        f"(replicate={mesh.shape['replicate']} × dp={mesh.shape['dp']})"
    )
    spec: list[object] = [None] * full_global.ndim
    spec[batch_axis] = DATA_AXES
    sharding = NamedSharding(mesh, P(*spec))

    per_proc = B // n_proc
    idx = jax.process_index()
    sl = [slice(None)] * full_global.ndim
    sl[batch_axis] = slice(idx * per_proc, (idx + 1) * per_proc)
    local = full_global[tuple(sl)]
    return jax.make_array_from_process_local_data(sharding, local, full_global.shape)
