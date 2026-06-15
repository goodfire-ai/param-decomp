"""GSPMD sharding helpers — the JAX analog of FSDP2.

The single-pool SPMD design (the recommended JAX target, see
`jax_spike/SYNTHESIS.md`): data is sharded `P('dp')` over a 1-D device mesh,
params + PGD sources are placed with an explicit sharding, and `jax.jit` inserts
every collective (the grad all-reduce, the source-grad reduction) automatically
because the mean-losses reduce over the sharded batch axis. No manual NCCL, no
pool-coordination code.

Placement is expressed as `NamedSharding`: `replicate` for params (target-specific
plans like `llama8b_sharding.py` layer C-sharding on top), `shard_batch` for the
data axis.
"""

import os

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def init_distributed() -> bool:
    """Bring up `jax.distributed` under SLURM. No-op (False) off SLURM.

    The cluster recipe (from the spike): all GPUs visible per task
    (`--gres=gpu:8`), each process claims `local_device_ids=[SLURM_LOCALID]`.
    """
    if "SLURM_PROCID" not in os.environ:
        return False
    local_id = int(os.environ["SLURM_LOCALID"])
    n_visible = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
    assert local_id < n_visible, (
        f"SLURM_LOCALID={local_id} >= {n_visible} visible GPUs — srun packed tasks onto "
        f"too few nodes (job 50416 failure mode); launch steps with an explicit "
        f"--ntasks-per-node=<gpus-per-node>"
    )
    jax.distributed.initialize(local_device_ids=[local_id])
    return True


def dp_mesh() -> Mesh:
    return Mesh(np.array(jax.devices()), axis_names=("dp",))


def replicate(x: jax.Array, mesh: Mesh) -> jax.Array:
    return jax.device_put(x, NamedSharding(mesh, P()))


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
    assert B % mesh.devices.size == 0, (
        f"batch {B} (axis {batch_axis}) not divisible by mesh size {mesh.devices.size}"
    )
    spec: list[str | None] = [None] * full_global.ndim
    spec[batch_axis] = "dp"
    sharding = NamedSharding(mesh, P(*spec))

    per_proc = B // n_proc
    idx = jax.process_index()
    sl = [slice(None)] * full_global.ndim
    sl[batch_axis] = slice(idx * per_proc, (idx + 1) * per_proc)
    local = full_global[tuple(sl)]
    return jax.make_array_from_process_local_data(sharding, local, full_global.shape)
