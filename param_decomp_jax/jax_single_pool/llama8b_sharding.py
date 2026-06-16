"""GSPMD sharding plan for the Llama-8B single-pool step — the FSDP-style memory story.

The memory consumers, and how each is placed on the 1-D `dp` mesh:

  * frozen suffix (`Target`): REPLICATED. ~3.6B bf16 params (14 blocks + lm_head) ~=
    7.3GB/device. Small relative to activations; replicating avoids all-gathering the
    target every forward.
  * components (V/U) + their Adam states: SHARDED over `dp` (the FSDP analog). The fp32
    masters + fp32 Adam m/v are the dominant non-activation footprint; sharding each
    site's C axis splits all three across devices -> 1/n_dev per device.
  * CI fn + Adam states: SHARDED over `dp` along the largest axis (out head, in_proj).
  * PGD source (broadcast scope, `{site: (1,T,C+1)}`): REPLICATED. A single adversarial
    source shared across the global batch; it combines elementwise with the batch-sharded
    CI and its grad reduction falls out of the global-mean loss (torch
    `reduce_source_grads` analog). Tiny vs activations, so replicating costs nothing;
    the C+1 axis is odd and cannot tile the mesh anyway. `SrcAdamState` mirrors it.
  * residual input + all activations: BATCH-sharded over `dp`. The masked suffix
    re-forwards then run on per-device sub-batches -> activation memory scales 1/n_dev.
    This is what unlocks a global batch that OOMs replicated on one device.

Sharding V/U over the C axis keeps every einsum valid: `x @ V` contracts d_in and
produces a C-sharded result; `(.) @ U` contracts the sharded C and `jax.jit` inserts
the reduce-scatter / all-reduce. No manual collectives.
"""

from functools import partial
from typing import Any

import equinox as eqx
import jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PRNGKeyArray

from jax_single_pool.adversary import SourceParameterization, init_persistent_sources
from jax_single_pool.ci_fn import CIArch, CIFn, init_ci_fn
from jax_single_pool.llama8b import DecompVU, Target, init_decomp_vu
from jax_single_pool.lm import SiteSpec
from jax_single_pool.sharding import dp_mesh
from jax_single_pool.sharding import shard_batch as _generic_shard_batch

__all__ = [
    "dp_mesh",
    "replicate_target",
    "init_decomp_vu_sharded",
    "init_ci_fn_sharded",
    "init_sources_sharded",
    "shard_batch",
]


def _put(x: Any, sharding: NamedSharding) -> Any:
    return jax.device_put(x, sharding) if eqx.is_array(x) else x


def replicate_target(tgt: Target, mesh: Mesh) -> Target:
    repl = NamedSharding(mesh, P())
    return jax.tree.map(lambda a: _put(a, repl), tgt)


def init_decomp_vu_sharded(sites: tuple[SiteSpec, ...], key: PRNGKeyArray, mesh: Mesh) -> DecompVU:
    """Seeded per-site V/U init directly into the C-sharded global placement.

    The init runs under jit with `out_shardings`, so each device generates only its own
    shard and no host-side full tree ever exists. (Eager `device_put` of a host tree
    onto a multi-process non-replicated sharding triggers jax's cross-process
    value-equality check — a `process_allgather` of the whole tree, a 168 GiB
    allocation for a 12-layer chunk at C=24576.)

    Per site: V `(d_in, C_s)` shards C on axis 1; U `(C_s, d_out)` on axis 0."""
    n = mesh.devices.size
    for spec in sites:
        assert spec.C % n == 0, f"{spec.name}: C={spec.C} not divisible by mesh size {n}"
    site_c = {spec.name: spec.C for spec in sites}
    shard_V = NamedSharding(mesh, P(None, "dp"))
    shard_U = NamedSharding(mesh, P("dp", None))
    init = partial(init_decomp_vu, sites)

    def place(path: tuple[Any, ...], shape: jax.ShapeDtypeStruct) -> NamedSharding:
        *_, site_key, vu_key = path
        assert isinstance(site_key, jax.tree_util.DictKey), path
        assert isinstance(vu_key, jax.tree_util.SequenceKey), path
        C = site_c[site_key.key]
        is_V = vu_key.idx == 0
        assert shape.shape[1 if is_V else 0] == C, (path, shape.shape)
        return shard_V if is_V else shard_U

    out_shardings = jax.tree_util.tree_map_with_path(place, jax.eval_shape(init, key))
    return jax.jit(init, out_shardings=out_shardings)(key)


def init_ci_fn_sharded(
    arch: CIArch, sites: tuple[SiteSpec, ...], key: PRNGKeyArray, mesh: Mesh
) -> CIFn:
    """Seeded CI-fn init directly into its sharded global placement (jit +
    `out_shardings`; same no-host-tree rationale as `init_decomp_vu_sharded`).

    Placement: the largest matrices shard over `dp` — `out_w` (d_model, ΣC) on the ΣC
    axis, per-block 2-D weights on their last axis where divisible; 1-D vectors
    (biases, inv_freq) replicate."""
    n = mesh.devices.size
    repl = NamedSharding(mesh, P())
    shard_last = NamedSharding(mesh, P(None, "dp"))
    init = partial(init_ci_fn, arch, sites)

    def place(shape: jax.ShapeDtypeStruct) -> NamedSharding:
        return shard_last if shape.ndim == 2 and shape.shape[-1] % n == 0 else repl

    out_shardings = jax.tree.map(place, jax.eval_shape(init, key))
    return jax.jit(init, out_shardings=out_shardings)(key)


def init_sources_sharded(
    site_names: tuple[str, ...],
    site_component_counts: tuple[int, ...],
    seq_len: int,
    parameterization: SourceParameterization,
    key: PRNGKeyArray,
    mesh: Mesh,
) -> dict[str, Array]:
    """Seeded PGD-source init `{site: (1, T, C+1)}` -> REPLICATED over `dp` (jit +
    `out_shardings`; same no-host-tree rationale as `init_decomp_vu_sharded`).

    The source is a single adversarial source shared across the whole global batch
    (leading batch axis = 1, broadcast); it combines elementwise with the batch-sharded
    CI (`mask = ci + (1-ci)*source[..., :-1]`) and its grad is AVG-reduced across shards
    (torch `reduce_source_grads`). Replication is the semantically correct placement and
    the torch analog. Sharding the trailing C+1 axis is invalid anyway: with the
    weight-delta channel C+1 is odd (8193) and not divisible by the mesh size, and would
    also fight the batch-sharded elementwise combine."""
    repl = NamedSharding(mesh, P())
    init = partial(
        init_persistent_sources, site_names, site_component_counts, seq_len, parameterization
    )
    return jax.jit(init, out_shardings=repl)(key)


def shard_batch(resid_global: jax.Array, mesh: Mesh) -> jax.Array:
    """Batch-shard the residual input (b, t, d) over `dp` (axis 0)."""
    return _generic_shard_batch(resid_global, mesh, batch_axis=0)
