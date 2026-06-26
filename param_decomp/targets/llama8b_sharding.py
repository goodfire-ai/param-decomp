"""GSPMD sharding plan for the Llama-8B single-pool step — the HSDP×TP memory story.

The mesh is 3-D `(replicate, dp, tp)` (`dp·tp = node size`, `replicate` cross-node). The
memory consumers, and how each is placed:

  * frozen target: FSDP-sharded over `dp` (the ~14 GB layer bulk shards `/(dp·tp)`,
    gathered per layer in the scan), embed/lm_head/norm REPLICATED. Replicated over
    `replicate` (no cross-node gather).
  * components (V/U) + their Adam states: SHARDED over `dp` (FSDP: V's d_in / U's d_out)
    + `tp` (Megatron: C). The fp32 masters + fp32 Adam m/v are the dominant non-activation
    footprint; this splits all three `/(dp·tp)` per rank — ZeRO-1 over the INTRA-node mesh.
    Replicated over `replicate` (the bf16 compute weight gathers on `dp` only — no weight
    collective crosses IB; the proven HSDP×TP repro's tradeoff: master replicated cross-node
    so `replicate` carries ONLY the grad all-reduce, not a weight gather).
  * CI fn + Adam states: SHARDED over `dp` (FSDP) + `tp` (Megatron), same story.
  * PGD source (broadcast scope, `{site: (1,T,C+1)}`): REPLICATED. A single adversarial
    source shared across the global batch; it combines elementwise with the batch-sharded
    CI and its grad reduction falls out of the global-mean loss (torch
    `reduce_source_grads` analog). Tiny vs activations, so replicating costs nothing;
    the C+1 axis is odd and cannot tile the mesh anyway. `SrcAdamState` mirrors it.
  * token input + all activations: BATCH-sharded over `DATA_AXES` (`replicate × dp`). The
    masked re-forwards then run on per-rank sub-batches -> activation memory scales
    1/(replicate·dp). This is what unlocks a global batch that OOMs replicated on one device.

Sharding V/U over the C axis keeps every einsum valid: `x @ V` contracts d_in and
produces a C-sharded result; `(.) @ U` contracts the sharded C and `jax.jit` inserts
the reduce-scatter / all-reduce. No manual collectives.

Placement is MODEL-OWNED: each param owner declares its per-leaf `NamedSharding` via a
`.shardings(mesh)` method (V/U on `DecompVU`, the Megatron layout on the chunkwise CI fn,
all-replicate on the frozen target). The helpers below only drive the apply: compute the
shardings on the `eqx.filter_eval_shape`'d abstract model, then run the seeded init under
`jax.jit(init, out_shardings=...)` so each device generates only its own shard and no
host-side full tree exists — eager `device_put` of a host tree onto a multi-process
non-replicated sharding triggers a `process_allgather` (a 168 GiB allocation for a
12-layer chunk at C=24576). A non-dividing declared shard axis is a loud crash inside
`.shardings` (fail-fast), never a silent replicate.
"""

from functools import partial

import equinox as eqx
import jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, PRNGKeyArray

from param_decomp.adversary import init_persistent_sources
from param_decomp.ci_fn import CIFn, CIFnArch, build_ci_fn
from param_decomp.components import DecompVU, SiteSpec, init_decomp_vu
from param_decomp.configs import BSCScope, SCScope
from param_decomp.sharding import DATA_AXES, dp_mesh, place_via_shardings
from param_decomp.sharding import shard_batch as _generic_shard_batch
from param_decomp.targets.llama8b import LlamaDecomposedModel, parse_site_name

__all__ = [
    "dp_mesh",
    "place_target",
    "init_decomp_vu_placed",
    "init_ci_fn_placed",
    "init_sources_sharded",
    "shard_batch",
]


def place_target(tgt: LlamaDecomposedModel, mesh: Mesh) -> LlamaDecomposedModel:
    """Eager `device_put` of the already-loaded frozen target onto its own declared
    placement (`tgt.shardings(mesh)` — all-replicate)."""
    return place_via_shardings(tgt, tgt.shardings(mesh))


def _replicate_u_dout_sites(sites: tuple[SiteSpec, ...]) -> frozenset[str]:
    """Attention q/k/v sites, whose U d_out is the head dim: it can't FSDP on `dp` (the head
    layout re-shards to `tp` at the attention seam, and `tp` is taken by C), so its d_out
    stays replicated. o_proj/MLP d_out are plain features and FSDP normally."""
    return frozenset(s.name for s in sites if parse_site_name(s.name)[1] in ("q", "k", "v"))


def init_decomp_vu_placed(sites: tuple[SiteSpec, ...], key: PRNGKeyArray, mesh: Mesh) -> DecompVU:
    """Seeded per-site V/U init placed by `DecompVU.shardings` (uniform FSDP(`dp`)×TP(`tp`);
    q/k/v U d_out replicated). Shardings computed on the abstract model, init under jit."""
    init = partial(init_decomp_vu, sites)
    replicate_u_dout = _replicate_u_dout_sites(sites)
    out_shardings = eqx.filter_eval_shape(init, key).shardings(mesh, replicate_u_dout)
    return jax.jit(init, out_shardings=out_shardings)(key)


def init_ci_fn_placed(
    arch: CIFnArch, sites: tuple[SiteSpec, ...], key: PRNGKeyArray, mesh: Mesh
) -> CIFn:
    """Seeded CI-fn init (any arch, via `build_ci_fn`) placed by the CI fn's own
    `.shardings` (the chunkwise transformer's Megatron layout; the toy MLPs shard each
    weight's output axis). Shardings computed on the abstract model, init under jit."""
    init = partial(build_ci_fn, arch, sites)
    out_shardings = eqx.filter_eval_shape(init, key).shardings(mesh)
    return jax.jit(init, out_shardings=out_shardings)(key)


def init_sources_sharded(
    site_names: tuple[str, ...],
    site_component_counts: tuple[int, ...],
    seq_len: int,
    scope: SCScope | BSCScope,
    global_batch: int,
    source_dtype: DTypeLike,
    key: PRNGKeyArray,
    mesh: Mesh,
) -> dict[str, Array]:
    """Seeded PPGD-source init -> placed per scope (jit + `out_shardings`; same
    no-host-tree rationale as `init_decomp_vu_placed`).

    `sc`: `{site: (1, T, C+1)}` REPLICATED over `dp`. One adversarial source shared
    across the whole global batch (leading batch axis = 1, broadcast); it combines
    elementwise with the batch-sharded CI (`mask = ci + (1-ci)*source[..., :-1]`) and its
    grad is AVG-reduced across shards (torch `reduce_source_grads`).

    `bsc`: `{site: (B, T, C+1)}` BATCH-SHARDED over `DATA_AXES` (`replicate × dp`, axis 0),
    aligning each batch element's source with that element's `shard_batch`-placed
    residual/CI. The source is independent per element, so the per-element grad is already
    shard-local — NO cross-rank reduction, matching torch's `_skip_all_reduce`. (Requires
    `global_batch % (replicate·dp) == 0`, the same divisibility `shard_batch` needs.)

    Sharding the trailing C+1 axis is invalid for either scope: with the weight-delta
    channel C+1 is odd and not divisible by the mesh size, and would also fight the
    batch-sharded elementwise combine."""
    match scope:
        case SCScope():
            leading_shape, placement = (1, seq_len), NamedSharding(mesh, P())
        case BSCScope():
            leading_shape = (global_batch, seq_len)
            placement = NamedSharding(mesh, P(DATA_AXES, None, None))
    init = partial(
        init_persistent_sources, site_names, site_component_counts, leading_shape, source_dtype
    )
    return jax.jit(init, out_shardings=placement)(key)


def shard_batch(resid_global: jax.Array, mesh: Mesh) -> jax.Array:
    """Batch-shard the residual input (b, t, d) over `dp` (axis 0)."""
    return _generic_shard_batch(resid_global, mesh, batch_axis=0)
