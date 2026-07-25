"""GSPMD sharding plan for the GLU-transformer (Llama-8B / Qwen3-8B) single-pool step — the pure-HSDP memory story.

The 2-D `(replicate, fsdp)` mesh: `fsdp` is the 8 intra-node NVLink GPUs (the FSDP
weight-gather / grad-reduce axis), `replicate` the across-node axis. There is NO TP /
Megatron-C. The memory consumers, and how each is placed:

  * frozen target: FSDP-sharded on `fsdp` (the `d`-dim of every per-layer weight); the
    ~16 GB bulk shards `/fsdp` (8), gathered per layer in the scan on NVLink. embed /
    lm_head / norm / inv_freq replicate.
  * components (V/U) + their Adam states: placed by the run's `PlacementRules`
    (`placement.component_stacks_shardings` — `runtime.sharding`). Under the `owner`
    preset: the shape-group STACK axis ÷`replicate` (whole matrices owned per node-group),
    d dims ÷`fsdp`, C ÷`tp` — ÷N total, same memory as the intra-matrix `zero1`
    preset. The fp32 masters + fp32 Adam m/v are the dominant
    non-activation footprint (÷N, so the per-GPU cost shrinks with device count).
    COMPUTE re-pins the bf16
    weights to `fsdp`-only ONCE per step (in ENTRY, off the per-layer hot path; see
    `glu_transformer._reconstruct_compute_weights` — hand-written until the staged placement
    migration wires it through the `params.forward` row).
  * CI fn + Adam states: sharded ÷N over the full mesh along d_model (in_proj / blocks /
    heads), same ZeRO-1 reconstruction to `fsdp`-only before the chunk scan.
  * PGD source (`shared` scope, `{site: (1,P,C+1)}`): REPLICATED. A single adversarial
    source shared across the global batch; it combines elementwise with the batch-sharded
    CI and its grad reduction falls out of the global-mean loss (torch
    `reduce_source_grads` analog). Tiny vs activations, so replicating costs nothing;
    the C+1 axis is odd and cannot tile the mesh anyway. `SrcAdamState` mirrors it.
  * token input + all activations: BATCH-sharded over the FULL mesh
    (`('replicate', 'fsdp')`). The masked re-forwards then run on per-rank sub-batches ->
    activation memory scales 1/N. This is what unlocks a global batch that OOMs replicated.

Sharding V/U keeps every einsum valid: the compute weights are reconstructed `fsdp`-only
before the layer scan, so `x @ V` gathers the `fsdp`-sharded d_in on NVLink and contracts it;
`(.) @ U` produces a `fsdp`-sharded d_out and `jax.jit` inserts the reduce-scatter /
all-reduce. No manual collectives.

Placement is declared, never inferred: the frozen target and the CI fn declare per-leaf
`NamedSharding`s via `.shardings(mesh)` methods; the trainable V/U are placed by the run's
`PlacementRules` (`placement.component_stacks_shardings` — the per-group row assignment
resolved at config build). The helpers below only drive the apply: compute the
shardings on the `eqx.filter_eval_shape`'d abstract model, then run the seeded init under
`jax.jit(init, out_shardings=...)` so each device generates only its own shard and no
host-side full tree exists — eager `device_put` of a host tree onto a multi-process
non-replicated sharding triggers a `process_allgather` (a host allocation of the FULL
unsharded tree per process). A non-dividing declared shard axis is a loud crash at
placement construction / inside `.shardings` (fail-fast), never a silent replicate.
"""

from functools import partial

import equinox as eqx
import jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, PRNGKeyArray

from param_decomp.adversary import (
    init_persistent_sources_stacked,
    sources_c_groups,
    unstack_persistent_sources,
)
from param_decomp.ci_fn import CIFn, CIFnArch, build_ci_fn
from param_decomp.components import (
    ComponentStacks,
    SiteSpec,
    WeightInit,
    init_component_stacks,
)
from param_decomp.configs import SourceShape
from param_decomp.model import PositionAxis, Positioned, Positionless
from param_decomp.placement import PlacementRules, component_stacks_shardings
from param_decomp.sharding import hsdp_mesh, place_via_shardings
from param_decomp.sharding import shard_batch as _generic_shard_batch
from param_decomp.targets.glu_transformer import GLUDecomposedModel

__all__ = [
    "hsdp_mesh",
    "place_target",
    "init_component_stacks_placed",
    "init_ci_fn_placed",
    "init_sources_sharded",
    "shard_batch",
]


def place_target(tgt: GLUDecomposedModel, mesh: Mesh) -> GLUDecomposedModel:
    """Eager `device_put` of the already-loaded frozen target onto its own declared
    placement (`tgt.shardings(mesh)` — FSDP-on-`fsdp`)."""
    return place_via_shardings(tgt, tgt.shardings(mesh))


def init_component_stacks_placed(
    sites: tuple[SiteSpec, ...],
    key: PRNGKeyArray,
    rules: PlacementRules,
    weight_init: WeightInit = "kaiming",
    target_weights: dict[str, Array] | None = None,
) -> ComponentStacks:
    """`init_component_stacks` placed by `component_stacks_shardings(_, rules)` (the run's
    placement policy); kaiming values bit-identical to the retired per-site init (pinned
    by `test_sharding`). One jit, 2×n_shapes sharded outputs — the persistence layout IS
    the stacked layout, so no host-side full tree ever exists; `key`/`target_weights` ride
    as jit ARGUMENTS (not closure constants) so the frozen W arrays are not baked into the
    compiled init."""
    init = partial(init_component_stacks, sites, weight_init=weight_init)
    placement = component_stacks_shardings(
        eqx.filter_eval_shape(init, key, target_weights=target_weights), rules
    )
    return jax.jit(init, out_shardings=placement)(key, target_weights=target_weights)


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
    positions: PositionAxis,
    source_shape: SourceShape,
    global_batch: int,
    source_dtype: DTypeLike,
    key: PRNGKeyArray,
    mesh: Mesh,
) -> dict[str, Array]:
    """Seeded PPGD-source init -> placed per `source_shape` (jit + `out_shardings`; same
    no-host-tree rationale as `init_component_stacks_placed`). Every stored (positions x
    source_shape) leading shape is written out in the match below; the rank always
    matches the waist, with size-1 broadcast axes for the letters `source_shape` omits
    (`configs.SourceShape`).

    Batch-1 shapes (`c`, `sc`) are REPLICATED: one source (per position for `sc`) shared
    across the whole global batch; it combines elementwise with the batch-sharded CI
    (`mask = ci + (1-ci)*source[..., :-1]`) and its grad is AVG-reduced across shards
    (torch `reduce_source_grads`).

    Batch-B shapes (`bc`, `bsc`) are BATCH-SHARDED over the FULL mesh
    (`('replicate', 'fsdp')`, axis 0), aligning each batch element's source with that
    element's `shard_batch`-placed residual/CI. The source is independent per element, so
    the per-element grad is already shard-local — NO cross-rank reduction, matching
    torch's `_skip_all_reduce`. (Requires `global_batch % n_dev == 0`, the same
    divisibility `shard_batch` needs.)

    Sharding the trailing C+1 axis is invalid for every shape: with the weight-delta
    channel C+1 is odd and not divisible by the mesh size, and would also fight the
    batch-sharded elementwise combine."""
    match positions, source_shape:
        case Positionless(), "c":
            leading_shape, spec = (1,), P()
        case Positionless(), "bc":
            leading_shape, spec = (global_batch,), P(("replicate", "fsdp"), None)
        case Positionless(), "sc" | "bsc":
            raise ValueError(
                f"source_shape {source_shape!r} names a position axis; target is positionless"
            )
        case Positioned(), "c":
            leading_shape, spec = (1, 1), P()
        case Positioned(), "bc":
            leading_shape, spec = (global_batch, 1), P(("replicate", "fsdp"), None, None)
        case Positioned(n_positions=n), "sc":
            leading_shape, spec = (1, n), P()
        case Positioned(n_positions=n), "bsc":
            leading_shape = (global_batch, n)
            spec = P(("replicate", "fsdp"), None, None)
    # Two cheap compiles instead of one n_sites-sharded-output graph (same shape as
    # `init_component_stacks_placed`, incl. the transient: the stacked copy can't donate into
    # split outputs, so init briefly holds one extra shard-local copy of the sources).
    stacked_shardings = {
        c: NamedSharding(mesh, P(None, *spec))
        for c in sources_c_groups(site_names, site_component_counts)
    }
    stacked = jax.jit(
        partial(
            init_persistent_sources_stacked,
            site_names,
            site_component_counts,
            leading_shape,
            source_dtype,
        ),
        out_shardings=stacked_shardings,
    )(key)
    return jax.jit(
        partial(unstack_persistent_sources, site_names, site_component_counts),
        out_shardings=NamedSharding(mesh, spec),
    )(stacked)


def shard_batch(resid_global: jax.Array, mesh: Mesh) -> jax.Array:
    """Batch-shard the residual input (b, t, d) over the full mesh (axis 0)."""
    return _generic_shard_batch(resid_global, mesh, batch_axis=0)
