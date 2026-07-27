"""Seeded init → placed arrays, with no host-side full tree.

Each helper computes the declared shardings on an `eqx.filter_eval_shape`'d abstract
value, then runs the seeded init under `jax.jit(init, out_shardings=...)` so each device
generates only its own shard — an eager `device_put` of a host tree onto a multi-process
non-replicated sharding triggers a `process_allgather` (a host allocation of the FULL
unsharded tree per process). A non-dividing declared shard axis is a loud crash at
placement construction / inside `.shardings` (fail-fast), never a silent replicate.

Compile-time doctrine: keep seeded inits FEW-OUTPUTS-under-jit — a jit returning n_sites
(hundreds of) sharded outputs, or n_chunks unrolled RNG bodies, is a multi-minute
SPMD/layout compile. vmap-stack over the same per-site/per-chunk keys (bit-identical
values), then fan out with a trivial slice jit. `init_component_stacks_placed` is the
template; `init_ci_fn_placed` / `init_sources_sharded` follow it.
"""

from functools import partial

import equinox as eqx
import jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.adversary import (
    init_persistent_sources_stacked,
    sources_c_groups,
    unstack_persistent_sources,
)
from param_decomp.core.ci_fn import CIFn, CIFnArch, build_ci_fn
from param_decomp.core.components import (
    ComponentStacks,
    SiteSpec,
    init_component_stacks,
)
from param_decomp.core.configs import SourceShape
from param_decomp.core.model import PositionAxis, Positioned, Positionless
from param_decomp.core.placement import PlacementRules, component_stacks_shardings


def init_component_stacks_placed(
    sites: tuple[SiteSpec, ...], key: PRNGKeyArray, rules: PlacementRules
) -> ComponentStacks:
    """Seeded V/U init placed by `component_stacks_shardings(_, rules)` (the run's placement
    policy), values bit-identical to the retired per-site init (pinned by `test_sharding`).
    One jit, 2×n_shapes sharded outputs — the persistence layout IS the stacked layout, so
    the old two-stage stack-then-unstack fan-out (and its transient extra copy) is gone."""
    abstract = eqx.filter_eval_shape(partial(init_component_stacks, sites), key)
    placement = component_stacks_shardings(abstract, rules)
    return jax.jit(partial(init_component_stacks, sites), out_shardings=placement)(key)


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
