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
from jaxtyping import PRNGKeyArray

from param_decomp.core.adversary import (
    SiteSource,
    Sources,
    init_persistent_sources_stacked,
    sources_c_groups,
    unstack_persistent_sources,
)
from param_decomp.core.axes import MeshAxis
from param_decomp.core.ci_fn import (
    ChunkwiseTransformerCIFn,
    CIFn,
    CIFnArch,
    GlobalMLPCIFn,
    LayerwiseMLPCIFn,
    build_ci_fn,
)
from param_decomp.core.components import (
    ComponentStacks,
    SiteSpec,
    init_component_stacks,
)
from param_decomp.core.configs import SourceShape
from param_decomp.core.model import BATCH_AXES, PositionAxis, Positioned, Positionless
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


def ci_fn_shardings(abstract: CIFn, mesh: Mesh, rules: PlacementRules) -> CIFn:
    """The CI fn's own declared placement, dispatched by arch: the chunkwise transformer
    from the placement table's `ci_fn` rows; the toy MLPs shard each weight's output
    axis. THE single source of truth for the CI fn's persist layout — seeded init places
    onto it, and AOT consumers (the fit check) type their abstract state with it."""
    match abstract:
        case ChunkwiseTransformerCIFn():
            return abstract.shardings(mesh, rules.ci_fn)
        case LayerwiseMLPCIFn() | GlobalMLPCIFn():
            return abstract.shardings(mesh)
        case _:
            raise AssertionError(f"unknown CI fn {type(abstract)}")


def init_ci_fn_placed(
    arch: CIFnArch,
    sites: tuple[SiteSpec, ...],
    key: PRNGKeyArray,
    mesh: Mesh,
    rules: PlacementRules,
) -> CIFn:
    """Seeded CI-fn init (any arch, via `build_ci_fn`) placed by `ci_fn_shardings`.
    Shardings computed on the abstract model, init under jit."""
    init = partial(build_ci_fn, arch, sites)
    abstract = eqx.filter_eval_shape(init, key)
    return jax.jit(init, out_shardings=ci_fn_shardings(abstract, mesh, rules))(key)


def _source_leading(
    positions: PositionAxis, source_shape: SourceShape, global_batch: int
) -> tuple[tuple[int, ...], tuple[tuple[MeshAxis, ...] | None, ...]]:
    """Each stored (positions x source_shape) leading shape, with its mesh spec: batch-B
    shapes batch-shard over the data axes, batch-1 and position axes replicate."""
    match positions, source_shape:
        case Positionless(), "c":
            return (1,), (None,)
        case Positionless(), "bc":
            return (global_batch,), (BATCH_AXES,)
        case Positionless(), "sc" | "bsc":
            raise ValueError(
                f"source_shape {source_shape!r} names a position axis; target is positionless"
            )
        case Positioned(), "c":
            return (1, 1), (None, None)
        case Positioned(), "bc":
            return (global_batch, 1), (BATCH_AXES, None)
        case Positioned(n_positions=n), "sc":
            return (1, n), (None, None)
        case Positioned(n_positions=n), "bsc":
            return (global_batch, n), (BATCH_AXES, None)


def persistent_sources_shardings(
    site_names: tuple[str, ...],
    positions: PositionAxis,
    source_shape: SourceShape,
    global_batch: int,
    mesh: Mesh,
) -> dict[str, SiteSource[NamedSharding]]:
    """The per-site declared placement of one persistent adversary's sources — component
    sources follow the CI's TP-sharded C axis, scalar delta sources are TP-replicated,
    batch-B shapes batch-shard. THE single source of truth for the sources' persist
    layout — `init_sources_sharded` places onto it, and AOT consumers (the fit check)
    type their abstract state with it."""
    _, leading_spec = _source_leading(positions, source_shape, global_batch)
    component_sharding = NamedSharding(mesh, P(*leading_spec, "tp"))
    delta_sharding = NamedSharding(mesh, P(*leading_spec))
    return {
        name: SiteSource(components=component_sharding, delta=delta_sharding) for name in site_names
    }


def init_sources_sharded(
    site_names: tuple[str, ...],
    site_component_counts: tuple[int, ...],
    positions: PositionAxis,
    source_shape: SourceShape,
    global_batch: int,
    source_dtype: DTypeLike,
    key: PRNGKeyArray,
    mesh: Mesh,
) -> Sources:
    """Seeded PPGD-source init -> placed per `source_shape` (jit + `out_shardings`; same
    no-host-tree rationale as `init_component_stacks_placed`). Every stored (positions x
    source_shape) leading shape is written out in the match below; the rank always
    matches the waist, with size-1 broadcast axes for the letters `source_shape` omits
    (`configs.SourceShape`).

    Batch-1 shapes (`c`, `sc`) share one source across the global batch. Component
    sources follow the CI's TP-sharded C axis; scalar delta sources are TP-replicated.

    Batch-B shapes (`bc`, `bsc`) are BATCH-SHARDED over the data-parallel axes
    (`('replicate', 'fsdp')`, axis 0), aligning each batch element's source with that
    element's `shard_batch`-placed residual/CI. The source is independent per element, so
    the per-element grad is already shard-local — NO cross-rank reduction, matching
    torch's `_skip_all_reduce`. (Requires `global_batch % n_dev == 0`, the same
    divisibility `shard_batch` needs.)

    The structured representation makes those independent placements explicit; no
    runtime slice has to redistribute a monolithic C+1 value."""
    leading_shape, leading_spec = _source_leading(positions, source_shape, global_batch)
    # Two cheap compiles instead of one n_sites-sharded-output graph (same shape as
    # `init_component_stacks_placed`, incl. the transient: the stacked copy can't donate into
    # split outputs, so init briefly holds one extra shard-local copy of the sources).
    stacked_shardings = {
        c: SiteSource(
            components=NamedSharding(mesh, P(None, *leading_spec, "tp")),
            delta=NamedSharding(mesh, P(None, *leading_spec)),
        )
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
    out_shardings = persistent_sources_shardings(
        site_names, positions, source_shape, global_batch, mesh
    )
    return jax.jit(
        partial(unstack_persistent_sources, site_names, site_component_counts),
        out_shardings=out_shardings,
    )(stacked)
