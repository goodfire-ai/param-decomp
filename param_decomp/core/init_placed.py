"""Seeded init → placed arrays, with no host-side full tree.

Each helper computes the declared shardings on an `eqx.filter_eval_shape`'d abstract
value, then runs the seeded init under `jax.jit(init, out_shardings=...)` so each device
generates only its own shard — an eager `device_put` of a host tree onto a multi-process
non-replicated sharding triggers a `process_allgather` (a host allocation of the FULL
unsharded tree per process). A non-dividing declared shard axis is a loud crash at
placement construction / inside `.shardings` (fail-fast), never a silent replicate.
The final `reshard` preserves placement during shape evaluation; nested
`out_shardings` alone does not propagate it to the enclosing computation.

Compile-time doctrine: keep seeded inits FEW-OUTPUTS-under-jit — a jit returning n_sites
(hundreds of) sharded outputs, or n_chunks unrolled RNG bodies, is a multi-minute
SPMD/layout compile. vmap-stack over the same per-site/per-chunk keys (bit-identical
values), then fan out with a trivial slice jit. `init_component_stacks_placed` is the
template; `run_state.init_decomposition`'s CI fn init / `init_sources_sharded` follow it.
"""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import PRNGKeyArray

from param_decomp.core.adversary import (
    BlockedSourceComponents,
    SourceStack,
    SourceStacks,
    init_persistent_sources,
)
from param_decomp.core.axes import MeshAxis
from param_decomp.core.ci_fn.architecture import CIFnArchitecture
from param_decomp.core.ci_fn.interface import CIFn
from param_decomp.core.components import (
    BlockedFactorization,
    ComponentStacks,
    DenseFactorization,
    SiteSpec,
    init_component_stacks,
    pad_component_stacks,
    site_stack_indices_for,
    vu_groups,
)
from param_decomp.core.configs import (
    BatchSourceShape,
    FrequencyMinimalityConfig,
    MergedStochasticSubsetPooledPPGDReconLossConfig,
    PersistentPGDLossConfig,
    SourcePoolConfig,
)
from param_decomp.core.losses import (
    BatchFrequency,
    EmaFrequency,
    FrequencyEstimator,
    init_frequency_estimator,
)
from param_decomp.core.model import (
    ComponentActivations,
    DecomposedModel,
    PlacedModel,
    PositionAxis,
    Positioned,
    Positionless,
)
from param_decomp.core.placement import (
    PlacementRules,
    batch_axes,
    component_stacks_shardings,
)

type ComponentInitializer[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
] = Callable[
    [DecomposedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT], PRNGKeyArray],
    ComponentStacks,
]
"""A target-aware, unplaced V/U initializer. The placed wrapper below owns sharding."""

type CIFnInitializer[Conditioning] = Callable[[ComponentStacks, PRNGKeyArray], CIFn[Conditioning]]
"""A CI fn initializer given the decomposition's fresh components. `init_decomposition`
places its output. The arrays it reads must be its own pytree leaves (an `eqx.Module`),
never closure captures: the placed init's `eqx.filter_jit` traces its array leaves as
arguments and hashes the rest as static, while a captured array is baked into the
program as a constant."""


def seeded_ci_fn_initializer[Conditioning](
    arch: CIFnArchitecture[Conditioning], sites: tuple[SiteSpec, ...], rules: PlacementRules
) -> CIFnInitializer[Conditioning]:
    """The architecture's own seeded init, which reads no components."""
    return lambda _components, key: arch.initialize(sites, rules, key)


def random_component_initializer[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model: DecomposedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    key: PRNGKeyArray,
) -> ComponentStacks:
    """The domain-neutral random initializer used unless a composition root selects another."""
    return init_component_stacks(model.sites, key)


def _census_stack_pads(rules: PlacementRules) -> dict[str, int]:
    """The rules' resolved persist-stack pads, ready for `pad_component_stacks`."""
    return {
        group: entry.stack_pad
        for group, entry in rules.components.group_census.items()
        if entry.stack_pad
    }


def init_component_stacks_placed(
    sites: tuple[SiteSpec, ...], key: PRNGKeyArray, rules: PlacementRules
) -> ComponentStacks:
    """Seed random V/U directly into the component persistence layout, the census'
    persist-stack pads appended as all-zero matrices (the real stack draws identically
    to an unpadded init)."""
    pads = _census_stack_pads(rules)
    init = lambda k: pad_component_stacks(init_component_stacks(sites, k), pads)
    abstract = eqx.filter_eval_shape(init, key)
    placement = component_stacks_shardings(abstract, rules)
    return jax.reshard(jax.jit(init, out_shardings=placement)(key), placement)


def padded_component_initializer[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    rules: PlacementRules,
    initializer: ComponentInitializer[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
) -> ComponentInitializer[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT]:
    """`initializer` with the census' persist-stack pads appended as all-zero matrices — the
    tree the component persistence layout is declared over, so every consumer of the
    initializer's shape (the placed init, the pre-init placement audit) sees the pads."""
    pads = _census_stack_pads(rules)
    return lambda m, k: pad_component_stacks(initializer(m, k), pads)


def init_model_component_stacks_placed[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    key: PRNGKeyArray,
    rules: PlacementRules,
    initializer: ComponentInitializer[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
) -> ComponentStacks:
    """Run a target-aware initializer directly into the component persistence layout,
    the census' persist-stack pads appended as all-zero matrices.

    The frozen model stays a traced argument: an aligned initializer may read target weights.
    Initializers return semantic-group stacks, preserving the no-host-full-tree contract.
    """
    init = padded_component_initializer(rules, initializer)
    abstract = eqx.filter_eval_shape(init, model.model, key)
    placement = component_stacks_shardings(abstract, rules)
    return jax.reshard(jax.jit(init, out_shardings=placement)(model.model, key), placement)


@eqx.filter_jit
def init_frequency_estimator_placed(
    cfg: FrequencyMinimalityConfig | None,
    sites: tuple[SiteSpec, ...],
    sharding: NamedSharding,
) -> FrequencyEstimator:
    estimator = init_frequency_estimator(cfg, sites)
    match estimator:
        case BatchFrequency():
            return estimator
        case EmaFrequency():
            return eqx.tree_at(
                lambda f: f.estimate,
                estimator,
                jax.sharding.reshard(estimator.estimate, sharding),
            )


def _source_leading(
    positions: PositionAxis, source_shape: BatchSourceShape, global_batch: int, mesh: Mesh
) -> tuple[tuple[int, ...], tuple[tuple[MeshAxis, ...] | None, ...]]:
    """Each stored (positions x source_shape) leading shape, with its mesh spec: batch-B
    shapes batch-shard over the data axes, position axes replicate."""
    data_axes = batch_axes(mesh)
    match positions, source_shape:
        case Positionless(), "bc":
            return (global_batch,), (data_axes,)
        case Positionless(), "bsc":
            raise ValueError(
                f"source_shape {source_shape!r} names a position axis; target is positionless"
            )
        case Positioned(), "bc":
            return (global_batch, 1), (data_axes, None)
        case Positioned(n_positions=n), "bsc":
            return (global_batch, n), (data_axes, None)


def _source_stacks_shardings(
    sites: tuple[SiteSpec, ...],
    leading_spec: tuple[tuple[MeshAxis, ...] | None, ...],
    mesh: Mesh,
) -> SourceStacks[NamedSharding]:
    stacks: dict[str, SourceStack[NamedSharding]] = {}
    for group, members in vu_groups(sites).items():
        match members.factorization:
            case DenseFactorization():
                components: NamedSharding | BlockedSourceComponents = NamedSharding(
                    mesh, P(None, *leading_spec, "tp")
                )
            case BlockedFactorization():
                components = BlockedSourceComponents(
                    values=NamedSharding(mesh, P(None, *leading_spec, "tp", None))  # pyright: ignore[reportArgumentType]
                )
        stacks[group] = SourceStack(
            components=components, delta=NamedSharding(mesh, P(None, *leading_spec))
        )
    return SourceStacks(stacks=stacks, site_stack_indices=site_stack_indices_for(sites))


def persistent_sources_shardings(
    sites: tuple[SiteSpec, ...],
    positions: PositionAxis,
    source_shape: BatchSourceShape,
    global_batch: int,
    mesh: Mesh,
) -> SourceStacks[NamedSharding]:
    """The declared placement of one ordinary persistent adversary's source stacks.

    Batch-B shapes shard over the data axes; position axes replicate. The
    storage component axis uses TP, and source-delta values replicate over TP.
    This is a storage policy; source reads establish the consumer's compute layout.
    """
    _, leading_spec = _source_leading(positions, source_shape, global_batch, mesh)
    return _source_stacks_shardings(sites, leading_spec, mesh)


def source_pool_shardings(sites: tuple[SiteSpec, ...], mesh: Mesh) -> SourceStacks[NamedSharding]:
    """Shard minipools with their batch elements; retain ordinary component TP placement."""
    return _source_stacks_shardings(sites, (batch_axes(mesh), None), mesh)


def init_source_pool_sharded(
    sites: tuple[SiteSpec, ...],
    pool: SourcePoolConfig,
    global_batch: int,
    source_dtype: DTypeLike,
    key: PRNGKeyArray,
    mesh: Mesh,
) -> SourceStacks:
    """Initialize cross-site particles directly into their declared pool placement."""
    leading = (global_batch, pool.size_per_batch_element)
    shardings = source_pool_shardings(sites, mesh)
    initialized = jax.jit(
        partial(init_persistent_sources, sites, leading, source_dtype),
        out_shardings=shardings,
    )(key)
    return jax.reshard(initialized, shardings)


def persistent_sources_shardings_from_config(
    sites: tuple[SiteSpec, ...],
    positions: PositionAxis,
    cfg: PersistentPGDLossConfig | MergedStochasticSubsetPooledPPGDReconLossConfig,
    global_batch: int,
    mesh: Mesh,
) -> SourceStacks[NamedSharding]:
    """The runtime placement declared by one persistent-source config."""
    match cfg:
        case MergedStochasticSubsetPooledPPGDReconLossConfig():
            return source_pool_shardings(sites, mesh)
        case PersistentPGDLossConfig(source_shape=source_shape):
            return persistent_sources_shardings(sites, positions, source_shape, global_batch, mesh)


def init_persistent_sources_from_config(
    sites: tuple[SiteSpec, ...],
    positions: PositionAxis,
    cfg: PersistentPGDLossConfig | MergedStochasticSubsetPooledPPGDReconLossConfig,
    global_batch: int,
    key: PRNGKeyArray,
    mesh: Mesh,
) -> SourceStacks:
    """Initialize one persistent-source config directly into its runtime placement."""
    match cfg:
        case MergedStochasticSubsetPooledPPGDReconLossConfig(pool=pool):
            return init_source_pool_sharded(sites, pool, global_batch, cfg.source_dtype, key, mesh)
        case PersistentPGDLossConfig(source_shape=source_shape):
            return init_sources_sharded(
                sites,
                positions,
                source_shape,
                global_batch,
                cfg.source_dtype,
                key,
                mesh,
            )


def init_sources_sharded(
    sites: tuple[SiteSpec, ...],
    positions: PositionAxis,
    source_shape: BatchSourceShape,
    global_batch: int,
    source_dtype: DTypeLike,
    key: PRNGKeyArray,
    mesh: Mesh,
) -> SourceStacks:
    """Seeded PPGD-source init placed onto `persistent_sources_shardings` (jit +
    `out_shardings`; same no-host-tree rationale as `init_component_stacks_placed`, and
    the same few-outputs doctrine — one sharded output per semantic group). Every stored
    (positions x source_shape) leading shape is enumerated in `_source_leading`; the rank
    always matches the waist, with size-1 broadcast axes for the letters `source_shape`
    omits (`configs.BatchSourceShape`).

    Sources shard over the data-parallel axes (`placement.batch_axes`),
    aligning each batch element's source with that element's `shard_batch`-placed
    residual/CI. The source is independent per element, so the per-element grad is
    already shard-local — NO cross-rank reduction, matching torch's `_skip_all_reduce`.
    (Requires `global_batch % n_dev == 0`, the same divisibility `shard_batch` needs.)"""
    leading_shape, _ = _source_leading(positions, source_shape, global_batch, mesh)
    shardings = persistent_sources_shardings(sites, positions, source_shape, global_batch, mesh)
    initialized = jax.jit(
        partial(init_persistent_sources, sites, leading_shape, source_dtype),
        out_shardings=shardings,
    )(key)
    return jax.reshard(initialized, shardings)
