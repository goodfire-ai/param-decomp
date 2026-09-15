"""Mask materialization shared by reconstruction objectives and model targets.

Every builder is pointwise on the CI values, so a narrow site's mask inherits the
`NarrowCI` bundle: the mask values ride `values` and the SAME router indices travel with
them — the seam contract to the masked forward lives in the type, never in a side-channel
convention. Persistent/fresh sources are stored FULL-C; a narrow site reads its routed
entries out of the source by the bundle's indices (`_narrow_source_values`) — a read of
the routed rows, never a scatter to `[.., C]`."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float, PRNGKeyArray

from param_decomp.core.adversary import (
    ExpertBlockedSource,
    SiteSource,
    Sources,
    SourceStacks,
    full_source_components,
)
from param_decomp.core.components import (
    NarrowCI,
    SiteCI,
    SiteSpec,
    map_site_ci,
    require_full_emission,
    site_ci_values,
)
from param_decomp.core.linear_plan import uniform_like, value_mesh
from param_decomp.core.model import Masking, MaterializedMasking, SourceMasking, StochasticMasking


def _narrow_source_values(source: ExpertBlockedSource, ci: NarrowCI) -> Array:
    """One narrow site's routed slice of a block-dim source, in the CI's dtype:
    `out[.., m, :] = source.values[.., ids[.., m], :]`. The cast happens on the table
    BEFORE the select — pointwise, so it commutes with the select — and fuses with the
    source's dequant into the selecting op, which then also fixes the transpose's
    dtype: routed cotangents land on the stored source's shape at the CI dtype (the
    velocity's), never widened to the table's fp32 view.

    Two spellings, by the source's lead (`SourceShape`): a source spanning the CI's
    full lead (`bsc`) is a take along the expert axis — exact, its transpose a
    scatter of each routed cotangent onto its own entry; a source with size-1
    broadcast lead axes is a one-hot contraction over the expert axis, which
    broadcasts through the einsum's typed rule where the explicit sharding rule
    cannot resolve a take of a size-1 table axis against a batch-sharded index axis."""
    table = source.values.astype(ci.values.dtype)
    lead = ci.values.shape[:-1]
    assert table.shape[-2:] == (ci.n_experts, ci.c_per_expert), (table.shape, ci.C)
    assert all(extent in (1, full) for extent, full in zip(table.shape[:-2], lead, strict=True)), (
        table.shape,
        lead,
    )
    if table.shape[:-2] == lead:
        gathered = _take_routed_rows(table, ci)
    else:
        gathered = _contract_routed_rows(table, ci)
    return gathered.reshape(*lead, ci.values.shape[-1])


def _take_routed_rows(table: Float[Array, "*lead E c"], ci: NarrowCI) -> Float[Array, "*lead k c"]:
    """`table[.., ids[.., m], :]` for a table spanning the CI's lead. On a mesh the
    expert axis is TP-sharded, so the take is shard-local: the expert axis viewed
    `(shard, local expert)`, each shard taking its own experts' rows by shard-local
    index (a foreign slot reads an arbitrary in-range row and is zeroed), then the
    typed sum over shards — the sharding rule's one all-reduce, exact since every slot
    is live on exactly one shard."""
    ids = ci.router_indices
    mesh = value_mesh(ids)
    if mesh.empty:
        return jnp.take_along_axis(table, ids[..., None], axis=-2)
    table_spec = jax.typeof(table).sharding.spec
    lead_spec = jax.typeof(ids).sharding.spec[:-1]
    expert_axis = table_spec[-2]
    assert expert_axis is None or isinstance(expert_axis, str), table_spec
    n_shards = 1 if expert_axis is None else mesh.shape[expert_axis]
    e_local = ci.n_experts // n_shards
    view = jnp.reshape(
        table,
        (*table.shape[:-2], n_shards, e_local, ci.c_per_expert),
        out_sharding=NamedSharding(mesh, P(*table_spec[:-2], expert_axis, None, None)),
    )
    local_ids = jax.sharding.reshard(
        ids[..., None, :] - (jnp.arange(n_shards) * e_local)[:, None],
        NamedSharding(mesh, P(*lead_spec, expert_axis, None)),
    )
    live = (local_ids >= 0) & (local_ids < e_local)
    rows = jnp.take_along_axis(view, jnp.clip(local_ids, 0, e_local - 1)[..., None], axis=-2)
    return jnp.einsum(
        "...skc->...kc",
        jnp.where(live[..., None], rows, 0.0),
        out_sharding=NamedSharding(mesh, P(*lead_spec, None, None)),
    )


def _contract_routed_rows(
    table: Float[Array, "*lead E c"], ci: NarrowCI
) -> Float[Array, "*lead k c"]:
    """`table[.., ids[.., m], :]` as a one-hot contraction over the expert axis, for a
    table whose size-1 lead axes broadcast against the CI's; exact for 0/1 weights."""
    one_hot = jax.nn.one_hot(ci.router_indices, ci.n_experts, dtype=table.dtype)
    mesh = value_mesh(ci.router_indices)
    if mesh.empty:
        return jnp.einsum("...ke,...ec->...kc", one_hot, table)
    ids_spec = jax.typeof(ci.router_indices).sharding.spec
    return jnp.einsum(
        "...ke,...ec->...kc",
        one_hot,
        table,
        out_sharding=NamedSharding(mesh, P(*ids_spec, None)),
    )


def all_live_masking_no_delta(
    sites: tuple[SiteSpec, ...], *, leading_shape: tuple[int, ...], dtype: DTypeLike
) -> MaterializedMasking:
    """Turn every component on while disabling frozen-weight delta corrections."""
    return MaterializedMasking(
        component_masks={site.name: jnp.ones((*leading_shape, site.C), dtype) for site in sites}
    )


def _sample_stochastic_masks(
    ci_lower: Mapping[str, SiteCI], draw_key: Array
) -> tuple[dict[str, SiteCI], dict[str, Array]]:
    """Draw fresh component and weight-delta masks for every site.

    The per-site fold index follows `ci_lower`'s insertion order — the CI fn's canonical
    output order, stable across traces. Narrow sites draw at the narrow shape (their
    unrouted components are structurally absent, so no draw exists for them)."""
    mask_key, delta_key = random.split(draw_key)
    masks: dict[str, SiteCI] = {}
    delta_masks: dict[str, Array] = {}
    for site_idx, (site, ci) in enumerate(ci_lower.items()):
        site_key = random.fold_in(mask_key, site_idx)
        masks[site] = map_site_ci(lambda v, k=site_key: v + (1.0 - v) * uniform_like(k, v), ci)
        delta_masks[site] = uniform_like(
            random.fold_in(delta_key, site_idx), site_ci_values(ci), drop_last_axis=True
        )
    return masks, delta_masks


def materialize_masking(masking: Masking) -> MaterializedMasking:
    """Return concrete mask arrays for a target that cannot sample inside its blocks.

    Materialized inputs — including adversarial masks — pass through unchanged. Stochastic
    inputs are sampled eagerly here. Such targets implement ``stack_ci`` as identity, so
    ``ci_stacked`` remains the ordinary per-site ``ci_lower`` dictionary; the scan target
    consumes its stacked recipe directly and never calls this helper.
    """
    match masking:
        case MaterializedMasking():
            return masking
        case StochasticMasking(ci_stacked=ci_lower, draw_key=draw_key, routes=routes):
            assert isinstance(ci_lower, dict), (
                f"materialize_masking requires identity stack_ci; got {type(ci_lower).__name__}"
            )
            component_masks, weight_delta_masks = _sample_stochastic_masks(ci_lower, draw_key)
            return MaterializedMasking(
                component_masks=component_masks,
                weight_delta_masks=weight_delta_masks,
                routes=routes,
            )
        case SourceMasking(
            ci_stacked=ci_lower,
            source_values_stacked=source_values,
            delta_values_stacked=delta_values,
            routes=routes,
        ):
            assert isinstance(ci_lower, dict) and isinstance(source_values, dict), (
                f"materialize_masking requires identity stack_ci; got "
                f"{type(ci_lower).__name__} / {type(source_values).__name__}"
            )
            return MaterializedMasking(
                component_masks={
                    site: compose_source_mask(ci, source_values[site])
                    for site, ci in ci_lower.items()
                },
                weight_delta_masks=delta_values,
                routes=routes,
            )


def stochastic_delta_pinned_masks(
    ci_lower: Mapping[str, SiteCI], draw_key: Array
) -> tuple[dict[str, SiteCI], dict[str, Array]]:
    """Stochastic component masks with every weight-delta mask pinned to 1.0 — the tPD
    non-target pass, where `components + Δ` must reconstruct the frozen output.

    Pre-built (`MaterializedMasking`) rather than the in-target `StochasticMasking`
    rebuild, which draws its own `U[0,1]` delta inside each block and cannot pin it. The
    key split mirrors `_sample_stochastic_masks` (source half used, delta half discarded —
    the delta is deterministic here), as does the fold order (`ci_lower` insertion order,
    the CI fn's canonical output order)."""
    mask_key, _ = random.split(draw_key)
    masks: dict[str, SiteCI] = {}
    delta_masks: dict[str, Array] = {}
    for site_idx, (site, ci) in enumerate(ci_lower.items()):
        site_key = random.fold_in(mask_key, site_idx)
        masks[site] = map_site_ci(lambda v, k=site_key: v + (1.0 - v) * uniform_like(k, v), ci)
        values = site_ci_values(ci)
        delta_masks[site] = jnp.ones(values.shape[:-1], values.dtype)
    return masks, delta_masks


def constant_delta_pinned_masks(
    value: float, ci_lower: Mapping[str, SiteCI]
) -> tuple[dict[str, SiteCI], dict[str, Array]]:
    """Constant component masks (`ci + (1-ci)·value`) with every weight-delta mask pinned
    to 1.0 — the tPD non-target pass's constant-source arm. The plain objective's
    constant arm carries NO delta path at all; here the delta must be fully on."""
    masks = {
        site: map_site_ci(lambda v: v + (1.0 - v) * value, ci) for site, ci in ci_lower.items()
    }
    delta_masks = {
        site: jnp.ones(site_ci_values(ci).shape[:-1], site_ci_values(ci).dtype)
        for site, ci in ci_lower.items()
    }
    return masks, delta_masks


def unmasked_no_delta_masks(
    ci_lower: Mapping[str, SiteCI],
) -> tuple[dict[str, SiteCI], dict[str, Array]]:
    """Every component mask `1.0` with every weight-delta mask pinned to `0.0` — the tPD
    non-target pass's one delta-OFF arm (the enumerated exception): the FULL
    component sum alone must reconstruct the frozen output, so components that never
    activate cannot hide behind the delta. Deterministic — no sources are drawn;
    `ci_lower` supplies only shapes and dtypes."""
    masks = {site: map_site_ci(jnp.ones_like, ci) for site, ci in ci_lower.items()}
    delta_masks = {
        site: jnp.zeros(site_ci_values(ci).shape[:-1], site_ci_values(ci).dtype)
        for site, ci in ci_lower.items()
    }
    return masks, delta_masks


def sample_source_pool(
    key: PRNGKeyArray,
    ci_lower: Mapping[str, SiteCI],
    source_pool: SourceStacks,
) -> Sources:
    """Sample one cross-site particle per batch element and broadcast it over positions.

    The same row index is used at every site, so row ``i`` is one jointly trained attack
    across the model. Gathering keeps the pool as the graph leaf; its transpose
    scatter-adds each document's source gradient onto the selected row.
    """
    reference = site_ci_values(next(iter(ci_lower.values())))
    per_site = source_pool.per_site()
    sizes = {source.delta.shape for source in per_site.values()}
    assert len(sizes) == 1, f"pool rows must align across sites, got delta shapes {sizes}"
    ((n,),) = sizes

    index_shape = (reference.shape[0], *(1 for _ in reference.shape[1:-1]))
    mesh = value_mesh(reference)
    if mesh.empty:
        idx = random.randint(key, index_shape, 0, n, dtype=jnp.int32)
    else:
        reference_spec = jax.typeof(reference).sharding.spec.partitions
        index_spec = P(reference_spec[0], *(None for _ in reference_spec[1:-1]))
        idx = random.randint(
            key,
            index_shape,
            0,
            n,
            dtype=jnp.int32,
            out_sharding=NamedSharding(mesh, index_spec),
        )

    def gather(table: Array) -> Array:
        assert table.shape[0] == n, (table.shape, n)
        if mesh.empty:
            return table[idx]
        table_spec = jax.typeof(table).sharding.spec
        idx_spec = jax.typeof(idx).sharding.spec
        assert table_spec[0] is None, table_spec
        return table.at[idx].get(out_sharding=NamedSharding(mesh, P(*idx_spec, *table_spec[1:])))

    sampled: Sources = {}
    for site, source in per_site.items():
        assert all(jnp.issubdtype(leaf.dtype, jnp.floating) for leaf in jax.tree.leaves(source)), (
            f"site {site!r}: source-pool sampling reads the float source view"
        )
        match source.components:
            case ExpertBlockedSource(values=values):
                components = ExpertBlockedSource(values=gather(values))
            case jax.Array():
                components = gather(source.components)
        sampled[site] = SiteSource(components=components, delta=gather(source.delta))
    return sampled


def source_value_cis(
    ci_lower: Mapping[str, SiteCI], sources: Mapping[str, SiteSource]
) -> tuple[dict[str, SiteCI], dict[str, Array]]:
    """Per-site source VALUES in the CI's own emission geometry, plus the delta values —
    mask ingredients, not yet composed (`compose_source_mask` is the compose).

    Sources broadcast over the leading dimensions left singleton by their source shape.
    Casting the fp32 source state to the CI dtype here matches torch under autocast while
    preserving the source gradient through the cast. A narrow site's CI meets its
    BLOCK-DIM source (the typed agreement fails closed: an expert-blocked site stores
    `ExpertBlockedSource`) and reads its routed entries by the bundle's indices; the
    source's unrouted entries are inert this batch and carry zero gradient — exactly
    the full-width semantics, since an unrouted component's mask cannot affect any
    reconstruction. Full-emission sites read the flat expert-major view
    (`full_source_components`, bit-identical bytes).
    """
    assert set(sources) == set(ci_lower), (sources.keys(), ci_lower.keys())
    values: dict[str, SiteCI] = {}
    deltas: dict[str, Array] = {}
    for site, ci in ci_lower.items():
        source = sources[site]
        assert all(jnp.issubdtype(leaf.dtype, jnp.floating) for leaf in jax.tree.leaves(source)), (
            f"site {site!r}: masks read the FLOAT source view "
            "(source_values_to_float), never the storage representation"
        )
        match ci:
            case NarrowCI():
                assert isinstance(source.components, ExpertBlockedSource), (
                    f"narrow site {site!r} needs a block-dim source, "
                    f"got {type(source.components).__name__}"
                )
                values[site] = NarrowCI(
                    values=_narrow_source_values(source.components, ci),
                    router_indices=ci.router_indices,
                    n_experts=ci.n_experts,
                )
                deltas[site] = source.delta.astype(ci.values.dtype)
            case jax.Array():
                components = full_source_components(source.components)
                values[site] = components.astype(ci.dtype)
                deltas[site] = source.delta.astype(ci.dtype)
    return values, deltas


def compose_source_mask(ci: SiteCI, source_values: SiteCI) -> SiteCI:
    """One site's mask from its CI and source values: `ci + (1 - ci)·source`,
    pointwise at whatever leading layout the pair rides — per-site, target-stacked, or a
    stage slice inside a checkpointed block. The two emissions must agree; a narrow
    pair composes on `values` with the CI's indices carried through."""
    match ci, source_values:
        case NarrowCI(), NarrowCI():
            assert ci.values.shape == source_values.values.shape, (
                ci.values.shape,
                source_values.values.shape,
            )
            return NarrowCI(
                values=ci.values + (1.0 - ci.values) * source_values.values,
                router_indices=ci.router_indices,
                n_experts=ci.n_experts,
            )
        case jax.Array(), jax.Array():
            # dense source values may carry size-1 broadcast lead axes (SourceShape)
            return ci + (1.0 - ci) * source_values
        case (NarrowCI(), jax.Array()) | (jax.Array(), NarrowCI()):
            raise AssertionError(
                f"mixed mask emission: {type(ci).__name__} CI with "
                f"{type(source_values).__name__} source values"
            )


def masks_from_sources(
    ci_lower: Mapping[str, SiteCI], sources: Mapping[str, SiteSource]
) -> tuple[dict[str, SiteCI], dict[str, Array]]:
    """Build component and weight-delta masks from per-site sources — the
    eager spelling of the `SourceMasking` recipe, op-for-op the same composition."""
    values, delta_masks = source_value_cis(ci_lower, sources)
    masks = {site: compose_source_mask(ci, values[site]) for site, ci in ci_lower.items()}
    return masks, delta_masks


def _per_sample_adversarial_assignment(
    key: PRNGKeyArray, adv_fraction: Array, leading: tuple[int, ...]
) -> Array:
    """Draw one Bernoulli selector per sample, broadcast across position axes."""
    one_flag_per_sample = (leading[0], *(1,) * (len(leading) - 1))
    return random.bernoulli(key, adv_fraction, one_flag_per_sample)


def mixed_persistent_stochastic_masks(
    key: PRNGKeyArray,
    ci_lower: Mapping[str, SiteCI],
    persistent_sources: Sources,
    leading: tuple[int, ...],
    adv_fraction: Array,
    stochastic_routes: dict[str, Array] | None,
) -> tuple[dict[str, Array], dict[str, Array], dict[str, Array] | None]:
    """Build the merged stochastic+PPGD term's forward inputs.

    Adversarial samples use the persistent bundle and route every site; the rest use
    fresh uniform sources and the stochastic routes. The persistent sources remain graph
    leaves, so the per-sample selection gates their gradient to adversarial samples.
    The per-site fold index follows `ci_lower`'s insertion order — the CI fn's canonical
    output order, stable across traces.
    """
    # The merged term has no narrow arm: its fresh uniform sources are full-width
    # per-token draws — a narrow-emitting run authors separate stochastic and
    # persistent terms (require_full_emission refuses per site).
    full_ci_lower = {site: require_full_emission(ci) for site, ci in ci_lower.items()}
    ci_lower = full_ci_lower
    assignment_key, uniform_key = random.split(key)
    component_key, delta_key = random.split(uniform_key)
    adversarial = _per_sample_adversarial_assignment(assignment_key, adv_fraction, leading)
    fresh_uniform_sources: Sources = {
        site: SiteSource(
            components=uniform_like(random.fold_in(component_key, site_idx), ci, dtype=jnp.float32),
            delta=uniform_like(
                random.fold_in(delta_key, site_idx), ci, drop_last_axis=True, dtype=jnp.float32
            ),
        )
        for site_idx, (site, ci) in enumerate(ci_lower.items())
    }
    adv_masks, adv_deltas = masks_from_sources(full_ci_lower, persistent_sources)
    stoch_masks, stoch_deltas = masks_from_sources(full_ci_lower, fresh_uniform_sources)

    def blend(selector: Array, adversarial_arm: Array, stochastic_arm: Array) -> Array:
        """`where` with a 0/1 float selector, spelled arithmetically: exact for finite
        arms, and — unlike `select_n`, which demands exactly equal shardings across
        `which` and every case — broadcast rules accept a replicated selector against
        axis-typed arms."""
        selector = selector.astype(adversarial_arm.dtype)
        return selector * adversarial_arm + (1 - selector) * stochastic_arm

    masks = {
        site: blend(
            adversarial[..., None],
            require_full_emission(adv_masks[site]),
            require_full_emission(stoch_masks[site]),
        )
        for site in ci_lower
    }
    delta_masks = {
        site: blend(adversarial, adv_deltas[site], stoch_deltas[site]) for site in ci_lower
    }
    routes = (
        None
        if stochastic_routes is None
        else {site: jnp.logical_or(adversarial, stochastic_routes[site]) for site in ci_lower}
    )
    return masks, delta_masks, routes
