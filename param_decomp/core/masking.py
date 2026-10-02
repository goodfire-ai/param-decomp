"""Mask materialization shared by reconstruction objectives and model targets.

Selected masks carry token-ordered values and their block indices together.
Blocked source tables are read at those selected rows without forming a full-C
per-token mask."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float, PRNGKeyArray

from param_decomp.core.adversary import (
    BlockedSourceComponents,
    SiteSource,
    Sources,
    SourceStacks,
    full_source_components,
)
from param_decomp.core.components import (
    SelectedCI,
    SiteCI,
    SiteSpec,
    map_site_ci,
    site_ci_leading,
    site_ci_values,
)
from param_decomp.core.linear_plan import uniform_like, value_mesh
from param_decomp.core.model import (
    Masking,
    MaterializedMasking,
    SiteRoutes,
    SourceMasking,
    StochasticMasking,
)
from param_decomp.core.source_mask import SourceMaskIngredients


def _selected_source_values(source: BlockedSourceComponents, ci: SelectedCI) -> Array:
    """One selected-emitting site's selected slice of a block-dim source, in the CI's
    dtype: `out[.., m, :] = source.values[.., ids[.., m], :]`. The cast happens on the
    table BEFORE the select — pointwise, so it commutes with the select — and fuses with
    the source's dequant into the selecting op, which then also fixes the transpose's
    dtype: selected cotangents land on the stored source's shape at the CI dtype (the
    velocity's), never widened to the table's fp32 view.

    Two spellings, by the source's lead (`SourceShape`): a source spanning the CI's
    full lead (`bsc`) is a take along the block axis — exact, its transpose a
    scatter of each selected cotangent onto its own entry; a source with size-1
    broadcast lead axes is a one-hot contraction over the block axis, which
    broadcasts through the einsum's typed rule where the explicit sharding rule
    cannot resolve a take of a size-1 table axis against a batch-sharded index axis."""
    table = source.values.astype(ci.values.dtype)
    lead = ci.values.shape[:-1]
    assert table.shape[-2:] == (ci.n_blocks, ci.c_per_block), (table.shape, ci.C)
    assert all(extent in (1, full) for extent, full in zip(table.shape[:-2], lead, strict=True)), (
        table.shape,
        lead,
    )
    if table.shape[:-2] == lead:
        gathered = _take_selected_rows(table, ci)
    else:
        gathered = _contract_selected_rows(table, ci)
    return gathered.reshape(*lead, ci.values.shape[-1])


def _take_selected_rows(
    table: Float[Array, "*lead E c"], ci: SelectedCI
) -> Float[Array, "*lead k c"]:
    """`table[.., ids[.., m], :]` for a table spanning the CI's lead. On a mesh the
    block axis is TP-sharded, so the take is shard-local: the block axis viewed
    `(shard, local block)`, each shard taking its own blocks' rows by shard-local
    index (a foreign pick reads an arbitrary in-range row and is zeroed), then the
    typed sum over shards — the sharding rule's one all-reduce, exact since every pick
    is live on exactly one shard."""
    ids = ci.block_indices
    mesh = value_mesh(ids)
    if mesh.empty:
        return jnp.take_along_axis(table, ids[..., None], axis=-2)
    table_spec = jax.typeof(table).sharding.spec
    lead_spec = jax.typeof(ids).sharding.spec[:-1]
    block_axis = table_spec[-2]
    assert block_axis is None or isinstance(block_axis, str), table_spec
    n_shards = 1 if block_axis is None else mesh.shape[block_axis]
    n_blocks_local = ci.n_blocks // n_shards
    view = jnp.reshape(
        table,
        (*table.shape[:-2], n_shards, n_blocks_local, ci.c_per_block),
        out_sharding=NamedSharding(mesh, P(*table_spec[:-2], block_axis, None, None)),
    )
    local_ids = jax.sharding.reshard(
        ids[..., None, :] - (jnp.arange(n_shards) * n_blocks_local)[:, None],
        NamedSharding(mesh, P(*lead_spec, block_axis, None)),
    )
    live = (local_ids >= 0) & (local_ids < n_blocks_local)
    rows = jnp.take_along_axis(view, jnp.clip(local_ids, 0, n_blocks_local - 1)[..., None], axis=-2)
    return jnp.einsum(
        "...skc->...kc",
        jnp.where(live[..., None], rows, 0.0),
        out_sharding=NamedSharding(mesh, P(*lead_spec, None, None)),
    )


def _contract_selected_rows(
    table: Float[Array, "*lead E c"], ci: SelectedCI
) -> Float[Array, "*lead k c"]:
    """`table[.., ids[.., m], :]` as a one-hot contraction over the block axis, for a
    table whose size-1 lead axes broadcast against the CI's; exact for 0/1 weights."""
    one_hot = jax.nn.one_hot(ci.block_indices, ci.n_blocks, dtype=table.dtype)
    mesh = value_mesh(ci.block_indices)
    if mesh.empty:
        return jnp.einsum("...ke,...ec->...kc", one_hot, table)
    ids_spec = jax.typeof(ci.block_indices).sharding.spec
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


def sample_component_mask(ci: SiteCI, key: Array) -> SiteCI:
    return map_site_ci(lambda v: v + (1.0 - v) * uniform_like(key, v), ci)


def sample_delta_mask(ci: SiteCI, key: Array) -> Array:
    return uniform_like(key, site_ci_values(ci), drop_last_axis=True)


def _constant_delta_mask(ci: SiteCI, value: float) -> Array:
    values = site_ci_values(ci)
    return jnp.full(site_ci_leading(ci), value, values.dtype)


def materialize_masking(masking: Masking) -> MaterializedMasking:
    """Materialize a complete per-site recipe, independently of target layout.

    Stochastic fold indices follow the CI mapping's insertion order. Selected sites
    draw only their token/selected-slot values; no unselected-component draw exists.
    """
    match masking:
        case MaterializedMasking():
            return masking
        case StochasticMasking(ci=ci_lower, draw_key=draw_key):
            mask_key, delta_key = random.split(draw_key)
            component_masks: dict[str, SiteCI] = {}
            weight_delta_masks: dict[str, Array] = {}
            for site_idx, (site, ci) in enumerate(ci_lower.items()):
                component_masks[site] = sample_component_mask(
                    ci, random.fold_in(mask_key, site_idx)
                )
                weight_delta_masks[site] = sample_delta_mask(
                    ci, random.fold_in(delta_key, site_idx)
                )
            return MaterializedMasking(
                component_masks=component_masks,
                weight_delta_masks=weight_delta_masks,
            )
        case SourceMasking(ingredients=ingredients):
            return MaterializedMasking(
                component_masks={site: pair.compose() for site, pair in ingredients.items()},
                weight_delta_masks={site: pair.delta for site, pair in ingredients.items()},
            )


def stochastic_delta_pinned_masking(
    ci_lower: Mapping[str, SiteCI], draw_key: Array
) -> MaterializedMasking:
    """Stochastic component masks with every weight-delta mask pinned to 1.0 — the tPD
    non-target pass, where `components + Δ` must reconstruct the frozen output.

    Pre-built (`MaterializedMasking`) rather than the in-target `StochasticMasking`
    rebuild, which draws its own `U[0,1]` delta inside each block and cannot pin it. The
    key split mirrors `materialize_masking` (source half used, delta half discarded —
    the delta is deterministic here), as does the fold order (`ci_lower` insertion order,
    the CI fn's canonical output order)."""
    mask_key, _ = random.split(draw_key)
    masks: dict[str, SiteCI] = {}
    delta_masks: dict[str, Array] = {}
    for site_idx, (site, ci) in enumerate(ci_lower.items()):
        site_key = random.fold_in(mask_key, site_idx)
        masks[site] = sample_component_mask(ci, site_key)
        delta_masks[site] = _constant_delta_mask(ci, 1.0)
    return MaterializedMasking(component_masks=masks, weight_delta_masks=delta_masks)


def constant_delta_pinned_masking(
    value: float, ci_lower: Mapping[str, SiteCI]
) -> MaterializedMasking:
    """Constant component masks (`ci + (1-ci)·value`) with every weight-delta mask pinned
    to 1.0 — the tPD non-target pass's constant-source arm. The plain objective's
    constant arm carries NO delta path at all; here the delta must be fully on."""
    masks = {
        site: map_site_ci(lambda v: v + (1.0 - v) * value, ci) for site, ci in ci_lower.items()
    }
    delta_masks = {site: _constant_delta_mask(ci, 1.0) for site, ci in ci_lower.items()}
    return MaterializedMasking(component_masks=masks, weight_delta_masks=delta_masks)


def unmasked_no_delta_masking(ci_lower: Mapping[str, SiteCI]) -> MaterializedMasking:
    """Set component masks to 1, with no weight delta, for non-target reconstruction.

    The full component sum must reconstruct without help from the delta, including
    components that never activate. `ci_lower` supplies only shapes and dtypes."""
    masks = {site: map_site_ci(jnp.ones_like, ci) for site, ci in ci_lower.items()}
    return MaterializedMasking(component_masks=masks, weight_delta_masks=None)


def sample_source_pool(
    key: PRNGKeyArray,
    source_pool: SourceStacks,
    leading: tuple[int, ...],
) -> Sources:
    """Draw one cross-site particle per batch index and broadcast over positions.

    Storage owns the particle count and batch placement. Sampling whole stacks shares
    the draw across sites without changing their component or expert partitions.
    """
    reference = next(iter(source_pool.stacks.values())).delta
    _, batch, particles = reference.shape
    consumer_batch, *positions = leading
    assert consumer_batch == batch, (leading, reference.shape)
    mesh = value_mesh(reference)
    batch_axis = jax.typeof(reference).sharding.spec[1]
    batch_keys = random.split(key, batch)
    if not mesh.empty:
        batch_keys = jax.sharding.reshard(batch_keys, NamedSharding(mesh, P(batch_axis)))

    def sample_stack(table: Array) -> Array:
        assert jnp.issubdtype(table.dtype, jnp.floating), (
            "Source-pool sampling requires the floating source view"
        )
        assert table.shape[1:3] == (batch, particles), table.shape
        if not mesh.empty:
            spec = jax.typeof(table).sharding.spec
            assert spec[1] == batch_axis and spec[2] is None, spec
        selected = jax.vmap(
            lambda batch_key, minipool: random.choice(batch_key, minipool, axis=1),
            in_axes=(0, 1),
            out_axes=1,
        )(batch_keys, table)
        return selected.reshape(table.shape[0], batch, *(1 for _ in positions), *table.shape[3:])

    return jax.tree.map(sample_stack, source_pool).per_site()


def read_source_mask(ci: SiteCI, source: SiteSource) -> SourceMaskIngredients:
    """Align source storage to this CI's component coordinates, preserving source gradients."""
    assert all(jnp.issubdtype(leaf.dtype, jnp.floating) for leaf in jax.tree.leaves(source)), (
        "Masks require the floating source view, never the storage representation"
    )
    match ci:
        case SelectedCI():
            assert isinstance(source.components, BlockedSourceComponents)
            values = _selected_source_values(source.components, ci)
        case jax.Array():
            values = full_source_components(source.components).astype(ci.dtype)
    return SourceMaskIngredients(ci, values, source.delta.astype(site_ci_values(ci).dtype))


def source_mask_ingredients(
    ci_lower: Mapping[str, SiteCI], sources: Mapping[str, SiteSource]
) -> dict[str, SourceMaskIngredients]:
    """Read source tables into each CI's frame once, before stacking or rematerialization."""
    assert set(sources) == set(ci_lower), (sources.keys(), ci_lower.keys())
    return {site: read_source_mask(ci, sources[site]) for site, ci in ci_lower.items()}


def source_masking(
    ci_lower: Mapping[str, SiteCI],
    sources: Mapping[str, SiteSource],
) -> SourceMasking:
    """Align source tables with per-site CI to construct a logical masking recipe."""
    return SourceMasking(ingredients=source_mask_ingredients(ci_lower, sources))


def _per_sample_adversarial_assignment(
    key: PRNGKeyArray, adv_fraction: Array, leading: tuple[int, ...]
) -> Array:
    """Draw one Bernoulli selector per sample, broadcast across position axes."""
    one_flag_per_sample = (leading[0], *(1,) * (len(leading) - 1))
    return random.bernoulli(key, adv_fraction, one_flag_per_sample)


def _uniform_source_values(ci: SiteCI, key: PRNGKeyArray) -> Array:
    """Draw one source value per token and selected component."""
    values = site_ci_values(ci)
    return uniform_like(key, values, dtype=jnp.float32).astype(values.dtype)


def _uniform_delta_source(ci: SiteCI, key: PRNGKeyArray) -> Array:
    values = site_ci_values(ci)
    return uniform_like(key, values, drop_last_axis=True, dtype=jnp.float32).astype(values.dtype)


def _select_source_samples(adversarial: Array, source: Array, noise: Array) -> Array:
    """Select one family per document, broadcasting over every payload coordinate."""
    selector = adversarial.reshape(adversarial.shape[0], *(1 for _ in noise.shape[1:]))
    selector = selector.astype(source.dtype)
    return selector * source + (1 - selector) * noise


def mixed_persistent_stochastic_masking(
    key: PRNGKeyArray,
    ci_lower: Mapping[str, SiteCI],
    persistent_sources: Sources,
    leading: tuple[int, ...],
    adv_fraction: Array,
    stochastic_routes: SiteRoutes | None,
) -> tuple[SourceMasking, SiteRoutes | None]:
    """Select persistent or fresh sources per document in one CI frame, with the routes
    that forward runs under: adversarial documents route every position, the rest keep
    `stochastic_routes`.

    Source gradients reach adversarial documents only. Selected noise retains canonical
    token/slot identity; delta and site routing retain logical token coordinates.
    Targets compose the selected source with CI under their ordinary rematerialization.
    """
    assignment_key, uniform_key = random.split(key)
    component_key, delta_key = random.split(uniform_key)
    adversarial = _per_sample_adversarial_assignment(assignment_key, adv_fraction, leading)
    ingredients = {}
    for site_idx, (site, pair) in enumerate(
        source_mask_ingredients(ci_lower, persistent_sources).items()
    ):
        ingredients[site] = SourceMaskIngredients(
            ci=pair.ci,
            source_values=_select_source_samples(
                adversarial,
                pair.source_values,
                _uniform_source_values(pair.ci, random.fold_in(component_key, site_idx)),
            ),
            delta=_select_source_samples(
                adversarial,
                pair.delta,
                _uniform_delta_source(pair.ci, random.fold_in(delta_key, site_idx)),
            ),
        )
    routes = (
        None
        if stochastic_routes is None
        else {site: jnp.logical_or(adversarial, stochastic_routes[site]) for site in ci_lower}
    )
    return SourceMasking(ingredients=ingredients), routes


def sample_source_rows(
    key: PRNGKeyArray, ci_lower: Mapping[str, SiteCI], sources: Sources
) -> Sources:
    """A persistent term's sources as a batch OTHER than the training batch reads them.

    A source's stored batch extent is its `SourceShape` spelling (`configs.SourceShape`):
    1 means no `b` — the source is shared over the batch and applies to any batch as it
    is. A batch-indexed source's (`bc`/`bsc`) rows index the TRAINING batch's samples,
    which this batch does not have: each batch element draws one training row uniformly,
    the same row at every site (row `i` stays one jointly trained attack), so in
    expectation the batch sees the adversary the training batch saw — whatever the two
    batch sizes are. The gather runs along the sources' batch axis in place; on a mesh
    the drawn rows ride the batch axis's sharding and the result keeps every other axis's
    spec."""
    extents = {source.delta.shape[0] for source in sources.values()}
    assert len(extents) == 1, f"sources' batch extents must agree across sites, got {extents}"
    (n_train,) = extents
    if n_train == 1:
        return dict(sources)
    reference = site_ci_values(next(iter(ci_lower.values())))
    n_rows = reference.shape[0]
    mesh = value_mesh(reference)
    if mesh.empty:
        rows = random.randint(key, (n_rows,), 0, n_train, dtype=jnp.int32)
    else:
        batch_axis = jax.typeof(reference).sharding.spec[0]
        rows = random.randint(
            key,
            (n_rows,),
            0,
            n_train,
            dtype=jnp.int32,
            out_sharding=NamedSharding(mesh, P(batch_axis)),
        )

    def gather(table: Array) -> Array:
        assert table.shape[0] == n_train, (table.shape, n_train)
        if mesh.empty:
            return table[rows]
        table_spec = jax.typeof(table).sharding.spec
        rows_spec = jax.typeof(rows).sharding.spec
        return table.at[rows].get(out_sharding=NamedSharding(mesh, P(*rows_spec, *table_spec[1:])))

    sampled: Sources = {}
    for site, source in sources.items():
        assert all(jnp.issubdtype(leaf.dtype, jnp.floating) for leaf in jax.tree.leaves(source)), (
            f"site {site!r}: source-row sampling reads the float source view"
        )
        match source.components:
            case BlockedSourceComponents(values=values):
                components = BlockedSourceComponents(values=gather(values))
            case jax.Array():
                components = gather(source.components)
        sampled[site] = SiteSource(components=components, delta=gather(source.delta))
    return sampled
