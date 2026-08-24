"""Measure whether a higher causal importance preactivation means that ablating the
component has a greater effect on the model's output.

For randomly selected inputs (and token positions for language models), this code samples
components and ablates them one at a time. The effect of each ablation is the model's
reconstruction loss at the selected location between the ablated output and the output with
all components present; downstream language-model positions are deliberately not scored. It
reports how often, across pairs of components, the component with the higher preactivation also
has the larger effect. It makes this comparison separately for preactivations below 0, between
0 and 1, and above 1, and compares components from all heads and layers. Random ordering scores
0.5; perfect ordering scores 1. A near-chance `below_zero` score can mean that the ablation
effects are below the bf16 measurement floor rather than that the ordering is ill-tempered.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int, PRNGKeyArray
from numpy.typing import NDArray

from param_decomp.core.ci_fn import (
    PlacedCIFn,
    evaluate_compute_ci,
    materialize_ci_compute_weights,
)
from param_decomp.core.components import ComponentStacks
from param_decomp.core.configs import WellTemperednessConfig
from param_decomp.core.jit_util import filter_jit
from param_decomp.core.linear_plan import value_mesh
from param_decomp.core.masking import all_live_masking_no_delta
from param_decomp.core.model import (
    BATCH_AXES,
    CaptureKeys,
    MaterializedMasking,
    PlacedModel,
    prepare_compute_weights,
)
from param_decomp.core.precision import COMPUTE_DT
from param_decomp.core.sharding import batch_shard_leading

type Region = Literal["below_zero", "zero_to_one", "above_one"]
type NumericArray = Array | NDArray[Any]

REGIONS: tuple[Region, ...] = ("below_zero", "zero_to_one", "above_one")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Ablations:
    preactivations: Float[NumericArray, "n_regions n_locations n_components"]
    damage: Float[NumericArray, "n_regions n_locations n_components"]
    site_indices: Int[NumericArray, "n_regions n_locations n_components"]


type WellTemperednessStep = Callable[
    [PlacedModel, ComponentStacks, PlacedCIFn, Any, PRNGKeyArray], Ablations
]


def in_region[ArrayT: (Array, NDArray[Any])](preactivations: ArrayT, region: Region) -> ArrayT:
    match region:
        case "below_zero":
            return preactivations <= 0.0
        case "zero_to_one":
            return (preactivations > 0.0) & (preactivations < 1.0)
        case "above_one":
            return preactivations >= 1.0


def _components_to_ablate(
    preactivations: Float[Array, "n_locations C"],
    n_components: int,
    sampling_key: PRNGKeyArray,
) -> Int[Array, "n_regions n_locations n_components"]:
    """Draw without replacement in each preactivation range.

    If a range has too few components, the returned array is padded with components outside
    the range. The scoring code ignores those entries.
    """
    sampling_scores = jnp.stack(
        [
            jnp.where(
                in_region(preactivations, region),
                random.uniform(random.fold_in(sampling_key, region_index), preactivations.shape),
                -jnp.inf,
            )
            for region_index, region in enumerate(REGIONS)
        ]
    )
    _, global_component_indices = jax.lax.top_k(sampling_scores, n_components)
    return global_component_indices


def make_well_temperedness_step(
    model_static: PlacedModel,
    ci_capture_keys: CaptureKeys,
    config: WellTemperednessConfig,
    mesh: Mesh | None = None,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> WellTemperednessStep:
    site_names = model_static.site_names
    site_specs = model_static.sites
    has_position_axis = model_static.has_position_axis
    per_ablation_recon_loss = jax.vmap(model_static.recon_loss_fn)
    ablations_per_forward = config.ablations_per_forward
    indices_within_forward = jnp.arange(ablations_per_forward)
    ablation_shape = (len(REGIONS), config.n_locations, config.n_components_per_region)
    n_ablations = math.prod(ablation_shape)
    assert n_ablations % ablations_per_forward == 0, (
        f"WellTemperedness makes {n_ablations} ablations, not divisible by "
        f"ablations_per_forward {ablations_per_forward}"
    )
    if mesh is not None:
        batch_mesh_extent = math.prod(mesh.shape[axis] for axis in BATCH_AXES)
        assert ablations_per_forward % batch_mesh_extent == 0, (
            f"WellTemperedness ablations_per_forward {ablations_per_forward} is not divisible "
            f"by batch mesh extent {batch_mesh_extent}"
        )
    location_indices_by_forward = jnp.broadcast_to(
        jnp.arange(config.n_locations)[None, :, None], ablation_shape
    ).reshape(-1, ablations_per_forward)
    site_component_offsets = jnp.asarray(
        np.cumsum((0, *(site_spec.C for site_spec in site_specs)))[:-1], dtype=jnp.int32
    )
    site_component_boundaries = site_component_offsets[1:]
    total_components = sum(site_spec.C for site_spec in site_specs)
    assert config.n_components_per_region <= total_components, (
        f"WellTemperedness draws {config.n_components_per_region} components per region from "
        f"{total_components} components"
    )

    def select_locations(
        values: Float[Array, "B *position d"],
        batch_indices: Int[Array, " n_locations"],
        position_indices: Int[Array, " n_locations"],
    ) -> Float[Array, "n_locations d"]:
        # A batch-typed operand's location gather has no inferable output sharding
        # (n_locations does not follow the batch axes); the tiny selected slab is
        # replicated, like every downstream ablation quantity.
        if not value_mesh(values).empty:
            values = jax.sharding.reshard(
                values,
                NamedSharding(value_mesh(values), P(*([None] * values.ndim))),
            )
        selected_values = (
            values[batch_indices, position_indices] if has_position_axis else values[batch_indices]
        )
        return selected_values.astype(jnp.float32)

    # `model` is the filter_jit ARG: frozen array fields stay traced instead of becoming HLO
    # constants. Only static topology and the array-free recon loss close over the factory.
    def step(
        model: PlacedModel,
        components: ComponentStacks,
        placed_ci_fn: PlacedCIFn,
        inputs: Any,
        sampling_key: PRNGKeyArray,
    ) -> Ablations:
        ci_inputs = model.clean_forward(inputs, ci_capture_keys).captures
        # The step's CI lifecycle: materialize the compute residents first — evaluating
        # persistence-layout weights leaves their gathers (and, under Explicit typing,
        # ambiguous weight-grad contractions) inside the chunk scan.
        compute_ci_fn = materialize_ci_compute_weights(placed_ci_fn)
        ci_preactivations = evaluate_compute_ci(
            compute_ci_fn, ci_inputs, remat=False
        ).preactivations
        location_shape = ci_preactivations[site_names[0]].shape[:-1]
        expected_location_rank = 2 if has_position_axis else 1
        assert len(location_shape) == expected_location_rank, location_shape
        location_key, component_sampling_key = random.split(sampling_key)
        batch_indices, position_indices = _choose_locations(
            location_key, location_shape, config.n_locations
        )
        position_shape = location_shape[1:]

        def ablated_masking(
            site_indices: Int[Array, " ablations_per_forward"],
            position_indices: Int[Array, " ablations_per_forward"],
            component_indices: Int[Array, " ablations_per_forward"],
        ) -> MaterializedMasking:
            component_masks = all_live_masking_no_delta(
                site_specs,
                leading_shape=(ablations_per_forward, *position_shape),
                dtype=COMPUTE_DT,
            ).component_masks
            for site_index, site_spec in enumerate(site_specs):
                belongs_to_site = site_indices == site_index
                component_indices_at_site = jnp.where(belongs_to_site, component_indices, 0)
                mask_indices = (
                    (indices_within_forward, position_indices, component_indices_at_site)
                    if has_position_axis
                    else (indices_within_forward, component_indices_at_site)
                )
                component_masks[site_spec.name] = (
                    component_masks[site_spec.name]
                    .at[mask_indices]
                    .set(jnp.where(belongs_to_site, 0.0, 1.0))
                )
            return MaterializedMasking(component_masks=component_masks)

        preactivations_at_locations = jnp.concatenate(
            [
                select_locations(ci_preactivations[site_name], batch_indices, position_indices)
                for site_name in site_names
            ],
            axis=-1,
        )
        global_component_indices = _components_to_ablate(
            preactivations_at_locations, config.n_components_per_region, component_sampling_key
        )
        selected_site_indices = jnp.searchsorted(
            site_component_boundaries, global_component_indices, side="right"
        ).astype(jnp.int32)
        selected_component_indices = (
            global_component_indices - site_component_offsets[selected_site_indices]
        )
        selected_preactivations = jnp.take_along_axis(
            preactivations_at_locations[None], global_component_indices, axis=-1
        )

        prepared_components = prepare_compute_weights(model, components)
        all_components_output = model.masked_forward(
            prepared_components,
            inputs,
            masking=all_live_masking_no_delta(
                site_specs, leading_shape=location_shape, dtype=COMPUTE_DT
            ),
            remat=False,
        ).output
        reference_outputs = select_locations(all_components_output, batch_indices, position_indices)
        site_indices_by_forward = selected_site_indices.reshape(-1, ablations_per_forward)
        component_indices_by_forward = selected_component_indices.reshape(-1, ablations_per_forward)

        def damage_forward(
            ablation_indices: tuple[
                Int[Array, " ablations_per_forward"],
                Int[Array, " ablations_per_forward"],
                Int[Array, " ablations_per_forward"],
            ],
        ) -> Float[Array, " ablations_per_forward"]:
            location_indices, site_indices, component_indices = ablation_indices

            def take_locations(values: Array) -> Array:
                # Location gathers off batch-typed inputs have no inferable output
                # sharding; the tiny per-forward subset materializes replicated and
                # batch_shard_leading re-pins it.
                if not value_mesh(values).empty:
                    values = jax.sharding.reshard(
                        values,
                        NamedSharding(value_mesh(values), P(*([None] * values.ndim))),
                    )
                return batch_shard_leading(
                    jnp.take(values, batch_indices[location_indices], axis=0), mesh
                )

            inputs_for_ablations = jax.tree.map(take_locations, inputs)
            ablated_outputs = model.masked_forward(
                prepared_components,
                inputs_for_ablations,
                masking=ablated_masking(
                    site_indices, position_indices[location_indices], component_indices
                ),
                remat=False,
            ).output
            if not value_mesh(ablated_outputs).empty:
                # Same as `select_locations`: the per-location gather off the
                # batch-typed output replicates the tiny selected slab.
                ablated_outputs = jax.sharding.reshard(
                    ablated_outputs,
                    NamedSharding(value_mesh(ablated_outputs), P(*([None] * ablated_outputs.ndim))),
                )
            ablated_outputs_at_locations = (
                ablated_outputs[indices_within_forward, position_indices[location_indices]]
                if has_position_axis
                else ablated_outputs
            ).astype(jnp.float32)
            return per_ablation_recon_loss(
                ablated_outputs_at_locations, reference_outputs[location_indices]
            )

        damage_by_forward = jax.lax.map(
            damage_forward,
            (
                location_indices_by_forward,
                site_indices_by_forward,
                component_indices_by_forward,
            ),
        )
        ablations = Ablations(
            preactivations=selected_preactivations,
            damage=damage_by_forward.reshape(ablation_shape),
            site_indices=selected_site_indices,
        )
        if mesh is None:
            return ablations
        replicated = NamedSharding(mesh, P())
        return jax.tree.map(lambda value: jax.sharding.reshard(value, replicated), ablations)

    return filter_jit(step, compiler_options=compiler_options)


def _choose_locations(
    sampling_key: PRNGKeyArray, location_shape: tuple[int, ...], n_locations: int
) -> tuple[Int[Array, " n_locations"], Int[Array, " n_locations"]]:
    """Choose input locations and return their batch and optional position indices."""
    n_available = math.prod(location_shape)
    assert n_locations <= n_available, (
        f"{n_locations} input locations requested from {n_available} available"
    )
    n_positions = math.prod(location_shape[1:])
    flat_indices = random.choice(sampling_key, n_available, (n_locations,), replace=False)
    return flat_indices // n_positions, flat_indices % n_positions


def fraction_well_ordered_pairs(
    preactivations: Float[NDArray[Any], "n_locations n_components"],
    damage: Float[NDArray[Any], "n_locations n_components"],
    region: Region,
    included_component_mask: NDArray[Any] | None,
) -> float | None:
    """Return the fraction of pairs ordered the same way, or `None` if there are no pairs."""
    assert np.isfinite(preactivations).all(), "well-temperedness preactivations must be finite"
    assert np.isfinite(damage).all(), "well-temperedness ablation damage must be finite"
    in_region_mask = in_region(preactivations, region)
    if included_component_mask is not None:
        assert included_component_mask.shape == in_region_mask.shape, (
            included_component_mask.shape,
            in_region_mask.shape,
        )
        in_region_mask &= included_component_mask
    concordance_score = 0.0
    n_pairs = 0
    for location in np.flatnonzero(in_region_mask.sum(1) >= 2):
        location_preactivations = preactivations[location][in_region_mask[location]]
        location_damage = damage[location][in_region_mask[location]]
        first_indices, second_indices = np.triu_indices(location_preactivations.size, k=1)
        ordering_products = (
            location_preactivations[first_indices] - location_preactivations[second_indices]
        ) * (location_damage[first_indices] - location_damage[second_indices])
        concordance_score += float((ordering_products > 0).sum()) + 0.5 * float(
            (ordering_products == 0).sum()
        )
        n_pairs += first_indices.size
    return concordance_score / n_pairs if n_pairs else None


def _region_fractions(
    ablations: Ablations,
    included_site_indices: tuple[int, ...] | None,
) -> dict[Region, float]:
    included_component_mask = (
        None
        if included_site_indices is None
        else np.isin(ablations.site_indices, included_site_indices)
    )
    return {
        region: fraction
        for region_index, region in enumerate(REGIONS)
        if (
            fraction := fraction_well_ordered_pairs(
                np.asarray(ablations.preactivations[region_index]),
                np.asarray(ablations.damage[region_index]),
                region,
                None if included_component_mask is None else included_component_mask[region_index],
            )
        )
        is not None
    }


def well_temperedness_log_entries(
    ablations: Ablations,
    site_groups: dict[str, tuple[int, ...]],
) -> dict[str, float]:
    site_scopes: dict[str, tuple[int, ...] | None] = {"all_sites": None, **site_groups}
    return {
        f"{scope}/fraction_well_ordered_pairs_{region}": fraction
        for scope, included_sites in site_scopes.items()
        for region, fraction in _region_fractions(ablations, included_sites).items()
    }
