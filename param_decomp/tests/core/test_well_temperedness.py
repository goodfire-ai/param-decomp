"""Tests for the well-temperedness measurement."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    LayerwiseMLPCIArch,
    MHACIAttention,
    PlacedCIFn,
    build_ci_fn,
    evaluate_ci,
)
from param_decomp.core.components import ComponentStacks, SiteC, init_component_stacks
from param_decomp.core.configs import WellTemperednessConfig
from param_decomp.core.model import (
    DecomposedModel,
    MaterializedMasking,
    PlacedModel,
    prepare_compute_weights,
)
from param_decomp.core.placement import from_config
from param_decomp.core.precision import COMPUTE_DT
from param_decomp.core.sharding import hsdp_mesh, place_target
from param_decomp.core.well_temperedness import (
    REGIONS,
    Ablations,
    Region,
    _choose_locations,
    _components_to_ablate,
    _region_fractions,
    fraction_well_ordered_pairs,
    in_region,
    make_well_temperedness_step,
    well_temperedness_log_entries,
)
from param_decomp.core.well_temperedness_eval import (
    _plot_preactivation_vs_damage,
    _resolve_groups,
)
from param_decomp.targets.glu_transformer import glu_site_specs, mlp_family_site_cs
from param_decomp.targets.llama_simple_mlp import canonical_site_cs
from param_decomp.targets.llama_simple_mlp import site_specs as simple_mlp_site_specs
from param_decomp.targets.testing import SIMPLE_MLP_MIXED_SITE_CS as _MIXED_SITE_CS
from param_decomp.targets.testing import tiny_glu_cfg as _tiny_cfg
from param_decomp.targets.testing import tiny_glu_decomposed_lm as _tiny_decomposed_lm
from param_decomp.targets.testing import tiny_simple_mlp_cfg as _simple_mlp_cfg
from param_decomp.targets.testing import (
    tiny_simple_mlp_decomposed_model as _tiny_decomposed_simple_mlp,
)
from param_decomp.targets.tms import (
    TMSConfig,
    init_tms_target,
    sample_sparse_features,
    site_input_tap_keys,
    tms_decomposed_model,
)
from param_decomp.targets.tms import (
    site_specs as tms_site_specs,
)

_BATCH, _SEQ, _C = 3, 7, 6
_DAMAGE_NOISE_FLOOR = 1e-6


def _assert_damage_close(actual: np.ndarray, expected: np.ndarray, message: str) -> None:
    expected_scale = float(np.abs(expected).max())
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0,
        atol=max(0.01 * expected_scale, _DAMAGE_NOISE_FLOOR),
        err_msg=message,
    )


def _build_ci_fn(model: DecomposedModel, n_embd: int, key: jax.Array) -> PlacedCIFn:
    first_block = min(int(name.split(".")[1]) for name in model.site_names)
    architecture = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(f"resid.{first_block}",), output_sites=model.site_names),),
        input_dim=n_embd,
        d_model=16,
        n_blocks=1,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )
    return PlacedCIFn(fn=build_ci_fn(architecture, model.sites, key), placement=None)


def _setup_glu_transformer() -> tuple[PlacedModel, ComponentStacks, PlacedCIFn, jax.Array]:
    target_config = _tiny_cfg()
    site_specs = glu_site_specs(target_config, mlp_family_site_cs(4, 5, _C))
    model = _tiny_decomposed_lm(target_config, site_specs, jax.random.PRNGKey(0))
    components = init_component_stacks(site_specs, jax.random.PRNGKey(1))
    ci_fn = _build_ci_fn(model, target_config.n_embd, jax.random.PRNGKey(2))
    tokens = jax.random.randint(jax.random.PRNGKey(3), (_BATCH, _SEQ), 0, target_config.vocab_size)
    return PlacedModel(model=model, placement=None), components, ci_fn, tokens


def _setup_simple_mlp() -> tuple[PlacedModel, ComponentStacks, PlacedCIFn, jax.Array]:
    target_config = _simple_mlp_cfg()
    site_specs = simple_mlp_site_specs(target_config, canonical_site_cs(_MIXED_SITE_CS))
    model = _tiny_decomposed_simple_mlp(target_config, site_specs, jax.random.PRNGKey(0))
    components = init_component_stacks(site_specs, jax.random.PRNGKey(1))
    ci_fn = _build_ci_fn(model, target_config.n_embd, jax.random.PRNGKey(2))
    tokens = jax.random.randint(jax.random.PRNGKey(3), (_BATCH, _SEQ), 0, target_config.vocab_size)
    return PlacedModel(model=model, placement=None), components, ci_fn, tokens


TARGET_SETUPS = {"glu_transformer": _setup_glu_transformer, "llama_simple_mlp": _setup_simple_mlp}


def _well_temperedness_config(
    *,
    groups: dict[str, list[str]] | None = None,
    n_locations: int,
    n_components_per_region: int,
    ablations_per_forward: int = 1,
) -> WellTemperednessConfig:
    return WellTemperednessConfig(
        groups=groups,
        n_locations=n_locations,
        n_components_per_region=n_components_per_region,
        ablations_per_forward=ablations_per_forward,
    )


def _measure_ablations(
    model: PlacedModel,
    components: ComponentStacks,
    ci_fn: PlacedCIFn,
    tokens: jax.Array,
    config: WellTemperednessConfig,
) -> tuple[Ablations, jax.Array, jax.Array]:
    sampling_key = jax.random.PRNGKey(11)
    location_key, _ = jax.random.split(sampling_key)
    batch_indices, position_indices = _choose_locations(
        location_key, (_BATCH, _SEQ), config.n_locations
    )
    ablations = make_well_temperedness_step(model, ci_fn.fn.capture_keys, config)(
        model,
        components,
        ci_fn,
        tokens,
        sampling_key,
    )
    return ablations, batch_indices, position_indices


def _run_glu_transformer(config: WellTemperednessConfig) -> Ablations:
    model, components, ci_fn, tokens = _setup_glu_transformer()
    ablations, _, _ = _measure_ablations(model, components, ci_fn, tokens, config)
    return jax.device_get(ablations)


def _preactivations_inside(
    region: Region, shape: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    match region:
        case "below_zero":
            return -np.abs(rng.normal(2.0, 1.5, shape)) - 1e-3
        case "zero_to_one":
            return rng.uniform(1e-3, 1.0 - 1e-3, shape)
        case "above_one":
            return 1.0 + np.abs(rng.normal(1.5, 1.0, shape))


def _synthetic_ablations(site_names: tuple[str, ...]) -> Ablations:
    rng = np.random.default_rng(0)
    n_locations, components_per_site = 4, 8
    n_components = len(site_names) * components_per_site
    ablation_shape = (n_locations, n_components)
    site_indices = np.broadcast_to(
        np.repeat(np.arange(len(site_names)), components_per_site),
        (len(REGIONS), n_locations, n_components),
    )
    return Ablations(
        preactivations=np.stack(
            [_preactivations_inside(region, ablation_shape, rng) for region in REGIONS]
        ),
        damage=np.abs(rng.normal(0, 1.0, (len(REGIONS), *ablation_shape))),
        site_indices=site_indices,
    )


def _expected_component_selection(
    model: PlacedModel,
    ci_fn: PlacedCIFn,
    tokens: jax.Array,
    config: WellTemperednessConfig,
    batch_indices: jax.Array,
    position_indices: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    preactivations_by_site = evaluate_ci(
        ci_fn,
        model.clean_forward(tokens, ci_fn.fn.capture_keys).captures,
        remat=False,
    ).preactivations
    preactivations_at_locations = jnp.concatenate(
        [
            preactivations_by_site[site_name][batch_indices, position_indices].astype(jnp.float32)
            for site_name in model.site_names
        ],
        axis=-1,
    )
    global_component_indices = _components_to_ablate(
        preactivations_at_locations,
        config.n_components_per_region,
        jax.random.split(jax.random.PRNGKey(11))[1],
    )
    site_component_offsets = jnp.asarray(
        np.cumsum((0, *(site_spec.C for site_spec in model.sites)))[:-1], dtype=jnp.int32
    )
    selected_site_indices = (
        (global_component_indices[..., None] >= site_component_offsets[1:])
        .sum(axis=-1)
        .astype(jnp.int32)
    )
    selected_component_indices = (
        global_component_indices - site_component_offsets[selected_site_indices]
    )
    return (
        preactivations_at_locations,
        global_component_indices,
        selected_site_indices,
        selected_component_indices,
    )


@pytest.mark.parametrize("target", TARGET_SETUPS)
def test_swept_damage_matches_hand_built_single_ablation(target: str):
    model, components, ci_fn, tokens = TARGET_SETUPS[target]()
    config = _well_temperedness_config(
        n_locations=2, n_components_per_region=3, ablations_per_forward=6
    )
    ablations, batch_indices, position_indices = _measure_ablations(
        model, components, ci_fn, tokens, config
    )
    _, _, selected_site_indices, selected_component_indices = _expected_component_selection(
        model, ci_fn, tokens, config, batch_indices, position_indices
    )
    np.testing.assert_array_equal(
        np.asarray(ablations.site_indices), np.asarray(selected_site_indices)
    )
    prepared_components = prepare_compute_weights(model, components)
    all_components_masks = {
        site_spec.name: jnp.ones((_BATCH, _SEQ, site_spec.C), COMPUTE_DT)
        for site_spec in model.sites
    }
    all_components_output = model.masked_forward(
        prepared_components,
        tokens,
        masking=MaterializedMasking(component_masks=all_components_masks),
        remat=False,
    ).output

    @eqx.filter_jit
    def hand_built_at_location(
        model: PlacedModel,
        masks: dict[str, jax.Array],
        batch_index: int,
        position_index: int,
    ) -> jax.Array:
        output = model.masked_forward(
            prepared_components,
            tokens,
            masking=MaterializedMasking(component_masks=masks),
            remat=False,
        ).output
        return output[batch_index, position_index].astype(jnp.float32)

    for region_index in range(len(REGIONS)):
        for location_index in range(config.n_locations):
            batch = int(batch_indices[location_index])
            position = int(position_indices[location_index])
            reference_output = all_components_output[batch, position].astype(jnp.float32)
            expected_damage = []
            for sample_index in range(config.n_components_per_region):
                site_name = model.site_names[
                    int(selected_site_indices[region_index, location_index, sample_index])
                ]
                component_index = int(
                    selected_component_indices[region_index, location_index, sample_index]
                )
                masks = dict(all_components_masks)
                masks[site_name] = masks[site_name].at[batch, position, component_index].set(0.0)
                ablated_output = hand_built_at_location(model, masks, batch, position)
                expected_damage.append(float(model.recon_loss_fn(ablated_output, reference_output)))
            _assert_damage_close(
                np.asarray(ablations.damage[region_index, location_index]),
                np.asarray(expected_damage),
                f"{target} {REGIONS[region_index]} location {location_index}",
            )


def test_components_are_sampled_uniformly_within_a_region():
    n_components = 12
    n_draws = 3
    n_trials = 256
    preactivations = jnp.arange(n_components, dtype=jnp.float32)[None] + 1.0
    keys = jax.random.split(jax.random.PRNGKey(17), n_trials)
    selections = jax.vmap(lambda key: _components_to_ablate(preactivations, n_draws, key)[2, 0])(
        keys
    )
    counts = np.bincount(np.asarray(selections).ravel(), minlength=n_components)
    expected = n_trials * n_draws / n_components
    np.testing.assert_allclose(counts, expected, rtol=0, atol=24)


def test_positionless_target_runs_the_complete_ablation_step():
    target_config = TMSConfig(n_features=5, n_hidden=2)
    sites = tms_site_specs(target_config, (SiteC("linear1", 8), SiteC("linear2", 6)))
    model = tms_decomposed_model(
        target_config, init_tms_target(target_config, jax.random.PRNGKey(0)), sites
    )
    components = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = build_ci_fn(
        LayerwiseMLPCIArch(
            hidden_dims=(16,),
            has_position_axis=False,
            input_names=site_input_tap_keys(tuple(site.name for site in sites)),
        ),
        sites,
        jax.random.PRNGKey(2),
    )
    inputs = sample_sparse_features(
        jax.random.PRNGKey(3), 8, target_config.n_features, 0.5, "at_least_zero_active"
    )
    config = _well_temperedness_config(
        n_locations=3, n_components_per_region=4, ablations_per_forward=4
    )

    placed = PlacedModel(model=model, placement=None)
    ablations = make_well_temperedness_step(placed, ci_fn.capture_keys, config)(
        placed, components, PlacedCIFn(fn=ci_fn, placement=None), inputs, jax.random.PRNGKey(4)
    )

    assert ablations.damage.shape == (len(REGIONS), 3, 4)
    assert np.isfinite(np.asarray(ablations.damage)).all()


@pytest.mark.multidevice
def test_mesh_outputs_are_replicated_and_chunk_batch_must_tile_mesh():
    model, components, ci_fn, tokens = _setup_glu_transformer()
    mesh = hsdp_mesh(1, jax.device_count(), 1)
    rules = from_config("ddp", mesh, model.sites)
    # A placed forward's batch must tile the data axes (the Explicit reshard refuses a
    # ragged split the constraint-based lowering used to pad-shard).
    n_data = mesh.shape["replicate"] * mesh.shape["fsdp"]
    reps = -(-n_data // tokens.shape[0])
    tokens = jnp.tile(tokens, (reps, 1))[:n_data]
    batch_mesh_extent = mesh.shape["replicate"] * mesh.shape["fsdp"]
    invalid_chunk = _well_temperedness_config(
        n_locations=2,
        n_components_per_region=batch_mesh_extent,
        ablations_per_forward=3 * batch_mesh_extent // 2,
    )
    with pytest.raises(AssertionError, match="batch mesh extent"):
        make_well_temperedness_step(model, ci_fn.fn.capture_keys, invalid_chunk, mesh)

    config = _well_temperedness_config(
        n_locations=1,
        n_components_per_region=batch_mesh_extent,
        ablations_per_forward=batch_mesh_extent,
    )
    model = place_target(model.model, rules)
    with jax.set_mesh(mesh):
        ablations = make_well_temperedness_step(model, ci_fn.fn.capture_keys, config, mesh)(
            model,
            components,
            PlacedCIFn(fn=ci_fn.fn, placement=rules.ci_fn),
            tokens,
            jax.random.PRNGKey(5),
        )
    assert all(value.sharding.is_fully_replicated for value in jax.tree.leaves(ablations))


def test_chunking_does_not_change_damage():
    two_per_forward = _run_glu_transformer(
        _well_temperedness_config(n_locations=2, n_components_per_region=3, ablations_per_forward=2)
    )
    six_per_forward = _run_glu_transformer(
        _well_temperedness_config(n_locations=2, n_components_per_region=3, ablations_per_forward=6)
    )
    _assert_damage_close(
        np.asarray(two_per_forward.damage), np.asarray(six_per_forward.damage), "chunking"
    )
    np.testing.assert_array_equal(two_per_forward.preactivations, six_per_forward.preactivations)
    np.testing.assert_array_equal(two_per_forward.site_indices, six_per_forward.site_indices)


def test_all_ones_reference_is_zero_for_dead_components():
    model, components, ci_fn, tokens = _setup_glu_transformer()
    zero_components = jax.tree.map(jnp.zeros_like, components)
    config = _well_temperedness_config(
        n_locations=2, n_components_per_region=2, ablations_per_forward=2
    )
    ablations, _, _ = _measure_ablations(model, zero_components, ci_fn, tokens, config)
    np.testing.assert_array_less(np.abs(ablations.damage), _DAMAGE_NOISE_FLOOR)


def test_selected_preactivations_components_and_regions_stay_joined():
    model, components, ci_fn, tokens = _setup_glu_transformer()
    config = _well_temperedness_config(
        n_locations=4, n_components_per_region=_C, ablations_per_forward=_C
    )
    ablations, batch_indices, position_indices = _measure_ablations(
        model, components, ci_fn, tokens, config
    )
    preactivations_at_locations, global_component_indices, selected_site_indices, _ = (
        _expected_component_selection(model, ci_fn, tokens, config, batch_indices, position_indices)
    )
    expected_preactivations = jnp.take_along_axis(
        preactivations_at_locations[None], global_component_indices, axis=-1
    )
    np.testing.assert_allclose(
        np.asarray(ablations.preactivations),
        np.asarray(expected_preactivations),
        rtol=0.01,
        atol=1e-3,
    )
    np.testing.assert_array_equal(
        np.asarray(ablations.site_indices), np.asarray(selected_site_indices)
    )

    region_has_samples = dict.fromkeys(REGIONS, False)
    for region_index, region in enumerate(REGIONS):
        for location in range(config.n_locations):
            global_components = np.asarray(global_component_indices[region_index, location])
            assert len(set(global_components.tolist())) == config.n_components_per_region
            selected_preactivations = np.asarray(ablations.preactivations[region_index, location])
            in_region_mask = np.asarray(in_region(selected_preactivations, region))
            region_has_samples[region] |= bool(in_region_mask.any())
    assert region_has_samples["below_zero"] and (
        region_has_samples["zero_to_one"] or region_has_samples["above_one"]
    )


def test_group_patterns_resolve_and_feed_the_log_surface():
    model, *_ = _setup_glu_transformer()
    first_site = model.site_names[0]
    site_groups = _resolve_groups(model.site_names, {"first": [first_site], "all": ["*"]})
    assert site_groups == {"first": (0,), "all": tuple(range(len(model.site_names)))}
    log_entries = well_temperedness_log_entries(_synthetic_ablations(model.site_names), site_groups)
    assert {name.split("/", 1)[0] for name in log_entries} == {"all_sites", "first", "all"}
    with pytest.raises(AssertionError, match="reserved"):
        _resolve_groups(model.site_names, {"all_sites": [first_site]})
    with pytest.raises(AssertionError, match="matches no sites"):
        _resolve_groups(model.site_names, {"missing": ["does.not.exist.*"]})


def test_invalid_sample_and_chunk_sizes_are_refused():
    model, *_ = _setup_glu_transformer()
    total_components = sum(site_spec.C for site_spec in model.sites)
    too_many_components = _well_temperedness_config(
        n_locations=2,
        n_components_per_region=total_components + 1,
        ablations_per_forward=1,
    )
    with pytest.raises(AssertionError, match="per region from"):
        make_well_temperedness_step(model, frozenset(), too_many_components)
    indivisible_ablation_count = _well_temperedness_config(
        n_locations=2, n_components_per_region=3, ablations_per_forward=5
    )
    with pytest.raises(AssertionError, match="ablations_per_forward"):
        make_well_temperedness_step(model, frozenset(), indivisible_ablation_count)


@pytest.mark.parametrize(
    ("damage", "expected"),
    [
        ([[9.0, 5.0, 1.0], [9.0, 5.0, 1.0]], 1.0),
        ([[1.0, 5.0, 9.0], [1.0, 5.0, 9.0]], 0.0),
        ([[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]], 0.5),
    ],
)
def test_fraction_well_ordered_pairs(damage: list[list[float]], expected: float):
    preactivations = np.array([[3.0, 2.0, 1.5], [3.0, 2.0, 1.5]])
    fraction = fraction_well_ordered_pairs(preactivations, np.asarray(damage), "above_one", None)
    assert fraction == pytest.approx(expected)


def test_locations_are_weighted_by_pair_count():
    preactivations = np.array([[3.0, 2.0, -1.0], [3.0, 2.0, 1.5]])
    damage = np.array([[3.0, 2.0, 0.0], [1.0, 2.0, 3.0]])
    assert fraction_well_ordered_pairs(preactivations, damage, "above_one", None) == pytest.approx(
        0.25
    )


def test_nonfinite_measurements_fail_instead_of_scoring_as_discordant():
    preactivations = np.array([[3.0, 2.0]])
    damage = np.array([[2.0, np.nan]])
    with pytest.raises(AssertionError, match="damage must be finite"):
        fraction_well_ordered_pairs(preactivations, damage, "above_one", None)


def test_padding_is_excluded():
    preactivations = np.array([[2.0, 1.5, -4.0, -9.0], [3.0, 1.2, -4.0, -9.0]])
    damage = np.array([[9.0, 5.0, 1.0, 0.1], [9.0, 5.0, 1.0, 0.1]])
    assert fraction_well_ordered_pairs(preactivations, damage, "above_one", None) == 1.0
    assert fraction_well_ordered_pairs(preactivations[:1], damage[:1], "above_one", None) == 1.0
    assert fraction_well_ordered_pairs(preactivations, damage, "zero_to_one", None) is None


def test_global_pairing_compares_components_from_different_sites():
    preactivations = np.array([[2.0, 1.5], [2.0, 1.5]])
    high_damage = np.array([[10.0, 9.0], [10.0, 9.0]])
    low_damage = np.array([[1.0, 0.0], [1.0, 0.0]])
    site_names = ("lower_preactivations_higher_damage", "higher_preactivations_lower_damage")
    combined_preactivations = np.concatenate((preactivations, preactivations + 2), axis=1)
    combined_damage = np.concatenate((high_damage, low_damage), axis=1)
    site_indices = np.broadcast_to(np.array([0, 0, 1, 1]), (len(REGIONS), 2, 4))
    ablations = Ablations(
        preactivations=np.stack(
            (-np.abs(combined_preactivations), combined_preactivations / 4, combined_preactivations)
        ),
        damage=np.stack((combined_damage, combined_damage, combined_damage)),
        site_indices=site_indices,
    )
    for site_name in site_names:
        fraction = _region_fractions(ablations, (site_names.index(site_name),))["above_one"]
        assert fraction == pytest.approx(1.0)

    assert _region_fractions(ablations, None)["above_one"] == pytest.approx(1 / 3)


def test_regions_partition_the_preactivation_axis():
    preactivations = np.array([[-2.0, 0.0, 0.5, 1.0, 3.0]])
    np.testing.assert_array_equal(
        sum(
            (in_region(preactivations, region) for region in REGIONS),
            start=np.zeros_like(preactivations, int),
        ),
        np.ones_like(preactivations, int),
    )


def test_log_surface_has_one_scalar_per_scope_and_region():
    model, *_ = _setup_glu_transformer()
    site_names = model.site_names[:2]
    ablations = _synthetic_ablations(site_names)
    log_entries = well_temperedness_log_entries(ablations, {"first": (0,)})
    assert set(log_entries) == {
        f"{scope}/fraction_well_ordered_pairs_{region}"
        for scope in ("all_sites", "first")
        for region in REGIONS
    }


def test_regions_without_pairs_are_absent_from_the_log_surface():
    model, *_ = _setup_glu_transformer()
    ablations = _synthetic_ablations(model.site_names)
    preactivations = np.array(ablations.preactivations, copy=True)
    preactivations[REGIONS.index("zero_to_one")] = 2.0
    sparse = Ablations(preactivations, ablations.damage, ablations.site_indices)

    log_entries = well_temperedness_log_entries(sparse, {})

    assert log_entries
    assert all("zero_to_one" not in name for name in log_entries)


def test_plot_returns_png():
    model, *_ = _setup_glu_transformer()
    png = _plot_preactivation_vs_damage(_synthetic_ablations(model.site_names))
    assert png.startswith(b"\x89PNG")


def test_choose_locations_covers_positioned_and_positionless_inputs():
    batch_indices, position_indices = _choose_locations(
        jax.random.PRNGKey(5), (_BATCH, _SEQ), _BATCH * _SEQ
    )
    np.testing.assert_array_equal(
        np.sort(np.asarray(batch_indices) * _SEQ + np.asarray(position_indices)),
        np.arange(_BATCH * _SEQ),
    )
    batch_indices, position_indices = _choose_locations(jax.random.PRNGKey(5), (_BATCH,), _BATCH)
    np.testing.assert_array_equal(np.sort(np.asarray(batch_indices)), np.arange(_BATCH))
    np.testing.assert_array_equal(np.asarray(position_indices), 0)
