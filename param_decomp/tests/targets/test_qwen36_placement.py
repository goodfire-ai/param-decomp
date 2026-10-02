"""The qwen36_moe Plan-A placement census, through the REAL forward.

The evidence bar (PLACEMENT_DESIGN.md): a declared resident placement is proven, not
assumed. These tests run the actual stage-scan forward — frozen and masked, with remat,
narrow component masks on the pinned routing, weight-delta masks, routes, expert-blocked
V/U, and both routed expert-parallel arms (frozen, and the production routed DECOMPOSED
execution) — on a simulated two-axis `(data, tp)` mesh, and check

- value and gradient parity against the unplaced execution (both expert arms);
- ZERO in-loop cross-`data` collectives in the compiled gradient module;
- the once-per-step masters→resident gather in entry, exit reductions present, and
  every surviving in-loop all-gather activation-shaped (no weight gather sank into a
  while body).
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from param_decomp.core.components import (
    ComponentStacks,
    SiteCI,
    init_component_stacks,
)
from param_decomp.core.configs import (
    PlacementPresetName,
    SequenceSharding,
)
from param_decomp.core.init_placed import init_component_stacks_placed
from param_decomp.core.model import (
    ForwardResult,
    MaterializedMasking,
    PlacedModel,
    StochasticMasking,
    faithfulness_weight_deltas,
)
from param_decomp.core.placement import component_stacks_shardings, from_config
from param_decomp.core.sharding import place_target, resident_abstract_mesh
from param_decomp.core.tools.hlo_census import collective_census
from param_decomp.lm.batch import LMBatch, LMBatchWithRouting
from param_decomp.targets.lm_output import LMOutput
from param_decomp.targets.qwen36_moe import (
    Qwen36MoeConfig,
    Qwen36MoeDecomposedModel,
    QwenPreparedMasking,
    QwenPreparedWeights,
    full_site_cs,
    qwen36_moe_site_specs,
    site_name,
)
from param_decomp.targets.testing import (
    constant_mask_values,
    materialized_logits,
    random_mask_values,
    site_masks,
    tiny_qwen36_cfg,
)
from param_decomp.targets.transformer_taps import resid_tap_key, site_output_tap_key
from param_decomp.tests.targets.qwen36_placement_helpers import (
    BATCH,
    CENSUS_CS,
    DATA,
    SEQ,
    TP,
    assert_no_weight_gather_in_any_loop,
    batch_placed,
    model_and_components,
    multidevice,
    placement_mesh,
)


def _masked_census_and_parity(
    model: Qwen36MoeDecomposedModel,
    capture_keys: frozenset[str],
    preset: PlacementPresetName = "zero1-replicated-resident-moe",
    sequence_sharding: SequenceSharding = "replicate",
):
    """The full masked forward (all six kinds decomposed, with routes, weight-delta
    masks, remat, and captures) placed on the (data, tp) mesh under a moe resident
    preset: values, captures, and V/U gradients must match the unplaced run, and the
    compiled gradient census must show zero in-loop cross-data collectives, entry-only
    weight movement, and exit reductions. Shared by both master flavors and sequence
    parallelism. Each arm's masked forward runs under ITS OWN clean forward's pinned
    routing (the placed one decided on the placed batch), which must agree."""
    mesh = placement_mesh()
    cfg, sites = model.cfg, model.sites
    components = init_component_stacks(sites, jax.random.PRNGKey(1))
    tokens = jax.random.randint(jax.random.PRNGKey(2), (BATCH, SEQ), 0, cfg.vocab_size)
    mask_values = random_mask_values(model, (BATCH, SEQ), jax.random.PRNGKey(3))
    deltas = {spec.name: jnp.full((BATCH, SEQ), 0.25) for spec in sites}
    routes = {spec.name: (jnp.arange(BATCH * SEQ).reshape(BATCH, SEQ) % 3 > 0) for spec in sites}

    def masked(
        target: PlacedModel[
            LMBatch,
            LMOutput,
            QwenPreparedWeights,
            LMBatchWithRouting[LMBatch],
            QwenPreparedMasking,
        ],
        prepared: QwenPreparedWeights,
        _batch: Array,
        component: dict[str, SiteCI],
        delta: dict[str, Array],
        route: dict[str, Array],
        conditioning: LMBatchWithRouting[LMBatch],
    ):
        return target.masked_forward(
            prepared,
            conditioning,
            masking=target.model.prepare_masking(
                MaterializedMasking(component_masks=component, weight_delta_masks=delta)
            ),
            routes=route,
            capture_keys=capture_keys,
            remat=True,
        )

    def loss_and_result(
        target: PlacedModel[
            LMBatch,
            LMOutput,
            QwenPreparedWeights,
            LMBatchWithRouting[LMBatch],
            QwenPreparedMasking,
        ],
        value: ComponentStacks,
        batch: Array,
        component: dict[str, SiteCI],
        delta: dict[str, Array],
        route: dict[str, Array],
        conditioning: LMBatchWithRouting[LMBatch],
    ) -> tuple[Array, ForwardResult[LMOutput, LMBatchWithRouting[LMBatch]]]:
        prepared = target.prepare_compute_weights(value)
        result = masked(target, prepared, batch, component, delta, route, conditioning)
        return jnp.sum(materialized_logits(result.output)), result

    # One program per arm carries the forward, its captures, and the V/U gradients; on
    # the placed arm the same executable also yields the census text, so nothing here
    # compiles twice.
    grads_and_result = jax.jit(jax.grad(loss_and_result, argnums=1, has_aux=True))
    unplaced = PlacedModel(model=model, placement=None)
    conditioning = unplaced.clean_forward(LMBatch(tokens)).conditioning
    masks = site_masks(model, conditioning.selection, mask_values)
    expected_grads, expected = grads_and_result(
        unplaced, components, tokens, masks, deltas, routes, conditioning
    )

    rules = from_config(preset, mesh, sites, sequence_sharding=sequence_sharding)
    placed_model = place_target(model, rules)
    placed_components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
    placed_tokens = batch_placed(tokens, mesh)
    placed_mask_values, placed_deltas, placed_routes = jax.tree.map(
        lambda value: batch_placed(value, mesh), (mask_values, deltas, routes)
    )

    with jax.set_mesh(mesh):
        placed_pinned = placed_model.clean_forward(LMBatch(placed_tokens)).conditioning
        np.testing.assert_array_equal(
            np.asarray(placed_pinned.selection.indices), np.asarray(conditioning.selection.indices)
        )
        placed_args = (
            placed_model,
            placed_components,
            placed_tokens,
            site_masks(model, placed_pinned.selection, placed_mask_values),
            placed_deltas,
            placed_routes,
            placed_pinned,
        )
        compiled = grads_and_result.lower(*placed_args).compile()
        hlo = compiled.as_text()
        got_grads, got = compiled(*placed_args)
    assert hlo is not None

    np.testing.assert_allclose(
        np.asarray(got.output), np.asarray(expected.output), rtol=2e-4, atol=2e-4
    )
    for key in sorted(capture_keys):
        np.testing.assert_allclose(
            np.asarray(got.captures[key]),
            np.asarray(expected.captures[key]),
            rtol=2e-4,
            atol=2e-4,
            err_msg=key,
        )
    for got_leaf, expected_leaf in zip(
        jax.tree.leaves(got_grads), jax.tree.leaves(expected_grads), strict=True
    ):
        # master grads pass through the bf16 compute cast in both arms; reassociation
        # across the tp splits then lands 1-2 bf16 ulps apart — and a small grad element
        # can inherit a whole ulp from an O(1) upstream intermediate, so the absolute
        # floor is one bf16 ulp at scale 1 (2⁻⁷), not one at the element's own scale.
        np.testing.assert_allclose(
            np.asarray(got_leaf), np.asarray(expected_leaf), rtol=2e-2, atol=2**-7
        )

    census = collective_census(hlo, replica_stride=TP, n_devices=DATA * TP)
    assert census.in_loop_cross_replicate == 0, census.counts
    assert census.exit_reductions > 0, census.counts
    # the once-per-step masters→resident entry gather crosses data
    assert census.counts.get("entry:all-gather[xrep]", 0) > 0, census.counts
    assert_no_weight_gather_in_any_loop(hlo, BATCH // DATA)


def _census_capture_keys(cfg: Qwen36MoeConfig, fused_taps: bool) -> frozenset[str]:
    keys = {
        resid_tap_key(0),
        resid_tap_key(cfg.n_layer),
        f"mlp_in.{cfg.n_layer - 1}",
        site_output_tap_key(site_name(cfg.n_layer - 2, "shared_up")),
        site_output_tap_key(site_name(cfg.n_layer - 1, "experts_down")),
    }
    if fused_taps:
        keys.add(site_output_tap_key(site_name(2, "experts_gate")))
    return frozenset(keys)


@multidevice
@pytest.mark.multidevice
def test_placed_routed_decomposed_census_and_parity():
    """The PRODUCTION masked forward — the routed decomposed arm on the expert-sharded
    schedule — placed vs unplaced, plus the census. Fused gate/up tap captures are
    excluded: the placed routed arms refuse them (pinned below)."""
    model, _components = model_and_components(CENSUS_CS)
    _masked_census_and_parity(model, _census_capture_keys(model.cfg, fused_taps=False))


@multidevice
@pytest.mark.multidevice
def test_placed_sequence_parallel_census_and_parity():
    """`sequence_sharding: sequence_parallel` — the masked scan carries the residual
    position-sharded over tp between blocks (block entries gather, block exits land
    sharded). A pure resharding of the same math: values and V/U gradients must match
    the unplaced run at the replicated arm's tolerances, with the residency census
    intact. Captures under sequence parallelism are an enumerated gap and refuse."""
    model, _components = model_and_components(CENSUS_CS)
    _masked_census_and_parity(model, frozenset(), sequence_sharding="sequence_parallel")

    mesh = placement_mesh()
    sites = model.sites
    rules = from_config(
        "zero1-replicated-resident-moe", mesh, sites, sequence_sharding="sequence_parallel"
    )
    assert rules.activations.masked_external.rule == {
        **rules.activations.external.rule,
        "position": ("tp",),
    }
    placed_model = place_target(model, rules)
    components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
    tokens = batch_placed(
        jax.random.randint(jax.random.PRNGKey(2), (BATCH, SEQ), 0, model.cfg.vocab_size), mesh
    )
    mask_values = jax.tree.map(
        lambda value: batch_placed(value, mesh),
        random_mask_values(model, (BATCH, SEQ), jax.random.PRNGKey(3)),
    )
    with jax.set_mesh(mesh):
        prepared = PlacedModel(model=placed_model.model, placement=rules).prepare_compute_weights(
            components
        )
        conditioning = placed_model.clean_forward(LMBatch(tokens)).conditioning
        with pytest.raises(AssertionError, match="capture under sequence parallelism"):
            placed_model.model.masked_forward(
                prepared,
                conditioning,
                masking=placed_model.model.prepare_masking(
                    MaterializedMasking(
                        component_masks=site_masks(model, conditioning.selection, mask_values),
                        weight_delta_masks=None,
                    )
                ),
                routes=None,
                placement=rules,
                capture_keys=frozenset({resid_tap_key(0)}),
                remat=True,
            )


def test_sequence_sharding_default_is_the_external_row():
    """`replicate`, `masked_external` IS the external row — object identity, so the
    audit and every compiled program are unchanged from the pre-arm spelling."""
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, CENSUS_CS))
    mesh = resident_abstract_mesh(DATA, TP)
    rules = from_config("zero1-replicated-resident-moe", mesh, sites)
    assert rules.activations.masked_external is rules.activations.external


@multidevice
@pytest.mark.multidevice
def test_placed_owner_moe_census_and_parity():
    """The owner master flavor through the production masked forward: same zero
    in-loop cross-data census and unplaced parity, with masters stack-cut
    (`{stack: data, expert: tp}` — whole V/U blocks per device)."""
    model, _components = model_and_components(CENSUS_CS)
    _masked_census_and_parity(
        model,
        _census_capture_keys(model.cfg, fused_taps=False),
        preset="owner-replicated-resident-moe",
    )


@multidevice
@pytest.mark.multidevice
def test_owner_moe_faithfulness_delta_path_is_data_local():
    """The owner flavor's headline faithfulness property: masters rest whole blocks per
    device, so the compiled delta path (masters → faithfulness weights → W − V·U)
    carries NO cross-`data` collective at all — against the zero1 flavor's pinned
    contrast, whose data-cut `C_block` contraction must land through one."""
    from param_decomp.core.model import faithfulness_weight_deltas
    from param_decomp.core.tools.hlo_census import _COLLECTIVE_OP, _spans_replicate

    mesh = placement_mesh()
    model, _components = model_and_components(CENSUS_CS)
    sites = model.sites

    def cross_data_collectives(preset: PlacementPresetName) -> int:
        rules = from_config(preset, mesh, sites)
        placed_model = place_target(model, rules)
        placed_components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
        with jax.set_mesh(mesh):
            fn = jax.jit(faithfulness_weight_deltas)
            hlo = fn.lower(placed_model, placed_components).compile().as_text()
        assert hlo is not None
        return sum(
            1
            for line in hlo.splitlines()
            if (m := _COLLECTIVE_OP.search(line)) is not None
            and _spans_replicate(line, m.group(1), TP, DATA * TP)
        )

    assert cross_data_collectives("owner-replicated-resident-moe") == 0
    assert cross_data_collectives("zero1-replicated-resident-moe") > 0


@multidevice
@pytest.mark.multidevice
def test_placed_moe_faithfulness_deltas_match_unplaced():
    """Both moe resident presets land the bf16 frozen stack on the delta row and convert
    it there: the placed deltas match the unplaced fp32 reference to fp32 reassociation
    (the sharded C / `C_block` contractions land through collectives), and with no
    components they ARE the frozen matrices, bit for bit — the landing moves bytes only."""
    mesh = placement_mesh()
    model, components = model_and_components(CENSUS_CS)
    model = eqx.tree_at(
        lambda m: m.moe, model, jax.tree.map(lambda a: a.astype(jnp.bfloat16), model.moe)
    )
    no_components = jax.tree.map(jnp.zeros_like, components)
    reference = model.weight_deltas(components)
    frozen_only = model.weight_deltas(no_components)
    for preset in ("owner-replicated-resident-moe", "zero1-replicated-resident-moe"):
        rules = from_config(preset, mesh, model.sites)
        placed_model = place_target(model, rules)
        with jax.set_mesh(mesh):
            deltas_fn = jax.jit(faithfulness_weight_deltas)
            placed = deltas_fn(
                placed_model,
                jax.device_put(components, component_stacks_shardings(components, rules)),
            )
            placed_frozen_only = deltas_fn(
                placed_model,
                jax.device_put(no_components, component_stacks_shardings(no_components, rules)),
            )
        for group in reference:
            np.testing.assert_allclose(
                jax.device_get(placed[group]),
                reference[group],
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"{preset} {group}",
            )
            np.testing.assert_array_equal(
                jax.device_get(placed_frozen_only[group]),
                frozen_only[group],
                err_msg=(preset, group),
            )


@multidevice
@pytest.mark.multidevice
def test_placed_routed_decomposed_refuses_fused_tap_captures():
    """The placed routed decomposed arm has no scattered full-width tap spelling; a
    fused gate/up tap capture must die loudly at trace, not silently materialize."""
    mesh = placement_mesh()
    model, _components = model_and_components(CENSUS_CS)
    rules = from_config("zero1-replicated-resident-moe", mesh, model.sites)
    placed_model = place_target(model, rules)
    placed_components = init_component_stacks_placed(model.sites, jax.random.PRNGKey(1), rules)
    tokens = batch_placed(
        jax.random.randint(jax.random.PRNGKey(2), (BATCH, SEQ), 0, model.cfg.vocab_size), mesh
    )
    mask_values = jax.tree.map(
        lambda value: batch_placed(value, mesh), constant_mask_values(model, (BATCH, SEQ), 1.0)
    )
    with jax.set_mesh(mesh):
        prepared = placed_model.prepare_compute_weights(placed_components)
        conditioning = placed_model.clean_forward(LMBatch(tokens)).conditioning
        masks = site_masks(model, conditioning.selection, mask_values)
        with pytest.raises(AssertionError, match="fused expert gate/up taps"):
            jax.jit(
                lambda m, p, b, c, r: m.masked_forward(
                    p,
                    r,
                    masking=m.model.prepare_masking(MaterializedMasking(component_masks=c)),
                    routes=None,
                    capture_keys=frozenset({site_output_tap_key(site_name(2, "experts_gate"))}),
                    remat=True,
                )
            )(placed_model, prepared, tokens, masks, conditioning)


@multidevice
@pytest.mark.multidevice
def test_placed_clean_forward_runs_the_expert_parallel_arm():
    """The placed clean forward (routed frozen experts, expert-parallel) matches the
    unplaced global-sort routed arm to reassociation tolerance and compiles with zero
    in-loop cross-data collectives."""
    mesh = placement_mesh()
    model, _components = model_and_components(CENSUS_CS)
    tokens = jax.random.randint(jax.random.PRNGKey(4), (BATCH, SEQ), 0, model.cfg.vocab_size)
    keys = frozenset({resid_tap_key(1), f"mlp_in.{model.cfg.n_layer - 1}"})

    expected = PlacedModel(model=model, placement=None).clean_forward(LMBatch(tokens), keys)

    rules = from_config("zero1-replicated-resident-moe", mesh, model.sites)
    placed_model = place_target(model, rules)
    placed_tokens = batch_placed(tokens, mesh)
    with jax.set_mesh(mesh):
        clean = jax.jit(lambda m, b: m.clean_forward(LMBatch(b), keys))
        got = clean(placed_model, placed_tokens)
        hlo = clean.lower(placed_model, placed_tokens).compile().as_text()
    assert hlo is not None
    # per-job expert matmuls are the same dot products, but the placed mixers split
    # their contractions over tp (partial sums + all-reduce), so f32 reassociation
    # compounds through the residual stream.
    np.testing.assert_allclose(
        np.asarray(got.output), np.asarray(expected.output), rtol=2e-4, atol=2e-4
    )
    for key in sorted(keys):
        np.testing.assert_allclose(
            np.asarray(got.captures[key]),
            np.asarray(expected.captures[key]),
            rtol=2e-4,
            atol=2e-4,
            err_msg=key,
        )
    census = collective_census(hlo, replica_stride=TP, n_devices=DATA * TP)
    assert census.in_loop_cross_replicate == 0, census.counts


@multidevice
@pytest.mark.multidevice
def test_placed_shared_only_masked_grads_flow_through_expert_parallel_arm():
    """Decomposing only the shared kinds leaves the expert arm routed INSIDE the placed
    masked forward (the dense resident preset — no expert-blocked group exists, so the
    moe preset correctly refuses): gradients flow through the expert-parallel custom
    VJPs under remat, match the unplaced run, and the census stays clean."""
    # its own (2, 2) mesh: the dense resident preset scatters delta d_in over
    # (tp, data), and the shared expert's intermediate width (12) tiles ÷4, not ÷8.
    devices = np.asarray(jax.devices()[:4]).reshape(2, 2)
    mesh = Mesh(devices, ("data", "tp"), axis_types=(AxisType.Explicit,) * 2)
    shared_cs = {kind: c for kind, c in CENSUS_CS.items() if kind.startswith("shared_")}
    model, components = model_and_components(shared_cs)
    sites = model.sites
    tokens = jax.random.randint(jax.random.PRNGKey(5), (BATCH, SEQ), 0, model.cfg.vocab_size)
    masks = {
        spec.name: jax.random.uniform(jax.random.PRNGKey(6), (BATCH, SEQ, spec.C)) for spec in sites
    }
    deltas = {spec.name: jnp.full((BATCH, SEQ), 0.5) for spec in sites}

    def loss(
        target: PlacedModel[
            LMBatch,
            LMOutput,
            QwenPreparedWeights,
            LMBatchWithRouting[LMBatch],
            QwenPreparedMasking,
        ],
        value: ComponentStacks,
        _batch: Array,
        component: dict[str, Array],
        delta: dict[str, Array],
        conditioning: LMBatchWithRouting[LMBatch],
    ) -> Array:
        out = target.masked_forward(
            target.prepare_compute_weights(value),
            conditioning,
            masking=target.model.prepare_masking(
                MaterializedMasking(component_masks=component, weight_delta_masks=delta)
            ),
            routes=None,
            remat=True,
        ).output
        return jnp.sum(materialized_logits(out))

    unplaced = PlacedModel(model=model, placement=None)
    expected_grads = jax.jit(jax.grad(loss, argnums=1))(
        unplaced,
        components,
        tokens,
        masks,
        deltas,
        unplaced.clean_forward(LMBatch(tokens)).conditioning,
    )

    with pytest.raises(AssertionError, match="name no semantic axis"):
        from_config("zero1-replicated-resident-moe", mesh, sites)
    rules = from_config("zero1-replicated-resident", mesh, sites)
    placed_model = place_target(model, rules)
    placed_components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
    placed_tokens = batch_placed(tokens, mesh)
    placed_masks, placed_deltas = jax.tree.map(
        lambda value: batch_placed(value, mesh), (masks, deltas)
    )
    with jax.set_mesh(mesh):
        placed_pinned = placed_model.clean_forward(LMBatch(placed_tokens)).conditioning
        grad_fn = jax.jit(jax.grad(loss, argnums=1))
        placed_args = (
            placed_model,
            placed_components,
            placed_tokens,
            placed_masks,
            placed_deltas,
            placed_pinned,
        )
        got_grads = grad_fn(*placed_args)
        hlo = grad_fn.lower(*placed_args).compile().as_text()
    assert hlo is not None
    for got_leaf, expected_leaf in zip(
        jax.tree.leaves(got_grads), jax.tree.leaves(expected_grads), strict=True
    ):
        # Both arms run bf16 compute, and the two routed spellings group the expert
        # matmuls differently (global sort vs per-(row, shard)): the compounded bf16
        # drift leaves a handful of ELEMENTS ~10% apart, so the bound is per-leaf
        # relative Frobenius error — a wrong or missing reduction lands O(1), not 1e-2.
        got_np, expected_np = np.asarray(got_leaf), np.asarray(expected_leaf)
        error = np.linalg.norm(got_np - expected_np) / np.linalg.norm(expected_np)
        assert error < 2e-2, error
    census = collective_census(hlo, replica_stride=2, n_devices=4)
    assert census.in_loop_cross_replicate == 0, census.counts


@multidevice
@pytest.mark.multidevice
@pytest.mark.parametrize("sequence_sharding", ["replicate", "sequence_parallel"])
def test_placed_stochastic_masked_forward_matches_unplaced(sequence_sharding: SequenceSharding):
    """The stochastic masked forward (masks rebuilt from the shared CI + draw keys
    INSIDE the checkpointed stage bodies) placed vs unplaced: threefry is counter-based,
    so the batch-sharded draws are value-identical and the V/U gradients match
    to bf16-cast tolerance — under both `sequence_sharding` arms (sequence parallelism
    is a resharding of the same draws)."""
    mesh = placement_mesh()
    model, components = model_and_components(CENSUS_CS)
    sites = model.sites
    tokens = jax.random.randint(jax.random.PRNGKey(7), (BATCH, SEQ), 0, model.cfg.vocab_size)
    ci_values = constant_mask_values(model, (BATCH, SEQ), 0.4)

    def loss(
        target: PlacedModel[
            LMBatch,
            LMOutput,
            QwenPreparedWeights,
            LMBatchWithRouting[LMBatch],
            QwenPreparedMasking,
        ],
        value: ComponentStacks,
        _batch: Array,
        ci_lower: dict[str, SiteCI],
        conditioning: LMBatchWithRouting[LMBatch],
    ) -> Array:
        masking = StochasticMasking(ci=ci_lower, draw_key=jax.random.PRNGKey(8))
        out = target.masked_forward(
            target.prepare_compute_weights(value),
            conditioning,
            masking=target.model.prepare_masking(masking),
            routes=None,
            remat=True,
        ).output
        return jnp.sum(materialized_logits(out))

    unplaced = PlacedModel(model=model, placement=None)
    conditioning = unplaced.clean_forward(LMBatch(tokens)).conditioning
    expected_grads = jax.jit(jax.grad(loss, argnums=1))(
        unplaced,
        components,
        tokens,
        site_masks(model, conditioning.selection, ci_values),
        conditioning,
    )

    rules = from_config(
        "zero1-replicated-resident-moe", mesh, sites, sequence_sharding=sequence_sharding
    )
    placed_model = place_target(model, rules)
    placed_components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
    placed_tokens = batch_placed(tokens, mesh)
    placed_ci_values = jax.tree.map(lambda value: batch_placed(value, mesh), ci_values)
    with jax.set_mesh(mesh):
        placed_pinned = placed_model.clean_forward(LMBatch(placed_tokens)).conditioning
        grad_fn = jax.jit(jax.grad(loss, argnums=1))
        placed_args = (
            placed_model,
            placed_components,
            placed_tokens,
            site_masks(model, placed_pinned.selection, placed_ci_values),
            placed_pinned,
        )
        got_grads = grad_fn(*placed_args)
        hlo = grad_fn.lower(*placed_args).compile().as_text()
    assert hlo is not None
    for got_leaf, expected_leaf in zip(
        jax.tree.leaves(got_grads), jax.tree.leaves(expected_grads), strict=True
    ):
        got_np, expected_np = np.asarray(got_leaf), np.asarray(expected_leaf)
        error = np.linalg.norm(got_np - expected_np) / np.linalg.norm(expected_np)
        assert error < 2e-2, error
    census = collective_census(hlo, replica_stride=TP, n_devices=DATA * TP)
    assert census.in_loop_cross_replicate == 0, census.counts
    assert census.exit_reductions > 0, census.counts
    assert_no_weight_gather_in_any_loop(hlo, BATCH // DATA)


@multidevice
@pytest.mark.multidevice
def test_placed_prepared_weights_and_frozen_leaves_follow_the_declared_rows():
    """Residency spot checks: the frozen leaves land on their declared rows (experts
    ÷tp expert-major, KV replicated, DeltaNet heads ÷tp) and the prepared compute
    stacks rest at the compute-weights row (expert ÷tp, dense C ÷tp)."""
    mesh = placement_mesh()
    model, _components = model_and_components(CENSUS_CS)
    rules = from_config("zero1-replicated-resident-moe", mesh, model.sites)
    placed_model = place_target(model, rules)
    target = placed_model.model
    assert isinstance(target, Qwen36MoeDecomposedModel)

    def assert_spec(value: Array, spec: P) -> None:
        assert isinstance(value.sharding, NamedSharding)
        assert value.sharding.is_equivalent_to(NamedSharding(mesh, spec), value.ndim), (
            value.sharding.spec,
            spec,
        )

    assert_spec(target.moe.experts_gate, P(None, "tp", None))
    assert_spec(target.moe.experts_down, P(None, None, "tp"))
    assert_spec(target.moe.router, P())
    assert_spec(target.attn.attn.wq, P(None, "tp", None))
    assert_spec(target.attn.attn.wk, P())
    assert_spec(target.deltanet.mixer.w_v, P(None, None, "tp", None))
    assert_spec(target.deltanet.mixer.conv_v, P(None, None, "tp", None))
    assert_spec(target.deltanet.mixer.norm_w, P())
    assert_spec(target.embed, P())

    placed_components = init_component_stacks_placed(model.sites, jax.random.PRNGKey(1), rules)
    with jax.set_mesh(mesh):
        prepared = jax.jit(lambda m, c: m.prepare_compute_weights(c))(
            placed_model, placed_components
        )
    assert_spec(prepared.per_kind["experts_gate"]["V"], P(None, "tp", None, None))
    assert_spec(prepared.per_kind["experts_gate"]["U"], P(None, "tp", None, None))
    assert_spec(prepared.per_kind["shared_gate"]["V"], P(None, None, "tp"))


@multidevice
@pytest.mark.multidevice
def test_placed_streamed_output_recon_census_and_parity():
    """The streamed output edge through the placed production forward: the recon KL's
    gradient over the factored package matches the unplaced MATERIALIZED edge (one
    check covering the edge flip and the placement), and the compiled module keeps
    zero in-loop cross-data collectives — the vocab-chunk scan is a while loop, so a
    stray cross-data reduction inside the streamed kernels would show here."""
    from param_decomp.core.recon import reconstruction_observations
    from param_decomp.targets.lm_output import StreamedOutputEdge

    mesh = placement_mesh()
    model, _components = model_and_components(CENSUS_CS)
    streamed_model = dataclasses.replace(model, output_edge=StreamedOutputEdge(n_vocab_chunks=4))
    sites = model.sites
    cfg = model.cfg
    tokens = jax.random.randint(jax.random.PRNGKey(11), (BATCH, SEQ), 0, cfg.vocab_size)
    masks = random_mask_values(model, (BATCH, SEQ), jax.random.PRNGKey(12))
    deltas = {spec.name: jnp.full((BATCH, SEQ), 0.25) for spec in sites}

    def loss(
        target: PlacedModel[
            LMBatch,
            LMOutput,
            QwenPreparedWeights,
            LMBatchWithRouting[LMBatch],
            QwenPreparedMasking,
        ],
        value: ComponentStacks,
        _batch: LMBatch,
        component: dict[str, Array],
        delta: dict[str, Array],
        loss_mesh: jax.sharding.Mesh | None,
    ) -> Array:
        clean = jax.tree.map(jax.lax.stop_gradient, target.clean_forward(_batch))
        clean_observed = reconstruction_observations(
            clean, target.pin_output_batch, capture_keys=frozenset(), mesh=loss_mesh
        )
        masked = target.masked_forward(
            target.prepare_compute_weights(value),
            clean.conditioning,
            masking=target.model.prepare_masking(
                MaterializedMasking(
                    component_masks=site_masks(model, clean.conditioning.selection, component),
                    weight_delta_masks=delta,
                )
            ),
            routes=None,
            remat=True,
        )
        masked_observed = reconstruction_observations(
            masked, target.pin_output_batch, capture_keys=frozenset(), mesh=loss_mesh
        )
        return target.recon_loss_fn(masked_observed.output, clean_observed.output)

    unplaced = PlacedModel(model=model, placement=None)
    components = init_component_stacks(sites, jax.random.PRNGKey(1))
    expected_loss, expected_grads = jax.jit(jax.value_and_grad(loss, argnums=1))(
        unplaced, components, LMBatch(tokens), masks, deltas, None
    )

    rules = from_config("zero1-replicated-resident-moe", mesh, sites)
    placed_model = place_target(streamed_model, rules)
    placed_components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
    placed_batch = LMBatch(batch_placed(tokens, mesh))
    placed_masks, placed_deltas = jax.tree.map(
        lambda value: batch_placed(value, mesh), (masks, deltas)
    )
    with jax.set_mesh(mesh):
        grad_fn = jax.jit(jax.value_and_grad(loss, argnums=1), static_argnums=(5,))
        got_loss, got_grads = grad_fn(
            placed_model, placed_components, placed_batch, placed_masks, placed_deltas, mesh
        )
        hlo = (
            grad_fn.lower(
                placed_model, placed_components, placed_batch, placed_masks, placed_deltas, mesh
            )
            .compile()
            .as_text()
        )
    assert hlo is not None

    np.testing.assert_allclose(
        np.asarray(got_loss), np.asarray(expected_loss), rtol=2e-4, atol=2e-4
    )
    for got_leaf, expected_leaf in zip(
        jax.tree.leaves(got_grads), jax.tree.leaves(expected_grads), strict=True
    ):
        got_np, expected_np = np.asarray(got_leaf), np.asarray(expected_leaf)
        denom = np.linalg.norm(expected_np)
        error = np.linalg.norm(got_np - expected_np) / (denom if denom > 0 else 1.0)
        assert error < 2e-2, error

    census = collective_census(hlo, replica_stride=TP, n_devices=DATA * TP)
    assert census.in_loop_cross_replicate == 0, census.counts
    assert census.exit_reductions > 0, census.counts
    assert census.counts.get("entry:all-gather[xrep]", 0) > 0, census.counts
    assert_no_weight_gather_in_any_loop(hlo, BATCH // DATA)
