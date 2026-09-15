"""The qwen36_moe Plan-A placement census, through the REAL forward.

The evidence bar (PLACEMENT_DESIGN.md): a declared resident placement is proven, not
assumed. These tests run the actual stage-scan forward — frozen and masked, with remat,
component masks, weight-delta masks, routes, expert-blocked V/U, and both routed
expert-parallel arms (frozen, and the production routed DECOMPOSED execution; the dense
oracle keeps its own placed cell) — on a simulated two-axis `(data, tp)` mesh, and check

- value and gradient parity against the unplaced execution (both expert arms);
- ZERO in-loop cross-`data` collectives in the compiled gradient module;
- the once-per-step masters→resident gather in entry, exit reductions present, and
  every surviving in-loop all-gather activation-shaped (no weight gather sank into a
  while body).
"""

import dataclasses
import re

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from param_decomp.core.components import ComponentStacks, init_component_stacks
from param_decomp.core.configs import PlacementPresetName, SequenceSharding
from param_decomp.core.init_placed import init_component_stacks_placed
from param_decomp.core.model import (
    ForwardResult,
    MaterializedMasking,
    PlacedModel,
    StochasticMasking,
    faithfulness_weight_deltas,
    prepare_compute_weights,
)
from param_decomp.core.placement import batch_axes, component_stacks_shardings, from_config
from param_decomp.core.sharding import place_target, resident_abstract_mesh
from param_decomp.core.tools.hlo_census import (
    _computations,
    _loop_computations,
    collective_census,
)
from param_decomp.targets.lm_output import LMOutput
from param_decomp.targets.qwen36_moe import (
    Qwen36MoeConfig,
    Qwen36MoeDecomposedModel,
    full_site_cs,
    qwen36_moe_site_specs,
    site_name,
)
from param_decomp.targets.testing import (
    materialized_logits,
    tiny_qwen36_cfg,
    tiny_qwen36_decomposed_model,
)
from param_decomp.targets.transformer_taps import resid_tap_key, site_output_tap_key

# data=4, tp=2 on the 8 simulated devices; every extent of the tiny config tiles it
# (4 experts ÷2, 2 q-heads ÷2, 2/4 DeltaNet k/v-heads ÷2, fused 32 ÷2, shared 12 ÷2),
# expert C_block=4 ÷data, dense C=8 ÷(tp·data).
DATA, TP = 4, 2
CENSUS_CS: dict[str, int] = {
    "experts_gate": 16,
    "experts_up": 16,
    "experts_down": 16,
    "shared_gate": 8,
    "shared_up": 8,
    "shared_down": 8,
}
BATCH, SEQ = 4, 8

multidevice = pytest.mark.skipif(len(jax.devices()) < 8, reason="requires eight local devices")


def _mesh() -> Mesh:
    devices = np.asarray(jax.devices()[: DATA * TP]).reshape(DATA, TP)
    return Mesh(devices, ("data", "tp"), axis_types=(AxisType.Explicit,) * 2)


def _model_and_components(
    c_of: dict[str, int],
) -> tuple[Qwen36MoeDecomposedModel, ComponentStacks]:
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, c_of))
    model = tiny_qwen36_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    return model, init_component_stacks(sites, jax.random.PRNGKey(1))


def _batch_placed(value: Array, mesh: Mesh) -> Array:
    spec = P(batch_axes(mesh), *(None for _ in value.shape[1:]))
    return jax.device_put(value, NamedSharding(mesh, spec))


def _assert_no_weight_gather_in_any_loop(hlo: str, batch_shard: int) -> None:
    """Every all-gather inside a while body must be activation-shaped (leading dim = the
    per-shard batch) — a weight shape leading with a stack/expert/matrix dim means
    GSPMD sank a weight gather into the loop."""
    comps = _computations(hlo)
    for name in _loop_computations(comps):
        for line in comps[name]:
            m = re.search(r"all-gather(?:-start)?\(", line)
            if m is None:
                continue
            for dims_text in re.findall(
                r"\b(?:pred|bf16|f16|f32|s32|u32)\[([\d,]+)\]", line[: m.start()]
            ):
                dims = tuple(int(d) for d in dims_text.split(","))
                assert len(dims) >= 2 and dims[0] == batch_shard, (
                    f"in-loop all-gather is not activation-shaped — a weight gather "
                    f"survived residency: {dims} in {line.strip()[:160]}"
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
    weight movement, and exit reductions. Shared by the routed (production) and dense
    (oracle) `ExpertsExecution` census tests, both master flavors, and sequence
    parallelism."""
    mesh = _mesh()
    cfg, sites = model.cfg, model.sites
    components = init_component_stacks(sites, jax.random.PRNGKey(1))
    tokens = jax.random.randint(jax.random.PRNGKey(2), (BATCH, SEQ), 0, cfg.vocab_size)
    masks = {
        spec.name: jax.random.uniform(jax.random.PRNGKey(3), (BATCH, SEQ, spec.C)) for spec in sites
    }
    deltas = {spec.name: jnp.full((BATCH, SEQ), 0.25) for spec in sites}
    routes = {spec.name: (jnp.arange(BATCH * SEQ).reshape(BATCH, SEQ) % 3 > 0) for spec in sites}

    def masked(
        target: PlacedModel[LMOutput],
        prepared: dict[str, dict[str, Array]],
        batch: Array,
        component: dict[str, Array],
        delta: dict[str, Array],
        route: dict[str, Array],
    ):
        return target.masked_forward(
            prepared,
            batch,
            masking=MaterializedMasking(
                component_masks=component, weight_delta_masks=delta, routes=route
            ),
            capture_keys=capture_keys,
            remat=True,
        )

    def loss_and_result(
        target: PlacedModel[LMOutput],
        value: ComponentStacks,
        batch: Array,
        component: dict[str, Array],
        delta: dict[str, Array],
        route: dict[str, Array],
    ) -> tuple[Array, ForwardResult[LMOutput]]:
        prepared = prepare_compute_weights(target, value)
        result = masked(target, prepared, batch, component, delta, route)
        return jnp.sum(materialized_logits(result.output)), result

    # One program per arm carries the forward, its captures, and the V/U gradients; on
    # the placed arm the same executable also yields the census text, so nothing here
    # compiles twice.
    grads_and_result = jax.jit(jax.grad(loss_and_result, argnums=1, has_aux=True))
    unplaced = PlacedModel(model=model, placement=None)
    expected_grads, expected = grads_and_result(unplaced, components, tokens, masks, deltas, routes)

    rules = from_config(preset, mesh, sites, sequence_sharding=sequence_sharding)
    placed_model = place_target(model, rules)
    placed_components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
    placed_tokens = _batch_placed(tokens, mesh)
    placed_masks, placed_deltas, placed_routes = jax.tree.map(
        lambda value: _batch_placed(value, mesh), (masks, deltas, routes)
    )

    with jax.set_mesh(mesh):
        placed_args = (
            placed_model,
            placed_components,
            placed_tokens,
            placed_masks,
            placed_deltas,
            placed_routes,
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
    _assert_no_weight_gather_in_any_loop(hlo, BATCH // DATA)


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
    model, _components = _model_and_components(CENSUS_CS)
    _masked_census_and_parity(model, _census_capture_keys(model.cfg, fused_taps=False))


@multidevice
@pytest.mark.multidevice
def test_placed_sequence_parallel_census_and_parity():
    """`sequence_sharding: sequence_parallel` — the masked scan carries the residual
    position-sharded over tp between blocks (block entries gather, block exits land
    sharded). A pure resharding of the same math: values and V/U gradients must match
    the unplaced run at the replicated arm's tolerances, with the residency census
    intact. Captures under sequence parallelism are an enumerated gap and refuse."""
    model, _components = _model_and_components(CENSUS_CS)
    _masked_census_and_parity(model, frozenset(), sequence_sharding="sequence_parallel")

    mesh = _mesh()
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
    tokens = _batch_placed(
        jax.random.randint(jax.random.PRNGKey(2), (BATCH, SEQ), 0, model.cfg.vocab_size), mesh
    )
    masks = {
        spec.name: _batch_placed(
            jax.random.uniform(jax.random.PRNGKey(3), (BATCH, SEQ, spec.C)), mesh
        )
        for spec in sites
    }
    with jax.set_mesh(mesh):
        prepared = prepare_compute_weights(
            PlacedModel(model=placed_model.model, placement=rules), components
        )
        with pytest.raises(AssertionError, match="capture under sequence parallelism"):
            placed_model.model.masked_forward(
                prepared,
                tokens,
                masking=MaterializedMasking(
                    component_masks=masks, weight_delta_masks=None, routes=None
                ),
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
    model, _components = _model_and_components(CENSUS_CS)
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

    mesh = _mesh()
    model, _components = _model_and_components(CENSUS_CS)
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
    mesh = _mesh()
    model, components = _model_and_components(CENSUS_CS)
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
def test_placed_dense_oracle_census_and_parity():
    """The `"dense"` ExpertsExecution oracle keeps its placed spelling green — the one
    placed arm that materializes the fused gate/up taps, so those captures stay."""
    model, _components = _model_and_components(CENSUS_CS)
    dense = dataclasses.replace(model, experts_execution="dense")
    _masked_census_and_parity(dense, _census_capture_keys(model.cfg, fused_taps=True))


@multidevice
@pytest.mark.multidevice
def test_placed_routed_decomposed_refuses_fused_tap_captures():
    """The placed routed decomposed arm has no scattered full-width tap spelling; a
    fused gate/up tap capture must die loudly at trace, not silently materialize."""
    mesh = _mesh()
    model, _components = _model_and_components(CENSUS_CS)
    rules = from_config("zero1-replicated-resident-moe", mesh, model.sites)
    placed_model = place_target(model, rules)
    placed_components = init_component_stacks_placed(model.sites, jax.random.PRNGKey(1), rules)
    tokens = _batch_placed(
        jax.random.randint(jax.random.PRNGKey(2), (BATCH, SEQ), 0, model.cfg.vocab_size), mesh
    )
    masks = {spec.name: _batch_placed(jnp.ones((BATCH, SEQ, spec.C)), mesh) for spec in model.sites}
    with jax.set_mesh(mesh):
        prepared = prepare_compute_weights(placed_model, placed_components)
        with pytest.raises(AssertionError, match="fused expert gate/up taps"):
            jax.jit(
                lambda m, p, b, c: m.masked_forward(
                    p,
                    b,
                    masking=MaterializedMasking(component_masks=c),
                    capture_keys=frozenset({site_output_tap_key(site_name(2, "experts_gate"))}),
                    remat=True,
                )
            )(placed_model, prepared, tokens, masks)


@multidevice
@pytest.mark.multidevice
def test_placed_clean_forward_runs_the_expert_parallel_arm():
    """The placed clean forward (routed frozen experts, expert-parallel) matches the
    unplaced global-sort routed arm to reassociation tolerance and compiles with zero
    in-loop cross-data collectives."""
    mesh = _mesh()
    model, _components = _model_and_components(CENSUS_CS)
    tokens = jax.random.randint(jax.random.PRNGKey(4), (BATCH, SEQ), 0, model.cfg.vocab_size)
    keys = frozenset({resid_tap_key(1), f"mlp_in.{model.cfg.n_layer - 1}"})

    expected = PlacedModel(model=model, placement=None).clean_forward(tokens, keys)

    rules = from_config("zero1-replicated-resident-moe", mesh, model.sites)
    placed_model = place_target(model, rules)
    placed_tokens = _batch_placed(tokens, mesh)
    with jax.set_mesh(mesh):
        clean = jax.jit(lambda m, b: m.clean_forward(b, keys))
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
    model, components = _model_and_components(shared_cs)
    sites = model.sites
    tokens = jax.random.randint(jax.random.PRNGKey(5), (BATCH, SEQ), 0, model.cfg.vocab_size)
    masks = {
        spec.name: jax.random.uniform(jax.random.PRNGKey(6), (BATCH, SEQ, spec.C)) for spec in sites
    }
    deltas = {spec.name: jnp.full((BATCH, SEQ), 0.5) for spec in sites}

    def loss(
        target: PlacedModel[LMOutput],
        value: ComponentStacks,
        batch: Array,
        component: dict[str, Array],
        delta: dict[str, Array],
    ) -> Array:
        out = target.masked_forward(
            prepare_compute_weights(target, value),
            batch,
            masking=MaterializedMasking(component_masks=component, weight_delta_masks=delta),
            remat=True,
        ).output
        return jnp.sum(materialized_logits(out))

    unplaced = PlacedModel(model=model, placement=None)
    expected_grads = jax.jit(jax.grad(loss, argnums=1))(unplaced, components, tokens, masks, deltas)

    with pytest.raises(AssertionError, match="name no semantic axis"):
        from_config("zero1-replicated-resident-moe", mesh, sites)
    rules = from_config("zero1-replicated-resident", mesh, sites)
    placed_model = place_target(model, rules)
    placed_components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
    placed_tokens = _batch_placed(tokens, mesh)
    placed_masks, placed_deltas = jax.tree.map(
        lambda value: _batch_placed(value, mesh), (masks, deltas)
    )
    with jax.set_mesh(mesh):
        grad_fn = jax.jit(jax.grad(loss, argnums=1))
        got_grads = grad_fn(
            placed_model, placed_components, placed_tokens, placed_masks, placed_deltas
        )
        hlo = (
            grad_fn.lower(
                placed_model, placed_components, placed_tokens, placed_masks, placed_deltas
            )
            .compile()
            .as_text()
        )
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
    mesh = _mesh()
    model, components = _model_and_components(CENSUS_CS)
    sites = model.sites
    tokens = jax.random.randint(jax.random.PRNGKey(7), (BATCH, SEQ), 0, model.cfg.vocab_size)
    ci = {spec.name: jnp.full((BATCH, SEQ, spec.C), 0.4) for spec in sites}

    def loss(
        target: PlacedModel[LMOutput],
        value: ComponentStacks,
        batch: Array,
        ci_lower: dict[str, Array],
    ) -> Array:
        masking = StochasticMasking(
            ci_stacked=target.stack_ci(ci_lower), draw_key=jax.random.PRNGKey(8), routes=None
        )
        out = target.masked_forward(
            prepare_compute_weights(target, value), batch, masking=masking, remat=True
        ).output
        return jnp.sum(materialized_logits(out))

    unplaced = PlacedModel(model=model, placement=None)
    expected_grads = jax.jit(jax.grad(loss, argnums=1))(unplaced, components, tokens, ci)

    rules = from_config(
        "zero1-replicated-resident-moe", mesh, sites, sequence_sharding=sequence_sharding
    )
    placed_model = place_target(model, rules)
    placed_components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
    placed_tokens = _batch_placed(tokens, mesh)
    placed_ci = jax.tree.map(lambda value: _batch_placed(value, mesh), ci)
    with jax.set_mesh(mesh):
        grad_fn = jax.jit(jax.grad(loss, argnums=1))
        got_grads = grad_fn(placed_model, placed_components, placed_tokens, placed_ci)
        hlo = (
            grad_fn.lower(placed_model, placed_components, placed_tokens, placed_ci)
            .compile()
            .as_text()
        )
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
    _assert_no_weight_gather_in_any_loop(hlo, BATCH // DATA)


@multidevice
@pytest.mark.multidevice
def test_placed_prepared_weights_and_frozen_leaves_follow_the_declared_rows():
    """Residency spot checks: the frozen leaves land on their declared rows (experts
    ÷tp expert-major, KV replicated, DeltaNet heads ÷tp) and the prepared compute
    stacks rest at the compute-weights row (expert ÷tp, dense C ÷tp)."""
    mesh = _mesh()
    model, _components = _model_and_components(CENSUS_CS)
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
        prepared = jax.jit(lambda m, c: prepare_compute_weights(m, c))(
            placed_model, placed_components
        )
    assert_spec(prepared["experts_gate"]["V"], P(None, "tp", None, None))
    assert_spec(prepared["experts_gate"]["U"], P(None, "tp", None, None))
    assert_spec(prepared["shared_gate"]["V"], P(None, None, "tp"))


@multidevice
@pytest.mark.multidevice
@pytest.mark.parametrize(
    ("preset", "optimizer_impl"),
    [
        ("zero1-replicated-resident-moe", "adamw"),
        ("owner-replicated-resident-moe", "stacked_muon"),
    ],
    ids=["zero1-adamw", "owner-muon"],
)
def test_placed_full_train_step_runs_with_faithfulness(
    preset: PlacementPresetName, optimizer_impl: str
):
    """The REAL train step (make_train_step: faithfulness + importance-minimality +
    stochastic recon, chunkwise CI) placed at each moe resident preset with its paired
    components optimizer — the whole training path, including the master-layout
    V·U faithfulness contraction and, on the owner flavor, stacked Muon's NS staging
    from the stack-cut masters."""
    import equinox as eqx
    import optax

    from param_decomp.core.ci_fn import (
        Chunk,
        ChunkwiseTransformerCIArch,
        MHACIAttention,
        resolve_ci_placement,
    )
    from param_decomp.core.components import group_factorizations
    from param_decomp.core.configs import (
        FaithfulnessLossConfig,
        ImportanceMinimalityLossConfig,
        StochasticReconLossConfig,
    )
    from param_decomp.core.faithfulness import faithfulness_loss_for
    from param_decomp.core.init_placed import init_ci_fn_placed
    from param_decomp.core.muon_stacked import stacked_muon
    from param_decomp.core.objective import build_objective
    from param_decomp.core.placement import (
        assert_stacked_muon_component_staging,
        ns_staging_sharding,
    )
    from param_decomp.core.run_state import component_muon_dimension_numbers
    from param_decomp.core.schedule import Knot, ScheduleConfig
    from param_decomp.core.train import (
        Decomposition,
        ForwardSubstrate,
        TrainingItem,
        TrainState,
        make_train_step,
    )

    mesh = _mesh()
    model, _components = _model_and_components(CENSUS_CS)
    sites = model.sites
    rules = from_config(preset, mesh, sites)
    placed_model = place_target(model, rules)

    match optimizer_impl:
        case "adamw":
            opt_vu = optax.adamw(1e-3, weight_decay=0.0)
        case "stacked_muon":
            # run_state.build_optimizers' plumbing exactly: the pairing claim, then the
            # ns_compute waypoint for every muon leaf.
            assert_stacked_muon_component_staging(rules)
            waypoint = ns_staging_sharding(rules.components.ns_compute, mesh)
            opt_vu = stacked_muon(
                1e-3,
                beta=0.95,
                weight_decay=0.0,
                consistent_rms=0.2,
                muon_weight_dimension_numbers=component_muon_dimension_numbers(
                    group_factorizations(sites)
                ),
                ns_steps=5,
                ns_dtype=jnp.dtype(jnp.float32),
                waypoints=lambda tree: jax.tree.map(lambda _: waypoint, tree),
            )
        case _:
            raise AssertionError(optimizer_impl)

    arch = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(resid_tap_key(0),), output_sites=model.site_names),),
        input_dim=model.cfg.n_embd,
        d_model=8,
        n_blocks=1,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=16,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )
    objective = build_objective(
        (
            FaithfulnessLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(
                coeff=1e-4,
                gamma=ScheduleConfig(
                    max_val=1.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.5))
                ),
            ),
            StochasticReconLossConfig(coeff=1.0),
        ),
        model.site_names,
    )
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)

    with jax.set_mesh(mesh):
        components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
        ci_fn = init_ci_fn_placed(arch, sites, jax.random.PRNGKey(2), mesh, rules)
        state = TrainState(
            decomposition=Decomposition(components=components, ci_fn=ci_fn),
            training=TrainingItem(
                components_opt_state=opt_vu.init(eqx.filter(components, eqx.is_array)),
                ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
                adversaries={},
                freq_ema=None,
                step=jnp.zeros((), jnp.int32),
            ),
        )
        step_fn = make_train_step(
            model_static=placed_model,
            substrate=ForwardSubstrate.of(
                placed_model,
                remat_recon_forwards=True,
                remat_ci_fn=True,
                ci_capture_keys=frozenset({resid_tap_key(0)}),
                ci_placement=resolve_ci_placement(arch, rules),
            ),
            objective=objective,
            components_optimizer=opt_vu,
            ci_fn_optimizer=opt_ci,
            total_steps=4,
            faithfulness=faithfulness_loss_for(placed_model),
        )
        tokens = _batch_placed(
            jax.random.randint(jax.random.PRNGKey(3), (BATCH, SEQ), 0, model.cfg.vocab_size), mesh
        )
        before = np.asarray(components.stacks["experts_gate"][0])
        for step_index in range(2):
            state, metrics = step_fn(
                placed_model, state, tokens, jax.random.fold_in(jax.random.PRNGKey(4), step_index)
            )
            assert jnp.isfinite(metrics["total"]), (step_index, metrics["total"])
        moved = np.asarray(state.decomposition.components.stacks["experts_gate"][0])
    assert not np.allclose(moved, before), "V did not move — the step is a no-op"


# ── the MoE CI fn placed: parity, census, staging claims ─────────────────────


def _moe_ci_arch(cfg: Qwen36MoeConfig):
    """One chunk per stage, expert kinds narrow, shared kinds full — dims tiling the
    (data=4, tp=2) mesh: expert_ffn_hidden ÷data, C_block=4 ÷data, q-heads ÷tp."""
    from param_decomp.core.ci_fn import (
        FullSlot,
        MHACIAttention,
        MoEChunk,
        MoEChunkwiseTransformerCIArch,
        NarrowSlot,
        RoutingTap,
    )
    from param_decomp.targets.qwen36_moe import (
        is_expert_kind,
        router_idx_tap_key,
        router_weights_tap_key,
    )

    interval = cfg.full_attention_interval
    chunks = []
    for start in range(0, cfg.n_layer, interval):
        layers = tuple(range(start, start + interval))
        slots = []
        for router, layer in enumerate(layers):
            for kind in CENSUS_CS:
                site = site_name(layer, kind)
                slots.append(
                    NarrowSlot(site=site, router=router)
                    if is_expert_kind(kind)
                    else FullSlot(site=site)
                )
        chunks.append(
            MoEChunk(
                input_taps=(resid_tap_key(start),),
                routing=tuple(
                    RoutingTap(
                        ids_key=router_idx_tap_key(layer),
                        weights_key=router_weights_tap_key(layer),
                    )
                    for layer in layers
                ),
                slots=tuple(slots),
            )
        )
    return MoEChunkwiseTransformerCIArch(
        chunks=tuple(chunks),
        input_dim=cfg.n_embd + interval * cfg.n_experts,
        d_model=8,
        n_blocks=2,
        attention=MHACIAttention(n_heads=2),
        n_experts=cfg.n_experts,
        expert_ffn_hidden=8,
        shared_ffn_hidden=8,
        learned_norm_scale=False,
        grouped_matmul_backend="ragged_dot",
    )


@multidevice
@pytest.mark.multidevice
def test_placed_moe_ci_fn_census_and_parity():
    """The MoE chunkwise CI fn placed at the moe resident preset: the placed CI values
    (narrow bundles included) and CI-weight gradients match the unplaced run, and the
    compiled forward+backward census pins ZERO in-loop cross-data collectives through
    the CI MoE blocks — the narrow combine's all-reduce rides tp, an activation
    collective; the masters→resident entry gather is the only data crossing."""
    import equinox as eqx

    from param_decomp.core.ci_fn import (
        MoEChunkwiseTransformerCIFn,
        PlacedCIFn,
        build_ci_fn,
        evaluate_ci,
        resolve_ci_placement,
    )
    from param_decomp.core.components import NarrowCI, site_ci_values
    from param_decomp.core.init_placed import init_ci_fn_placed

    mesh = _mesh()
    model, _components = _model_and_components(CENSUS_CS)
    arch = _moe_ci_arch(model.cfg)
    rules = from_config("zero1-replicated-resident-moe", mesh, model.sites)
    ci_placement = resolve_ci_placement(arch, rules)
    assert ci_placement is not None and ci_placement.moe is not None

    seed = jax.random.PRNGKey(5)
    unplaced_fn = build_ci_fn(arch, model.sites, seed)
    assert isinstance(unplaced_fn, MoEChunkwiseTransformerCIFn)
    tokens = jax.random.randint(jax.random.PRNGKey(6), (BATCH, SEQ), 0, model.cfg.vocab_size)
    unplaced_model = PlacedModel(model=model, placement=None)
    taps = dict(unplaced_model.clean_forward(tokens, unplaced_fn.capture_keys).captures)

    expected_ci = evaluate_ci(PlacedCIFn(fn=unplaced_fn, placement=None), taps, remat=False)

    def unplaced_total(fn: MoEChunkwiseTransformerCIFn) -> Array:
        ci = evaluate_ci(PlacedCIFn(fn=fn, placement=None), taps, remat=True)
        # raw preactivations, not a squashing: the squashings' piecewise-derivative
        # regions flip under bf16 divergence, amplifying reassociation into O(1)
        # gradient noise at these tiny widths
        return sum(
            (
                jnp.sum(site_ci_values(v).astype(jnp.float32) ** 2)
                for v in ci.preactivations.values()
            ),
            start=jnp.zeros((), jnp.float32),
        )

    expected_total, expected_grads = eqx.filter_value_and_grad(unplaced_total)(unplaced_fn)

    with jax.set_mesh(mesh):
        placed_fn = init_ci_fn_placed(arch, model.sites, seed, mesh, rules)
        assert isinstance(placed_fn, MoEChunkwiseTransformerCIFn)
        placed_model = place_target(model, rules)
        placed_tokens = _batch_placed(tokens, mesh)
        placed_taps = dict(
            placed_model.clean_forward(placed_tokens, placed_fn.capture_keys).captures
        )

        def placed_total(fn: MoEChunkwiseTransformerCIFn) -> Array:
            ci = evaluate_ci(PlacedCIFn(fn=fn, placement=ci_placement), placed_taps, remat=True)
            return sum(
                (
                    jnp.sum(site_ci_values(v).astype(jnp.float32) ** 2)
                    for v in ci.preactivations.values()
                ),
                start=jnp.zeros((), jnp.float32),
            )

        placed_ci = jax.jit(
            lambda fn, t: evaluate_ci(PlacedCIFn(fn=fn, placement=ci_placement), t, remat=False)
        )(placed_fn, placed_taps)
        grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(placed_total))
        got_total, got_grads = grad_fn(placed_fn)
        hlo = (
            jax.jit(lambda arrays: eqx.filter_value_and_grad(placed_total)(arrays))
            .lower(placed_fn)
            .compile()
            .as_text()
        )
    assert hlo is not None

    def assert_close_frobenius(got: np.ndarray, expected: np.ndarray, name: str) -> None:
        # bf16 compute at tiny widths: the tp-split matmuls, the EP job spelling, and
        # the auto-axes attention arm all reassociate, and the divergence compounds
        # through the blocks — pointwise ulp bounds don't hold, the norm-level one does
        # (the grad comparison below uses the same criterion). A 12-wide residual puts
        # the shared-expert values' reassociation floor at ~2e-2; 3e-2 keeps headroom.
        denom = np.linalg.norm(expected)
        error = np.linalg.norm(got - expected) / (denom if denom > 0 else 1.0)
        assert error < 3e-2, (name, error)

    for name in unplaced_fn.output_names:
        expected_value, got_value = expected_ci.lower[name], placed_ci.lower[name]
        match expected_value:
            case NarrowCI():
                assert isinstance(got_value, NarrowCI)
                np.testing.assert_array_equal(
                    np.asarray(got_value.router_indices), np.asarray(expected_value.router_indices)
                )
                assert_close_frobenius(
                    np.asarray(got_value.values), np.asarray(expected_value.values), name
                )
            case _:
                assert_close_frobenius(np.asarray(got_value), np.asarray(expected_value), name)
    np.testing.assert_allclose(np.asarray(got_total), np.asarray(expected_total), rtol=2e-3)
    for got_leaf, expected_leaf in zip(
        jax.tree.leaves(eqx.filter(got_grads, eqx.is_array)),
        jax.tree.leaves(eqx.filter(expected_grads, eqx.is_array)),
        strict=True,
    ):
        got_np, expected_np = np.asarray(got_leaf), np.asarray(expected_leaf)
        denom = np.linalg.norm(expected_np)
        error = np.linalg.norm(got_np - expected_np) / (denom if denom > 0 else 1.0)
        # bf16 silu·up products at width 8 compound reassociation a little past the
        # masked-forward suite's 2e-2; the bound keeps headroom over the observed 2.1e-2
        assert error < 4e-2, error

    census = collective_census(hlo, replica_stride=TP, n_devices=DATA * TP)
    # The one sanctioned in-loop cross-data collective: the replicated-persisted
    # bias/norm-vector grads' whole-batch sums (the dense chunkwise residency test's
    # carve-out, test_tp_boundary_topology). The byte bound keeps a matrix grad from
    # hiding behind it; the expert banks, heads, and every matrix defer to the entry
    # reductions, and the narrow combine's all-reduce rides tp, not data.
    assert census.in_loop_cross_replicate <= 1, census.counts
    assert all(size <= 2**12 for size in census.in_loop_cross_replicate_bytes), (
        census.in_loop_cross_replicate_bytes
    )
    assert census.counts.get("entry:all-gather[xrep]", 0) > 0, census.counts
    _assert_no_weight_gather_in_any_loop(hlo, BATCH // DATA)


def test_stacked_muon_moe_ci_staging_claims():
    """The MoE CI staging claim: dense families tile `n_chunks`, expert families the
    canonical fold `n_chunks·E` — a non-tiling chunk count refuses with the remedy."""
    from jax.sharding import AbstractMesh

    from param_decomp.core.placement import (
        CIFnPlacement,
        StackCensus,
        assert_stacked_muon_moe_ci_staging,
    )

    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, CENSUS_CS))
    mesh = AbstractMesh((2, 2), ("data", "tp"))
    rules = from_config("zero1-replicated-resident-moe", mesh, sites)
    # the moe CI rows rest intra-matrix, so no chunk-stack pad resolves under them

    def ci_placement(n_chunks: int) -> CIFnPlacement:
        return CIFnPlacement.resolved(rules.ci_fn, StackCensus(stack_len=n_chunks, stack_pad=0))

    assert_stacked_muon_moe_ci_staging(ci_placement(2), n_experts=cfg.n_experts)
    with pytest.raises(AssertionError, match="do not tile"):
        assert_stacked_muon_moe_ci_staging(ci_placement(3), n_experts=cfg.n_experts)


@multidevice
@pytest.mark.multidevice
def test_placed_full_train_step_runs_with_the_moe_ci_fn():
    """The REAL train step with the MoE chunkwise CI fn — narrow emission through
    smooth-L0 imp-min (the no-[C]-accumulator lp path), stochastic recon whose narrow
    masks drive the routed decomposed expert arm, and faithfulness — placed at the moe
    resident preset: two steps, finite, V and the CI expert banks both move."""
    import equinox as eqx
    import optax

    from param_decomp.core.ci_fn import MoEChunkwiseTransformerCIFn, resolve_ci_placement
    from param_decomp.core.configs import (
        FaithfulnessLossConfig,
        ImportanceMinimalityLossConfig,
        StochasticReconLossConfig,
    )
    from param_decomp.core.faithfulness import faithfulness_loss_for
    from param_decomp.core.init_placed import init_ci_fn_placed
    from param_decomp.core.objective import build_objective
    from param_decomp.core.schedule import Knot, ScheduleConfig
    from param_decomp.core.train import (
        Decomposition,
        ForwardSubstrate,
        TrainingItem,
        TrainState,
        make_train_step,
    )

    mesh = _mesh()
    model, _components = _model_and_components(CENSUS_CS)
    sites = model.sites
    rules = from_config("zero1-replicated-resident-moe", mesh, sites)
    placed_model = place_target(model, rules)
    arch = _moe_ci_arch(model.cfg)
    objective = build_objective(
        (
            FaithfulnessLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(
                coeff=1e-4,
                gamma=ScheduleConfig(
                    max_val=1.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.01))
                ),
            ),
            StochasticReconLossConfig(coeff=1.0),
        ),
        model.site_names,
    )
    opt_vu = optax.adamw(1e-3, weight_decay=0.0)
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)

    with jax.set_mesh(mesh):
        components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
        ci_fn = init_ci_fn_placed(arch, sites, jax.random.PRNGKey(2), mesh, rules)
        assert isinstance(ci_fn, MoEChunkwiseTransformerCIFn)
        state = TrainState(
            decomposition=Decomposition(components=components, ci_fn=ci_fn),
            training=TrainingItem(
                components_opt_state=opt_vu.init(eqx.filter(components, eqx.is_array)),
                ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
                adversaries={},
                freq_ema=None,
                step=jnp.zeros((), jnp.int32),
            ),
        )
        step_fn = make_train_step(
            model_static=placed_model,
            substrate=ForwardSubstrate.of(
                placed_model,
                remat_recon_forwards=True,
                remat_ci_fn=True,
                ci_capture_keys=ci_fn.capture_keys,
                ci_placement=resolve_ci_placement(arch, rules),
            ),
            objective=objective,
            components_optimizer=opt_vu,
            ci_fn_optimizer=opt_ci,
            total_steps=4,
            faithfulness=faithfulness_loss_for(placed_model),
        )
        tokens = _batch_placed(
            jax.random.randint(jax.random.PRNGKey(3), (BATCH, SEQ), 0, model.cfg.vocab_size), mesh
        )
        v_before = np.asarray(components.stacks["experts_gate"][0])
        bank_before = np.asarray(ci_fn.chunks.blocks[0].expert_gate[0])
        for step_index in range(2):
            state, metrics = step_fn(
                placed_model, state, tokens, jax.random.fold_in(jax.random.PRNGKey(4), step_index)
            )
            assert jnp.isfinite(metrics["total"]), (step_index, metrics["total"])
        v_moved = np.asarray(state.decomposition.components.stacks["experts_gate"][0])
        ci_moved = state.decomposition.ci_fn
        assert isinstance(ci_moved, MoEChunkwiseTransformerCIFn)
        bank_moved = np.asarray(ci_moved.chunks.blocks[0].expert_gate[0])
    assert not np.allclose(v_moved, v_before), "V did not move — the step is a no-op"
    assert not np.allclose(bank_moved, bank_before), "the CI expert bank did not move"


@multidevice
@pytest.mark.multidevice
def test_placed_train_step_with_bsc_sources_census_and_slot_locality():
    """The large-capacity adversary placed: persistent-PGD `bsc` sources (per-datapoint
    slots, uint16 fixed-point values with stochastic-rounding stores, SRC_STEP
    `momentum_sgd` with its bf16 velocity) through the REAL train step with narrow
    emission on the owner moe preset. Pins the double sharding {batch: data, expert: tp}
    on the stored sources, per-slot update parity against the unplaced step (each batch
    slot ascends by ITS datapoint's gradient — the bsc semantics), representation +
    sharding preserved through the step, and the compiled step's census: nothing
    tensor-sized crosses `data` inside any loop — the warmup scan's global-mean loss
    scalars and the CI bias/norm-vector grads are the only sanctioned in-loop
    cross-data reductions. The `replicate` arm only: sequence parallelism's placement is
    pinned by its own census and stochastic-parity tests, and the full train step is the
    most expensive program in the suite."""
    import equinox as eqx
    import optax

    from param_decomp.core.adversary import (
        ExpertBlockedSource as EBS,
    )
    from param_decomp.core.adversary import (
        PersistentAdversary,
        SourceComponents,
        init_persistent_sources,
        init_sources_opt_state,
    )
    from param_decomp.core.ci_fn import (
        MoEChunkwiseTransformerCIFn,
        build_ci_fn,
        resolve_ci_placement,
    )
    from param_decomp.core.components import ExpertBlocked
    from param_decomp.core.configs import (
        FaithfulnessLossConfig,
        ImportanceMinimalityLossConfig,
        MomentumSgdPGDConfig,
        PersistentPGDReconLossConfig,
        StochasticReconLossConfig,
    )
    from param_decomp.core.faithfulness import faithfulness_loss_for
    from param_decomp.core.init_placed import init_ci_fn_placed, init_sources_sharded
    from param_decomp.core.model import Positioned
    from param_decomp.core.objective import build_objective
    from param_decomp.core.schedule import Knot, ScheduleConfig
    from param_decomp.core.train import (
        Decomposition,
        ForwardSubstrate,
        TrainingItem,
        TrainState,
        make_train_step,
    )

    ppgd = PersistentPGDReconLossConfig(
        coeff=0.5,
        n_warmup_steps=1,
        source_shape="bsc",
        source_dtype="uint16",
        optimizer=MomentumSgdPGDConfig(momentum=0.9, lr_schedule=ScheduleConfig.constant(0.05)),
    )
    objective_cfgs = (
        FaithfulnessLossConfig(coeff=1.0),
        ImportanceMinimalityLossConfig(
            coeff=1e-4,
            gamma=ScheduleConfig(
                max_val=1.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.01))
            ),
        ),
        StochasticReconLossConfig(coeff=1.0),
        ppgd,
    )
    model, _components = _model_and_components(CENSUS_CS)
    sites = model.sites
    arch = _moe_ci_arch(model.cfg)
    objective = build_objective(objective_cfgs, model.site_names)
    opt_vu = optax.adamw(1e-3, weight_decay=0.0)
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    tokens = jax.random.randint(jax.random.PRNGKey(3), (BATCH, SEQ), 0, model.cfg.vocab_size)
    src_key = jax.random.PRNGKey(7)
    step_keys = [jax.random.fold_in(jax.random.PRNGKey(4), i) for i in range(2)]

    def run_two_steps(placed: bool) -> TrainState:
        mesh = _mesh() if placed else None
        if placed:
            rules = from_config("owner-replicated-resident-moe", _mesh(), sites)
            target = place_target(model, rules)
        else:
            rules = None
            target = PlacedModel(model=model, placement=None)

        def build_and_step() -> TrainState:
            if rules is not None:
                components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
                ci_fn = init_ci_fn_placed(arch, sites, jax.random.PRNGKey(2), _mesh(), rules)
                sources = init_sources_sharded(
                    sites, Positioned(SEQ), "bsc", BATCH, jnp.uint16, src_key, _mesh()
                )
                ci_placement = resolve_ci_placement(arch, rules)
                batch = _batch_placed(tokens, _mesh())
            else:
                components = init_component_stacks(sites, jax.random.PRNGKey(1))
                ci_fn = build_ci_fn(arch, sites, jax.random.PRNGKey(2))
                sources = init_persistent_sources(sites, (BATCH, SEQ), jnp.uint16, src_key)
                ci_placement = None
                batch = tokens
            assert isinstance(ci_fn, MoEChunkwiseTransformerCIFn)
            state = TrainState(
                decomposition=Decomposition(components=components, ci_fn=ci_fn),
                training=TrainingItem(
                    components_opt_state=opt_vu.init(eqx.filter(components, eqx.is_array)),
                    ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
                    adversaries={
                        ppgd.type: PersistentAdversary(
                            sources=sources,
                            opt_state=init_sources_opt_state(ppgd.optimizer, sources),
                            state_key=ppgd.type,
                            optimizer=ppgd.optimizer,
                            n_warmup=ppgd.n_warmup_steps,
                        )
                    },
                    freq_ema=None,
                    step=jnp.zeros((), jnp.int32),
                ),
            )
            step_fn = make_train_step(
                model_static=target,
                substrate=ForwardSubstrate.of(
                    target,
                    remat_recon_forwards=True,
                    remat_ci_fn=True,
                    ci_capture_keys=ci_fn.capture_keys,
                    ci_placement=ci_placement,
                ),
                objective=objective,
                components_optimizer=opt_vu,
                ci_fn_optimizer=opt_ci,
                total_steps=4,
                faithfulness=faithfulness_loss_for(target),
            )
            run = step_fn
            if rules is not None:
                # the census: the compiled step's in-loop collectives — an OUTER plain
                # jit (the engine's eqx jit inlines under it) so the lowered text is
                # reachable, the fit check's own AOT pattern. The same executable then
                # runs the steps: the placed step is compiled exactly once.
                outer = jax.jit(lambda m, s, b, k: step_fn(m, s, b, k))
                compiled = outer.lower(target, state, batch, step_keys[0]).compile()
                hlo = compiled.as_text()
                assert hlo is not None
                census = collective_census(hlo, replica_stride=TP, n_devices=DATA * TP)
                assert all(size <= 2**12 for size in census.in_loop_cross_replicate_bytes), (
                    census.in_loop_cross_replicate_bytes,
                    census.counts,
                )
                _assert_no_weight_gather_in_any_loop(hlo, BATCH // DATA)
                run = compiled
            for key in step_keys:
                state, metrics = run(target, state, batch, key)
                assert jnp.isfinite(metrics["total"]), metrics["total"]
            return state

        if mesh is not None:
            with jax.set_mesh(mesh):
                return build_and_step()
        return build_and_step()

    expert_site = next(s.name for s in sites if isinstance(s.factorization, ExpertBlocked))
    dense_site = next(s.name for s in sites if not isinstance(s.factorization, ExpertBlocked))

    placed_state = run_two_steps(placed=True)
    placed_stacks = placed_state.training.adversaries[ppgd.type].sources
    mesh = _mesh()
    expert_group, _ = placed_stacks.slot_of(expert_site)
    dense_group, _ = placed_stacks.slot_of(dense_site)
    expert_values = placed_stacks.stacks[expert_group].components
    assert isinstance(expert_values, EBS)
    assert expert_values.values.dtype == jnp.uint16
    # the double sharding on the STORED stacks, preserved through the step: the slot
    # (layer) axis replicated, {batch: data, expert: tp}
    assert expert_values.values.sharding.is_equivalent_to(
        NamedSharding(mesh, P(None, batch_axes(mesh), None, "tp", None)),
        expert_values.values.ndim,
    )
    dense_values = placed_stacks.stacks[dense_group].components
    assert isinstance(dense_values, jax.Array)
    assert dense_values.sharding.is_equivalent_to(
        NamedSharding(mesh, P(None, batch_axes(mesh), None, "tp")), dense_values.ndim
    )
    assert placed_stacks.stacks[expert_group].delta.sharding.is_equivalent_to(
        NamedSharding(mesh, P(None, batch_axes(mesh), None)), 3
    )

    unplaced_state = run_two_steps(placed=False)
    placed_sources = placed_stacks.per_site()
    unplaced_sources = unplaced_state.training.adversaries[ppgd.type].sources.per_site()
    init_sources = init_persistent_sources(sites, (BATCH, SEQ), jnp.uint16, src_key).per_site()

    def flat(components: SourceComponents) -> np.ndarray:
        return np.asarray(
            (components.flat if isinstance(components, EBS) else components), np.float32
        )

    slot_deltas: dict[str, np.ndarray] = {}
    for spec in sites:
        got = flat(placed_sources[spec.name].components)
        want = flat(unplaced_sources[spec.name].components)
        started = flat(init_sources[spec.name].components)
        # both arms quantize under the SAME key chain, so placed-vs-unplaced can only
        # diverge where bf16 reassociation through the tp/data splits flips a
        # stochastic-rounding decision by one 1/65535 step — norm-level tolerance,
        # like the CI census (comparison at integer scale: same units both sides)
        denom = np.linalg.norm(want)
        assert np.linalg.norm(got - want) / (denom if denom > 0 else 1.0) < 2e-2, spec.name
        slot_deltas[spec.name] = want - started
    # slot locality is judged on the sites that moved: per-datapoint grads mean batch
    # slots (which saw different tokens) cannot all have stepped identically
    moved = {name: delta for name, delta in slot_deltas.items() if np.any(delta != 0.0)}
    assert moved, "no persistent source moved in two steps"
    assert any(not np.array_equal(delta[0], delta[1]) for delta in moved.values()), (
        "batch slots received identical updates — bsc grads are not per-datapoint"
    )


@multidevice
@pytest.mark.multidevice
def test_placed_streamed_output_recon_census_and_parity():
    """The streamed output edge through the placed production forward: the recon KL's
    gradient over the factored package matches the unplaced MATERIALIZED edge (one
    check covering the edge flip and the placement), and the compiled module keeps
    zero in-loop cross-data collectives — the vocab-chunk scan is a while loop, so a
    stray cross-data reduction inside the streamed kernels would show here."""
    from param_decomp.core.recon import reconstruction_observations
    from param_decomp.targets.qwen36_moe import StreamedOutputEdge

    mesh = _mesh()
    model, _components = _model_and_components(CENSUS_CS)
    streamed_model = dataclasses.replace(model, output_edge=StreamedOutputEdge(n_vocab_chunks=4))
    sites = model.sites
    cfg = model.cfg
    tokens = jax.random.randint(jax.random.PRNGKey(11), (BATCH, SEQ), 0, cfg.vocab_size)
    masks = {
        spec.name: jax.random.uniform(jax.random.PRNGKey(12), (BATCH, SEQ, spec.C))
        for spec in sites
    }
    deltas = {spec.name: jnp.full((BATCH, SEQ), 0.25) for spec in sites}

    def loss(
        target: PlacedModel[LMOutput],
        value: ComponentStacks,
        batch: Array,
        component: dict[str, Array],
        delta: dict[str, Array],
        loss_mesh: jax.sharding.Mesh | None,
    ) -> Array:
        clean = jax.tree.map(jax.lax.stop_gradient, target.clean_forward(batch))
        clean_observed = reconstruction_observations(
            clean, target.pin_output_batch, hidden_acts_capture_keys=frozenset(), mesh=loss_mesh
        )
        masked = target.masked_forward(
            prepare_compute_weights(target, value),
            batch,
            masking=MaterializedMasking(component_masks=component, weight_delta_masks=delta),
            remat=True,
        )
        masked_observed = reconstruction_observations(
            masked, target.pin_output_batch, hidden_acts_capture_keys=frozenset(), mesh=loss_mesh
        )
        return target.recon_loss_fn(masked_observed.output, clean_observed.output)

    unplaced = PlacedModel(model=model, placement=None)
    components = init_component_stacks(sites, jax.random.PRNGKey(1))
    expected_loss, expected_grads = jax.jit(jax.value_and_grad(loss, argnums=1))(
        unplaced, components, tokens, masks, deltas, None
    )

    rules = from_config("zero1-replicated-resident-moe", mesh, sites)
    placed_model = place_target(streamed_model, rules)
    placed_components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
    placed_tokens = _batch_placed(tokens, mesh)
    placed_masks, placed_deltas = jax.tree.map(
        lambda value: _batch_placed(value, mesh), (masks, deltas)
    )
    with jax.set_mesh(mesh):
        grad_fn = jax.jit(jax.value_and_grad(loss, argnums=1), static_argnums=(5,))
        got_loss, got_grads = grad_fn(
            placed_model, placed_components, placed_tokens, placed_masks, placed_deltas, mesh
        )
        hlo = (
            grad_fn.lower(
                placed_model, placed_components, placed_tokens, placed_masks, placed_deltas, mesh
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
    _assert_no_weight_gather_in_any_loop(hlo, BATCH // DATA)


@multidevice
@pytest.mark.multidevice
def test_placed_narrow_slow_eval_runs_at_data_gt_1():
    """The bsc smoke's launch blocker, at tiny C: the STEP-0 SLOW-EVAL tier
    (ComponentActivationDensity counts, mean-CI sums, CIHistograms bins, CI_L0) on the
    placed seat at data>1 feeds dp-sharded NARROW values into the per-component
    reductions — the trace gate and fit check never run this tier, so it must lower and
    execute here. Density/sums are also pinned exactly against the same reductions on
    host-replicated copies of the step's own CI values."""
    from param_decomp.core.ci_fn import (
        MoEChunkwiseTransformerCIFn,
        PlacedCIFn,
        build_ci_fn,
        evaluate_ci,
        resolve_ci_placement,
    )
    from param_decomp.core.ci_l0_eval import make_ci_l0_eval_step
    from param_decomp.core.components import NarrowCI
    from param_decomp.core.slow_eval import make_ci_reduction_step
    from param_decomp.tests.core.test_narrow_ci import _scatter_to_full

    mesh = _mesh()
    model, components = _model_and_components(CENSUS_CS)
    arch = _moe_ci_arch(model.cfg)
    rules = from_config("zero1-replicated-resident-moe", mesh, model.sites)
    ci_placement = resolve_ci_placement(arch, rules)
    assert ci_placement is not None
    tokens = jax.random.randint(jax.random.PRNGKey(31), (BATCH, SEQ), 0, model.cfg.vocab_size)
    ci_fn = build_ci_fn(arch, model.sites, jax.random.PRNGKey(32))
    assert isinstance(ci_fn, MoEChunkwiseTransformerCIFn)
    with jax.set_mesh(mesh):
        placed_model = place_target(model, rules)
        placed_fn = jax.device_put(ci_fn, ci_fn.shardings(mesh, ci_placement))
        placed_ci_fn = PlacedCIFn(fn=placed_fn, placement=ci_placement)
        placed_tokens = _batch_placed(tokens, mesh)
        # The shared-context shape of the production tier: one CI envelope, then the
        # jitted reduction over its compute-precision preactivations.
        context_ci = evaluate_ci(
            placed_ci_fn,
            placed_model.clean_forward(placed_tokens, placed_fn.capture_keys).captures,
            remat=False,
        )
        reduction_step = make_ci_reduction_step(0.0, None, 4)
        density, ci_sums, n_positions, binned_lower, binned_pre, density_hist = reduction_step(
            context_ci.preactivations
        )
        l0_step = make_ci_l0_eval_step(
            placed_model, placed_fn.capture_keys, 0.0, groups=None, mesh=mesh
        )
        l0 = l0_step(placed_model, components, placed_ci_fn, placed_tokens, jax.random.PRNGKey(0))
        jax.block_until_ready((density, ci_sums, binned_lower, binned_pre, l0))
        # The oracle: the SAME envelope's CI values, replicated to host, reduced full-width.
        lower = context_ci.lower

    assert int(n_positions) == BATCH * SEQ
    assert not density_hist
    for spec in model.sites:
        value = lower[spec.name]
        if isinstance(value, NarrowCI):
            host = NarrowCI(
                jnp.asarray(np.asarray(value.values)),
                jnp.asarray(np.asarray(value.router_indices)),
                value.n_experts,
            )
            full = np.asarray(_scatter_to_full(host), dtype=np.float32)
        else:
            full = np.asarray(value, dtype=np.float32)
        flat = full.reshape(-1, spec.C)
        np.testing.assert_array_equal(np.asarray(density[spec.name]), (flat > 0.0).sum(0))
        np.testing.assert_allclose(
            np.asarray(ci_sums[spec.name]), flat.sum(0), rtol=1e-5, atol=1e-5
        )
        counts, lo, hi = binned_lower[spec.name]
        assert np.isfinite(np.asarray(counts)).all() and float(lo) <= float(hi)
        np.testing.assert_allclose(
            np.asarray(l0[f"l0/0.0_{spec.name}"]), (flat > 0.0).sum(-1).mean(), rtol=1e-5
        )


@multidevice
@pytest.mark.multidevice
def test_placed_owner_nonlinearity_eval_reads_slots_of_the_stack_sharded_masters():
    """The standing nonlinearity eval over the owner-placed fp32 masters, whose persist
    stack axis is sharded over `data` (one slot per device here): the device step reduces
    each group's whole stack and the per-site read happens on the host, so no program
    slices the sharded stack axis. Pinned site by site against the same statistics of
    the unplaced masters."""
    from param_decomp.core.nonlinearity_eval import (
        component_nonlinearity_stats,
        make_nonlinearity_eval_step,
        site_nonlinearity_stats,
    )

    mesh = _mesh()
    model, components = _model_and_components(CENSUS_CS)
    rules = from_config("owner-replicated-resident-moe", mesh, model.sites)
    with jax.set_mesh(mesh):
        placed = jax.device_put(components, component_stacks_shardings(components, rules))
        assert all(jax.typeof(us).sharding.spec[0] == "data" for _, us in placed.stacks.values()), {
            group: jax.typeof(us).sharding.spec for group, (_, us) in placed.stacks.items()
        }
        stats = site_nonlinearity_stats(
            make_nonlinearity_eval_step(model.sites, {})(placed), model.sites
        )

    partitioned = [site for site in model.sites if site.nonlinearity_partition is not None]
    assert partitioned and set(stats) == {site.name for site in partitioned}
    for site in partitioned:
        assert site.nonlinearity_partition is not None
        u = components.site(site.name).U
        expected = component_nonlinearity_stats(
            u.reshape(-1, u.shape[-1]), site.nonlinearity_partition
        )
        assert stats[site.name].soft_use_count.shape == (site.C,)
        np.testing.assert_allclose(
            stats[site.name].soft_use_count, expected.soft_use_count, rtol=1e-6, err_msg=site.name
        )
        np.testing.assert_allclose(
            stats[site.name].effective_use_count_per_subcomponent,
            expected.effective_use_count_per_subcomponent,
            rtol=1e-6,
            err_msg=site.name,
        )
