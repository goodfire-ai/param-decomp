"""Placed persist-stack padding: the entry's pad strip and real-slot gather, its grad
transpose, the faithfulness lane, and a full padded train step — for the V/U groups and for the chunkwise CI fn's chunk stack.

Three worlds over the 8-device suite: `owner` at (4,2,1), where the 6-layer tiny target's
kind stacks pad to 8 naturally; `owner` at (2,2,1) — pad-free — where a census surgery
re-runs the SAME world padded, pinning real-slot bit-identity; and
`owner-replicated-resident` at (data=4, tp=2), where the kind stacks pad to 8 and a
3-chunk CI fn pads to 4 (the Qwen3-4B seat's mechanism at toy scale)."""

import dataclasses
import re
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax import random
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core.adversary import (
    PersistentAdversary,
    init_persistent_sources,
    init_sources_adam_state,
)
from param_decomp.core.axes import Axes
from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    ChunkwiseTransformerCIFn,
    MHACIAttention,
    PlacedCIFn,
    build_ci_fn,
    evaluate_ci,
    materialize_ci_compute_weights,
    pad_ci_fn,
    resolve_ci_placement,
)
from param_decomp.core.components import (
    ComponentStacks,
    SiteC,
    SiteSpec,
    init_component_stacks,
    require_full_emission,
)
from param_decomp.core.configs import (
    AdamPGDConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    StochasticReconSubsetLossConfig,
    UniformKSubsetRoutingConfig,
)
from param_decomp.core.faithfulness import faithfulness_loss_for
from param_decomp.core.init_placed import (
    init_ci_fn_placed,
    init_component_stacks_placed,
    padded_component_initializer,
    random_component_initializer,
)
from param_decomp.core.model import (
    MaterializedMasking,
    PlacedModel,
    faithfulness_weight_deltas,
)
from param_decomp.core.objective import build_objective
from param_decomp.core.placement import (
    CIFnPlacement,
    PlacementRules,
    StackCensus,
    component_stacks_audit,
    component_stacks_to_compute_weights,
    dropped_mesh_axes,
    from_config,
)
from param_decomp.core.schedule import Knot, ScheduleConfig
from param_decomp.core.sharding import place_target, shard_batch
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    TrainingItem,
    TrainState,
    make_train_step,
)
from param_decomp.targets.glu_transformer import GLUDecomposedModel
from param_decomp.targets.llama_simple_mlp import (
    KIND_ORDER,
    canonical_site_cs,
    site_name,
    site_specs,
)
from param_decomp.targets.testing import (
    materialized_logits,
    tiny_simple_mlp_cfg,
    tiny_simple_mlp_decomposed_model,
)

pytestmark = [
    pytest.mark.multidevice,
    pytest.mark.skipif(
        jax.default_backend() != "cpu" or jax.device_count() < 8,
        reason="requires the eight-device CPU topology from make test-multidevice",
    ),
]

_B, _T, _C = 8, 16, 8
_N_LAYER = tiny_simple_mlp_cfg().n_layer  # 6: pads to 8 under a ÷4 stack cut


def _mesh(replicate: int, fsdp: int) -> Mesh:
    return Mesh(
        np.array(jax.devices()[: replicate * fsdp]).reshape(replicate, fsdp, 1),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )


def _resident_mesh(data: int, tp: int) -> Mesh:
    return Mesh(
        np.array(jax.devices()[: data * tp]).reshape(data, tp),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )


def _ci_arch(site_names: tuple[str, ...], n_chunks: int) -> ChunkwiseTransformerCIArch:
    """`n_chunks` equal chunks over the sites in order (every site has one C, so the
    per-slot heads stack whatever the boundaries)."""
    per_chunk = len(site_names) // n_chunks
    assert per_chunk * n_chunks == len(site_names), (len(site_names), n_chunks)
    return ChunkwiseTransformerCIArch(
        chunks=tuple(
            Chunk(
                input_taps=("resid.0",),
                output_sites=site_names[i * per_chunk : (i + 1) * per_chunk],
            )
            for i in range(n_chunks)
        ),
        input_dim=tiny_simple_mlp_cfg().n_embd,
        d_model=16,
        n_blocks=2,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )


def _model_and_sites() -> tuple[GLUDecomposedModel, tuple[SiteSpec, ...]]:
    cfg = tiny_simple_mlp_cfg()
    site_cs = canonical_site_cs(
        tuple(
            SiteC(site_name(layer, kind), _C) for layer in range(cfg.n_layer) for kind in KIND_ORDER
        )
    )
    sites = site_specs(cfg, site_cs)
    return tiny_simple_mlp_decomposed_model(cfg, sites, random.PRNGKey(0)), sites


def _tokens() -> jax.Array:
    return random.randint(random.PRNGKey(4), (_B, _T), 0, tiny_simple_mlp_cfg().vocab_size)


def test_padding_world_init_entry_slice_and_grad_transpose():
    """At (4,2,1) the 6-stacks pad to 8: the placed init seeds real slots identically to
    the unpadded init and zeros the pads; the entry gather's slice hands compute exactly
    the real stacks; and its transpose returns real grads with exact zeros on pads."""
    model, sites = _model_and_sites()
    mesh = _mesh(4, 2)
    rules = from_config("owner", mesh, model.sites)
    assert all(entry.stack_pad == 2 for entry in rules.components.group_census.values())
    reference = init_component_stacks(sites, random.PRNGKey(1))
    weights = jnp.arange(1.0, 1.0 + len(reference.stacks))
    with jax.set_mesh(mesh):
        vu = init_component_stacks_placed(sites, random.PRNGKey(1), rules)
        for group, (Vs, Us) in vu.stacks.items():
            for placed, expected in (
                (Vs, reference.stacks[group][0]),
                (Us, reference.stacks[group][1]),
            ):
                gathered = jax.device_get(placed)
                assert gathered.shape == (_N_LAYER + 2, *expected.shape[1:])
                np.testing.assert_array_equal(gathered[:_N_LAYER], expected, strict=True)
                assert (gathered[_N_LAYER:] == 0.0).all(), group

        def compute_loss(stacks: ComponentStacks) -> jax.Array:
            resident = component_stacks_to_compute_weights(stacks, rules.components)
            total = jnp.zeros(())
            for w, (Vs, Us) in zip(weights, resident.stacks.values(), strict=True):
                assert Vs.shape[0] == _N_LAYER and Us.shape[0] == _N_LAYER
                total = total + w * (jnp.sum(Vs * Vs) + jnp.sum(Us * Us))
            return total

        grads = eqx.filter_grad(compute_loss)(vu)

    def reference_loss(stacks: ComponentStacks) -> jax.Array:
        total = jnp.zeros(())
        for w, (Vs, Us) in zip(weights, stacks.stacks.values(), strict=True):
            total = total + w * (jnp.sum(Vs * Vs) + jnp.sum(Us * Us))
        return total

    reference_grads = eqx.filter_grad(reference_loss)(reference)
    for group, (gV, gU) in grads.stacks.items():
        for placed, expected in (
            (gV, reference_grads.stacks[group][0]),
            (gU, reference_grads.stacks[group][1]),
        ):
            gathered = jax.device_get(placed)
            assert (gathered[_N_LAYER:] == 0.0).all(), group
            np.testing.assert_allclose(gathered[:_N_LAYER], expected, rtol=1e-6)


def test_padding_world_pre_init_audit_sees_the_pads():
    """The engine audits the initializer's abstract tree against the rules before seeding
    (`run._prepare_run`): at a padded world that tree must carry the census pads — the raw
    initializer's does not, and the audit rightly refuses it."""
    model, _ = _model_and_sites()
    rules = from_config("owner", _mesh(4, 2), model.sites)
    raw = eqx.filter_eval_shape(random_component_initializer, model, random.PRNGKey(1))
    with pytest.raises(AssertionError, match="stack pad of 2"):
        component_stacks_audit(raw, rules)
    padded = eqx.filter_eval_shape(
        padded_component_initializer(rules, random_component_initializer), model, random.PRNGKey(1)
    )
    audit = component_stacks_audit(padded, rules)
    assert all(shape[0] == _N_LAYER + 2 for _, _, shape in audit.values())


def test_padding_world_faithfulness_lane_pads_are_exact_zeros():
    model, sites = _model_and_sites()
    mesh = _mesh(4, 2)
    rules = from_config("owner", mesh, model.sites)
    placed = place_target(model, rules)
    reference_vu = init_component_stacks(sites, random.PRNGKey(1))
    reference_deltas = model.weight_deltas(reference_vu)
    reference_loss = faithfulness_loss_for(PlacedModel(model=model, placement=None))(
        reference_deltas
    )
    with jax.set_mesh(mesh):
        vu = init_component_stacks_placed(sites, random.PRNGKey(1), rules)
        deltas = jax.jit(lambda c: faithfulness_weight_deltas(placed, c))(vu)
        loss = jax.jit(faithfulness_loss_for(placed))(deltas)
    for group, delta in deltas.items():
        gathered = jax.device_get(delta)
        assert gathered.shape[0] == _N_LAYER + 2
        assert (gathered[_N_LAYER:] == 0.0).all(), group
        # W - V@U is a near-cancellation of O(0.1) operands; the placed and unplaced
        # products can land one fp32 ulp of the operands apart (host FMA contraction),
        # which no relative bound on the small difference absorbs.
        np.testing.assert_allclose(
            gathered[:_N_LAYER], reference_deltas[group], rtol=1e-5, atol=2**-23
        )
    np.testing.assert_allclose(jax.device_get(loss), reference_loss, rtol=1e-6)


def _full_objective_state_and_step(
    model: GLUDecomposedModel,
    sites: tuple[SiteSpec, ...],
    mesh: Mesh,
    rules: PlacementRules,
    arch: ChunkwiseTransformerCIArch,
):
    """The real `make_train_step` over the committed seat's term classes (faithfulness +
    imp-min + stochastic subset + persistent PPGD), assembled placed."""
    placed = place_target(model, rules)
    vu = init_component_stacks_placed(sites, random.PRNGKey(1), rules)
    ci_fn = init_ci_fn_placed(arch, placed.sites, random.PRNGKey(2), mesh, rules)
    src = init_persistent_sources(placed.sites, (1, _T), jnp.float32, random.PRNGKey(3))
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    ppgd_cfg = PersistentPGDReconLossConfig(
        coeff=0.5,
        source_shape="sc",
        optimizer=AdamPGDConfig(
            beta1=0.5,
            beta2=0.99,
            lr_schedule=ScheduleConfig(
                max_val=0.01,
                points=(Knot(at=0.0, frac=0.0), Knot(at=0.025, frac=1.0), Knot(at=1.0, frac=1.0)),
            ),
        ),
        n_warmup_steps=2,
    )
    state = TrainState(
        decomposition=Decomposition(components=vu, ci_fn=ci_fn),
        training=TrainingItem(
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={
                ppgd_cfg.type: PersistentAdversary(
                    sources=src,
                    opt_state=init_sources_adam_state(src),
                    state_key=ppgd_cfg.type,
                    optimizer=ppgd_cfg.optimizer,
                    n_warmup=ppgd_cfg.n_warmup_steps,
                )
            },
            freq_ema=None,
            step=jnp.zeros((), jnp.int32),
        ),
    )
    loss_terms = build_objective(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                gamma=ScheduleConfig(
                    max_val=1.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.2))
                ),
            ),
            StochasticReconSubsetLossConfig(
                routing=UniformKSubsetRoutingConfig(), coeff=0.5, n_mask_samples=1
            ),
            ppgd_cfg,
        ),
        placed.site_names,
    )
    step = make_train_step(
        model_static=placed,
        substrate=ForwardSubstrate.of(
            placed,
            remat_recon_forwards=True,
            remat_ci_fn=False,
            ci_capture_keys=ci_fn.capture_keys,
            ci_placement=resolve_ci_placement(arch, rules),
        ),
        objective=loss_terms,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=100,
        faithfulness=faithfulness_loss_for(placed),
    )
    return placed, state, step


def _assert_pads_exactly_zero(state: TrainState, real: int) -> None:
    vu = state.decomposition.components
    for group, (Vs, Us) in vu.stacks.items():
        assert (jax.device_get(Vs)[real:] == 0.0).all(), group
        assert (jax.device_get(Us)[real:] == 0.0).all(), group
    for leaf in jax.tree.leaves(eqx.filter(state.training.components_opt_state, eqx.is_array)):
        if leaf.ndim >= 3 and leaf.shape[0] == real + 2:
            assert (jax.device_get(leaf)[real:] == 0.0).all(), "moments grew pad mass"


def test_padded_owner_full_train_step_keeps_pads_exactly_zero():
    model, sites = _model_and_sites()
    mesh = _mesh(4, 2)
    rules = from_config("owner", mesh, model.sites)
    # 4 chunks tile owner's ÷replicate=4 CI rows: this world pads the components only
    arch = _ci_arch(model.site_names, 4)
    with jax.set_mesh(mesh):
        placed, state, step = _full_objective_state_and_step(model, sites, mesh, rules, arch)
        tokens = shard_batch(_tokens(), mesh, batch_axis=0)
        metrics: dict[str, jax.Array] = {}
        for i in range(2):
            state, metrics = step(placed, state, tokens, random.PRNGKey(100 + i))
    assert jnp.isfinite(metrics["total"]), metrics
    _assert_pads_exactly_zero(state, _N_LAYER)


def _sum_of_squares(leaves: list[jax.Array]) -> jax.Array:
    return sum((jnp.sum(leaf * leaf) for leaf in leaves), jnp.zeros(()))


def _compiled_grad_hlo(loss: Callable[..., jax.Array], *args: object) -> str:
    text = jax.jit(jax.grad(loss)).lower(*args).compile().as_text()
    assert text is not None
    return text


def _all_gather_leading_dims(hlo: str) -> set[int]:
    """The leading (stack) dim of every all-gather result in the compiled module."""
    dims: set[int] = set()
    for line in hlo.splitlines():
        m = re.search(r"all-gather(?:-start)?\(", line)
        if m is None:
            continue
        for dims_text in re.findall(r"\b(?:bf16|f32)\[([\d,]+)\]", line[: m.start()]):
            dims.add(int(dims_text.split(",")[0]))
    return dims


def test_padded_entry_matches_the_gather_then_strip_spelling_bit_for_bit():
    """The padded entry against the spelling it replaces — gather the PADDED stack along
    its persist layout, then strip — on the same placed masters at the (4,2,1) padded
    world: residents and raw gradients through the two routes are bit-identical (the
    all-to-all moves the same slots to the same places; only the pads' bytes stay home)."""
    model, sites = _model_and_sites()
    mesh = _mesh(4, 2)
    rules = from_config("owner", mesh, model.sites)
    components = rules.components
    weights = jnp.arange(1.0, 1.0 + len(components.group_census))

    def entry(stacks: ComponentStacks) -> ComponentStacks:
        return component_stacks_to_compute_weights(stacks, components)

    def gather_then_strip(stacks: ComponentStacks) -> ComponentStacks:
        def one(value: jax.Array, census: StackCensus, axes: Axes) -> jax.Array:
            reduced = dropped_mesh_axes(
                components.optimizer_state, components.compute_weights, axes
            )
            gathered = jax.sharding.reshard(
                jax.lax.optimization_barrier(value),
                NamedSharding(mesh, P(*components.compute_weights.spec_for(axes), reduced=reduced)),
            )
            return jax.lax.slice_in_dim(gathered, 0, census.stack_len, axis=0)

        stacks_out: dict[str, tuple[jax.Array, jax.Array]] = {}
        for group, (vs, us) in stacks.stacks.items():
            census = components.group_census[group]
            f = census.factorization
            stacks_out[group] = (one(vs, census, f.v_axes), one(us, census, f.u_axes))
        return ComponentStacks(stacks=stacks_out, site_slots=stacks.site_slots)

    def loss_through(
        route: Callable[[ComponentStacks], ComponentStacks],
    ) -> Callable[[ComponentStacks], jax.Array]:
        def loss(stacks: ComponentStacks) -> jax.Array:
            resident = route(stacks)
            return sum(
                (
                    w * (jnp.sum(Vs * Vs) + jnp.sum(Us * Us))
                    for w, (Vs, Us) in zip(weights, resident.stacks.values(), strict=True)
                ),
                jnp.zeros(()),
            )

        return loss

    routes = (entry, gather_then_strip)
    with jax.set_mesh(mesh):
        vu = init_component_stacks_placed(sites, random.PRNGKey(1), rules)
        residents = [eqx.filter_jit(route)(vu) for route in routes]
        grads = [eqx.filter_jit(eqx.filter_grad(loss_through(route)))(vu) for route in routes]
    for new, old in (residents, grads):
        for got, expected in zip(jax.tree.leaves(new), jax.tree.leaves(old), strict=True):
            np.testing.assert_array_equal(
                jax.device_get(got), jax.device_get(expected), strict=True
            )


def test_padded_entry_gathers_real_slots_only():
    """The compiled entry at the resident world (kinds 6 → 8, chunks 3 → 4) carries no pad
    through any gather: every all-gather result — forward and its transpose — leads with
    a REAL stack length, never the padded one, and the persist→waypoint hop lowers as an
    all-to-all. The pad-free (2,2,1) owner world keeps the direct stack-axis gather and
    lowers no all-to-all at all."""
    model, sites, mesh, rules = _resident_world()
    arch = _ci_arch(model.site_names, _N_CHUNKS)
    placement = resolve_ci_placement(arch, rules)

    def components_loss(stacks: ComponentStacks) -> jax.Array:
        return _sum_of_squares(
            jax.tree.leaves(component_stacks_to_compute_weights(stacks, rules.components))
        )

    def ci_loss(fn: ChunkwiseTransformerCIFn) -> jax.Array:
        compute = materialize_ci_compute_weights(PlacedCIFn(fn=fn, placement=placement)).fn
        assert isinstance(compute, ChunkwiseTransformerCIFn)
        return _sum_of_squares(_chunk_leaves(compute))

    with jax.set_mesh(mesh):
        vu = init_component_stacks_placed(sites, random.PRNGKey(1), rules)
        fn = init_ci_fn_placed(arch, sites, random.PRNGKey(2), mesh, rules)
        components_hlo = _compiled_grad_hlo(components_loss, vu)
        ci_hlo = _compiled_grad_hlo(ci_loss, fn)
    assert _all_gather_leading_dims(components_hlo) == {_N_LAYER}
    assert _all_gather_leading_dims(ci_hlo) == {_N_CHUNKS}
    assert "all-to-all" in components_hlo and "all-to-all" in ci_hlo

    pad_free_mesh = _mesh(2, 2)
    pad_free_rules = from_config("owner", pad_free_mesh, model.sites)
    assert all(entry.stack_pad == 0 for entry in pad_free_rules.components.group_census.values())

    def pad_free_loss(stacks: ComponentStacks) -> jax.Array:
        return _sum_of_squares(
            jax.tree.leaves(component_stacks_to_compute_weights(stacks, pad_free_rules.components))
        )

    with jax.set_mesh(pad_free_mesh):
        vu = init_component_stacks_placed(sites, random.PRNGKey(1), pad_free_rules)
        hlo = _compiled_grad_hlo(pad_free_loss, vu)
    assert _all_gather_leading_dims(hlo) == {_N_LAYER}
    assert "all-to-all" not in hlo


def _padded_by_surgery(rules: PlacementRules, pad: int) -> PlacementRules:
    """The SAME rules with every census entry's pad bumped — the pads are data, so a
    test can author a padded world where the mesh alone would not demand one."""
    census = {
        group: dataclasses.replace(entry, stack_pad=pad)
        for group, entry in rules.components.group_census.items()
    }
    return dataclasses.replace(
        rules, components=dataclasses.replace(rules.components, group_census=census)
    )


def test_padded_grads_are_bit_identical_to_unpadded_at_the_same_world():
    """Census surgery at (2,2,1) — where 6 stacks tile ÷2 pad-free — re-runs the SAME
    world padded (+2): raw gradients through the entry gather + the faithfulness lane
    must match the unpadded run's on the real slots BIT FOR BIT, with exact zeros on
    the pads (the pad slots ride only zero-contribution paths)."""
    model, sites = _model_and_sites()
    mesh = _mesh(2, 2)
    rules = from_config("owner", mesh, model.sites)
    assert all(entry.stack_pad == 0 for entry in rules.components.group_census.values())
    padded_rules = _padded_by_surgery(rules, 2)
    tokens_host = _tokens()
    masking = MaterializedMasking(
        component_masks={s.name: 0.5 * jnp.ones((_B, _T, s.C)) for s in model.sites}
    )

    def run(world_rules: PlacementRules) -> ComponentStacks:
        placed = place_target(model, world_rules)
        faithfulness = faithfulness_loss_for(placed)

        def loss(vu: ComponentStacks, tokens: jax.Array) -> jax.Array:
            prepared = model.prepare_compute_weights(vu, world_rules)
            output = materialized_logits(
                model.masked_forward(
                    prepared,
                    tokens,
                    masking=masking,
                    placement=world_rules,
                    capture_keys=frozenset(),
                    remat=False,
                ).output
            )
            faith = faithfulness(faithfulness_weight_deltas(placed, vu))
            return jnp.sum(output * output) + faith

        with jax.set_mesh(mesh):
            vu = init_component_stacks_placed(sites, random.PRNGKey(1), world_rules)
            tokens = shard_batch(tokens_host, mesh, batch_axis=0)
            return eqx.filter_jit(eqx.filter_grad(loss))(vu, tokens)

    unpadded, padded = run(rules), run(padded_rules)
    for group, (gV, gU) in unpadded.stacks.items():
        pV, pU = padded.stacks[group]
        for real, pad in ((gV, pV), (gU, pU)):
            gathered = jax.device_get(pad)
            assert (gathered[_N_LAYER:] == 0.0).all(), group
            np.testing.assert_array_equal(gathered[:_N_LAYER], jax.device_get(real), strict=True)


def test_padded_train_step_matches_unpadded_at_the_same_world():
    """The full step at the surgically padded (2,2,1) world lands the same real master
    slots as the unpadded run. Not quite bitwise: the grad-clip's global norm reduces
    over the shape-changed (padded) leaves, and XLA's reduction tree over the larger
    array regroups the REAL terms — a one-ulp wobble in the clip scale. Raw grads are
    pinned bitwise above; here the tolerance is a few fp32 ulps."""
    model, sites = _model_and_sites()
    mesh = _mesh(2, 2)
    rules = from_config("owner", mesh, model.sites)
    padded_rules = _padded_by_surgery(rules, 2)
    tokens_host = _tokens()
    arch = _ci_arch(model.site_names, 4)

    def run(world_rules: PlacementRules) -> TrainState:
        with jax.set_mesh(mesh):
            placed, state, step = _full_objective_state_and_step(
                model, sites, mesh, world_rules, arch
            )
            tokens = shard_batch(tokens_host, mesh, batch_axis=0)
            state, _ = step(placed, state, tokens, random.PRNGKey(100))
        return state

    unpadded, padded = run(rules), run(padded_rules)
    for group, (Vs, Us) in unpadded.decomposition.components.stacks.items():
        pVs, pUs = padded.decomposition.components.stacks[group]
        for real, pad in ((Vs, pVs), (Us, pUs)):
            np.testing.assert_allclose(
                jax.device_get(pad)[:_N_LAYER], jax.device_get(real), rtol=1e-6, atol=1e-7
            )
    _assert_pads_exactly_zero(padded, _N_LAYER)


# ── the CI fn's chunk stack under owner-replicated-resident ──────────────────
# The Qwen3-4B seat's mechanism at toy scale: `{data: 4, tp: 2}`, 3 chunks that do not
# tile the ÷data cut of the CI-fn persist rows, so the chunk stack pads to 4.

_N_CHUNKS = 3


def _resident_world() -> tuple[GLUDecomposedModel, tuple[SiteSpec, ...], Mesh, PlacementRules]:
    model, sites = _model_and_sites()
    mesh = _resident_mesh(4, 2)
    return model, sites, mesh, from_config("owner-replicated-resident", mesh, model.sites)


def _chunk_leaves(fn: ChunkwiseTransformerCIFn) -> list[jax.Array]:
    return jax.tree.leaves(fn.chunks)


def test_ci_chunk_padding_world_init_entry_slice_and_grad_transpose():
    """At (data=4, tp=2) the 3-chunk CI fn pads to 4: the census says so, the placed init
    seeds the real chunk slots like the unplaced init and zeros the pad on EVERY leaf, the
    entry (`materialize_ci_compute_weights`) hands the scan exactly the real chunks, and
    its transpose returns real-chunk grads with exact zeros on the pad."""
    model, sites, mesh, rules = _resident_world()
    arch = _ci_arch(model.site_names, _N_CHUNKS)
    placement = resolve_ci_placement(arch, rules)
    assert placement is not None
    assert placement.chunks == StackCensus(stack_len=_N_CHUNKS, stack_pad=1)
    reference = build_ci_fn(arch, sites, random.PRNGKey(2))
    assert isinstance(reference, ChunkwiseTransformerCIFn)

    def compute_loss(f: ChunkwiseTransformerCIFn, ci_placement: CIFnPlacement | None) -> jax.Array:
        compute = materialize_ci_compute_weights(PlacedCIFn(fn=f, placement=ci_placement)).fn
        assert isinstance(compute, ChunkwiseTransformerCIFn) and compute.stack_pad == 0
        total = jnp.zeros(())
        for leaf in _chunk_leaves(compute):
            assert leaf.shape[0] == _N_CHUNKS
            leaf = leaf.astype(jnp.float32)
            total = total + jnp.sum(leaf * leaf)
        return total

    with jax.set_mesh(mesh):
        fn = init_ci_fn_placed(arch, sites, random.PRNGKey(2), mesh, rules)
        assert isinstance(fn, ChunkwiseTransformerCIFn) and fn.stack_pad == 1
        for placed_leaf, expected in zip(_chunk_leaves(fn), _chunk_leaves(reference), strict=True):
            gathered = jax.device_get(placed_leaf)
            assert gathered.shape == (_N_CHUNKS + 1, *expected.shape[1:])
            np.testing.assert_allclose(gathered[:_N_CHUNKS], expected, rtol=1e-6)
            assert (gathered[_N_CHUNKS:] == 0.0).all()
        grads = eqx.filter_grad(compute_loss)(fn, placement)

    reference_grads = eqx.filter_grad(compute_loss)(reference, None)
    for placed_leaf, expected in zip(
        _chunk_leaves(grads), _chunk_leaves(reference_grads), strict=True
    ):
        gathered = jax.device_get(placed_leaf)
        assert (gathered[_N_CHUNKS:] == 0.0).all()
        np.testing.assert_allclose(gathered[:_N_CHUNKS], expected, rtol=1e-6)


def test_padded_ci_fn_forward_and_grads_are_bit_identical_to_unpadded_at_the_same_world():
    """Census surgery at (data=4, tp=2): a 4-chunk fn tiles ÷data=4 pad-free; the SAME
    world re-run with its chunk stack padded (+4) must emit bit-identical CI and
    bit-identical real-chunk raw grads, with exact zeros on the pad slots."""
    model, sites, mesh, rules = _resident_world()
    arch = _ci_arch(model.site_names, 4)
    unpadded = resolve_ci_placement(arch, rules)
    assert unpadded is not None and unpadded.chunks == StackCensus(stack_len=4, stack_pad=0)
    padded = CIFnPlacement.resolved(rules.ci_fn, StackCensus(stack_len=4, stack_pad=4))
    taps_host = random.normal(random.PRNGKey(5), (_B, _T, tiny_simple_mlp_cfg().n_embd))

    def run(placement: CIFnPlacement) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
        fn = pad_ci_fn(build_ci_fn(arch, sites, random.PRNGKey(2)), placement)
        assert isinstance(fn, ChunkwiseTransformerCIFn)

        def loss(f: ChunkwiseTransformerCIFn) -> tuple[jax.Array, dict[str, jax.Array]]:
            ci = evaluate_ci(PlacedCIFn(fn=f, placement=placement), taps, remat=False)
            preactivations = {
                site: require_full_emission(value) for site, value in ci.preactivations.items()
            }
            total = sum(
                (
                    jnp.sum(value.astype(jnp.float32) * value.astype(jnp.float32))
                    for value in preactivations.values()
                ),
                jnp.zeros(()),
            )
            return total, preactivations

        with jax.set_mesh(mesh):
            placed_fn = jax.device_put(fn, fn.shardings(mesh, placement))
            taps = {"resid.0": shard_batch(taps_host, mesh, batch_axis=0)}
            (_, preactivations), grads = eqx.filter_jit(
                eqx.filter_value_and_grad(loss, has_aux=True)
            )(placed_fn)
        return jax.device_get(preactivations), [jax.device_get(g) for g in _chunk_leaves(grads)]

    (ci_unpadded, grads_unpadded), (ci_padded, grads_padded) = run(unpadded), run(padded)
    for site, expected in ci_unpadded.items():
        np.testing.assert_array_equal(ci_padded[site], expected, strict=True)
    for expected, wide in zip(grads_unpadded, grads_padded, strict=True):
        assert (wide[4:] == 0.0).all()
        np.testing.assert_array_equal(wide[:4], expected, strict=True)


def test_padded_ci_chunk_stack_full_train_step_keeps_pads_exactly_zero():
    """The real step at the resident world with BOTH persist stacks padded (kinds 6 → 8,
    chunks 3 → 4): the loss is finite, and every pad slot — CI masters and their moments
    alike — stays exactly zero (zero grads through the entry slice, wd = 0)."""
    model, sites, mesh, rules = _resident_world()
    arch = _ci_arch(model.site_names, _N_CHUNKS)
    with jax.set_mesh(mesh):
        placed, state, step = _full_objective_state_and_step(model, sites, mesh, rules, arch)
        tokens = shard_batch(_tokens(), mesh, batch_axis=0)
        metrics: dict[str, jax.Array] = {}
        for i in range(2):
            state, metrics = step(placed, state, tokens, random.PRNGKey(100 + i))
    assert jnp.isfinite(metrics["total"]), metrics
    _assert_pads_exactly_zero(state, _N_LAYER)
    ci_fn = state.decomposition.ci_fn
    assert isinstance(ci_fn, ChunkwiseTransformerCIFn) and ci_fn.stack_pad == 1
    for leaf in _chunk_leaves(ci_fn):
        assert (jax.device_get(leaf)[_N_CHUNKS:] == 0.0).all()
    moments = [
        leaf
        for leaf in jax.tree.leaves(eqx.filter(state.training.ci_fn_opt_state, eqx.is_array))
        if leaf.ndim >= 2 and leaf.shape[0] == _N_CHUNKS + 1
    ]
    assert moments, "the CI optimizer state carries no chunk-stacked moments"
    for leaf in moments:
        assert (jax.device_get(leaf)[_N_CHUNKS:] == 0.0).all(), "CI moments grew pad mass"
