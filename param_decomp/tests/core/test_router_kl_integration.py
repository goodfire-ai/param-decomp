"""Router reconstruction follows pinned experts and differentiates only upstream sites."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core.ci_fn.implementations.block_selected.arch import (
    BlockSelectedChunkwiseTransformerCIFn,
)
from param_decomp.core.components import ComponentStacks, init_component_stacks
from param_decomp.core.configs import AdamWOptimizerConfig
from param_decomp.core.init_placed import init_component_stacks_placed
from param_decomp.core.losses import BatchFrequency, categorical_kl_from_logits
from param_decomp.core.model import MaterializedMasking, PlacedModel
from param_decomp.core.placement import from_config
from param_decomp.core.run_state import _adamw_optimizer
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.sharding import place_target
from param_decomp.lm.batch import LMBatch, LMBatchWithRouting
from param_decomp.routed.experts import ExpertImplementation
from param_decomp.targets.lm_output import LMOutput
from param_decomp.targets.qwen36_moe import (
    Qwen36MoeDecomposedModel,
    QwenPreparedMasking,
    QwenPreparedWeights,
    expert_mixing_weights,
    full_site_cs,
    qwen36_moe_site_specs,
    select_experts,
)
from param_decomp.targets.testing import (
    identity_masking,
    random_mask_values,
    site_masks,
    tiny_qwen36_cfg,
    tiny_qwen36_decomposed_model,
    tiny_qwen36_moe_ci_fn,
)


def _fixture() -> tuple[Qwen36MoeDecomposedModel, ComponentStacks, jax.Array]:
    cfg = replace(tiny_qwen36_cfg(), n_layer=2)
    cs = {
        "gdn_out": 8,
        "attn_o": 8,
        "experts_gate": 16,
        "experts_up": 16,
        "experts_down": 16,
        "shared_gate": 8,
        "shared_up": 8,
        "shared_down": 8,
    }
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, cs))
    model = replace(
        tiny_qwen36_decomposed_model(cfg, sites, jax.random.key(0)),
        expert_implementation="dense_masked",
    )
    return (
        model,
        init_component_stacks(sites, jax.random.key(1)),
        jax.random.randint(jax.random.key(2), (2, 4), 0, cfg.vocab_size),
    )


def _keys(model: Qwen36MoeDecomposedModel) -> frozenset[str]:
    return frozenset(
        f"{tap}.{layer}"
        for tap in ("router_logits", "router_probs", "router_weights")
        for layer in model.expert_router_layers
    )


def _router_kl(clean: dict[str, jax.Array], masked: dict[str, jax.Array]) -> jax.Array:
    values = []
    for key in clean:
        if key.startswith("router_logits."):
            values.append(categorical_kl_from_logits(masked[key], clean[key]))
    return jnp.mean(jnp.stack(values))


@pytest.mark.parametrize("implementation", ["dense_masked", "ragged_dot"])
def test_router_identity_and_counterfactual_selection_are_distinct(
    implementation: ExpertImplementation,
) -> None:
    model, components, tokens = _fixture()
    model = replace(model, expert_implementation=implementation)
    placed = PlacedModel(model, None)
    keys = _keys(model)
    clean = placed.clean_forward(LMBatch(tokens), keys)
    prepared = placed.prepare_compute_weights(components)
    identity = placed.masked_forward(
        prepared,
        clean.conditioning,
        masking=placed.model.prepare_masking(identity_masking(model, clean.conditioning.selection)),
        routes=None,
        capture_keys=keys,
        remat=True,
    )
    np.testing.assert_allclose(_router_kl(clean.captures, identity.captures), 0.0, atol=1e-7)
    values = random_mask_values(model, tokens.shape, jax.random.key(3))
    masked = placed.masked_forward(
        prepared,
        clean.conditioning,
        masking=placed.model.prepare_masking(
            MaterializedMasking(
                component_masks=site_masks(model, clean.conditioning.selection, values),
                weight_delta_masks=None,
            )
        ),
        routes=None,
        capture_keys=keys,
        remat=True,
    )
    np.testing.assert_array_equal(
        masked.conditioning.selection.indices, clean.conditioning.selection.indices
    )
    changed = False
    for layer in model.expert_router_layers:
        probabilities = masked.captures[f"router_probs.{layer}"]
        indices = clean.conditioning.selection.indices[layer]
        counterfactual = select_experts(probabilities, model.cfg.n_experts_per_token)
        changed = changed or (
            bool(jnp.any(jnp.sort(counterfactual, axis=-1) != jnp.sort(indices, axis=-1)))
        )
        np.testing.assert_allclose(
            masked.captures[f"router_weights.{layer}"],
            expert_mixing_weights(probabilities, indices),
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            jax.nn.softmax(masked.captures[f"router_logits.{layer}"]), probabilities, rtol=1e-6
        )
    assert changed, "the intervention must exercise a counterfactual change of expert identities"
    assert float(_router_kl(clean.captures, masked.captures)) > 0.0


def test_router_only_gradient_reaches_mixers_and_ci_but_not_later_experts() -> None:
    model, components, tokens = _fixture()
    placed = PlacedModel(model, None)
    ci_fn = tiny_qwen36_moe_ci_fn(model, jax.random.key(4))
    keys = _keys(model)
    clean = placed.clean_forward(LMBatch(tokens), keys | ci_fn.capture_keys)

    def loss(parameters: tuple[ComponentStacks, BlockSelectedChunkwiseTransformerCIFn]):
        vu, ci = parameters
        coefficients = ci.prepare()(
            clean.captures, clean.conditioning, components, sequence=None, remat=True
        ).lower
        masked = placed.masked_forward(
            placed.prepare_compute_weights(vu),
            clean.conditioning,
            masking=placed.model.prepare_masking(
                MaterializedMasking(component_masks=coefficients, weight_delta_masks=None)
            ),
            routes=None,
            capture_keys=keys,
            remat=True,
        )
        return _router_kl(clean.captures, masked.captures)

    value, (vu_grad, ci_grad) = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=False))(
        (components, ci_fn)
    )
    assert float(value) > 0.0
    assert jnp.linalg.norm(vu_grad.stacks["gdn_out"][0]) > 0
    for kind in ("experts_gate", "experts_up", "experts_down"):
        for leaf in vu_grad.stacks[kind]:
            np.testing.assert_array_equal(leaf[-1], jnp.zeros_like(leaf[-1]))
    ci_arrays = jax.tree.leaves(eqx.filter(ci_grad, eqx.is_array))
    assert sum(float(jnp.sum(jnp.square(leaf))) for leaf in ci_arrays) > 0


@pytest.mark.multidevice
@pytest.mark.parametrize(
    "implementation,tp", [("dense_masked", 1), ("dense_masked", 2), ("ragged_dot", 2)]
)
def test_router_kl_gradient_preserves_dense_and_expert_placement(
    implementation: ExpertImplementation, tp: int
) -> None:
    if jax.device_count() < 2 * tp:
        pytest.skip("requires four simulated CPU devices")
    model, components, tokens = _fixture()
    model = replace(model, expert_implementation=implementation)
    keys = _keys(model)
    reference = PlacedModel(model, None)
    clean = reference.clean_forward(LMBatch(tokens), keys)
    values = random_mask_values(model, tokens.shape, jax.random.key(5))

    def gradient(
        placed: PlacedModel[
            LMBatch,
            LMOutput,
            QwenPreparedWeights,
            LMBatchWithRouting[LMBatch],
            QwenPreparedMasking,
        ],
        vu: ComponentStacks,
        conditioning: LMBatchWithRouting[LMBatch],
        mask_values: dict[str, jax.Array],
        clean_captures: dict[str, jax.Array],
    ):
        masks = site_masks(model, conditioning.selection, mask_values)

        def loss(candidate: ComponentStacks):
            # fp32 operands isolate placement algebra from bf16 reduction rounding.
            masked = placed.masked_forward(
                placed.model.prepare_compute_weights(candidate, placed.placement),
                conditioning,
                masking=placed.model.prepare_masking(
                    MaterializedMasking(component_masks=masks, weight_delta_masks=None)
                ),
                routes=None,
                capture_keys=keys,
                remat=True,
            )
            return _router_kl(clean_captures, masked.captures)

        return jax.value_and_grad(loss)(vu)

    expected = jax.jit(gradient)(reference, components, clean.conditioning, values, clean.captures)
    mesh = Mesh(
        np.asarray(jax.devices()[: 2 * tp]).reshape(2, tp),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    rules = from_config("zero1-replicated-resident-moe", mesh, model.sites)
    placed = place_target(model, rules)
    with jax.set_mesh(mesh):
        vu = init_component_stacks_placed(model.sites, jax.random.key(1), rules)
        batch = jax.device_put(tokens, NamedSharding(mesh, P("data", None)))
        placed_clean = placed.clean_forward(LMBatch(batch), keys)
        placed_values = jax.tree.map(
            lambda x: jax.device_put(x, NamedSharding(mesh, P("data", None, None))), values
        )
        actual = jax.jit(gradient)(
            placed, vu, placed_clean.conditioning, placed_values, placed_clean.captures
        )
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(got, want, rtol=4e-3, atol=3e-5)


@pytest.mark.multidevice
def test_pooled_step_keeps_e2e_source_ascent_separate_from_router_reconstruction() -> None:
    from param_decomp.core.adversary import PersistentAdversary, init_sources_opt_state
    from param_decomp.core.configs import (
        FaithfulnessLossConfig,
        ImportanceMinimalityLossConfig,
        MergedStochasticSubsetPooledPPGDReconLossConfig,
    )
    from param_decomp.core.faithfulness import faithfulness_loss_for
    from param_decomp.core.init_placed import init_source_pool_sharded
    from param_decomp.core.objective import build_objective
    from param_decomp.core.train import (
        Decomposition,
        ForwardSubstrate,
        PDState,
        PDTrainingState,
        TrainState,
        make_train_step,
    )
    from param_decomp.targets.testing import tiny_qwen36_moe_ci_fn_arch
    from param_decomp.tests.placed_ci_fn import placed_ci_fn

    if jax.device_count() < 2:
        pytest.skip("requires two simulated CPU devices")
    model, _, tokens = _fixture()
    mesh = Mesh(
        np.asarray(jax.devices()[:2]).reshape(2, 1),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    rules = from_config("zero1-replicated-resident-moe", mesh, model.sites)
    placed = place_target(model, rules)
    arch = replace(tiny_qwen36_moe_ci_fn_arch(model), expert_implementation="dense_masked")
    raw = {
        "coeff": 1.0,
        "adv_fraction": 0.5,
        "pool": {"size_per_batch_element": 4},
        "n_warmup_steps": 1,
        "routing": {"type": "uniform_k_subset"},
        "optimizer": {"type": "adam", "beta1": 0.01, "beta2": 0.99, "lr_schedule": 0.02},
        "adversary_objective": "e2e",
    }
    baseline = MergedStochasticSubsetPooledPPGDReconLossConfig.model_validate(raw)
    comparisons = [
        {"capture": key, "distance": "categorical_kl_from_logits"}
        for key in model.router_logits_capture_keys
    ]
    auxiliaries = [{"name": "router_kl", "coeff": 2.0, "comparisons": comparisons}]
    auxiliary = MergedStochasticSubsetPooledPPGDReconLossConfig.model_validate(
        {**raw, "auxiliaries": auxiliaries}
    )
    full_term = MergedStochasticSubsetPooledPPGDReconLossConfig.model_validate(
        {**raw, "auxiliaries": auxiliaries, "adversary_objective": "term"}
    )

    def objective_for(term: MergedStochasticSubsetPooledPPGDReconLossConfig):
        return build_objective(
            (
                FaithfulnessLossConfig(coeff=1.0),
                ImportanceMinimalityLossConfig(coeff=1e-4, gamma=ScheduleConfig.constant(1.0)),
                term,
            ),
            model.sites,
        )

    opt = _adamw_optimizer(AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-3)), 1)
    with jax.set_mesh(mesh):
        components = init_component_stacks_placed(model.sites, jax.random.key(1), rules)
        ci = placed_ci_fn(arch, model.sites, jax.random.key(4), mesh, rules)
        sources = init_source_pool_sharded(
            model.sites, baseline.pool, tokens.shape[0], jnp.float32, jax.random.key(7), mesh
        )
        initial = TrainState(
            decomposition=Decomposition(components=components, ci_fn=ci),
            training=PDTrainingState(
                frequency=BatchFrequency(),
                objective=objective_for(baseline),
                components_opt_state=opt.init(eqx.filter(components, eqx.is_array)),
                ci_fn_opt_state=opt.init(eqx.filter(ci, eqx.is_array)),
                adversaries={
                    baseline.type: PersistentAdversary(
                        sources=sources,
                        opt_state=init_sources_opt_state(baseline.optimizer, sources),
                        state_key=baseline.type,
                        optimizer=baseline.optimizer,
                        n_warmup=baseline.n_warmup_steps,
                    )
                },
                step=jnp.zeros((), jnp.int32),
            ),
        )
        batch = jax.device_put(tokens, NamedSharding(mesh, P("data", None)))
        substrate = ForwardSubstrate.of(
            placed,
            remat_recon_forwards=True,
            remat_ci_fn=True,
            ci_capture_keys=ci.capture_keys,
        )

        def advance(
            term: MergedStochasticSubsetPooledPPGDReconLossConfig,
        ) -> tuple[PDState[LMBatchWithRouting[LMBatch]], dict[str, jax.Array]]:
            objective = objective_for(term)
            step = jax.jit(
                make_train_step(
                    model_static=placed,
                    substrate=substrate,
                    components_optimizer=opt,
                    ci_fn_optimizer=opt,
                    total_steps=4,
                    faithfulness=faithfulness_loss_for(placed),
                )
            )
            state = jax.tree.map(lambda leaf: leaf.copy(), initial)
            state = replace(
                state,
                training=replace(state.training, objective=objective),
            )
            return step(placed, state, LMBatch(batch.copy()), jax.random.key(8))

        plain, plain_metrics = advance(baseline)
        with_aux, aux_metrics = advance(auxiliary)
        with_term, term_metrics = advance(full_term)
    for metrics in (plain_metrics, aux_metrics, term_metrics):
        assert jnp.isfinite(metrics["total"])
    assert float(aux_metrics["total"]) > float(plain_metrics["total"])
    plain_source = plain.training.adversaries[baseline.type]
    aux_source = with_aux.training.adversaries[baseline.type]
    term_source = with_term.training.adversaries[baseline.type]
    for got, want in zip(jax.tree.leaves(aux_source), jax.tree.leaves(plain_source), strict=True):
        np.testing.assert_allclose(got, want, rtol=2e-5, atol=1e-7)
    for field in ("components", "ci_fn"):
        before = eqx.filter(getattr(plain.decomposition, field), eqx.is_array)
        after = eqx.filter(getattr(with_aux.decomposition, field), eqx.is_array)
        assert any(
            not np.array_equal(a, b)
            for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True)
        ), field
    assert any(
        not np.allclose(a, b, rtol=1e-5, atol=1e-8)
        for a, b in zip(jax.tree.leaves(term_source), jax.tree.leaves(aux_source), strict=True)
    ), "term-mode source ascent must include router reconstruction"
