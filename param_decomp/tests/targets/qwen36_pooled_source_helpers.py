"""Shared training assertions for the three placed Qwen pooled-source configurations."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh

from param_decomp.core.configs import AdamWOptimizerConfig
from param_decomp.core.init_placed import init_component_stacks_placed
from param_decomp.core.placement import from_config
from param_decomp.core.run_state import _adamw_optimizer
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.sharding import place_target
from param_decomp.lm.batch import LMBatch
from param_decomp.routed.experts import ExpertImplementation
from param_decomp.tests.targets.qwen36_placement_helpers import (
    BATCH,
    CENSUS_CS,
    DATA,
    SEQ,
    batch_placed,
    model_and_components,
    moe_ci_fn_arch,
)


def assert_placed_pooled_source_train_step(
    *,
    implementation: ExpertImplementation,
    tp: int,
    include_mixers: bool,
) -> None:
    """Merged pooled PGD updates sources, components, selected CI and sharded frequency EMA."""

    from param_decomp.core.adversary import (
        BlockedSourceComponents,
        PersistentAdversary,
        init_sources_opt_state,
    )
    from param_decomp.core.ci_fn.implementations.block_selected.arch import (
        BlockSelectedChunkwiseTransformerCIFn,
    )
    from param_decomp.core.configs import (
        AdamPGDConfig,
        FaithfulnessLossConfig,
        FrequencyMinimalityConfig,
        ImportanceMinimalityLossConfig,
        MergedStochasticSubsetPooledPPGDReconLossConfig,
        NonlinearityLocalityLossConfig,
        SourcePoolConfig,
        UniformKSubsetRoutingConfig,
    )
    from param_decomp.core.faithfulness import faithfulness_loss_for
    from param_decomp.core.init_placed import (
        init_frequency_estimator_placed,
        init_source_pool_sharded,
    )
    from param_decomp.core.losses import EmaFrequency
    from param_decomp.core.objective import build_objective
    from param_decomp.core.schedule import Knot
    from param_decomp.core.train import (
        Decomposition,
        ForwardSubstrate,
        PDTrainingState,
        TrainState,
        make_train_step,
    )
    from param_decomp.targets.qwen36_moe import KIND_ORDER, sublayer_of
    from param_decomp.targets.testing import tiny_qwen36_moe_ci_fn_arch
    from param_decomp.tests.placed_ci_fn import placed_ci_fn

    mesh = Mesh(
        np.asarray(jax.devices()[: DATA * tp]).reshape(DATA, tp),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    cs = dict(CENSUS_CS)
    if include_mixers:
        cs.update({kind: 8 for kind in KIND_ORDER if sublayer_of(kind) != "moe"})
    model, _components = model_and_components(cs)
    model = dataclasses.replace(model, expert_implementation=implementation)
    sites = model.sites
    rules = from_config("zero1-replicated-resident-moe", mesh, sites)
    placed_model = place_target(model, rules)
    base_arch = tiny_qwen36_moe_ci_fn_arch(model) if include_mixers else moe_ci_fn_arch(model.cfg)
    arch = dataclasses.replace(
        base_arch,
        expert_implementation=implementation,
    )
    pooled = MergedStochasticSubsetPooledPPGDReconLossConfig(
        coeff=1.0,
        adv_fraction=ScheduleConfig.constant(0.5),
        pool=SourcePoolConfig(size_per_batch_element=2),
        n_warmup_steps=2,
        routing=UniformKSubsetRoutingConfig(),
        optimizer=AdamPGDConfig(beta1=0.01, beta2=0.99, lr_schedule=ScheduleConfig.constant(0.02)),
    )
    frequency = FrequencyMinimalityConfig(
        coeff=1e-5, reference_datapoint_count=BATCH * SEQ, ema_halflife_steps=8
    )
    objective = build_objective(
        (
            FaithfulnessLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(
                coeff=1e-4,
                frequency=frequency,
                gamma=ScheduleConfig(
                    max_val=1.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.01))
                ),
            ),
            pooled,
            *(
                (
                    NonlinearityLocalityLossConfig(
                        coeff=3e-5,
                        relative_threshold=ScheduleConfig.constant(4.0),
                        unit_kind_coefficients={
                            "neuron": 1.0,
                            "attention_head": 1.0,
                            "deltanet_head": 1.0,
                        },
                    ),
                )
                if include_mixers
                else ()
            ),
        ),
        model.sites,
    )
    opt_vu = _adamw_optimizer(AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-3)), 1)
    opt_ci = _adamw_optimizer(AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-3)), 1)

    with jax.set_mesh(mesh):
        components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
        if tp == 1:
            prepared = placed_model.prepare_compute_weights(components)
            for leaf in jax.tree.leaves(prepared):
                assert leaf.sharding.is_fully_replicated, leaf.sharding
        ci_fn = placed_ci_fn(arch, sites, jax.random.PRNGKey(2), mesh, rules)
        assert isinstance(ci_fn, BlockSelectedChunkwiseTransformerCIFn)
        sources = init_source_pool_sharded(
            sites, pooled.pool, BATCH, jnp.float32, jax.random.key(7), mesh
        )
        source_components = sources.stacks["experts_gate"].components
        assert isinstance(source_components, BlockedSourceComponents)
        source_before = np.asarray(source_components.values)
        state = TrainState(
            decomposition=Decomposition(components=components, ci_fn=ci_fn),
            training=PDTrainingState(
                frequency=init_frequency_estimator_placed(
                    frequency, sites, rules.frequency_sharding
                ),
                objective=objective,
                components_opt_state=opt_vu.init(eqx.filter(components, eqx.is_array)),
                ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
                adversaries={
                    pooled.type: PersistentAdversary(
                        sources=sources,
                        opt_state=init_sources_opt_state(pooled.optimizer, sources),
                        state_key=pooled.type,
                        optimizer=pooled.optimizer,
                        n_warmup=pooled.n_warmup_steps,
                    )
                },
                step=jnp.zeros((), jnp.int32),
            ),
        )
        step_fn = jax.jit(
            make_train_step(
                model_static=placed_model,
                substrate=ForwardSubstrate.of(
                    placed_model,
                    remat_recon_forwards=True,
                    remat_ci_fn=True,
                    ci_capture_keys=ci_fn.capture_keys,
                ),
                components_optimizer=opt_vu,
                ci_fn_optimizer=opt_ci,
                total_steps=4,
                faithfulness=faithfulness_loss_for(placed_model),
            )
        )
        tokens = batch_placed(
            jax.random.randint(jax.random.PRNGKey(3), (BATCH, SEQ), 0, model.cfg.vocab_size), mesh
        )
        v_before = np.asarray(components.stacks["experts_gate"][0])
        mixer_before = np.asarray(components.stacks["gdn_q"][0]) if include_mixers else None
        bank_before = np.asarray(ci_fn.chunks.blocks[0].selected_gate[0])
        for step_index in range(2):
            state, metrics = step_fn(
                placed_model,
                state,
                LMBatch(tokens),
                jax.random.fold_in(jax.random.PRNGKey(4), step_index),
            )
            assert jnp.isfinite(metrics["total"]), (step_index, metrics["total"])
            assert f"loss/{pooled.type}" in metrics
            estimator = state.training.frequency
            assert isinstance(estimator, EmaFrequency)
            assert estimator.estimate.keys() == set(model.site_names)
            for site, ema in estimator.estimate.items():
                assert ema.sharding == rules.frequency_sharding, (step_index, site)
                frequencies = np.asarray(ema)
                assert np.isfinite(frequencies).all(), (step_index, site)
                assert np.any(frequencies > 0), (step_index, site)
            if include_mixers:
                assert "loss/NonlinearityLocalityLoss_deltanet_head" in metrics
        if mixer_before is not None:
            assert not np.allclose(state.decomposition.components.stacks["gdn_q"][0], mixer_before)
        v_moved = np.asarray(state.decomposition.components.stacks["experts_gate"][0])
        ci_moved = state.decomposition.ci_fn
        assert isinstance(ci_moved, BlockSelectedChunkwiseTransformerCIFn)
        bank_moved = np.asarray(ci_moved.chunks.blocks[0].selected_gate[0])
    assert not np.allclose(v_moved, v_before), "V did not move — the step is a no-op"
    assert not np.allclose(bank_moved, bank_before), "the CI expert bank did not move"

    source_components = (
        state.training.adversaries[pooled.type].sources.stacks["experts_gate"].components
    )
    assert isinstance(source_components, BlockedSourceComponents)
    source_after = np.asarray(source_components.values)
    assert not np.allclose(source_after, source_before), "the merged source pool did not update"
