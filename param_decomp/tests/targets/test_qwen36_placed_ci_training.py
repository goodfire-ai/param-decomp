"""Placed narrow MoE CI: real train steps with AdamW and stacked Muon."""

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.core.configs import (
    AdamWOptimizerConfig,
    MuonOptimizerConfig,
    PlacementPresetName,
)
from param_decomp.core.init_placed import init_component_stacks_placed
from param_decomp.core.losses import BatchFrequency
from param_decomp.core.placement import from_config
from param_decomp.core.run_state import _adamw_optimizer, _muon_optimizer
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.sharding import place_target
from param_decomp.lm.batch import LMBatch
from param_decomp.tests.targets.qwen36_placement_helpers import (
    BATCH,
    CENSUS_CS,
    SEQ,
    batch_placed,
    model_and_components,
    moe_ci_fn_arch,
    multidevice,
    placement_mesh,
)


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
def test_placed_full_train_step_runs_with_the_moe_ci_fn(
    preset: PlacementPresetName, optimizer_impl: Literal["adamw", "stacked_muon"]
):
    """The REAL train step with the MoE chunkwise CI fn — narrow emission through
    smooth-L0 imp-min (the no-[C]-accumulator lp path), stochastic recon whose narrow
    masks drive the routed decomposed expert arm, and faithfulness — placed at the moe
    resident presets with their paired optimizers: two steps, finite, V and the CI
    expert banks both move. This exercises the master-layout V·U faithfulness
    contraction and owner/Muon's Newton–Schulz staging
    from stack-cut masters. TP=1 is covered by the pooled-source training test."""
    import equinox as eqx

    from param_decomp.core.ci_fn.implementations.block_selected.arch import (
        BlockSelectedChunkwiseTransformerCIFn,
    )
    from param_decomp.core.components import group_factorizations
    from param_decomp.core.configs import (
        FaithfulnessLossConfig,
        ImportanceMinimalityLossConfig,
        StochasticReconLossConfig,
    )
    from param_decomp.core.faithfulness import faithfulness_loss_for
    from param_decomp.core.objective import build_objective
    from param_decomp.core.placement import (
        assert_stacked_muon_component_staging,
        ns_staging_sharding,
    )
    from param_decomp.core.run_state import component_muon_dimension_numbers
    from param_decomp.core.schedule import Knot
    from param_decomp.core.train import (
        Decomposition,
        ForwardSubstrate,
        PDTrainingState,
        TrainState,
        make_train_step,
    )
    from param_decomp.tests.placed_ci_fn import placed_ci_fn

    mesh = placement_mesh()
    model, _components = model_and_components(CENSUS_CS)
    sites = model.sites
    rules = from_config(preset, mesh, sites)
    placed_model = place_target(model, rules)
    arch = moe_ci_fn_arch(model.cfg)
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
        model.sites,
    )
    match optimizer_impl:
        case "adamw":
            opt_vu = _adamw_optimizer(
                AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-3)), 1
            )
        case "stacked_muon":
            assert_stacked_muon_component_staging(rules)
            waypoint = ns_staging_sharding(rules.components.ns_compute, ("stack",))
            opt_vu = _muon_optimizer(
                MuonOptimizerConfig(type="muon", lr_schedule=ScheduleConfig.constant(1e-3)),
                1,
                component_muon_dimension_numbers(group_factorizations(sites)),
                lambda tree: jax.tree.map(lambda _: waypoint, tree),
            )
    opt_ci = _adamw_optimizer(AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-3)), 1)

    with jax.set_mesh(mesh):
        components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
        ci_fn = placed_ci_fn(arch, sites, jax.random.PRNGKey(2), mesh, rules)
        assert isinstance(ci_fn, BlockSelectedChunkwiseTransformerCIFn)
        state = TrainState(
            decomposition=Decomposition(components=components, ci_fn=ci_fn),
            training=PDTrainingState(
                frequency=BatchFrequency(),
                objective=objective,
                components_opt_state=opt_vu.init(eqx.filter(components, eqx.is_array)),
                ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
                adversaries={},
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
        bank_before = np.asarray(ci_fn.chunks.blocks[0].selected_gate[0])
        for step_index in range(2):
            state, metrics = step_fn(
                placed_model,
                state,
                LMBatch(tokens),
                jax.random.fold_in(jax.random.PRNGKey(4), step_index),
            )
            assert jnp.isfinite(metrics["total"]), (step_index, metrics["total"])
        v_moved = np.asarray(state.decomposition.components.stacks["experts_gate"][0])
        ci_moved = state.decomposition.ci_fn
        assert isinstance(ci_moved, BlockSelectedChunkwiseTransformerCIFn)
        bank_moved = np.asarray(ci_moved.chunks.blocks[0].selected_gate[0])
    assert not np.allclose(v_moved, v_before), "V did not move — the step is a no-op"
    assert not np.allclose(bank_moved, bank_before), "the CI expert bank did not move"
