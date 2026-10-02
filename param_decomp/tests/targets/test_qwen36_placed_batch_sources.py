"""Placed per-datapoint source updates: uint16 storage, slot-local gradients and collectives."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core.components import init_component_stacks
from param_decomp.core.configs import AdamWOptimizerConfig
from param_decomp.core.init_placed import init_component_stacks_placed
from param_decomp.core.losses import BatchFrequency
from param_decomp.core.model import PlacedModel
from param_decomp.core.placement import batch_axes, from_config
from param_decomp.core.run_state import _adamw_optimizer
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.sharding import place_target
from param_decomp.core.tools.hlo_census import collective_census
from param_decomp.lm.batch import LMBatch, LMBatchWithRouting
from param_decomp.tests.targets.qwen36_placement_helpers import (
    BATCH,
    CENSUS_CS,
    DATA,
    SEQ,
    TP,
    assert_no_weight_gather_in_any_loop,
    batch_placed,
    model_and_components,
    moe_ci_fn_arch,
    multidevice,
    placement_mesh,
)


@multidevice
@pytest.mark.multidevice
def test_placed_train_step_with_bsc_sources_census_and_slot_locality():
    """The large-capacity adversary placed: persistent-PGD `bsc` sources (per-datapoint
    slots, uint16 fixed-point values with stochastic-rounding stores, source optimizer
    `momentum_sgd` with its bf16 velocity) through the REAL train step with narrow
    emission on the owner moe preset. Pins the double sharding {batch: data, expert: tp}
    on the stored sources, per-slot update parity against the unplaced step (each batch
    slot ascends by ITS datapoint's gradient — the bsc semantics), representation +
    sharding preserved through the step, and the compiled step's census: nothing
    tensor-sized crosses `data` inside any loop — the warmup scan's global-mean loss
    scalars and the CI bias/norm-vector grads are the only sanctioned in-loop
    cross-data reductions. The `replicate` arm only: sequence parallelism's placement is
    pinned by its own census and stochastic-parity tests, and the full train step is the
    most expensive program in the suite. Reconstruction here is persistent-only;
    combined stochastic + persistent loss execution is covered by
    test_qwen36_narrow.test_e2e_train_step_with_narrow_emission."""

    from param_decomp.core.adversary import (
        BlockedSourceComponents as EBS,
    )
    from param_decomp.core.adversary import (
        PersistentAdversary,
        SourceComponents,
        init_persistent_sources,
        init_sources_opt_state,
    )
    from param_decomp.core.ci_fn.implementations.block_selected.arch import (
        BlockSelectedChunkwiseTransformerCIFn,
    )
    from param_decomp.core.components import BlockedFactorization
    from param_decomp.core.configs import (
        FaithfulnessLossConfig,
        ImportanceMinimalityLossConfig,
        MomentumSgdPGDConfig,
        PersistentPGDReconLossConfig,
    )
    from param_decomp.core.faithfulness import faithfulness_loss_for
    from param_decomp.core.init_placed import init_sources_sharded
    from param_decomp.core.model import Positioned
    from param_decomp.core.objective import build_objective
    from param_decomp.core.schedule import Knot
    from param_decomp.core.train import (
        Decomposition,
        ForwardSubstrate,
        PDState,
        PDTrainingState,
        TrainState,
        make_train_step,
    )
    from param_decomp.tests.placed_ci_fn import placed_ci_fn

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
        ppgd,
    )
    model, _components = model_and_components(CENSUS_CS)
    sites = model.sites
    arch = moe_ci_fn_arch(model.cfg)
    objective = build_objective(objective_cfgs, model.sites)
    opt_vu = _adamw_optimizer(AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-3)), 1)
    opt_ci = _adamw_optimizer(AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-3)), 1)
    tokens = jax.random.randint(jax.random.PRNGKey(3), (BATCH, SEQ), 0, model.cfg.vocab_size)
    src_key = jax.random.PRNGKey(7)
    step_keys = [jax.random.fold_in(jax.random.PRNGKey(4), i) for i in range(2)]

    def run_two_steps(placed: bool) -> PDState[LMBatchWithRouting[LMBatch]]:
        mesh = placement_mesh() if placed else None
        if placed:
            rules = from_config("owner-replicated-resident-moe", placement_mesh(), sites)
            target = place_target(model, rules)
        else:
            rules = None
            target = PlacedModel(model=model, placement=None)

        def build_and_step() -> PDState[LMBatchWithRouting[LMBatch]]:
            if rules is not None:
                components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
                ci_fn = placed_ci_fn(arch, sites, jax.random.PRNGKey(2), placement_mesh(), rules)
                sources = init_sources_sharded(
                    sites, Positioned(SEQ), "bsc", BATCH, jnp.uint16, src_key, placement_mesh()
                )
                batch = batch_placed(tokens, placement_mesh())
            else:
                components = init_component_stacks(sites, jax.random.PRNGKey(1))
                ci_fn = arch.initialize(sites, None, jax.random.PRNGKey(2))
                sources = init_persistent_sources(sites, (BATCH, SEQ), jnp.uint16, src_key)
                batch = tokens
            assert isinstance(ci_fn, BlockSelectedChunkwiseTransformerCIFn)
            state = TrainState(
                decomposition=Decomposition[LMBatchWithRouting[LMBatch]](
                    components=components, ci_fn=ci_fn
                ),
                training=PDTrainingState(
                    frequency=BatchFrequency(),
                    objective=objective,
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
                    step=jnp.zeros((), jnp.int32),
                ),
            )
            step_fn = jax.jit(
                make_train_step(
                    model_static=target,
                    substrate=ForwardSubstrate.of(
                        target,
                        remat_recon_forwards=True,
                        remat_ci_fn=True,
                        ci_capture_keys=ci_fn.capture_keys,
                    ),
                    components_optimizer=opt_vu,
                    ci_fn_optimizer=opt_ci,
                    total_steps=4,
                    faithfulness=faithfulness_loss_for(target),
                )
            )
            run = step_fn
            if rules is not None:
                # the census: the compiled step's in-loop collectives — an OUTER plain
                # jit (the engine's eqx jit inlines under it) so the lowered text is
                # reachable, the fit check's own AOT pattern. The same executable then
                # runs the steps: the placed step is compiled exactly once.
                outer = jax.jit(lambda m, s, b, k: step_fn(m, s, b, k))
                compiled = outer.lower(target, state, LMBatch(batch), step_keys[0]).compile()
                hlo = compiled.as_text()
                assert hlo is not None
                census = collective_census(hlo, replica_stride=TP, n_devices=DATA * TP)
                assert all(size <= 2**12 for size in census.in_loop_cross_replicate_bytes), (
                    census.in_loop_cross_replicate_bytes,
                    census.counts,
                )
                assert_no_weight_gather_in_any_loop(hlo, BATCH // DATA)
                run = compiled
            for key in step_keys:
                state, metrics = run(target, state, LMBatch(batch), key)
                assert jnp.isfinite(metrics["total"]), metrics["total"]
            return state

        if mesh is not None:
            with jax.set_mesh(mesh):
                return build_and_step()
        return build_and_step()

    expert_site = next(s.name for s in sites if isinstance(s.factorization, BlockedFactorization))
    dense_site = next(
        s.name for s in sites if not isinstance(s.factorization, BlockedFactorization)
    )

    placed_state = run_two_steps(placed=True)
    placed_stacks = placed_state.training.adversaries[ppgd.type].sources
    mesh = placement_mesh()
    expert_group, _ = placed_stacks.stack_index_of(expert_site)
    dense_group, _ = placed_stacks.stack_index_of(dense_site)
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
