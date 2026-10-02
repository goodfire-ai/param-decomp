"""Construct decompositions and fresh training state for plain and targeted PD."""

from collections.abc import Callable, Mapping
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random
from jax.sharding import Mesh, NamedSharding
from jaxtyping import Array, Float32, PRNGKeyArray

from param_decomp.core.adversary import PersistentAdversary, init_sources_opt_state
from param_decomp.core.ci_fn.interface import CIFn
from param_decomp.core.ci_fn.optimizer import ci_fn_muon_dimension_numbers, ci_fn_muon_waypoints
from param_decomp.core.components import (
    COMPONENT_STACK_AXES,
    BlockedFactorization,
    ComponentStacks,
    DenseFactorization,
    Factorization,
    SiteSpec,
    group_factorizations,
)
from param_decomp.core.configs import (
    AdamWOptimizerConfig,
    ImportanceMinimalityLossConfig,
    MuonOptimizerConfig,
    NontargetConfig,
    PDConfig,
    PDConfigBase,
    TargetedPDConfig,
)
from param_decomp.core.init_placed import (
    CIFnInitializer,
    ComponentInitializer,
    init_frequency_estimator_placed,
    init_model_component_stacks_placed,
    init_persistent_sources_from_config,
    random_component_initializer,
)
from param_decomp.core.model import ComponentActivations, PlacedModel, PositionAxis, Positioned
from param_decomp.core.muon_stacked import NSWaypoints, stacked_muon
from param_decomp.core.objective import (
    build_objective,
    build_targeted_objective,
)
from param_decomp.core.optimizer import ScheduledOptimizer, ScheduledOptimizerState
from param_decomp.core.placement import (
    PlacementRules,
    assert_stacked_muon_component_staging,
    ns_staging_sharding,
)
from param_decomp.core.recon import AnyReconLossTerm, persistent_configs
from param_decomp.core.runtime_schedule import RuntimeSchedule, train_frac_at
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.train import (
    CIScaledWeightDecay,
    Decomposition,
    PDState,
    PDTrainingState,
    TargetedPDState,
    TargetedPDTrainingState,
    TrainState,
)


def clip_by_global_norm_with_eps(max_norm: float, eps: float) -> optax.GradientTransformation:
    """Global-norm clip matching torch's `clip_grad_norm_`: scale by
    `clip(max_norm / (global_norm + eps), max=1)`. optax's `clip_by_global_norm` omits
    `eps`; at small `max_norm` (0.01) the clip fires almost every step so this ~1e-4
    relative offset is per-step."""

    def init(params: optax.Params) -> optax.EmptyState:
        del params
        return optax.EmptyState()

    def update(
        updates: optax.Updates, state: optax.OptState, params: optax.Params | None = None
    ) -> tuple[optax.Updates, optax.OptState]:
        del params
        global_norm = optax.global_norm(updates)
        scale = jnp.minimum(max_norm / (global_norm + eps), 1.0)
        updates = jax.tree.map(lambda g: g * scale, updates)
        return updates, state

    return optax.GradientTransformation(init, update)


def component_muon_dimension_numbers(
    factorizations: Mapping[str, Factorization],
) -> Callable[[optax.Params], optax.Params]:
    """Label the V/U tree's muon leaves from each group's declared factorization. The
    labeling never inspects leaf rank, so a factorization without an arm here fails at
    optimizer build instead of silently falling back to Adam."""

    def label(params: optax.Params) -> optax.Params:
        stacks = params
        assert isinstance(stacks, ComponentStacks), type(stacks)
        assert stacks.stacks.keys() == factorizations.keys(), (
            sorted(stacks.stacks),
            sorted(factorizations),
        )

        def group_dims(group: str) -> optax.contrib.MuonDimensionNumbers:
            match factorizations[group]:
                case DenseFactorization():
                    # [stack, a, b]: orthogonalize the trailing matrix, stack batched.
                    return optax.contrib.MuonDimensionNumbers(reduction_axis=-2, output_axis=-1)
                case BlockedFactorization():
                    # [stack, block, a, b]: each block's matrix orthogonalized on its
                    # own, both leading axes batched (stacked NS folds them into one —
                    # `muon_stacked._canonicalize`).
                    return optax.contrib.MuonDimensionNumbers(reduction_axis=-2, output_axis=-1)

        labeled = {group: (group_dims(group), group_dims(group)) for group in stacks.stacks}
        return cast(
            optax.Params,
            cast(
                object,
                ComponentStacks(
                    stacks=labeled,
                    site_stack_indices=stacks.site_stack_indices,
                    stack_pads=stacks.stack_pads,
                ),
            ),
        )

    return label


def _adamw_optimizer(opt: AdamWOptimizerConfig, total_steps: int) -> ScheduledOptimizer:
    def optimizer(lr: Float32[Array, ""]) -> optax.GradientTransformation:
        return optax.adamw(
            lr, b1=opt.betas[0], b2=opt.betas[1], eps=1e-8, weight_decay=opt.weight_decay
        )

    return _scheduled_optimizer(optimizer, opt.lr_schedule, total_steps, opt.grad_clip_norm)


def _muon_optimizer(
    opt: MuonOptimizerConfig,
    total_steps: int,
    muon_dimension_numbers: Callable[[optax.Params], optax.Params] | None,
    waypoints: NSWaypoints | None,
) -> ScheduledOptimizer:
    def optimizer(lr: Float32[Array, ""]) -> optax.GradientTransformation:
        return stacked_muon(
            lr,
            beta=opt.beta,
            weight_decay=opt.weight_decay,
            consistent_rms=opt.consistent_rms,
            muon_weight_dimension_numbers=muon_dimension_numbers,
            ns_steps=opt.ns_steps,
            ns_dtype=jnp.dtype(opt.ns_dtype),
            waypoints=waypoints,
        )

    return _scheduled_optimizer(optimizer, opt.lr_schedule, total_steps, opt.grad_clip_norm)


def _scheduled_optimizer(
    inner: Callable[[Float32[Array, ""]], optax.GradientTransformation],
    lr_schedule: ScheduleConfig,
    total_steps: int,
    grad_clip_norm: float | None,
) -> ScheduledOptimizer:
    def optimizer(lr: Float32[Array, ""]) -> optax.GradientTransformation:
        if grad_clip_norm is None:
            return inner(lr)
        return optax.chain(clip_by_global_norm_with_eps(grad_clip_norm, eps=1e-6), inner(lr))

    def init(params: optax.Params) -> ScheduledOptimizerState:
        schedule = RuntimeSchedule.from_coeff(lr_schedule)
        count = jnp.zeros((), jnp.int32)
        lr = schedule.at(train_frac_at(count, total_steps))
        return ScheduledOptimizerState(schedule, count, lr, optimizer(lr).init(params))

    def update(
        updates: optax.Updates, state: ScheduledOptimizerState, params: optax.Params
    ) -> tuple[optax.Updates, ScheduledOptimizerState]:
        fraction = jnp.minimum(train_frac_at(state.count, total_steps), 1.0)
        lr = state.schedule.at(fraction)
        updates, inner_state = optimizer(lr).update(updates, state.inner_state, params)
        return updates, ScheduledOptimizerState(
            state.schedule,
            jnp.asarray(optax.safe_increment(state.count), jnp.int32),
            lr,
            inner_state,
        )

    return ScheduledOptimizer(init, update)


def _uniform_waypoints(sharding: NamedSharding) -> NSWaypoints:
    """Stage every component matrix at the shared Newton–Schulz placement."""
    return lambda tree: jax.tree.map(lambda _: sharding, tree)


def build_optimizers(
    pd: PDConfigBase,
    placement: PlacementRules,
    sites: tuple[SiteSpec, ...],
) -> tuple[ScheduledOptimizer, ScheduledOptimizer]:
    """Build the component and CI optimizers from their authored configs.

    Components supply their matrix structure through site factorizations and stage at
    the table's Newton–Schulz row; the CI parameters declare their independent matrices
    together with their own staging."""
    match pd.components_optimizer:
        case AdamWOptimizerConfig() as opt:
            opt_vu = _adamw_optimizer(opt, pd.steps)
        case MuonOptimizerConfig() as opt:
            assert_stacked_muon_component_staging(placement)
            opt_vu = _muon_optimizer(
                opt,
                pd.steps,
                component_muon_dimension_numbers(group_factorizations(sites)),
                _uniform_waypoints(
                    ns_staging_sharding(placement.components.ns_compute, COMPONENT_STACK_AXES)
                ),
            )
    match pd.ci_fn_optimizer:
        case AdamWOptimizerConfig() as opt:
            opt_ci = _adamw_optimizer(opt, pd.steps)
        case MuonOptimizerConfig() as opt:
            opt_ci = _muon_optimizer(
                opt, pd.steps, ci_fn_muon_dimension_numbers, ci_fn_muon_waypoints()
            )
    return opt_vu, opt_ci


def _placed_init_geometry[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
) -> tuple[PlacementRules, Mesh]:
    """The bundle's own rules + mesh. Seeded init places real arrays, so an unplaced
    bundle and the abstract (spec-check) arm of `PlacementRules.mesh` are both refused."""
    rules = model.placement
    assert rules is not None, "seeded init is placed init: the bundle must carry rules"
    mesh = rules.mesh
    assert isinstance(mesh, Mesh), type(mesh)
    return rules, mesh


def init_decomposition[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    ci_fn_initializer: CIFnInitializer[Conditioning],
    init_key: PRNGKeyArray,
    component_initializer: ComponentInitializer[
        TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT
    ] = random_component_initializer,
) -> Decomposition[Conditioning]:
    """Initialize the trained product independently so a consumer can
    `jax.eval_shape` it to recover the saved `decomposition` item's tree structure
    without building (or knowing about) the optimizers/adversaries.

    The CI fn initializes from the fresh components, directly into the storage its
    abstract value declares (`init_placed`'s no-host-tree contract). Callers run it under
    the model's mesh (`jax.set_mesh`, entered outside any trace): a seeded initializer
    reads no components, leaving only the uncommitted key as a traced input, so the CI
    fn's reshard takes its mesh from that context."""
    rules, mesh = _placed_init_geometry(model)
    assert jax.sharding.get_abstract_mesh() == mesh.abstract_mesh, (
        "init_decomposition runs under the model's mesh",
        jax.sharding.get_abstract_mesh(),
    )
    ci_key = random.fold_in(init_key, 1)
    components = init_model_component_stacks_placed(model, init_key, rules, component_initializer)

    @eqx.filter_jit
    def init_placed_ci_fn(
        initializer: CIFnInitializer[Conditioning], components: ComponentStacks, key: PRNGKeyArray
    ) -> CIFn[Conditioning]:
        ci_fn = initializer(components, key)
        return jax.reshard(ci_fn, ci_fn.shardings(mesh))

    ci_fn = init_placed_ci_fn(ci_fn_initializer, components, ci_key)
    assert ci_fn.has_position_axis == model.model.has_position_axis, (
        f"CI fn has_position_axis={ci_fn.has_position_axis} but model declares "
        f"{model.model.has_position_axis}"
    )
    return Decomposition(components=components, ci_fn=ci_fn)


def _init_adversaries[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    positions: PositionAxis,
    batch_size: int,
    recon: tuple[AnyReconLossTerm, ...],
    src_key: PRNGKeyArray,
) -> dict[str, PersistentAdversary]:
    """Initialize the persistent sources required by the reconstruction terms."""
    _, mesh = _placed_init_geometry(model)
    assert isinstance(positions, Positioned) == model.model.has_position_axis, (
        f"{positions} does not match the model's has_position_axis={model.model.has_position_axis}"
    )
    adversaries: dict[str, PersistentAdversary] = {}
    for term_idx, (state_key, cfg) in enumerate(persistent_configs(recon).items()):
        sources = init_persistent_sources_from_config(
            model.model.sites, positions, cfg, batch_size, random.fold_in(src_key, term_idx), mesh
        )
        adversaries[state_key] = PersistentAdversary(
            sources=sources,
            opt_state=init_sources_opt_state(cfg.optimizer, sources),
            state_key=state_key,
            optimizer=cfg.optimizer,
            n_warmup=cfg.n_warmup_steps,
        )
    return adversaries


def init_pd_state[TargetIn, Out, PreparedT: ComponentActivations, Conditioning, PreparedMaskingT](
    pd: PDConfig,
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    ci_fn_initializer: CIFnInitializer[Conditioning],
    positions: PositionAxis,
    opt_vu: ScheduledOptimizer,
    opt_ci: ScheduledOptimizer,
    init_key: PRNGKeyArray,
    src_key: PRNGKeyArray,
    component_initializer: ComponentInitializer[
        TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT
    ] = random_component_initializer,
) -> PDState[Conditioning]:
    decomposition = init_decomposition(model, ci_fn_initializer, init_key, component_initializer)
    return TrainState(
        decomposition=decomposition,
        training=init_pd_training(pd, model, positions, opt_vu, opt_ci, decomposition, src_key),
    )


def init_pd_training[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    pd: PDConfig,
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    positions: PositionAxis,
    opt_vu: ScheduledOptimizer,
    opt_ci: ScheduledOptimizer,
    decomposition: Decomposition[Conditioning],
    src_key: PRNGKeyArray,
) -> PDTrainingState:
    rules, _ = _placed_init_geometry(model)
    objective = build_objective(pd.loss_metrics, model.model.sites)
    [importance] = [
        cfg for cfg in pd.loss_metrics if isinstance(cfg, ImportanceMinimalityLossConfig)
    ]
    return PDTrainingState(
        objective=objective,
        frequency=init_frequency_estimator_placed(
            importance.frequency, model.model.sites, rules.frequency_sharding
        ),
        components_opt_state=opt_vu.init(eqx.filter(decomposition.components, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(decomposition.ci_fn, eqx.is_array)),
        adversaries=_init_adversaries(model, positions, pd.batch_size, objective.recon, src_key),
        step=jnp.zeros((), jnp.int32),
    )


def init_targeted_pd_state[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    pd: TargetedPDConfig,
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    ci_fn_initializer: CIFnInitializer[Conditioning],
    positions: PositionAxis,
    opt_vu: ScheduledOptimizer,
    opt_ci: ScheduledOptimizer,
    init_key: PRNGKeyArray,
    src_key: PRNGKeyArray,
    nontarget: NontargetConfig,
    component_initializer: ComponentInitializer[
        TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT
    ] = random_component_initializer,
) -> TargetedPDState[Conditioning]:
    decomposition = init_decomposition(model, ci_fn_initializer, init_key, component_initializer)
    return TrainState(
        decomposition=decomposition,
        training=init_targeted_pd_training(
            pd, model, positions, opt_vu, opt_ci, decomposition, src_key, nontarget
        ),
    )


def init_targeted_pd_training[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    pd: TargetedPDConfig,
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    positions: PositionAxis,
    opt_vu: ScheduledOptimizer,
    opt_ci: ScheduledOptimizer,
    decomposition: Decomposition[Conditioning],
    src_key: PRNGKeyArray,
    nontarget: NontargetConfig,
) -> TargetedPDTrainingState:
    rules, _ = _placed_init_geometry(model)
    objective = build_targeted_objective(pd.loss_metrics, nontarget, model.model.sites)
    [importance] = [
        cfg for cfg in pd.loss_metrics if isinstance(cfg, ImportanceMinimalityLossConfig)
    ]
    return TargetedPDTrainingState(
        objective=objective,
        target_frequency=init_frequency_estimator_placed(
            importance.frequency, model.model.sites, rules.frequency_sharding
        ),
        nontarget_frequency=init_frequency_estimator_placed(
            importance.frequency, model.model.sites, rules.frequency_sharding
        ),
        ci_scaled_weight_decay=(
            CIScaledWeightDecay(jnp.asarray(pd.ci_scaled_weight_decay, jnp.float32))
            if pd.ci_scaled_weight_decay is not None
            else None
        ),
        components_opt_state=opt_vu.init(eqx.filter(decomposition.components, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(decomposition.ci_fn, eqx.is_array)),
        adversaries=_init_adversaries(
            model, positions, pd.batch_size, objective.target.recon, src_key
        ),
        step=jnp.zeros((), jnp.int32),
    )
