"""Construction of a run's optimizers + initial `TrainState` from the pydantic `PDConfig`
plus the lab-built CI-fn arch and the target's position extents.

Shared by the trainer (`run.py`) and the run-loading consumers (`load_run.py`): orbax
restores ONTO a reference pytree, so anything that wants to read a checkpoint must
rebuild the state exactly as the run did — same init fns, same key derivation, same
optimizer-state structure.
"""

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random
from jax.sharding import Mesh
from jax.typing import ArrayLike
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.adversary import PersistentAdversary, init_sources_adam_state
from param_decomp.core.ci_fn import ChunkwiseTransformerCIArch, CIFnArch
from param_decomp.core.components import ComponentStacks
from param_decomp.core.configs import (
    AdamPGDConfig,
    AdamWOptimizerConfig,
    ComponentUpdateScaling,
    MuonOptimizerConfig,
    PDConfig,
)
from param_decomp.core.init_placed import (
    init_ci_fn_placed,
    init_component_stacks_placed,
    init_sources_sharded,
)
from param_decomp.core.losses import scheduled_value_traced
from param_decomp.core.model import DecomposedModel, PositionAxis, Positioned
from param_decomp.core.muon_stacked import stacked_muon
from param_decomp.core.objective import build_objective
from param_decomp.core.placement import PlacementRules
from param_decomp.core.recon import (
    MixedPersistentStochasticSources,
    PersistentSources,
    persistent_configs,
)
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.train import Decomposition, TrainingItem, TrainState


def optax_schedule(config: ScheduleConfig, total_steps: int) -> Callable[[ArrayLike], Array]:
    """`scheduled_value_traced` curried into an optax schedule over the update count.
    Torch cosine parity (the `decay_steps - 1` denominator, SPEC S20) is pinned by
    `test_optim_torch_parity.py`."""

    def schedule(count: ArrayLike) -> Array:
        return scheduled_value_traced(jnp.asarray(count, jnp.float32), total_steps, config)

    return schedule


def clip_by_global_norm_with_eps(max_norm: float, eps: float) -> optax.GradientTransformation:
    """Global-norm clip matching torch's `clip_grad_norm_`: scale by
    `clip(max_norm / (global_norm + eps), max=1)`. optax's `clip_by_global_norm` omits
    `eps`; at small `max_norm` (0.01) the clip fires almost every step so this ~1e-4
    relative offset is per-step (SPEC S19)."""

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


def scale_component_updates_c_covariant() -> optax.GradientTransformation:
    """Scale post-Adam V/U updates so the represented-matrix step transfers across C.

    With the canonical init ``V~N(0, d_in^-1)``, ``U~N(0, C^-1)``, Adam's first
    per-parameter step has approximately fixed magnitude. In function space, ``V dU``
    therefore grows as C and ``dV U`` as sqrt(C). Multiplying the already-preconditioned
    updates by ``d_in/C`` and ``sqrt(d_in/C)`` respectively makes both terms C-invariant,
    anchored to the configured LR at C=d_in. The scaling must follow Adam: applying it to
    raw gradients would be cancelled by Adam's second-moment normalization.
    """

    def init(params: optax.Params) -> optax.EmptyState:
        assert isinstance(params, ComponentStacks)
        return optax.EmptyState()

    def update(
        updates: optax.Updates, state: optax.OptState, params: optax.Params | None = None
    ) -> tuple[optax.Updates, optax.OptState]:
        del params
        assert isinstance(updates, ComponentStacks)
        stacks = {
            shape: (
                dV * (shape[0] / shape[2]) ** 0.5,
                dU * (shape[0] / shape[2]),
            )
            for shape, (dV, dU) in updates.stacks.items()
        }
        return ComponentStacks(stacks=stacks, site_slots=updates.site_slots), state

    return optax.GradientTransformation(init, update)


class GaugeBalancedState(NamedTuple):
    inner_state: optax.OptState


def balance_component_stacks(
    components: ComponentStacks,
) -> tuple[ComponentStacks, dict[tuple[int, int, int], Array]]:
    """Choose the minimum-factor-norm representative of every non-null rank-one term.

    ``V_c U_c`` has the exact gauge ``(a V_c)(U_c / a)``. The unique positive scale that
    makes both factor norms equal is ``a=sqrt(||U_c||/||V_c||)``. A component with either
    factor exactly zero is already functionally null and has no finite balanced
    representative, so it stays untouched.
    """
    scales: dict[tuple[int, int, int], Array] = {}
    stacks: dict[tuple[int, int, int], tuple[Array, Array]] = {}
    for shape, (V, U) in components.stacks.items():
        v_norm = jnp.linalg.norm(V, axis=1)
        u_norm = jnp.linalg.norm(U, axis=2)
        non_null = (v_norm != 0) & (u_norm != 0)
        ratio = jnp.where(non_null, u_norm, 1.0) / jnp.where(non_null, v_norm, 1.0)
        scale = jnp.sqrt(ratio)
        scales[shape] = scale
        stacks[shape] = (V * scale[:, None, :], U / scale[:, :, None])
    return ComponentStacks(stacks=stacks, site_slots=components.site_slots), scales


def _transform_adam_moments_to_balanced_gauge(
    state: optax.OptState, scales: dict[tuple[int, int, int], Array]
) -> optax.OptState:
    """Transform Adam covector moments under ``V'=aV, U'=U/a``.

    Gradients transform inversely to parameters, so ``m_V'=m_V/a``,
    ``nu_V'=nu_V/a²``, ``m_U'=a m_U``, and ``nu_U'=a² nu_U``.
    """

    def transform(x: object) -> object:
        if not isinstance(x, optax.ScaleByAdamState):
            return x
        assert isinstance(x.mu, ComponentStacks) and isinstance(x.nu, ComponentStacks)
        mu_stacks = {
            shape: (mV / scales[shape][:, None, :], mU * scales[shape][:, :, None])
            for shape, (mV, mU) in x.mu.stacks.items()
        }
        nu_stacks = {
            shape: (
                nV / scales[shape][:, None, :] ** 2,
                nU * scales[shape][:, :, None] ** 2,
            )
            for shape, (nV, nU) in x.nu.stacks.items()
        }
        return optax.ScaleByAdamState(
            count=x.count,
            mu=ComponentStacks(stacks=mu_stacks, site_slots=x.mu.site_slots),
            nu=ComponentStacks(stacks=nu_stacks, site_slots=x.nu.site_slots),
        )

    return jax.tree.map(transform, state, is_leaf=lambda x: isinstance(x, optax.ScaleByAdamState))


def gauge_balance_component_optimizer(
    inner: optax.GradientTransformation,
) -> optax.GradientTransformation:
    """Retract each Adam step to balanced V/U gauge and carry its moments with it."""

    def init(params: optax.Params) -> GaugeBalancedState:
        assert isinstance(params, ComponentStacks)
        return GaugeBalancedState(inner.init(params))

    def update(
        updates: optax.Updates,
        state: GaugeBalancedState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, GaugeBalancedState]:
        assert isinstance(updates, ComponentStacks)
        assert isinstance(params, ComponentStacks)
        inner_updates, inner_state = inner.update(updates, state.inner_state, params)
        assert isinstance(inner_updates, ComponentStacks)
        proposed = optax.apply_updates(params, inner_updates)
        assert isinstance(proposed, ComponentStacks)
        balanced, scales = balance_component_stacks(proposed)
        balanced_updates = jax.tree.map(lambda new, old: new - old, balanced, params)
        inner_state = _transform_adam_moments_to_balanced_gauge(inner_state, scales)
        return balanced_updates, GaugeBalancedState(inner_state)

    return optax.GradientTransformation(init, update)


def scale_component_updates_c_covariant_balanced() -> optax.GradientTransformation:
    """C-covariant Adam scaling for balanced factors.

    At canonical balanced initialization both factor entries scale as C^-1/4, so either
    first-order product update grows as C^3/4. Scaling both post-Adam factor updates by
    ``(d_in/C)^3/4`` removes that growth, anchored to the configured LR at C=d_in.
    """

    def init(params: optax.Params) -> optax.EmptyState:
        assert isinstance(params, ComponentStacks)
        return optax.EmptyState()

    def update(
        updates: optax.Updates, state: optax.OptState, params: optax.Params | None = None
    ) -> tuple[optax.Updates, optax.OptState]:
        del params
        assert isinstance(updates, ComponentStacks)
        stacks = {
            shape: (dV * (shape[0] / shape[2]) ** 0.75, dU * (shape[0] / shape[2]) ** 0.75)
            for shape, (dV, dU) in updates.stacks.items()
        }
        return ComponentStacks(stacks=stacks, site_slots=updates.site_slots), state

    return optax.GradientTransformation(init, update)


def balanced_adam_product_step_geometry(d_in: int, d_out: int, C: int) -> float:
    """First-Adam-step geometry of a balanced ``V @ U`` factorization.

    With canonical ``Var(W_ij)=1/d_in`` scaling, bias-corrected Adam behaves like a sign
    step and its residual-aligned product motion is proportional to ``C**0.75 * A``, where
    ``A`` is the sum of the ``dV @ U`` and ``V @ dU`` shape contributions below. Returning
    that geometry separately makes the reciprocal update transform explicit and testable.
    """
    return C**0.75 * (d_in**0.5 * d_out**-0.75 + d_out**0.25 * d_in**-0.5)


def scale_component_updates_function_covariant_balanced() -> optax.GradientTransformation:
    """Map a product-space Adam LR to balanced factor updates.

    The authored LR is ``eta_product``. Each site's post-Adam factor updates are multiplied
    by ``1 / balanced_adam_product_step_geometry(d_in, d_out, C)`` so the leading relative
    ``V @ U`` motion is invariant to C, a uniform width change, and transposing a rectangular
    matrix. This is the shape-complete counterpart of ``c_covariant_balanced``; both still
    require realized function-step telemetry beyond the first-step regime.
    """

    def init(params: optax.Params) -> optax.EmptyState:
        assert isinstance(params, ComponentStacks)
        return optax.EmptyState()

    def update(
        updates: optax.Updates, state: optax.OptState, params: optax.Params | None = None
    ) -> tuple[optax.Updates, optax.OptState]:
        del params
        assert isinstance(updates, ComponentStacks)
        stacks = {
            shape: (
                dV / balanced_adam_product_step_geometry(*shape),
                dU / balanced_adam_product_step_geometry(*shape),
            )
            for shape, (dV, dU) in updates.stacks.items()
        }
        return ComponentStacks(stacks=stacks, site_slots=updates.site_slots), state

    return optax.GradientTransformation(init, update)


def uses_balanced_component_gauge(update_scaling: ComponentUpdateScaling) -> bool:
    return update_scaling in {"c_covariant_balanced", "function_covariant_balanced"}


def stacked_muon_dimension_numbers(params: optax.Params) -> optax.Params:
    """Muon leaf labeling for matrix-STACK trees: every 3D leaf is a `[stack, a, b]` stack
    of matrices — orthogonalize the trailing two axes, stack axis batched — and everything
    else (e.g. the CI fn's `[n_chunks, d]` bias stacks) takes the Adam fallback. Covers
    BOTH optimizer groups now: the chunkwise CI fn's per-chunk stacks (`ci_fn.py`) and the
    owner-partitioned V/U shape-group stacks (`components.py` — all leaves 3D, so the
    fallback never fires there). optax's default rule (2D → muon) would Adam every V/U
    leaf silently and, on the CI tree, NS-orthogonalize the bias stacks instead."""
    dims = optax.contrib.MuonDimensionNumbers(reduction_axis=-2, output_axis=-1)
    return jax.tree.map(lambda leaf: dims if leaf.ndim == 3 else None, params)


def _optimizer_with_clip(
    opt: AdamWOptimizerConfig | MuonOptimizerConfig,
    schedule: Callable[[ArrayLike], Array],
    muon_dimension_numbers: Callable[[optax.Params], optax.Params] | None,
    mesh: Mesh | None,
):
    """The group optimizer (fp32 master) over `schedule`, optionally preceded by
    torch-parity global-norm clip (SPEC S19/N1). AdamW is canonical (eps is the torch/optax
    default 1e-8, not exposed on `AdamWOptimizerConfig`; optax's wd default overridden to the
    config's — torch's is 0); Muon is a config-gated experimental variant (SPEC S19').
    `muon_dimension_numbers` labels the group's leaves for muon (None = optax's default
    2D-matrix rule, correct for the MLP CI fns); ignored for adamw.
    `mesh` shards the stacked-impl NS batch axis; None (toys, CPU tests) = unsharded."""
    match opt:
        case AdamWOptimizerConfig():
            inner = optax.adamw(
                schedule, b1=opt.betas[0], b2=opt.betas[1], eps=1e-8, weight_decay=opt.weight_decay
            )
        case MuonOptimizerConfig(impl="optax"):
            assert opt.ns_dtype == "float32", "ns_dtype is a stacked-impl knob (optax NS is fp32)"
            inner = optax.contrib.muon(
                schedule,
                beta=opt.beta,
                weight_decay=opt.weight_decay,
                consistent_rms=opt.consistent_rms,
                muon_weight_dimension_numbers=muon_dimension_numbers,
                ns_steps=opt.ns_steps,
            )
        case MuonOptimizerConfig():
            assert opt.impl == "stacked", opt.impl
            inner = stacked_muon(
                schedule,
                beta=opt.beta,
                weight_decay=opt.weight_decay,
                consistent_rms=opt.consistent_rms,
                muon_weight_dimension_numbers=muon_dimension_numbers,
                ns_steps=opt.ns_steps,
                ns_dtype=jnp.dtype(opt.ns_dtype),
                mesh=mesh,
            )
    if opt.grad_clip_norm is None:
        return inner
    return optax.chain(clip_by_global_norm_with_eps(opt.grad_clip_norm, eps=1e-6), inner)


def configure_component_optimizer(
    optimizer: optax.GradientTransformation,
    update_scaling: ComponentUpdateScaling,
    config: AdamWOptimizerConfig | MuonOptimizerConfig,
) -> optax.GradientTransformation:
    """Apply the V/U post-optimizer rule to any component optimizer phase.

    Main training and faithfulness warmup use separate Adam instances; routing both through
    this boundary prevents warmup from silently reintroducing the C/gauge dependence that
    the configured main update removes.
    """
    match update_scaling:
        case "none":
            return optimizer
        case "c_covariant":
            assert isinstance(config, AdamWOptimizerConfig), (
                "c_covariant component scaling is derived for AdamW updates"
            )
            return optax.chain(optimizer, scale_component_updates_c_covariant())
        case "c_covariant_balanced":
            assert isinstance(config, AdamWOptimizerConfig), (
                "c_covariant_balanced component scaling is derived for AdamW updates"
            )
            scaled = optax.chain(optimizer, scale_component_updates_c_covariant_balanced())
            return gauge_balance_component_optimizer(scaled)
        case "function_covariant_balanced":
            assert isinstance(config, AdamWOptimizerConfig), (
                "function_covariant_balanced component scaling is derived for AdamW updates"
            )
            scaled = optax.chain(optimizer, scale_component_updates_function_covariant_balanced())
            return gauge_balance_component_optimizer(scaled)
        case _:
            raise AssertionError(update_scaling)


def build_optimizers(pd: PDConfig, ci_fn_arch: CIFnArch, mesh: Mesh | None):
    """Returns (opt_vu, opt_ci, schedules): the schedule fns are returned too so the
    log path reports the exact LR the optimizer applies (single source of truth).

    Every knob is read straight off `PDConfig` and honored as written — the full
    `ScheduleConfig` shape, both optimizer types, and a per-group clip that is simply
    absent when `grad_clip_norm` is null."""
    sched_vu = optax_schedule(pd.components_optimizer.lr_schedule, pd.steps)
    sched_ci = optax_schedule(pd.ci_fn_optimizer.lr_schedule, pd.steps)
    opt_vu = _optimizer_with_clip(
        pd.components_optimizer, sched_vu, stacked_muon_dimension_numbers, mesh=mesh
    )
    opt_vu = configure_component_optimizer(
        opt_vu, pd.component_update_scaling, pd.components_optimizer
    )
    ci_muon_dim_nums = (
        stacked_muon_dimension_numbers
        if isinstance(ci_fn_arch, ChunkwiseTransformerCIArch)
        else None
    )
    opt_ci = _optimizer_with_clip(pd.ci_fn_optimizer, sched_ci, ci_muon_dim_nums, mesh=mesh)
    return opt_vu, opt_ci, (sched_vu, sched_ci)


def init_decomposition(
    model: DecomposedModel,
    ci_fn_arch: CIFnArch,
    init_key: PRNGKeyArray,
    mesh: Mesh,
    rules: PlacementRules,
) -> Decomposition:
    """The trained-product half of `init_train_state`, factored out so a consumer can
    `jax.eval_shape` it to recover the saved `decomposition` item's tree structure
    without building (or knowing about) the optimizers/adversaries."""
    ci_key = random.fold_in(init_key, 1)
    # V/U placement derives from the rules table; the CI fn still declares its own
    # per-leaf shardings (PLACEMENT_DESIGN.md migration stage 3).
    components = init_component_stacks_placed(model.sites, init_key, rules)
    ci_fn = init_ci_fn_placed(ci_fn_arch, model.sites, ci_key, mesh)
    assert ci_fn.has_position_axis == model.has_position_axis, (
        f"CI fn has_position_axis={ci_fn.has_position_axis} but model declares "
        f"{model.has_position_axis}"
    )
    return Decomposition(components=components, ci_fn=ci_fn)


def init_train_state(
    pd: PDConfig,
    model: DecomposedModel,
    ci_fn_arch: CIFnArch,
    positions: PositionAxis,
    opt_vu: optax.GradientTransformation,
    opt_ci: optax.GradientTransformation,
    init_key: PRNGKeyArray,
    src_key: PRNGKeyArray,
    mesh: Mesh,
    rules: PlacementRules,
) -> TrainState:
    """Persistent sources are shaped from `positions` (the run's waist geometry)."""
    assert isinstance(positions, Positioned) == model.has_position_axis, (
        f"{positions} does not match the model's has_position_axis={model.has_position_axis}"
    )
    decomposition = init_decomposition(model, ci_fn_arch, init_key, mesh, rules)
    components, ci_fn = decomposition.components, decomposition.ci_fn
    if uses_balanced_component_gauge(pd.component_update_scaling):
        components, _ = balance_component_stacks(components)
        decomposition = Decomposition(components=components, ci_fn=ci_fn)
    losses = build_objective(pd.loss_metrics, model.site_names)
    persistent = persistent_configs(losses.recon)
    term_coeff_by_state_key = {
        entry.sources.state_key: term.coeff
        for term in losses.recon
        for entry in term.plan
        if isinstance(entry.sources, (PersistentSources, MixedPersistentStochasticSources))
    }
    assert set(term_coeff_by_state_key) == set(persistent)
    adversaries: dict[str, PersistentAdversary] = {}
    if persistent:
        for term_idx, state_key in enumerate(persistent):
            cfg = persistent[state_key]
            assert isinstance(cfg.optimizer, AdamPGDConfig)
            sources = init_sources_sharded(
                model.site_names,
                tuple(s.C for s in model.sites),
                positions,
                cfg.source_shape,
                pd.batch_size,
                jnp.dtype(cfg.source_dtype),
                random.fold_in(src_key, term_idx),
                mesh,
            )
            adversaries[state_key] = PersistentAdversary(
                sources=sources,
                opt_state=init_sources_adam_state(sources),
                state_key=state_key,
                coeff=term_coeff_by_state_key[state_key],
                adam=cfg.optimizer,
                n_warmup=cfg.n_warmup_steps,
            )
    return TrainState(
        decomposition=decomposition,
        training=TrainingItem(
            components_opt_state=opt_vu.init(eqx.filter(components, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries=adversaries,
            step=jnp.zeros((), jnp.int32),
        ),
    )
