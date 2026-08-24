"""Construction of a run's optimizers + initial `TrainState` from the pydantic `PDConfig`
plus the lab-built CI-fn arch and the target's position extents.

Shared by the trainer (`run.py`) and the run-loading consumers (`load_run.py`): orbax
restores ONTO a reference pytree, so anything that wants to read a checkpoint must
rebuild the state exactly as the run did — same init fns, same key derivation, same
optimizer-state structure.
"""

from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import ArrayLike
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.adversary import PersistentAdversary, init_sources_adam_state
from param_decomp.core.ci_fn import (
    ChunkwiseTransformerCIArch,
    ChunkwiseTransformerCIFn,
    CIFnArch,
    ns_compute_shardings,
)
from param_decomp.core.configs import (
    AdamPGDConfig,
    AdamWOptimizerConfig,
    AnyImportanceMinimalityLossConfig,
    AnyPDConfig,
    ImportanceMinimalityLossConfig,
    MuonOptimizerConfig,
    PDConfigBase,
    SmoothL0ImportanceMinimalityLossConfig,
)
from param_decomp.core.init_placed import (
    init_ci_fn_placed,
    init_component_stacks_placed,
    init_sources_sharded,
)
from param_decomp.core.losses import EmaFrequency, resolve_frequency, scheduled_value_traced
from param_decomp.core.model import PlacedModel, PositionAxis, Positioned
from param_decomp.core.muon_stacked import NSWaypoints, stacked_muon
from param_decomp.core.objective import build_recon_terms
from param_decomp.core.placement import (
    CIFnPlacement,
    PlacementRules,
    assert_stacked_muon_ci_staging,
    assert_stacked_muon_component_staging,
    ns_staging_sharding,
)
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


def stacked_muon_dimension_numbers(params: optax.Params) -> optax.Params:
    """Muon leaf labeling for matrix-STACK trees: every 3D leaf is a `[stack, a, b]` stack
    of matrices — orthogonalize the trailing two axes, stack axis batched — and everything
    else (e.g. the CI fn's `[n_chunks, d]` bias stacks) takes the Adam fallback. Covers
    BOTH optimizer groups now: the chunkwise CI fn's per-chunk stacks (`ci_fn.py`) and the
    owner-partitioned V/U semantic-group stacks (`components.py` — all leaves 3D, so the
    fallback never fires there). optax's default rule (2D → muon) would Adam every V/U
    leaf silently and, on the CI tree, NS-orthogonalize the bias stacks instead."""
    dims = optax.contrib.MuonDimensionNumbers(reduction_axis=-2, output_axis=-1)
    return jax.tree.map(lambda leaf: dims if leaf.ndim == 3 else None, params)


def _optimizer_with_clip(
    opt: AdamWOptimizerConfig | MuonOptimizerConfig,
    schedule: Callable[[ArrayLike], Array],
    muon_dimension_numbers: Callable[[optax.Params], optax.Params] | None,
    waypoints: NSWaypoints | None,
):
    """The group optimizer (fp32 master) over `schedule`, optionally preceded by
    torch-parity global-norm clip (SPEC S19/N1). AdamW is canonical (eps is the torch/optax
    default 1e-8, not exposed on `AdamWOptimizerConfig`; optax's wd default overridden to the
    config's — torch's is 0); Muon is a config-gated experimental variant (SPEC S19').
    `muon_dimension_numbers` labels the group's leaves for muon (None = optax's default
    2D-matrix rule, correct for the MLP CI fns); it and `waypoints` (the group's declared
    stacked-NS staging) are read only by the stacked-muon arm."""
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
                waypoints=waypoints,
            )
    if opt.grad_clip_norm is None:
        return inner
    return optax.chain(clip_by_global_norm_with_eps(opt.grad_clip_norm, eps=1e-6), inner)


def _uniform_waypoints(sharding: NamedSharding) -> NSWaypoints:
    """Every muon leaf stages at the same `ns_compute` waypoint (the V/U components tree
    shares one row; an MLP CI fn has no placement rows and stages replicated)."""
    return lambda tree: jax.tree.map(lambda _: sharding, tree)


def build_optimizers(
    pd: PDConfigBase,
    ci_fn_arch: CIFnArch,
    mesh: Mesh,
    placement: PlacementRules,
    ci_placement: CIFnPlacement | None,
):
    """Returns (opt_vu, opt_ci, schedules): the schedule fns are returned too so the
    log path reports the exact LR the optimizer applies (single source of truth).

    Every knob is read straight off `PDConfig` and honored as written — the full
    `ScheduleConfig` shape, both optimizer types, and a per-group clip that is simply
    absent when `grad_clip_norm` is null. Each group's stacked-NS staging comes from its
    `ns_compute` placement rows: one row for the V/U stacks, one per CI weight family
    (`ci_fn.ns_compute_shardings`). `ci_placement` is the run's RESOLVED CI-fn placement
    (`resolve_ci_placement`) — never re-derived from `placement` here."""
    sched_vu = optax_schedule(pd.components_optimizer.lr_schedule, pd.steps)
    sched_ci = optax_schedule(pd.ci_fn_optimizer.lr_schedule, pd.steps)
    match pd.components_optimizer:
        case MuonOptimizerConfig(impl="stacked"):
            assert_stacked_muon_component_staging(placement)
        case _:
            pass
    opt_vu = _optimizer_with_clip(
        pd.components_optimizer,
        sched_vu,
        stacked_muon_dimension_numbers,
        waypoints=_uniform_waypoints(ns_staging_sharding(placement.components.ns_compute, mesh)),
    )
    ci_muon_dim_nums: Callable[[optax.Params], optax.Params] | None
    ci_waypoints: NSWaypoints
    match ci_fn_arch:
        case ChunkwiseTransformerCIArch():
            assert ci_placement is not None, "a placed run's chunkwise CI fn carries its rows"
            match pd.ci_fn_optimizer:
                case MuonOptimizerConfig(impl="stacked"):
                    assert_stacked_muon_ci_staging(placement, len(ci_fn_arch.chunks))
                case _:
                    pass
            ci_stage_rows = ci_placement
            ci_muon_dim_nums = stacked_muon_dimension_numbers
            # The muon-masked update tree keeps the chunkwise treedef (its class included),
            # so the structural navigation in `ns_compute_shardings` applies to it directly.
            ci_waypoints = lambda tree: ns_compute_shardings(
                cast(ChunkwiseTransformerCIFn, cast(object, tree)), mesh, ci_stage_rows
            )
        case _:
            assert ci_placement is None, f"{type(ci_fn_arch).__name__} runs unplaced"
            ci_muon_dim_nums = None
            ci_waypoints = _uniform_waypoints(NamedSharding(mesh, P(None, None, None)))
    opt_ci = _optimizer_with_clip(
        pd.ci_fn_optimizer, sched_ci, ci_muon_dim_nums, waypoints=ci_waypoints
    )
    return opt_vu, opt_ci, (sched_vu, sched_ci)


def _placed_init_geometry(model: PlacedModel) -> tuple[PlacementRules, Mesh]:
    """The bundle's own rules + mesh. Seeded init places real arrays, so an unplaced
    bundle and the abstract (spec-check) arm of `PlacementRules.mesh` are both refused."""
    rules = model.placement
    assert rules is not None, "seeded init is placed init: the bundle must carry rules"
    mesh = rules.mesh
    assert isinstance(mesh, Mesh), type(mesh)
    return rules, mesh


def init_decomposition(
    model: PlacedModel,
    ci_fn_arch: CIFnArch,
    init_key: PRNGKeyArray,
) -> Decomposition:
    """The trained-product half of `init_train_state`, factored out so a consumer can
    `jax.eval_shape` it to recover the saved `decomposition` item's tree structure
    without building (or knowing about) the optimizers/adversaries."""
    rules, mesh = _placed_init_geometry(model)
    ci_key = random.fold_in(init_key, 1)
    components = init_component_stacks_placed(model.sites, init_key, rules)
    ci_fn = init_ci_fn_placed(ci_fn_arch, model.sites, ci_key, mesh, rules)
    assert ci_fn.has_position_axis == model.has_position_axis, (
        f"CI fn has_position_axis={ci_fn.has_position_axis} but model declares "
        f"{model.has_position_axis}"
    )
    return Decomposition(components=components, ci_fn=ci_fn)


def _imp_min_config(pd: AnyPDConfig) -> AnyImportanceMinimalityLossConfig:
    [imp_cfg] = [
        m
        for m in pd.loss_metrics
        if isinstance(m, ImportanceMinimalityLossConfig | SmoothL0ImportanceMinimalityLossConfig)
    ]
    return imp_cfg


def init_train_state(
    pd: AnyPDConfig,
    model: PlacedModel,
    ci_fn_arch: CIFnArch,
    positions: PositionAxis,
    opt_vu: optax.GradientTransformation,
    opt_ci: optax.GradientTransformation,
    init_key: PRNGKeyArray,
    src_key: PRNGKeyArray,
) -> TrainState:
    """Persistent sources are shaped from `positions` (the run's waist geometry)."""
    _, mesh = _placed_init_geometry(model)
    assert isinstance(positions, Positioned) == model.has_position_axis, (
        f"{positions} does not match the model's has_position_axis={model.has_position_axis}"
    )
    decomposition = init_decomposition(model, ci_fn_arch, init_key)
    components, ci_fn = decomposition.components, decomposition.ci_fn
    freq_role = resolve_frequency(_imp_min_config(pd).frequency)
    # Recon terms only — persistent adversaries derive from these, and a targeted run's
    # loss list carries no faithfulness role for a full objective build to demand.
    recon_terms = build_recon_terms(pd.loss_metrics, model.site_names)
    persistent = persistent_configs(recon_terms)
    term_coeff_by_state_key = {
        term.sources.state_key: term.coeff
        for term in recon_terms
        if isinstance(term.sources, (PersistentSources, MixedPersistentStochasticSources))
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
                adam=cfg.optimizer,
                n_warmup=cfg.n_warmup_steps,
            )
    return TrainState(
        decomposition=decomposition,
        training=TrainingItem(
            components_opt_state=opt_vu.init(eqx.filter(components, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries=adversaries,
            freq_ema=freq_role.initial_state(model.sites)
            if isinstance(freq_role, EmaFrequency)
            else None,
            step=jnp.zeros((), jnp.int32),
        ),
    )
