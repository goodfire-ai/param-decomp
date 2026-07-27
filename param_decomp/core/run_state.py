"""Construction of a run's optimizers + initial `TrainState` from the pydantic `PDConfig`
plus the lab-built CI-fn arch and the target's position extents.

Shared by the trainer (`run.py`) and the run-loading consumers (`load_run.py`): orbax
restores ONTO a reference pytree, so anything that wants to read a checkpoint must
rebuild the state exactly as the run did — same init fns, same key derivation, same
optimizer-state structure.
"""

from collections.abc import Callable

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
from param_decomp.core.configs import (
    AdamPGDConfig,
    AdamWOptimizerConfig,
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
from param_decomp.core.placement import PlacementRules
from param_decomp.core.recon import (
    MixedPersistentStochasticSources,
    PersistentSources,
    build_loss_terms,
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


def build_optimizers(pd: PDConfig, ci_fn_arch: CIFnArch, mesh: Mesh | None):
    """Returns (opt_vu, opt_ci, schedules): the schedule fns are returned too so the
    log path reports the exact LR the optimizer applies (single source of truth).

    The canonical-shape asserts (cosine-to-0.1, canonical optimizer shape, required components
    clip, optional CI-fn clip) live in
    the lab conversion (`experiments.config.assert_canonical_algorithm_config`); here we
    read the values straight off `PDConfig` so there is no second source of truth."""
    sched_vu = optax_schedule(pd.components_optimizer.lr_schedule, pd.steps)
    sched_ci = optax_schedule(pd.ci_fn_optimizer.lr_schedule, pd.steps)
    opt_vu = _optimizer_with_clip(
        pd.components_optimizer, sched_vu, stacked_muon_dimension_numbers, mesh=mesh
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
    losses = build_loss_terms(pd.loss_metrics, model.site_names)
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
