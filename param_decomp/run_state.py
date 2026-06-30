"""Construction of a run's optimizers + initial `TrainState` from the pydantic `PDConfig`
plus the lab-built CI-fn arch and data source.

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

from param_decomp.adversary import PersistentAdversary, init_sources_adam_state
from param_decomp.built_run import DataConfig
from param_decomp.ci_fn import CIFnArch
from param_decomp.configs import AdamPGDConfig, OptimizerConfig, PDConfig
from param_decomp.lm import DecomposedModel
from param_decomp.recon import (
    PersistentSources,
    build_loss_terms,
    persistent_configs,
)
from param_decomp.targets.llama8b_sharding import (
    init_ci_fn_placed,
    init_decomp_vu_placed,
    init_sources_sharded,
)
from param_decomp.train import TrainState


def torch_cosine_schedule(
    peak_lr: float, total_steps: int, alpha: float
) -> Callable[[ArrayLike], Array]:
    """Cosine decay matching torch's `get_scheduled_value` denominator: `progress =
    step / (total_steps - 1)` (no warmup), so `0.1×` is reached at `step = total_steps - 1`.
    optax's `cosine_decay_schedule` divides by `total_steps`, reaching it one step later
    (SPEC S20). `total_steps == 1` collapses to a constant `peak_lr`.

    Same annealing shape as `schedule.get_scheduled_value`'s cosine branch — that one is a
    host-numpy `float` lookup for the per-step PGD / p-anneal coeffs (warmup-aware, denom
    `decay_steps - 1`); this one is a `jnp` callable traced into the optimizer LR. The two
    can't share a runtime fn (host float vs traced array); each is pinned to torch by its
    own parity test (`test_schedule.py` / `test_optim_torch_parity.py`)."""
    denom = max(total_steps - 1, 1)

    def schedule(count: ArrayLike) -> Array:
        progress = jnp.asarray(count, jnp.float32) / denom
        return peak_lr * (alpha + (1 - alpha) * 0.5 * (1 + jnp.cos(jnp.pi * progress)))

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


def _adamw_with_clip(opt: OptimizerConfig, schedule: Callable[[ArrayLike], Array]):
    """AdamW (fp32 master, optax wd default overridden to the config's — torch's is 0)
    over `schedule`, optionally preceded by torch-parity global-norm clip (SPEC S19/N1).
    Adam eps is the torch/optax default 1e-8 (not exposed on `OptimizerConfig`)."""
    adamw = optax.adamw(
        schedule, b1=opt.betas[0], b2=opt.betas[1], eps=1e-8, weight_decay=opt.weight_decay
    )
    if opt.grad_clip_norm is None:
        return adamw
    return optax.chain(clip_by_global_norm_with_eps(opt.grad_clip_norm, eps=1e-6), adamw)


def build_optimizers(pd: PDConfig):
    """Returns (opt_vu, opt_ci, schedules): the schedule fns are returned too so the
    log path reports the exact LR the optimizer applies (single source of truth).

    The canonical-shape asserts (cosine-to-0.1, plain AdamW, components-only clip) live in
    the lab conversion (`experiments.config.assert_canonical_algorithm_config`); here we
    read the values straight off `PDConfig` so there is no second source of truth."""
    sched_vu = torch_cosine_schedule(
        pd.components_optimizer.lr_schedule.start_val, pd.steps, alpha=0.1
    )
    sched_ci = torch_cosine_schedule(pd.ci_fn_optimizer.lr_schedule.start_val, pd.steps, alpha=0.1)
    opt_vu = _adamw_with_clip(pd.components_optimizer, sched_vu)
    opt_ci = _adamw_with_clip(pd.ci_fn_optimizer, sched_ci)
    return opt_vu, opt_ci, (sched_vu, sched_ci)


def init_train_state(
    pd: PDConfig,
    lm: DecomposedModel,
    ci_fn_arch: CIFnArch,
    data: DataConfig | None,
    opt_vu: optax.GradientTransformation,
    opt_ci: optax.GradientTransformation,
    init_key: PRNGKeyArray,
    src_key: PRNGKeyArray,
    mesh: Mesh,
) -> TrainState:
    ci_key = random.fold_in(init_key, 1)
    # Placement is MODEL-OWNED: V/U + CI declare their own per-leaf shardings (asserting
    # divisibility), uniformly across mesh sizes — no scale inference, no replicate fallback.
    components = init_decomp_vu_placed(lm.sites, init_key, mesh)
    ci_fn = init_ci_fn_placed(ci_fn_arch, lm.sites, ci_key, mesh)
    assert ci_fn.expects_axes == lm.leading_axes, (
        f"CI fn expects leading axes {ci_fn.expects_axes} but model has {lm.leading_axes}"
    )
    losses = build_loss_terms(pd.loss_metrics, lm.site_names)
    persistent = persistent_configs(losses.recon)
    term_coeff_by_state_key = {
        entry.sources.state_key: term.coeff
        for term in losses.recon
        for entry in term.plan
        if isinstance(entry.sources, PersistentSources)
    }
    assert set(term_coeff_by_state_key) == set(persistent)
    adversaries: dict[str, PersistentAdversary] = {}
    if persistent:
        # Persistent sources live on a position axis; TMS (no position axis) carries none.
        assert isinstance(data, DataConfig), (
            "persistent PGD sources need a sequence axis; TMS (leading_axes=()) has none"
        )
        for term_idx, state_key in enumerate(persistent):
            cfg = persistent[state_key]
            assert isinstance(cfg.optimizer, AdamPGDConfig)
            sources = init_sources_sharded(
                lm.site_names,
                tuple(s.C for s in lm.sites),
                data.seq_len,
                cfg.scope,
                data.global_batch,
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
        components=components,
        ci_fn=ci_fn,
        components_opt_state=opt_vu.init(eqx.filter(components, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        adversaries=adversaries,
        step=jnp.zeros((), jnp.int32),
    )
