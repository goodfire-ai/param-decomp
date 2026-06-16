"""Construction of a run's optimizers + initial `TrainState` from an `ExperimentConfig`.

Shared by the trainer (`run.py`) and the checkpoint exporter (`export.py`): orbax
restores ONTO a reference pytree, so anything that wants to read a checkpoint must
rebuild the state exactly as the run did — same init fns, same key derivation, same
optimizer-state structure.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random
from jax.sharding import Mesh
from jaxtyping import Array, PRNGKeyArray

from jax_single_pool.adversary import init_sources_adam_state
from jax_single_pool.config import ExperimentConfig
from jax_single_pool.llama8b_sharding import (
    init_ci_fn_sharded,
    init_decomp_vu_sharded,
    init_sources_sharded,
)
from jax_single_pool.lm import DecomposedLM
from jax_single_pool.recon import build_recon_terms
from jax_single_pool.train import TrainState


def torch_cosine_schedule(peak_lr: float, total_steps: int, alpha: float) -> optax.Schedule:
    """Cosine decay matching torch's `get_scheduled_value` denominator: `progress =
    step / (total_steps - 1)` (no warmup), so `0.1×` is reached at `step = total_steps - 1`.
    optax's `cosine_decay_schedule` divides by `total_steps`, reaching it one step later
    (SPEC S20). `total_steps == 1` collapses to a constant `peak_lr`."""
    denom = max(total_steps - 1, 1)

    def schedule(count: Array) -> Array:
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
        updates: optax.Updates, state: optax.EmptyState, params: optax.Params | None = None
    ) -> tuple[optax.Updates, optax.EmptyState]:
        del params
        global_norm = optax.global_norm(updates)
        scale = jnp.minimum(max_norm / (global_norm + eps), 1.0)
        updates = jax.tree.map(lambda g: g * scale, updates)
        return updates, state

    return optax.GradientTransformation(init, update)


def build_optimizers(cfg: ExperimentConfig):
    """Returns (opt_vu, opt_ci, schedules): the schedule fns are returned too so the
    log path reports the exact LR the optimizer applies (single source of truth)."""
    sched_vu = torch_cosine_schedule(cfg.vu_optimizer.lr, cfg.steps, alpha=0.1)
    sched_ci = torch_cosine_schedule(cfg.ci_optimizer.lr, cfg.steps, alpha=0.1)
    opt_vu = optax.chain(
        clip_by_global_norm_with_eps(cfg.vu_optimizer.grad_clip_norm, eps=1e-6),
        optax.adamw(sched_vu, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0),
    )
    opt_ci = optax.adamw(sched_ci, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0)
    return opt_vu, opt_ci, (sched_vu, sched_ci)


def init_train_state(
    cfg: ExperimentConfig,
    lm: DecomposedLM,
    opt_vu: optax.GradientTransformation,
    opt_ci: optax.GradientTransformation,
    init_key: PRNGKeyArray,
    src_key: PRNGKeyArray,
    mesh: Mesh,
) -> TrainState:
    components = init_decomp_vu_sharded(lm.sites, init_key, mesh)
    ci_fn = init_ci_fn_sharded(cfg.ci_fn, lm.sites, random.fold_in(init_key, 1), mesh)
    loss_spec = build_recon_terms(cfg.loss_metrics, lm.site_names, cfg.n_mask_samples, cfg.sampling)
    sources = {
        state_key: init_sources_sharded(
            lm.site_names,
            tuple(s.C for s in lm.sites),
            cfg.data.seq_len,
            loss_spec.persistent[state_key].scope,
            cfg.data.global_batch,
            random.fold_in(src_key, term_idx),
            mesh,
        )
        for term_idx, state_key in enumerate(loss_spec.persistent)
    }
    return TrainState(
        components=components,
        ci_fn=ci_fn,
        components_opt_state=opt_vu.init(eqx.filter(components, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        sources=sources,
        sources_opt_state={k: init_sources_adam_state(v) for k, v in sources.items()},
        step=jnp.zeros((), jnp.int32),
    )
