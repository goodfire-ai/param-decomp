"""The generic single-pool VPD training step over a `DecomposedModel` (SPEC §4).

One `jax.jit` step: clean target → CI envelope → per-persistent-term supplemental
ascents + per-fresh-entry sign-PGD ascents (`adversary.py`) → faith + imp-min +
the recon loss TERMS (`recon.py`; each term = plan × mask-source strategy, SPEC
S10') → one fused backward over (components, ci_fn, all persistent sources) →
optimizer updates → each persistent term's final ascent from the same graph,
unscaled by ITS coeff (SPEC S13'/S14'/S23). All trainable state is fp32 masters
(SPEC N1); forwards run in bf16 via explicit casts.

Schedules (imp-min p anneal, source-LR warmup) are computed inside the step from
`state.step`, so the jit signature is stable across the whole run (SPEC S9, S13).
Per-term RNG: term i draws from `fold_in(step_key, 1 + i)` in config-list order
(SPEC R1) — this reproduces the pre-unification production key derivation exactly.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from beartype import beartype
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float, PRNGKeyArray, jaxtyped

from param_decomp.adversary import (
    PersistentAdversary,
    init_fresh_pgd_sources,
    source_masks,
)
from param_decomp.ci_fn import CI, CIFn
from param_decomp.components import DecompVU
from param_decomp.configs import SmoothL0ImportanceMinimalityLossConfig
from param_decomp.lm import DecomposedModel
from param_decomp.losses import (
    annealed_imp_min_param,
    faithfulness_loss,
    imp_min_terms,
)
from param_decomp.recon import (
    ConstantSources,
    FreshPGDSources,
    LossSurface,
    PersistentSources,
    ReconForward,
    Routes,
    StochasticSources,
)
from param_decomp.sharding import batch_shard_leading

COMPUTE_DT = jnp.bfloat16


def cast_floating(tree: Any, dtype: Any) -> Any:
    return jax.tree.map(lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, tree)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainState:
    components: DecompVU  # the universal trainable V/U pytree, fp32 masters
    ci_fn: CIFn  # fp32 masters
    components_opt_state: optax.OptState
    ci_fn_opt_state: optax.OptState
    adversaries: dict[str, PersistentAdversary]
    """Persistent-PGD adversaries, `state_key -> adversary` (each owns its sources + Adam
    state + static config). One state_key per persistent loss term (SPEC S23); empty when
    no persistent term."""
    step: Array


def _grad_norm_metrics(components_grad: DecompVU, ci_fn_grad: Any) -> dict[str, Array]:
    """Pre-clip gradient L2 norms, matching the torch `component_grad_norms` families:
    per-leaf `grad_norms/components<path>` / `grad_norms/ci_fns<path>` (paths are this
    pytree's own — e.g. `.vu['layers.18.mlp.gate_proj'][0]` for the per-site Llama
    layout, vs torch's per-site names) and the overlay-critical
    `grad_norms/summary/{components,ci_fns,total}`."""
    out: dict[str, Array] = {}

    def family(grad_tree: Any, prefix: str) -> Array:
        sum_sq = jnp.zeros((), jnp.float32)
        for path, leaf in jax.tree_util.tree_flatten_with_path(grad_tree)[0]:
            leaf_sum_sq = jnp.sum(leaf.astype(jnp.float32) ** 2)
            out[f"grad_norms/{prefix}{jax.tree_util.keystr(path)}"] = jnp.sqrt(leaf_sum_sq)
            sum_sq = sum_sq + leaf_sum_sq
        out[f"grad_norms/summary/{prefix}"] = jnp.sqrt(sum_sq)
        return sum_sq

    total_sq = family(components_grad, "components") + family(ci_fn_grad, "ci_fns")
    out["grad_norms/summary/total"] = jnp.sqrt(total_sq)
    return out


# ───────────────────────────── the step factory ─────────────────────────────


def make_train_step(
    lm: DecomposedModel,
    *,
    losses: LossSurface,
    components_optimizer: optax.GradientTransformation,
    ci_fn_optimizer: optax.GradientTransformation,
    total_steps: int,
    remat_recon_forwards: bool,
    remat_ci_fn: bool,
    mesh: Mesh | None,
):
    """Build the `eqx.filter_jit`'d `step(model, state, batch, key) -> (state, metrics)`.

    `model` is the jit ARG (frozen 8B weights traced as array leaves, never baked); the
    factory closes over only static config (`site_names`, `recon_loss_fn`, term wiring) read
    off `lm` here. `losses` (from `build_loss_terms`) is the `LossSurface` record — the
    faithfulness + importance-minimality singletons and the recon Σ, read by name. `mesh`
    (when given) pins every batch-leading activation over the full mesh
    (`P(('replicate', 'fsdp'), ...)`) so the masked re-forwards stay on per-rank sub-batches
    (activation memory 1/N)."""
    site_names = lm.site_names
    sites = lm.sites
    recon_loss_fn = lm.recon_loss_fn  # static method: pure, holds no arrays — safe to close
    recon_terms = losses.recon
    faith_term = losses.faith
    imp_term = losses.imp
    faith_coeff = faith_term.coeff
    imp_min = imp_term.cfg
    imp_coeff = imp_term.coeff
    freq_coeff = imp_min.frequency.coeff if imp_min.frequency is not None else 0.0
    # Log the imp-min loss + its annealed param under penalty-kind-specific keys: the param
    # is `p` for L_p / `gamma` for smooth-L0, and the loss carries the penalty's class name.
    is_smooth_l0 = isinstance(imp_min, SmoothL0ImportanceMinimalityLossConfig)
    imp_loss_key = "imp_smooth_l0" if is_smooth_l0 else "imp"
    imp_min_param_key = "gamma_imp" if is_smooth_l0 else "p_imp"

    def batch_sharded(x: Array) -> Array:
        return batch_shard_leading(x, mesh)

    def ci_shard(x: Array) -> Array:
        """Pin a CI / mask tensor `[batch, *positions, C]` batch over the full mesh, C
        REPLICATED. No-op off-mesh (single device / toys)."""
        if mesh is None:
            return x
        spec = (("replicate", "fsdp"), *((None,) * (x.ndim - 1)))
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(*spec)))

    def ci_batch_sharded(ci: CI) -> CI:
        """Pin the CI-fn output batch over the full mesh, C REPLICATED — the layout `site_out`
        pins `x@V` to (SPEC §4.1), so the downstream mask multiply `xV * mask` needs no
        reshard. The explicit constraint stops GSPMD re-deciding it in the backward (same
        rationale as `site_out`'s activation pin, bf072ef01). `logits` is passed through
        (unused in the step — only the squashings are; DCE drops it)."""
        return CI(
            logits=ci.logits,
            lower={site: ci_shard(v) for site, v in ci.lower.items()},
            upper={site: ci_shard(v) for site, v in ci.upper.items()},
        )

    # ONE masked re-forward for recon AND the adversary ascents, sharing the same remat policy.
    # `remat_recon_forwards` gates gradient-checkpointing inside the target's `masked_output` at
    # the target's natural granularity (a deep target recomputes one layer at a time in the
    # backward instead of storing every layer's activations). This is load-bearing for the
    # ASCENTS too: though they backprop only to the SOURCES (params + CI detached), the source
    # gradient still flows through the per-layer activations (the masks MULTIPLY them), so an
    # un-rematted ascent forward stacks `[n_layer, *leading, d_ff]` MLP intermediates — measured
    # as the dominant step-memory term at depth (~6.6x peak vs rematted; the full-32L OOM).
    # Remat off stores all activations: faster when memory allows.
    @jaxtyped(typechecker=beartype)
    def masked_forward(
        model: DecomposedModel,
        prepared: Any,
        batch: Any,
        masks: dict[str, Float[Array, "*leading _"]],
        delta_masks: dict[str, Float[Array, "..."]],
        routes: dict[str, Bool[Array, "*leading"]] | None,
        live_sites: tuple[str, ...],
        has_delta: bool,
    ) -> Any:
        # `prepared` = `model.prepare_compute_weights(components_bf16)`, built ONCE per step and
        # shared across all forwards (the ÷N→÷fsdp gather is not re-run per forward).
        return batch_sharded(
            model.masked_output(
                prepared,
                batch,
                masks,
                delta_masks,
                routes,
                live_sites,
                has_delta,
                remat=remat_recon_forwards,
            )
        )

    def stochastic_entry_masks(
        ci_lower: dict[str, Array],
        live_sites: tuple[str, ...],
        leading_shape: tuple[int, ...],
        draw_key: PRNGKeyArray,
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        mask_source_key, delta_mask_key = random.split(draw_key)
        masks = {}
        delta_masks = {}
        for site_idx, site in enumerate(live_sites):
            ci_site = ci_lower[site]
            source_key = random.fold_in(mask_source_key, site_idx)
            stochastic_source = random.uniform(source_key, ci_site.shape, COMPUTE_DT)
            masks[site] = ci_site + (1.0 - ci_site) * stochastic_source
            delta_masks[site] = random.uniform(
                random.fold_in(delta_mask_key, site_idx), leading_shape, COMPUTE_DT
            )
        return masks, delta_masks

    def constant_entry_masks(
        strategy: ConstantSources,
        ci_lower: dict[str, Array],
        live_sites: tuple[str, ...],
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        """No delta masks: `has_delta` is False for constant sources, so the forward
        skips the `x @ Δ` matmul and never indexes the (empty) delta dict (§4b)."""
        masks = {
            site: ci_lower[site] + (1.0 - ci_lower[site]) * strategy.value for site in live_sites
        }
        return masks, {}

    def entry_loss_for_sources(
        entry: ReconForward,
        sources: dict[str, Array],
        routes_per_draw: tuple[Routes, ...],
        model: DecomposedModel,
        prepared: Any,
        ci_lower: dict[str, Array],
        batch: Any,
        clean_output: Array,
        forward_fn: Any,
    ) -> Array:
        """Mean KL over the entry's draws with FIXED source values — the adversarial
        ascent objective (shared by fresh and persistent ascents, SPEC S12'). `prepared` is
        the shared per-step compute weights (`prepare_compute_weights`)."""
        masks, delta_masks = source_masks(ci_lower, sources, entry.live_sites)
        total = jnp.zeros((), jnp.float32)
        for routes in routes_per_draw:
            masked = forward_fn(
                model,
                prepared,
                batch,
                masks,
                delta_masks,
                routes,
                entry.live_sites,
                entry.has_delta,
            )
            total = total + recon_loss_fn(masked, clean_output)
        return total / len(routes_per_draw)

    @eqx.filter_jit(donate="all-except-first")
    @jaxtyped(typechecker=beartype)
    def step(
        model: DecomposedModel,
        state: TrainState,
        batch: Any,
        key: PRNGKeyArray,
    ) -> tuple[TrainState, dict[str, Array]]:
        step_f32 = state.step.astype(jnp.float32)
        imp_min_param = annealed_imp_min_param(step_f32, total_steps, imp_min)

        batch = batch_sharded(batch)
        with jax.named_scope("pd_clean_fwd"):
            clean_output = jax.lax.stop_gradient(batch_sharded(model.clean_output(batch)))
        with jax.named_scope("pd_read_taps"):
            taps = model.read_activations(batch, state.ci_fn.input_names)
        # `leading` (batch, *positions) — the shape masks/sources/routes live in. Sourced
        # from a tap (always `[*leading, d_tap]`), not the opaque batch, so the engine never
        # assumes the batch's rank/feature dim.
        leading = next(iter(taps.values())).shape[:-1]

        # ── adversary ascents: params + CI detached (SPEC §4.5) ──
        components_detached = jax.lax.stop_gradient(cast_floating(state.components, COMPUTE_DT))
        prepared_detached = model.prepare_compute_weights(components_detached)
        ci_fn_detached = jax.lax.stop_gradient(cast_floating(state.ci_fn, COMPUTE_DT))
        with jax.named_scope("pd_ci_fn_fwd_detached"):
            ci_lower_detached = ci_batch_sharded(ci_fn_detached(taps, remat=False)).lower

        # ── persistent adversaries: each runs its supplemental ascents vs the route-ALL
        # all-sites forward (SPEC S24 — torch warmup parity, NOT the term's loss plan),
        # params + CI detached. The warmed sources then enter the main backward as leaves;
        # the LR schedule + `start_frac` gating (SPEC S32) live in `PersistentAdversary`. ──
        def warmup_scoring_loss(sources: dict[str, Array]) -> Array:
            masks, delta_masks = source_masks(ci_lower_detached, sources, site_names)
            masked = masked_forward(
                model, prepared_detached, batch, masks, delta_masks, None, site_names, True
            )
            return recon_loss_fn(masked, clean_output)

        with jax.named_scope("pd_pgd_warmup_ascend"):
            warmed_advs = {
                state_key: adv.warmup_ascend(warmup_scoring_loss, step_f32, total_steps)
                for state_key, adv in state.adversaries.items()
            }

        # Fresh-PGD entries: ONE routing draw per entry per step, shared by all
        # ascents and the main loss forward (SPEC S24); sign-ascend `n_steps`, then
        # the sources are constants in the main backward (torch parity).
        fresh_sources: dict[tuple[int, int], dict[str, Array]] = {}
        fixed_routes: dict[tuple[int, int], tuple[Routes, ...]] = {}
        for term_idx, term in enumerate(recon_terms):
            term_key = random.fold_in(key, 1 + term_idx)
            for entry_idx, entry in enumerate(term.plan):
                if not isinstance(entry.sources, FreshPGDSources):
                    continue
                fresh_cfg = entry.sources
                routing_key, init_key = random.split(random.fold_in(term_key, entry_idx))
                routes_per_draw = entry.sample_routing(routing_key, leading)
                fixed_routes[(term_idx, entry_idx)] = routes_per_draw
                live_specs = tuple(s for s in sites if s.name in entry.live_sites)
                init = init_fresh_pgd_sources(
                    live_specs, fresh_cfg.init, fresh_cfg.scope, leading, init_key
                )

                def ascent_loss(
                    sources: dict[str, Array],
                    entry: ReconForward = entry,
                    routes: tuple[Routes, ...] = routes_per_draw,
                ) -> Array:
                    return entry_loss_for_sources(
                        entry,
                        sources,
                        routes,
                        model,
                        prepared_detached,
                        ci_lower_detached,
                        batch,
                        clean_output,
                        masked_forward,
                    )

                def sign_ascend_body(
                    sources: dict[str, Array],
                    _: None,
                    ascent_loss: Callable[[dict[str, Array]], Array] = ascent_loss,
                    step_size: float = fresh_cfg.step_size,
                ) -> tuple[dict[str, Array], None]:
                    sources_grad = jax.grad(ascent_loss)(sources)
                    return {
                        site: jnp.clip(
                            sources[site] + step_size * jnp.sign(sources_grad[site]),
                            0.0,
                            1.0,
                        )
                        for site in sources
                    }, None

                with jax.named_scope("pd_fresh_pgd_ascend"):
                    ascended, _ = jax.lax.scan(
                        sign_ascend_body, init, None, length=fresh_cfg.n_steps
                    )
                fresh_sources[(term_idx, entry_idx)] = jax.lax.stop_gradient(ascended)

        # ── main losses: live components/ci; the PERSISTENT sources participate in
        # the graph so their gradient comes from the SAME backward (SPEC S14'); they
        # are NOT detached here, but components/ci grads through them are what torch
        # gets too (sources are leaves). ──
        def loss_fn(
            trainable: tuple[DecompVU, CIFn, dict[str, dict[str, Array]]],
        ) -> tuple[Array, tuple[Array, Array, Array, tuple[Array, ...]]]:
            components, ci_fn, persistent_sources = trainable
            components_bf16 = cast_floating(components, COMPUTE_DT)
            # ONE ÷N→÷fsdp reconstruction for the whole recon grid (shared by every forward +
            # its backward), instead of re-gathering per forward — the hoist.
            prepared = model.prepare_compute_weights(components_bf16)
            ci_fn_bf16 = cast_floating(ci_fn, COMPUTE_DT)
            with jax.named_scope("pd_ci_fn_fwd_main"):
                ci = ci_batch_sharded(ci_fn_bf16(taps, remat=remat_ci_fn))
            faith_loss = faithfulness_loss(model.weight_deltas(components))
            imp_lp, imp_freq = imp_min_terms(ci.upper, imp_min, imp_min_param)

            term_losses: list[Array] = []
            for term_idx, term in enumerate(recon_terms):
                term_key = random.fold_in(key, 1 + term_idx)
                total = jnp.zeros((), jnp.float32)
                n_forwards = 0
                for entry_idx, entry in enumerate(term.plan):
                    entry_key, routing_key = random.split(random.fold_in(term_key, entry_idx))
                    match entry.sources:
                        case FreshPGDSources():
                            routes_per_draw = fixed_routes[(term_idx, entry_idx)]
                        case _:
                            routes_per_draw = entry.sample_routing(routing_key, leading)
                    for draw_idx, routes in enumerate(routes_per_draw):
                        draw_key = random.fold_in(entry_key, draw_idx)
                        match entry.sources:
                            case StochasticSources():
                                masks, delta_masks = stochastic_entry_masks(
                                    ci.lower, entry.live_sites, leading, draw_key
                                )
                            case ConstantSources() as strategy:
                                masks, delta_masks = constant_entry_masks(
                                    strategy, ci.lower, entry.live_sites
                                )
                            case FreshPGDSources():
                                masks, delta_masks = source_masks(
                                    ci.lower,
                                    fresh_sources[(term_idx, entry_idx)],
                                    entry.live_sites,
                                )
                            case PersistentSources(state_key=state_key):
                                masks, delta_masks = source_masks(
                                    ci.lower, persistent_sources[state_key], entry.live_sites
                                )
                        with jax.named_scope("pd_recon_masked_fwd"):
                            masked = masked_forward(
                                model,
                                prepared,
                                batch,
                                masks,
                                delta_masks,
                                routes,
                                entry.live_sites,
                                entry.has_delta,
                            )
                        total = total + recon_loss_fn(masked, clean_output)
                        n_forwards += 1
                assert n_forwards > 0, f"term {term.name!r} produced no forwards"
                term_loss = total / n_forwards
                first_sources = term.plan[0].sources
                if isinstance(first_sources, PersistentSources):
                    start_frac = state.adversaries[first_sources.state_key].start_frac
                    if start_frac > 0.0:
                        term_loss = (step_f32 >= start_frac * total_steps) * term_loss
                term_losses.append(term_loss)

            total_loss = faith_coeff * faith_loss + imp_coeff * imp_lp + freq_coeff * imp_freq
            for term, term_loss in zip(recon_terms, term_losses, strict=True):
                total_loss = total_loss + term.coeff * term_loss
            return total_loss, (faith_loss, imp_lp, imp_freq, tuple(term_losses))

        with jax.named_scope("pd_value_and_grad"):
            (total_loss, (faith_loss, imp_lp, imp_freq, term_losses)), grads = (
                eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                    (state.components, state.ci_fn, {k: a.sources for k, a in warmed_advs.items()})
                )
            )
        components_grad, ci_fn_grad, persistent_grads_scaled = grads
        if mesh is not None and os.environ.get("PD_REPLICATE_WEIGHTS", "") == "1":
            # ZeRO-1 with replicated compute weights: the grad arrives REPLICATED (full V/U);
            # reduce-scatter it back to the ÷N master layout BEFORE the sharded optimizer
            # update, else the optimizer gathers master+Adam to full-replicated → OOM.
            components_grad = jax.lax.with_sharding_constraint(
                components_grad, state.components.shardings(mesh)
            )
        grad_norm_metrics = _grad_norm_metrics(components_grad, ci_fn_grad)

        # ── each adversary's final ascent from the fused graph (SPEC S13'/S14'): the
        # backward saw coeff·L_term, so it ascends on L_term itself (unscaled by its coeff
        # inside `final_ascend`, exact since one source bundle feeds one term, S23). ──
        new_adversaries = {
            state_key: warmed_advs[state_key].final_ascend(
                persistent_grads_scaled[state_key], step_f32, total_steps
            )
            for state_key in warmed_advs
        }

        components_updates, new_components_opt_state = components_optimizer.update(
            components_grad,
            state.components_opt_state,
            eqx.filter(state.components, eqx.is_array),
        )
        ci_fn_updates, new_ci_fn_opt_state = ci_fn_optimizer.update(
            ci_fn_grad, state.ci_fn_opt_state, eqx.filter(state.ci_fn, eqx.is_array)
        )
        new_components = eqx.apply_updates(state.components, components_updates)
        new_ci_fn = eqx.apply_updates(state.ci_fn, ci_fn_updates)

        new_state = TrainState(
            components=new_components,
            ci_fn=new_ci_fn,
            components_opt_state=new_components_opt_state,
            ci_fn_opt_state=new_ci_fn_opt_state,
            adversaries=new_adversaries,
            step=state.step + 1,
        )
        metrics = {
            "total": total_loss,
            "faith": faith_loss,
            imp_loss_key: imp_lp,
            "freq": imp_freq,
            imp_min_param_key: imp_min_param,
            **{f"loss/{t.name}": v for t, v in zip(recon_terms, term_losses, strict=True)},
            **grad_norm_metrics,
        }
        source_lrs = {
            k: adv.source_lr(step_f32, total_steps) for k, adv in state.adversaries.items()
        }
        if len(source_lrs) == 1:
            metrics["src_lr"] = next(iter(source_lrs.values()))
        else:
            metrics |= {f"schedules/lr/src/{k}": v for k, v in source_lrs.items()}
        return new_state, metrics

    return step


# ───────────────────────────── faithfulness warmup (SPEC S21) ─────────────────────────────


def make_faith_warmup_step(
    opt: optax.GradientTransformation,
) -> Callable[[DecomposedModel, DecompVU, optax.OptState], tuple[DecompVU, optax.OptState, Array]]:
    """`model` is the jit ARG (frozen weights traced, not baked) — `weight_deltas` reads its
    per-site W slices, so closing over the model would bake them into the HLO."""

    @eqx.filter_jit
    def warmup_step(
        model: DecomposedModel, components: DecompVU, opt_state: optax.OptState
    ) -> tuple[DecompVU, optax.OptState, Array]:
        def loss_fn(components_: DecompVU) -> Array:
            return faithfulness_loss(model.weight_deltas(components_))

        loss, grad = eqx.filter_value_and_grad(loss_fn)(components)
        updates, opt_state = opt.update(grad, opt_state, eqx.filter(components, eqx.is_array))
        return eqx.apply_updates(components, updates), opt_state, loss

    return warmup_step
