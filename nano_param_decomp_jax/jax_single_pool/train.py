"""The generic single-pool VPD training step over a `DecomposedLM` (SPEC §4).

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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, PRNGKeyArray

from jax_single_pool.adversary import (
    SourcesAdamState,
    init_fresh_pgd_sources,
    source_masks,
    sources_adam_ascend_project,
)
from jax_single_pool.ci_fn import CIFn, CIValues
from jax_single_pool.lm import DecomposedLM
from jax_single_pool.losses import (
    annealed_pnorm,
    faithfulness_loss,
    importance_minimality_terms,
    kl_per_position,
    warmup_then_constant_lr,
)
from jax_single_pool.recon import (
    ConstantSources,
    FreshPGDSources,
    LossSpec,
    PersistentSources,
    ReconForward,
    Routes,
    StochasticSources,
)
from param_decomp_config.losses import AdamPGDConfig

COMPUTE_DT = jnp.bfloat16


def cast_floating(tree: Any, dtype: Any) -> Any:
    return jax.tree.map(lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, tree)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainState:
    components: Any  # LM-specific trainable pytree (V/U), fp32 masters
    ci_fn: CIFn  # fp32 masters
    components_opt_state: optax.OptState
    ci_fn_opt_state: optax.OptState
    sources: dict[str, dict[str, Array]]
    """Persistent adversarial sources, `state_key -> site -> (1, T, C+1)` in [0,1].
    One state_key per persistent loss term (SPEC S23); empty when no persistent term."""
    sources_opt_state: dict[str, SourcesAdamState]
    step: Array


def _grad_norm_metrics(components_grad: Any, ci_fn_grad: Any) -> dict[str, Array]:
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
    lm: DecomposedLM,
    *,
    loss_spec: LossSpec,
    components_optimizer: optax.GradientTransformation,
    ci_fn_optimizer: optax.GradientTransformation,
    total_steps: int,
    remat_recon_forwards: bool,
    mesh: Mesh | None,
):
    """Build the jit'd `step(state, frozen, residual, key) -> (state, metrics)`.

    `loss_spec` (from `build_recon_terms`) carries the SHARED torch loss configs
    mapped onto recon terms; the supported subset is asserted there. `mesh` (when
    given) pins every batch-leading activation to `P('dp', ...)` so the masked
    re-forwards stay on per-device sub-batches (activation memory 1/n_dev)."""
    site_names = lm.site_names
    recon_terms = loss_spec.recon_terms
    imp_min = loss_spec.imp_min
    faith_coeff = loss_spec.faith_coeff
    assert imp_min.coeff is not None and imp_min.p_anneal_final_p is not None
    imp_coeff = imp_min.coeff
    term_coeff_by_state_key = {
        entry.sources.state_key: term.coeff
        for term in recon_terms
        for entry in term.plan
        if isinstance(entry.sources, PersistentSources)
    }
    assert set(term_coeff_by_state_key) == set(loss_spec.persistent)
    persistent_adams: dict[str, AdamPGDConfig] = {}
    for state_key, ppgd_cfg in loss_spec.persistent.items():
        optimizer = ppgd_cfg.optimizer
        assert isinstance(optimizer, AdamPGDConfig)
        persistent_adams[state_key] = optimizer

    def batch_sharded(x: Array) -> Array:
        if mesh is None:
            return x
        spec = ["dp"] + [None] * (x.ndim - 1)
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(*spec)))

    def batch_sharded_ci(ci_values: CIValues) -> CIValues:
        """Reshard the CI-fn output to batch-sharded ONCE, here. The CI head's `out_w`
        is ΣC-sharded, so its output is born C-sharded; without a single producer-side
        pin, GSPMD inserts a separate C→batch reshard for every consumer (each plan
        forward, the adversaries, imp-min — forward and backward), and those
        all-to-all buffers dominate the temp arena at scale."""
        return CIValues(
            lower={site: batch_sharded(v) for site, v in ci_values.lower.items()},
            upper={site: batch_sharded(v) for site, v in ci_values.upper.items()},
        )

    def masked_forward(
        frozen: Any,
        components_bf16: Any,
        residual: Array,
        masks: dict[str, Array],
        delta_masks: dict[str, Array],
        routes: dict[str, Array] | None,
        live_sites: tuple[str, ...],
    ) -> Array:
        return batch_sharded(
            lm.masked_logits(
                frozen, components_bf16, residual, masks, delta_masks, routes, live_sites
            )
        )

    # Recomputing each masked forward in backward bounds activation memory to one
    # forward at a time (the torch 2-pool streaming profile) at the cost of the
    # recompute; with few recon forwards and memory headroom, remat off is faster.
    checkpointed_masked_forward = (
        jax.checkpoint(masked_forward, static_argnums=(6,))
        if remat_recon_forwards
        else masked_forward
    )

    def stochastic_entry_masks(
        strategy: StochasticSources,
        ci_lower: dict[str, Array],
        live_sites: tuple[str, ...],
        batch_seq: tuple[int, int],
        draw_key: PRNGKeyArray,
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        mask_source_key, delta_mask_key = random.split(draw_key)
        masks = {}
        delta_masks = {}
        for site_idx, site in enumerate(live_sites):
            ci_site = ci_lower[site]
            source_key = random.fold_in(mask_source_key, site_idx)
            match strategy.sampling:
                case "continuous":
                    stochastic_source = random.uniform(source_key, ci_site.shape, COMPUTE_DT)
                case "binomial":
                    stochastic_source = random.bernoulli(source_key, 0.5, ci_site.shape).astype(
                        COMPUTE_DT
                    )
            masks[site] = ci_site + (1.0 - ci_site) * stochastic_source
            delta_masks[site] = random.uniform(
                random.fold_in(delta_mask_key, site_idx), batch_seq, COMPUTE_DT
            )
        return masks, delta_masks

    def constant_entry_masks(
        strategy: ConstantSources,
        ci_lower: dict[str, Array],
        live_sites: tuple[str, ...],
        batch_seq: tuple[int, int],
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        masks = {}
        delta_masks = {}
        for site in live_sites:
            ci_site = ci_lower[site]
            masks[site] = ci_site + (1.0 - ci_site) * strategy.value
            delta_masks[site] = jnp.zeros(batch_seq, ci_site.dtype)
        return masks, delta_masks

    def entry_loss_for_sources(
        entry: ReconForward,
        sources: dict[str, Array],
        routes_per_draw: tuple[Routes, ...],
        frozen: Any,
        components_bf16: Any,
        ci_lower: dict[str, Array],
        residual: Array,
        clean_logits: Array,
        forward_fn: Any,
    ) -> Array:
        """Mean KL over the entry's draws with FIXED source values — the adversarial
        ascent objective (shared by fresh and persistent ascents, SPEC S12')."""
        masks, delta_masks = source_masks(ci_lower, sources, entry.live_sites)
        total = jnp.zeros((), jnp.float32)
        for routes in routes_per_draw:
            masked = forward_fn(
                frozen, components_bf16, residual, masks, delta_masks, routes, entry.live_sites
            )
            total = total + kl_per_position(masked, clean_logits)
        return total / len(routes_per_draw)

    @jax.jit
    def step(
        state: TrainState, frozen: Any, residual: Float[Array, "b t d"], key: PRNGKeyArray
    ) -> tuple[TrainState, dict[str, Array]]:
        step_f32 = state.step.astype(jnp.float32)
        pnorm = annealed_pnorm(step_f32, total_steps, imp_min)
        batch, seq = residual.shape[0], residual.shape[1]

        residual = batch_sharded(residual)
        clean_logits = jax.lax.stop_gradient(batch_sharded(lm.clean_logits(frozen, residual)))
        site_inputs = lm.site_inputs(frozen, residual)

        # ── adversary ascents: params + CI detached (SPEC §4.5) ──
        components_detached = jax.lax.stop_gradient(cast_floating(state.components, COMPUTE_DT))
        ci_fn_detached = jax.lax.stop_gradient(cast_floating(state.ci_fn, COMPUTE_DT))
        ci_lower_detached = batch_sharded_ci(ci_fn_detached(site_inputs)).lower

        # Persistent terms: n_warmup supplemental Adam ascents each, against the
        # route-ALL all-sites forward (SPEC S24 — torch warmup parity, NOT the term's
        # loss plan), sequential per term as in torch.
        source_lrs: dict[str, Array] = {}
        warmed_sources: dict[str, dict[str, Array]] = {}
        warmup_opt_states: dict[str, SourcesAdamState] = {}
        for state_key, ppgd_cfg in loss_spec.persistent.items():
            adam = persistent_adams[state_key]
            source_lr = warmup_then_constant_lr(
                step_f32,
                total_steps,
                adam.lr_schedule.start_val,
                adam.lr_schedule.warmup_pct,
            )
            source_lrs[state_key] = source_lr

            def warmup_loss(sources: dict[str, Array]) -> Array:
                masks, delta_masks = source_masks(ci_lower_detached, sources, site_names)
                masked = masked_forward(
                    frozen, components_detached, residual, masks, delta_masks, None, site_names
                )
                return kl_per_position(masked, clean_logits)

            def warmup_body(
                carry: tuple[dict[str, Array], SourcesAdamState], _: None
            ) -> tuple[tuple[dict[str, Array], SourcesAdamState], None]:
                sources, adam_state = carry
                sources_grad = jax.grad(warmup_loss)(sources)
                sources, adam_state = sources_adam_ascend_project(
                    sources, sources_grad, adam_state, source_lr, adam
                )
                return (sources, adam_state), None

            (warmed, warmed_opt), _ = jax.lax.scan(
                warmup_body,
                (state.sources[state_key], state.sources_opt_state[state_key]),
                None,
                length=ppgd_cfg.n_warmup_steps,
            )
            warmed_sources[state_key] = jax.lax.stop_gradient(warmed)
            warmup_opt_states[state_key] = warmed_opt

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
                routes_per_draw = entry.sample_routing(routing_key, (batch, seq))
                fixed_routes[(term_idx, entry_idx)] = routes_per_draw
                live_specs = tuple(s for s in lm.sites if s.name in entry.live_sites)
                init = init_fresh_pgd_sources(
                    live_specs, fresh_cfg.init, fresh_cfg.scope, batch, seq, init_key
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
                        frozen,
                        components_detached,
                        ci_lower_detached,
                        residual,
                        clean_logits,
                        masked_forward,
                    )

                def sign_ascend_body(
                    sources: dict[str, Array],
                    _: None,
                    ascent_loss: Callable[[dict[str, Array]], Array] = ascent_loss,
                ) -> tuple[dict[str, Array], None]:
                    sources_grad = jax.grad(ascent_loss)(sources)
                    return {
                        site: jnp.clip(
                            sources[site] + fresh_cfg.step_size * jnp.sign(sources_grad[site]),
                            0.0,
                            1.0,
                        )
                        for site in sources
                    }, None

                ascended, _ = jax.lax.scan(sign_ascend_body, init, None, length=fresh_cfg.n_steps)
                fresh_sources[(term_idx, entry_idx)] = jax.lax.stop_gradient(ascended)

        # ── main losses: live components/ci; the PERSISTENT sources participate in
        # the graph so their gradient comes from the SAME backward (SPEC S14'); they
        # are NOT detached here, but components/ci grads through them are what torch
        # gets too (sources are leaves). ──
        def loss_fn(
            trainable: tuple[Any, CIFn, dict[str, dict[str, Array]]],
        ) -> tuple[Array, tuple[Array, Array, Array, tuple[Array, ...]]]:
            components, ci_fn, persistent_sources = trainable
            components_bf16 = cast_floating(components, COMPUTE_DT)
            ci_fn_bf16 = cast_floating(ci_fn, COMPUTE_DT)
            ci = batch_sharded_ci(ci_fn_bf16(site_inputs))
            faith_loss = faithfulness_loss(lm.weight_deltas(frozen, components))
            imp_lp, imp_entropy = importance_minimality_terms(ci.upper, pnorm, imp_min.eps)
            imp_loss = imp_lp + imp_min.beta * imp_entropy

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
                            routes_per_draw = entry.sample_routing(routing_key, (batch, seq))
                    for draw_idx, routes in enumerate(routes_per_draw):
                        draw_key = random.fold_in(entry_key, draw_idx)
                        match entry.sources:
                            case StochasticSources() as strategy:
                                masks, delta_masks = stochastic_entry_masks(
                                    strategy, ci.lower, entry.live_sites, (batch, seq), draw_key
                                )
                            case ConstantSources() as strategy:
                                masks, delta_masks = constant_entry_masks(
                                    strategy, ci.lower, entry.live_sites, (batch, seq)
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
                        masked = checkpointed_masked_forward(
                            frozen,
                            components_bf16,
                            residual,
                            masks,
                            delta_masks,
                            routes,
                            entry.live_sites,
                        )
                        total = total + kl_per_position(masked, clean_logits)
                        n_forwards += 1
                assert n_forwards > 0, f"term {term.name!r} produced no forwards"
                term_losses.append(total / n_forwards)

            total_loss = faith_coeff * faith_loss + imp_coeff * imp_loss
            for term, term_loss in zip(recon_terms, term_losses, strict=True):
                total_loss = total_loss + term.coeff * term_loss
            return total_loss, (faith_loss, imp_loss, imp_lp, tuple(term_losses))

        (total_loss, (faith_loss, imp_loss, imp_lp, term_losses)), grads = (
            eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                (state.components, state.ci_fn, warmed_sources)
            )
        )
        components_grad, ci_fn_grad, persistent_grads_scaled = grads
        grad_norm_metrics = _grad_norm_metrics(components_grad, ci_fn_grad)

        # ── each persistent term's final ascent, from the fused graph (SPEC S13'/S14');
        # the backward saw coeff·L_term, the adversary ascends on L_term itself, and the
        # division is exact because each source bundle feeds exactly one term (S23). ──
        new_sources: dict[str, dict[str, Array]] = {}
        new_sources_opt_state: dict[str, SourcesAdamState] = {}
        for state_key, ppgd_cfg in loss_spec.persistent.items():
            coeff = term_coeff_by_state_key[state_key]
            sources_grad = {
                site: g / coeff for site, g in persistent_grads_scaled[state_key].items()
            }
            ascended, ascended_opt = sources_adam_ascend_project(
                warmed_sources[state_key],
                sources_grad,
                warmup_opt_states[state_key],
                source_lrs[state_key],
                persistent_adams[state_key],
            )
            new_sources[state_key] = ascended
            new_sources_opt_state[state_key] = ascended_opt

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
            sources=new_sources,
            sources_opt_state=new_sources_opt_state,
            step=state.step + 1,
        )
        metrics = {
            "total": total_loss,
            "faith": faith_loss,
            "imp": imp_loss,
            "imp_no_beta": imp_lp,
            "p_imp": pnorm,
            **{f"loss/{t.name}": v for t, v in zip(recon_terms, term_losses, strict=True)},
            **grad_norm_metrics,
        }
        if len(source_lrs) == 1:
            metrics["src_lr"] = next(iter(source_lrs.values()))
        else:
            metrics |= {f"schedules/lr/src/{k}": v for k, v in source_lrs.items()}
        return new_state, metrics

    return step


# ───────────────────────────── faithfulness warmup (SPEC S21) ─────────────────────────────


def make_faith_warmup_step(
    lm: DecomposedLM, opt: optax.GradientTransformation
) -> Callable[[Any, optax.OptState, Any], tuple[Any, optax.OptState, Array]]:
    @jax.jit
    def warmup_step(
        components: Any, opt_state: optax.OptState, frozen: Any
    ) -> tuple[Any, optax.OptState, Array]:
        def loss_fn(components_: Any) -> Array:
            return faithfulness_loss(lm.weight_deltas(frozen, components_))

        loss, grad = eqx.filter_value_and_grad(loss_fn)(components)
        updates, opt_state = opt.update(grad, opt_state, eqx.filter(components, eqx.is_array))
        return eqx.apply_updates(components, updates), opt_state, loss

    return warmup_step
