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

from param_decomp.core.adversary import (
    PersistentAdversary,
    init_fresh_pgd_sources,
    mixed_persistent_stochastic_masks,
    source_masks,
)
from param_decomp.core.ci_fn import CI, CIFn
from param_decomp.core.components import ComponentStacks
from param_decomp.core.configs import SmoothL0ImportanceMinimalityLossConfig
from param_decomp.core.jit_util import filter_jit
from param_decomp.core.losses import (
    annealed_imp_min_param,
    faithfulness_loss,
    imp_min_terms,
    scheduled_value_traced,
)
from param_decomp.core.model import DecomposedModel
from param_decomp.core.recon import (
    ConstantSources,
    FreshPGDSources,
    LossSurface,
    MixedPersistentStochasticSources,
    PersistentSources,
    ReconForward,
    ReconLossTerm,
    Routes,
    StochasticSources,
)
from param_decomp.core.sharding import batch_shard_leading

COMPUTE_DT = jnp.bfloat16


def cast_floating(tree: Any, dtype: Any) -> Any:
    return jax.tree.map(lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, tree)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Decomposition:
    """The trained PRODUCT: V/U components + the CI fn (fp32 masters). Checkpointed as
    its own orbax item so consumers (harvest/autointerp/clustering/app) restore it with
    zero knowledge of the training process (optimizer states, adversaries, step)."""

    components: ComponentStacks  # the universal trainable V/U pytree, fp32 masters
    ci_fn: CIFn  # fp32 masters


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainingItem:
    """The trainer-only trajectory tail: both optimizer states, the persistent adversaries,
    the step counter. Checkpointed as its own orbax item — no consumer restores it."""

    components_opt_state: optax.OptState
    ci_fn_opt_state: optax.OptState
    adversaries: dict[str, PersistentAdversary]
    """Persistent-PGD adversaries, `state_key -> adversary` (each owns its sources + Adam
    state + static config). One state_key per persistent loss term (SPEC S23); empty when
    no persistent term."""
    step: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainState:
    """The full training pytree, composed of the two checkpoint items so there is ONE
    representation: `decomposition` is the trained product, `training` the trajectory tail.
    Save/restore maps directly onto these two fields — no regrouping."""

    decomposition: Decomposition
    training: TrainingItem


def _grad_norm_metrics(components_grad: ComponentStacks, ci_fn_grad: Any) -> dict[str, Array]:
    """Pre-clip gradient L2 norms, matching the torch `component_grad_norms` families.

    Components norms are per SITE per factor — `grad_norms/components.vu['<site>'][0|1]`
    (0=V, 1=U), e.g. `grad_norms/components.vu['layers.18.mlp.gate_proj'][0]`. Sites are
    the semantic unit; the shape-group stacks they're stored in are not. The key spells
    the retired per-site pytree path so wandb histories overlay across the stacking
    refactor. Ci-fn norms are per LEAF of whatever pytree the CI fn is
    (`grad_norms/ci_fns<path>`), plus the overlay-critical
    `grad_norms/summary/{components,ci_fns,total}`."""
    out: dict[str, Array] = {}

    def per_slice_sq(stack: Float[Array, "g a b"]) -> Float[Array, " g"]:
        return jnp.sum(stack.astype(jnp.float32) ** 2, axis=(1, 2))

    factor_sq = {
        shape: (per_slice_sq(Vs), per_slice_sq(Us))
        for shape, (Vs, Us) in components_grad.stacks.items()
    }
    for name, shape, slot in components_grad.site_slots:
        v_sq, u_sq = factor_sq[shape]
        out[f"grad_norms/components.vu['{name}'][0]"] = jnp.sqrt(v_sq[slot])
        out[f"grad_norms/components.vu['{name}'][1]"] = jnp.sqrt(u_sq[slot])
    components_sq = jnp.zeros((), jnp.float32)
    for v_sq, u_sq in factor_sq.values():
        components_sq = components_sq + jnp.sum(v_sq) + jnp.sum(u_sq)
    out["grad_norms/summary/components"] = jnp.sqrt(components_sq)

    ci_fn_sq = jnp.zeros((), jnp.float32)
    for path, leaf in jax.tree_util.tree_flatten_with_path(ci_fn_grad)[0]:
        leaf_sq = jnp.sum(leaf.astype(jnp.float32) ** 2)
        out[f"grad_norms/ci_fns{jax.tree_util.keystr(path)}"] = jnp.sqrt(leaf_sq)
        ci_fn_sq = ci_fn_sq + leaf_sq
    out["grad_norms/summary/ci_fns"] = jnp.sqrt(ci_fn_sq)

    out["grad_norms/summary/total"] = jnp.sqrt(components_sq + ci_fn_sq)
    return out


# ───────────────────────────── the step factory ─────────────────────────────


def make_train_step(
    model_static: DecomposedModel,
    *,
    losses: LossSurface,
    components_optimizer: optax.GradientTransformation,
    ci_fn_optimizer: optax.GradientTransformation,
    total_steps: int,
    remat_recon_forwards: bool,
    remat_ci_fn: bool,
    mesh: Mesh | None,
    ascend_replicate: bool = False,
    compiler_options: dict[str, bool | int | str] | None = None,
):
    """Build the `eqx.filter_jit`'d `step(model, state, batch, key) -> (state, metrics)`.

    `model` is the jit ARG (frozen 8B weights traced as array leaves, never baked); the
    factory closes over only static config (`site_names`, `recon_loss_fn`, term wiring) read
    off `model_static` here — the distinct name keeps an accidental closure over the
    array-bearing model (the HLO-baking hazard) a loud NameError, never silent.
    `losses` (from `build_loss_terms`) is the `LossSurface` record — the
    faithfulness + importance-minimality singletons and the recon Σ, read by name. `mesh`
    (when given) pins every batch-leading activation over the full mesh
    (`P(('replicate', 'fsdp'), ...)`) so the masked re-forwards stay on per-rank sub-batches
    (activation memory 1/N)."""
    site_names = model_static.site_names
    sites = model_static.sites
    c_by_site = {spec.name: spec.C for spec in sites}
    recon_loss_fn = model_static.recon_loss_fn  # static: pure, holds no arrays — safe to close
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

    def replicate_for_ascend(prepared: Any) -> Any:
        """Lever #5 (`RuntimeConfig.ascend_replicate`): gather the ÷fsdp compute weights to
        FULL/replicated ONCE before the adversary ascents, so the `n_warmup` ascend forwards run
        plain matmuls with NO per-layer ÷fsdp→full NVLink gather. The gather is
        mask-INDEPENDENT and the V/U are detached (constant) across ascend steps, so the
        re-gather is pure redundancy — `n_warmup × n_layer × (fwd+bwd)` collectives collapse to
        one full gather. Trades the full V/U resident (≈ `fsdp`× the ÷fsdp stack) during the
        ascend phase for the eliminated re-gathers. Pure data movement (bf16 values unchanged) →
        numerics bit-identical. No-op off-flag / off-mesh."""
        if not ascend_replicate or mesh is None or jax.sharding.get_abstract_mesh().empty:
            return prepared
        replicated = NamedSharding(mesh, P())
        return jax.tree.map(lambda a: jax.lax.with_sharding_constraint(a, replicated), prepared)

    # ONE masked re-forward for recon AND the adversary ascents, sharing the same remat policy.
    # `remat_recon_forwards` gates gradient-checkpointing inside the target's `masked_output` at
    # the target's natural granularity (a deep target recomputes one layer at a time in the
    # backward instead of storing every layer's activations). This is load-bearing for the
    # ASCENTS too: though they backprop only to the SOURCES (params + CI detached), the source
    # gradient still flows through the per-layer activations (the masks MULTIPLY them), so an
    # un-rematted ascent forward stacks `[n_layer, *leading, d_ff]` MLP intermediates.
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
    ) -> Array:
        """Mean KL over the entry's draws with FIXED source values — the adversarial
        ascent objective (shared by fresh and persistent ascents, SPEC S12'). `prepared` is
        the shared per-step compute weights (`prepare_compute_weights`)."""
        masks, delta_masks = source_masks(ci_lower, sources, entry.live_sites)
        total = jnp.zeros((), jnp.float32)
        for routes in routes_per_draw:
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
        return total / len(routes_per_draw)

    @jaxtyped(typechecker=beartype)
    def step(
        model: DecomposedModel,
        state: TrainState,
        batch: Any,
        key: PRNGKeyArray,
    ) -> tuple[TrainState, dict[str, Array]]:
        decomposition = state.decomposition
        training = state.training
        step_f32 = training.step.astype(jnp.float32)
        imp_min_param = annealed_imp_min_param(step_f32, total_steps, imp_min)

        batch = batch_sharded(batch)
        with jax.named_scope("pd_clean_fwd_and_taps"):
            clean_output, taps = model.clean_output_and_activations(
                batch, decomposition.ci_fn.input_names
            )
            clean_output = jax.lax.stop_gradient(batch_sharded(clean_output))
        # `leading` (batch, *positions) — the shape masks/sources/routes live in. Sourced
        # from a tap (always `[*leading, d_tap]`), not the opaque batch, so the engine never
        # assumes the batch's rank/feature dim.
        leading = next(iter(taps.values())).shape[:-1]

        # ── adversary ascents: params + CI detached (SPEC §4.5) ──
        prepared, recon_vjp = jax.vjp(
            lambda c: model.prepare_compute_weights(cast_floating(c, COMPUTE_DT)),
            decomposition.components,
        )
        prepared_detached = jax.lax.stop_gradient(prepared)
        prepared_ascend = replicate_for_ascend(prepared_detached)
        # The CI envelope is a pure fn of the batch, so compute it ONCE per step — the value +
        # its vjp, mirroring `prepared`/`recon_vjp`. The ascend uses the stop_gradient'd value;
        # `loss_fn` takes the live value and its ci-fn grad is pulled back through `ci_vjp`. So the
        # (≈10x-the-target) CI fn is forward-evaluated ONCE, not once detached for the ascend +
        # once inside the main backward.
        with jax.named_scope("pd_ci_fn_fwd"):
            ci, ci_vjp = eqx.filter_vjp(
                lambda cf: ci_batch_sharded(cast_floating(cf, COMPUTE_DT)(taps, remat=remat_ci_fn)),
                decomposition.ci_fn,
            )
        ci_lower_detached = jax.lax.stop_gradient(ci).lower

        # ── persistent adversaries: each runs its supplemental ascents vs the route-ALL
        # all-sites forward (SPEC S24 — torch warmup parity, NOT the term's loss plan),
        # params + CI detached. The warmed sources then enter the main backward as leaves;
        # the LR schedule (S13′) lives in `PersistentAdversary`. ──
        def warmup_scoring_loss(sources: dict[str, Array]) -> Array:
            masks, delta_masks = source_masks(ci_lower_detached, sources, site_names)
            masked = masked_forward(
                model, prepared_ascend, batch, masks, delta_masks, None, site_names, True
            )
            return recon_loss_fn(masked, clean_output)

        with jax.named_scope("pd_pgd_warmup_ascend"):
            warmed_advs = {
                state_key: adv.warmup_ascend(warmup_scoring_loss, step_f32, total_steps)
                for state_key, adv in training.adversaries.items()
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
                    live_specs, fresh_cfg.init, fresh_cfg.source_shape, leading, init_key
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
                        prepared_ascend,
                        ci_lower_detached,
                        batch,
                        clean_output,
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
        warmed_sources = {k: a.sources for k, a in warmed_advs.items()}

        def draw_recon_loss(
            prepared: Any,
            ci_lower: dict[str, Array],
            persistent_sources: dict[str, dict[str, Array]],
            ci_stacked: dict[str, Array] | None,
            term_idx: int,
            entry_idx: int,
            entry: ReconForward,
            draw_key: PRNGKeyArray,
            routes: Routes,
        ) -> Array:
            """One recon forward's KL: dispatch on the entry's mask-source strategy, run the
            masked forward, compare against the clean output (SPEC S10')."""
            with jax.named_scope("pd_recon_masked_fwd"):
                match entry.sources:
                    case StochasticSources():
                        # Stochastic recon builds its masks INSIDE the target's
                        # `masked_output_stochastic` from the shared `ci_stacked` — a scan
                        # target recomputes them in its checkpointed block (mask never held,
                        # the memory win); others build masks then `masked_output`. Either way
                        # the engine holds no per-forward mask stacks.
                        assert ci_stacked is not None
                        masked = batch_sharded(
                            model.masked_output_stochastic(
                                prepared,
                                batch,
                                ci_stacked,
                                draw_key,
                                routes,
                                entry.live_sites,
                                entry.has_delta,
                                remat=remat_recon_forwards,
                            )
                        )
                    case ConstantSources() as strategy:
                        masks, delta_masks = constant_entry_masks(
                            strategy, ci_lower, entry.live_sites
                        )
                        masked = masked_forward(
                            model, prepared, batch, masks, delta_masks,
                            routes, entry.live_sites, entry.has_delta,
                        )  # fmt: skip
                    case FreshPGDSources():
                        masks, delta_masks = source_masks(
                            ci_lower, fresh_sources[(term_idx, entry_idx)], entry.live_sites
                        )
                        masked = masked_forward(
                            model, prepared, batch, masks, delta_masks,
                            routes, entry.live_sites, entry.has_delta,
                        )  # fmt: skip
                    case PersistentSources(state_key=state_key):
                        masks, delta_masks = source_masks(
                            ci_lower, persistent_sources[state_key], entry.live_sites
                        )
                        masked = masked_forward(
                            model, prepared, batch, masks, delta_masks,
                            routes, entry.live_sites, entry.has_delta,
                        )  # fmt: skip
                    case MixedPersistentStochasticSources(state_key=state_key):
                        adv_fraction = scheduled_value_traced(
                            step_f32, total_steps, entry.sources.cfg.adv_fraction
                        )
                        mixed_masks, mixed_deltas, mixed_routes = mixed_persistent_stochastic_masks(
                            draw_key,
                            ci_lower,
                            persistent_sources[state_key],
                            entry.live_sites,
                            c_by_site,
                            leading,
                            adv_fraction,
                            routes,
                        )
                        masked = masked_forward(
                            model, prepared, batch, mixed_masks, mixed_deltas,
                            mixed_routes, entry.live_sites, entry.has_delta,
                        )  # fmt: skip
            return recon_loss_fn(masked, clean_output)

        def term_draws(
            term_idx: int, term: ReconLossTerm
        ) -> list[tuple[int, ReconForward, PRNGKeyArray, Routes]]:
            """The term's flat `(entry_idx, entry, draw_key, routes)` forwards. Key derivation
            reproduces the pre-unification production trace exactly (SPEC R1); fresh-PGD
            entries reuse the ascent phase's fixed routes (SPEC S24)."""
            term_key = random.fold_in(key, 1 + term_idx)
            draws: list[tuple[int, ReconForward, PRNGKeyArray, Routes]] = []
            for entry_idx, entry in enumerate(term.plan):
                entry_key, routing_key = random.split(random.fold_in(term_key, entry_idx))
                match entry.sources:
                    case FreshPGDSources():
                        routes_per_draw = fixed_routes[(term_idx, entry_idx)]
                    case _:
                        routes_per_draw = entry.sample_routing(routing_key, leading)
                draws.extend(
                    (entry_idx, entry, random.fold_in(entry_key, draw_idx), routes)
                    for draw_idx, routes in enumerate(routes_per_draw)
                )
            assert draws, f"term {term.name!r} produced no forwards"
            return draws

        per_term_draws = [term_draws(term_idx, term) for term_idx, term in enumerate(recon_terms)]

        def loss_fn(
            trainable: tuple[Any, ComponentStacks, CI, dict[str, dict[str, Array]]],
        ) -> tuple[Array, tuple[Array, Array, Array, tuple[Array, ...]]]:
            prepared, components, ci, persistent_sources = trainable
            ci_stacked = model.stack_ci(ci.lower)
            faith_loss = faithfulness_loss(model.weight_deltas(components))
            imp_lp, imp_freq = imp_min_terms(ci.upper, imp_min, imp_min_param)

            term_losses: list[Array] = []
            for term_idx, draws in enumerate(per_term_draws):
                total = jnp.zeros((), jnp.float32)
                for entry_idx, entry, draw_key, routes in draws:
                    total = total + draw_recon_loss(
                        prepared, ci.lower, persistent_sources, ci_stacked,
                        term_idx, entry_idx, entry, draw_key, routes,
                    )  # fmt: skip
                term_losses.append(total / len(draws))

            total_loss = faith_coeff * faith_loss + imp_coeff * imp_lp + freq_coeff * imp_freq
            for term, term_loss in zip(recon_terms, term_losses, strict=True):
                total_loss = total_loss + term.coeff * term_loss
            return total_loss, (faith_loss, imp_lp, imp_freq, tuple(term_losses))

        with jax.named_scope("pd_value_and_grad"):
            (total_loss, (faith_loss, imp_lp, imp_freq, term_losses)), grads = (
                eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                    (prepared, decomposition.components, ci, warmed_sources)
                )
            )
        prepared_grad, components_grad_faith, ci_grad, persistent_grads_scaled = grads
        components_grad_recon = recon_vjp(prepared_grad)[0]
        ci_fn_grad = ci_vjp(ci_grad)[0]
        components_grad = jax.tree.map(
            lambda recon_g, faith_g: recon_g + faith_g,
            components_grad_recon,
            components_grad_faith,
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
            training.components_opt_state,
            eqx.filter(decomposition.components, eqx.is_array),
        )
        ci_fn_updates, new_ci_fn_opt_state = ci_fn_optimizer.update(
            ci_fn_grad, training.ci_fn_opt_state, eqx.filter(decomposition.ci_fn, eqx.is_array)
        )
        new_components = eqx.apply_updates(decomposition.components, components_updates)
        new_ci_fn = eqx.apply_updates(decomposition.ci_fn, ci_fn_updates)

        new_state = TrainState(
            decomposition=Decomposition(components=new_components, ci_fn=new_ci_fn),
            training=TrainingItem(
                components_opt_state=new_components_opt_state,
                ci_fn_opt_state=new_ci_fn_opt_state,
                adversaries=new_adversaries,
                step=training.step + 1,
            ),
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
            k: adv.source_lr(step_f32, total_steps) for k, adv in training.adversaries.items()
        }
        if len(source_lrs) == 1:
            metrics["src_lr"] = next(iter(source_lrs.values()))
        else:
            metrics |= {f"schedules/lr/src/{k}": v for k, v in source_lrs.items()}
        return new_state, metrics

    return filter_jit(step, donate="all-except-first", compiler_options=compiler_options)


# ───────────────────────────── faithfulness warmup (SPEC S21) ─────────────────────────────


def make_faith_warmup_step(
    opt: optax.GradientTransformation,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> Callable[
    [DecomposedModel, ComponentStacks, optax.OptState],
    tuple[ComponentStacks, optax.OptState, Array],
]:
    """`model` is the jit ARG (frozen weights traced, not baked) — `weight_deltas` reads its
    per-site W slices, so closing over the model would bake them into the HLO."""

    def warmup_step(
        model: DecomposedModel, components: ComponentStacks, opt_state: optax.OptState
    ) -> tuple[ComponentStacks, optax.OptState, Array]:
        def loss_fn(components_: ComponentStacks) -> Array:
            return faithfulness_loss(model.weight_deltas(components_))

        loss, grad = eqx.filter_value_and_grad(loss_fn)(components)
        updates, opt_state = opt.update(grad, opt_state, eqx.filter(components, eqx.is_array))
        return eqx.apply_updates(components, updates), opt_state, loss

    return filter_jit(warmup_step, compiler_options=compiler_options)
