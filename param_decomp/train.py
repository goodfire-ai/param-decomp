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

from param_decomp.adversary import (
    PersistentAdversary,
    init_fresh_pgd_sources,
    source_masks,
)
from param_decomp.ci_fn import CI, CIFn
from param_decomp.components import DecompVU
from param_decomp.configs import SmoothL0ImportanceMinimalityLossConfig
from param_decomp.jit_util import filter_jit
from param_decomp.lm import DecomposedModel
from param_decomp.losses import (
    annealed_imp_min_param,
    faithfulness_loss,
    imp_min_terms,
    scheduled_value_traced,
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
from param_decomp.schedule import ScheduleConfig
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


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CIEnvelope:
    """A step's detached CI squashings (stop-gradient'd, compute-dtype, sharding-pinned).
    Under stale-CI replay (SPEC S34) the window-first step returns its envelope and the
    window's repeat steps consume it as a CONSTANT."""

    lower: dict[str, Array]
    upper: dict[str, Array]


@dataclass(frozen=True)
class StaleCITrainSteps:
    """The jitted step pair for stale-CI replay (SPEC S34), plus the mid-window resume
    rebuild.

    `fresh` is the full step (taps → CI-fn forward/vjp → ci_fn update), additionally
    returning its `CIEnvelope` for the window's repeats. `repeat` consumes that envelope
    as a constant — no taps, no CI-fn forward/backward; the ci_fn and its optimizer state
    pass through untouched. Its first arg bundles `(model, envelope)` so donation
    (`all-except-first`) spares both across the window's repeats.

    `compute_ci(model, ci_fn, batch) -> CIEnvelope` rebuilds the envelope when a
    requeue-resume lands mid-window. The ci_fn updates only on window-first steps, so the
    rebuilt envelope reflects the last fresh step's POST-update ci_fn — one ci_fn Adam
    step fresher than the envelope the interrupted window was using (plus vjp-primal vs
    plain-forward compilation differences). A tiny, preemption-only trajectory wobble."""

    fresh: Callable[
        [DecomposedModel, TrainState, Any, PRNGKeyArray],
        tuple[TrainState, dict[str, Array], CIEnvelope],
    ]
    repeat: Callable[
        [tuple[DecomposedModel, CIEnvelope], TrainState, Any, PRNGKeyArray],
        tuple[TrainState, dict[str, Array]],
    ]
    compute_ci: Callable[[DecomposedModel, Any, Any], CIEnvelope]


def _grad_norm_metrics(components_grad: DecompVU, ci_fn_grad: Any | None) -> dict[str, Array]:
    """Pre-clip gradient L2 norms, matching the torch `component_grad_norms` families:
    per-leaf `grad_norms/components<path>` / `grad_norms/ci_fns<path>` (paths are this
    pytree's own — e.g. `.vu['layers.18.mlp.gate_proj'][0]` for the per-site Llama
    layout, vs torch's per-site names) and the overlay-critical
    `grad_norms/summary/{components,ci_fns,total}`. `ci_fn_grad` is None on a stale-CI
    repeat step (no CI-fn backward): the `ci_fns` family is absent and `total` covers
    the components only."""
    out: dict[str, Array] = {}

    def family(grad_tree: Any, prefix: str) -> Array:
        sum_sq = jnp.zeros((), jnp.float32)
        for path, leaf in jax.tree_util.tree_flatten_with_path(grad_tree)[0]:
            leaf_sum_sq = jnp.sum(leaf.astype(jnp.float32) ** 2)
            out[f"grad_norms/{prefix}{jax.tree_util.keystr(path)}"] = jnp.sqrt(leaf_sum_sq)
            sum_sq = sum_sq + leaf_sum_sq
        out[f"grad_norms/summary/{prefix}"] = jnp.sqrt(sum_sq)
        return sum_sq

    total_sq = family(components_grad, "components")
    if ci_fn_grad is not None:
        total_sq = total_sq + family(ci_fn_grad, "ci_fns")
    out["grad_norms/summary/total"] = jnp.sqrt(total_sq)
    return out


# ───────────────────────────── the step factory ─────────────────────────────


def _build_step_impl(
    lm: DecomposedModel,
    *,
    losses: LossSurface,
    components_optimizer: optax.GradientTransformation,
    ci_fn_optimizer: optax.GradientTransformation,
    total_steps: int,
    remat_recon_forwards: bool,
    remat_ci_fn: bool,
    mesh: Mesh | None,
    ascend_replicate: bool,
    stale_ci_fn_lr: ScheduleConfig | None,
):
    """The un-jitted step machinery shared by `make_train_step` and
    `make_stale_ci_train_steps`: returns `(step_impl, compute_ci_impl)` where
    `step_impl(model, state, batch, key, cached)` runs the full step when `cached is None`
    (computing taps + the CI envelope, training the ci_fn) and the stale-CI repeat step
    when `cached` is a `CIEnvelope` (SPEC S34: CI a constant, ci_fn untouched). The
    `cached` switch is a Python-level trace decision — each variant jits to its own
    program, and the `cached is None` program is the pre-S34 step unchanged.

    `stale_ci_fn_lr` (stale-CI mode only) is the ci_fn LR schedule applied IN-STEP at the
    global step: the ci_fn optimizer must then carry unit LR (`unit_lr_ci_fn_optimizer`),
    since its optax update count advances only on window-first steps and a count-driven
    schedule would stretch by the replay factor."""
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

    @jaxtyped(typechecker=beartype)
    def step_impl(
        model: DecomposedModel,
        state: TrainState,
        batch: Any,
        key: PRNGKeyArray,
        cached: CIEnvelope | None,
    ) -> tuple[TrainState, dict[str, Array], CIEnvelope]:
        step_f32 = state.step.astype(jnp.float32)
        imp_min_param = annealed_imp_min_param(step_f32, total_steps, imp_min)

        batch = batch_sharded(batch)
        with jax.named_scope("pd_clean_fwd"):
            clean_output = jax.lax.stop_gradient(batch_sharded(model.clean_output(batch)))
        if cached is None:
            with jax.named_scope("pd_read_taps"):
                taps = model.read_activations(batch, state.ci_fn.input_names)
            # `leading` (batch, *positions) — the shape masks/sources/routes live in. Sourced
            # from a tap (always `[*leading, d_tap]`), not the opaque batch, so the engine never
            # assumes the batch's rank/feature dim.
            leading = next(iter(taps.values())).shape[:-1]
        else:
            taps = None
            leading = next(iter(cached.lower.values())).shape[:-1]

        # ── adversary ascents: params + CI detached (SPEC §4.5) ──
        prepared, recon_vjp = jax.vjp(
            lambda c: model.prepare_compute_weights(cast_floating(c, COMPUTE_DT)),
            state.components,
        )
        prepared_detached = jax.lax.stop_gradient(prepared)
        prepared_ascend = replicate_for_ascend(prepared_detached)
        # The CI envelope is a pure fn of the batch, so compute it ONCE per step — the value +
        # its vjp, mirroring `prepared`/`recon_vjp`. The ascend uses the stop_gradient'd value;
        # `loss_fn` takes the live value and its ci-fn grad is pulled back through `ci_vjp`. So the
        # (≈10x-the-target) CI fn is forward-evaluated ONCE, not once detached for the ascend +
        # once inside the main backward. On a stale-CI repeat step it is not evaluated at all:
        # the window-first envelope is the constant CI everywhere.
        if cached is None:
            assert taps is not None
            with jax.named_scope("pd_ci_fn_fwd"):
                ci, ci_vjp = eqx.filter_vjp(
                    lambda cf: ci_batch_sharded(
                        cast_floating(cf, COMPUTE_DT)(taps, remat=remat_ci_fn)
                    ),
                    state.ci_fn,
                )
            ci_detached = jax.lax.stop_gradient(ci)
            envelope = CIEnvelope(lower=ci_detached.lower, upper=ci_detached.upper)
        else:
            ci, ci_vjp = None, None
            envelope = cached
        ci_lower_detached = envelope.lower

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
                        prepared_ascend,
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

        # ── main losses: live components (+ live ci when fresh — on a stale-CI repeat the
        # envelope enters as a CONSTANT); the PERSISTENT sources participate in
        # the graph so their gradient comes from the SAME backward (SPEC S14'); they
        # are NOT detached here, but components/ci grads through them are what torch
        # gets too (sources are leaves). ──
        def losses_body(
            prepared: Any,
            components: DecompVU,
            ci_lower: dict[str, Array],
            ci_upper: dict[str, Array],
            persistent_sources: dict[str, dict[str, Array]],
        ) -> tuple[Array, tuple[Array, Array, Array, tuple[Array, ...]]]:
            # Stochastic recon builds its masks INSIDE the target's `masked_output_stochastic`
            # from this once-per-step shared CI form — a scan target recomputes them in its
            # checkpointed block (mask never held, the memory win); others fall back to building
            # masks then `masked_output`. Either way the engine holds no per-forward mask stacks.
            ci_stacked = model.stack_ci(ci_lower)
            faith_loss = faithfulness_loss(model.weight_deltas(components))
            imp_lp, imp_freq = imp_min_terms(ci_upper, imp_min, imp_min_param)

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

                        def pre_built_fwd(
                            mds: tuple[dict[str, Array], dict[str, Array]],
                            routes: Routes = routes,
                            entry: ReconForward = entry,
                        ) -> Any:
                            return masked_forward(
                                model,
                                prepared,
                                batch,
                                mds[0],
                                mds[1],
                                routes,
                                entry.live_sites,
                                entry.has_delta,
                            )

                        with jax.named_scope("pd_recon_masked_fwd"):
                            match entry.sources:
                                case StochasticSources():  # masks built inside the target
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
                                    masked = pre_built_fwd(
                                        constant_entry_masks(strategy, ci_lower, entry.live_sites)
                                    )
                                case FreshPGDSources():
                                    masked = pre_built_fwd(
                                        source_masks(
                                            ci_lower,
                                            fresh_sources[(term_idx, entry_idx)],
                                            entry.live_sites,
                                        )
                                    )
                                case PersistentSources(state_key=state_key):
                                    masked = pre_built_fwd(
                                        source_masks(
                                            ci_lower,
                                            persistent_sources[state_key],
                                            entry.live_sites,
                                        )
                                    )
                        total = total + recon_loss_fn(masked, clean_output)
                        n_forwards += 1
                assert n_forwards > 0, f"term {term.name!r} produced no forwards"
                term_loss = total / n_forwards
                term_losses.append(term_loss)

            total_loss = faith_coeff * faith_loss + imp_coeff * imp_lp + freq_coeff * imp_freq
            for term, term_loss in zip(recon_terms, term_losses, strict=True):
                total_loss = total_loss + term.coeff * term_loss
            return total_loss, (faith_loss, imp_lp, imp_freq, tuple(term_losses))

        warmed_sources = {k: a.sources for k, a in warmed_advs.items()}
        if cached is None:
            assert ci is not None and ci_vjp is not None

            def loss_fn(
                trainable: tuple[Any, DecompVU, CI, dict[str, dict[str, Array]]],
            ) -> tuple[Array, tuple[Array, Array, Array, tuple[Array, ...]]]:
                prepared_t, components_t, ci_t, persistent_t = trainable
                return losses_body(prepared_t, components_t, ci_t.lower, ci_t.upper, persistent_t)

            with jax.named_scope("pd_value_and_grad"):
                (total_loss, (faith_loss, imp_lp, imp_freq, term_losses)), grads = (
                    eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                        (prepared, state.components, ci, warmed_sources)
                    )
                )
            prepared_grad, components_grad_faith, ci_grad, persistent_grads_scaled = grads
            ci_fn_grad = ci_vjp(ci_grad)[0]
        else:

            def loss_fn_repeat(
                trainable: tuple[Any, DecompVU, dict[str, dict[str, Array]]],
            ) -> tuple[Array, tuple[Array, Array, Array, tuple[Array, ...]]]:
                prepared_t, components_t, persistent_t = trainable
                return losses_body(
                    prepared_t, components_t, envelope.lower, envelope.upper, persistent_t
                )

            with jax.named_scope("pd_value_and_grad"):
                (total_loss, (faith_loss, imp_lp, imp_freq, term_losses)), grads = (
                    eqx.filter_value_and_grad(loss_fn_repeat, has_aux=True)(
                        (prepared, state.components, warmed_sources)
                    )
                )
            prepared_grad, components_grad_faith, persistent_grads_scaled = grads
            ci_fn_grad = None
        components_grad_recon = recon_vjp(prepared_grad)[0]
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
            state.components_opt_state,
            eqx.filter(state.components, eqx.is_array),
        )
        if ci_fn_grad is not None:
            ci_fn_updates, new_ci_fn_opt_state = ci_fn_optimizer.update(
                ci_fn_grad, state.ci_fn_opt_state, eqx.filter(state.ci_fn, eqx.is_array)
            )
            if stale_ci_fn_lr is not None:
                # Unit-lr optimizer (see factory docstring): apply the run's ci_fn LR
                # schedule here, at the GLOBAL step — matching what optax's count-driven
                # schedule applies when every step updates the ci_fn.
                ci_fn_lr = scheduled_value_traced(step_f32, total_steps, stale_ci_fn_lr)
                ci_fn_updates = jax.tree.map(lambda u: ci_fn_lr * u, ci_fn_updates)
        else:
            ci_fn_updates, new_ci_fn_opt_state = None, state.ci_fn_opt_state
        new_components = eqx.apply_updates(state.components, components_updates)
        new_ci_fn = (
            eqx.apply_updates(state.ci_fn, ci_fn_updates)
            if ci_fn_updates is not None
            else state.ci_fn
        )

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
        return new_state, metrics, envelope

    def compute_ci_impl(model: DecomposedModel, ci_fn: CIFn, batch: Any) -> CIEnvelope:
        batch = batch_sharded(batch)
        taps = model.read_activations(batch, ci_fn.input_names)
        ci = jax.lax.stop_gradient(
            ci_batch_sharded(cast_floating(ci_fn, COMPUTE_DT)(taps, remat=remat_ci_fn))
        )
        return CIEnvelope(lower=ci.lower, upper=ci.upper)

    return step_impl, compute_ci_impl


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
    ascend_replicate: bool = False,
    compiler_options: dict[str, bool | int | str] | None = None,
):
    """Build the `eqx.filter_jit`'d `step(model, state, batch, key) -> (state, metrics)`.

    `model` is the jit ARG (frozen 8B weights traced as array leaves, never baked); the
    factory closes over only static config (`site_names`, `recon_loss_fn`, term wiring) read
    off `lm` here. `losses` (from `build_loss_terms`) is the `LossSurface` record — the
    faithfulness + importance-minimality singletons and the recon Σ, read by name. `mesh`
    (when given) pins every batch-leading activation over the full mesh
    (`P(('replicate', 'fsdp'), ...)`) so the masked re-forwards stay on per-rank sub-batches
    (activation memory 1/N)."""
    step_impl, _ = _build_step_impl(
        lm,
        losses=losses,
        components_optimizer=components_optimizer,
        ci_fn_optimizer=ci_fn_optimizer,
        total_steps=total_steps,
        remat_recon_forwards=remat_recon_forwards,
        remat_ci_fn=remat_ci_fn,
        mesh=mesh,
        ascend_replicate=ascend_replicate,
        stale_ci_fn_lr=None,
    )

    def step(
        model: DecomposedModel, state: TrainState, batch: Any, key: PRNGKeyArray
    ) -> tuple[TrainState, dict[str, Array]]:
        new_state, metrics, _ = step_impl(model, state, batch, key, None)
        return new_state, metrics

    return filter_jit(step, donate="all-except-first", compiler_options=compiler_options)


def make_stale_ci_train_steps(
    lm: DecomposedModel,
    *,
    losses: LossSurface,
    components_optimizer: optax.GradientTransformation,
    ci_fn_optimizer: optax.GradientTransformation,
    ci_fn_lr_schedule: ScheduleConfig,
    total_steps: int,
    remat_recon_forwards: bool,
    remat_ci_fn: bool,
    mesh: Mesh | None,
    ascend_replicate: bool = False,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> StaleCITrainSteps:
    """Build the stale-CI replay step pair (SPEC S34; `DataConfig.replay_stale_ci`).

    `ci_fn_optimizer` MUST carry unit LR (`run_state.unit_lr_ci_fn_optimizer`): the fresh
    step applies `ci_fn_lr_schedule` itself at the GLOBAL step, so the ci_fn follows the
    run's LR schedule even though its optax update count only advances on window-first
    steps (a count-driven schedule would stretch by the replay factor)."""
    step_impl, compute_ci_impl = _build_step_impl(
        lm,
        losses=losses,
        components_optimizer=components_optimizer,
        ci_fn_optimizer=ci_fn_optimizer,
        total_steps=total_steps,
        remat_recon_forwards=remat_recon_forwards,
        remat_ci_fn=remat_ci_fn,
        mesh=mesh,
        ascend_replicate=ascend_replicate,
        stale_ci_fn_lr=ci_fn_lr_schedule,
    )

    def fresh(
        model: DecomposedModel, state: TrainState, batch: Any, key: PRNGKeyArray
    ) -> tuple[TrainState, dict[str, Array], CIEnvelope]:
        return step_impl(model, state, batch, key, None)

    def repeat(
        model_and_envelope: tuple[DecomposedModel, CIEnvelope],
        state: TrainState,
        batch: Any,
        key: PRNGKeyArray,
    ) -> tuple[TrainState, dict[str, Array]]:
        model, cached_envelope = model_and_envelope
        new_state, metrics, _ = step_impl(model, state, batch, key, cached_envelope)
        return new_state, metrics

    return StaleCITrainSteps(
        fresh=filter_jit(fresh, donate="all-except-first", compiler_options=compiler_options),
        repeat=filter_jit(repeat, donate="all-except-first", compiler_options=compiler_options),
        compute_ci=filter_jit(compute_ci_impl, compiler_options=compiler_options),
    )


# ───────────────────────────── faithfulness warmup (SPEC S21) ─────────────────────────────


def make_faith_warmup_step(
    opt: optax.GradientTransformation,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> Callable[[DecomposedModel, DecompVU, optax.OptState], tuple[DecompVU, optax.OptState, Array]]:
    """`model` is the jit ARG (frozen weights traced, not baked) — `weight_deltas` reads its
    per-site W slices, so closing over the model would bake them into the HLO."""

    def warmup_step(
        model: DecomposedModel, components: DecompVU, opt_state: optax.OptState
    ) -> tuple[DecompVU, optax.OptState, Array]:
        def loss_fn(components_: DecompVU) -> Array:
            return faithfulness_loss(model.weight_deltas(components_))

        loss, grad = eqx.filter_value_and_grad(loss_fn)(components)
        updates, opt_state = opt.update(grad, opt_state, eqx.filter(components, eqx.is_array))
        return eqx.apply_updates(components, updates), opt_state, loss

    return filter_jit(warmup_step, compiler_options=compiler_options)
