"""The generic single-pool VPD training step over a `DecomposedLM` (SPEC §4).

One `jax.jit` step: clean target → CI envelope → the adversary's supplemental ascents
(`adversary.py`) → the four losses (`losses.py`, recon plans in `recon.py`) → one fused
backward over (components, ci_fn, sources) → optimizer updates → for the PERSISTENT
adversary, the final (n_warmup+1)-th source ascent from the same graph (SPEC S13/S14).
All trainable state is fp32 masters (SPEC N1); forwards run in bf16 via explicit casts.

Schedules (imp-min p anneal, source-LR warmup) are computed inside the step from
`state.step`, so the jit signature is stable across the whole run (SPEC S9, S13).
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
    AdversaryConfig,
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
from jax_single_pool.recon import ReconPlan
from param_decomp_config.losses import (
    AdamPGDConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    PGDReconLossConfig,
    SCScope,
)

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
    sources: dict[str, Float[Array, "1 T Cp1"]]  # per-site raw sources, always in [0,1]
    sources_adam_state: SourcesAdamState
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
    faith_coeff: float,
    stoch_coeff: float,
    imp_min: ImportanceMinimalityLossConfig,
    adversary: AdversaryConfig,
    components_optimizer: optax.GradientTransformation,
    ci_fn_optimizer: optax.GradientTransformation,
    total_steps: int,
    recon_plan: ReconPlan,
    remat_recon_forwards: bool,
    mesh: Mesh | None,
):
    """Build the jit'd `step(state, frozen, residual, key) -> (state, metrics)`.

    `imp_min` / `adversary` are the SHARED torch loss configs; the asserts below pin
    the subset this trainer implements. `mesh` (when given) pins every batch-leading
    activation to `P('dp', ...)` so the masked re-forwards stay on per-device
    sub-batches (activation memory 1/n_dev)."""
    site_names = lm.site_names
    assert recon_plan, "empty recon plan"
    for recon_forward in recon_plan:
        assert recon_forward.live_sites and set(recon_forward.live_sites) <= set(site_names), (
            recon_forward
        )

    assert imp_min.coeff is not None and imp_min.p_anneal_final_p is not None
    imp_coeff = imp_min.coeff
    adversary_coeff = adversary.coeff
    assert adversary_coeff is not None
    match adversary:
        case PersistentPGDReconLossConfig():
            assert isinstance(adversary.scope, SCScope), adversary.scope
            assert not adversary.use_sigmoid_parameterization and adversary.start_frac == 0.0, (
                adversary
            )
            assert adversary.n_samples == 1, adversary
            source_adam = adversary.optimizer
            assert isinstance(source_adam, AdamPGDConfig), source_adam
            source_lr_schedule = source_adam.lr_schedule
            assert (
                source_lr_schedule.fn_type == "constant"
                and source_lr_schedule.final_val_frac == 1.0
            ), source_lr_schedule
        case PGDReconLossConfig():
            source_adam = None
            source_lr_schedule = None

    def batch_sharded(x: Array) -> Array:
        if mesh is None:
            return x
        spec = ["dp"] + [None] * (x.ndim - 1)
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(*spec)))

    def batch_sharded_ci(ci_values: CIValues) -> CIValues:
        """Reshard the CI-fn output to batch-sharded ONCE, here. The CI head's `out_w`
        is ΣC-sharded, so its output is born C-sharded; without a single producer-side
        pin, GSPMD inserts a separate C→batch reshard for every consumer (each chunk
        forward, the adversary, imp-min — forward and backward), and those all-to-all
        buffers dominate the temp arena at scale."""
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

    def adversarial_recon_loss(
        frozen: Any,
        components_bf16: Any,
        ci_lower: dict[str, Array],
        sources: dict[str, Array],
        residual: Array,
        clean_logits: Array,
        masked_forward_fn: Any,
    ) -> Array:
        """The source-masked recon KL (SPEC S12) — shared by BOTH adversaries; what
        differs between PPGD and fresh PGD is who owns `sources` and how they ascend."""
        masks, delta_masks = source_masks(ci_lower, sources, site_names)
        masked = masked_forward_fn(
            frozen, components_bf16, residual, masks, delta_masks, None, site_names
        )
        return kl_per_position(masked, clean_logits)

    def stochastic_recon_loss(
        frozen: Any,
        components_bf16: Any,
        ci_lower: dict[str, Array],
        residual: Array,
        clean_logits: Array,
        key: PRNGKeyArray,
    ) -> Array:
        batch, seq = residual.shape[0], residual.shape[1]
        total = jnp.zeros((), jnp.float32)
        n_forwards = 0
        for entry_idx, recon_forward in enumerate(recon_plan):
            entry_key, routing_key = random.split(random.fold_in(key, entry_idx))
            for draw_idx, routes in enumerate(
                recon_forward.sample_routing(routing_key, (batch, seq))
            ):
                mask_source_key, delta_mask_key = random.split(random.fold_in(entry_key, draw_idx))
                masks = {}
                delta_masks = {}
                for site_idx, site in enumerate(recon_forward.live_sites):
                    ci_site = ci_lower[site]
                    stochastic_source = random.uniform(
                        random.fold_in(mask_source_key, site_idx), ci_site.shape, COMPUTE_DT
                    )
                    masks[site] = ci_site + (1.0 - ci_site) * stochastic_source
                    delta_masks[site] = random.uniform(
                        random.fold_in(delta_mask_key, site_idx), (batch, seq), COMPUTE_DT
                    )
                masked = checkpointed_masked_forward(
                    frozen,
                    components_bf16,
                    residual,
                    masks,
                    delta_masks,
                    routes,
                    recon_forward.live_sites,
                )
                total = total + kl_per_position(masked, clean_logits)
                n_forwards += 1
        assert n_forwards > 0, "recon plan produced no forwards"
        return total / n_forwards

    @jax.jit
    def step(state: TrainState, frozen: Any, residual: Float[Array, "b t d"], key: PRNGKeyArray):
        step_f32 = state.step.astype(jnp.float32)
        pnorm = annealed_pnorm(step_f32, total_steps, imp_min)

        residual = batch_sharded(residual)
        clean_logits = jax.lax.stop_gradient(batch_sharded(lm.clean_logits(frozen, residual)))
        site_inputs = lm.site_inputs(frozen, residual)

        # ── supplemental adversary ascents: params + CI detached (SPEC §4.5) ──
        components_detached = jax.lax.stop_gradient(cast_floating(state.components, COMPUTE_DT))
        ci_fn_detached = jax.lax.stop_gradient(cast_floating(state.ci_fn, COMPUTE_DT))
        ci_lower_detached = batch_sharded_ci(ci_fn_detached(site_inputs)).lower

        def adversary_loss(sources: dict[str, Array]) -> Array:
            return adversarial_recon_loss(
                frozen,
                components_detached,
                ci_lower_detached,
                sources,
                residual,
                clean_logits,
                masked_forward,
            )

        match adversary:
            case PersistentPGDReconLossConfig():
                assert source_lr_schedule is not None and source_adam is not None
                adam = source_adam
                sources_lr = warmup_then_constant_lr(
                    step_f32,
                    total_steps,
                    source_lr_schedule.start_val,
                    source_lr_schedule.warmup_pct,
                )

                def warmup_body(
                    carry: tuple[dict[str, Array], SourcesAdamState], _: None
                ) -> tuple[tuple[dict[str, Array], SourcesAdamState], None]:
                    sources, adam_state = carry
                    sources_grad = jax.grad(adversary_loss)(sources)
                    sources, adam_state = sources_adam_ascend_project(
                        sources, sources_grad, adam_state, sources_lr, adam
                    )
                    return (sources, adam_state), None

                (refined_sources, sources_adam_state), _ = jax.lax.scan(
                    warmup_body,
                    (state.sources, state.sources_adam_state),
                    None,
                    length=adversary.n_warmup_steps,
                )
            case PGDReconLossConfig() as fresh_pgd_config:
                sources_lr = None
                sources_adam_state = state.sources_adam_state
                batch, seq = residual.shape[0], residual.shape[1]
                fresh_sources = init_fresh_pgd_sources(
                    lm.sites, adversary, batch, seq, random.fold_in(key, 2)
                )

                def sign_ascend_body(
                    sources: dict[str, Array], _: None
                ) -> tuple[dict[str, Array], None]:
                    sources_grad = jax.grad(adversary_loss)(sources)
                    return {
                        site: jnp.clip(
                            sources[site]
                            + fresh_pgd_config.step_size * jnp.sign(sources_grad[site]),
                            0.0,
                            1.0,
                        )
                        for site in sources
                    }, None

                refined_sources, _ = jax.lax.scan(
                    sign_ascend_body, fresh_sources, None, length=adversary.n_steps
                )
        refined_sources = jax.lax.stop_gradient(refined_sources)

        # ── main losses: live components/ci; the PERSISTENT adversary's sources
        # participate in the graph so their gradient comes from the SAME backward
        # (SPEC S14); they are NOT detached here, but components/ci grads through them
        # are what torch gets too (sources are leaves). ──
        def loss_fn(trainable: tuple[Any, CIFn, dict[str, Array]]):
            components, ci_fn, sources = trainable
            components_bf16 = cast_floating(components, COMPUTE_DT)
            ci_fn_bf16 = cast_floating(ci_fn, COMPUTE_DT)
            ci = batch_sharded_ci(ci_fn_bf16(site_inputs))
            faith_loss = faithfulness_loss(lm.weight_deltas(frozen, components))
            imp_lp, imp_entropy = importance_minimality_terms(ci.upper, pnorm, imp_min.eps)
            imp_loss = imp_lp + imp_min.beta * imp_entropy
            stoch_loss = stochastic_recon_loss(
                frozen, components_bf16, ci.lower, residual, clean_logits, random.fold_in(key, 1)
            )
            adv_loss = adversarial_recon_loss(
                frozen,
                components_bf16,
                ci.lower,
                sources,
                residual,
                clean_logits,
                checkpointed_masked_forward,
            )
            total_loss = (
                faith_coeff * faith_loss
                + imp_coeff * imp_loss
                + stoch_coeff * stoch_loss
                + adversary_coeff * adv_loss
            )
            return total_loss, (faith_loss, imp_loss, imp_lp, stoch_loss, adv_loss)

        (total_loss, (faith_loss, imp_loss, imp_lp, stoch_loss, adv_loss)), grads = (
            eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                (state.components, state.ci_fn, refined_sources)
            )
        )
        components_grad, ci_fn_grad, sources_grad_scaled = grads
        grad_norm_metrics = _grad_norm_metrics(components_grad, ci_fn_grad)

        match adversary:
            case PersistentPGDReconLossConfig():
                assert sources_lr is not None and source_adam is not None
                # The backward saw coeff·L_adv; the adversary ascends on L_adv itself.
                sources_grad = {s: g / adversary_coeff for s, g in sources_grad_scaled.items()}
                # ── the (n_warmup+1)-th source ascent, from the fused graph (SPEC S13/S14) ──
                new_sources, sources_adam_state = sources_adam_ascend_project(
                    refined_sources, sources_grad, sources_adam_state, sources_lr, source_adam
                )
            case PGDReconLossConfig():
                # fresh sources die with the step; the cotangent wrt them is unused
                new_sources = state.sources

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
            sources_adam_state=sources_adam_state,
            step=state.step + 1,
        )
        adversary_metric_key = (
            "ppgd" if isinstance(adversary, PersistentPGDReconLossConfig) else "pgd"
        )
        metrics = {
            "total": total_loss,
            "faith": faith_loss,
            "imp": imp_loss,
            "imp_no_beta": imp_lp,
            "stoch": stoch_loss,
            adversary_metric_key: adv_loss,
            "p_imp": pnorm,
            **grad_norm_metrics,
        }
        if sources_lr is not None:
            metrics["src_lr"] = sources_lr
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
