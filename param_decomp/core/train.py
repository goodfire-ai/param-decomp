"""The generic single-pool VPD training step over a `DecomposedModel` (SPEC §4).

One `jax.jit` step: clean target forward → CI envelope → per-persistent-term supplemental
ascents + per-fresh-entry sign-PGD ascents (`adversary.py`) → faith + imp-min +
the recon loss TERMS (`recon.py`; each term = plan × mask-source strategy, SPEC
S10') → one fused backward over (components, ci_fn, all persistent sources) →
optimizer updates → each persistent term's final ascent from the same graph
(SPEC S13'/S14'/S23: the source path is never coeff-scaled — a persistent term's
coeff rides its model-side cotangents via `model_cotangents_scaled` — so the
backward hands each adversary `dL/ds` directly). All trainable state is fp32
masters (SPEC N1); forwards run in bf16 via explicit casts.

Schedules (imp-min p anneal, source-LR warmup, every scheduled loss coefficient) are
computed inside the step from `state.step`, so the jit signature is stable across the
whole run (SPEC S9, S13); each coefficient resolves ONCE at the top of the step and only
values flow into the loss math.
Per-term RNG: term i draws from `fold_in(step_key, offset + i)` in config-list order
(SPEC R1) — offset 1 for the main grid reproduces the pre-unification production key
derivation exactly.

The step machinery lives on `_StepAtoms` — the vocabulary a step factory composes its
body from — so a second factory (a tPD two-pass step) shares the machinery without a
mirrored body.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from beartype import beartype
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from param_decomp.core.adversary import PersistentAdversary, init_fresh_pgd_sources
from param_decomp.core.ci_fn import CI, CIFn, evaluate_ci
from param_decomp.core.components import ComponentStacks, VUShape
from param_decomp.core.configs import LossCoeff, SmoothL0ImportanceMinimalityLossConfig
from param_decomp.core.jit_util import filter_jit
from param_decomp.core.losses import (
    ReconstructionLoss,
    annealed_imp_min_param,
    coeff_at,
    faithfulness_loss,
    imp_min_terms,
    mean_reconstruction_losses,
    reconstruction_loss,
    reconstruction_loss_metrics,
    reconstruction_spec_at,
    scheduled_value_traced,
)
from param_decomp.core.masking import (
    constant_delta_pinned_masks,
    masks_from_sources,
    mixed_persistent_stochastic_masks,
    stochastic_delta_pinned_masks,
    unmasked_no_delta_masks,
)
from param_decomp.core.model import (
    CaptureKeys,
    DecomposedModel,
    Masking,
    MaterializedMasking,
    StochasticMasking,
    prepare_compute_weights,
    select_captures,
)
from param_decomp.core.objective import ImportanceMinimalityTerm, LossSurface, TargetedObjective
from param_decomp.core.recon import (
    AnyReconLossTerm,
    ConstantSources,
    ForwardObservations,
    FreshPGDSources,
    MaskSourceStrategy,
    MixedPersistentStochasticSources,
    PersistentSources,
    ReconForward,
    ReconLossTerm,
    ReconstructionSpec,
    Routes,
    StochasticSources,
    UnmaskedNoDeltaSources,
    reconstruction_observations,
    resolve_reconstruction_terms,
)
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.sharding import batch_shard_leading


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


@eqx.filter_jit
def uv_norm_ratio_metrics(components: ComponentStacks) -> dict[str, Array]:
    """Return each site's Frobenius-norm ratio ``||U|| / ||V||`` and summaries."""

    def per_slice_sq(stack: Float[Array, "g a b"]) -> Float[Array, " g"]:
        return jnp.sum(stack.astype(jnp.float32) ** 2, axis=(1, 2))

    factor_sq = {
        shape: (per_slice_sq(Vs), per_slice_sq(Us)) for shape, (Vs, Us) in components.stacks.items()
    }
    metrics: dict[str, Array] = {}
    ratios = []
    for name, shape, slot in components.site_slots:
        v_sq, u_sq = factor_sq[shape]
        ratio = jnp.sqrt(u_sq[slot] / v_sq[slot])
        metrics[f"uv_norm_ratio['{name}']"] = ratio
        ratios.append(ratio)

    stacked = jnp.stack(ratios)
    metrics["uv_norm_ratio_mean"] = jnp.mean(stacked)
    metrics["uv_norm_ratio_max"] = jnp.max(stacked)
    return metrics


def _scheduled_coeff_metrics(
    step_f32: Array, total_steps: int, coeffs: dict[str, LossCoeff]
) -> dict[str, Array]:
    """Per-step values of the SCHEDULED coefficients only — a constant would be log
    noise, and a moving coefficient invisible in wandb is a debugging trap."""
    return {
        f"schedules/coeff/{name}": scheduled_value_traced(step_f32, total_steps, coeff)
        for name, coeff in coeffs.items()
        if isinstance(coeff, ScheduleConfig)
    }


@jax.custom_vjp
def _cotangent_scaled(x: Array, by: Array) -> Array:
    del by  # forward-inert: consumed only by the vjp
    return x


def _cotangent_scaled_fwd(x: Array, by: Array) -> tuple[Array, Array]:
    return x, by


def _cotangent_scaled_bwd(by: Array, g: Array) -> tuple[Array, Array]:
    return g * by.astype(g.dtype), jnp.zeros_like(by)


_cotangent_scaled.defvjp(_cotangent_scaled_fwd, _cotangent_scaled_bwd)


def model_cotangents_scaled[T](tree: T, by: Array | float) -> T:
    """`tree`, bit-identical in the forward, with every backward cotangent scaled `by`
    the term's per-step coeff. This is WHERE a persistent term's coeff applies (SPEC
    S14'): the term's loss enters the differentiated total UNSCALED so the source path
    carries `dL/ds` directly, and the model-side inputs (prepared weights, CI envelope)
    are wrapped here so the components/CI fn still receive `coeff·dL/dθ` — an exact 0
    while an activation gate holds the coeff at 0, with no division anywhere."""
    by_arr = jnp.asarray(by, jnp.float32)
    return jax.tree.map(lambda leaf: _cotangent_scaled(leaf, by_arr), tree)


type CoeffApplication = Literal["scales_loss", "scales_model_cotangents"]


def coeff_application(term: AnyReconLossTerm) -> CoeffApplication:
    """WHERE this term's coeff applies — static structure, decided at trace time.

    A term that trains its sources FROM the shared backward (a persistent bundle in its
    plan) must keep the source path unscaled so the backward hands the adversary `dL/ds`
    (SPEC S14'): its coeff rides the model-side cotangents (`model_cotangents_scaled`)
    and the term enters the differentiated total at weight 1. Every other term's coeff
    scales its loss scalar in the total."""
    trains_sources_from_backward = any(
        isinstance(entry.sources, PersistentSources | MixedPersistentStochasticSources)
        for entry in term.plan
    )
    return "scales_model_cotangents" if trains_sources_from_backward else "scales_loss"


# ───────────────────────────── the step vocabulary ─────────────────────────────


@dataclass(frozen=True)
class StreamInputs:
    """One data stream's per-step trace inputs: the sharded opaque batch, the detached
    clean-forward observations it is scored against, the CI-fn input activations, and
    the waist `leading` shape masks/sources/routes live in."""

    batch: Any
    clean: ForwardObservations
    taps: dict[str, Array]
    leading: tuple[int, ...]


@dataclass(frozen=True)
class AscendedAdversaries:
    """The ascent phase's outputs: warmed persistent adversaries (SPEC S24), each
    fresh-PGD entry's ascended sources, and the per-entry routing draws those ascents
    fixed for the main grid to reuse (SPEC S24, torch parity)."""

    warmed: dict[str, PersistentAdversary]
    fresh_sources: dict[tuple[int, int], dict[str, Array]]
    fixed_routes: dict[tuple[int, int], tuple[Routes, ...]]


type DrawLoss[S: MaskSourceStrategy] = Callable[
    [int, int, ReconLossTerm[S], ReconForward[S], PRNGKeyArray, Routes], ReconstructionLoss
]
"""`(term_idx, entry_idx, term, entry, draw_key, routes) -> the draw's scored recon` —
one grid's per-draw dispatcher, built by the factory that owns the grid's trainables.
`S` is the grid's source-strategy width: the non-target grid's dispatcher takes only the
delta-pinnable strategies, so its match is exhaustive over two arms (SPEC T5)."""

type TermDraws[S: MaskSourceStrategy] = list[tuple[int, ReconForward[S], PRNGKeyArray, Routes]]
"""One term's flat `(entry_idx, entry, draw_key, routes)` forwards."""


def constant_source_masks(
    strategy: ConstantSources, ci_lower: dict[str, Array], live_sites: tuple[str, ...]
) -> dict[str, Array]:
    """Build constant component masks; no weight-delta path exists for this source."""
    return {site: ci_lower[site] + (1.0 - ci_lower[site]) * strategy.value for site in live_sites}


class _StepAtoms[PreparedT]:
    """The step vocabulary a step factory composes its body from — the run's statics as
    fields, the shared machinery (sharding pins, stream prep, the component/CI vjp
    scaffolding, adversary ascents, the recon-grid walk, the optimizer tail) as methods.
    Instantiated once per factory at trace-setup time.

    Holds NO arrays: `model_static` supplies static config in `__init__` and is NOT
    stored — every method takes the array-bearing model as an explicit argument (the
    HLO-baking rule)."""

    def __init__(
        self,
        model_static: DecomposedModel[PreparedT],
        *,
        recon_terms: tuple[AnyReconLossTerm, ...],
        imp: ImportanceMinimalityTerm,
        components_optimizer: optax.GradientTransformation,
        ci_fn_optimizer: optax.GradientTransformation,
        total_steps: int,
        remat_recon_forwards: bool,
        remat_ci_fn: bool,
        ci_capture_keys: CaptureKeys,
        mesh: Mesh | None,
        ascend_replicate: bool,
    ) -> None:
        self.components_optimizer = components_optimizer
        self.ci_fn_optimizer = ci_fn_optimizer
        self.total_steps = total_steps
        self.remat_recon_forwards = remat_recon_forwards
        self.remat_ci_fn = remat_ci_fn
        self.ci_capture_keys = ci_capture_keys
        self.mesh = mesh
        self.ascend_replicate = ascend_replicate

        self.site_names = model_static.site_names
        self.sites = model_static.sites
        self.c_by_site = {spec.name: spec.C for spec in self.sites}
        self.recon_loss_fn = (
            model_static.recon_loss_fn
        )  # static: pure, holds no arrays — safe to close
        resolved_recon = resolve_reconstruction_terms(model_static, recon_terms)
        self.recon_terms = resolved_recon.terms
        self.hidden_acts_capture_keys_by_term = resolved_recon.hidden_acts_capture_keys_by_term
        self.hidden_acts_capture_keys = resolved_recon.hidden_acts_capture_keys
        self.persistent_term_by_key = resolved_recon.persistent_term_by_key

        self.imp_min = imp.cfg
        self.imp_coeff: LossCoeff = imp.coeff
        self.freq_coeff: LossCoeff = (
            self.imp_min.frequency.coeff if self.imp_min.frequency is not None else 0.0
        )
        # Log the imp-min loss + its annealed param under penalty-kind-specific keys: the param
        # is `p` for L_p / `gamma` for smooth-L0, and the loss carries the penalty's class name.
        is_smooth_l0 = isinstance(self.imp_min, SmoothL0ImportanceMinimalityLossConfig)
        self.imp_loss_key = "imp_smooth_l0" if is_smooth_l0 else "imp"
        self.imp_min_param_key = "gamma_imp" if is_smooth_l0 else "p_imp"

    def shard_batch_tree[T](self, x: T) -> T:
        """Pin the leading (batch) axis of every array in the pytree. The batch and the
        model output are opaque protocol edges (`Any` — tokens for an LM, a dict or tuple
        for another target), so this maps over leaves rather than assuming one array."""
        return jax.tree.map(lambda leaf: batch_shard_leading(leaf, self.mesh), x)

    def _shard_ci_array(self, x: Array) -> Array:
        """Pin a CI / mask tensor `[batch, *positions, C]` batch over the full mesh, C
        REPLICATED. No-op off-mesh (single device / toys)."""
        if self.mesh is None:
            return x
        spec = (("replicate", "fsdp"), *((None,) * (x.ndim - 1)))
        return jax.lax.with_sharding_constraint(x, NamedSharding(self.mesh, P(*spec)))

    def shard_ci(self, ci: CI) -> CI:
        """Pin the CI-fn output batch over the full mesh, C REPLICATED — the layout `site_out`
        pins `x@V` to (SPEC §4.1), so the downstream mask multiply `xV * mask` needs no
        reshard. The explicit constraint stops GSPMD re-deciding it in the backward (same
        rationale as `site_out`'s activation pin, bf072ef01). `preactivations` is passed through
        (unused in the step — only the squashings are; DCE drops it)."""
        return CI(
            preactivations=ci.preactivations,
            lower={site: self._shard_ci_array(v) for site, v in ci.lower.items()},
            upper={site: self._shard_ci_array(v) for site, v in ci.upper.items()},
        )

    def replicate_for_ascend(self, prepared_weights: PreparedT) -> PreparedT:
        """Lever #5 (`runtime.ascend_replicate`): gather the ÷fsdp compute weights to
        FULL/replicated ONCE before the adversary ascents, so the `n_warmup` ascend forwards run
        plain matmuls with NO per-layer ÷fsdp→full NVLink gather. The gather is
        mask-INDEPENDENT and the V/U are detached (constant) across ascend steps, so the
        re-gather is pure redundancy — `n_warmup × n_layer × (fwd+bwd)` collectives collapse to
        one full gather. Trades the full V/U resident (≈ `fsdp`× the ÷fsdp stack) during the
        ascend phase for the eliminated re-gathers. Pure data movement (bf16 values unchanged) →
        numerics bit-identical. No-op off-flag / off-mesh."""
        if (
            not self.ascend_replicate
            or self.mesh is None
            or (jax.sharding.get_abstract_mesh().empty)
        ):
            return prepared_weights
        replicated = NamedSharding(self.mesh, P())
        return jax.tree.map(
            lambda a: jax.lax.with_sharding_constraint(a, replicated), prepared_weights
        )

    def prep_stream(
        self, model: DecomposedModel[PreparedT], batch: Any, hidden_acts_keys: CaptureKeys
    ) -> StreamInputs:
        """Shard one stream's batch, run its detached clean forward, and pull the CI taps +
        recon observations. `hidden_acts_keys` is the stream's own union — a stream whose
        grid carries no hidden-acts reconstruction captures none."""
        batch = self.shard_batch_tree(batch)
        with jax.named_scope("pd_clean_fwd_and_taps"):
            clean_forward_result = jax.tree.map(
                jax.lax.stop_gradient,
                model.clean_forward(batch, self.ci_capture_keys | hidden_acts_keys),
            )
            taps = select_captures(clean_forward_result.captures, self.ci_capture_keys)
            clean = reconstruction_observations(
                clean_forward_result,
                hidden_acts_capture_keys=hidden_acts_keys,
                mesh=self.mesh,
            )
        # `leading` (batch, *positions) — the shape masks/sources/routes live in. Sourced
        # from a tap (always `[*leading, d_tap]`), not the opaque batch, so the engine never
        # assumes the batch's rank/feature dim.
        leading = next(iter(taps.values())).shape[:-1]
        return StreamInputs(batch=batch, clean=clean, taps=taps, leading=leading)

    def component_weights_vjp(
        self, model: DecomposedModel[PreparedT], components: ComponentStacks
    ) -> tuple[PreparedT, Callable[[PreparedT], tuple[ComponentStacks]]]:
        """The compute-weights value + vjp — the recon gradient's pullback onto V/U."""
        return jax.vjp(lambda c: prepare_compute_weights(model, c), components)

    def ci_forward_vjp(
        self, ci_fn: CIFn, taps: dict[str, Array]
    ) -> tuple[CI, Callable[[CI], tuple[Any]]]:
        """The CI envelope's value + vjp. The CI envelope is a pure fn of the taps, so it is
        forward-evaluated ONCE per stream — the ascents use the stop_gradient'd value; the
        loss takes the live value and its ci-fn grad is pulled back through the vjp."""
        with jax.named_scope("pd_ci_fn_fwd"):
            return eqx.filter_vjp(
                lambda cf: self.shard_ci(evaluate_ci(cf, taps, remat=self.remat_ci_fn)),
                ci_fn,
            )

    # ONE masked-forward re-forward for recon AND the adversary ascents, sharing the same remat
    # policy. `remat_recon_forwards` gates gradient-checkpointing inside the target's
    # `masked_forward` at the target's natural granularity (a deep target recomputes one layer
    # at a time in the backward instead of storing every layer's activations). This is
    # load-bearing for the ASCENTS too: though they backprop only to the SOURCES (params + CI
    # detached), the source gradient still flows through the per-layer activations (the masks
    # MULTIPLY them), so an un-rematted ascent forward stacks `[n_layer, *leading, d_ff]` MLP
    # intermediates. Remat off stores all activations: faster when memory allows.
    @jaxtyped(typechecker=beartype)
    def masked_recon(
        self,
        model: DecomposedModel[PreparedT],
        *,
        prepared_weights: PreparedT,
        batch: Any,
        masking: Masking,
        capture_keys: CaptureKeys,
        reconstruction: ReconstructionSpec,
        clean: ForwardObservations,
    ) -> ReconstructionLoss:
        """Run one masked forward and score its complete recon objective — output and
        hidden-acts points alike."""
        masked_forward_result = model.masked_forward(
            prepared_weights,
            batch,
            masking=masking,
            capture_keys=capture_keys,
            remat=self.remat_recon_forwards,
        )
        masked = reconstruction_observations(
            masked_forward_result,
            hidden_acts_capture_keys=capture_keys,
            mesh=self.mesh,
        )
        return reconstruction_loss(
            self.recon_loss_fn,
            masked=masked,
            clean=clean,
            reconstruction=reconstruction,
        )

    def reconstruction_specs_at[S: MaskSourceStrategy](
        self, terms: tuple[ReconLossTerm[S], ...], step_f32: Array
    ) -> dict[str, ReconstructionSpec]:
        """Each term's value-level reconstruction spec at this step: the S35 rider's
        possibly-scheduled coeff resolved ONCE, high in the step (the pnorm pattern), so
        schedule objects never enter the draw dispatchers."""
        return {
            term.name: reconstruction_spec_at(
                term.hidden_acts_reconstruction, step_f32, self.total_steps
            )
            for term in terms
        }

    def recon_for_sources(
        self,
        *,
        term: AnyReconLossTerm,
        entry: ReconForward[MaskSourceStrategy],
        sources: dict[str, Array],
        routes_per_draw: tuple[Routes, ...],
        model: DecomposedModel[PreparedT],
        prepared_weights: PreparedT,
        ci_lower: dict[str, Array],
        stream: StreamInputs,
        reconstruction: ReconstructionSpec,
    ) -> Array:
        """Mean of one adversarial entry's fixed-source objective across its draws."""
        masks, delta_masks = masks_from_sources(ci_lower, sources, entry.live_sites)
        total = jnp.zeros((), jnp.float32)
        for routes in routes_per_draw:
            breakdown = self.masked_recon(
                model,
                prepared_weights=prepared_weights,
                batch=stream.batch,
                masking=MaterializedMasking(
                    component_masks=masks, weight_delta_masks=delta_masks, routes=routes
                ),
                capture_keys=self.hidden_acts_capture_keys_by_term[term.name],
                reconstruction=reconstruction,
                clean=stream.clean,
            )
            total = total + breakdown.total
        return total / len(routes_per_draw)

    def ascend_adversaries(
        self,
        model: DecomposedModel[PreparedT],
        stream: StreamInputs,
        ascend_prepared_weights: PreparedT,
        ci_lower_detached: dict[str, Array],
        adversaries: dict[str, PersistentAdversary],
        key: PRNGKeyArray,
        step_f32: Array,
        reconstruction_specs: dict[str, ReconstructionSpec],
    ) -> AscendedAdversaries:
        """The step's whole ascent phase, params + CI detached (SPEC §4.5).

        Persistent adversaries each run their supplemental ascents vs the route-ALL
        all-sites forward (SPEC S24 — torch warmup parity, NOT the term's loss plan); the
        warmed sources then enter the main backward as leaves; the LR schedule (S13′)
        lives in `PersistentAdversary`. Fresh-PGD entries draw routing ONCE per step,
        shared by all ascents and the main loss forward (SPEC S24); sign-ascend `n_steps`,
        then the sources are constants in the main backward (torch parity)."""

        def warmup_scoring_loss(term: AnyReconLossTerm) -> Callable[[dict[str, Array]], Array]:
            def objective(sources: dict[str, Array]) -> Array:
                masks, delta_masks = masks_from_sources(ci_lower_detached, sources, self.site_names)
                return self.masked_recon(
                    model,
                    prepared_weights=ascend_prepared_weights,
                    batch=stream.batch,
                    masking=MaterializedMasking(
                        component_masks=masks, weight_delta_masks=delta_masks, routes=None
                    ),
                    capture_keys=self.hidden_acts_capture_keys_by_term[term.name],
                    reconstruction=reconstruction_specs[term.name],
                    clean=stream.clean,
                ).total

            return objective

        with jax.named_scope("pd_pgd_warmup_ascend"):
            warmed = {
                state_key: adv.warmup_ascend(
                    warmup_scoring_loss(self.persistent_term_by_key[state_key]),
                    step_f32,
                    self.total_steps,
                )
                for state_key, adv in adversaries.items()
            }

        fresh_sources: dict[tuple[int, int], dict[str, Array]] = {}
        fixed_routes: dict[tuple[int, int], tuple[Routes, ...]] = {}
        for term_idx, term in enumerate(self.recon_terms):
            term_key = random.fold_in(key, 1 + term_idx)
            for entry_idx, entry in enumerate(term.plan):
                if not isinstance(entry.sources, FreshPGDSources):
                    continue
                fresh_cfg = entry.sources
                routing_key, init_key = random.split(random.fold_in(term_key, entry_idx))
                routes_per_draw = entry.sample_routing(routing_key, stream.leading)
                fixed_routes[(term_idx, entry_idx)] = routes_per_draw
                live_specs = tuple(s for s in self.sites if s.name in entry.live_sites)
                init = init_fresh_pgd_sources(
                    sites=live_specs,
                    init=fresh_cfg.init,
                    source_shape=fresh_cfg.source_shape,
                    leading=stream.leading,
                    key=init_key,
                )

                def ascent_loss(
                    sources: dict[str, Array],
                    term: AnyReconLossTerm = term,
                    entry: ReconForward[MaskSourceStrategy] = entry,
                    routes: tuple[Routes, ...] = routes_per_draw,
                ) -> Array:
                    return self.recon_for_sources(
                        term=term,
                        entry=entry,
                        sources=sources,
                        routes_per_draw=routes,
                        model=model,
                        prepared_weights=ascend_prepared_weights,
                        ci_lower=ci_lower_detached,
                        stream=stream,
                        reconstruction=reconstruction_specs[term.name],
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

        return AscendedAdversaries(
            warmed=warmed, fresh_sources=fresh_sources, fixed_routes=fixed_routes
        )

    def term_draws[S: MaskSourceStrategy](
        self,
        key: PRNGKeyArray,
        key_offset: int,
        term_idx: int,
        term: ReconLossTerm[S],
        fixed_routes: dict[tuple[int, int], tuple[Routes, ...]],
        leading: tuple[int, ...],
    ) -> TermDraws[S]:
        """The term's flat `(entry_idx, entry, draw_key, routes)` forwards. Key derivation
        reproduces the pre-unification production trace exactly (SPEC R1 — the main grid's
        `key_offset` is 1; a second grid keeps its per-term RNG disjoint by offsetting past
        the first); fresh-PGD entries reuse the ascent phase's fixed routes (SPEC S24)."""
        term_key = random.fold_in(key, key_offset + term_idx)
        draws: TermDraws[S] = []
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

    def grid_losses[S: MaskSourceStrategy](
        self,
        terms: tuple[ReconLossTerm[S], ...],
        draws_per_term: list[TermDraws[S]],
        draw_loss: DrawLoss[S],
    ) -> tuple[ReconstructionLoss, ...]:
        """Mean recon over each term's draws (SPEC S10'); the grid's owner supplies its
        per-draw dispatcher."""
        return tuple(
            mean_reconstruction_losses(
                tuple(
                    draw_loss(term_idx, entry_idx, term, entry, draw_key, routes)
                    for entry_idx, entry, draw_key, routes in draws
                )
            )
            for term_idx, (term, draws) in enumerate(zip(terms, draws_per_term, strict=True))
        )

    def main_draw_loss(
        self,
        model: DecomposedModel[PreparedT],
        *,
        prepared_weights: PreparedT,
        ci: CI,
        ci_stacked: Any,
        persistent_sources: dict[str, dict[str, Array]],
        ascended: AscendedAdversaries,
        stream: StreamInputs,
        step_f32: Array,
        reconstruction_specs: dict[str, ReconstructionSpec],
        term_coeffs: dict[str, Array | float],
    ) -> DrawLoss[MaskSourceStrategy]:
        """The main grid's per-draw dispatcher over the trainables: match the entry's
        mask-source strategy, run the masked forward, score against the stream's clean
        observations. Built INSIDE the loss fn — it closes over the live trainables.

        Persistent(-carrying) draws take the coeff on their MODEL-SIDE inputs
        (`model_cotangents_scaled`) and enter the total at weight 1, so the fused
        backward hands each adversary `dL/ds` unscaled (SPEC S14')."""

        def draw_loss(
            term_idx: int,
            entry_idx: int,
            term: ReconLossTerm[MaskSourceStrategy],
            entry: ReconForward[MaskSourceStrategy],
            draw_key: PRNGKeyArray,
            routes: Routes,
        ) -> ReconstructionLoss:
            match coeff_application(term):
                case "scales_model_cotangents":
                    draw_prepared = model_cotangents_scaled(
                        prepared_weights, term_coeffs[term.name]
                    )
                    draw_ci_lower = model_cotangents_scaled(ci.lower, term_coeffs[term.name])
                case "scales_loss":
                    draw_prepared, draw_ci_lower = prepared_weights, ci.lower
            with jax.named_scope("pd_recon_masked_fwd"):
                match entry.sources:
                    case StochasticSources():
                        # Stochastic recon passes `StochasticMasking` to `masked_forward`, so a scan
                        # target rebuilds masks from shared `ci_stacked` inside each checkpointed
                        # block (the full mask stack is never held). Explicit strategies pass
                        # `MaterializedMasking`; the engine holds no per-forward mask stacks.
                        assert ci_stacked is not None
                        masking: Masking = StochasticMasking(
                            ci_stacked=ci_stacked,
                            draw_key=draw_key,
                            live_sites=entry.live_sites,
                            routes=routes,
                        )
                    case ConstantSources() as strategy:
                        masking = MaterializedMasking(
                            component_masks=constant_source_masks(
                                strategy, ci.lower, entry.live_sites
                            ),
                            weight_delta_masks=None,
                            routes=routes,
                        )
                    case UnmaskedNoDeltaSources():
                        raise AssertionError(
                            "UnmaskedNoDeltaSources is non-target-pass vocabulary "
                            "(SPEC T4/T5); the main grid never carries it"
                        )
                    case FreshPGDSources():
                        component_masks, weight_delta_masks = masks_from_sources(
                            ci.lower,
                            ascended.fresh_sources[(term_idx, entry_idx)],
                            entry.live_sites,
                        )
                        masking = MaterializedMasking(
                            component_masks=component_masks,
                            weight_delta_masks=weight_delta_masks,
                            routes=routes,
                        )
                    case PersistentSources(state_key=state_key):
                        component_masks, weight_delta_masks = masks_from_sources(
                            draw_ci_lower, persistent_sources[state_key], entry.live_sites
                        )
                        masking = MaterializedMasking(
                            component_masks=component_masks,
                            weight_delta_masks=weight_delta_masks,
                            routes=routes,
                        )
                    case MixedPersistentStochasticSources(state_key=state_key):
                        adv_fraction = scheduled_value_traced(
                            step_f32, self.total_steps, entry.sources.cfg.adv_fraction
                        )
                        component_masks, weight_delta_masks, routes = (
                            mixed_persistent_stochastic_masks(
                                key=draw_key,
                                ci_lower=model_cotangents_scaled(ci.lower, term_coeffs[term.name]),
                                persistent_sources=persistent_sources[state_key],
                                live_sites=entry.live_sites,
                                components_per_site=self.c_by_site,
                                leading=stream.leading,
                                adv_fraction=adv_fraction,
                                stochastic_routes=routes,
                            )
                        )
                        masking = MaterializedMasking(
                            component_masks=component_masks,
                            weight_delta_masks=weight_delta_masks,
                            routes=routes,
                        )
                return self.masked_recon(
                    model,
                    prepared_weights=draw_prepared,
                    batch=stream.batch,
                    masking=masking,
                    capture_keys=self.hidden_acts_capture_keys_by_term[term.name],
                    reconstruction=reconstruction_specs[term.name],
                    clean=stream.clean,
                )

        return draw_loss

    def apply_gradients(
        self,
        decomposition: Decomposition,
        training: TrainingItem,
        warmed_advs: dict[str, PersistentAdversary],
        components_grad: Any,
        ci_fn_grad: Any,
        persistent_source_grads: dict[str, dict[str, Array]],
        step_f32: Array,
    ) -> tuple[TrainState, dict[str, Array]]:
        """The optimizer tail: grad-norm metrics, each adversary's final ascent from the
        fused graph (SPEC S13'/S14': the source path is never coeff-scaled, so the
        backward's grad IS dL_term/d(sources) — exact since one source bundle feeds one
        term, S23), then both optimizer updates into the next `TrainState`."""
        grad_norm_metrics = _grad_norm_metrics(components_grad, ci_fn_grad)

        new_adversaries = {
            state_key: warmed_advs[state_key].final_ascend(
                persistent_source_grads[state_key], step_f32, self.total_steps
            )
            for state_key in warmed_advs
        }

        components_updates, new_components_opt_state = self.components_optimizer.update(
            components_grad,
            training.components_opt_state,
            eqx.filter(decomposition.components, eqx.is_array),
        )
        ci_fn_updates, new_ci_fn_opt_state = self.ci_fn_optimizer.update(
            ci_fn_grad,
            training.ci_fn_opt_state,
            eqx.filter(decomposition.ci_fn, eqx.is_array),
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
        return new_state, grad_norm_metrics

    def train_metrics(
        self,
        *,
        total_loss: Array,
        imp_lp: Array,
        imp_freq: Array,
        imp_min_param: Array,
        term_breakdowns: tuple[ReconstructionLoss, ...],
        grad_norm_metrics: dict[str, Array],
        adversaries: dict[str, PersistentAdversary],
        step_f32: Array,
    ) -> dict[str, Array]:
        """The step's scalar record over THIS atoms' terms: totals, the imp-min pair under
        their penalty-kind keys, per-term losses + breakdowns, grad norms, source LRs."""
        term_losses = tuple(breakdown.total for breakdown in term_breakdowns)
        metrics = {
            "total": total_loss,
            self.imp_loss_key: imp_lp,
            "freq": imp_freq,
            self.imp_min_param_key: imp_min_param,
            **{f"loss/{t.name}": v for t, v in zip(self.recon_terms, term_losses, strict=True)},
            **grad_norm_metrics,
        }
        for term, breakdown in zip(self.recon_terms, term_breakdowns, strict=True):
            prefix = f"loss/{term.name}"
            metrics |= {
                f"{prefix}/{suffix}": value
                for suffix, value in reconstruction_loss_metrics(breakdown).items()
            }
        source_lrs = {
            k: adv.source_lr(step_f32, self.total_steps) for k, adv in adversaries.items()
        }
        if len(source_lrs) == 1:
            metrics["src_lr"] = next(iter(source_lrs.values()))
        else:
            metrics |= {f"schedules/lr/src/{k}": v for k, v in source_lrs.items()}
        return metrics


# ───────────────────────────── the step factory ─────────────────────────────


def make_train_step[PreparedT](
    model_static: DecomposedModel[PreparedT],
    *,
    losses: LossSurface,
    components_optimizer: optax.GradientTransformation,
    ci_fn_optimizer: optax.GradientTransformation,
    total_steps: int,
    remat_recon_forwards: bool,
    remat_ci_fn: bool,
    ci_capture_keys: CaptureKeys,
    mesh: Mesh | None = None,
    ascend_replicate: bool = False,
    compiler_options: dict[str, bool | int | str] | None = None,
):
    """Build the `eqx.filter_jit`'d `step(model, state, batch, key) -> (state, metrics)`.

    `model` is the jit ARG (frozen 8B weights traced as array leaves, never baked); the
    factory closes over only static config (`site_names`, `recon_loss_fn`, term wiring) read
    off `model_static` here — the distinct name keeps an accidental closure over the
    array-bearing model (the HLO-baking hazard) a loud NameError, never silent.
    `losses` (from `build_objective`) is the `LossSurface` record — the
    faithfulness + importance-minimality singletons and the recon Σ, read by name. `mesh`
    (when given) pins every batch-leading activation over the full mesh
    (`P(('replicate', 'fsdp'), ...)`) so masked forwards stay on per-rank sub-batches
    (activation memory 1/N). The body is a straight-line composition of the `_StepAtoms`
    vocabulary."""
    atoms = _StepAtoms(
        model_static,
        recon_terms=losses.recon,
        imp=losses.imp,
        components_optimizer=components_optimizer,
        ci_fn_optimizer=ci_fn_optimizer,
        total_steps=total_steps,
        remat_recon_forwards=remat_recon_forwards,
        remat_ci_fn=remat_ci_fn,
        ci_capture_keys=ci_capture_keys,
        mesh=mesh,
        ascend_replicate=ascend_replicate,
    )
    coeff_schedules: dict[str, LossCoeff] = {
        losses.faith.name: losses.faith.coeff,
        losses.imp.name: losses.imp.coeff,
        **{term.name: term.coeff for term in losses.recon},
        **{
            f"{term.name}/hidden_acts_reconstruction": term.hidden_acts_reconstruction.coeff
            for term in losses.recon
            if term.hidden_acts_reconstruction is not None
        },
    }
    if losses.imp.cfg.frequency is not None:
        coeff_schedules[f"{losses.imp.name}/frequency"] = losses.imp.cfg.frequency.coeff

    @jaxtyped(typechecker=beartype)
    def step(
        model: DecomposedModel[PreparedT],
        state: TrainState,
        batch: Any,
        key: PRNGKeyArray,
    ) -> tuple[TrainState, dict[str, Array]]:
        decomposition = state.decomposition
        training = state.training
        step_f32 = training.step.astype(jnp.float32)
        imp_min_param = annealed_imp_min_param(step_f32, atoms.total_steps, atoms.imp_min)
        # Every coefficient's per-step value, resolved once at the top of the step: the
        # loss math below sees only scalars, never schedule objects.
        faith_coeff = coeff_at(step_f32, atoms.total_steps, losses.faith.coeff)
        imp_coeff = coeff_at(step_f32, atoms.total_steps, atoms.imp_coeff)
        freq_coeff = coeff_at(step_f32, atoms.total_steps, atoms.freq_coeff)
        recon_coeffs = tuple(
            coeff_at(step_f32, atoms.total_steps, term.coeff) for term in atoms.recon_terms
        )
        term_coeffs: dict[str, Array | float] = {
            term.name: coeff for term, coeff in zip(atoms.recon_terms, recon_coeffs, strict=True)
        }
        reconstruction_specs = atoms.reconstruction_specs_at(atoms.recon_terms, step_f32)

        stream = atoms.prep_stream(model, batch, atoms.hidden_acts_capture_keys)

        # ── adversary ascents: params + CI detached (SPEC §4.5) ──
        prepared_weights, recon_vjp = atoms.component_weights_vjp(model, decomposition.components)
        detached_prepared_weights = jax.lax.stop_gradient(prepared_weights)
        ascend_prepared_weights = atoms.replicate_for_ascend(detached_prepared_weights)
        # The CI envelope is a pure fn of the batch, so compute it ONCE per step — the value +
        # its vjp, mirroring `prepared_weights`/`recon_vjp`. The ascend uses the stop_gradient'd
        # value; `loss_fn` takes the live value and its ci-fn grad is pulled back through
        # `ci_vjp`. So the (≈10x-the-target) CI fn is forward-evaluated ONCE, not once detached
        # for the ascend + once inside the main backward.
        ci, ci_vjp = atoms.ci_forward_vjp(decomposition.ci_fn, stream.taps)
        ci_lower_detached = jax.lax.stop_gradient(ci).lower

        ascended = atoms.ascend_adversaries(
            model,
            stream,
            ascend_prepared_weights,
            ci_lower_detached,
            training.adversaries,
            key,
            step_f32,
            reconstruction_specs,
        )

        # ── main losses: live components/ci; the PERSISTENT sources participate in
        # the graph so their gradient comes from the SAME backward (SPEC S14'); they
        # are NOT detached here, but components/ci grads through them are what torch
        # gets too (sources are leaves). ──
        warmed_sources = {k: a.sources for k, a in ascended.warmed.items()}
        draws_per_term = [
            atoms.term_draws(key, 1, term_idx, term, ascended.fixed_routes, stream.leading)
            for term_idx, term in enumerate(atoms.recon_terms)
        ]

        def loss_fn(
            trainable: tuple[PreparedT, ComponentStacks, CI, dict[str, dict[str, Array]]],
        ) -> tuple[Array, tuple[Array, Array, Array, Array, tuple[ReconstructionLoss, ...]]]:
            prepared_weights, components, ci, persistent_sources = trainable
            ci_stacked = model.stack_ci(ci.lower)
            faith_loss = faithfulness_loss(model.weight_deltas(components))
            imp_lp, imp_freq = imp_min_terms(ci.upper, atoms.imp_min, imp_min_param)

            term_breakdowns = atoms.grid_losses(
                atoms.recon_terms,
                draws_per_term,
                atoms.main_draw_loss(
                    model,
                    prepared_weights=prepared_weights,
                    ci=ci,
                    ci_stacked=ci_stacked,
                    persistent_sources=persistent_sources,
                    ascended=ascended,
                    stream=stream,
                    step_f32=step_f32,
                    reconstruction_specs=reconstruction_specs,
                    term_coeffs=term_coeffs,
                ),
            )
            term_losses = tuple(breakdown.total for breakdown in term_breakdowns)
            base = faith_coeff * faith_loss + imp_coeff * imp_lp + freq_coeff * imp_freq
            # The differentiated total: persistent-carrying terms enter at weight 1 —
            # their coeff already rides their model-side cotangents — so the backward
            # hands each adversary dL/ds unscaled (SPEC S14'). The OBJECTIVE (the
            # reported `total`, Σ coeff·L) has the same gradients up to that plumbing
            # and the identical value for every non-persistent term.
            total_loss = base
            reported_total = base
            for term, coeff, term_loss in zip(
                atoms.recon_terms, recon_coeffs, term_losses, strict=True
            ):
                match coeff_application(term):
                    case "scales_loss":
                        total_loss = total_loss + coeff * term_loss
                    case "scales_model_cotangents":
                        total_loss = total_loss + term_loss
                reported_total = reported_total + coeff * term_loss
            return total_loss, (reported_total, faith_loss, imp_lp, imp_freq, term_breakdowns)

        with jax.named_scope("pd_value_and_grad"):
            (_, (reported_total, faith_loss, imp_lp, imp_freq, term_breakdowns)), grads = (
                eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                    (prepared_weights, decomposition.components, ci, warmed_sources)
                )
            )
        prepared_grad, components_grad_faith, ci_grad, persistent_source_grads = grads
        components_grad = jax.tree.map(
            lambda recon_g, faith_g: recon_g + faith_g,
            recon_vjp(prepared_grad)[0],
            components_grad_faith,
        )
        ci_fn_grad = ci_vjp(ci_grad)[0]

        new_state, grad_norm_metrics = atoms.apply_gradients(
            decomposition,
            training,
            ascended.warmed,
            components_grad,
            ci_fn_grad,
            persistent_source_grads,
            step_f32,
        )
        metrics = (
            atoms.train_metrics(
                total_loss=reported_total,
                imp_lp=imp_lp,
                imp_freq=imp_freq,
                imp_min_param=imp_min_param,
                term_breakdowns=term_breakdowns,
                grad_norm_metrics=grad_norm_metrics,
                adversaries=training.adversaries,
                step_f32=step_f32,
            )
            | {"faith": faith_loss}
            | _scheduled_coeff_metrics(step_f32, atoms.total_steps, coeff_schedules)
        )
        return new_state, metrics

    return filter_jit(step, donate="all-except-first", compiler_options=compiler_options)


# ───────────────────────────── the targeted (tPD) step factory ─────────────────────────────


@dataclass(frozen=True)
class CIScaledWeightDecay:
    """The tPD CI-scaled weight decay (SPEC T11) as the targeted step consumes it: the
    authored coefficient joined with the components optimizer's LR schedule — the
    per-step decay rate is `components_lr(step) * coeff`, AdamW's decoupled-decay
    convention, so the decay tracks the LR schedule like AdamW's own `weight_decay`
    would."""

    coeff: float
    components_lr: Callable[[Array], Array]


def _per_component_batch_max(ci_lower: dict[str, Array]) -> dict[str, Array]:
    """Each site's per-subcomponent max CI over every leading (batch AND position) axis,
    fp32. Reads `lower` deliberately: `lower ≡ clip(upper, 0, 1)` pointwise (S6), so the
    two squashings agree on this statistic and no clamp is needed."""
    return {
        site: jnp.max(v.astype(jnp.float32), axis=tuple(range(v.ndim - 1)))
        for site, v in ci_lower.items()
    }


def _scale_subcomponents(
    components: ComponentStacks, scale: dict[str, Float[Array, " C"]]
) -> ComponentStacks:
    """Scale each site's V columns and U rows by that site's per-subcomponent factor,
    stacked per shape group so the multiply stays in the owner-partitioned layout."""
    rows_by_shape: dict[VUShape, list[Array]] = {}
    for name, shape, slot in components.site_slots:
        rows = rows_by_shape.setdefault(shape, [])
        assert slot == len(rows), (name, shape, slot)
        rows.append(scale[name])
    stacks = {}
    for shape, (vs, us) in components.stacks.items():
        keep = jnp.stack(rows_by_shape[shape])  # [g, C]
        stacks[shape] = (vs * keep[:, None, :], us * keep[:, :, None])
    return ComponentStacks(stacks=stacks, site_slots=components.site_slots)


def make_targeted_train_step[PreparedT](
    model_static: DecomposedModel[PreparedT],
    *,
    objective: TargetedObjective,
    ci_scaled_weight_decay: CIScaledWeightDecay | None,
    components_optimizer: optax.GradientTransformation,
    ci_fn_optimizer: optax.GradientTransformation,
    total_steps: int,
    remat_recon_forwards: bool,
    remat_ci_fn: bool,
    ci_capture_keys: CaptureKeys,
    mesh: Mesh | None = None,
    ascend_replicate: bool = False,
    compiler_options: dict[str, bool | int | str] | None = None,
):
    """Build the tPD `step(model, state, batch, nontarget_batch, key)` (SPEC §11).

    Two passes over one shared `_StepAtoms` vocabulary, summed into ONE `value_and_grad`:
    the TARGET pass runs the full decomposition objective (recon grid + adversary
    ascents) on the narrow stream, and the NON-TARGET pass runs its delta-pinned grid +
    importance-minimality on the broad stream. Each stream runs at its own natural
    geometry (SPEC T8) and every position is scored. There is no faithfulness role
    anywhere in it — `objective` cannot carry one.

    The batch args are the streams in pass order: `batch` the target stream (whose global
    batch is `pd.batch_size` — persistent sources size from it), `nontarget_batch` the
    broad stream."""
    # The library boundary behind `NontargetPass.recon`'s narrow type, for objectives
    # built outside it (SPEC T5).
    for term in objective.nontarget.recon:
        for entry in term.plan:
            assert isinstance(
                entry.sources, StochasticSources | ConstantSources | UnmaskedNoDeltaSources
            ), term.name
    atoms = _StepAtoms(
        model_static,
        recon_terms=objective.target.recon,
        imp=objective.target.imp,
        components_optimizer=components_optimizer,
        ci_fn_optimizer=ci_fn_optimizer,
        total_steps=total_steps,
        remat_recon_forwards=remat_recon_forwards,
        remat_ci_fn=remat_ci_fn,
        ci_capture_keys=ci_capture_keys,
        mesh=mesh,
        ascend_replicate=ascend_replicate,
    )
    nt_terms = objective.nontarget.recon
    coeff_schedules: dict[str, LossCoeff] = {
        objective.target.imp.name: objective.target.imp.coeff,
        **{term.name: term.coeff for term in objective.target.recon},
        **{
            f"{term.name}/hidden_acts_reconstruction": term.hidden_acts_reconstruction.coeff
            for term in objective.target.recon
            if term.hidden_acts_reconstruction is not None
        },
    }
    if objective.target.imp.cfg.frequency is not None:
        coeff_schedules[f"{objective.target.imp.name}/frequency"] = (
            objective.target.imp.cfg.frequency.coeff
        )

    imp_name = objective.target.imp.name
    nontarget_coeff_schedules: dict[str, LossCoeff] = {
        imp_name: objective.nontarget.impmin_coeff,
        **{term.name: term.coeff for term in nt_terms},
    }

    def nontarget_draw_loss(
        model: DecomposedModel[PreparedT],
        prepared_weights: PreparedT,
        nt_ci: CI,
        nt_stream: StreamInputs,
        nt_reconstruction_specs: dict[str, ReconstructionSpec],
    ) -> DrawLoss[StochasticSources | ConstantSources | UnmaskedNoDeltaSources]:
        """The non-target grid's per-draw dispatcher: every delta mask pinned to 1.0 —
        except the unmasked-no-delta arm, which pins it to 0.0 (SPEC T4) — scored
        against the broad stream's frozen output."""

        def draw_loss(
            term_idx: int,
            entry_idx: int,
            term: ReconLossTerm[StochasticSources | ConstantSources | UnmaskedNoDeltaSources],
            entry: ReconForward[StochasticSources | ConstantSources | UnmaskedNoDeltaSources],
            draw_key: PRNGKeyArray,
            routes: Routes,
        ) -> ReconstructionLoss:
            del term_idx, entry_idx
            with jax.named_scope("pd_nontarget_masked_fwd"):
                match entry.sources:
                    case StochasticSources():
                        component_masks, delta_masks = stochastic_delta_pinned_masks(
                            nt_ci.lower, entry.live_sites, draw_key
                        )
                    case ConstantSources(value=value):
                        component_masks, delta_masks = constant_delta_pinned_masks(
                            value, nt_ci.lower, entry.live_sites
                        )
                    case UnmaskedNoDeltaSources():
                        component_masks, delta_masks = unmasked_no_delta_masks(
                            nt_ci.lower, entry.live_sites
                        )
                return atoms.masked_recon(
                    model,
                    prepared_weights=prepared_weights,
                    batch=nt_stream.batch,
                    masking=MaterializedMasking(
                        component_masks=component_masks,
                        weight_delta_masks=delta_masks,
                        routes=routes,
                    ),
                    capture_keys=frozenset(),
                    reconstruction=nt_reconstruction_specs[term.name],
                    clean=nt_stream.clean,
                )

        return draw_loss

    @jaxtyped(typechecker=beartype)
    def targeted_step(
        model: DecomposedModel[PreparedT],
        state: TrainState,
        batch: Any,
        nontarget_batch: Any,
        key: PRNGKeyArray,
    ) -> tuple[TrainState, dict[str, Array]]:
        decomposition = state.decomposition
        training = state.training
        step_f32 = training.step.astype(jnp.float32)
        imp_min_param = annealed_imp_min_param(step_f32, atoms.total_steps, atoms.imp_min)
        # Every coefficient's per-step value, resolved once at the top of the step: the
        # loss math below sees only scalars, never schedule objects.
        imp_coeff = coeff_at(step_f32, atoms.total_steps, atoms.imp_coeff)
        freq_coeff = coeff_at(step_f32, atoms.total_steps, atoms.freq_coeff)
        recon_coeffs = tuple(
            coeff_at(step_f32, atoms.total_steps, term.coeff) for term in atoms.recon_terms
        )
        term_coeffs: dict[str, Array | float] = {
            term.name: coeff for term, coeff in zip(atoms.recon_terms, recon_coeffs, strict=True)
        }
        nt_imp_coeff = coeff_at(step_f32, atoms.total_steps, objective.nontarget.impmin_coeff)
        nt_recon_coeffs = tuple(
            coeff_at(step_f32, atoms.total_steps, term.coeff) for term in nt_terms
        )
        reconstruction_specs = atoms.reconstruction_specs_at(atoms.recon_terms, step_f32)
        nt_reconstruction_specs = atoms.reconstruction_specs_at(nt_terms, step_f32)

        stream = atoms.prep_stream(model, batch, atoms.hidden_acts_capture_keys)
        nt_stream = atoms.prep_stream(model, nontarget_batch, frozenset())

        # ── adversary ascents: TARGET pass only, params + CI detached (SPEC §4.5/§11) ──
        prepared_weights, recon_vjp = atoms.component_weights_vjp(model, decomposition.components)
        detached_prepared_weights = jax.lax.stop_gradient(prepared_weights)
        ascend_prepared_weights = atoms.replicate_for_ascend(detached_prepared_weights)
        ci, ci_vjp = atoms.ci_forward_vjp(decomposition.ci_fn, stream.taps)
        nt_ci, nt_ci_vjp = atoms.ci_forward_vjp(decomposition.ci_fn, nt_stream.taps)
        ci_lower_detached = jax.lax.stop_gradient(ci).lower

        ascended = atoms.ascend_adversaries(
            model,
            stream,
            ascend_prepared_weights,
            ci_lower_detached,
            training.adversaries,
            key,
            step_f32,
            reconstruction_specs,
        )

        warmed_sources = {k: a.sources for k, a in ascended.warmed.items()}
        draws_per_term = [
            atoms.term_draws(key, 1, term_idx, term, ascended.fixed_routes, stream.leading)
            for term_idx, term in enumerate(atoms.recon_terms)
        ]
        # The non-target grid's per-term RNG offsets past the target grid's, so the two
        # grids' draws stay disjoint under the one step key (SPEC R1).
        nt_draws_per_term = [
            atoms.term_draws(key, 1 + len(atoms.recon_terms), term_idx, term, {}, nt_stream.leading)
            for term_idx, term in enumerate(nt_terms)
        ]

        def loss_fn(
            trainable: tuple[PreparedT, CI, CI, dict[str, dict[str, Array]]],
        ) -> tuple[
            Array,
            tuple[
                Array,
                Array,
                Array,
                tuple[ReconstructionLoss, ...],
                dict[str, Array],
            ],
        ]:
            prepared_weights, ci, nt_ci, persistent_sources = trainable
            ci_stacked = model.stack_ci(ci.lower)
            imp_lp, imp_freq = imp_min_terms(ci.upper, atoms.imp_min, imp_min_param)

            term_breakdowns = atoms.grid_losses(
                atoms.recon_terms,
                draws_per_term,
                atoms.main_draw_loss(
                    model,
                    prepared_weights=prepared_weights,
                    ci=ci,
                    ci_stacked=ci_stacked,
                    persistent_sources=persistent_sources,
                    ascended=ascended,
                    stream=stream,
                    step_f32=step_f32,
                    reconstruction_specs=reconstruction_specs,
                    term_coeffs=term_coeffs,
                ),
            )
            base = imp_coeff * imp_lp + freq_coeff * imp_freq
            # Differentiated total vs reported total: see the plain factory — a
            # persistent-carrying term's coeff rides its model-side cotangents, so it
            # enters the total at weight 1 and its adversary receives dL/ds (SPEC S14').
            total_loss = base
            reported_total = base
            for term, coeff, breakdown in zip(
                atoms.recon_terms, recon_coeffs, term_breakdowns, strict=True
            ):
                match coeff_application(term):
                    case "scales_loss":
                        total_loss = total_loss + coeff * breakdown.total
                    case "scales_model_cotangents":
                        total_loss = total_loss + breakdown.total
                reported_total = reported_total + coeff * breakdown.total

            # ── the non-target pass: its imp-min (the shared annealed param, its own
            # coeff) + its delta-pinned grid, added to the SAME total so one backward
            # grads both passes (SPEC T1). ──
            nt_imp_lp, nt_imp_freq = imp_min_terms(nt_ci.upper, atoms.imp_min, imp_min_param)
            nt_total = nt_imp_coeff * nt_imp_lp + freq_coeff * nt_imp_freq
            nt_aux = {
                f"nontarget_data/loss/{imp_name}": nt_imp_lp,
                "nontarget_data/loss/FrequencyMinimalityLoss": nt_imp_freq,
            }
            nt_breakdowns = atoms.grid_losses(
                nt_terms,
                nt_draws_per_term,
                nontarget_draw_loss(
                    model, prepared_weights, nt_ci, nt_stream, nt_reconstruction_specs
                ),
            )
            for term, coeff, breakdown in zip(
                nt_terms, nt_recon_coeffs, nt_breakdowns, strict=True
            ):
                nt_total = nt_total + coeff * breakdown.total
                nt_aux[f"nontarget_data/loss/{term.name}"] = breakdown.total
            nt_aux["nontarget_data/loss/total"] = nt_total
            total_loss = total_loss + nt_total
            reported_total = reported_total + nt_total
            return total_loss, (reported_total, imp_lp, imp_freq, term_breakdowns, nt_aux)

        with jax.named_scope("pd_value_and_grad"):
            (_, (reported_total, imp_lp, imp_freq, term_breakdowns, nt_aux)), grads = (
                eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                    (prepared_weights, ci, nt_ci, warmed_sources)
                )
            )
        prepared_grad, ci_grad, nt_ci_grad, persistent_source_grads = grads
        # No faithfulness role ⇒ the components' whole gradient arrives through the
        # compute-weights pullback.
        components_grad = recon_vjp(prepared_grad)[0]
        # The CI fn saw both streams; its total gradient is the sum of the two pullbacks.
        ci_fn_grad = jax.tree.map(
            lambda target_g, nt_g: target_g + nt_g,
            ci_vjp(ci_grad)[0],
            nt_ci_vjp(nt_ci_grad)[0],
        )

        new_state, grad_norm_metrics = atoms.apply_gradients(
            decomposition,
            training,
            ascended.warmed,
            components_grad,
            ci_fn_grad,
            persistent_source_grads,
            step_f32,
        )
        wd_metrics: dict[str, Array] = {}
        if ci_scaled_weight_decay is not None:
            # T11: an update rule on the post-step component masters, not a loss term —
            # nothing differentiates through it. Off the step's own pre-update forward
            # CIs, maxed over BOTH streams: a component important on either is not dead.
            batch_max_ci = _per_component_batch_max(ci.lower)
            nt_batch_max_ci = _per_component_batch_max(nt_ci.lower)
            rate = ci_scaled_weight_decay.components_lr(step_f32) * ci_scaled_weight_decay.coeff
            decay = {
                site: rate * (1.0 - jnp.maximum(batch_max_ci[site], nt_batch_max_ci[site]))
                for site in atoms.site_names
            }
            decayed = _scale_subcomponents(
                new_state.decomposition.components, {site: 1.0 - d for site, d in decay.items()}
            )
            new_state = TrainState(
                decomposition=Decomposition(
                    components=decayed, ci_fn=new_state.decomposition.ci_fn
                ),
                training=new_state.training,
            )
            decay_all = jnp.concatenate(list(decay.values()))
            wd_metrics = {
                "ci_scaled_weight_decay/mean": jnp.mean(decay_all),
                "ci_scaled_weight_decay/max": jnp.max(decay_all),
            }
        metrics = (
            atoms.train_metrics(
                total_loss=reported_total,
                imp_lp=imp_lp,
                imp_freq=imp_freq,
                imp_min_param=imp_min_param,
                term_breakdowns=term_breakdowns,
                grad_norm_metrics=grad_norm_metrics,
                adversaries=training.adversaries,
                step_f32=step_f32,
            )
            | nt_aux
            | wd_metrics
            | _scheduled_coeff_metrics(step_f32, atoms.total_steps, coeff_schedules)
            | {
                f"nontarget_data/{key}": value
                for key, value in _scheduled_coeff_metrics(
                    step_f32, atoms.total_steps, nontarget_coeff_schedules
                ).items()
            }
        )
        return new_state, metrics

    return filter_jit(targeted_step, donate="all-except-first", compiler_options=compiler_options)


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
