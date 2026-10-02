"""Plain VPD and targeted-PD training steps over a `DecomposedModel`.

Each step computes clean outputs and CI, ascends adversarial sources, scores the
objective, and updates components, CI parameters, and persistent adversaries.
Reconstruction terms each run one masked forward. The default `e2e` adversary
retakes an output-only source gradient when the outer term includes auxiliaries;
`term` mode reuses the combined reconstruction gradient from the shared backward.

Model masters are fp32 and forwards use bf16 casts. Schedules resolve from the
step counter inside JIT. Reconstruction term keys follow config-list order via
`fold_in(step_key, offset + i)`; target and non-target grids use disjoint offsets.
The two factories share `ForwardSubstrate` and `ReconGrid` but keep separate
plain and targeted step bodies."""

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, NamedTuple, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from beartype import beartype
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Float32, Int32, PRNGKeyArray, jaxtyped

from param_decomp.core.adversary import (
    PersistentAdversary,
    Sources,
    SourceStacks,
    init_fresh_pgd_sources,
)
from param_decomp.core.ci_fn.interface import CI, CIFn
from param_decomp.core.ci_fn.runtime import evaluate_ci_from_captures
from param_decomp.core.components import (
    BlockedFactorization,
    ComponentStacks,
    DenseFactorization,
    Factorization,
    SelectedCI,
    SiteCI,
    SiteSpec,
    map_site_ci,
    selected_component_maxes,
    vu_groups,
)
from param_decomp.core.decomposed_linear import constrain_component_activation
from param_decomp.core.dict_utils import dict_safe_update_
from param_decomp.core.faithfulness import FaithfulnessLossFn
from param_decomp.core.losses import (
    BatchFrequencyPenalty,
    EmaFrequencyPenalty,
    FrequencyEstimator,
    FrequencyPenalty,
    ReconstructionLoss,
    per_component_frequencies,
    reconstruction_loss,
    reconstruction_loss_metrics,
)
from param_decomp.core.masking import (
    constant_delta_pinned_masking,
    materialize_masking,
    mixed_persistent_stochastic_masking,
    sample_source_pool,
    source_masking,
    stochastic_delta_pinned_masking,
    unmasked_no_delta_masking,
)
from param_decomp.core.model import (
    CaptureKeys,
    ComponentActivations,
    MaterializedMasking,
    PlacedModel,
    SiteRoutes,
    faithfulness_weight_deltas,
    select_captures,
)
from param_decomp.core.objective import (
    MinimalityResult,
    NonlinearityResult,
    PDObjective,
    ResolvedNonlinearity,
    TargetedPDObjective,
    evaluate_minimality,
)
from param_decomp.core.optimizer import ScheduledOptimizer, ScheduledOptimizerState
from param_decomp.core.placement import PlacementRules
from param_decomp.core.recon import (
    PERSISTENT_SOURCE_TYPES,
    AnyReconLossTerm,
    AuxiliaryReconstructionSpec,
    ConstantSources,
    ForwardObservations,
    FreshPGDSources,
    MaskSourceStrategy,
    MixedPersistentStochasticSources,
    PersistentSourcePool,
    PersistentSources,
    ReconLossTerm,
    Routes,
    StochasticSources,
    UnmaskedNoDeltaSources,
    auxiliary_capture_keys,
    reconstruction_observations,
)
from param_decomp.core.runtime_schedule import RuntimeSchedule, scheduled_value_at, train_frac_at
from param_decomp.core.sharding import batch_shard_leading
from param_decomp.sequence import SequenceLayout


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Decomposition[Conditioning]:
    """The trained PRODUCT: V/U components + the CI fn (fp32 masters). Checkpointed as
    its own orbax item so downstream consumers restore it with
    zero knowledge of the training process (optimizer states, adversaries, step)."""

    components: ComponentStacks  # the universal trainable V/U pytree, fp32 masters
    ci_fn: CIFn[Conditioning]  # fp32 masters


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PDTrainingState:
    """Plain PD's objective, optimizer history, adversaries, and frequency estimates."""

    objective: PDObjective
    frequency: FrequencyEstimator
    components_opt_state: ScheduledOptimizerState
    ci_fn_opt_state: ScheduledOptimizerState
    adversaries: dict[str, PersistentAdversary]
    step: Int32[Array, ""]


class TrainingProgress(Protocol):
    """What the run loop reads from either algorithm's training state."""

    @property
    def components_opt_state(self) -> ScheduledOptimizerState: ...

    @property
    def ci_fn_opt_state(self) -> ScheduledOptimizerState: ...

    @property
    def adversaries(self) -> dict[str, PersistentAdversary]: ...

    @property
    def step(self) -> Int32[Array, ""]: ...


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainState[Conditioning, Training: TrainingProgress]:
    """The trained decomposition paired with its algorithm's training state; checkpointed
    as those two orbax items."""

    decomposition: Decomposition[Conditioning]
    training: Training


type PDState[Conditioning] = TrainState[Conditioning, PDTrainingState]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TargetedPDTrainingState:
    """Targeted PD's two-stream objective, optimizer history, adversaries, and decay."""

    objective: TargetedPDObjective
    target_frequency: FrequencyEstimator
    nontarget_frequency: FrequencyEstimator
    components_opt_state: ScheduledOptimizerState
    ci_fn_opt_state: ScheduledOptimizerState
    adversaries: dict[str, PersistentAdversary]
    ci_scaled_weight_decay: "CIScaledWeightDecay | None"
    step: Int32[Array, ""]


type TargetedPDState[Conditioning] = TrainState[Conditioning, TargetedPDTrainingState]


def _grad_norm_metrics(
    components_grad: ComponentStacks, ci_fn_grad: object, mesh: Mesh | None
) -> dict[str, Array]:
    """Pre-clip gradient L2 norms, matching the torch `component_grad_norms` families.

    Components norms are per SITE per factor — `grad_norms/components.vu['<site>'][0|1]`
    (0=V, 1=U), e.g. `grad_norms/components.vu['layers.18.mlp.gate_proj'][0]`. Sites are
    the semantic unit; the grouped stacks they're stored in are not. The key spells
    the retired per-site pytree path so wandb histories overlay across the stacking
    refactor. Ci-fn norms are per LEAF of whatever pytree the CI fn is
    (`grad_norms/ci_fns<path>`), plus the overlay-critical
    `grad_norms/summary/{components,ci_fns,total}`."""
    out: dict[str, Array] = {}

    def per_slice_sq(stack: Float[Array, "g ..."]) -> Float[Array, " g"]:
        sq = jnp.sum(stack.astype(jnp.float32) ** 2, axis=tuple(range(1, stack.ndim)))
        # Replicate the [g] vector ONCE; the per-site scalar reads below are then local
        # slices instead of one tiny cross-mesh broadcast per site per factor (2·n_sites
        # collectives per step under a stack-sharded persist layout).
        if mesh is not None:
            sq = jax.sharding.reshard(sq, NamedSharding(mesh, P()))
        return sq

    factor_sq = {
        shape: (per_slice_sq(Vs), per_slice_sq(Us))
        for shape, (Vs, Us) in components_grad.stacks.items()
    }
    for name, shape, index in components_grad.site_stack_indices:
        v_sq, u_sq = factor_sq[shape]
        out[f"grad_norms/components.vu['{name}'][0]"] = jnp.sqrt(v_sq[index])
        out[f"grad_norms/components.vu['{name}'][1]"] = jnp.sqrt(u_sq[index])
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


def uv_norm_ratio_metrics(components: ComponentStacks) -> dict[str, Array]:
    """Return each site's Frobenius-norm ratio ``||U|| / ||V||`` and summaries."""

    def per_slice_sq(stack: Float[Array, "g ..."]) -> Float[Array, " g"]:
        sq = jnp.sum(stack.astype(jnp.float32) ** 2, axis=tuple(range(1, stack.ndim)))
        # Replicate the tiny [g] vector ONCE before the per-site reads: under a
        # stack-owned persist layout (`sharding: owner`) the stack axis is sharded, and
        # a static per-index slice of a sharded dim is unimplemented. Ambient-mesh guard
        # (the `site_forward` pattern): a no-op off-mesh (toys / CPU tests).
        if not jax.sharding.get_abstract_mesh().empty:
            sq = jax.sharding.reshard(sq, P())
        return sq

    factor_sq = {
        shape: (per_slice_sq(Vs), per_slice_sq(Us)) for shape, (Vs, Us) in components.stacks.items()
    }
    metrics: dict[str, Array] = {}
    ratios = []
    for name, shape, index in components.site_stack_indices:
        v_sq, u_sq = factor_sq[shape]
        ratio = jnp.sqrt(u_sq[index] / v_sq[index])
        metrics[f"uv_norm_ratio['{name}']"] = ratio
        ratios.append(ratio)

    stacked = jnp.stack(ratios)
    metrics["uv_norm_ratio_mean"] = jnp.mean(stacked)
    metrics["uv_norm_ratio_max"] = jnp.max(stacked)
    return metrics


@jax.custom_vjp
def _cotangent_scaled(x: Array, by: Float32[Array, ""]) -> Array:
    del by  # forward-inert: consumed only by the vjp
    return x


def _cotangent_scaled_fwd(x: Array, by: Float32[Array, ""]) -> tuple[Array, Float32[Array, ""]]:
    return x, by


def _cotangent_scaled_bwd(by: Float32[Array, ""], g: Array) -> tuple[Array, Float32[Array, ""]]:
    # An UNREDUCED cotangent (the chained-reduced weights') may only multiply a scalar
    # typed `reduced` over the same axes: (Σᵢ aᵢ)·c = Σᵢ(aᵢ·c), a pure retag.
    scale = by.astype(g.dtype)
    unreduced = frozenset(jax.typeof(g).sharding.spec.unreduced)
    if unreduced:
        scale = jax.sharding.reshard(scale, P(reduced=unreduced))
    return g * scale, jnp.zeros_like(by)


_cotangent_scaled.defvjp(_cotangent_scaled_fwd, _cotangent_scaled_bwd)


def model_cotangents_scaled[T](tree: T, by: Float32[Array, ""]) -> T:
    """Return `tree` unchanged in the forward and scale its backward cotangents by `by`.

    Wrapping the model-side inputs (prepared weights, CI envelope) applies a term's
    coefficient to component and CI gradients while persistent-source gradients
    stay unscaled. Sources can therefore ascend even when the loss coefficient is
    zero, with no division by that coefficient."""
    # Integer leaves (a SelectedCI's block indices) carry no cotangent; ride untouched.
    return jax.tree.map(
        lambda leaf: _cotangent_scaled(leaf, by) if eqx.is_inexact_array(leaf) else leaf, tree
    )


type CoeffApplication = Literal["scales_loss", "scales_model_cotangents"]


def coeff_application(term: AnyReconLossTerm) -> CoeffApplication:
    """Choose whether a term's coefficient scales its loss or its model-side gradients.

    Persistent-source terms use model-side scaling and enter the differentiated
    sum at weight 1, preserving the unscaled source gradient for ascent. Other
    terms multiply the loss scalar directly."""
    trains_sources_from_backward = isinstance(term.sources, PERSISTENT_SOURCE_TYPES)
    return "scales_model_cotangents" if trains_sources_from_backward else "scales_loss"


# ───────────────────────────── the step vocabulary ─────────────────────────────


@dataclass(frozen=True)
class StreamInputs[Out, Conditioning]:
    """Clean observations, CI inputs and target conditioning shared by a data stream.

    `leading` describes the mask/source waist; `sequence` preserves document isolation.
    """

    clean: ForwardObservations[Out]
    taps: dict[str, Array]
    conditioning: Conditioning
    sequence: SequenceLayout | None
    leading: tuple[int, ...]


@dataclass(frozen=True)
class AscendedAdversaries:
    """The ascent phase's outputs: warmed persistent adversaries, each
    fresh-PGD term's ascended sources, and the routing draw each ascent fixed for the
    main grid to reuse (torch parity)."""

    warmed: dict[str, PersistentAdversary]
    fresh_sources: dict[int, Sources]
    fixed_routes: dict[int, Routes]


type DrawLoss[S: MaskSourceStrategy] = Callable[
    [int, ReconLossTerm[S], PRNGKeyArray, Routes], ReconstructionLoss
]
"""`(term_idx, term, draw_key, routes) -> the term's scored recon` — one grid's
per-term dispatcher, built by the factory that owns the grid's trainables. `S` is the
grid's source-strategy width: the non-target grid's dispatcher takes only the enumerated
non-target strategies, so its match is exhaustive over those arms."""


@dataclass(frozen=True)
class TermDraw:
    """One term's forward for the step: its source key and its routing."""

    key: PRNGKeyArray
    routes: Routes


@dataclass(frozen=True)
class ReconGrid[S: MaskSourceStrategy]:
    """One reconstruction grid and its first reserved per-term RNG index."""

    terms: tuple[ReconLossTerm[S], ...]
    key_offset: int

    def __post_init__(self) -> None:
        assert self.terms, "a reconstruction grid must be non-empty"
        assert self.key_offset >= 1, self.key_offset
        assert len({term.name for term in self.terms}) == len(self.terms), (
            "duplicate reconstruction term names"
        )
        self.persistent_by_key()

    def persistent_by_key(self) -> dict[str, ReconLossTerm[S]]:
        persistent: dict[str, ReconLossTerm[S]] = {}
        for term in self.terms:
            match term.sources:
                case (
                    PersistentSources(state_key=state_key)
                    | MixedPersistentStochasticSources(state_key=state_key)
                    | PersistentSourcePool(state_key=state_key)
                ):
                    assert state_key not in persistent, (
                        f"persistent source {state_key!r} feeds multiple terms"
                    )
                    persistent[state_key] = term
                case (
                    StochasticSources()
                    | ConstantSources()
                    | UnmaskedNoDeltaSources()
                    | FreshPGDSources()
                ):
                    pass
        return persistent

    @property
    def capture_keys(self) -> CaptureKeys:
        return frozenset(key for term in self.terms for key in term.capture_keys)

    def reconstruction_specs(
        self, train_frac: Float32[Array, ""]
    ) -> dict[str, AuxiliaryReconstructionSpec]:
        return {
            term.name: tuple(auxiliary.at(train_frac) for auxiliary in term.auxiliaries)
            for term in self.terms
        }

    def adversary_reconstruction_specs(
        self, reconstruction_specs: dict[str, AuxiliaryReconstructionSpec]
    ) -> dict[str, AuxiliaryReconstructionSpec]:
        """Choose each adversary's source-ascent objective independently of the outer loss."""
        return {
            term.name: (
                ()
                if isinstance(term.sources, PERSISTENT_SOURCE_TYPES)
                and term.sources.cfg.adversary_objective == "e2e"
                else reconstruction_specs[term.name]
            )
            for term in self.terms
        }

    @property
    def e2e_terms_requiring_source_grad_retake_by_key(
        self,
    ) -> dict[str, ReconLossTerm[S]]:
        """Persistent e2e terms whose outer loss includes reconstruction auxiliaries."""
        return {
            state_key: term
            for state_key, term in self.persistent_by_key().items()
            if isinstance(term.sources, PERSISTENT_SOURCE_TYPES)
            and term.sources.cfg.adversary_objective == "e2e"
            and term.auxiliaries
        }

    def _term_keys(
        self, key: PRNGKeyArray, term_idx: int, term: ReconLossTerm[S]
    ) -> tuple[PRNGKeyArray, PRNGKeyArray, PRNGKeyArray | None]:
        """Derive disjoint draw, routing, and optional source-pool sampling streams."""
        draw_key, routing_key = random.split(random.fold_in(key, self.key_offset + term_idx))
        match term.sources:
            case PersistentSourcePool():
                draw_key, source_pool_sample_key = random.split(draw_key)
            case (
                StochasticSources()
                | ConstantSources()
                | UnmaskedNoDeltaSources()
                | FreshPGDSources()
                | PersistentSources()
                | MixedPersistentStochasticSources()
            ):
                source_pool_sample_key = None
        return draw_key, routing_key, source_pool_sample_key

    def source_pool_sample_keys(self, key: PRNGKeyArray) -> dict[str, PRNGKeyArray]:
        """One sampling key per source pool, shared by every ascent in the step."""
        keys: dict[str, PRNGKeyArray] = {}
        for term_idx, term in enumerate(self.terms):
            if isinstance(term.sources, PersistentSourcePool):
                _, _, sample_key = self._term_keys(key, term_idx, term)
                assert sample_key is not None
                keys[term.sources.state_key] = sample_key
        return keys

    def draws(
        self,
        key: PRNGKeyArray,
        fixed_routes: dict[int, Routes],
        leading: tuple[int, ...],
    ) -> list[TermDraw]:
        """Materialize every term's key chain."""
        draws: list[TermDraw] = []
        for term_idx, term in enumerate(self.terms):
            draw_key, routing_key, _ = self._term_keys(key, term_idx, term)
            match term.sources:
                case FreshPGDSources():
                    routes = fixed_routes[term_idx]
                case (
                    StochasticSources()
                    | ConstantSources()
                    | UnmaskedNoDeltaSources()
                    | PersistentSources()
                    | MixedPersistentStochasticSources()
                    | PersistentSourcePool()
                ):
                    routes = term.sample_routing(routing_key, leading)
            draws.append(TermDraw(draw_key, routes))
        return draws

    def losses(
        self,
        draws: list[TermDraw],
        draw_loss: DrawLoss[S],
    ) -> tuple[ReconstructionLoss, ...]:
        """Each term's scored reconstruction."""
        return tuple(
            draw_loss(term_idx, term, draw.key, draw.routes)
            for term_idx, (term, draw) in enumerate(zip(self.terms, draws, strict=True))
        )


@dataclass(frozen=True)
class ForwardSubstrate[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
]:
    """Array-free run statics owning forward preparation and VJP scaffolding.

    The model is never stored: every method keeps it as a traced argument, preserving
    the HLO-baking rule. `placement_rules` is the model bundle's own rules, pulled off
    it at `of` — the CI/batch constraints below share the model's placement by
    construction. `recon_loss_fn` and `pin_output_batch` are the target's two output
    operations the substrate composes — pure `@staticmethod`s holding no arrays, so
    they may ride the closed-over statics.
    """

    remat_recon_forwards: bool
    remat_ci_fn: bool
    placement_rules: PlacementRules | None
    ci_capture_keys: CaptureKeys
    recon_loss_fn: Callable[[Out, Out], Array]
    pin_output_batch: Callable[[Out, Mesh | None], Out]

    @property
    def mesh(self) -> Mesh | None:
        """The rules' own mesh — a substrate never carries a second copy to desync. A
        substrate that executes forwards needs a concrete mesh, so the abstract
        (spec-check) arm of `PlacementRules.mesh` is refused here."""
        if self.placement_rules is None:
            return None
        mesh = self.placement_rules.mesh
        assert isinstance(mesh, Mesh), type(mesh)
        return mesh

    @classmethod
    def of(
        cls,
        model_static: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
        *,
        remat_recon_forwards: bool,
        remat_ci_fn: bool,
        ci_capture_keys: CaptureKeys,
    ) -> "ForwardSubstrate[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT]":
        return cls(
            remat_recon_forwards=remat_recon_forwards,
            remat_ci_fn=remat_ci_fn,
            placement_rules=model_static.placement,
            ci_capture_keys=ci_capture_keys,
            recon_loss_fn=model_static.recon_loss_fn,
            pin_output_batch=model_static.pin_output_batch,
        )

    def shard_batch_tree[T](self, x: T) -> T:
        """Pin the leading (batch) axis of every array in the pytree. The batch is an
        opaque protocol edge (`TargetIn` — tokens for an LM, a dict or tuple for another
        target), so this maps over leaves rather than assuming one array."""
        return jax.tree.map(lambda leaf: batch_shard_leading(leaf, self.mesh), x)

    def shard_ci(self, ci: CI) -> CI:
        layout = (
            None if self.placement_rules is None else self.placement_rules.activations.component
        )
        return jax.tree.map(
            lambda value: constrain_component_activation(value, layout),
            ci,
            is_leaf=lambda value: isinstance(value, SelectedCI),
        )

    def component_frequencies(
        self, ci: CI, gamma: Float32[Array, ""], *, normalize_at_one: bool
    ) -> dict[str, Float32[Array, " _"]]:
        frequencies = per_component_frequencies(ci.upper, gamma, normalize_at_one=normalize_at_one)
        if self.placement_rules is None:
            return frequencies
        return jax.sharding.reshard(frequencies, self.placement_rules.frequency_sharding)

    def prep_stream(
        self,
        model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
        batch: TargetIn,
        reconstruction_keys: CaptureKeys,
    ) -> StreamInputs[Out, Conditioning]:
        """Shard one stream's batch, run its detached clean forward, and pull the CI taps +
        recon observations. `reconstruction_keys` is the stream's own union — a stream whose
        grid carries no reconstruction auxiliaries captures none."""
        batch = self.shard_batch_tree(batch)
        with jax.named_scope("pd_clean_fwd_and_taps"):
            clean_forward_result = jax.tree.map(
                jax.lax.stop_gradient,
                model.clean_forward(batch, self.ci_capture_keys | reconstruction_keys),
            )
            taps = select_captures(clean_forward_result.captures, self.ci_capture_keys)
            clean = reconstruction_observations(
                clean_forward_result,
                self.pin_output_batch,
                capture_keys=reconstruction_keys,
                mesh=self.mesh,
            )
        return StreamInputs(
            clean=clean,
            taps=taps,
            conditioning=clean_forward_result.conditioning,
            sequence=clean_forward_result.sequence,
            leading=clean_forward_result.leading_shape,
        )

    def component_weights_vjp(
        self,
        model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
        components: ComponentStacks,
    ) -> tuple[PreparedT, Callable[[PreparedT], tuple[ComponentStacks]]]:
        """The compute-weights value + vjp — the recon gradient's pullback onto V/U."""
        return jax.vjp(lambda c: model.prepare_compute_weights(c), components)

    def ci_fn_prepare_vjp(
        self, ci_fn: CIFn[Conditioning]
    ) -> tuple[CIFn[Conditioning], Callable[[CIFn[Conditioning]], tuple[CIFn[Conditioning]]]]:
        """The resident BF16 CI weights and their pullback onto the FP32 CI masters."""
        return eqx.filter_vjp(lambda cf: cf.prepare(), ci_fn)

    def ci_fn_forward_vjp(
        self,
        compute_ci_fn: CIFn[Conditioning],
        prepared_weights: PreparedT,
        stream: StreamInputs[Out, Conditioning],
    ) -> tuple[CI, Callable[[CI], tuple[CIFn[Conditioning], PreparedT]]]:
        """Evaluate once per stream and pull CI gradients back to its compute weights and
        to the target's prepared components it reads."""

        def forward(ci_fn: CIFn[Conditioning], components: PreparedT) -> CI:
            return self.shard_ci(
                evaluate_ci_from_captures(
                    ci_fn,
                    stream.taps,
                    stream.conditioning,
                    components,
                    sequence=stream.sequence,
                    remat=self.remat_ci_fn,
                )
            )

        with jax.named_scope("pd_ci_fn_fwd"):
            return eqx.filter_vjp(forward, compute_ci_fn, prepared_weights)

    # ONE masked-forward remat policy for recon AND the adversary ascents.
    # `remat_recon_forwards` picks the checkpoint policy of the target's per-block scan:
    # True = `nothing_saveable` — the backward re-forwards one block at a time instead of
    # holding its activations (deep targets need this to fit); False = `dots_saveable` — the
    # backward reads stored batch-scaled activation dots and re-forwards nothing (faster when
    # memory allows; gathered weight operands are never residuals either way — the scanned
    # linear re-derives them in its transpose). This is load-bearing for the ASCENTS too:
    # though they backprop only to the SOURCES (params + CI detached), the source gradient
    # still flows through the per-layer activations (the masks MULTIPLY them), so an
    # un-rematted ascent forward stores `[n_layer, *leading, d_ff]`-scale intermediates.
    @jaxtyped(typechecker=beartype)
    def masked_recon(
        self,
        model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
        *,
        prepared_weights: PreparedT,
        stream: StreamInputs[Out, Conditioning],
        masking: PreparedMaskingT,
        routes: SiteRoutes | None,
        reconstruction: AuxiliaryReconstructionSpec,
    ) -> ReconstructionLoss:
        """Run one masked forward of the stream's batch — pinned to its clean forward's
        decisions — and score its complete recon objective against the stream's clean
        observations, output and auxiliary captures alike."""
        capture_keys = auxiliary_capture_keys(reconstruction)
        masked_forward_result = model.masked_forward(
            prepared_weights,
            stream.conditioning,
            masking=masking,
            routes=routes,
            capture_keys=capture_keys,
            remat=self.remat_recon_forwards,
        )
        masked = reconstruction_observations(
            masked_forward_result,
            self.pin_output_batch,
            capture_keys=capture_keys,
            mesh=self.mesh,
        )
        return reconstruction_loss(
            self.recon_loss_fn,
            masked=masked,
            clean=stream.clean,
            reconstruction=reconstruction,
        )


def ascend_adversaries[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    substrate: ForwardSubstrate[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    grid: ReconGrid[MaskSourceStrategy],
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    stream: StreamInputs[Out, Conditioning],
    detached_prepared_weights: PreparedT,
    ci_lower_detached: Mapping[str, SiteCI],
    adversaries: dict[str, PersistentAdversary],
    key: PRNGKeyArray,
    train_frac: Float32[Array, ""],
    reconstruction_specs: dict[str, AuxiliaryReconstructionSpec],
) -> AscendedAdversaries:
    """Detached adversary ascents for the full-width target/main grid."""

    source_pool_sample_keys = grid.source_pool_sample_keys(key)

    def warmup_scoring_loss(term: AnyReconLossTerm) -> Callable[[SourceStacks], Array]:
        def objective(sources: SourceStacks) -> Array:
            match term.sources:
                case PersistentSourcePool(state_key=state_key):
                    sampled_sources = sample_source_pool(
                        source_pool_sample_keys[state_key], sources, stream.leading
                    )
                case PersistentSources() | MixedPersistentStochasticSources():
                    sampled_sources = sources.per_site()
                case (
                    StochasticSources()
                    | ConstantSources()
                    | UnmaskedNoDeltaSources()
                    | FreshPGDSources()
                ):
                    raise ValueError("Only persistent adversaries have warmup source ascents")
            return substrate.masked_recon(
                model,
                prepared_weights=detached_prepared_weights,
                stream=stream,
                masking=model.model.prepare_masking(
                    source_masking(ci_lower_detached, sampled_sources)
                ),
                routes=None,
                reconstruction=reconstruction_specs[term.name],
            ).total

        return objective

    # the uint16 stochastic-rounding stream: its own fold arm, disjoint from the
    # grid's small key_offset+term_idx folds; dead in the graph for float storage
    quantize_key = random.fold_in(key, 0x51C)
    with jax.named_scope("pd_pgd_warmup_ascend"):
        warmed = {
            state_key: adv.warmup_ascend(
                warmup_scoring_loss(grid.persistent_by_key()[state_key]),
                train_frac,
                random.fold_in(quantize_key, adv_idx),
            )
            for adv_idx, (state_key, adv) in enumerate(adversaries.items())
        }

    fresh_sources: dict[int, Sources] = {}
    fixed_routes: dict[int, Routes] = {}
    for term_idx, term in enumerate(grid.terms):
        if not isinstance(term.sources, FreshPGDSources):
            continue
        fresh_cfg = term.sources
        routing_key, init_key = random.split(random.fold_in(key, grid.key_offset + term_idx))
        routes = term.sample_routing(routing_key, stream.leading)
        fixed_routes[term_idx] = routes
        init = init_fresh_pgd_sources(
            sites=model.model.sites,
            init=fresh_cfg.init,
            source_shape=fresh_cfg.source_shape,
            leading=stream.leading,
            key=init_key,
        )

        def ascent_loss(
            sources: Sources,
            term: AnyReconLossTerm = term,
            routes: Routes = routes,
        ) -> Array:
            return substrate.masked_recon(
                model,
                prepared_weights=detached_prepared_weights,
                stream=stream,
                masking=model.model.prepare_masking(
                    materialize_masking(source_masking(ci_lower_detached, sources))
                ),
                routes=routes,
                reconstruction=reconstruction_specs[term.name],
            ).total

        def sign_ascend_body(
            sources: Sources,
            _: None,
            ascent_loss: Callable[[Sources], Array] = ascent_loss,
            step_size: float = fresh_cfg.step_size,
        ) -> tuple[Sources, None]:
            sources_grad = jax.grad(ascent_loss)(sources)
            return jax.tree.map(
                lambda source, gradient: jnp.clip(
                    source + step_size * jnp.sign(gradient), 0.0, 1.0
                ),
                sources,
                sources_grad,
            ), None

        with jax.named_scope("pd_fresh_pgd_ascend"):
            ascended, _ = jax.lax.scan(sign_ascend_body, init, None, length=fresh_cfg.n_steps)
        fresh_sources[term_idx] = jax.lax.stop_gradient(ascended)

    return AscendedAdversaries(
        warmed=warmed, fresh_sources=fresh_sources, fixed_routes=fixed_routes
    )


def main_draw_loss[TargetIn, Out, PreparedT: ComponentActivations, Conditioning, PreparedMaskingT](
    substrate: ForwardSubstrate[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    *,
    prepared_weights: PreparedT,
    ci: CI,
    draw_stochastic_masking: Callable[[Array], PreparedMaskingT],
    persistent_sources: dict[str, SourceStacks],
    source_pool_sample_keys: dict[str, PRNGKeyArray],
    ascended: AscendedAdversaries,
    stream: StreamInputs[Out, Conditioning],
    train_frac: Float32[Array, ""],
    reconstruction_specs: dict[str, AuxiliaryReconstructionSpec],
) -> DrawLoss[MaskSourceStrategy]:
    """The main grid's per-term dispatcher over the trainables: match the term's
    mask-source strategy, run the masked forward, score against the stream's clean
    observations. Built INSIDE the loss fn — it closes over the live trainables.

    Persistent(-carrying) terms take the coeff on their MODEL-SIDE inputs
    (`model_cotangents_scaled`) and enter the total at weight 1, so the fused
    backward hands each adversary `dL/ds` unscaled."""

    def draw_loss(
        term_idx: int,
        term: ReconLossTerm[MaskSourceStrategy],
        draw_key: PRNGKeyArray,
        routes: Routes,
    ) -> ReconstructionLoss:
        match coeff_application(term):
            case "scales_model_cotangents":
                draw_prepared = model_cotangents_scaled(prepared_weights, term.coeff.at(train_frac))
            case "scales_loss":
                draw_prepared = prepared_weights
        with jax.named_scope("pd_recon_masked_fwd"):
            match term.sources:
                case StochasticSources():
                    masking = draw_stochastic_masking(draw_key)
                case ConstantSources(value=value):
                    masking = model.model.prepare_masking(
                        MaterializedMasking(
                            component_masks={
                                site: map_site_ci(lambda v: v + (1.0 - v) * value, lower)
                                for site, lower in ci.lower.items()
                            },
                            weight_delta_masks=None,
                        )
                    )
                case UnmaskedNoDeltaSources():
                    raise AssertionError(
                        "UnmaskedNoDeltaSources is non-target-pass vocabulary; "
                        "the main grid never carries it"
                    )
                case FreshPGDSources():
                    masking = model.model.prepare_masking(
                        materialize_masking(
                            source_masking(ci.lower, ascended.fresh_sources[term_idx])
                        )
                    )
                case PersistentSources(state_key=state_key):
                    masking = model.model.prepare_masking(
                        source_masking(
                            model_cotangents_scaled(ci.lower, term.coeff.at(train_frac)),
                            persistent_sources[state_key].per_site(),
                        )
                    )
                case MixedPersistentStochasticSources(state_key=state_key):
                    adv_fraction = scheduled_value_at(train_frac, term.sources.cfg.adv_fraction)
                    mixed, routes = mixed_persistent_stochastic_masking(
                        key=draw_key,
                        ci_lower=model_cotangents_scaled(ci.lower, term.coeff.at(train_frac)),
                        persistent_sources=persistent_sources[state_key].per_site(),
                        leading=stream.leading,
                        adv_fraction=adv_fraction,
                        stochastic_routes=routes,
                    )
                    masking = model.model.prepare_masking(mixed)
                case PersistentSourcePool(state_key=state_key):
                    adv_fraction = scheduled_value_at(train_frac, term.sources.cfg.adv_fraction)
                    sampled_sources = sample_source_pool(
                        source_pool_sample_keys[state_key],
                        persistent_sources[state_key],
                        stream.leading,
                    )
                    mixed, routes = mixed_persistent_stochastic_masking(
                        key=draw_key,
                        ci_lower=model_cotangents_scaled(ci.lower, term.coeff.at(train_frac)),
                        persistent_sources=sampled_sources,
                        leading=stream.leading,
                        adv_fraction=adv_fraction,
                        stochastic_routes=routes,
                    )
                    masking = model.model.prepare_masking(mixed)
            return substrate.masked_recon(
                model,
                prepared_weights=draw_prepared,
                stream=stream,
                masking=masking,
                routes=routes,
                reconstruction=reconstruction_specs[term.name],
            )

    return draw_loss


def retake_e2e_source_grads[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    substrate: ForwardSubstrate[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    grid: ReconGrid[MaskSourceStrategy],
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    *,
    prepared_weights: PreparedT,
    ci: CI,
    ascended: AscendedAdversaries,
    stream: StreamInputs[Out, Conditioning],
    draws: list[TermDraw],
    train_frac: Float32[Array, ""],
    warmed_sources: dict[str, SourceStacks],
    source_pool_sample_keys: dict[str, PRNGKeyArray],
) -> dict[str, SourceStacks]:
    """Recompute final persistent-source gradients using output reconstruction only."""
    e2e_terms = grid.e2e_terms_requiring_source_grad_retake_by_key
    if not e2e_terms:
        return {}

    detached_ci = jax.lax.stop_gradient(ci)
    draw_detached_stochastic_masking = model.model.prepare_stochastic_masking(detached_ci.lower)
    term_indices = {term.name: idx for idx, term in enumerate(grid.terms)}
    grads: dict[str, SourceStacks] = {}
    for state_key, term in e2e_terms.items():
        term_idx = term_indices[term.name]

        def e2e_loss(
            sources: SourceStacks,
            state_key: str = state_key,
            term: ReconLossTerm[MaskSourceStrategy] = term,
            term_idx: int = term_idx,
        ) -> Array:
            draw_loss = main_draw_loss(
                substrate,
                model,
                prepared_weights=prepared_weights,
                ci=detached_ci,
                draw_stochastic_masking=draw_detached_stochastic_masking,
                persistent_sources=warmed_sources | {state_key: sources},
                source_pool_sample_keys=source_pool_sample_keys,
                ascended=ascended,
                stream=stream,
                train_frac=train_frac,
                reconstruction_specs={term.name: ()},
            )
            draw = draws[term_idx]
            return draw_loss(term_idx, term, draw.key, draw.routes).total

        with jax.named_scope("pd_pgd_e2e_final_grad"):
            grads[state_key] = jax.grad(e2e_loss)(warmed_sources[state_key])
    return grads


def apply_gradients[Conditioning](
    components_optimizer: ScheduledOptimizer,
    ci_fn_optimizer: ScheduledOptimizer,
    decomposition: Decomposition[Conditioning],
    components_opt_state: ScheduledOptimizerState,
    ci_fn_opt_state: ScheduledOptimizerState,
    warmed_advs: dict[str, PersistentAdversary],
    components_grad: Any,
    ci_fn_grad: Any,
    persistent_source_grads: dict[str, SourceStacks],
    train_frac: Float32[Array, ""],
    final_ascend_key: PRNGKeyArray,
    mesh: Mesh | None,
) -> tuple[
    Decomposition[Conditioning],
    ScheduledOptimizerState,
    ScheduledOptimizerState,
    dict[str, PersistentAdversary],
    dict[str, Array],
]:
    """Record gradient norms, take each adversary's final ascent, and update model state.

    Source gradients are unscaled because each source bundle feeds exactly one term
    whose coefficient applies only to model-side gradients."""
    grad_norm_metrics = _grad_norm_metrics(components_grad, ci_fn_grad, mesh)

    new_adversaries = {
        state_key: warmed_advs[state_key].final_ascend(
            persistent_source_grads[state_key],
            train_frac,
            random.fold_in(final_ascend_key, adv_idx),
        )
        for adv_idx, state_key in enumerate(warmed_advs)
    }

    components_updates, new_components_opt_state = components_optimizer.update(
        components_grad,
        components_opt_state,
        eqx.filter(decomposition.components, eqx.is_array),
    )
    ci_fn_updates, new_ci_fn_opt_state = ci_fn_optimizer.update(
        ci_fn_grad,
        ci_fn_opt_state,
        eqx.filter(decomposition.ci_fn, eqx.is_array),
    )
    new_components = eqx.apply_updates(decomposition.components, components_updates)
    new_ci_fn = eqx.apply_updates(decomposition.ci_fn, ci_fn_updates)

    return (
        Decomposition(components=new_components, ci_fn=new_ci_fn),
        new_components_opt_state,
        new_ci_fn_opt_state,
        new_adversaries,
        grad_norm_metrics,
    )


def _scheduled_coefficient_metrics(
    schedule: RuntimeSchedule, name: str, train_frac: Float32[Array, ""]
) -> dict[str, Float32[Array, ""]]:
    return {} if schedule.points is None else {f"schedules/coeff/{name}": schedule.at(train_frac)}


def _frequency_metrics(penalty: FrequencyPenalty | None) -> dict[str, Float32[Array, ""]]:
    match penalty:
        case None:
            return {"freq": jnp.zeros((), jnp.float32)}
        case BatchFrequencyPenalty():
            return {"freq": penalty.value}
        case EmaFrequencyPenalty():
            return {"freq": penalty.value, "freq_batch": penalty.batch_value}


def shared_step_metrics(
    terms: tuple[ReconLossTerm[MaskSourceStrategy], ...],
    *,
    total_loss: Float32[Array, ""],
    minimality: MinimalityResult,
    term_breakdowns: tuple[ReconstructionLoss, ...],
    grad_norm_metrics: dict[str, Array],
    adversaries: dict[str, PersistentAdversary],
    train_frac: Float32[Array, ""],
) -> dict[str, Array]:
    """Metrics shared by plain and targeted steps; each caller adds its own pass metrics."""
    term_losses = tuple(breakdown.total for breakdown in term_breakdowns)
    metrics = {
        "total": total_loss,
        "gamma_imp": minimality.gamma,
        "imp": minimality.activity,
    }
    dict_safe_update_(metrics, _frequency_metrics(minimality.frequency))
    dict_safe_update_(
        metrics, {f"loss/{t.name}": v for t, v in zip(terms, term_losses, strict=True)}
    )
    dict_safe_update_(metrics, grad_norm_metrics)
    for term, breakdown in zip(terms, term_breakdowns, strict=True):
        dict_safe_update_(
            metrics, _scheduled_coefficient_metrics(term.coeff, term.name, train_frac)
        )
        for auxiliary in term.auxiliaries:
            dict_safe_update_(
                metrics,
                _scheduled_coefficient_metrics(
                    auxiliary.coeff, f"{term.name}/{auxiliary.name}", train_frac
                ),
            )
        prefix = f"loss/{term.name}"
        dict_safe_update_(
            metrics,
            {
                f"{prefix}/{suffix}": value
                for suffix, value in reconstruction_loss_metrics(breakdown).items()
            },
        )
    source_lrs = {k: adv.source_lr(train_frac) for k, adv in adversaries.items()}
    if len(source_lrs) == 1:
        metrics["src_lr"] = next(iter(source_lrs.values()))
    else:
        dict_safe_update_(metrics, {f"schedules/lr/src/{k}": v for k, v in source_lrs.items()})
    return metrics


# ───────────────────────────── the step factory ─────────────────────────────


class PDLossEvaluation(NamedTuple):
    """Numerical losses and the estimator advanced by a plain PD evaluation."""

    reported_loss: Float32[Array, ""]
    faithfulness: Float32[Array, ""]
    minimality: MinimalityResult
    next_frequency: FrequencyEstimator
    nonlinearity: NonlinearityResult | None
    reconstruction: tuple[ReconstructionLoss, ...]


def make_train_step[TargetIn, Out, PreparedT: ComponentActivations, Conditioning, PreparedMaskingT](
    model_static: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    *,
    substrate: ForwardSubstrate[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    components_optimizer: ScheduledOptimizer,
    ci_fn_optimizer: ScheduledOptimizer,
    total_steps: int,
    faithfulness: FaithfulnessLossFn,
):
    """Build the plain VPD step from its forward substrate and objective."""
    assert total_steps > 0, total_steps

    @jaxtyped(typechecker=beartype)
    def step(
        model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
        state: PDState[Conditioning],
        batch: TargetIn,
        key: PRNGKeyArray,
    ) -> tuple[PDState[Conditioning], dict[str, Array]]:
        decomposition = state.decomposition
        training = state.training
        objective = training.objective
        grid = ReconGrid(objective.recon, key_offset=1)
        minimality_term = objective.minimality
        nonlinearity = (
            ResolvedNonlinearity.resolve(objective.nonlinearity, model_static.model.sites)
            if objective.nonlinearity is not None
            else None
        )
        train_frac = train_frac_at(training.step, total_steps)
        reconstruction_specs = grid.reconstruction_specs(train_frac)

        stream = substrate.prep_stream(model, batch, grid.capture_keys)

        # ── adversary ascents: params + CI detached ──
        prepared_weights, recon_vjp = substrate.component_weights_vjp(
            model, decomposition.components
        )
        detached_prepared_weights = jax.lax.stop_gradient(prepared_weights)
        # Ascents reuse the detached CI envelope; the main loss differentiates its
        # live value through the retained CI forward and compute-weight pullbacks.
        compute_ci_fn, ci_fn_prepare_vjp = substrate.ci_fn_prepare_vjp(decomposition.ci_fn)
        ci, ci_fn_forward_vjp = substrate.ci_fn_forward_vjp(compute_ci_fn, prepared_weights, stream)
        ci_lower_detached = jax.lax.stop_gradient(ci).lower

        ascended = ascend_adversaries(
            substrate,
            grid,
            model,
            stream,
            detached_prepared_weights,
            ci_lower_detached,
            training.adversaries,
            key,
            train_frac,
            grid.adversary_reconstruction_specs(reconstruction_specs),
        )

        # ── main losses: live components/ci; the PERSISTENT sources participate in
        # the graph so their gradient comes from the SAME backward; they
        # are NOT detached here, but components/ci grads through them are what torch
        # gets too (sources are leaves). ──
        warmed_sources = {k: a.float_sources for k, a in ascended.warmed.items()}
        draws = grid.draws(key, ascended.fixed_routes, stream.leading)
        source_pool_sample_keys = grid.source_pool_sample_keys(key)

        def loss_fn(
            trainable: tuple[PreparedT, ComponentStacks, CI, dict[str, SourceStacks]],
        ) -> tuple[Float32[Array, ""], PDLossEvaluation]:
            prepared_weights, components, ci, persistent_sources = trainable
            draw_stochastic_masking = model.model.prepare_stochastic_masking(ci.lower)
            # Δ = W − V·U derives from jit parameters alone (frozen W, fp32 masters), so
            # this checkpoint saves no residuals — without it every group's fp32 delta
            # stack persists from this forward to the backward's dV/dU contraction,
            # nearly the whole step. The remat re-runs any entry collectives the
            # placement's delta path carries (the `owner-replicated-resident-moe` rows
            # carry none).
            faith_loss = jax.checkpoint(
                lambda c: faithfulness(faithfulness_weight_deltas(model, c))
            )(components)
            faith_term = objective.faith.coeff.at(train_frac) * faith_loss
            minimality, next_frequency = evaluate_minimality(
                minimality_term, training.frequency, ci, train_frac, substrate.component_frequencies
            )

            draw_loss = main_draw_loss(
                substrate,
                model,
                prepared_weights=prepared_weights,
                ci=ci,
                draw_stochastic_masking=draw_stochastic_masking,
                persistent_sources=persistent_sources,
                source_pool_sample_keys=source_pool_sample_keys,
                ascended=ascended,
                stream=stream,
                train_frac=train_frac,
                reconstruction_specs=reconstruction_specs,
            )
            term_breakdowns = grid.losses(draws, draw_loss)
            term_losses = tuple(breakdown.total for breakdown in term_breakdowns)
            base = faith_term + minimality.weighted_loss
            nonlinearity_result = (
                nonlinearity.evaluate(train_frac, components) if nonlinearity is not None else None
            )
            if nonlinearity_result is not None:
                base = base + nonlinearity_result.weighted_loss
            # The differentiated total: persistent-carrying terms enter at weight 1 —
            # their coeff already rides their model-side cotangents — so the backward
            # hands each adversary dL/ds unscaled. The OBJECTIVE (the
            # reported `total`, Σ coeff·L) has the same gradients up to that plumbing
            # and the identical value for every non-persistent term.
            total_loss = base
            reported_total = base
            for term, term_loss in zip(grid.terms, term_losses, strict=True):
                coeff = term.coeff.at(train_frac)
                match coeff_application(term):
                    case "scales_loss":
                        total_loss = total_loss + coeff * term_loss
                    case "scales_model_cotangents":
                        total_loss = total_loss + term_loss
                reported_total = reported_total + coeff * term_loss
            return total_loss, PDLossEvaluation(
                reported_loss=reported_total,
                faithfulness=faith_loss,
                minimality=minimality,
                next_frequency=next_frequency,
                nonlinearity=nonlinearity_result,
                reconstruction=term_breakdowns,
            )

        with jax.named_scope("pd_value_and_grad"):
            (_, evaluation), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                (prepared_weights, decomposition.components, ci, warmed_sources)
            )
        prepared_grad, components_grad_direct, ci_grad, persistent_source_grads = grads
        compute_ci_fn_grad, prepared_grad_from_ci_fn = ci_fn_forward_vjp(ci_grad)
        (ci_fn_grad,) = ci_fn_prepare_vjp(compute_ci_fn_grad)
        (components_grad_prepared,) = recon_vjp(
            jax.tree.map(
                lambda recon_g, ci_g: recon_g + ci_g, prepared_grad, prepared_grad_from_ci_fn
            )
        )
        # faith and nonlinearity read the components DIRECTLY (weight-space terms, not
        # through the prepared-weights vjp), so the direct grads join the prepared-path grads.
        components_grad = jax.tree.map(
            lambda prepared_g, direct_g: prepared_g + direct_g,
            components_grad_prepared,
            components_grad_direct,
        )
        persistent_source_grads = persistent_source_grads | retake_e2e_source_grads(
            substrate,
            grid,
            model,
            prepared_weights=detached_prepared_weights,
            ci=ci,
            ascended=ascended,
            stream=stream,
            draws=draws,
            train_frac=train_frac,
            warmed_sources=warmed_sources,
            source_pool_sample_keys=source_pool_sample_keys,
        )

        updated, vu_opt_state, ci_fn_opt_state, adversaries, grad_norm_metrics = apply_gradients(
            components_optimizer,
            ci_fn_optimizer,
            decomposition,
            training.components_opt_state,
            training.ci_fn_opt_state,
            ascended.warmed,
            components_grad,
            ci_fn_grad,
            persistent_source_grads,
            train_frac,
            final_ascend_key=random.fold_in(random.fold_in(key, 0x51C), 0x51D),
            mesh=substrate.mesh,
        )
        new_state = TrainState(
            decomposition=updated,
            training=replace(
                training,
                components_opt_state=vu_opt_state,
                ci_fn_opt_state=ci_fn_opt_state,
                adversaries=adversaries,
                frequency=evaluation.next_frequency,
                step=training.step + 1,
            ),
        )
        metrics = shared_step_metrics(
            grid.terms,
            total_loss=evaluation.reported_loss,
            minimality=evaluation.minimality,
            term_breakdowns=evaluation.reconstruction,
            grad_norm_metrics=grad_norm_metrics,
            adversaries=training.adversaries,
            train_frac=train_frac,
        )
        if evaluation.nonlinearity is not None:
            assert nonlinearity is not None
            result = evaluation.nonlinearity
            name = nonlinearity.term.name
            dict_safe_update_(
                metrics,
                {
                    f"loss/{name}": result.loss,
                    "nonlinearity_relative_threshold": result.relative_threshold,
                    **{f"loss/{name}_{kind}": value for kind, value in result.by_kind.items()},
                },
            )
            dict_safe_update_(
                metrics, _scheduled_coefficient_metrics(nonlinearity.term.coeff, name, train_frac)
            )
        dict_safe_update_(
            metrics,
            _scheduled_coefficient_metrics(objective.faith.coeff, objective.faith.name, train_frac),
        )
        dict_safe_update_(
            metrics,
            _scheduled_coefficient_metrics(
                minimality_term.activity_coeff, minimality_term.name, train_frac
            ),
        )
        if minimality_term.frequency is not None:
            dict_safe_update_(
                metrics,
                _scheduled_coefficient_metrics(
                    minimality_term.frequency.coeff, f"{minimality_term.name}/frequency", train_frac
                ),
            )
        metrics["faith"] = evaluation.faithfulness
        return new_state, metrics

    return step


# ───────────────────────────── the targeted (tPD) step factory ─────────────────────────────


class CIScaledWeightDecay(eqx.Module):
    """Post-optimizer decay scaled by the step's CI maxima and applied learning rate."""

    coeff: Float32[Array, ""]

    def __check_init__(self) -> None:
        assert self.coeff.shape == () and self.coeff.dtype == jnp.float32

    def apply[Conditioning](
        self,
        state: TargetedPDState[Conditioning],
        target_ci: CI,
        nontarget_ci: CI,
        learning_rate: Float32[Array, ""],
        sites: tuple[SiteSpec, ...],
    ) -> tuple[TargetedPDState[Conditioning], dict[str, Array]]:
        """Apply post-optimizer weight decay using this step's pre-update CI maxima."""
        target_max = _per_component_batch_max(target_ci.lower)
        nontarget_max = _per_component_batch_max(nontarget_ci.lower)
        rate = learning_rate * self.coeff
        decay = {
            spec.name: rate * (1.0 - jnp.maximum(target_max[spec.name], nontarget_max[spec.name]))
            for spec in sites
        }
        decayed = _scale_subcomponents(
            state.decomposition.components,
            {site: 1.0 - value for site, value in decay.items()},
            {group: vu_group.factorization for group, vu_group in vu_groups(sites).items()},
        )
        new_state = TrainState(
            decomposition=Decomposition(components=decayed, ci_fn=state.decomposition.ci_fn),
            training=state.training,
        )
        # Scalar reductions per site, then combined: the per-site [C] vectors carry
        # heterogeneous shardings (tp-sharded full emission, replicated segment-max), so
        # a concatenation is untypeable where these scalars are not.
        n_total = sum(spec.C for spec in sites)
        decay_sum = sum((jnp.sum(value) for value in decay.values()), start=jnp.zeros(()))
        decay_max = functools.reduce(jnp.maximum, (jnp.max(value) for value in decay.values()))
        return new_state, {
            "ci_scaled_weight_decay/mean": decay_sum / n_total,
            "ci_scaled_weight_decay/max": decay_max,
        }


def _per_component_batch_max(ci_lower: dict[str, SiteCI]) -> dict[str, Array]:
    """Each site's per-subcomponent max CI over every leading (batch AND position) axis,
    fp32. Reads `lower` deliberately: `lower ≡ clip(upper, 0, 1)` pointwise, so the
    two squashings agree on this statistic and no clamp is needed. A selected-emitting
    site takes the exact segment-max (`selected_component_maxes`): an unselected
    component's CI is zero by definition, so a never-important component's max stays 0
    and CI-scaled weight decay drags it at the full rate."""

    def site_max(v: SiteCI) -> Array:
        match v:
            case SelectedCI():
                return selected_component_maxes(v, v.values)
            case jax.Array():
                return jnp.max(v.astype(jnp.float32), axis=tuple(range(v.ndim - 1)))

    return {site: site_max(v) for site, v in ci_lower.items()}


def _factor_like(factor: Array, leaf: Array, expand_axes: tuple[int, ...]) -> Array:
    """`factor` expanded with size-1 axes at `expand_axes` and resharded to the leaf's
    own spec on the axes they share — the scale multiply must be sharding-typed like the
    master it scales (the CI-derived factor arrives on activation shardings, the master
    on its persistence layout). Unplaced (empty-mesh) execution broadcasts as-is."""
    expanded = jnp.expand_dims(factor, expand_axes)
    if jax.sharding.get_abstract_mesh().empty:
        return expanded
    spec = tuple(jax.typeof(leaf).sharding.spec)
    spec += (None,) * (leaf.ndim - len(spec))
    return jax.sharding.reshard(
        expanded, P(*(None if i in expand_axes else spec[i] for i in range(leaf.ndim)))
    )


def _scale_subcomponents(
    components: ComponentStacks,
    scale: dict[str, Float[Array, " C"]],
    factorization_by_group: dict[str, Factorization],
) -> ComponentStacks:
    """Scale each site's V columns and U rows by that site's per-subcomponent factor,
    stacked per semantic group so the multiply stays in the declared layout. The flat
    `[C]` factor addresses a block-factored group's `[g, E, d, c]` leaves through the
    block-major `C = E·c` ordering."""
    rows_by_group: dict[str, list[Array]] = {}
    for name, group, index in components.site_stack_indices:
        rows = rows_by_group.setdefault(group, [])
        assert index == len(rows), (name, group, index)
        rows.append(scale[name])
    stacks = {}
    for group, (vs, us) in components.stacks.items():
        keep = jnp.stack(rows_by_group[group])  # [g, C]
        if pad := components.pad_of(group):
            # Persist-stack pads scale by 0 — they are exactly zero and stay so.
            keep = jnp.concatenate([keep, jnp.zeros((pad, *keep.shape[1:]), keep.dtype)])
        match factorization_by_group[group]:
            case DenseFactorization():
                stacks[group] = (
                    vs * _factor_like(keep, vs, (1,)),
                    us * _factor_like(keep, us, (2,)),
                )
            case BlockedFactorization(n_blocks=n_blocks, c_per_block=c_per_block):
                blocked = keep.reshape(keep.shape[0], n_blocks, c_per_block)
                stacks[group] = (
                    vs * _factor_like(blocked, vs, (2,)),
                    us * _factor_like(blocked, us, (3,)),
                )
    return ComponentStacks(
        stacks=stacks,
        site_stack_indices=components.site_stack_indices,
        stack_pads=components.stack_pads,
    )


class StreamLossResult(NamedTuple):
    reported_loss: Float32[Array, ""]
    minimality: MinimalityResult
    reconstruction: tuple[ReconstructionLoss, ...]


class TargetedLossEvaluation(NamedTuple):
    target: StreamLossResult
    nontarget: StreamLossResult
    next_target_frequency: FrequencyEstimator
    next_nontarget_frequency: FrequencyEstimator


def make_targeted_train_step[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model_static: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    *,
    substrate: ForwardSubstrate[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    components_optimizer: ScheduledOptimizer,
    ci_fn_optimizer: ScheduledOptimizer,
    total_steps: int,
):
    """Build the hand-written tPD two-stream step from its substrate and objective."""
    assert total_steps > 0, total_steps

    def nontarget_draw_loss(
        model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
        prepared_weights: PreparedT,
        nt_ci: CI,
        nt_stream: StreamInputs[Out, Conditioning],
        nt_reconstruction_specs: dict[str, AuxiliaryReconstructionSpec],
    ) -> DrawLoss[StochasticSources | ConstantSources | UnmaskedNoDeltaSources]:
        """The non-target grid's per-term dispatcher: every delta mask pinned to 1.0 —
        except the unmasked-no-delta arm, which carries no delta — scored
        against the broad stream's frozen output."""

        def draw_loss(
            term_idx: int,
            term: ReconLossTerm[StochasticSources | ConstantSources | UnmaskedNoDeltaSources],
            draw_key: PRNGKeyArray,
            routes: Routes,
        ) -> ReconstructionLoss:
            del term_idx
            with jax.named_scope("pd_nontarget_masked_fwd"):
                match term.sources:
                    case StochasticSources():
                        masking = stochastic_delta_pinned_masking(nt_ci.lower, draw_key)
                    case ConstantSources(value=value):
                        masking = constant_delta_pinned_masking(value, nt_ci.lower)
                    case UnmaskedNoDeltaSources():
                        masking = unmasked_no_delta_masking(nt_ci.lower)
                return substrate.masked_recon(
                    model,
                    prepared_weights=prepared_weights,
                    stream=nt_stream,
                    masking=model.model.prepare_masking(masking),
                    routes=routes,
                    reconstruction=nt_reconstruction_specs[term.name],
                )

        return draw_loss

    @jaxtyped(typechecker=beartype)
    def targeted_step(
        model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
        state: TargetedPDState[Conditioning],
        batch: TargetIn,
        nontarget_batch: TargetIn,
        key: PRNGKeyArray,
    ) -> tuple[TargetedPDState[Conditioning], dict[str, Array]]:
        decomposition = state.decomposition
        training = state.training
        objective = training.objective
        target = ReconGrid(objective.target.recon, key_offset=1)
        nontarget = ReconGrid(objective.nontarget.recon, key_offset=1 + len(target.terms))
        assert not nontarget.capture_keys, nontarget.capture_keys
        for term in nontarget.terms:
            assert isinstance(
                term.sources, StochasticSources | ConstantSources | UnmaskedNoDeltaSources
            ), term.name
        minimality_term = objective.target.minimality
        train_frac = train_frac_at(training.step, total_steps)
        reconstruction_specs = target.reconstruction_specs(train_frac)
        nt_reconstruction_specs = nontarget.reconstruction_specs(train_frac)

        stream = substrate.prep_stream(model, batch, target.capture_keys)
        nt_stream = substrate.prep_stream(model, nontarget_batch, nontarget.capture_keys)

        # ── adversary ascents: TARGET pass only, params + CI detached ──
        prepared_weights, recon_vjp = substrate.component_weights_vjp(
            model, decomposition.components
        )
        detached_prepared_weights = jax.lax.stop_gradient(prepared_weights)
        compute_ci_fn, ci_fn_prepare_vjp = substrate.ci_fn_prepare_vjp(decomposition.ci_fn)
        ci, ci_fn_forward_vjp = substrate.ci_fn_forward_vjp(compute_ci_fn, prepared_weights, stream)
        nt_ci, nt_ci_fn_forward_vjp = substrate.ci_fn_forward_vjp(
            compute_ci_fn, prepared_weights, nt_stream
        )
        ci_lower_detached = jax.lax.stop_gradient(ci).lower

        ascended = ascend_adversaries(
            substrate,
            target,
            model,
            stream,
            detached_prepared_weights,
            ci_lower_detached,
            training.adversaries,
            key,
            train_frac,
            target.adversary_reconstruction_specs(reconstruction_specs),
        )

        warmed_sources = {k: a.float_sources for k, a in ascended.warmed.items()}
        draws = target.draws(key, ascended.fixed_routes, stream.leading)
        source_pool_sample_keys = target.source_pool_sample_keys(key)
        # The non-target grid's per-term RNG offsets past the target grid's, so the two
        # grids' draws stay disjoint under the one step key.
        nt_draws = nontarget.draws(key, {}, nt_stream.leading)

        def loss_fn(
            trainable: tuple[PreparedT, CI, CI, dict[str, SourceStacks]],
        ) -> tuple[Float32[Array, ""], TargetedLossEvaluation]:
            prepared_weights, ci, nt_ci, persistent_sources = trainable
            draw_stochastic_masking = model.model.prepare_stochastic_masking(ci.lower)
            minimality, next_target_frequency = evaluate_minimality(
                minimality_term,
                training.target_frequency,
                ci,
                train_frac,
                substrate.component_frequencies,
            )
            nt_minimality, next_nontarget_frequency = evaluate_minimality(
                objective.nontarget.minimality,
                training.nontarget_frequency,
                nt_ci,
                train_frac,
                substrate.component_frequencies,
            )

            draw_loss = main_draw_loss(
                substrate,
                model,
                prepared_weights=prepared_weights,
                ci=ci,
                draw_stochastic_masking=draw_stochastic_masking,
                persistent_sources=persistent_sources,
                source_pool_sample_keys=source_pool_sample_keys,
                ascended=ascended,
                stream=stream,
                train_frac=train_frac,
                reconstruction_specs=reconstruction_specs,
            )
            term_breakdowns = target.losses(draws, draw_loss)
            base = minimality.weighted_loss
            # Differentiated total vs reported total: see the plain factory — a
            # persistent-carrying term's coeff rides its model-side cotangents, so it
            # enters the total at weight 1 and its adversary receives dL/ds.
            total_loss = base
            reported_total = base
            for term, breakdown in zip(target.terms, term_breakdowns, strict=True):
                coeff = term.coeff.at(train_frac)
                match coeff_application(term):
                    case "scales_loss":
                        total_loss = total_loss + coeff * breakdown.total
                    case "scales_model_cotangents":
                        total_loss = total_loss + breakdown.total
                reported_total = reported_total + coeff * breakdown.total

            # ── the non-target pass: its minimality (the shared annealed param, its own
            # coeff) + its delta-pinned grid, added to the SAME total so one backward
            # grads both passes. ──
            nt_total = nt_minimality.weighted_loss
            nt_breakdowns = nontarget.losses(
                nt_draws,
                nontarget_draw_loss(
                    model, prepared_weights, nt_ci, nt_stream, nt_reconstruction_specs
                ),
            )
            for term, breakdown in zip(nontarget.terms, nt_breakdowns, strict=True):
                nt_total = nt_total + term.coeff.at(train_frac) * breakdown.total
            total_loss = total_loss + nt_total
            return total_loss, TargetedLossEvaluation(
                target=StreamLossResult(reported_total, minimality, term_breakdowns),
                nontarget=StreamLossResult(nt_total, nt_minimality, nt_breakdowns),
                next_target_frequency=next_target_frequency,
                next_nontarget_frequency=next_nontarget_frequency,
            )

        with jax.named_scope("pd_value_and_grad"):
            (_, evaluation), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                (prepared_weights, ci, nt_ci, warmed_sources)
            )
        prepared_grad, ci_grad, nt_ci_grad, persistent_source_grads = grads
        compute_ci_fn_grad, prepared_grad_from_ci_fn = ci_fn_forward_vjp(ci_grad)
        nt_compute_ci_fn_grad, nt_prepared_grad_from_ci_fn = nt_ci_fn_forward_vjp(nt_ci_grad)
        (ci_fn_grad,) = ci_fn_prepare_vjp(
            jax.tree.map(
                lambda target_g, nt_g: target_g + nt_g, compute_ci_fn_grad, nt_compute_ci_fn_grad
            )
        )
        (components_grad,) = recon_vjp(
            jax.tree.map(
                lambda recon_g, ci_g, nt_ci_g: recon_g + ci_g + nt_ci_g,
                prepared_grad,
                prepared_grad_from_ci_fn,
                nt_prepared_grad_from_ci_fn,
            )
        )
        persistent_source_grads = persistent_source_grads | retake_e2e_source_grads(
            substrate,
            target,
            model,
            prepared_weights=detached_prepared_weights,
            ci=ci,
            ascended=ascended,
            stream=stream,
            draws=draws,
            train_frac=train_frac,
            warmed_sources=warmed_sources,
            source_pool_sample_keys=source_pool_sample_keys,
        )

        updated, vu_opt_state, ci_fn_opt_state, adversaries, grad_norm_metrics = apply_gradients(
            components_optimizer,
            ci_fn_optimizer,
            decomposition,
            training.components_opt_state,
            training.ci_fn_opt_state,
            ascended.warmed,
            components_grad,
            ci_fn_grad,
            persistent_source_grads,
            train_frac,
            final_ascend_key=random.fold_in(random.fold_in(key, 0x51C), 0x51D),
            mesh=substrate.mesh,
        )
        new_state = TrainState(
            decomposition=updated,
            training=replace(
                training,
                components_opt_state=vu_opt_state,
                ci_fn_opt_state=ci_fn_opt_state,
                adversaries=adversaries,
                target_frequency=evaluation.next_target_frequency,
                nontarget_frequency=evaluation.next_nontarget_frequency,
                step=training.step + 1,
            ),
        )
        wd_metrics: dict[str, Array] = {}
        if training.ci_scaled_weight_decay is not None:
            new_state, wd_metrics = training.ci_scaled_weight_decay.apply(
                new_state,
                ci,
                nt_ci,
                new_state.training.components_opt_state.applied_learning_rate,
                model_static.model.sites,
            )
        metrics = shared_step_metrics(
            target.terms,
            total_loss=evaluation.target.reported_loss + evaluation.nontarget.reported_loss,
            minimality=evaluation.target.minimality,
            term_breakdowns=evaluation.target.reconstruction,
            grad_norm_metrics=grad_norm_metrics,
            adversaries=training.adversaries,
            train_frac=train_frac,
        )
        dict_safe_update_(
            metrics,
            {
                "loss/nontarget/total": evaluation.nontarget.reported_loss,
                "loss/nontarget/imp": evaluation.nontarget.minimality.activity,
            },
        )
        dict_safe_update_(
            metrics,
            {
                f"loss/nontarget/{name}": value
                for name, value in _frequency_metrics(
                    evaluation.nontarget.minimality.frequency
                ).items()
            },
        )
        nt_minimality_term = objective.nontarget.minimality
        dict_safe_update_(
            metrics,
            _scheduled_coefficient_metrics(
                nt_minimality_term.activity_coeff, nt_minimality_term.name, train_frac
            ),
        )
        for term, result in zip(nontarget.terms, evaluation.nontarget.reconstruction, strict=True):
            dict_safe_update_(metrics, {f"loss/nontarget/{term.name}": result.total})
            dict_safe_update_(
                metrics,
                _scheduled_coefficient_metrics(term.coeff, f"nontarget/{term.name}", train_frac),
            )
        dict_safe_update_(metrics, wd_metrics)
        dict_safe_update_(
            metrics,
            _scheduled_coefficient_metrics(
                minimality_term.activity_coeff, minimality_term.name, train_frac
            ),
        )
        if minimality_term.frequency is not None:
            dict_safe_update_(
                metrics,
                _scheduled_coefficient_metrics(
                    minimality_term.frequency.coeff, f"{minimality_term.name}/frequency", train_frac
                ),
            )
        return new_state, metrics

    return targeted_step


# ───────────────────────────── faithfulness warmup ─────────────────────────────


type FaithWarmupStep[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
] = Callable[
    [
        PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
        ComponentStacks,
        optax.OptState,
    ],
    tuple[ComponentStacks, optax.OptState, Array],
]


def make_faith_warmup_step[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    opt: optax.GradientTransformation,
    faithfulness: FaithfulnessLossFn,
) -> FaithWarmupStep[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT]:
    """`model` is the jit ARG (frozen weights traced, not baked) — `weight_deltas` reads its
    per-site W slices, so closing over the model would bake them into the HLO."""

    def warmup_step(
        model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
        components: ComponentStacks,
        opt_state: optax.OptState,
    ) -> tuple[ComponentStacks, optax.OptState, Array]:
        def loss_fn(components_: ComponentStacks) -> Array:
            return faithfulness(faithfulness_weight_deltas(model, components_))

        loss, grad = eqx.filter_value_and_grad(loss_fn)(components)
        updates, opt_state = opt.update(grad, opt_state, eqx.filter(components, eqx.is_array))
        return eqx.apply_updates(components, updates), opt_state, loss

    return warmup_step
