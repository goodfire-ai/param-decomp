"""The closed VPD training objectives — plain and targeted.

Authored loss metrics become explicit objective roles: the plain objective is exactly one
faithfulness term, one CI-minimality term, a non-empty ordered tuple of recon
terms, and at most one nonlinearity-locality term; the targeted (tPD)
objective is a faithfulness-free target-pass surface plus a directly-authored
non-target pass (delta-pinned recon + minimality with its own activity coefficient).
The recon vocabulary (routing samplers, mask-source strategies) lives in `recon.py`;
this module alone composes it with the other objective roles.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float32

from param_decomp.core.ci_fn.interface import CI
from param_decomp.core.components import ComponentStacks, SiteSpec, nonlinearity_alignments
from param_decomp.core.configs import (
    AllRoutingConfig,
    AnyLossMetricConfig,
    AnyReconLossMetricConfig,
    CIMaskedReconLossConfig,
    CIMaskedReconSubsetLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    LossCoeff,
    MergedStochasticSubsetPooledPPGDReconLossConfig,
    MergedStochasticSubsetPPGDReconLossConfig,
    NonlinearityLocalityLossConfig,
    NontargetConfig,
    NontargetReconLossMetricConfig,
    PersistentPGDReconLossConfig,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    StochasticReconLossConfig,
    StochasticReconSubsetLossConfig,
    SubsetRoutingType,
    TargetedLossMetricConfig,
    UnmaskedNoDeltaReconLossConfig,
    UnmaskedReconLossConfig,
)
from param_decomp.core.losses import (
    FrequencyEstimator,
    FrequencyPenalty,
    activity_sum,
    activity_sum_from_ci,
    nonlinearity_loss,
)
from param_decomp.core.nonlinearity import NonlinearityAlignment, NonlinearityUnitKind
from param_decomp.core.recon import (
    AnyReconLossTerm,
    AuxiliaryReconstructionTerm,
    ConstantSources,
    FreshPGDSources,
    MaskSourceStrategy,
    MixedPersistentStochasticSources,
    PersistentSourcePool,
    PersistentSources,
    ReconLossTerm,
    StochasticSources,
    UnmaskedNoDeltaSources,
    routing_sampler_from_config,
)
from param_decomp.core.runtime_schedule import RuntimeSchedule, scheduled_value_at
from param_decomp.core.schedule import ScheduleConfig


class FaithfulnessTerm(eqx.Module):
    name: str = eqx.field(static=True)
    coeff: RuntimeSchedule


class ComponentFrequencies(Protocol):
    def __call__(
        self, ci: CI, gamma: Float32[Array, ""], *, normalize_at_one: bool
    ) -> dict[str, Float32[Array, " _"]]: ...


class MinimalityResult(NamedTuple):
    weighted_loss: Float32[Array, ""]
    activity: Float32[Array, ""]
    frequency: FrequencyPenalty | None
    gamma: Float32[Array, ""]


class FrequencyMinimalityTerm(eqx.Module):
    coeff: RuntimeSchedule
    reference_datapoint_count: int = eqx.field(static=True)


class MinimalityTerm(eqx.Module):
    """CI activity and an optional component-frequency penalty."""

    name: str = eqx.field(static=True)
    activity_coeff: RuntimeSchedule
    gamma: ScheduleConfig = eqx.field(static=True)
    normalize_at_one: bool = eqx.field(static=True)
    frequency: FrequencyMinimalityTerm | None


def evaluate_minimality(
    term: MinimalityTerm,
    estimator: FrequencyEstimator,
    ci: CI,
    train_frac: Float32[Array, ""],
    component_frequencies: ComponentFrequencies,
) -> tuple[MinimalityResult, FrequencyEstimator]:
    gamma = scheduled_value_at(train_frac, term.gamma)
    match term.frequency:
        case None:
            # Selected CI needs no component scatter when only the activity is read.
            activity = activity_sum_from_ci(ci.upper, gamma, normalize_at_one=term.normalize_at_one)
            frequency = None
            updated_estimator = estimator
            frequency_loss = jnp.zeros((), jnp.float32)
        case FrequencyMinimalityTerm() as frequency_term:
            frequencies = component_frequencies(ci, gamma, normalize_at_one=term.normalize_at_one)
            activity = activity_sum(frequencies)
            frequency, updated_estimator = estimator.evaluate(
                frequencies, frequency_term.reference_datapoint_count
            )
            frequency_loss = frequency_term.coeff.at(train_frac) * frequency.value
    return MinimalityResult(
        term.activity_coeff.at(train_frac) * activity + frequency_loss, activity, frequency, gamma
    ), updated_estimator


class NonlinearityResult(NamedTuple):
    weighted_loss: Float32[Array, ""]
    loss: Float32[Array, ""]
    by_kind: dict[NonlinearityUnitKind, Float32[Array, ""]]
    relative_threshold: Float32[Array, ""]


class NonlinearityTerm(eqx.Module):
    """Weight-space concentration over declared nonlinearity-facing units."""

    name: str = eqx.field(static=True)
    coeff: RuntimeSchedule
    relative_threshold: ScheduleConfig = eqx.field(static=True)
    unit_kind_coefficients: dict[NonlinearityUnitKind, float | None] = eqx.field(static=True)
    normalize_at_one: bool = eqx.field(static=True)


@dataclass(frozen=True)
class ResolvedNonlinearity:
    """A `NonlinearityTerm` joined with the target's declared partitions: the closed
    unit-kind check happens at `resolve`, once, and `None`-weighted kinds are filtered
    out here so no loss math ever sees an excluded kind."""

    term: NonlinearityTerm
    trained_alignments: dict[str, NonlinearityAlignment]
    kind_coefficients: dict[NonlinearityUnitKind, float]

    @staticmethod
    def resolve(term: NonlinearityTerm, sites: tuple[SiteSpec, ...]) -> "ResolvedNonlinearity":
        alignments = nonlinearity_alignments(sites)
        assert alignments, "NonlinearityLocalityLoss needs a partitioned site"
        declared_kinds = {a.partition.unit_kind for a in alignments.values()}
        authored = term.unit_kind_coefficients
        assert authored.keys() == declared_kinds, (
            f"unit_kind_coefficients must name exactly the target's partitioned kinds: "
            f"authored {sorted(authored)}, declared {sorted(declared_kinds)}"
        )
        kind_coefficients: dict[NonlinearityUnitKind, float] = {
            kind: w for kind, w in authored.items() if w is not None
        }
        return ResolvedNonlinearity(
            term,
            {
                name: alignment
                for name, alignment in alignments.items()
                if alignment.partition.unit_kind in kind_coefficients
            },
            kind_coefficients,
        )

    def evaluate(
        self, train_frac: Float32[Array, ""], components: ComponentStacks
    ) -> NonlinearityResult:
        threshold = scheduled_value_at(train_frac, self.term.relative_threshold)
        value, by_kind = nonlinearity_loss(
            components,
            self.trained_alignments,
            threshold,
            self.kind_coefficients,
            normalize_at_one=self.term.normalize_at_one,
        )
        return NonlinearityResult(self.term.coeff.at(train_frac) * value, value, by_kind, threshold)


class PDObjective(eqx.Module):
    """Faithfulness, CI minimality, reconstruction, and optional nonlinearity losses."""

    faith: FaithfulnessTerm
    minimality: MinimalityTerm
    recon: tuple[AnyReconLossTerm, ...]
    nonlinearity: NonlinearityTerm | None


class TargetPass(eqx.Module):
    """Target-stream reconstruction and CI-minimality losses.

    Faithfulness is excluded so the delta remains free to carry off-target behavior.
    Delta sources follow plain VPD: stochastic terms sample uniformly on [0, 1],
    and adversarial terms ascend and project their delta channel over [0, 1].
    This makes reconstruction from components alone an adversarial worst case
    while also covering hybrid ablations of components and delta; pinning the
    target delta off would omit those hybrids."""

    minimality: MinimalityTerm
    recon: tuple[AnyReconLossTerm, ...]


class NontargetPass(eqx.Module):
    """Broad-stream reconstruction and CI-minimality terms.

    Stochastic and constant-source terms keep the weight delta on; the unmasked
    term keeps it off and scores the full component sum alone. Minimality uses the
    target pass's penalty configuration with this pass's own activity coefficient."""

    recon: tuple[ReconLossTerm[StochasticSources | ConstantSources | UnmaskedNoDeltaSources], ...]
    """Non-target strategies exclude adversarial and mixed sources."""
    minimality: MinimalityTerm


class TargetedPDObjective(eqx.Module):
    """The complete two-pass tPD objective; both passes sum into ONE backward."""

    target: TargetPass
    nontarget: NontargetPass


def _minimality_term(
    cfg: ImportanceMinimalityLossConfig, name: str, activity_coeff: LossCoeff
) -> MinimalityTerm:
    assert all(k.frac > 0 for k in cfg.gamma.points), (
        f"gamma knots must all keep frac > 0, got {cfg.gamma.points}: a zero "
        "width collapses the smooth-L0 threshold band the gradient lives on"
    )
    return MinimalityTerm(
        name,
        RuntimeSchedule.from_coeff(activity_coeff),
        cfg.gamma,
        cfg.normalize_at_one,
        FrequencyMinimalityTerm(
            RuntimeSchedule.from_coeff(cfg.frequency.coeff),
            cfg.frequency.reference_datapoint_count,
        )
        if cfg.frequency is not None
        else None,
    )


def _collect_terms(
    loss_metrics: Sequence[AnyLossMetricConfig],
    sites: tuple[SiteSpec, ...],
) -> tuple[
    FaithfulnessTerm | None,
    MinimalityTerm | None,
    tuple[AnyReconLossTerm, ...],
    NonlinearityTerm | None,
]:
    """One pass over an authored loss list into its objective roles, names unique across
    all roles. Completeness (which roles must be present) is each objective builder's own
    claim, not this walk's.

    Recon-term order follows the authored list and is semantically load-bearing: per-term
    RNG keys derive from the recon index.
    """
    faith: FaithfulnessTerm | None = None
    minimality: MinimalityTerm | None = None
    recon_terms: list[AnyReconLossTerm] = []
    nonlinearity: NonlinearityTerm | None = None

    def unique_name(cfg: AnyLossMetricConfig) -> str:
        # Only committed terms are in `taken`, so persistent terms may call this once for
        # their state key and again inside `recon` without colliding with themselves.
        name = cfg.name if cfg.name is not None else cfg.type
        taken = {term.name for term in recon_terms}
        if faith is not None:
            taken.add(faith.name)
        if minimality is not None:
            taken.add(minimality.name)
        if nonlinearity is not None:
            taken.add(nonlinearity.name)
        assert name not in taken, f"duplicate loss instance_key {name!r}"
        return name

    def recon(
        cfg: AnyReconLossMetricConfig,
        routing: SubsetRoutingType,
        sources: MaskSourceStrategy,
    ) -> AnyReconLossTerm:
        # `sources` sits in parameter position, so the term is built width-erased
        # directly (storage is width-erased; narrower widths are the builders' concern).
        return ReconLossTerm(
            unique_name(cfg),
            RuntimeSchedule.from_coeff(cfg.coeff),
            routing_sampler_from_config(routing, tuple(site.name for site in sites)),
            sources,
            tuple(
                AuxiliaryReconstructionTerm(
                    auxiliary.name,
                    RuntimeSchedule.from_coeff(auxiliary.coeff),
                    auxiliary.comparisons,
                )
                for auxiliary in cfg.auxiliaries
            ),
        )

    for cfg in loss_metrics:
        match cfg:
            case FaithfulnessLossConfig():
                assert faith is None
                faith = FaithfulnessTerm(unique_name(cfg), RuntimeSchedule.from_coeff(cfg.coeff))
            case ImportanceMinimalityLossConfig():
                assert minimality is None
                minimality = _minimality_term(cfg, unique_name(cfg), cfg.coeff)
            case UnmaskedReconLossConfig():
                recon_terms.append(recon(cfg, AllRoutingConfig(), ConstantSources(1.0)))
            case (
                CIMaskedReconLossConfig()
                | CIMaskedReconSubsetLossConfig()
                | StochasticReconLossConfig()
                | StochasticReconSubsetLossConfig()
            ):
                routing, sources = _nontarget_recon_parts(cfg)
                recon_terms.append(recon(cfg, routing, sources))
            case PGDReconLossConfig() | PGDReconSubsetLossConfig():
                sources = FreshPGDSources(cfg.init, cfg.n_steps, cfg.step_size, cfg.source_shape)
                routing = (
                    cfg.routing if isinstance(cfg, PGDReconSubsetLossConfig) else AllRoutingConfig()
                )
                recon_terms.append(recon(cfg, routing, sources))
            case MergedStochasticSubsetPPGDReconLossConfig():
                key = unique_name(cfg)
                sources = MixedPersistentStochasticSources(state_key=key, cfg=cfg)
                recon_terms.append(recon(cfg, cfg.routing, sources))
            case MergedStochasticSubsetPooledPPGDReconLossConfig():
                key = unique_name(cfg)
                sources = PersistentSourcePool(state_key=key, cfg=cfg)
                recon_terms.append(recon(cfg, cfg.routing, sources))
            case PersistentPGDReconLossConfig():
                key = unique_name(cfg)
                sources = PersistentSources(state_key=key, cfg=cfg)
                recon_terms.append(recon(cfg, AllRoutingConfig(), sources))
            case NonlinearityLocalityLossConfig():
                assert nonlinearity is None
                nonlinearity = NonlinearityTerm(
                    unique_name(cfg),
                    RuntimeSchedule.from_coeff(cfg.coeff),
                    cfg.relative_threshold,
                    cfg.unit_kind_coefficients,
                    cfg.normalize_at_one,
                )

    return faith, minimality, tuple(recon_terms), nonlinearity


def build_objective(
    loss_metrics: Sequence[AnyLossMetricConfig],
    sites: tuple[SiteSpec, ...],
) -> PDObjective:
    """Build the closed plain-VPD objective, rejecting incomplete authored surfaces."""
    faith, minimality, recon_terms, nonlinearity = _collect_terms(loss_metrics, sites)
    assert faith is not None and minimality is not None, (
        f"need FaithfulnessLoss + ImportanceMinimalityLoss, got {[m.type for m in loss_metrics]}"
    )
    assert recon_terms, "no recon loss terms configured"
    return PDObjective(faith, minimality, recon_terms, nonlinearity)


def build_recon_terms(
    loss_metrics: Sequence[AnyLossMetricConfig],
    sites: tuple[SiteSpec, ...],
) -> tuple[AnyReconLossTerm, ...]:
    """Just the recon Σ of an authored loss list — the persistent-source layout derives
    from these (`recon.persistent_configs`), so state init shares this walk with both
    objective builders instead of demanding one builder's completeness rules."""
    return _collect_terms(loss_metrics, sites)[2]


def build_targeted_objective(
    loss_metrics: Sequence[TargetedLossMetricConfig],
    nontarget: NontargetConfig,
    sites: tuple[SiteSpec, ...],
) -> TargetedPDObjective:
    """Build the closed two-pass tPD objective.

    `loss_metrics` authors the TARGET pass — typed by `TargetedLossMetricConfig`, which
    has no faithfulness member (the delta is the unpenalized off-target escape valve,
    so a targeted config cannot spell a faithfulness role). The non-target pass is
    authored directly on `nontarget` — never derived from the target list — and its
    minimality shares the target's penalty config (shape + anneal) by construction,
    with the non-target pass's own activity coefficient."""
    faith, minimality, recon_terms, nonlinearity = _collect_terms(loss_metrics, sites)
    # The library boundary for lists built outside the schema; unreachable for a parsed
    # TargetedPDConfig.
    assert faith is None, "a targeted loss list carried a FaithfulnessLossConfig"
    assert nonlinearity is None, "a targeted loss list carried a NonlinearityLocalityLossConfig"
    assert minimality is not None, (
        f"need a ImportanceMinimalityLoss, got {[m.type for m in loss_metrics]}"
    )
    assert recon_terms, "no recon loss terms configured"

    (minimality_config,) = (
        cfg for cfg in loss_metrics if isinstance(cfg, ImportanceMinimalityLossConfig)
    )
    nt_minimality = _minimality_term(minimality_config, "nontarget/impmin", nontarget.impmin_coeff)
    nt_terms: list[ReconLossTerm[StochasticSources | ConstantSources | UnmaskedNoDeltaSources]] = []
    for cfg in nontarget.recon:
        name = cfg.name if cfg.name is not None else cfg.type
        assert name not in {t.name for t in nt_terms}, f"duplicate non-target loss {name!r}"
        routing, sources = _nontarget_recon_parts(cfg)
        nt_terms.append(
            ReconLossTerm(
                name,
                RuntimeSchedule.from_coeff(cfg.coeff),
                routing_sampler_from_config(routing, tuple(site.name for site in sites)),
                sources,
                (),
            )
        )
    return TargetedPDObjective(
        target=TargetPass(minimality=minimality, recon=recon_terms),
        nontarget=NontargetPass(minimality=nt_minimality, recon=tuple(nt_terms)),
    )


def _nontarget_recon_parts(
    cfg: NontargetReconLossMetricConfig,
) -> tuple[SubsetRoutingType, StochasticSources | ConstantSources | UnmaskedNoDeltaSources]:
    """Resolve a non-target reconstruction config into its routing and source strategy."""
    match cfg:
        case CIMaskedReconLossConfig():
            return AllRoutingConfig(), ConstantSources(0.0)
        case CIMaskedReconSubsetLossConfig():
            return cfg.routing, ConstantSources(0.0)
        case StochasticReconLossConfig():
            return AllRoutingConfig(), StochasticSources()
        case StochasticReconSubsetLossConfig():
            return cfg.routing, StochasticSources()
        case UnmaskedNoDeltaReconLossConfig():
            return AllRoutingConfig(), UnmaskedNoDeltaSources()
