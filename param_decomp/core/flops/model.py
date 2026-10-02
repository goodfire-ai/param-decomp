"""Useful matrix-contraction FLOPs for one parameter-decomposition training step.

A multiply-add counts as two FLOPs. These counts exclude rematerialization,
optimizer updates, communication, elementwise operations and reductions. Each
stream's costs cover its entire global batch; shared CI computation is paid once.
"""

import math
from dataclasses import dataclass

from param_decomp.core.components import BlockedFactorization, DenseFactorization, SiteSpec
from param_decomp.core.configs import (
    AllRoutingConfig,
    AnyReconLossMetricConfig,
    CIMaskedReconLossConfig,
    CIMaskedReconSubsetLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    MergedStochasticSubsetPooledPPGDReconLossConfig,
    MergedStochasticSubsetPPGDReconLossConfig,
    NonlinearityLocalityLossConfig,
    NontargetConfig,
    PDConfig,
    PersistentPGDReconLossConfig,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    StaticProbabilityRoutingConfig,
    StochasticReconLossConfig,
    StochasticReconSubsetLossConfig,
    TargetedPDConfig,
    UniformKSubsetRoutingConfig,
    UnmaskedNoDeltaReconLossConfig,
    UnmaskedReconLossConfig,
)
from param_decomp.core.flops.types import ForwardBackwardFlops
from param_decomp.core.schedule import ScheduleConfig, get_scheduled_value


@dataclass(frozen=True)
class StreamFlops:
    """Shared stream work and independently differentiated reconstructions."""

    clean_forward: int
    ci: ForwardBackwardFlops
    reconstructions: tuple["FlopsTerm", ...]

    def __post_init__(self) -> None:
        if self.clean_forward < 0:
            raise ValueError("Clean-forward FLOPs must be nonnegative")


@dataclass(frozen=True)
class FlopsTerm:
    name: str
    flops: ForwardBackwardFlops
    n_repetitions: float

    def __post_init__(self) -> None:
        if not self.name or not math.isfinite(self.n_repetitions) or self.n_repetitions <= 0:
            raise ValueError("A FLOP term needs a name and a positive repetition count")

    @property
    def total(self) -> float:
        return self.n_repetitions * self.flops.total


@dataclass(frozen=True)
class TrainingFlops:
    terms: tuple[FlopsTerm, ...]

    def __post_init__(self) -> None:
        names = tuple(term.name for term in self.terms)
        if len(set(names)) != len(names):
            raise ValueError("FLOP term names must be unique")

    @property
    def forward(self) -> float:
        return sum(term.n_repetitions * term.flops.forward for term in self.terms)

    @property
    def backward(self) -> float:
        return sum(term.n_repetitions * term.flops.backward for term in self.terms)

    @property
    def total(self) -> float:
        return self.forward + self.backward


def faithfulness_flops(sites: tuple[SiteSpec, ...]) -> ForwardBackwardFlops:
    """Use the cheaper direct residual or Gram formulation of squared weight error.

    The Gram form uses ||VU-W||² = tr(VᵀV UUᵀ) - 2<V,WUᵀ> + ||W||².
    Its symmetric Grams and WUᵀ are reused in the two factor gradients.
    """
    forward = backward = 0
    for site in sites:
        match site.factorization:
            case DenseFactorization(d_in=d_in, d_out=d_out, C=n_components):
                n_copies = 1
            case BlockedFactorization(
                d_in=d_in, d_out=d_out, c_per_block=n_components, n_blocks=n_copies
            ):
                pass
        direct = ForwardBackwardFlops(
            2 * d_in * n_components * d_out, 4 * d_in * n_components * d_out
        )
        gram = ForwardBackwardFlops(
            (d_in + d_out) * n_components * (n_components + 1) + 2 * d_in * d_out * n_components,
            2 * (d_in + d_out) * n_components**2 + 2 * d_in * d_out * n_components,
        )
        cost = min((direct, gram), key=lambda cost: cost.total)
        forward += n_copies * cost.forward
        backward += n_copies * cost.backward
    return ForwardBackwardFlops(forward, backward)


def nonlinearity_flops(
    loss: NonlinearityLocalityLossConfig, sites: tuple[SiteSpec, ...]
) -> ForwardBackwardFlops:
    """Partition norms are segmented reductions, requiring no matrix contractions."""
    declared_kinds = {
        site.alignment.partition.unit_kind for site in sites if site.alignment is not None
    }
    if not declared_kinds or loss.unit_kind_coefficients.keys() != declared_kinds:
        raise ValueError("Nonlinearity coefficients must name the target's partitioned kinds")
    return ForwardBackwardFlops(0, 0)


@dataclass(frozen=True)
class ReconstructionPlan:
    name: str
    include_frozen_paths: bool
    auxiliary_captures: frozenset[str]
    source_captures: frozenset[str]
    n_ascent_steps: int
    retake_source_gradient: bool
    source_gradient_fraction: float


def _coefficient(value: float | ScheduleConfig, step: int, n_steps: int) -> float:
    match value:
        case ScheduleConfig():
            return get_scheduled_value(step, n_steps, value)
        case float() | int():
            return value


def reconstruction_plan(
    loss: AnyReconLossMetricConfig, step: int, n_steps: int
) -> ReconstructionPlan:
    """Separate objective requirements from target-specific contraction counts."""
    captures = frozenset(
        comparison.capture
        for auxiliary in loss.auxiliaries
        if _coefficient(auxiliary.coeff, step, n_steps) > 0
        for comparison in auxiliary.comparisons
    )
    frozen_paths, n_ascent_steps, source_captures, retake = True, 0, captures, False
    match loss:
        case CIMaskedReconLossConfig() | UnmaskedReconLossConfig():
            frozen_paths = False
        case CIMaskedReconSubsetLossConfig(routing=routing):
            match routing:
                case AllRoutingConfig():
                    frozen_paths = False
                case UniformKSubsetRoutingConfig() | StaticProbabilityRoutingConfig():
                    pass
        case StochasticReconLossConfig() | StochasticReconSubsetLossConfig():
            pass
        case (
            PGDReconLossConfig(n_steps=n_ascent_steps)
            | PGDReconSubsetLossConfig(n_steps=n_ascent_steps)
        ):
            pass
        case (
            PersistentPGDReconLossConfig()
            | MergedStochasticSubsetPPGDReconLossConfig()
            | MergedStochasticSubsetPooledPPGDReconLossConfig()
        ):
            n_ascent_steps = loss.n_warmup_steps
            match loss.adversary_objective:
                case "e2e":
                    source_captures = frozenset()
                    retake = bool(captures)
                case "term":
                    pass
    match loss:
        case (
            MergedStochasticSubsetPPGDReconLossConfig()
            | MergedStochasticSubsetPooledPPGDReconLossConfig()
        ):
            fraction = get_scheduled_value(step, n_steps, loss.adv_fraction)
        case (
            CIMaskedReconLossConfig()
            | CIMaskedReconSubsetLossConfig()
            | StochasticReconLossConfig()
            | StochasticReconSubsetLossConfig()
            | UnmaskedReconLossConfig()
            | PGDReconLossConfig()
            | PGDReconSubsetLossConfig()
            | PersistentPGDReconLossConfig()
        ):
            fraction = 1.0
    return ReconstructionPlan(
        loss.name or loss.type,
        frozen_paths,
        captures,
        source_captures,
        n_ascent_steps,
        retake,
        fraction,
    )


def reconstruction_plans(
    pd: PDConfig | TargetedPDConfig, step: int
) -> tuple[ReconstructionPlan, ...]:
    plans = []
    for loss in pd.loss_metrics:
        match loss:
            case (
                FaithfulnessLossConfig()
                | ImportanceMinimalityLossConfig()
                | NonlinearityLocalityLossConfig()
            ):
                pass
            case (
                CIMaskedReconLossConfig()
                | CIMaskedReconSubsetLossConfig()
                | StochasticReconLossConfig()
                | StochasticReconSubsetLossConfig()
                | UnmaskedReconLossConfig()
                | PGDReconLossConfig()
                | PGDReconSubsetLossConfig()
                | PersistentPGDReconLossConfig()
                | MergedStochasticSubsetPPGDReconLossConfig()
                | MergedStochasticSubsetPooledPPGDReconLossConfig()
            ):
                plans.append(reconstruction_plan(loss, step, pd.steps))
    return tuple(plans)


def nontarget_reconstruction_plans(nontarget: NontargetConfig) -> tuple[ReconstructionPlan, ...]:
    plans = []
    for loss in nontarget.recon:
        match loss:
            case UnmaskedNoDeltaReconLossConfig():
                frozen_paths = False
            case (
                CIMaskedReconLossConfig()
                | CIMaskedReconSubsetLossConfig()
                | StochasticReconLossConfig()
                | StochasticReconSubsetLossConfig()
            ):
                frozen_paths = True
        plans.append(
            ReconstructionPlan(
                loss.name or loss.type, frozen_paths, frozenset(), frozenset(), 0, False, 1.0
            )
        )
    return tuple(plans)


def _shared_stream_terms(stream: StreamFlops, prefix: str) -> tuple[FlopsTerm, ...]:
    return (
        FlopsTerm(f"{prefix}clean", ForwardBackwardFlops(stream.clean_forward, 0), 1),
        FlopsTerm(f"{prefix}ci", stream.ci, 1),
    )


def decomposition_step_flops(
    pd: PDConfig, sites: tuple[SiteSpec, ...], stream: StreamFlops
) -> TrainingFlops:
    """Count a main training step, separate from faithfulness-only warmup steps."""
    terms = [*_shared_stream_terms(stream, ""), *stream.reconstructions]
    for loss in pd.loss_metrics:
        match loss:
            case FaithfulnessLossConfig():
                terms.append(FlopsTerm("faithfulness", faithfulness_flops(sites), 1))
            case ImportanceMinimalityLossConfig():
                pass
            case NonlinearityLocalityLossConfig():
                terms.append(FlopsTerm("nonlinearity", nonlinearity_flops(loss, sites), 1))
            case (
                CIMaskedReconLossConfig()
                | CIMaskedReconSubsetLossConfig()
                | StochasticReconLossConfig()
                | StochasticReconSubsetLossConfig()
                | UnmaskedReconLossConfig()
                | PGDReconLossConfig()
                | PGDReconSubsetLossConfig()
                | PersistentPGDReconLossConfig()
                | MergedStochasticSubsetPPGDReconLossConfig()
                | MergedStochasticSubsetPooledPPGDReconLossConfig()
            ):
                pass
    return TrainingFlops(tuple(terms))


def targeted_step_flops(
    target_stream: StreamFlops,
    nontarget_stream: StreamFlops,
) -> TrainingFlops:
    """Count both streams; their CI backwards share weights but each executes once."""
    return TrainingFlops(
        (
            *_shared_stream_terms(target_stream, "target/"),
            *target_stream.reconstructions,
            *_shared_stream_terms(nontarget_stream, "nontarget/"),
            *nontarget_stream.reconstructions,
        )
    )
