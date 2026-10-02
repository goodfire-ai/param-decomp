"""Recon terms for the VPD objective.

A recon term is one forward over ALL the model's sites: a routing sampler (which sites'
masks apply at each position) crossed with a mask-source strategy. This module knows
nothing about the objective's faithfulness or importance terms; `objective.py` composes
those with the recon terms into the complete loss surface.
"""

from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax
from jax import random
from jax.sharding import Mesh
from jaxtyping import Array, Float32, PRNGKeyArray

from param_decomp.core.configs import (
    AllRoutingConfig,
    AuxiliaryReconstructionConfig,
    BatchSourceShape,
    CaptureReconstruction,
    MergedStochasticSubsetPooledPPGDReconLossConfig,
    MergedStochasticSubsetPPGDReconLossConfig,
    PersistentPGDReconLossConfig,
    PGDInitStrategy,
    StaticProbabilityRoutingConfig,
    SubsetRoutingType,
    UniformKSubsetRoutingConfig,
)
from param_decomp.core.model import (
    CaptureKeys,
    ForwardResult,
    SiteRoutes,
    select_captures,
)
from param_decomp.core.runtime_schedule import RuntimeSchedule

Routes = SiteRoutes | None
type RoutingSampler = UniformKRouting | StaticProbabilityRouting | AllRouting


# ───────────────────────────── mask-source strategies ─────────────────────────────


@dataclass(frozen=True)
class StochasticSources:
    """Fresh sources each step: components `U[0,1]`, delta `U[0,1]`."""


@dataclass(frozen=True)
class ConstantSources:
    """`mask = ci + (1-ci)*value`: 0.0 = CI-masked, 1.0 = unmasked. No delta
    (LOSS_PARITY_DESIGN §4b)."""

    value: float


@dataclass(frozen=True)
class UnmaskedNoDeltaSources:
    """Non-target reconstruction using all components and no weight delta.

    Component masks are 1 and there is no delta. Delta polarity is carried by this
    type, not by a flag on the delta-on strategies."""


@dataclass(frozen=True)
class FreshPGDSources:
    """Per-step sign-PGD-ascended sources (torch `PGDRecon*` as TRAINING losses): init
    per `init`, `n_steps` of `step_size * sign(grad)` with clamp to [0,1], no state
    across steps. The entry's routing is drawn ONCE per step and shared by every
    ascent and the final loss forward (torch parity)."""

    init: PGDInitStrategy
    n_steps: int
    step_size: float
    source_shape: BatchSourceShape


@dataclass(frozen=True)
class PersistentSources:
    """Sources living in `state.training.adversaries[state_key]` across steps (PPGD). Carries
    the shared `PersistentPGDReconLossConfig` so the term is self-describing — the step
    reads its scope/optimizer/warmup straight off `cfg`. `state_key` indexes
    `state.training.adversaries` (one key per persistent term)."""

    state_key: str
    cfg: PersistentPGDReconLossConfig


@dataclass(frozen=True)
class MixedPersistentStochasticSources:
    """The merged stochastic+PPGD strategy: per batch element, the persistent bundle's
    sources (probability `cfg.adv_fraction`, routed all-live) or fresh `U[0,1]` (routed
    per the entry's sampler). `state_key` indexes `state.training.adversaries` like
    `PersistentSources`."""

    state_key: str
    cfg: "MergedStochasticSubsetPPGDReconLossConfig"


@dataclass(frozen=True)
class PersistentSourcePool:
    """One persistent minipool of cross-site particles per batch index.

    Each batch element samples from its own particles and shares the draw across
    sites and positions. ``state_key`` identifies the stored pool and optimizer.
    """

    state_key: str
    cfg: MergedStochasticSubsetPooledPPGDReconLossConfig


MaskSourceStrategy = (
    StochasticSources
    | ConstantSources
    | UnmaskedNoDeltaSources
    | FreshPGDSources
    | PersistentSources
    | MixedPersistentStochasticSources
    | PersistentSourcePool
)


@dataclass(frozen=True)
class AuxiliaryReconstruction:
    name: str
    coeff: Float32[Array, ""] | float
    comparisons: tuple[CaptureReconstruction, ...]


class AuxiliaryReconstructionTerm(eqx.Module):
    name: str = eqx.field(static=True)
    coeff: RuntimeSchedule
    comparisons: tuple[CaptureReconstruction, ...] = eqx.field(static=True)

    def at(self, train_frac: Float32[Array, ""]) -> AuxiliaryReconstruction:
        return AuxiliaryReconstruction(self.name, self.coeff.at(train_frac), self.comparisons)


type AuxiliaryReconstructionSpec = tuple[AuxiliaryReconstruction, ...]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ForwardObservations[Out]:
    """The output and named activations one reconstruction comparison consumes."""

    output: Out
    captures: dict[str, Array]


def reconstruction_observations[Out, Conditioning](
    result: ForwardResult[Out, Conditioning],
    pin_output_batch: Callable[[Out, Mesh | None], Out],
    *,
    capture_keys: CaptureKeys,
    mesh: Mesh | None,
) -> ForwardObservations[Out]:
    """Convert one forward result into the exact view a reconstruction consumes;
    `pin_output_batch` is the target's (`DecomposedModel.pin_output_batch`)."""
    return ForwardObservations(
        pin_output_batch(result.output, mesh),
        select_captures(result.captures, capture_keys),
    )


def resolve_auxiliary_reconstruction(
    configs: tuple[AuxiliaryReconstructionConfig, ...],
) -> AuxiliaryReconstructionSpec:
    """Resolve constant eval coefficients; schedules require a training step."""
    auxiliaries: list[AuxiliaryReconstruction] = []
    for config in configs:
        assert isinstance(config.coeff, float), (
            "an eval probe's auxiliary coeff must be a constant float"
        )
        auxiliaries.append(AuxiliaryReconstruction(config.name, config.coeff, config.comparisons))
    return tuple(auxiliaries)


def auxiliary_capture_keys(reconstruction: AuxiliaryReconstructionSpec) -> CaptureKeys:
    return frozenset(
        comparison.capture for auxiliary in reconstruction for comparison in auxiliary.comparisons
    )


class ReconLossTerm[SourcesT: MaskSourceStrategy](eqx.Module):
    """One weighted reconstruction term with one masked forward per training step.

    `sample_routing` selects sites per position; `sources` supplies mask values.
    The target model defines the output comparison, and `auxiliaries` adds named
    activation comparisons. Coefficients are resolved from schedules in the step.
    `SourcesT` restricts the allowed strategies, and `name` gives the `loss/<name>`
    metric key."""

    name: str = eqx.field(static=True)
    coeff: RuntimeSchedule
    sample_routing: RoutingSampler = eqx.field(static=True)
    sources: SourcesT = eqx.field(static=True)
    auxiliaries: tuple[AuxiliaryReconstructionTerm, ...]

    @property
    def capture_keys(self) -> CaptureKeys:
        return frozenset(
            comparison.capture
            for auxiliary in self.auxiliaries
            for comparison in auxiliary.comparisons
        )


AnyReconLossTerm = ReconLossTerm[MaskSourceStrategy]
"""A reconstruction term with any admitted source strategy."""


# ───────────────────────────── routing samplers ─────────────────────────────


@dataclass(frozen=True)
class UniformKRouting:
    """Draw a uniform nonempty subset of sites independently at each position."""

    sites: tuple[str, ...]

    def __call__(self, key: PRNGKeyArray, leading_shape: tuple[int, ...]) -> Routes:
        n_sites = len(self.sites)
        k_key, perm_key = random.split(key)
        k = random.randint(k_key, leading_shape, 1, n_sites + 1)
        perms = random.uniform(perm_key, (n_sites, *leading_shape)).argsort(axis=0)
        routed = perms < k
        return {name: routed[j] for j, name in enumerate(self.sites)}


@dataclass(frozen=True)
class StaticProbabilityRouting:
    """Route each position to each site independently with probability `p`."""

    sites: tuple[str, ...]
    p: float

    def __call__(self, key: PRNGKeyArray, leading_shape: tuple[int, ...]) -> Routes:
        return {
            name: random.bernoulli(random.fold_in(key, j), self.p, leading_shape)
            for j, name in enumerate(self.sites)
        }


@dataclass(frozen=True)
class AllRouting:
    """Route every position to every site."""

    def __call__(self, key: PRNGKeyArray, leading_shape: tuple[int, ...]) -> Routes:
        return None


def routing_sampler_from_config(
    routing: SubsetRoutingType, sites: tuple[str, ...]
) -> RoutingSampler:
    match routing:
        case UniformKSubsetRoutingConfig():
            return UniformKRouting(sites)
        case StaticProbabilityRoutingConfig():
            return StaticProbabilityRouting(sites, routing.p)
        case AllRoutingConfig():
            return AllRouting()


# ───────────────────────────── shared-config -> flat terms ─────────────────────────────


type AnyPersistentLossConfig = (
    PersistentPGDReconLossConfig
    | MergedStochasticSubsetPPGDReconLossConfig
    | MergedStochasticSubsetPooledPPGDReconLossConfig
)

PERSISTENT_SOURCE_TYPES = (
    PersistentSources,
    MixedPersistentStochasticSources,
    PersistentSourcePool,
)


def persistent_configs(
    recon_terms: tuple[AnyReconLossTerm, ...],
) -> dict[str, AnyPersistentLossConfig]:
    """``state_key -> config`` for every persistent-source-carrying recon term."""
    out: dict[str, AnyPersistentLossConfig] = {}
    for term in recon_terms:
        if isinstance(term.sources, PERSISTENT_SOURCE_TYPES):
            assert term.sources.state_key not in out, term.sources.state_key
            out[term.sources.state_key] = term.sources.cfg
    return out
