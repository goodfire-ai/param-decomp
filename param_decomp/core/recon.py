"""Recon terms for the VPD objective.

A recon term is one forward family over ALL the model's sites: a routing sampler (which
sites' masks apply at each position, per draw) crossed with a mask-source strategy. This
module knows nothing about the objective's faithfulness or importance terms; `objective.py`
composes those with the recon terms into the complete loss surface.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
from jax import random
from jax.sharding import Mesh
from jaxtyping import Array, Float, PRNGKeyArray

from param_decomp.core.configs import (
    AllRoutingConfig,
    HiddenActsReconstruction,
    LossCoeff,
    MergedStochasticSubsetPPGDReconLossConfig,
    PersistentPGDReconLossConfig,
    PGDInitStrategy,
    SourceShape,
    StaticProbabilityRoutingConfig,
    SubsetRoutingType,
    UniformKSubsetRoutingConfig,
)
from param_decomp.core.model import (
    CaptureKeys,
    ForwardResult,
    select_captures,
)
from param_decomp.core.sharding import batch_shard_leading

Routes = dict[str, Array] | None
RoutingSampler = Callable[[PRNGKeyArray, tuple[int, ...]], tuple[Routes, ...]]
"""`(key, leading_shape) -> (routes, ...)` — a STATICALLY-sized family of routing draws,
each `{site: bool[*leading]}` (or None = route everywhere) becoming ONE forward. The torch
`Router.get_masks` made pure: fresh draws per step require the key threaded in —
samplers run INSIDE the jitted step, so they must be traceable (SPEC R1). Returning
several draws from one invocation enables JOINTLY-sampled families (independent
repeats, antithetic/complementary subsets, per-step random covers) that independent
per-draw keys alone cannot express. The term's structure — sampler identity, family
size, strategy kind — is static; only the key varies per step."""


# ───────────────────────────── mask-source strategies ─────────────────────────────


@dataclass(frozen=True)
class StochasticSources:
    """Fresh per-draw sources: components `U[0,1]`, delta `U[0,1]`."""


@dataclass(frozen=True)
class ConstantSources:
    """`mask = ci + (1-ci)*value`: 0.0 = CI-masked, 1.0 = unmasked. `delta_mask = 0`
    (torch passes no delta path at all for these; multiplying by zero is
    mathematically identical, it just pays the delta matmul)."""

    value: float


@dataclass(frozen=True)
class UnmaskedNoDeltaSources:
    """Every component mask `1.0`, every weight-delta mask `0.0` — the full component sum
    alone reconstructs. The tPD non-target pass's one delta-OFF arm (SPEC T4's enumerated
    exception): the polarity rides this type, never a flag on the delta-pinned strategies.
    Deterministic — no sources are drawn."""


@dataclass(frozen=True)
class FreshPGDSources:
    """Per-step sign-PGD-ascended sources (torch `PGDRecon*` as TRAINING losses): init
    per `init`, `n_steps` of `step_size * sign(grad)` with clamp to [0,1], no state
    across steps. The entry's routing is drawn ONCE per step and shared by every
    ascent and the final loss forward (SPEC S24, torch parity)."""

    init: PGDInitStrategy
    n_steps: int
    step_size: float
    source_shape: SourceShape


@dataclass(frozen=True)
class PersistentSources:
    """Sources living in `TrainState.adversaries[state_key]` across steps (PPGD). Carries
    the shared `PersistentPGDReconLossConfig` so the term is self-describing — the step
    reads its scope/optimizer/warmup straight off `cfg`. `state_key` indexes
    `TrainState.adversaries` (one key per persistent term, SPEC S23)."""

    state_key: str
    cfg: PersistentPGDReconLossConfig


@dataclass(frozen=True)
class MixedPersistentStochasticSources:
    """The merged stochastic+PPGD strategy: per batch element, the persistent bundle's
    sources (probability `cfg.adv_fraction`, routed all-live) or fresh `U[0,1]` (routed
    per the entry's sampler). `state_key` indexes `TrainState.adversaries` like
    `PersistentSources`."""

    state_key: str
    cfg: "MergedStochasticSubsetPPGDReconLossConfig"


MaskSourceStrategy = (
    StochasticSources
    | ConstantSources
    | UnmaskedNoDeltaSources
    | FreshPGDSources
    | PersistentSources
    | MixedPersistentStochasticSources
)


@dataclass(frozen=True)
class OutputOnlyReconstruction:
    """A reconstruction specification that compares only the model output."""


@dataclass(frozen=True)
class OutputAndHiddenActsReconstruction:
    """A reconstruction specification that also compares named hidden activations.

    Value-level: `coeff` is the hidden-activation reconstruction strength AT one step —
    a literal for static contexts (eval probes), a traced scalar when the step resolved a schedule
    (`losses.reconstruction_spec_at`) — never a schedule object, so the loss math
    downstream stays step-blind."""

    coeff: Float[Array, ""] | float
    points: tuple[str, ...]


type ReconstructionSpec = OutputOnlyReconstruction | OutputAndHiddenActsReconstruction


class ForwardObservations(NamedTuple):
    """The output and named activations one reconstruction comparison consumes."""

    output: Any
    hidden_acts_by_point: dict[str, Array]


def reconstruction_observations(
    result: ForwardResult,
    *,
    hidden_acts_capture_keys: CaptureKeys,
    mesh: Mesh | None,
) -> ForwardObservations:
    """Convert one forward result into the exact view a reconstruction consumes."""
    output = jax.tree.map(lambda value: batch_shard_leading(value, mesh), result.output)
    return ForwardObservations(
        output,
        select_captures(result.captures, hidden_acts_capture_keys),
    )


def resolve_reconstruction_spec(
    hidden_acts_reconstruction: HiddenActsReconstruction | None,
) -> ReconstructionSpec:
    """Resolve the authored optional into one explicit reconstruction specification —
    STATIC contexts only (eval probes): a scheduled hidden-activation reconstruction
    coefficient needs the step threaded
    in (`losses.reconstruction_spec_at`), which an eval probe deliberately lacks."""
    if hidden_acts_reconstruction is None:
        return OutputOnlyReconstruction()
    coeff = hidden_acts_reconstruction.coeff
    assert isinstance(coeff, float), (
        "an eval probe's hidden-activation reconstruction coeff must be a constant "
        f"float, got {coeff}"
    )
    return OutputAndHiddenActsReconstruction(coeff, hidden_acts_reconstruction.points)


def hidden_acts_capture_keys(reconstruction: ReconstructionSpec) -> CaptureKeys:
    """Return the hidden activations required before evaluating this specification."""
    match reconstruction:
        case OutputOnlyReconstruction():
            return frozenset()
        case OutputAndHiddenActsReconstruction(points=points):
            return frozenset(points)


@dataclass(frozen=True)
class ReconLossTerm[SourcesT: MaskSourceStrategy]:
    """One coefficiented recon loss: mean over its routing draws of `kl_per_position`
    (SPEC S10'). Every draw runs all sites decomposed; `sample_routing` produces the
    term's statically-sized family of draws and `sources` generates each draw's
    mask/delta sources. `SourcesT` narrows which strategies a term can carry — the tPD
    non-target pass admits only the enumerated non-target strategies IN THE TYPE (SPEC
    T5). `name` is the config's `instance_key` — the metric log key is `loss/<name>`.
    `coeff` and the S35 hidden-activation reconstruction coeff may be schedules, so the
    term stays a static description; the step resolves both to per-step values (`losses.coeff_at` /
    `losses.reconstruction_spec_at`)."""

    name: str
    coeff: LossCoeff
    sample_routing: RoutingSampler
    sources: SourcesT
    hidden_acts_reconstruction: HiddenActsReconstruction | None

    @property
    def uses_weight_deltas(self) -> bool:
        """`ConstantSources` carries no delta path (torch passes no `weight_deltas` for the
        Unmasked/CIMasked losses); its `delta_mask` would be a constant 0, so the `x @ Δ`
        matmul is skipped entirely (static, retrace-safe — LOSS_PARITY_DESIGN §4b).
        `UnmaskedNoDeltaSources` carries a materialized delta mask pinned to 0 (SPEC T4's
        exception); every other strategy drives a live delta mask."""
        return not isinstance(self.sources, ConstantSources)

    @property
    def hidden_acts_capture_keys(self) -> CaptureKeys:
        match self.hidden_acts_reconstruction:
            case None:
                return frozenset()
            case HiddenActsReconstruction(points=points):
                return frozenset(points)


AnyReconLossTerm = ReconLossTerm[MaskSourceStrategy]
"""The width-erased term — what heterogeneous storage (the plain objective and ``ReconGrid``) holds. Machinery that must preserve a narrower width is generic over `SourcesT` instead
(dataclass type params are invariant on 3.13: the synthesized `__replace__` puts them in
parameter position)."""


# ───────────────────────────── routing samplers ─────────────────────────────


def uniform_k_subset_routes(
    key: PRNGKeyArray, sites: tuple[str, ...], leading_shape: tuple[int, ...]
) -> dict[str, Array]:
    """Per position: `k ~ U{1..|sites|}`, then a uniform k-subset of the sites
    routes True (SPEC S11). Distributionally identical to torch's double-argsort ranks."""
    n_sites = len(sites)
    k_key, perm_key = random.split(key)
    k = random.randint(k_key, leading_shape, 1, n_sites + 1)
    perms = random.uniform(perm_key, (n_sites, *leading_shape)).argsort(axis=0)
    routed = perms < k
    return {name: routed[j] for j, name in enumerate(sites)}


def uniform_k_routing(sites: tuple[str, ...], n_draws: int) -> RoutingSampler:
    """`n_draws` independent per-position uniform-k-subset draws over `sites`."""

    def sample(key: PRNGKeyArray, leading_shape: tuple[int, ...]) -> tuple[Routes, ...]:
        return tuple(
            uniform_k_subset_routes(draw_key, sites, leading_shape)
            for draw_key in random.split(key, n_draws)
        )

    return sample


def static_probability_routing(sites: tuple[str, ...], p: float, n_draws: int) -> RoutingSampler:
    """`n_draws` independent draws routing each position to each site with
    probability `p` (torch `StaticProbabilityRouter`)."""

    def sample(key: PRNGKeyArray, leading_shape: tuple[int, ...]) -> tuple[Routes, ...]:
        return tuple(
            {
                name: random.bernoulli(random.fold_in(draw_key, j), p, leading_shape)
                for j, name in enumerate(sites)
            }
            for draw_key in random.split(key, n_draws)
        )

    return sample


def route_all_n(n_draws: int) -> RoutingSampler:
    """`n_draws` forwards, each routing every position to every site (`AllRoutingConfig`)."""

    def sample(_key: PRNGKeyArray, _leading_shape: tuple[int, ...]) -> tuple[Routes, ...]:
        return (None,) * n_draws

    return sample


def routing_sampler_from_config(
    routing: SubsetRoutingType, sites: tuple[str, ...], n_draws: int
) -> RoutingSampler:
    match routing:
        case UniformKSubsetRoutingConfig():
            return uniform_k_routing(sites, n_draws)
        case StaticProbabilityRoutingConfig():
            return static_probability_routing(sites, routing.p, n_draws)
        case AllRoutingConfig():
            return route_all_n(n_draws)


# ───────────────────────────── shared-config -> flat terms ─────────────────────────────


def persistent_configs(
    recon_terms: tuple[AnyReconLossTerm, ...],
) -> "dict[str, PersistentPGDReconLossConfig | MergedStochasticSubsetPPGDReconLossConfig]":
    """`state_key -> config` for every persistent-source-carrying recon term (SPEC S23:
    each key feeds exactly one term). Derived from the terms, not stored separately — the
    config rides each `PersistentSources` / `MixedPersistentStochasticSources` strategy;
    both carry the same adversary fields (optimizer/scope/source_dtype/n_warmup_steps)."""
    out: dict[str, PersistentPGDReconLossConfig | MergedStochasticSubsetPPGDReconLossConfig] = {}
    for term in recon_terms:
        if isinstance(term.sources, (PersistentSources, MixedPersistentStochasticSources)):
            assert term.sources.state_key not in out, term.sources.state_key
            out[term.sources.state_key] = term.sources.cfg
    return out
