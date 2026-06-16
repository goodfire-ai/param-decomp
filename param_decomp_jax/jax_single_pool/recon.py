"""Recon loss terms (SPEC S10'): plans (which sites go live per forward, how positions
route), mask-SOURCE strategies (where the [0,1] source values come from), and the
mapping from the shared torch loss configs onto them (`build_recon_terms`).

The torch loss-class cartesian product (`CIMasked`/`Stochastic`/`Unmasked`/`PGD`/
`PersistentPGD` x `_`/`Subset`/`Layerwise`) factors exactly as plan shape x source
strategy — see LOSS_PARITY_DESIGN.md. Everything here is static structure closed
over by the jit'd step; only keys (and, for persistent terms, `TrainState` entries)
vary per step.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from jax import random
from jaxtyping import Array, PRNGKeyArray

from jax_single_pool.lm import chunk_sites
from param_decomp_config.losses import (
    AdamPGDConfig,
    ChunkwiseSubsetReconLossConfig,
    CIMaskedReconLayerwiseLossConfig,
    CIMaskedReconLossConfig,
    CIMaskedReconSubsetLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
    PGDReconLayerwiseLossConfig,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    SCScope,
    StochasticReconLayerwiseLossConfig,
    StochasticReconLossConfig,
    StochasticReconSubsetLossConfig,
    UnmaskedReconLossConfig,
)
from param_decomp_config.pd import AnyLossMetricConfig
from param_decomp_config.routing import (
    AllRoutingConfig,
    StaticProbabilityRoutingConfig,
    SubsetRoutingType,
    UniformKSubsetRoutingConfig,
)

Routes = dict[str, Array] | None
RoutingSampler = Callable[[PRNGKeyArray, tuple[int, int]], tuple[Routes, ...]]
"""`(key, (B, T)) -> (routes, ...)` — a STATICALLY-sized family of routing draws, each
`{site: bool[B, T]}` (or None = route everywhere) becoming ONE forward. The torch
`Router.get_masks` made pure: fresh draws per step require the key threaded in —
samplers run INSIDE the jitted step, so they must be traceable (SPEC R1). Returning
several draws from one invocation enables JOINTLY-sampled families (independent
repeats, antithetic/complementary subsets, per-step random covers) that duplicated
plan entries with independent keys cannot express. The plan's structure — live-sets,
sampler identities, family sizes — is static; only the key varies per step."""


# ───────────────────────────── mask-source strategies ─────────────────────────────


@dataclass(frozen=True)
class StochasticSources:
    """Fresh per-draw sources: components `U[0,1]` (or Bernoulli), delta `U[0,1]`."""

    sampling: Literal["continuous", "binomial"]


@dataclass(frozen=True)
class ConstantSources:
    """`mask = ci + (1-ci)*value`: 0.0 = CI-masked, 1.0 = unmasked. `delta_mask = 0`
    (torch passes no delta path at all for these; multiplying by zero is
    mathematically identical, it just pays the delta matmul)."""

    value: float


@dataclass(frozen=True)
class FreshPGDSources:
    """Per-step sign-PGD-ascended sources (torch `PGDRecon*` as TRAINING losses): init
    per `init`, `n_steps` of `step_size * sign(grad)` with clamp to [0,1], no state
    across steps. The entry's routing is drawn ONCE per step and shared by every
    ascent and the final loss forward (SPEC S24, torch parity)."""

    init: Literal["random", "ones", "zeroes"]
    n_steps: int
    step_size: float
    scope: Literal["c", "bc", "bsc"]


@dataclass(frozen=True)
class PersistentSources:
    """Sources living in `TrainState.sources[state_key]` across steps (PPGD). The
    scope/optimizer/warmup config rides the shared `PersistentPGD*LossConfig`,
    resolved by the step factory; this is just the state pointer."""

    state_key: str


MaskSourceStrategy = StochasticSources | ConstantSources | FreshPGDSources | PersistentSources


def strategy_has_delta(sources: MaskSourceStrategy) -> bool:
    """`ConstantSources` carries no delta path (torch passes no `weight_deltas` for the
    Unmasked/CIMasked losses); its `delta_mask` would be a constant 0, so the `x @ Δ`
    matmul is skipped entirely (static, retrace-safe — LOSS_PARITY_DESIGN §4b). Every
    other strategy drives a live delta mask."""
    return not isinstance(sources, ConstantSources)


@dataclass(frozen=True)
class ReconForward:
    """One plan entry: which sites run their decomposed path (`live_sites` — everything
    else takes the frozen `x @ W` path, the ~9x-cheaper non-decomposed matmul), a
    sampler producing this entry's family of routing draws, and the strategy that
    generates each draw's mask/delta sources. `has_delta` (static, derived from the
    strategy) skips the `x @ Δ` matmul for constant-source entries."""

    live_sites: tuple[str, ...]
    sample_routing: RoutingSampler
    sources: MaskSourceStrategy

    @property
    def has_delta(self) -> bool:
        return strategy_has_delta(self.sources)


ReconPlan = tuple[ReconForward, ...]


@dataclass(frozen=True)
class ReconLossTerm:
    """One coefficiented recon loss: mean over ALL draws of ALL plan entries of
    `kl_per_position` (SPEC S10'). `name` is the torch `instance_key` (`cfg.name` or
    the type literal) — the metric log key is `loss/<name>`."""

    name: str
    coeff: float
    plan: ReconPlan


ReconLossTerms = tuple[ReconLossTerm, ...]


# ───────────────────────────── routing samplers ─────────────────────────────


def uniform_k_subset_routes(
    key: PRNGKeyArray, live_sites: tuple[str, ...], batch_seq_shape: tuple[int, ...]
) -> dict[str, Array]:
    """Per position: `k ~ U{1..|live|}`, then a uniform k-subset of the live sites
    routes True (SPEC S11). Distributionally identical to torch's double-argsort ranks."""
    n_sites = len(live_sites)
    k_key, perm_key = random.split(key)
    k = random.randint(k_key, batch_seq_shape, 1, n_sites + 1)
    perms = random.uniform(perm_key, (n_sites, *batch_seq_shape)).argsort(axis=0)
    routed = perms < k
    return {name: routed[j] for j, name in enumerate(live_sites)}


def uniform_k_routing(live_sites: tuple[str, ...], n_draws: int) -> RoutingSampler:
    """`n_draws` independent per-position uniform-k-subset draws over `live_sites`."""

    def sample(key: PRNGKeyArray, batch_seq_shape: tuple[int, int]) -> tuple[Routes, ...]:
        return tuple(
            uniform_k_subset_routes(draw_key, live_sites, batch_seq_shape)
            for draw_key in random.split(key, n_draws)
        )

    return sample


def static_probability_routing(
    live_sites: tuple[str, ...], p: float, n_draws: int
) -> RoutingSampler:
    """`n_draws` independent draws routing each position to each live site with
    probability `p` (torch `StaticProbabilityRouter`)."""

    def sample(key: PRNGKeyArray, batch_seq_shape: tuple[int, int]) -> tuple[Routes, ...]:
        return tuple(
            {
                name: random.bernoulli(random.fold_in(draw_key, j), p, batch_seq_shape)
                for j, name in enumerate(live_sites)
            }
            for draw_key in random.split(key, n_draws)
        )

    return sample


def route_all_n(n_draws: int) -> RoutingSampler:
    """`n_draws` forwards, each routing every position to every live site."""

    def sample(_key: PRNGKeyArray, _batch_seq_shape: tuple[int, int]) -> tuple[Routes, ...]:
        return (None,) * n_draws

    return sample


def route_all(_key: PRNGKeyArray, _batch_seq_shape: tuple[int, int]) -> tuple[Routes, ...]:
    """One draw routing every position to every live site."""
    return (None,)


def routing_sampler_from_config(
    routing: SubsetRoutingType, live_sites: tuple[str, ...], n_draws: int
) -> RoutingSampler:
    match routing:
        case UniformKSubsetRoutingConfig():
            return uniform_k_routing(live_sites, n_draws)
        case StaticProbabilityRoutingConfig():
            return static_probability_routing(live_sites, routing.p, n_draws)
        case AllRoutingConfig():
            return route_all_n(n_draws)


# ───────────────────────────── plan builders ─────────────────────────────


def subset_chunk_plan(
    site_names: tuple[str, ...],
    sites_per_chunk: int,
    n_samples: int,
    sources: MaskSourceStrategy,
) -> ReconPlan:
    """The production plan: partition into sequential chunks, `n_samples` uniform-k
    forwards per chunk (torch `SubsetReconPlan` over `ThreePoolTopology` chunks)."""
    return tuple(
        ReconForward(
            live_sites=chunk,
            sample_routing=uniform_k_routing(chunk, n_samples),
            sources=sources,
        )
        for chunk in chunk_sites(site_names, sites_per_chunk)
    )


def per_site_plan(site_names: tuple[str, ...], sources: MaskSourceStrategy) -> ReconPlan:
    """One forward per site, routed everywhere — the torch `*Layerwise` plan shape."""
    return tuple(
        ReconForward(live_sites=(site,), sample_routing=route_all, sources=sources)
        for site in site_names
    )


def all_sites_plan(
    site_names: tuple[str, ...], sample_routing: RoutingSampler, sources: MaskSourceStrategy
) -> ReconPlan:
    """One entry with every site live."""
    return (ReconForward(live_sites=site_names, sample_routing=sample_routing, sources=sources),)


# ───────────────────────────── shared-config -> terms ─────────────────────────────


@dataclass(frozen=True)
class LossSpec:
    """The trainer's whole loss surface, built from the shared `pd.loss_metrics` list:
    the two non-recon terms + the recon terms + the persistent-source registry
    (state_key -> shared config; SPEC S23: each key feeds exactly one term)."""

    faith_coeff: float
    imp_min: ImportanceMinimalityLossConfig
    recon_terms: ReconLossTerms
    persistent: dict[str, PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig]


def _assert_supported_persistent(
    cfg: PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig,
) -> None:
    assert isinstance(cfg.scope, SCScope), f"persistent scope {cfg.scope} unsupported (sc only)"
    assert not cfg.use_sigmoid_parameterization and cfg.start_frac == 0.0, cfg
    optimizer = cfg.optimizer
    assert isinstance(optimizer, AdamPGDConfig), optimizer
    schedule = optimizer.lr_schedule
    assert schedule.fn_type == "constant" and schedule.final_val_frac == 1.0, schedule


def build_recon_terms(
    loss_metrics: tuple[AnyLossMetricConfig, ...] | list[AnyLossMetricConfig],
    site_names: tuple[str, ...],
    n_mask_samples: int,
    sampling: Literal["continuous", "binomial"],
) -> LossSpec:
    """Map the shared torch loss configs onto recon terms (LOSS_PARITY_DESIGN §3).

    Asserts the subset this trainer implements; refuses everything else loudly.
    Term ORDER follows the config list (recon terms only) — per-term RNG keys are
    derived from that order, so it is semantically load-bearing (SPEC R1)."""
    faith_coeff: float | None = None
    imp_min: ImportanceMinimalityLossConfig | None = None
    terms: list[ReconLossTerm] = []
    persistent: dict[str, PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig] = {}

    def instance_key(cfg: AnyLossMetricConfig) -> str:
        name = cfg.name if cfg.name is not None else cfg.type
        assert all(t.name != name for t in terms), f"duplicate loss instance_key {name!r}"
        return name

    for cfg in loss_metrics:
        assert cfg.coeff is not None, f"{cfg.type}: training losses need a coeff"
        match cfg:
            case FaithfulnessLossConfig():
                assert faith_coeff is None
                faith_coeff = cfg.coeff
            case ImportanceMinimalityLossConfig():
                assert imp_min is None
                assert cfg.p_anneal_final_p is not None
                imp_min = cfg
            case UnmaskedReconLossConfig() | CIMaskedReconLossConfig():
                value = 1.0 if isinstance(cfg, UnmaskedReconLossConfig) else 0.0
                plan = all_sites_plan(site_names, route_all, ConstantSources(value))
                terms.append(ReconLossTerm(instance_key(cfg), cfg.coeff, plan))
            case CIMaskedReconSubsetLossConfig():
                sampler = routing_sampler_from_config(cfg.routing, site_names, n_draws=1)
                plan = all_sites_plan(site_names, sampler, ConstantSources(0.0))
                terms.append(ReconLossTerm(instance_key(cfg), cfg.coeff, plan))
            case CIMaskedReconLayerwiseLossConfig():
                plan = per_site_plan(site_names, ConstantSources(0.0))
                terms.append(ReconLossTerm(instance_key(cfg), cfg.coeff, plan))
            case StochasticReconLossConfig():
                plan = all_sites_plan(
                    site_names, route_all_n(n_mask_samples), StochasticSources(sampling)
                )
                terms.append(ReconLossTerm(instance_key(cfg), cfg.coeff, plan))
            case StochasticReconSubsetLossConfig():
                sampler = routing_sampler_from_config(cfg.routing, site_names, n_mask_samples)
                plan = all_sites_plan(site_names, sampler, StochasticSources(sampling))
                terms.append(ReconLossTerm(instance_key(cfg), cfg.coeff, plan))
            case StochasticReconLayerwiseLossConfig():
                plan = tuple(
                    ReconForward(
                        live_sites=(site,),
                        sample_routing=route_all_n(n_mask_samples),
                        sources=StochasticSources(sampling),
                    )
                    for site in site_names
                )
                terms.append(ReconLossTerm(instance_key(cfg), cfg.coeff, plan))
            case ChunkwiseSubsetReconLossConfig():
                assert isinstance(cfg.routing, UniformKSubsetRoutingConfig), cfg.routing
                plan = subset_chunk_plan(
                    site_names, cfg.sites_per_chunk, cfg.n_samples, StochasticSources(sampling)
                )
                terms.append(ReconLossTerm(instance_key(cfg), cfg.coeff, plan))
            case PGDReconLossConfig() | PGDReconSubsetLossConfig():
                fresh = FreshPGDSources(cfg.init, cfg.n_steps, cfg.step_size, cfg.mask_scope)
                sampler = (
                    routing_sampler_from_config(cfg.routing, site_names, n_draws=1)
                    if isinstance(cfg, PGDReconSubsetLossConfig)
                    else route_all
                )
                plan = all_sites_plan(site_names, sampler, fresh)
                terms.append(ReconLossTerm(instance_key(cfg), cfg.coeff, plan))
            case PGDReconLayerwiseLossConfig():
                fresh = FreshPGDSources(cfg.init, cfg.n_steps, cfg.step_size, cfg.mask_scope)
                plan = per_site_plan(site_names, fresh)
                terms.append(ReconLossTerm(instance_key(cfg), cfg.coeff, plan))
            case PersistentPGDReconLossConfig() | PersistentPGDReconSubsetLossConfig():
                _assert_supported_persistent(cfg)
                key = instance_key(cfg)
                assert key not in persistent
                persistent[key] = cfg
                sampler = (
                    routing_sampler_from_config(cfg.routing, site_names, cfg.n_samples)
                    if isinstance(cfg, PersistentPGDReconSubsetLossConfig)
                    else route_all_n(cfg.n_samples)
                )
                plan = all_sites_plan(site_names, sampler, PersistentSources(state_key=key))
                terms.append(ReconLossTerm(key, cfg.coeff, plan))
            case _:
                raise AssertionError(f"unsupported training loss {cfg.type!r}")

    assert faith_coeff is not None and imp_min is not None, (
        f"need FaithfulnessLoss + ImportanceMinimalityLoss, got {[m.type for m in loss_metrics]}"
    )
    assert terms, "no recon loss terms configured"
    for term in terms:
        for entry in term.plan:
            assert entry.live_sites and set(entry.live_sites) <= set(site_names), entry
    return LossSpec(
        faith_coeff=faith_coeff, imp_min=imp_min, recon_terms=tuple(terms), persistent=persistent
    )
