"""The trainer's loss surface (SPEC S10'): the `LossSurface` record — the faithfulness +
importance-minimality singletons and the recon Σ (each recon term a plan of which sites go
live per forward, how positions route, and the mask-SOURCE strategy for the [0,1] source
values). `build_loss_terms` validates the shared torch loss configs into that record.

A plan is built from two orthogonal choices: how the sites are CHUNKED (a chunking
helper `tuple[str, ...] -> list[Chunk]`) and how each chunk is turned into routed
forwards (`make_plan`, sharing one routing config + source strategy across chunks).
The torch loss-class cartesian product (`CIMasked`/`Stochastic`/`Unmasked`/`PGD`/
`PersistentPGD` x `_`/`Subset`/`Layerwise`) factors exactly as chunking x routing x
source strategy — see LOSS_PARITY_DESIGN.md. Everything here is static structure
closed over by the jit'd step; only keys (and, for persistent terms, `TrainState`
entries) vary per step.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from jax import random
from jaxtyping import Array, PRNGKeyArray

from param_decomp.configs import (
    AllRoutingConfig,
    AnyImportanceMinimalityLossConfig,
    AnyLossMetricConfig,
    ChunkwiseSubsetReconLossConfig,
    CIMaskedReconLayerwiseLossConfig,
    CIMaskedReconLossConfig,
    CIMaskedReconSubsetLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    MaskScopeLiteral,
    PersistentPGDReconLossConfig,
    PGDInitStrategy,
    PGDReconLayerwiseLossConfig,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    SmoothL0ImportanceMinimalityLossConfig,
    StaticProbabilityRoutingConfig,
    StochasticReconLayerwiseLossConfig,
    StochasticReconLossConfig,
    StochasticReconSubsetLossConfig,
    SubsetRoutingType,
    UniformKSubsetRoutingConfig,
    UnmaskedReconLossConfig,
)
from param_decomp.lm import chunk_sites

Routes = dict[str, Array] | None
RoutingSampler = Callable[[PRNGKeyArray, tuple[int, ...]], tuple[Routes, ...]]
"""`(key, leading_shape) -> (routes, ...)` — a STATICALLY-sized family of routing draws,
each `{site: bool[*leading]}` (or None = route everywhere) becoming ONE forward. The torch
`Router.get_masks` made pure: fresh draws per step require the key threaded in —
samplers run INSIDE the jitted step, so they must be traceable (SPEC R1). Returning
several draws from one invocation enables JOINTLY-sampled families (independent
repeats, antithetic/complementary subsets, per-step random covers) that duplicated
plan entries with independent keys cannot express. The plan's structure — live-sets,
sampler identities, family sizes — is static; only the key varies per step."""


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
class FreshPGDSources:
    """Per-step sign-PGD-ascended sources (torch `PGDRecon*` as TRAINING losses): init
    per `init`, `n_steps` of `step_size * sign(grad)` with clamp to [0,1], no state
    across steps. The entry's routing is drawn ONCE per step and shared by every
    ascent and the final loss forward (SPEC S24, torch parity)."""

    init: PGDInitStrategy
    n_steps: int
    step_size: float
    scope: MaskScopeLiteral


@dataclass(frozen=True)
class PersistentSources:
    """Sources living in `TrainState.adversaries[state_key]` across steps (PPGD). Carries
    the shared `PersistentPGDReconLossConfig` so the term is self-describing — the step
    reads its scope/optimizer/warmup straight off `cfg`. `state_key` indexes
    `TrainState.adversaries` (one key per persistent term, SPEC S23)."""

    state_key: str
    cfg: PersistentPGDReconLossConfig


MaskSourceStrategy = StochasticSources | ConstantSources | FreshPGDSources | PersistentSources


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
        """`ConstantSources` carries no delta path (torch passes no `weight_deltas` for the
        Unmasked/CIMasked losses); its `delta_mask` would be a constant 0, so the `x @ Δ`
        matmul is skipped entirely (static, retrace-safe — LOSS_PARITY_DESIGN §4b). Every
        other strategy drives a live delta mask."""
        return not isinstance(self.sources, ConstantSources)


ReconPlan = tuple[ReconForward, ...]


@dataclass(frozen=True)
class ReconLossTerm:
    """One coefficiented recon loss: mean over ALL draws of ALL plan entries of
    `kl_per_position` (SPEC S10'). `name` is the torch `instance_key` (`cfg.name` or
    the type literal) — the metric log key is `loss/<name>`."""

    name: str
    coeff: float
    plan: ReconPlan


@dataclass(frozen=True)
class FaithfulnessTerm:
    """Weight-space term: `Σ_s ‖Δ_s‖² / Σ_s numel` (SPEC S17). Carries no plan — its
    contribution reads the live weight deltas, not the masked forwards."""

    name: str
    coeff: float


@dataclass(frozen=True)
class ImportanceMinimalityTerm:
    """CI-space imp-min + entropy term (SPEC S7-S9). Carries the config so the step reads
    the annealed penalty parameter (`pnorm` / `gamma`) and `beta` straight off `cfg`. The
    config is either of the two imp-min penalties (`L_p` or smooth-L0)."""

    name: str
    coeff: float
    cfg: AnyImportanceMinimalityLossConfig


LossTerm = FaithfulnessTerm | ImportanceMinimalityTerm | ReconLossTerm
"""One built loss term. Each owns its `name` (the torch `instance_key`; log key
`loss/<name>` for recon / `faith` / `imp`), its `coeff`, and the data its contribution
needs."""


@dataclass(frozen=True)
class LossSurface:
    """The VPD objective's fixed shape — `L = c·faith + c·imp + Σ c·recon` (SPEC S10').
    faith and imp are singletons (exactly one each); recon is the genuinely plural sum.
    Built by `build_loss_terms`; the step reads each role by name and runs its distinct
    execution pattern — no isinstance dispatch over an arbitrary-order flat list."""

    faith: FaithfulnessTerm
    imp: ImportanceMinimalityTerm
    recon: tuple[ReconLossTerm, ...]


# ───────────────────────────── routing samplers ─────────────────────────────


def uniform_k_subset_routes(
    key: PRNGKeyArray, live_sites: tuple[str, ...], leading_shape: tuple[int, ...]
) -> dict[str, Array]:
    """Per position: `k ~ U{1..|live|}`, then a uniform k-subset of the live sites
    routes True (SPEC S11). Distributionally identical to torch's double-argsort ranks."""
    n_sites = len(live_sites)
    k_key, perm_key = random.split(key)
    k = random.randint(k_key, leading_shape, 1, n_sites + 1)
    perms = random.uniform(perm_key, (n_sites, *leading_shape)).argsort(axis=0)
    routed = perms < k
    return {name: routed[j] for j, name in enumerate(live_sites)}


def uniform_k_routing(live_sites: tuple[str, ...], n_draws: int) -> RoutingSampler:
    """`n_draws` independent per-position uniform-k-subset draws over `live_sites`."""

    def sample(key: PRNGKeyArray, leading_shape: tuple[int, ...]) -> tuple[Routes, ...]:
        return tuple(
            uniform_k_subset_routes(draw_key, live_sites, leading_shape)
            for draw_key in random.split(key, n_draws)
        )

    return sample


def static_probability_routing(
    live_sites: tuple[str, ...], p: float, n_draws: int
) -> RoutingSampler:
    """`n_draws` independent draws routing each position to each live site with
    probability `p` (torch `StaticProbabilityRouter`)."""

    def sample(key: PRNGKeyArray, leading_shape: tuple[int, ...]) -> tuple[Routes, ...]:
        return tuple(
            {
                name: random.bernoulli(random.fold_in(draw_key, j), p, leading_shape)
                for j, name in enumerate(live_sites)
            }
            for draw_key in random.split(key, n_draws)
        )

    return sample


def route_all_n(n_draws: int) -> RoutingSampler:
    """`n_draws` forwards, each routing every position to every live site (`AllRoutingConfig`)."""

    def sample(_key: PRNGKeyArray, _leading_shape: tuple[int, ...]) -> tuple[Routes, ...]:
        return (None,) * n_draws

    return sample


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


# ───────────────────────────── chunking helpers ─────────────────────────────

Chunk = tuple[str, ...]
"""A live-set: the sites that run their decomposed path in one forward, everything else
on the frozen `x@W` path (SPEC S2)."""


def one_chunk(sites: tuple[str, ...]) -> list[Chunk]:
    """The whole model as a single chunk (torch `all_sites`: every site live)."""
    return [tuple(sites)]


def per_site(sites: tuple[str, ...]) -> list[Chunk]:
    """One single-site chunk per site (torch `*Layerwise`)."""
    return [(s,) for s in sites]


def into_groups(sites: tuple[str, ...], k: int) -> list[Chunk]:
    """Sequential size-`k` groups in canonical site order (torch chunkwise; SPEC S10)."""
    return list(chunk_sites(sites, k))


# ───────────────────────────── the plan constructor ─────────────────────────────


def make_plan(
    chunks: list[Chunk],
    routing: SubsetRoutingType,
    sources: MaskSourceStrategy,
    n_samples: int,
) -> ReconPlan:
    """One `ReconForward` per chunk: that chunk live, the rest frozen `x@W` (SPEC S2),
    with `n_samples` routing draws from `routing` over the chunk's own sites (SPEC S11)
    and the shared `sources`. The chunking (`one_chunk`/`per_site`/`into_groups`) and the
    routing/source choices are orthogonal — see LOSS_PARITY_DESIGN.md."""
    return tuple(
        ReconForward(
            live_sites=chunk,
            sample_routing=routing_sampler_from_config(routing, chunk, n_samples),
            sources=sources,
        )
        for chunk in chunks
    )


def subset_chunk_plan(
    site_names: tuple[str, ...],
    sites_per_chunk: int,
    n_samples: int,
    sources: MaskSourceStrategy,
) -> ReconPlan:
    """The production plan: `n_samples` uniform-k forwards per chunk (torch
    `SubsetReconPlan` over `ThreePoolTopology` chunks)."""
    return make_plan(
        into_groups(site_names, sites_per_chunk), UniformKSubsetRoutingConfig(), sources, n_samples
    )


# ───────────────────────────── shared-config -> flat terms ─────────────────────────────


def persistent_configs(
    recon_terms: tuple[ReconLossTerm, ...],
) -> dict[str, PersistentPGDReconLossConfig]:
    """`state_key -> config` for every persistent recon term (SPEC S23: each key feeds
    exactly one term). Derived from the terms, not stored separately — the config rides
    each `PersistentSources` strategy."""
    out: dict[str, PersistentPGDReconLossConfig] = {}
    for term in recon_terms:
        for entry in term.plan:
            if isinstance(entry.sources, PersistentSources):
                assert entry.sources.state_key not in out, entry.sources.state_key
                out[entry.sources.state_key] = entry.sources.cfg
    return out


def build_loss_terms(
    loss_metrics: Sequence[AnyLossMetricConfig],
    site_names: tuple[str, ...],
) -> LossSurface:
    """Validate the shared torch loss configs into the `LossSurface` record
    (LOSS_PARITY_DESIGN §3): exactly one faithfulness + one importance-minimality term,
    plus the recon terms (the genuinely plural Σ).

    Asserts the subset this trainer implements; refuses everything else loudly.
    Recon-term ORDER follows the config list — per-term RNG keys derive from the recon
    index, so it is semantically load-bearing (SPEC R1). Stochastic terms read
    `n_mask_samples` off their own config (no longer a trainer-level knob)."""
    faith: FaithfulnessTerm | None = None
    imp: ImportanceMinimalityTerm | None = None
    recon_terms: list[ReconLossTerm] = []

    def unique_name(cfg: AnyLossMetricConfig) -> str:
        # Checks the already-committed terms (not this in-progress one), so the persistent
        # case calling this twice — once for the state_key, once inside recon() — is safe.
        name = cfg.name if cfg.name is not None else cfg.type
        taken = {t.name for t in recon_terms}
        if faith is not None:
            taken.add(faith.name)
        if imp is not None:
            taken.add(imp.name)
        assert name not in taken, f"duplicate loss instance_key {name!r}"
        return name

    def recon(cfg: AnyLossMetricConfig, plan: ReconPlan) -> ReconLossTerm:
        assert cfg.coeff is not None
        return ReconLossTerm(unique_name(cfg), cfg.coeff, plan)

    for cfg in loss_metrics:
        assert cfg.coeff is not None, f"{cfg.type}: training losses need a coeff"
        match cfg:
            case FaithfulnessLossConfig():
                assert faith is None
                faith = FaithfulnessTerm(unique_name(cfg), cfg.coeff)
            case ImportanceMinimalityLossConfig():
                assert imp is None
                assert cfg.p_anneal_final_p is not None
                imp = ImportanceMinimalityTerm(unique_name(cfg), cfg.coeff, cfg)
            case SmoothL0ImportanceMinimalityLossConfig():
                assert imp is None
                assert cfg.gamma_anneal_final_gamma is not None
                imp = ImportanceMinimalityTerm(unique_name(cfg), cfg.coeff, cfg)
            case UnmaskedReconLossConfig() | CIMaskedReconLossConfig():
                value = 1.0 if isinstance(cfg, UnmaskedReconLossConfig) else 0.0
                plan = make_plan(
                    one_chunk(site_names), AllRoutingConfig(), ConstantSources(value), n_samples=1
                )
                recon_terms.append(recon(cfg, plan))
            case CIMaskedReconSubsetLossConfig():
                plan = make_plan(
                    one_chunk(site_names), cfg.routing, ConstantSources(0.0), n_samples=1
                )
                recon_terms.append(recon(cfg, plan))
            case CIMaskedReconLayerwiseLossConfig():
                plan = make_plan(
                    per_site(site_names), AllRoutingConfig(), ConstantSources(0.0), n_samples=1
                )
                recon_terms.append(recon(cfg, plan))
            case StochasticReconLossConfig():
                plan = make_plan(
                    one_chunk(site_names),
                    AllRoutingConfig(),
                    StochasticSources(),
                    cfg.n_mask_samples,
                )
                recon_terms.append(recon(cfg, plan))
            case StochasticReconSubsetLossConfig():
                plan = make_plan(
                    one_chunk(site_names), cfg.routing, StochasticSources(), cfg.n_mask_samples
                )
                recon_terms.append(recon(cfg, plan))
            case StochasticReconLayerwiseLossConfig():
                plan = make_plan(
                    per_site(site_names),
                    AllRoutingConfig(),
                    StochasticSources(),
                    cfg.n_mask_samples,
                )
                recon_terms.append(recon(cfg, plan))
            case ChunkwiseSubsetReconLossConfig():
                assert isinstance(cfg.routing, UniformKSubsetRoutingConfig), cfg.routing
                plan = make_plan(
                    into_groups(site_names, cfg.sites_per_chunk),
                    cfg.routing,
                    StochasticSources(),
                    cfg.n_samples,
                )
                recon_terms.append(recon(cfg, plan))
            case PGDReconLossConfig() | PGDReconSubsetLossConfig():
                fresh = FreshPGDSources(cfg.init, cfg.n_steps, cfg.step_size, cfg.mask_scope)
                routing = (
                    cfg.routing if isinstance(cfg, PGDReconSubsetLossConfig) else AllRoutingConfig()
                )
                plan = make_plan(one_chunk(site_names), routing, fresh, n_samples=1)
                recon_terms.append(recon(cfg, plan))
            case PGDReconLayerwiseLossConfig():
                fresh = FreshPGDSources(cfg.init, cfg.n_steps, cfg.step_size, cfg.mask_scope)
                plan = make_plan(per_site(site_names), AllRoutingConfig(), fresh, n_samples=1)
                recon_terms.append(recon(cfg, plan))
            case PersistentPGDReconLossConfig():
                schedule = cfg.optimizer.lr_schedule
                assert schedule.fn_type == "constant" and schedule.final_val_frac == 1.0, schedule
                key = unique_name(cfg)
                plan = make_plan(
                    one_chunk(site_names),
                    AllRoutingConfig(),
                    PersistentSources(state_key=key, cfg=cfg),
                    cfg.n_samples,
                )
                recon_terms.append(recon(cfg, plan))
            case _:
                # StochasticHiddenActsReconLoss lands here by design: it is keep-on-bridge
                # (offline-eval only), not a JAX training loss (SPEC S31, LOSS_PARITY_DESIGN §4c).
                raise AssertionError(f"unsupported training loss {cfg.type!r}")

    assert faith is not None and imp is not None, (
        f"need FaithfulnessLoss + ImportanceMinimalityLoss, got {[m.type for m in loss_metrics]}"
    )
    assert recon_terms, "no recon loss terms configured"
    for term in recon_terms:
        for entry in term.plan:
            assert entry.live_sites and set(entry.live_sites) <= set(site_names), entry
    return LossSurface(faith, imp, tuple(recon_terms))
