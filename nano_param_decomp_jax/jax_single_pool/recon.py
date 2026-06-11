"""Stochastic-recon plans (SPEC S10/S11): which sites go live per forward, and how
positions route to them. Plans are static structure; only the key varies per step."""

from collections.abc import Callable
from dataclasses import dataclass

from jax import random
from jaxtyping import Array, PRNGKeyArray

from jax_single_pool.lm import chunk_sites

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


@dataclass(frozen=True)
class ReconForward:
    """One plan entry: which sites run their decomposed path (`live_sites` — everything
    else takes the frozen `x @ W` path, the ~9x-cheaper non-decomposed matmul) and a
    sampler producing this entry's family of routing draws. Each draw is one forward,
    with its own fresh mask/delta sources."""

    live_sites: tuple[str, ...]
    sample_routing: RoutingSampler


ReconPlan = tuple[ReconForward, ...]
"""The stochastic-recon loss is the mean over ALL forwards (every draw of every entry)
of KL/(B·T). Live-sets may differ across entries (each traces its own forward); the
plan is fixed across steps (varying it would retrace)."""


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


def route_all(_key: PRNGKeyArray, _batch_seq_shape: tuple[int, int]) -> tuple[Routes, ...]:
    """One draw routing every position to every live site."""
    return (None,)


def subset_chunk_plan(
    site_names: tuple[str, ...], sites_per_chunk: int, n_samples: int
) -> ReconPlan:
    """The production plan: partition into sequential chunks, `n_samples` uniform-k
    forwards per chunk (torch `SubsetReconPlan` over `ThreePoolTopology` chunks)."""
    return tuple(
        ReconForward(live_sites=chunk, sample_routing=uniform_k_routing(chunk, n_samples))
        for chunk in chunk_sites(site_names, sites_per_chunk)
    )


def per_site_plan(site_names: tuple[str, ...]) -> ReconPlan:
    """One forward per site, routed everywhere — the historical "layerwise" loop
    (torch `PerSitePlan` / `StochasticReconLayerwiseLoss`)."""
    return tuple(ReconForward(live_sites=(site,), sample_routing=route_all) for site in site_names)
