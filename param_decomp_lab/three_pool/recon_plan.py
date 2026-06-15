"""How the chunkwise pool turns a chunk's sites into a list of recon forwards.

The chunkwise pool reconstructs by running masked forwards of the target model with some
of the chunk's sites swapped in for their decomposition (V/U). A ``ChunkReconPlan``
parameterises the *list of forwards* a chunk performs each step: each entry is one
forward, described by which sites participate and how each position routes to them
(``RoutingMasks``).

  * ``PerSitePlan`` — one forward per site, that site routed everywhere. The original
    layerwise loop ("swap one matrix at a time"); the default.
  * ``SubsetReconPlan`` — ``n_samples`` forwards, each over *all* the chunk's sites with
    a freshly-drawn per-position routing (``routing``). With ``routing=all`` this is the
    joint "swap everything at once" forward (``n_samples=1`` → one forward instead of N →
    ~N× less chunkwise compute); with ``routing=uniform_k`` it is per-position subset
    recon.

``n_forwards`` is the (deterministic) length of the list ``generate`` produces — the
per-chunk contribution to the global ``N_est`` that scales the stoch gradient (see
``step_chunkwise`` and ``three_pool/CLAUDE.md``).
"""

from typing import Annotated, Literal

import torch
from pydantic import Field

from param_decomp.masks import RoutingMasks, get_subset_router
from param_decomp_config.base import BaseConfig
from param_decomp_config.routing import SubsetRoutingType

ForwardRouting = tuple[tuple[str, ...], RoutingMasks]
"""One recon forward: ``(sites, routing)``. ``sites`` are the chunk sites to swap in
(the keys of the forward's ``mask_infos``); ``routing`` is ``"all"`` or a per-site
boolean mask over positions."""


class PerSitePlan(BaseConfig):
    type: Literal["per_site"] = "per_site"

    def n_forwards(self, sites: tuple[str, ...]) -> int:
        return len(sites)

    def generate(
        self, sites: tuple[str, ...], _mask_shape: tuple[int, ...], _device: torch.device
    ) -> list[ForwardRouting]:
        return [((s,), "all") for s in sites]


class SubsetReconPlan(BaseConfig):
    type: Literal["subset"] = "subset"
    routing: Annotated[SubsetRoutingType, Field(discriminator="type")]
    n_samples: int = 1

    def n_forwards(self, _sites: tuple[str, ...]) -> int:
        return self.n_samples

    def generate(
        self, sites: tuple[str, ...], mask_shape: tuple[int, ...], device: torch.device
    ) -> list[ForwardRouting]:
        router = get_subset_router(self.routing, device)
        return [(sites, router.get_masks(list(sites), mask_shape)) for _ in range(self.n_samples)]


ChunkReconPlan = PerSitePlan | SubsetReconPlan
