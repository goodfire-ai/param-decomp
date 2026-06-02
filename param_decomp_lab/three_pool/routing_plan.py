"""How the LW pool turns a block's owned sites into a list of recon forwards.

The LW pool reconstructs by running masked forwards of the target model with some
of the block's owned sites swapped in for their decomposition (V/U). A
``RoutingPlan`` parameterises the *list of forwards* a block performs each step:
each entry is one forward, described by which owned sites participate and how each
position routes to them (``RoutingMasks``).

  * ``PerSitePlan`` — one forward per owned site, that site routed everywhere. The
    original layerwise loop ("swap one matrix at a time"); the default.
  * ``SubsetRoutingPlan`` — ``n_samples`` forwards, each over *all* owned sites with
    a freshly-drawn per-position routing (``routing``). With ``routing=all`` this is
    the joint "swap everything at once" forward (``n_samples=1`` → one forward
    instead of N → ~N× less LW compute); with ``routing=uniform_k`` it is per-position
    subset recon.

``n_forwards`` is the (deterministic) length of the list ``generate`` produces — the
per-block contribution to the global ``N_est`` that scales the stoch gradient (see
``step_layerwise`` and ``three_pool/CLAUDE.md``).
"""

from typing import Annotated, Literal

import torch
from pydantic import Field

from param_decomp.base_config import BaseConfig
from param_decomp.masks import RoutingMasks, SubsetRoutingType, get_subset_router

ForwardRouting = tuple[tuple[str, ...], RoutingMasks]
"""One recon forward: ``(sites, routing)``. ``sites`` are the owned sites to swap in
(the keys of the forward's ``mask_infos``); ``routing`` is ``"all"`` or a per-site
boolean mask over positions."""


class PerSitePlan(BaseConfig):
    type: Literal["per_site"] = "per_site"

    def n_forwards(self, owned_sites: tuple[str, ...]) -> int:
        return len(owned_sites)

    def generate(
        self, owned_sites: tuple[str, ...], _mask_shape: tuple[int, ...], _device: torch.device
    ) -> list[ForwardRouting]:
        return [((s,), "all") for s in owned_sites]


class SubsetRoutingPlan(BaseConfig):
    type: Literal["subset"] = "subset"
    routing: Annotated[SubsetRoutingType, Field(discriminator="type")]
    n_samples: int = 1

    def n_forwards(self, _owned_sites: tuple[str, ...]) -> int:
        return self.n_samples

    def generate(
        self, owned_sites: tuple[str, ...], mask_shape: tuple[int, ...], device: torch.device
    ) -> list[ForwardRouting]:
        router = get_subset_router(self.routing, device)
        return [
            (owned_sites, router.get_masks(list(owned_sites), mask_shape))
            for _ in range(self.n_samples)
        ]


RoutingPlan = PerSitePlan | SubsetRoutingPlan
