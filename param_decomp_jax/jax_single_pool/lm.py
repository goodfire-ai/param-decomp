"""`DecomposedModel` — the interface a vendored target implements for the generic trainer.

The trainer (`train.py`) is abstract over the target model: it sees an ordered set of
decomposed **sites** (SPEC §1.2) and a handful of pure functions over `(frozen, vu)`
pytrees. Everything at the boundary is keyed by site name (flat dicts, torch-module-path
style); how a target lays its parameters out internally (e.g. the Llama target's stacked
layer axis) is its own business.

The `[B,T,d]` residual is the FIXED WAIST: masking, routing, source scopes, imp-min, the
CI fn, and normalization all operate on `(B,T)` and stay LM-shaped. Only the three EDGES
are generic — the model's INPUT (whatever `site_inputs`/`clean_output`/`masked_output`
read upstream of the residual; tokens for an LM, a dict for a bio target), the model's
OUTPUT (`clean_output`/`masked_output` return `Any` — logits, a tuple of heads, coords),
and the recon comparison (`recon_loss_fn`, default `kl_per_position`).

Every function takes the frozen-target pytree as a RUNTIME argument. Never close over
it: a frozen 8B target captured as a jit constant bakes multi-GB weights into the HLO.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jaxtyping import Array, Bool, Float

from jax_single_pool.losses import kl_per_position

SiteMasks = dict[str, Float[Array, "B T C"]]
SiteDeltaMasks = dict[str, Float[Array, "B T"]]
SiteRoutes = dict[str, Bool[Array, "B T"]] | None
"""Per-site per-position routing; `None` routes every position to the decomposition
(SPEC §1.3). Positions routing False take the frozen `x @ W` path."""


@dataclass(frozen=True)
class SiteC:
    """A decomposed site as configured: its torch-module-path name and its C.

    The shape-carrying `SiteSpec` is derived from this plus the target's config."""

    name: str
    C: int


@dataclass(frozen=True)
class SiteSpec:
    name: str
    d_in: int
    d_out: int
    C: int


@dataclass(frozen=True)
class DecomposedModel:
    """Pure-function table over `(frozen, vu)` pytrees (see module docstring).

    `sites` fixes the canonical site order — chunking (SPEC S10) and the CI fn's
    input/output concatenation both follow it.

    `masked_output(frozen, vu, resid, masks, delta_masks, routes, live, has_delta)`:
    `live` (a tuple of site names, static under jit) lists the sites running their
    decomposed forward; all other sites MUST run the frozen `x @ W` path (SPEC S2).
    `masks`/`delta_masks` may broadcast over the batch dim (the PPGD source case).
    `has_delta` (static) False skips the `x @ Δ` matmul for constant-source entries
    whose delta mask is a constant 0 (LOSS_PARITY_DESIGN §4b).

    `clean_output` is the all-frozen forward — the recon target (SPEC S3); never the
    `mask=1` decomposed identity. Its output (and `masked_output`') is `Any`: an LM
    emits `[B,T,vocab]` logits, a bio target a tuple of heads or coordinates.

    `weight_deltas` returns fp32 `W − V@U` per site from fp32 master `vu` (SPEC N2).

    `masked_site_outputs` shares `masked_output`'s masking machinery but returns the
    per-`live`-site decomposed LINEAR OUTPUT (`((x@V)*m)@U + (x@Δ)*d`, the intermediate
    `masked_output` threads onward) instead of final logits, keyed by site (SPEC S31).
    It exists ONLY for the offline hidden-acts recon eval metrics — never the recon grid,
    which stays KL-on-final-logits (SPEC §2.3). The clean (target) per-site output is the
    frozen `x @ W` of `site_inputs`, derivable without this fn.

    `recon_loss_fn(clean_output, masked_output) -> scalar` is the recon comparison the
    step minimizes (SPEC §2.3). It defaults to `kl_per_position` (the LM cross-entropy
    surrogate); a non-LM target supplies its own (e.g. an MSE/geometric loss). It must
    reduce to a scalar and contract whatever shape its model emits.
    """

    sites: tuple[SiteSpec, ...]
    clean_output: Callable[[Any, Float[Array, "B T d"]], Any]
    site_inputs: Callable[[Any, Float[Array, "B T d"]], dict[str, Float[Array, "B T d_in"]]]
    masked_output: Callable[
        [
            Any,
            Any,
            Float[Array, "B T d"],
            SiteMasks,
            SiteDeltaMasks,
            SiteRoutes,
            tuple[str, ...],
            bool,
        ],
        Any,
    ]
    masked_site_outputs: Callable[
        [
            Any,
            Any,
            Float[Array, "B T d"],
            SiteMasks,
            SiteDeltaMasks,
            SiteRoutes,
            tuple[str, ...],
            bool,
        ],
        dict[str, Float[Array, "B T d_out"]],
    ]
    weight_deltas: Callable[[Any, Any], dict[str, Float[Array, "d_out d_in"]]]
    recon_loss_fn: Callable[[Any, Any], Float[Array, ""]] = field(default=kl_per_position)

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sites)


def chunk_sites(site_names: tuple[str, ...], sites_per_chunk: int) -> tuple[tuple[str, ...], ...]:
    """Sequential `sites_per_chunk`-groups in the canonical site order (SPEC S10)."""
    assert len(site_names) % sites_per_chunk == 0, (
        f"{len(site_names)} sites not divisible by sites_per_chunk={sites_per_chunk}"
    )
    return tuple(
        tuple(site_names[i : i + sites_per_chunk])
        for i in range(0, len(site_names), sites_per_chunk)
    )
