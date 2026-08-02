"""Host-only orchestration helpers for the recon-budget capacity lifecycle."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.components import ComponentStacks
from param_decomp.core.slot_surgery import find_inactive_slot, set_null_probe_factors
from param_decomp.core.train import TrainState

type ComponentGradientProbe = Callable[
    [
        TrainState,
        Any,
        PRNGKeyArray,
        dict[str, Array] | None,
        dict[str, Array] | None,
    ],
    tuple[Array, ComponentStacks],
]


@dataclass(frozen=True)
class BirthCandidate:
    site: str
    slot: int
    direction: Array
    sigma: float
    normalized_score: float


def _with_slot(mask: dict[str, Array], site: str, slot: int, value: bool) -> dict[str, Array]:
    out = dict(mask)
    out[site] = out[site].at[slot].set(value)
    return out


def choose_birth_candidate(
    state: TrainState,
    active: dict[str, Array],
    probe: ComponentGradientProbe,
    batch: Any,
    key: PRNGKeyArray,
    n_power_iters: int = 3,
) -> BirthCandidate | None:
    """Price all sites' next exact-null slots by implicit GradMax power iteration.

    The same batch and fresh-PGD key are reused for every alternating backward, so the
    implicit represented-matrix gradient ``G`` is fixed. Candidate slots are active and
    protected only in the ephemeral probe state. Each backward is function-preserving:
    ``(V=v,U=0)`` yields ``v^T G`` in grad-U, then ``(V=0,U=q)`` yields ``Gq`` in grad-V.
    No dense ``d_in × d_out`` gradient is materialized.
    """
    assert n_power_iters > 0, n_power_iters
    assert set(active) == set(state.decomposition.components.site_names)
    candidate_slots = {
        site: slot
        for site in state.decomposition.components.site_names
        if (slot := find_inactive_slot(state.decomposition.components, site)) is not None
    }
    if not candidate_slots:
        return None

    probe_active = dict(active)
    protected: dict[str, Array] = {}
    probe_state = state
    for site, slot in candidate_slots.items():
        probe_active = _with_slot(probe_active, site, slot, True)
        protected[site] = jnp.zeros_like(active[site]).at[slot].set(True)
        v_factor, u_factor = probe_state.decomposition.components.site(site)
        v = v_factor[:, slot].astype(jnp.float32)
        v = v / (jnp.linalg.norm(v) + 1e-30)
        probe_state = set_null_probe_factors(
            probe_state, site, slot, v, jnp.zeros((u_factor.shape[1],), jnp.float32)
        )

    sigma_by_site: dict[str, float] = {}
    for _ in range(n_power_iters):
        _, grad_v_state = probe(probe_state, batch, key, probe_active, protected)
        for site, slot in candidate_slots.items():
            v_factor, u_factor = probe_state.decomposition.components.site(site)
            _, grad_u = grad_v_state.site(site)
            q = grad_u[slot].astype(jnp.float32)
            sigma = float(jnp.linalg.norm(q))
            sigma_by_site[site] = sigma
            q = q / (sigma + 1e-30)
            probe_state = set_null_probe_factors(
                probe_state, site, slot, jnp.zeros((v_factor.shape[0],), jnp.float32), q
            )

        _, grad_u_state = probe(probe_state, batch, key, probe_active, protected)
        for site, slot in candidate_slots.items():
            v_factor, u_factor = probe_state.decomposition.components.site(site)
            grad_v, _ = grad_u_state.site(site)
            p = grad_v[:, slot].astype(jnp.float32)
            p = p / (jnp.linalg.norm(p) + 1e-30)
            probe_state = set_null_probe_factors(
                probe_state, site, slot, p, jnp.zeros((u_factor.shape[1],), jnp.float32)
            )

    candidates = []
    for site, slot in candidate_slots.items():
        v_factor, u_factor = probe_state.decomposition.components.site(site)
        sigma = sigma_by_site[site]
        if not (sigma > 0.0 and jnp.isfinite(sigma)):
            continue
        score = sigma / (v_factor.shape[0] * u_factor.shape[1]) ** 0.5
        candidates.append(BirthCandidate(site, slot, v_factor[:, slot], sigma, score))
    return max(candidates, key=lambda candidate: candidate.normalized_score, default=None)
