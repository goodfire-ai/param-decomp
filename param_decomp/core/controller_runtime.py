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


@dataclass(frozen=True)
class BirthSiteBatch:
    site: str
    slots: tuple[int, ...]
    directions: Array  # [d_in, k], orthonormal columns
    sigmas: Array  # [k], training-referee singular values
    validation_cosines: Array  # [n_validation], basis-invariant block transfer


@dataclass(frozen=True)
class BirthBatchCandidate:
    sites: tuple[BirthSiteBatch, ...]

    @property
    def size(self) -> int:
        return sum(len(site.slots) for site in self.sites)


def _block_transfer_cosines(training: Array, validation: Array) -> Array:
    """Cosine of one proposed block update with independent gradient blocks.

    The row basis is arbitrary: applying the same orthogonal rotation to ``training``
    and ``validation`` leaves every cosine unchanged.
    """
    assert training.ndim == 2, training.shape
    assert validation.ndim == 3 and validation.shape[1:] == training.shape, (
        training.shape,
        validation.shape,
    )
    numerator = jnp.einsum("kd,nkd->n", training, validation)
    denominator = jnp.linalg.norm(training) * jnp.linalg.norm(validation, axis=(1, 2))
    return jnp.where(denominator > 0.0, numerator / denominator, jnp.nan)


def _set_probe_blocks(
    state: TrainState,
    slots_by_site: dict[str, tuple[int, ...]],
    v_by_site: dict[str, Array],
    u_by_site: dict[str, Array],
) -> TrainState:
    for site, slots in slots_by_site.items():
        assert v_by_site[site].shape[1] == len(slots)
        assert u_by_site[site].shape[0] == len(slots)
        for index, slot in enumerate(slots):
            state = set_null_probe_factors(
                state, site, slot, v_by_site[site][:, index], u_by_site[site][index]
            )
    return state


def choose_birth_batch(
    state: TrainState,
    active: dict[str, Array],
    probe: ComponentGradientProbe,
    training_batch: Any,
    training_key: PRNGKeyArray,
    validation: tuple[tuple[Any, PRNGKeyArray], ...],
    max_slots_per_site: int,
    n_power_iters: int = 3,
) -> BirthBatchCandidate | None:
    """Price a block of exact-null slots with implicit block GradMax.

    Scratch slots materialize neither represented-matrix gradients nor model changes:
    alternating factor-gradient probes compute ``P^T G`` and ``G Q``. A small Rayleigh-
    Ritz SVD recovers an orthonormal approximate singular block. Selection has no raw
    singular-value floor: the whole finite, nonzero block survives only when its predicted
    first-step update is a descent direction on EVERY independent validation referee.
    This blockwise Frobenius alignment is invariant to rotations inside degenerate singular
    subspaces; validating individual SVD coordinates is not. The block cap is a
    systems/work limit; a caller must report when it truncates demand.
    """
    assert max_slots_per_site > 0, max_slots_per_site
    assert n_power_iters > 0, n_power_iters
    assert validation, "batched birth requires at least one independent referee"
    assert set(active) == set(state.decomposition.components.site_names)

    slots_by_site: dict[str, tuple[int, ...]] = {}
    p_by_site: dict[str, Array] = {}
    for site in state.decomposition.components.site_names:
        V, U = state.decomposition.components.site(site)
        inactive = tuple(
            slot
            for slot in range(U.shape[0])
            if not bool(active[site][slot]) and bool(jnp.all(U[slot] == 0.0))
        )
        k = min(len(inactive), max_slots_per_site, V.shape[0], U.shape[1])
        if k == 0:
            continue
        slots = inactive[:k]
        p_block, _ = jnp.linalg.qr(V[:, jnp.asarray(slots)].astype(jnp.float32), mode="reduced")
        slots_by_site[site] = slots
        p_by_site[site] = p_block[:, :k]
    if not slots_by_site:
        return None

    probe_active = dict(active)
    protected: dict[str, Array] = {}
    for site, slots in slots_by_site.items():
        mask = jnp.zeros_like(active[site])
        for slot in slots:
            mask = mask.at[slot].set(True)
            probe_active = _with_slot(probe_active, site, slot, True)
        protected[site] = mask

    probe_state = state
    for _ in range(n_power_iters):
        zeros_u = {
            site: jnp.zeros((len(slots), state.decomposition.components.site(site)[1].shape[1]))
            for site, slots in slots_by_site.items()
        }
        probe_state = _set_probe_blocks(probe_state, slots_by_site, p_by_site, zeros_u)
        _, grad = probe(probe_state, training_batch, training_key, probe_active, protected)
        q_by_site: dict[str, Array] = {}
        for site, slots in slots_by_site.items():
            _, grad_u = grad.site(site)
            projected = grad_u[jnp.asarray(slots)].astype(jnp.float32)
            q_block, _ = jnp.linalg.qr(projected.T, mode="reduced")
            q_by_site[site] = q_block[:, : len(slots)]

        zeros_v = {
            site: jnp.zeros((state.decomposition.components.site(site)[0].shape[0], len(slots)))
            for site, slots in slots_by_site.items()
        }
        probe_state = _set_probe_blocks(
            probe_state,
            slots_by_site,
            zeros_v,
            {site: Q.T for site, Q in q_by_site.items()},
        )
        _, grad = probe(probe_state, training_batch, training_key, probe_active, protected)
        for site, slots in slots_by_site.items():
            grad_v, _ = grad.site(site)
            p_block, _ = jnp.linalg.qr(
                grad_v[:, jnp.asarray(slots)].astype(jnp.float32), mode="reduced"
            )
            p_by_site[site] = p_block[:, : len(slots)]

    zeros_u = {
        site: jnp.zeros((len(slots), state.decomposition.components.site(site)[1].shape[1]))
        for site, slots in slots_by_site.items()
    }
    probe_state = _set_probe_blocks(probe_state, slots_by_site, p_by_site, zeros_u)
    _, training_grad = probe(probe_state, training_batch, training_key, probe_active, protected)

    q_by_site = {}
    sigma_by_site = {}
    for site, slots in slots_by_site.items():
        _, grad_u = training_grad.site(site)
        projected = grad_u[jnp.asarray(slots)].astype(jnp.float32)
        rotation, sigmas, Qh = jnp.linalg.svd(projected, full_matrices=False)
        p_by_site[site] = p_by_site[site] @ rotation
        q_by_site[site] = Qh
        sigma_by_site[site] = sigmas

    probe_state = _set_probe_blocks(probe_state, slots_by_site, p_by_site, zeros_u)
    validation_blocks: dict[str, list[Array]] = {site: [] for site in slots_by_site}
    for validation_batch, validation_key in validation:
        _, validation_grad = probe(
            probe_state, validation_batch, validation_key, probe_active, protected
        )
        for site, slots in slots_by_site.items():
            _, grad_u = validation_grad.site(site)
            validation_blocks[site].append(grad_u[jnp.asarray(slots)].astype(jnp.float32))

    site_batches = []
    for site, slots in slots_by_site.items():
        sigmas = sigma_by_site[site]
        positive_rank = jnp.isfinite(sigmas) & (sigmas > 0.0)
        keep = tuple(int(i) for i in jnp.flatnonzero(positive_rank))
        if not keep:
            continue
        index = jnp.asarray(keep)
        training_block = sigmas[index, None] * q_by_site[site][index]
        validation_stack = jnp.stack(validation_blocks[site])[:, index]
        cosines = _block_transfer_cosines(training_block, validation_stack)
        if not bool(jnp.all(jnp.isfinite(cosines) & (cosines > 0.0))):
            continue
        site_batches.append(
            BirthSiteBatch(
                site=site,
                slots=tuple(slots[i] for i in keep),
                directions=p_by_site[site][:, index],
                sigmas=sigmas[index],
                validation_cosines=cosines,
            )
        )
    candidate = BirthBatchCandidate(tuple(site_batches))
    return candidate if candidate.size > 0 else None
