"""The recon adversary: source state, ascent updates, and source→mask materialization.

Two semantically distinct adversaries share the source/mask machinery but nothing
else (SPEC §3):

- **Persistent PGD (PPGD)** — `PersistentPGDReconLossConfig`. Per-site sources + their
  `SRC_STEP` optimizer state (Adam `m`/`v`; nothing for the stateless `sign` variant)
  live in `TrainState` across steps; `sc` scope is `(1, T, C+1)` (shared across batch),
  `bsc` is `(B, T, C+1)` (independent per batch element, batch-sharded). Each step runs
  `n_warmup_steps` supplemental ascents plus one final ascent from the main backward
  (SPEC S13/S14), projecting to [0,1] after every update (S15).
- **Fresh PGD** — `PGDReconLossConfig` (torch `PGDReconLoss` as a TRAINING loss).
  Sources are re-initialized every step, ascended `n_steps` times by
  `step_size * sign(grad)` with clamp to [0,1], and carry NO state across steps —
  `TrainState.adversaries` stays empty for this variant.
"""

from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jax.typing import DTypeLike
from jaxtyping import Array, Float, PRNGKeyArray

from param_decomp.components import SiteSpec
from param_decomp.configs import (
    AdamPGDConfig,
    MaskScopeLiteral,
    PGDInitStrategy,
    SignPGDConfig,
)
from param_decomp.losses import scheduled_value_traced


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcesAdamState:
    m: dict[str, Array]
    v: dict[str, Array]
    step_count: Float[Array, ""]


def init_persistent_sources(
    site_names: tuple[str, ...],
    site_component_counts: tuple[int, ...],
    leading_shape: tuple[int, ...],
    source_dtype: DTypeLike,
    key: PRNGKeyArray,
) -> dict[str, Array]:
    """Per-site PPGD sources `(*leading_shape, C+1)`, init U[0,1] (SPEC S15; clamp
    parameterization); trailing channel = the weight-delta source. `leading_shape` spells
    the scope over the model's leading axes (SPEC §1.6): for an LM `(1, T)` for `sc`
    (batch axis collapsed -> shared across batch, free per position) and `(B, T)` for `bsc`
    (independent per batch element and position).

    `source_dtype` is the resident storage dtype (SPEC N1 fp32 for oracle parity; bf16 to
    halve footprint). Drawing in fp32 then casting keeps the U[0,1] draw dtype-stable."""
    keys = random.split(key, len(site_names))
    return {
        name: random.uniform(k, (*leading_shape, c + 1), jnp.float32).astype(source_dtype)
        for name, c, k in zip(site_names, site_component_counts, keys, strict=True)
    }


def init_fresh_pgd_sources(
    sites: tuple[SiteSpec, ...],
    init: PGDInitStrategy,
    scope: MaskScopeLiteral,
    leading: tuple[int, ...],
    key: PRNGKeyArray,
) -> dict[str, Array]:
    """Per-site fresh adversarial sources (torch `_init_adv_sources`): trailing channel
    is the weight-delta source; `leading = (B,) + position axes`. The source's leading
    shape spells `scope` over those axes (LM `(B, T)`): `bsc` keeps the full leading,
    `bc` collapses every position axis to 1 (`(B, 1)`), `c` collapses every axis to 1
    (`(1, 1)`)."""
    batch, *positions = leading
    match scope:
        case "bsc":
            source_leading = leading
        case "bc":
            source_leading = (batch, *(1 for _ in positions))
        case "c":
            source_leading = tuple(1 for _ in leading)
    keys = random.split(key, len(sites))
    sources = {}
    for site, site_key in zip(sites, keys, strict=True):
        shape = (*source_leading, site.C + 1)
        match init:
            case "random":
                sources[site.name] = random.uniform(site_key, shape, jnp.float32)
            case "ones":
                sources[site.name] = jnp.ones(shape, jnp.float32)
            case "zeroes":
                sources[site.name] = jnp.zeros(shape, jnp.float32)
    return sources


def init_sources_adam_state(sources: dict[str, Array]) -> SourcesAdamState:
    return SourcesAdamState(
        m={site: jnp.zeros_like(v) for site, v in sources.items()},
        v={site: jnp.zeros_like(v) for site, v in sources.items()},
        step_count=jnp.zeros(()),
    )


def sources_ascend_project(
    sources: dict[str, Array],
    sources_grad: dict[str, Array],
    opt_state: SourcesAdamState | None,
    lr: Array,
    optimizer: AdamPGDConfig | SignPGDConfig,
) -> tuple[dict[str, Array], SourcesAdamState | None]:
    """One `SRC_STEP` ASCENT on the persistent sources, then project to [0,1]
    (SPEC S13/S15). The variation point `SRC_STEP` (SPEC §6): `adam` carries persistent
    `m`/`v` moments (`opt_state`); `sign` is stateless (`opt_state is None`) — same
    projection contract."""
    match optimizer:
        case AdamPGDConfig():
            assert opt_state is not None
            return sources_adam_ascend_project(sources, sources_grad, opt_state, lr, optimizer)
        case SignPGDConfig():
            assert opt_state is None
            new_sources = {
                s: jnp.clip(
                    sources[s] + (lr * jnp.sign(sources_grad[s])).astype(sources[s].dtype),
                    0.0,
                    1.0,
                )
                for s in sources
            }
            return new_sources, None


def sources_adam_ascend_project(
    sources: dict[str, Array],
    sources_grad: dict[str, Array],
    adam_state: SourcesAdamState,
    lr: Array,
    adam: AdamPGDConfig,
) -> tuple[dict[str, Array], SourcesAdamState]:
    step_count = adam_state.step_count + 1.0
    # `sources_grad` arrives in the masked-forward compute dtype (bf16); cast to the moment
    # dtype so the persistent `m`/`v` keep their declared storage dtype across steps.
    grad = {s: sources_grad[s].astype(adam_state.m[s].dtype) for s in sources}
    m = {s: adam.beta1 * adam_state.m[s] + (1 - adam.beta1) * grad[s] for s in sources}
    v = {s: adam.beta2 * adam_state.v[s] + (1 - adam.beta2) * grad[s] * grad[s] for s in sources}
    bias_correction1 = 1 - adam.beta1**step_count
    bias_correction2 = 1 - adam.beta2**step_count
    new_sources = {
        s: jnp.clip(
            sources[s]
            + (
                lr * (m[s] / bias_correction1) / (jnp.sqrt(v[s] / bias_correction2) + adam.eps)
            ).astype(sources[s].dtype),
            0.0,
            1.0,
        )
        for s in sources
    }
    return new_sources, SourcesAdamState(m=m, v=v, step_count=step_count)


def source_masks(
    ci_lower: dict[str, Array], sources: dict[str, Array], site_names: tuple[str, ...]
) -> tuple[dict[str, Array], dict[str, Array]]:
    """`mask = ci + (1−ci)·source[:, :C]`; delta mask = raw trailing channel (SPEC S1).
    Shared by both adversaries; sources broadcast over whatever leading dims their
    scope left singleton. The fp32 source state is cast to the CI dtype here
    (torch-under-autocast behavior); the source gradient flows back through the cast."""
    masks = {}
    delta_masks = {}
    for site in site_names:
        source = sources[site].astype(ci_lower[site].dtype)
        masks[site] = ci_lower[site] + (1.0 - ci_lower[site]) * source[..., :-1]
        delta_masks[site] = source[..., -1]
    return masks, delta_masks


class PersistentAdversary(eqx.Module):
    """One persistent-PGD adversary (SPEC §3): the per-site sources + their `SRC_STEP`
    optimizer state that persist across steps, plus the lifecycle the trainer drives
    around the shared backward. `sources` / `opt_state` are dynamic state (`opt_state`
    is None iff the optimizer is the stateless `sign` variant); the rest is static
    config.

    Per step: `warmup_ascend` (n_warmup supplemental ascents vs a scoring forward, params
    + CI detached) → the warmed sources enter the main `value_and_grad` as leaves →
    `final_ascend` (one more ascent from the SAME backward's source-grad, unscaled by the
    term's `coeff` — exact since one source bundle feeds exactly one term, SPEC S23)."""

    sources: dict[str, Array]  # site -> source in [0,1], `(*scope_leading, C+1)`
    opt_state: SourcesAdamState | None
    state_key: str = eqx.field(static=True)
    coeff: float = eqx.field(static=True)
    optimizer: AdamPGDConfig | SignPGDConfig = eqx.field(static=True)
    n_warmup: int = eqx.field(static=True)

    def source_lr(self, step_f32: Array, total_steps: int) -> Array:
        return scheduled_value_traced(step_f32, total_steps, self.optimizer.lr_schedule)

    def warmup_ascend(
        self, scoring_loss: Callable[[dict[str, Array]], Array], step_f32: Array, total_steps: int
    ) -> "PersistentAdversary":
        """`n_warmup` supplemental Adam ascents on the sources vs `scoring_loss` (the
        route-all all-sites recon forward, params/CI detached — provided by the step). The
        warmed sources are `stop_gradient`'d: they enter the main backward as leaves, so
        the main graph differentiates w.r.t. them, not back through this scan."""
        lr = self.source_lr(step_f32, total_steps)

        def body(
            carry: tuple[dict[str, Array], SourcesAdamState | None], _: None
        ) -> tuple[tuple[dict[str, Array], SourcesAdamState | None], None]:
            sources, opt = carry
            grad = jax.grad(scoring_loss)(sources)
            return sources_ascend_project(sources, grad, opt, lr, self.optimizer), None

        (warmed, warmed_opt), _ = jax.lax.scan(
            body, (self.sources, self.opt_state), None, length=self.n_warmup
        )
        return eqx.tree_at(
            lambda a: (a.sources, a.opt_state), self, (jax.lax.stop_gradient(warmed), warmed_opt)
        )

    def final_ascend(
        self, source_grad_scaled: dict[str, Array], step_f32: Array, total_steps: int
    ) -> "PersistentAdversary":
        """One final ascent from the shared backward's source-grad (SPEC S13'/S14'). The
        backward saw `coeff·L_term`, so the grad is unscaled by `coeff` to ascend on
        `L_term` itself (exact: each source bundle feeds exactly one term, SPEC S23)."""
        lr = self.source_lr(step_f32, total_steps)
        grad = {s: g / self.coeff for s, g in source_grad_scaled.items()}
        ascended, ascended_opt = sources_ascend_project(
            self.sources, grad, self.opt_state, lr, self.optimizer
        )
        return eqx.tree_at(lambda a: (a.sources, a.opt_state), self, (ascended, ascended_opt))
