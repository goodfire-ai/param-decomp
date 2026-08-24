"""Adversarial source state, initialization, and optimization.

Two semantically distinct adversaries share source initialization and optimization but
nothing else (SPEC §3):

- **Persistent PGD (PPGD)** — `PersistentPGDReconLossConfig`. Per-site sources + their
  Adam moments live in `TrainState` across steps, stored per `source_shape`
  (`configs.SourceShape`).
  Each step runs `n_warmup_steps` supplemental Adam ascents plus one final ascent from
  the main backward (SPEC S13/S14), projecting to [0,1] after every update (S15).
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

from param_decomp.core.components import SiteSpec
from param_decomp.core.configs import AdamPGDConfig, PGDInitStrategy, SourceShape
from param_decomp.core.losses import scheduled_value_at


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SiteSource[T]:
    components: T
    delta: T


type Sources = dict[str, SiteSource[Array]]


def split_source_channels(packed: Array) -> SiteSource[Array]:
    return SiteSource(components=packed[..., :-1], delta=packed[..., -1])


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcesAdamState:
    m: Sources
    v: Sources
    step_count: Float[Array, ""]


def init_persistent_sources(
    site_names: tuple[str, ...],
    site_component_counts: tuple[int, ...],
    leading_shape: tuple[int, ...],
    source_dtype: DTypeLike,
    key: PRNGKeyArray,
) -> Sources:
    """Per-site PPGD component and weight-delta sources, initialized U[0,1].

    `leading_shape` spells
    the `source_shape` over the model's leading axes (SPEC §1.6), rank matching the
    waist with size-1 broadcast axes — e.g. an LM's `(1, T)` for `sc` (shared across
    batch, free per position), `(B, 1)` for `bc` (per batch element, shared over
    positions).

    `source_dtype` is the resident storage dtype (SPEC N1 fp32 for oracle parity; bf16 to
    halve footprint). Drawing in fp32 then casting keeps the U[0,1] draw dtype-stable."""
    keys = random.split(key, len(site_names))
    sources = {}
    for name, c, site_key in zip(site_names, site_component_counts, keys, strict=True):
        draw = random.uniform(site_key, (*leading_shape, c + 1), jnp.float32).astype(source_dtype)
        sources[name] = split_source_channels(draw)
    return sources


def sources_c_groups(
    site_names: tuple[str, ...], site_component_counts: tuple[int, ...]
) -> dict[int, tuple[str, ...]]:
    """Site names grouped by C, site order preserved within a group."""
    groups: dict[int, list[str]] = {}
    for name, c in zip(site_names, site_component_counts, strict=True):
        groups.setdefault(c, []).append(name)
    return {c: tuple(names) for c, names in groups.items()}


def init_persistent_sources_stacked(
    site_names: tuple[str, ...],
    site_component_counts: tuple[int, ...],
    leading_shape: tuple[int, ...],
    source_dtype: DTypeLike,
    key: PRNGKeyArray,
) -> dict[int, SiteSource[Array]]:
    """`init_persistent_sources` computed as same-C stacks,
    vmapped over the SAME per-site keys — `unstack_persistent_sources` recovers the
    per-site dict BIT-IDENTICALLY. Under jit the graph has n_C_groups sharded outputs
    instead of n_sites (the per-site form was a ~55s XLA compile at 224 sites)."""
    keys = random.split(key, len(site_names))
    site_index = {name: idx for idx, name in enumerate(site_names)}
    stacked: dict[int, SiteSource[Array]] = {}
    for c, names in sources_c_groups(site_names, site_component_counts).items():
        idxs = jnp.array([site_index[n] for n in names])
        draws = jax.vmap(lambda k, s=(*leading_shape, c + 1): random.uniform(k, s, jnp.float32))(
            keys[idxs]
        )
        draws = draws.astype(source_dtype)
        stacked[c] = split_source_channels(draws)
    return stacked


def unstack_persistent_sources(
    site_names: tuple[str, ...],
    site_component_counts: tuple[int, ...],
    stacked: dict[int, SiteSource[Array]],
) -> Sources:
    """Slice `init_persistent_sources_stacked`'s per-C stacks back into the per-site dict."""
    sources: Sources = {}
    for c, names in sources_c_groups(site_names, site_component_counts).items():
        assert stacked[c].components.shape[0] == len(names), (
            stacked[c].components.shape,
            len(names),
        )
        for j, name in enumerate(names):
            sources[name] = SiteSource(
                components=stacked[c].components[j], delta=stacked[c].delta[j]
            )
    return {name: sources[name] for name in site_names}


def init_fresh_pgd_sources(
    sites: tuple[SiteSpec, ...],
    init: PGDInitStrategy,
    source_shape: SourceShape,
    leading: tuple[int, ...],
    key: PRNGKeyArray,
) -> Sources:
    """Per-site fresh adversarial component and weight-delta sources.

    `leading = (B,) + position axes`. The source's leading
    shape spells `source_shape` (`configs.SourceShape`) over those axes: `bsc` keeps the
    full leading, `bc` collapses every position axis to 1 (`(B, 1)`), `c` collapses every
    axis to 1 (`(1, 1)`). The persistent counterpart enumerates its config-time shapes in
    `init_sources_sharded`."""
    batch, *positions = leading
    match source_shape:
        case "bsc":
            source_leading = leading
        case "bc":
            source_leading = (batch, *(1 for _ in positions))
        case "c":
            source_leading = tuple(1 for _ in leading)
        case "sc":
            raise AssertionError("unreachable: PGDConfig validation rejects `sc`")
    keys = random.split(key, len(sites))
    sources = {}
    for site, site_key in zip(sites, keys, strict=True):
        component_shape = (*source_leading, site.C)
        delta_shape = source_leading
        match init:
            case "random":
                sources[site.name] = split_source_channels(
                    random.uniform(site_key, (*source_leading, site.C + 1), jnp.float32)
                )
            case "ones":
                sources[site.name] = SiteSource(
                    jnp.ones(component_shape, jnp.float32), jnp.ones(delta_shape, jnp.float32)
                )
            case "zeroes":
                sources[site.name] = SiteSource(
                    jnp.zeros(component_shape, jnp.float32), jnp.zeros(delta_shape, jnp.float32)
                )
    return sources


def init_sources_adam_state(sources: Sources) -> SourcesAdamState:
    return SourcesAdamState(
        m=jax.tree.map(jnp.zeros_like, sources),
        v=jax.tree.map(jnp.zeros_like, sources),
        step_count=jnp.zeros(()),
    )


def sources_adam_ascend_project(
    sources: Sources,
    sources_grad: Sources,
    adam_state: SourcesAdamState,
    lr: Array,
    adam: AdamPGDConfig,
) -> tuple[Sources, SourcesAdamState]:
    """One Adam ASCENT on the persistent sources, then project to [0,1] (SPEC S13/S15).

    The variation point `SRC_STEP` (SPEC §6): a `sign` variant would replace the Adam
    update with `lr * sign(grad)` (stateless) — same projection contract."""
    step_count = adam_state.step_count + 1.0
    # `sources_grad` arrives in the masked-forward compute dtype (bf16); cast to the moment
    # dtype so the persistent `m`/`v` keep their declared storage dtype across steps.
    grad = jax.tree.map(lambda g, moment: g.astype(moment.dtype), sources_grad, adam_state.m)
    m = jax.tree.map(
        lambda moment, g: adam.beta1 * moment + (1 - adam.beta1) * g, adam_state.m, grad
    )
    v = jax.tree.map(
        lambda moment, g: adam.beta2 * moment + (1 - adam.beta2) * g * g, adam_state.v, grad
    )
    bias_correction1 = 1 - adam.beta1**step_count
    bias_correction2 = 1 - adam.beta2**step_count
    new_sources = jax.tree.map(
        lambda source, first, second: jnp.clip(
            source
            + (
                lr * (first / bias_correction1) / (jnp.sqrt(second / bias_correction2) + adam.eps)
            ).astype(source.dtype),
            0.0,
            1.0,
        ),
        sources,
        m,
        v,
    )
    return new_sources, SourcesAdamState(m=m, v=v, step_count=step_count)


class PersistentAdversary(eqx.Module):
    """One persistent-PGD adversary (SPEC §3): the per-site sources + their Adam moments
    that persist across steps, plus the lifecycle the trainer drives around the shared
    backward. `sources` / `opt_state` are dynamic state; the rest is static config.

    Per step: `warmup_ascend` (n_warmup supplemental ascents vs a scoring forward, params
    + CI detached) → the warmed sources enter the main `value_and_grad` as leaves →
    `final_ascend` (one more ascent from the SAME backward's source-grad — which IS
    `dL_term/d(sources)`: the source path is never coeff-scaled, SPEC S14'/S23)."""

    sources: Sources
    opt_state: SourcesAdamState
    state_key: str = eqx.field(static=True)
    adam: AdamPGDConfig = eqx.field(static=True)
    n_warmup: int = eqx.field(static=True)

    def source_lr(self, train_frac: Array) -> Array:
        return scheduled_value_at(train_frac, self.adam.lr_schedule)

    def warmup_ascend(
        self, scoring_loss: Callable[[Sources], Array], train_frac: Array
    ) -> "PersistentAdversary":
        """`n_warmup` supplemental Adam ascents on the sources vs `scoring_loss` (the
        route-all all-sites recon forward, params/CI detached — provided by the step). The
        warmed sources are `stop_gradient`'d: they enter the main backward as leaves, so
        the main graph differentiates w.r.t. them, not back through this scan."""
        lr = self.source_lr(train_frac)

        def body(
            carry: tuple[Sources, SourcesAdamState], _: None
        ) -> tuple[tuple[Sources, SourcesAdamState], None]:
            sources, opt = carry
            grad = jax.grad(scoring_loss)(sources)
            return sources_adam_ascend_project(sources, grad, opt, lr, self.adam), None

        (warmed, warmed_opt), _ = jax.lax.scan(
            body, (self.sources, self.opt_state), None, length=self.n_warmup
        )
        return eqx.tree_at(
            lambda a: (a.sources, a.opt_state), self, (jax.lax.stop_gradient(warmed), warmed_opt)
        )

    def after_one_adam_ascent(self, grad: Sources, train_frac: Array) -> "PersistentAdversary":
        """The adversary one Adam ascent-and-project (SPEC S13/S15) further along `grad`."""
        lr = self.source_lr(train_frac)
        sources, opt_state = sources_adam_ascend_project(
            self.sources, grad, self.opt_state, lr, self.adam
        )
        return eqx.tree_at(lambda a: (a.sources, a.opt_state), self, (sources, opt_state))

    def final_ascend(self, source_grad: Sources, train_frac: Array) -> "PersistentAdversary":
        """One final ascent recycled from the shared backward (SPEC S13'/S14'): the
        source path enters the backward UNSCALED — the term's coeff scales only the
        model-side cotangents (`train.model_cotangents_scaled`) — so `source_grad` IS
        `dL_term/d(sources)`, with nothing to unscale, at every step of any coeff
        schedule, activation gates included."""
        return self.after_one_adam_ascent(source_grad, train_frac)
