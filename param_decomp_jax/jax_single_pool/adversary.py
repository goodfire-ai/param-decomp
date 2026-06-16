"""The recon adversary: source state, ascent updates, and source→mask materialization.

Two semantically distinct adversaries share the source/mask machinery but nothing
else (SPEC §3):

- **Persistent PGD (PPGD)** — `PersistentPGDReconLossConfig`. Per-site `(1, T, C+1)`
  sources + their SRC_STEP optimizer state live in `TrainState` across steps; each step
  runs `n_warmup_steps` supplemental ascents plus one final ascent from the main backward
  (SPEC S13/S14), projecting to [0,1] after every update (S15). The SRC_STEP variation
  point (SPEC §6) is `adam` (carries bias-corrected moments) or `sign` (stateless).
- **Fresh PGD** — `PGDReconLossConfig` (torch `PGDReconLoss` as a TRAINING loss).
  Sources are re-initialized every step, ascended `n_steps` times by
  `step_size * sign(grad)` with clamp to [0,1], and carry NO state across steps —
  `TrainState.sources` stays empty for this variant.
"""

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, PRNGKeyArray

from jax_single_pool.lm import SiteSpec
from param_decomp_config.losses import AdamPGDConfig, PGDOptimizerConfig, SignPGDConfig


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcesAdamState:
    m: dict[str, Array]
    v: dict[str, Array]
    step_count: Float[Array, ""]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcesSignState:
    """SRC_STEP `sign` variant carries no moments (SPEC §6); an empty pytree keeps the
    `TrainState.sources_opt_state` slot uniform across optimizer choices."""


SourcesOptState = SourcesAdamState | SourcesSignState


def init_persistent_sources(
    site_names: tuple[str, ...],
    site_component_counts: tuple[int, ...],
    seq_len: int,
    key: PRNGKeyArray,
) -> dict[str, Array]:
    """PPGD `sc` scope (SPEC §1.6): `(1, T, C+1)` per site — shared across batch
    elements, free per position — init U[0,1] (SPEC S15; clamp parameterization).
    Trailing channel = the weight-delta source."""
    keys = random.split(key, len(site_names))
    return {
        name: random.uniform(k, (1, seq_len, c + 1), jnp.float32)
        for name, c, k in zip(site_names, site_component_counts, keys, strict=True)
    }


def init_fresh_pgd_sources(
    sites: tuple[SiteSpec, ...],
    init: Literal["random", "ones", "zeroes"],
    scope: Literal["c", "bc", "bsc"],
    batch: int,
    seq: int,
    key: PRNGKeyArray,
) -> dict[str, Array]:
    """Per-site fresh adversarial sources (torch `_init_adv_sources`): trailing channel
    is the weight-delta source; shape per `scope` (shape-spelled: `bsc` ->
    `(B, T, C+1)`, `bc` -> `(B, 1, C+1)`, `c` -> `(1, 1, C+1)`)."""
    match scope:
        case "bsc":
            leading = (batch, seq)
        case "bc":
            leading = (batch, 1)
        case "c":
            leading = (1, 1)
    keys = random.split(key, len(sites))
    sources = {}
    for site, site_key in zip(sites, keys, strict=True):
        shape = (*leading, site.C + 1)
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


def init_sources_opt_state(
    sources: dict[str, Array], optimizer: PGDOptimizerConfig
) -> SourcesOptState:
    match optimizer:
        case AdamPGDConfig():
            return init_sources_adam_state(sources)
        case SignPGDConfig():
            return SourcesSignState()


def _sources_adam_ascend(
    sources: dict[str, Array],
    sources_grad: dict[str, Array],
    adam_state: SourcesAdamState,
    lr: Array,
    adam: AdamPGDConfig,
) -> tuple[dict[str, Array], SourcesAdamState]:
    step_count = adam_state.step_count + 1.0
    m = {s: adam.beta1 * adam_state.m[s] + (1 - adam.beta1) * sources_grad[s] for s in sources}
    v = {
        s: adam.beta2 * adam_state.v[s] + (1 - adam.beta2) * sources_grad[s] * sources_grad[s]
        for s in sources
    }
    bias_correction1 = 1 - adam.beta1**step_count
    bias_correction2 = 1 - adam.beta2**step_count
    new_sources = {
        s: sources[s]
        + lr * (m[s] / bias_correction1) / (jnp.sqrt(v[s] / bias_correction2) + adam.eps)
        for s in sources
    }
    return new_sources, SourcesAdamState(m=m, v=v, step_count=step_count)


def sources_ascend_project(
    sources: dict[str, Array],
    sources_grad: dict[str, Array],
    opt_state: SourcesOptState,
    lr: Array,
    optimizer: PGDOptimizerConfig,
) -> tuple[dict[str, Array], SourcesOptState]:
    """One ASCENT on the persistent sources via the SRC_STEP variation point (SPEC §6),
    then project to [0,1] (SPEC S13/S15). `adam` carries bias-corrected moments across
    steps; `sign` is stateless (`sources += lr * sign(grad)`)."""
    match optimizer:
        case AdamPGDConfig():
            assert isinstance(opt_state, SourcesAdamState)
            new_sources, new_opt_state = _sources_adam_ascend(
                sources, sources_grad, opt_state, lr, optimizer
            )
        case SignPGDConfig():
            assert isinstance(opt_state, SourcesSignState)
            new_sources = {s: sources[s] + lr * jnp.sign(sources_grad[s]) for s in sources}
            new_opt_state = opt_state
    return {s: jnp.clip(v, 0.0, 1.0) for s, v in new_sources.items()}, new_opt_state


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
