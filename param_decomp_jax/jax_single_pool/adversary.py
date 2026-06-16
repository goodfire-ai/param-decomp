"""The recon adversary: source state, ascent updates, and source→mask materialization.

Two semantically distinct adversaries share the source/mask machinery but nothing
else (SPEC §3):

- **Persistent PGD (PPGD)** — `PersistentPGDReconLossConfig`. Per-site `(1, T, C+1)`
  sources + their Adam moments live in `TrainState` across steps; each step runs
  `n_warmup_steps` supplemental Adam ascents plus one final ascent from the main
  backward (SPEC S13/S14), projecting to [0,1] after every update (S15).
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
from param_decomp_config.losses import (
    AdamPGDConfig,
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
)

SourceParameterization = Literal["clamp", "sigmoid"]
"""The `PROJ`/`EFFECTIVE` variation point (SPEC §6, line `PROJ`/`EFFECTIVE`). `clamp`:
sources live in `[0,1]`, projected by clamp after each ascent, read as-is, init U[0,1].
`sigmoid`: latent sources are unbounded (no projection), read through a sigmoid, init
N(0,1). Torch counterpart: `use_sigmoid_parameterization` in `persistent_pgd_state.py`."""


def source_parameterization(
    cfg: PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig,
) -> SourceParameterization:
    return "sigmoid" if cfg.use_sigmoid_parameterization else "clamp"


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcesAdamState:
    m: dict[str, Array]
    v: dict[str, Array]
    step_count: Float[Array, ""]


def init_persistent_sources(
    site_names: tuple[str, ...],
    site_component_counts: tuple[int, ...],
    seq_len: int,
    parameterization: SourceParameterization,
    key: PRNGKeyArray,
) -> dict[str, Array]:
    """PPGD `sc` scope (SPEC §1.6): `(1, T, C+1)` per site — shared across batch
    elements, free per position. Init is U[0,1] under `clamp`, N(0,1) (unbounded
    latent) under `sigmoid` (SPEC S15 / §6 `PROJ`/`EFFECTIVE`). Trailing channel =
    the weight-delta source."""
    keys = random.split(key, len(site_names))
    init_fn = random.normal if parameterization == "sigmoid" else random.uniform
    return {
        name: init_fn(k, (1, seq_len, c + 1), jnp.float32)
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


def sources_adam_ascend_project(
    sources: dict[str, Array],
    sources_grad: dict[str, Array],
    adam_state: SourcesAdamState,
    lr: Array,
    adam: AdamPGDConfig,
    parameterization: SourceParameterization,
) -> tuple[dict[str, Array], SourcesAdamState]:
    """One Adam ASCENT on the persistent sources, then `PROJ` (SPEC S13/S15). `PROJ` is
    clamp to [0,1] under `clamp`, identity under `sigmoid` (the latent stays unbounded
    and is squashed at read time by `source_masks`; SPEC §6 `PROJ`/`EFFECTIVE`).

    The variation point `SRC_STEP` (SPEC §6): a `sign` variant would replace the Adam
    update with `lr * sign(grad)` (stateless) — same projection contract."""
    step_count = adam_state.step_count + 1.0
    m = {s: adam.beta1 * adam_state.m[s] + (1 - adam.beta1) * sources_grad[s] for s in sources}
    v = {
        s: adam.beta2 * adam_state.v[s] + (1 - adam.beta2) * sources_grad[s] * sources_grad[s]
        for s in sources
    }
    bias_correction1 = 1 - adam.beta1**step_count
    bias_correction2 = 1 - adam.beta2**step_count
    ascended = {
        s: sources[s]
        + lr * (m[s] / bias_correction1) / (jnp.sqrt(v[s] / bias_correction2) + adam.eps)
        for s in sources
    }
    new_sources = (
        ascended
        if parameterization == "sigmoid"
        else {s: jnp.clip(a, 0.0, 1.0) for s, a in ascended.items()}
    )
    return new_sources, SourcesAdamState(m=m, v=v, step_count=step_count)


def source_masks(
    ci_lower: dict[str, Array],
    sources: dict[str, Array],
    site_names: tuple[str, ...],
    parameterization: SourceParameterization,
) -> tuple[dict[str, Array], dict[str, Array]]:
    """`mask = ci + (1−ci)·EFFECTIVE(source)[:, :C]`; delta mask = `EFFECTIVE(source)`
    trailing channel (SPEC S1). `EFFECTIVE` is identity under `clamp`, sigmoid under
    `sigmoid` (SPEC §6 `PROJ`/`EFFECTIVE`). Shared by both adversaries; sources
    broadcast over whatever leading dims their scope left singleton. The fp32 source
    state is cast to the CI dtype here (torch-under-autocast behavior); the source
    gradient flows back through the cast (and through the sigmoid)."""
    masks = {}
    delta_masks = {}
    for site in site_names:
        source = sources[site].astype(ci_lower[site].dtype)
        if parameterization == "sigmoid":
            source = jax.nn.sigmoid(source)
        masks[site] = ci_lower[site] + (1.0 - ci_lower[site]) * source[..., :-1]
        delta_masks[site] = source[..., -1]
    return masks, delta_masks
