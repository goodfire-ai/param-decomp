"""JAX-native hidden-activation reconstruction eval metrics (`CIHiddenActsReconLoss`,
`StochasticHiddenActsReconLoss`), the offline counterparts of the torch eval metrics of
the same names (`param_decomp/metrics/stochastic_hidden_acts_recon.py`).

Both compute, per decomposed site, `MSE(masked_site_output, clean_site_output)` with
`reduction="sum"` (torch `F.mse_loss(reduction="sum")`), divided once by the element
count at the end. Log keys mirror torch exactly: `"<ClassName>/<site>"` per site plus a
combined `"<ClassName>"` (= Σ_sites sum_mse / Σ_sites n_elements).

- `CIHiddenActsReconLoss`: the deterministic CI mask (`lower_leaky`), no stochastic draw,
  no weight delta — one masked forward per batch.
- `StochasticHiddenActsReconLoss`: `n_mask_samples` stochastic CI-mask draws WITH weight
  deltas (the delta component is always built in JAX runs), per-draw per-site MSE
  accumulated. The stochastic
  draws are NOT seed-aligned to torch, so exact bitwise parity is impossible for this one
  (expected); the deterministic CI variant is tight.

The clean (target) per-site output is the frozen `x @ W`: `masked_site_outputs` with
every site live but routed FALSE everywhere falls onto `site_out`'s frozen `x @ W.T`
branch — reusing the one seam instead of a separate frozen-W accessor. Masked and clean
both run in COMPUTE_DT (bf16, matching the trained model, mirroring `load_run.py`); the
MSE reduction is fp32.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import random
from jaxtyping import Array, Float, PRNGKeyArray

from param_decomp.components import DecompVU
from param_decomp.jit_util import filter_jit
from param_decomp.lm import DecomposedModel, all_false_routes
from param_decomp.train import COMPUTE_DT, cast_floating


@dataclass(frozen=True)
class SiteMSEReduction:
    """Per-site `(Σ sum_mse, Σ n_elements)` across the eval pass (torch's
    `(summed_mse, n_elements)` accumulator). Combined and per-site means divide once."""

    sum_mse: float
    n_elements: int


def _per_site_sum_mse(
    masked: dict[str, Array], clean: dict[str, Array], site_names: tuple[str, ...]
) -> dict[str, Array]:
    """`Σ (masked − clean)^2` per site in fp32 (torch `F.mse_loss(reduction="sum")`)."""
    return {
        s: jnp.sum((masked[s].astype(jnp.float32) - clean[s].astype(jnp.float32)) ** 2)
        for s in site_names
    }


HiddenActsStep = Callable[
    [DecomposedModel, Any, Any, Float[Array, "*leading d"], PRNGKeyArray],
    tuple[dict[str, Array], dict[str, int]],
]
"""`(model, components, ci_fn, residual, key) -> ({site: sum_mse}, {site: n_elements})`
— one batch's per-site summed MSE (fp32) and element counts (the host folds these into
`SiteMSEReduction`s). `key` is unused by the deterministic CI step. `model`
(frozen-weight-bearing) is the jit ARG."""


def make_ci_hidden_acts_step(
    lm: DecomposedModel, compiler_options: dict[str, bool | int | str] | None = None
) -> HiddenActsStep:
    """Deterministic CI-mask hidden-acts step: `lower_leaky` CI, no delta, one forward."""
    site_names = lm.site_names

    def step(
        model: DecomposedModel,
        components: DecompVU,
        ci_fn: Any,
        residual: Float[Array, "*leading d"],
        _key: PRNGKeyArray,
    ) -> tuple[dict[str, Array], dict[str, int]]:
        taps = model.read_activations(residual, ci_fn.input_names)
        components_bf16 = cast_floating(components, COMPUTE_DT)
        prepared = model.prepare_compute_weights(components_bf16)
        ci_fn_bf16 = cast_floating(ci_fn, COMPUTE_DT)
        ci_lower = ci_fn_bf16(taps, remat=False).lower

        # `residual` is the model INPUT (tokens for an LM — no trailing d), so the waist
        # `*leading` comes off the CI output (`[*leading, C]`), not off `residual`.
        leading = ci_lower[site_names[0]].shape[:-1]
        zeros_delta = {s: jnp.zeros(leading, COMPUTE_DT) for s in site_names}
        clean = model.masked_site_outputs(
            prepared, residual,
            {s: jnp.ones_like(ci_lower[s]) for s in site_names}, zeros_delta,
            all_false_routes(site_names, leading), site_names, False,
        )  # fmt: skip
        masked = model.masked_site_outputs(
            prepared, residual, ci_lower, zeros_delta, None, site_names, False
        )
        sum_mse = _per_site_sum_mse(masked, clean, site_names)
        n_elements = {s: clean[s].size for s in site_names}
        return sum_mse, n_elements

    return filter_jit(step, compiler_options=compiler_options)


def make_stochastic_hidden_acts_step(
    lm: DecomposedModel,
    n_mask_samples: int,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> HiddenActsStep:
    """Stochastic-mask hidden-acts step: `n_mask_samples` draws of `mask = ci + (1−ci)·s`
    (with weight deltas), per-draw per-site MSE summed. RNG via per-draw / per-site
    `fold_in` (the eval-step discipline)."""
    assert n_mask_samples >= 1, n_mask_samples
    site_names = lm.site_names

    def step(
        model: DecomposedModel,
        components: DecompVU,
        ci_fn: Any,
        residual: Float[Array, "*leading d"],
        key: PRNGKeyArray,
    ) -> tuple[dict[str, Array], dict[str, int]]:
        taps = model.read_activations(residual, ci_fn.input_names)
        components_bf16 = cast_floating(components, COMPUTE_DT)
        prepared = model.prepare_compute_weights(components_bf16)
        ci_fn_bf16 = cast_floating(ci_fn, COMPUTE_DT)
        ci_lower = ci_fn_bf16(taps, remat=False).lower

        # `*leading` off the CI output, not `residual` — see make_ci_hidden_acts_step.
        leading = ci_lower[site_names[0]].shape[:-1]
        clean = model.masked_site_outputs(
            prepared, residual,
            {s: jnp.ones_like(ci_lower[s]) for s in site_names},
            {s: jnp.zeros(leading, COMPUTE_DT) for s in site_names},
            all_false_routes(site_names, leading), site_names, False,
        )  # fmt: skip

        sum_mse = {s: jnp.zeros((), jnp.float32) for s in site_names}
        for draw_idx in range(n_mask_samples):
            mask_key, delta_key = random.split(random.fold_in(key, draw_idx))
            masks = {}
            delta_masks = {}
            for site_idx, site in enumerate(site_names):
                ci_site = ci_lower[site]
                source_key = random.fold_in(mask_key, site_idx)
                source = random.uniform(source_key, ci_site.shape, COMPUTE_DT)
                masks[site] = ci_site + (1.0 - ci_site) * source
                delta_masks[site] = random.uniform(
                    random.fold_in(delta_key, site_idx), leading, COMPUTE_DT
                )
            masked = model.masked_site_outputs(
                prepared, residual, masks, delta_masks, None, site_names, True
            )
            draw_sum = _per_site_sum_mse(masked, clean, site_names)
            sum_mse = {s: sum_mse[s] + draw_sum[s] for s in site_names}

        n_elements = {s: clean[s].size * n_mask_samples for s in site_names}
        return sum_mse, n_elements

    return filter_jit(step, compiler_options=compiler_options)


def accumulate_hidden_acts(
    step: HiddenActsStep,
    model: DecomposedModel,
    components: DecompVU,
    ci_fn: Any,
    residual_batches: list[Float[Array, "*leading d"]],
    base_key: PRNGKeyArray,
) -> dict[str, SiteMSEReduction]:
    """Drive `step` over the eval batches, host-accumulating `(Σ sum_mse, Σ n)` per site
    (token-weighted, exact under micro-batching — same pattern as the density/mean
    accumulators). Per-batch RNG is `fold_in(base_key, batch_idx)`."""
    assert residual_batches, "hidden-acts eval needs at least one batch"
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch_idx, residual in enumerate(residual_batches):
        batch_sum, batch_n = step(
            model, components, ci_fn, residual, random.fold_in(base_key, batch_idx)
        )
        for site in batch_sum:
            sums[site] = sums.get(site, 0.0) + float(np.asarray(batch_sum[site]))
            counts[site] = counts.get(site, 0) + int(batch_n[site])
    return {s: SiteMSEReduction(sum_mse=sums[s], n_elements=counts[s]) for s in sums}


def hidden_acts_log_entries(
    class_name: str, reductions: dict[str, SiteMSEReduction]
) -> dict[str, float]:
    """`{class_name/site: mse}` per site plus a combined `{class_name}` (torch
    `compute_per_module_metrics`): per-site is `sum_mse/n`, combined is `Σ sum_mse / Σ n`
    over all sites."""
    assert reductions, "no hidden-acts data accumulated"
    out = {f"{class_name}/{s}": r.sum_mse / r.n_elements for s, r in reductions.items()}
    total_sum = sum(r.sum_mse for r in reductions.values())
    total_n = sum(r.n_elements for r in reductions.values())
    out[class_name] = total_sum / total_n
    return out
