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

The target supplies each linear output's canonical activation key. One clean-forward
request unions those keys with the CI inputs, so frozen `x @ W.T` references and CI taps
come from one forward; the masked forward requests the same keys.
Masked and clean run in COMPUTE_DT (bf16, matching the trained model, mirroring
`load_run.py`); the MSE reduction is fp32.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import random
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.ci_fn import PlacedCIFn, evaluate_compute_ci, materialize_ci_compute_weights
from param_decomp.core.components import ComponentStacks
from param_decomp.core.jit_util import filter_jit
from param_decomp.core.linear_plan import uniform_like
from param_decomp.core.model import (
    CaptureKeys,
    MaterializedMasking,
    PlacedModel,
    prepare_compute_weights,
)
from param_decomp.core.precision import COMPUTE_DT


@dataclass(frozen=True)
class SiteMSEReduction:
    """Per-site `(Σ sum_mse, Σ n_elements)` across the eval pass (torch's
    `(summed_mse, n_elements)` accumulator). Combined and per-site means divide once."""

    sum_mse: float
    n_elements: int


def _per_site_sum_mse(
    masked_site_outputs_by_site: dict[str, Array],
    clean_site_outputs_by_site: dict[str, Array],
    site_names: tuple[str, ...],
) -> dict[str, Array]:
    """`Σ (masked − clean)^2` per site in fp32 (torch `F.mse_loss(reduction="sum")`)."""
    return {
        s: jnp.sum(
            (
                masked_site_outputs_by_site[s].astype(jnp.float32)
                - clean_site_outputs_by_site[s].astype(jnp.float32)
            )
            ** 2
        )
        for s in site_names
    }


HiddenActsStep = Callable[
    [PlacedModel, Any, Any, Any, PRNGKeyArray],
    tuple[dict[str, Array], dict[str, int]],
]
"""`(model, components, placed_ci_fn, inputs, key) -> ({site: sum_mse}, {site: n_elements})`
— one batch's per-site summed MSE (fp32) and element counts (the host folds these into
`SiteMSEReduction`s). `inputs` is the model's target-specific input (an LM's `[batch, seq]`
token ids), exactly as the unified clean/masked forwards take it. `key` is unused
by the deterministic CI step. `model` (frozen-weight-bearing) is the jit ARG."""


def make_ci_hidden_acts_step(
    model_static: PlacedModel,
    ci_capture_keys: CaptureKeys,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> HiddenActsStep:
    """Deterministic CI-mask hidden-acts step: `lower_leaky` CI, no delta, one forward."""
    site_names = model_static.site_names
    site_output_keys = model_static.site_output_keys(site_names)
    clean_capture_keys = ci_capture_keys | frozenset(site_output_keys)

    output_key_by_site = dict(zip(site_names, site_output_keys, strict=True))

    def step(
        model: PlacedModel,
        components: ComponentStacks,
        placed_ci_fn: PlacedCIFn,
        inputs: Any,
        _key: PRNGKeyArray,
    ) -> tuple[dict[str, Array], dict[str, int]]:
        clean_captures_by_key = model.clean_forward(inputs, clean_capture_keys).captures
        clean_ci_inputs_by_key = {key: clean_captures_by_key[key] for key in ci_capture_keys}
        clean_site_outputs_by_site = {
            site: clean_captures_by_key[key] for site, key in output_key_by_site.items()
        }
        prepared_weights = prepare_compute_weights(model, components)
        ci_lower = evaluate_compute_ci(
            materialize_ci_compute_weights(placed_ci_fn), clean_ci_inputs_by_key, remat=False
        ).lower

        masked_captures_by_key = model.masked_forward(
            prepared_weights,
            inputs,
            masking=MaterializedMasking(component_masks=ci_lower),
            capture_keys=frozenset(site_output_keys),
            remat=False,
        ).captures
        masked_site_outputs_by_site = {
            site: masked_captures_by_key[key] for site, key in output_key_by_site.items()
        }
        sum_mse = _per_site_sum_mse(
            masked_site_outputs_by_site, clean_site_outputs_by_site, site_names
        )
        n_elements = {site: clean_site_outputs_by_site[site].size for site in site_names}
        return sum_mse, n_elements

    return filter_jit(step, compiler_options=compiler_options)


def make_stochastic_hidden_acts_step(
    model_static: PlacedModel,
    ci_capture_keys: CaptureKeys,
    n_mask_samples: int,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> HiddenActsStep:
    """Stochastic-mask hidden-acts step: `n_mask_samples` draws of `mask = ci + (1−ci)·s`
    (with weight deltas), per-draw per-site MSE summed. RNG via per-draw / per-site
    `fold_in` (the eval-step discipline)."""
    assert n_mask_samples >= 1, n_mask_samples
    site_names = model_static.site_names
    site_output_keys = model_static.site_output_keys(site_names)
    clean_capture_keys = ci_capture_keys | frozenset(site_output_keys)

    output_key_by_site = dict(zip(site_names, site_output_keys, strict=True))

    def step(
        model: PlacedModel,
        components: ComponentStacks,
        placed_ci_fn: PlacedCIFn,
        inputs: Any,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, Array], dict[str, int]]:
        clean_captures_by_key = model.clean_forward(inputs, clean_capture_keys).captures
        clean_ci_inputs_by_key = {key: clean_captures_by_key[key] for key in ci_capture_keys}
        clean_site_outputs_by_site = {
            site: clean_captures_by_key[key] for site, key in output_key_by_site.items()
        }
        prepared_weights = prepare_compute_weights(model, components)
        ci_lower = evaluate_compute_ci(
            materialize_ci_compute_weights(placed_ci_fn), clean_ci_inputs_by_key, remat=False
        ).lower

        sum_mse = {s: jnp.zeros((), jnp.float32) for s in site_names}
        for draw_idx in range(n_mask_samples):
            mask_key, delta_key = random.split(random.fold_in(key, draw_idx))
            masks = {}
            delta_masks = {}
            for site_idx, site in enumerate(site_names):
                ci_site = ci_lower[site]
                source_key = random.fold_in(mask_key, site_idx)
                source = uniform_like(source_key, ci_site, dtype=COMPUTE_DT)
                masks[site] = ci_site + (1.0 - ci_site) * source
                delta_masks[site] = uniform_like(
                    random.fold_in(delta_key, site_idx),
                    ci_site,
                    drop_last_axis=True,
                    dtype=COMPUTE_DT,
                )
            masked_captures_by_key = model.masked_forward(
                prepared_weights,
                inputs,
                masking=MaterializedMasking(component_masks=masks, weight_delta_masks=delta_masks),
                capture_keys=frozenset(site_output_keys),
                remat=False,
            ).captures
            masked_site_outputs_by_site = {
                site: masked_captures_by_key[key] for site, key in output_key_by_site.items()
            }
            draw_sum = _per_site_sum_mse(
                masked_site_outputs_by_site, clean_site_outputs_by_site, site_names
            )
            sum_mse = {s: sum_mse[s] + draw_sum[s] for s in site_names}

        n_elements = {
            site: clean_site_outputs_by_site[site].size * n_mask_samples for site in site_names
        }
        return sum_mse, n_elements

    return filter_jit(step, compiler_options=compiler_options)


def accumulate_hidden_acts(
    step: HiddenActsStep,
    model: PlacedModel,
    components: ComponentStacks,
    placed_ci_fn: PlacedCIFn,
    input_batches: list[Any],
    base_key: PRNGKeyArray,
) -> dict[str, SiteMSEReduction]:
    """Drive `step` over the eval batches, host-accumulating `(Σ sum_mse, Σ n)` per site
    (token-weighted, exact under micro-batching — same pattern as the density/mean
    accumulators). Per-batch RNG is `fold_in(base_key, batch_idx)`."""
    assert input_batches, "hidden-acts eval needs at least one batch"
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch_idx, inputs in enumerate(input_batches):
        batch_sum, batch_n = step(
            model, components, placed_ci_fn, inputs, random.fold_in(base_key, batch_idx)
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
    log_entries = {
        f"{class_name}/{site}": reduction.sum_mse / reduction.n_elements
        for site, reduction in reductions.items()
    }
    total_sum = sum(r.sum_mse for r in reductions.values())
    total_n = sum(r.n_elements for r in reductions.values())
    log_entries[class_name] = total_sum / total_n
    return log_entries
