"""In-loop eval pass: scalar parity with the torch eval metrics.

Implements independent pure JAX kernels for the scalar core of the torch reference
`eval:` block: `CEandKLLosses` (six masking variants), `CI_L0`, and fresh PGD
reconstruction. Each authored operation compiles only its own kernel; `make_eval_step`
composes them only for the fixed arithmetic probe and parity tests.
Plot-type metrics (CI histograms, activation density, per-component means, the
permutation/UV figures) ride the in-loop SLOW tier instead — natively in JAX
(`slow_eval.py`, SPEC S28; in-loop only, no offline CLI).

Variant semantics mirror `param_decomp/eval_metrics/ce_and_kl_losses.py`: each
variant is a masked forward with ALL sites live and no routing; only `stoch_masked`
carries a weight-delta mask (torch `make_mask_infos` without weight deltas drops the
delta term — delta mask 0 here). CE is next-token cross-entropy with the first label
ignored; KL is per-position vs the clean (frozen) logits.

Cross-batch aggregation (the multi-`n_steps` eval pass in `run.py`): every key this
function returns is a per-BATCH scalar that the caller averages uniformly over the
eval batches. This is mean-safe against the torch reference — i.e. it matches torch's
accumulate-then-`compute()` to within float reassociation — only because every emitted
key is itself a per-batch reduction that torch *also* averages across batches, and the
eval batches are uniform `(B, T)`. The S8/D2 Jensen trap (a nonlinearity applied AFTER
the cross-batch reduction, so mean-of-batch-results ≠ result-of-global-batch) does NOT
arise here, because no emitted key wraps the cross-batch axis in a nonlinearity:

- `ce_kl/kl_<variant>`: torch `CEandKLLosses` accumulates `kl * n_positions` and divides
  by total positions (token-weighted mean of a per-batch mean). Uniform `(B, T)` makes
  token-weighting equal to the uniform `1/n_steps` average here.
- `ce_kl/ce_difference_<variant>` = `ce_v - ce_target`: torch averages this per-batch
  DIFFERENCE (computed inside `_calc_ce_and_kl_losses`), not a difference of grand means.
  Linear, so uniform-average parity holds.
- `l0/<threshold>_<site|group>`: torch `CI_L0` collects per-batch L0 and averages them
  uniformly (`sum / count`); group L0 is a per-batch sum of member L0s. Linear.
- `loss/PGDReconLoss`: torch `PGDReconLoss` accumulates `kl * n` over batches and divides
  by total `n` (example-weighted mean of a per-batch mean KL); equals the uniform average
  under uniform `(B, T)`.

At `eval.n_steps: 1` the cross-batch average is a no-op; the parity argument above is
what keeps it correct when `n_steps` is raised.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh
from jaxtyping import Array, Float, Int, PRNGKeyArray

from param_decomp.core.ci_fn import PlacedCIFn, evaluate_ci
from param_decomp.core.ci_l0_eval import ci_l0_scalars, resolve_site_groups
from param_decomp.core.components import ComponentStacks
from param_decomp.core.decomposed_linear import constrain_component_activation
from param_decomp.core.jit_util import filter_jit
from param_decomp.core.linear_plan import uniform_like
from param_decomp.core.losses import (
    ReconstructionLoss,
    reconstruction_loss,
    reconstruction_loss_metrics,
)
from param_decomp.core.masking import masks_from_sources
from param_decomp.core.model import (
    EMPTY_CAPTURE_KEYS,
    CaptureKeys,
    MaterializedMasking,
    PlacedModel,
    prepare_compute_weights,
    select_captures,
)
from param_decomp.core.precision import COMPUTE_DT
from param_decomp.core.recon import ForwardObservations, reconstruction_observations
from param_decomp.core.recon_eval import FreshPGDReconEval, fresh_pgd_recon_sources
from param_decomp.core.sharding import batch_shard_leading
from param_decomp.targets.losses import kl_per_position

type ScalarStep = Callable[
    [PlacedModel, ComponentStacks, PlacedCIFn, Array, PRNGKeyArray], Mapping[str, Array]
]


def next_token_cross_entropy(
    logits: Float[Array, "B T vocab"], token_ids: Int[Array, "B T"]
) -> Array:
    """Mean fp32 CE of positions 0..T-2 predicting tokens 1..T-1 (torch: labels with
    the first position set to ignore_index)."""
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    label_log_probs = jnp.take_along_axis(log_probs[:, :-1], token_ids[:, 1:, None], axis=-1)[
        ..., 0
    ]
    return -label_log_probs.mean()


def _row_masked_mean(per_position: Float[Array, "B ..."], row_mask: Float[Array, " B"]) -> Array:
    """Mean of `per_position` over the rows where `row_mask` is 1 (all positions of a masked
    row weigh 0). `per_position` is fp32 `(B, *positions)`."""
    positions_per_row = math.prod(per_position.shape[1:])
    mask = row_mask.reshape(row_mask.shape[0], *((1,) * (per_position.ndim - 1)))
    return jnp.sum(per_position * mask) / (jnp.sum(row_mask) * positions_per_row)


def _row_masked_kl(
    masked_output: Float[Array, "B T vocab"],
    clean_output: Float[Array, "B T vocab"],
    row_mask: Float[Array, " B"],
) -> Array:
    """`kl_per_position` restricted to the rows where `row_mask` is 1 (same fp32 math,
    per-position KL weighted before the mean)."""
    log_q = jax.nn.log_softmax(masked_output.astype(jnp.float32), axis=-1)
    log_p = jax.nn.log_softmax(clean_output.astype(jnp.float32), axis=-1)
    p = jnp.exp(log_p)
    return _row_masked_mean(jnp.sum(p * (log_p - log_q), axis=-1), row_mask)


def _row_masked_cross_entropy(
    logits: Float[Array, "B T vocab"], token_ids: Int[Array, "B T"], row_mask: Float[Array, " B"]
) -> Array:
    """`next_token_cross_entropy` restricted to the rows where `row_mask` is 1."""
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    label_log_probs = jnp.take_along_axis(log_probs[:, :-1], token_ids[:, 1:, None], axis=-1)[
        ..., 0
    ]
    return _row_masked_mean(-label_log_probs, row_mask)


@dataclass(frozen=True)
class _PreparedLMBatch[PreparedT]:
    tokens: Array
    clean: ForwardObservations
    prepared_weights: PreparedT
    ci_lower: dict[str, Array]
    valid_row_mask: Array | None


def _prepare_lm_batch[PreparedT](
    model: PlacedModel[PreparedT],
    components: ComponentStacks,
    placed_ci_fn: PlacedCIFn,
    token_ids: Int[Array, "B T"],
    mesh: Mesh | None,
    n_valid_rows: int | None,
    ci_capture_keys: CaptureKeys,
    activation_capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
) -> _PreparedLMBatch[PreparedT]:
    """Pure shared preparation required by independent LM metric kernels."""
    tokens = batch_shard_leading(token_ids, mesh)
    capture_keys = ci_capture_keys | activation_capture_keys
    clean_forward_result = model.clean_forward(tokens, capture_keys)
    ci_input_activations = select_captures(clean_forward_result.captures, ci_capture_keys)
    clean = reconstruction_observations(
        clean_forward_result,
        hidden_acts_capture_keys=activation_capture_keys,
        mesh=mesh,
    )
    ci_lower = evaluate_ci(placed_ci_fn, ci_input_activations, remat=False).lower
    ci_lower = {
        site: constrain_component_activation(value, model.placement)
        for site, value in ci_lower.items()
    }
    valid_row_mask = None
    if n_valid_rows is not None:
        assert n_valid_rows <= tokens.shape[0], (n_valid_rows, tokens.shape)
        valid_row_mask = (jnp.arange(tokens.shape[0]) < n_valid_rows).astype(jnp.float32)
    prepared_weights = prepare_compute_weights(model, components)
    return _PreparedLMBatch(
        tokens=tokens,
        clean=clean,
        prepared_weights=prepared_weights,
        ci_lower=ci_lower,
        valid_row_mask=valid_row_mask,
    )


def _compute_masked_output[PreparedT](
    model: PlacedModel[PreparedT],
    batch: _PreparedLMBatch[PreparedT],
    masks: dict[str, Array],
    delta_masks: dict[str, Array],
    mesh: Mesh | None,
    capture_keys: CaptureKeys,
) -> Array:
    masked_forward_result = model.masked_forward(
        batch.prepared_weights,
        batch.tokens,
        masking=MaterializedMasking(component_masks=masks, weight_delta_masks=delta_masks),
        capture_keys=capture_keys,
        remat=False,
    )
    return batch_shard_leading(masked_forward_result.output, mesh)


def _kl[PreparedT](batch: _PreparedLMBatch[PreparedT], logits: Array) -> Array:
    if batch.valid_row_mask is None:
        return kl_per_position(logits, batch.clean.output)
    return _row_masked_kl(logits, batch.clean.output, batch.valid_row_mask)


def _ce[PreparedT](batch: _PreparedLMBatch[PreparedT], logits: Array) -> Array:
    if batch.valid_row_mask is None:
        return next_token_cross_entropy(logits, batch.tokens)
    return _row_masked_cross_entropy(logits, batch.tokens, batch.valid_row_mask)


def make_ce_kl_step[PreparedT](
    model_static: PlacedModel[PreparedT],
    ci_capture_keys: CaptureKeys,
    rounding_threshold: float,
    mesh: Mesh | None = None,
    compiler_options: dict[str, bool | int | str] | None = None,
    *,
    n_valid_rows: int | None = None,
) -> ScalarStep:
    """Build the single-purpose CE/KL evaluator."""
    assert model_static.has_position_axis, "CEandKLLosses is LM-only and requires a position axis"

    def eval_step(
        model: PlacedModel[PreparedT],
        components: ComponentStacks,
        placed_ci_fn: PlacedCIFn,
        token_ids: Array,
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        batch = _prepare_lm_batch(
            model,
            components,
            placed_ci_fn,
            token_ids,
            mesh,
            n_valid_rows,
            ci_capture_keys,
        )
        # Every mask source draws via `uniform_like` (never a bare `random.uniform`): a
        # bare draw lowers REPLICATED under the Explicit mesh, and the masked forward
        # stacks masks per kind into `[n_layers, B, T, C]` scan inputs — replicated, that
        # stack held the FULL eval batch on every rank (112 GiB per big kind at the 32L
        # production shape). Threefry is counter-based, so the sharded draw is
        # value-identical (SPEC D4).
        zeros_delta = {
            site: jnp.zeros_like(batch.tokens, dtype=COMPUTE_DT) for site in model.site_names
        }
        stoch_key, random_key, _ = random.split(key, 3)
        stochastic_masks: dict[str, Array] = {}
        stochastic_deltas: dict[str, Array] = {}
        for site_idx, site in enumerate(model.site_names):
            ci = batch.ci_lower[site]
            source = uniform_like(random.fold_in(stoch_key, site_idx), ci)
            stochastic_masks[site] = ci + (1.0 - ci) * source
            stochastic_deltas[site] = uniform_like(
                random.fold_in(stoch_key, len(model.site_names) + site_idx),
                ci,
                drop_last_axis=True,
            )
        variants = {
            "ci_masked": (batch.ci_lower, zeros_delta),
            "unmasked": (
                {site: jnp.ones_like(batch.ci_lower[site]) for site in model.site_names},
                zeros_delta,
            ),
            "stoch_masked": (stochastic_masks, stochastic_deltas),
            "random_masked": (
                {
                    site: uniform_like(random.fold_in(random_key, site_idx), batch.ci_lower[site])
                    for site_idx, site in enumerate(model.site_names)
                },
                zeros_delta,
            ),
            "rounded_masked": (
                {
                    site: (batch.ci_lower[site] > rounding_threshold).astype(COMPUTE_DT)
                    for site in model.site_names
                },
                zeros_delta,
            ),
            "zero_masked": (
                {site: jnp.zeros_like(batch.ci_lower[site]) for site in model.site_names},
                zeros_delta,
            ),
        }
        variant_logits = {
            name: _compute_masked_output(model, batch, masks, deltas, mesh, frozenset())
            for name, (masks, deltas) in variants.items()
        }
        target_ce = _ce(batch, batch.clean.output)
        metrics = {
            f"ce_kl/kl_{name}": _kl(batch, logits) for name, logits in variant_logits.items()
        }
        metrics.update(
            {
                f"ce_kl/ce_difference_{name}": _ce(batch, variant_logits[name]) - target_ce
                for name in variants
                if name != "zero_masked"
            }
        )
        return metrics

    return filter_jit(eval_step, compiler_options=compiler_options)


def make_ci_l0_step[PreparedT](
    model_static: PlacedModel[PreparedT],
    ci_capture_keys: CaptureKeys,
    ci_alive_threshold: float,
    groups: dict[str, tuple[str, ...]] | None,
    mesh: Mesh | None = None,
    compiler_options: dict[str, bool | int | str] | None = None,
    *,
    n_valid_rows: int | None = None,
) -> ScalarStep:
    """Bind the generic `CI_L0` arithmetic (`core.ci_l0_eval`) to the LM batch: the shared
    `_prepare_lm_batch` CI and, for the padded arithmetic probes, the row-masked mean."""
    assert model_static.has_position_axis, "CI_L0 is LM-only and requires a position axis"
    resolved_groups = resolve_site_groups(model_static.site_names, groups)

    def eval_step(
        model: PlacedModel[PreparedT],
        components: ComponentStacks,
        placed_ci_fn: PlacedCIFn,
        token_ids: Array,
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        del key
        batch = _prepare_lm_batch(
            model,
            components,
            placed_ci_fn,
            token_ids,
            mesh,
            n_valid_rows,
            ci_capture_keys,
        )

        def mean(value: Array) -> Array:
            if batch.valid_row_mask is None:
                return value.mean()
            return _row_masked_mean(value, batch.valid_row_mask)

        return ci_l0_scalars(
            batch.ci_lower, model.site_names, ci_alive_threshold, resolved_groups, mean
        )

    return filter_jit(eval_step, compiler_options=compiler_options)


def make_fresh_pgd_step[PreparedT](
    model_static: PlacedModel[PreparedT],
    ci_capture_keys: CaptureKeys,
    fresh_pgd: FreshPGDReconEval,
    mesh: Mesh | None = None,
    compiler_options: dict[str, bool | int | str] | None = None,
    *,
    n_valid_rows: int | None = None,
) -> ScalarStep:
    """Build fresh-PGD evaluation for end-to-end reconstruction and optional hidden-activation reconstruction."""
    assert model_static.has_position_axis, "LM PGDReconLoss requires a position axis"
    reconstruction_capture_keys = fresh_pgd.hidden_acts_capture_keys
    model_static.assert_hidden_acts_reconstruction_points(
        tuple(sorted(reconstruction_capture_keys))
    )

    def eval_step(
        model: PlacedModel[PreparedT],
        components: ComponentStacks,
        placed_ci_fn: PlacedCIFn,
        token_ids: Array,
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        batch = _prepare_lm_batch(
            model,
            components,
            placed_ci_fn,
            token_ids,
            mesh,
            n_valid_rows,
            ci_capture_keys,
            reconstruction_capture_keys,
        )
        _, _, pgd_key = random.split(key, 3)

        def end_to_end_error(masked: Array, clean_value: Array) -> Array:
            if batch.valid_row_mask is None:
                return model.recon_loss_fn(masked, clean_value)
            return _row_masked_kl(masked, clean_value, batch.valid_row_mask)

        def objective_with_breakdown(
            masks: dict[str, Array], delta_masks: dict[str, Array]
        ) -> ReconstructionLoss:
            masked_forward_result = model.masked_forward(
                batch.prepared_weights,
                batch.tokens,
                masking=MaterializedMasking(
                    component_masks=masks,
                    weight_delta_masks=delta_masks,
                ),
                capture_keys=reconstruction_capture_keys,
                remat=False,
            )
            masked = reconstruction_observations(
                masked_forward_result,
                hidden_acts_capture_keys=reconstruction_capture_keys,
                mesh=mesh,
            )
            return reconstruction_loss(
                end_to_end_error,
                masked=masked,
                clean=batch.clean,
                reconstruction=fresh_pgd.reconstruction,
                valid_row_mask=batch.valid_row_mask,
            )

        def loss_at_masks(masks: dict[str, Array], delta_masks: dict[str, Array]) -> Array:
            return objective_with_breakdown(masks, delta_masks).total

        sources = fresh_pgd_recon_sources(
            model.sites,
            batch.ci_lower,
            batch.tokens.shape,
            pgd_key,
            fresh_pgd,
            loss_at_masks,
        )
        masks, delta_masks = masks_from_sources(batch.ci_lower, sources)
        breakdown = objective_with_breakdown(masks, delta_masks)
        prefix = f"loss/{fresh_pgd.name}"
        metrics = {prefix: breakdown.total}
        metrics |= {
            f"{prefix}/{suffix}": value
            for suffix, value in reconstruction_loss_metrics(breakdown).items()
        }
        return metrics

    return filter_jit(eval_step, compiler_options=compiler_options)


def make_eval_step[PreparedT](
    model_static: PlacedModel[PreparedT],
    ci_capture_keys: CaptureKeys,
    rounding_threshold: float,
    ci_alive_threshold: float,
    l0_group_patterns: dict[str, tuple[str, ...]] | None,
    fresh_pgd: FreshPGDReconEval | None,
    mesh: Mesh | None = None,
    *,
    n_valid_rows: int | None,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> ScalarStep:
    """Compose independent metric kernels for arithmetic probes and parity tests."""
    ce_kl = make_ce_kl_step(
        model_static,
        ci_capture_keys,
        rounding_threshold,
        mesh,
        compiler_options,
        n_valid_rows=n_valid_rows,
    )
    ci_l0 = make_ci_l0_step(
        model_static,
        ci_capture_keys,
        ci_alive_threshold,
        l0_group_patterns,
        mesh,
        compiler_options,
        n_valid_rows=n_valid_rows,
    )
    pgd = (
        make_fresh_pgd_step(
            model_static,
            ci_capture_keys,
            fresh_pgd,
            mesh,
            compiler_options,
            n_valid_rows=n_valid_rows,
        )
        if fresh_pgd is not None
        else None
    )

    def evaluate(
        model: PlacedModel[PreparedT],
        components: ComponentStacks,
        placed_ci_fn: PlacedCIFn,
        token_ids: Array,
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        record = dict(ce_kl(model, components, placed_ci_fn, token_ids, key))
        record.update(ci_l0(model, components, placed_ci_fn, token_ids, key))
        if pgd is not None:
            record.update(pgd(model, components, placed_ci_fn, token_ids, key))
        return record

    return evaluate
