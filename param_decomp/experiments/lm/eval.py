"""In-loop eval pass: scalar parity with the torch eval metrics.

Implements independent pure JAX kernels for the scalar core of the torch reference
`eval:` block: `CEandKLLosses` (six masking variants), `CI_L0`, and fresh PGD
reconstruction. Each authored operation compiles only its own kernel; `make_eval_step`
composes them only for the fixed arithmetic probe and parity tests.
Plot-type metrics (CI histograms, activation density, per-component means, the
permutation/UV figures) ride the in-loop SLOW tier instead — natively in JAX
(`slow_eval.py`; in-loop only, no offline CLI).

Variant semantics mirror `param_decomp/eval_metrics/ce_and_kl_losses.py`: each
variant is a masked forward with ALL sites live and no routing; only `stoch_masked`
carries a weight-delta mask (torch `make_mask_infos` without weight deltas drops the
delta term — delta mask 0 here). CE averages adjacent tokens within each sequence, excluding document boundaries
when document IDs are supplied; KL is per-position vs the clean (frozen) logits.

Cross-batch aggregation (the multi-`n_steps` eval pass in `run.py`): every key this
function returns is a per-BATCH scalar that the caller averages uniformly over the
eval batches. This is mean-safe against the torch reference — i.e. it matches torch's
accumulate-then-`compute()` to within float reassociation — only because every emitted
key is itself a per-batch reduction that torch *also* averages across batches, and the
eval batches are uniform `(B, T)`. The Jensen trap (a nonlinearity applied AFTER
the cross-batch reduction, so mean-of-batch-results ≠ result-of-global-batch) does NOT
arise here, because no emitted key wraps the cross-batch axis in a nonlinearity:

- `ce_kl/kl_<variant>`: torch `CEandKLLosses` accumulates `kl * n_positions` and divides
  by total positions (token-weighted mean of a per-batch mean). Uniform `(B, T)` makes
  token-weighting equal to the uniform `1/n_steps` average here.
- `ce_kl/ce_difference_<variant>` = `ce_v - ce_target`: torch averages this per-batch
  DIFFERENCE (computed inside `_calc_ce_and_kl_losses`), not a difference of grand means.
  Linear, so averaging batch differences equals the difference of batch averages.
  Each batch CE uses its own count of valid next-token labels; this is not a
  corpus-wide token-weighted CE when valid-label counts vary between batches.
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
from jaxtyping import Array, Float, PRNGKeyArray

from param_decomp.core.base_config import Probability
from param_decomp.core.ci_fn.interface import CIFn
from param_decomp.core.ci_fn.runtime import evaluate_ci_from_captures
from param_decomp.core.ci_l0_eval import ci_l0_scalars, resolve_site_groups
from param_decomp.core.components import (
    ComponentStacks,
    SiteCI,
    map_site_ci,
)
from param_decomp.core.decomposed_linear import constrain_component_activation
from param_decomp.core.dict_utils import dict_safe_update_
from param_decomp.core.losses import (
    ReconstructionLoss,
    reconstruction_loss,
    reconstruction_loss_metrics,
)
from param_decomp.core.masking import (
    materialize_masking,
    sample_component_mask,
    sample_delta_mask,
    source_masking,
)
from param_decomp.core.model import (
    EMPTY_CAPTURE_KEYS,
    CaptureKeys,
    ComponentActivations,
    MaterializedMasking,
    PlacedModel,
    select_captures,
)
from param_decomp.core.precision import COMPUTE_DT
from param_decomp.core.recon import (
    ForwardObservations,
    reconstruction_observations,
)
from param_decomp.core.recon_eval import FreshPGDReconEval, fresh_pgd_recon_sources
from param_decomp.core.sharding import batch_shard_leading
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.lm.inputs import input_next_token_mask, input_token_ids
from param_decomp.targets.lm_output import LMOutput, StreamedLinearOutput
from param_decomp.targets.losses import (
    kl_per_position,
    streamed_position_kl,
    streamed_position_next_token_ce,
)

type ScalarStep[TargetIn, PreparedT: ComponentActivations, Conditioning, PreparedMaskingT] = (
    Callable[
        [
            PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
            ComponentStacks,
            CIFn[Conditioning],
            TargetIn,
            PRNGKeyArray,
        ],
        Mapping[str, Array],
    ]
)

type ScalarScorer[TargetIn, PreparedT: ComponentActivations, Conditioning, PreparedMaskingT] = (
    Callable[
        [
            PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
            "PreparedLMBatch[TargetIn, PreparedT, Conditioning]",
            PRNGKeyArray,
        ],
        dict[str, Array],
    ]
)
"""The pure scoring half of a scalar kernel over an already-prepared batch — jitted by
the caller: the shared-context eval path jits it alone (the batch comes from the pass's
one context step), the fused `make_*_step` path jits it composed with `prepare_lm_batch`."""


def next_token_cross_entropy(
    logits: Float[Array, "B T vocab"], batch: LMBatch | LMBatchWithDocuments
) -> Array:
    return _row_masked_cross_entropy(logits, batch, jnp.ones(input_token_ids(batch).shape[0]))


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


def _next_token_mean(
    values: Array, batch: LMBatch | LMBatchWithDocuments, row_weights: Array
) -> Array:
    """Average valid next-token predictions; no observations contribute zero."""
    weights = input_next_token_mask(batch) * row_weights[:, None]
    return jnp.sum(jnp.where(weights > 0, values * weights, 0)) / jnp.maximum(jnp.sum(weights), 1)


def _row_masked_cross_entropy(
    logits: Float[Array, "B T vocab"],
    batch: LMBatch | LMBatchWithDocuments,
    row_mask: Float[Array, " B"],
) -> Array:
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    label_log_probs = jnp.take_along_axis(
        log_probs[:, :-1], input_token_ids(batch)[:, 1:, None], axis=-1
    )[..., 0]
    return _next_token_mean(-label_log_probs, batch, row_mask)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PreparedLMBatch[TargetIn, PreparedT: ComponentActivations, Conditioning]:
    """The shared per-batch values every scalar kernel scores against — built either by
    `prepare_lm_batch` (the fused arithmetic/parity path) or from the pass's one
    `LMBatchContext` (the corpus eval path). A pytree, so it crosses jit boundaries.
    `conditioning` is the clean forward's pinned decisions, which every masked forward over
    `tokens` reproduces."""

    tokens: TargetIn
    clean: ForwardObservations[LMOutput]
    prepared_weights: PreparedT
    conditioning: Conditioning
    ci_lower: dict[str, SiteCI]
    valid_row_mask: Array | None


def prepare_lm_batch[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    components: ComponentStacks,
    ci_fn: CIFn[Conditioning],
    token_ids: TargetIn,
    mesh: Mesh | None,
    n_valid_rows: int | None,
    ci_capture_keys: CaptureKeys,
    activation_capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
) -> PreparedLMBatch[TargetIn, PreparedT, Conditioning]:
    """Pure shared preparation required by independent LM metric kernels."""
    tokens = jax.tree.map(lambda a: batch_shard_leading(a, mesh), token_ids)
    capture_keys = ci_capture_keys | activation_capture_keys
    clean_forward_result = model.clean_forward(tokens, capture_keys)
    ci_input_activations = select_captures(clean_forward_result.captures, ci_capture_keys)
    clean = reconstruction_observations(
        clean_forward_result,
        model.pin_output_batch,
        capture_keys=activation_capture_keys,
        mesh=mesh,
    )
    prepared_weights = model.prepare_compute_weights(components)
    ci_lower = evaluate_ci_from_captures(
        ci_fn.prepare(),
        ci_input_activations,
        clean_forward_result.conditioning,
        prepared_weights,
        sequence=clean_forward_result.sequence,
        remat=False,
    ).lower
    ci_lower = {
        site: constrain_component_activation(
            value, None if model.placement is None else model.placement.activations.component
        )
        for site, value in ci_lower.items()
    }
    valid_row_mask = None
    if n_valid_rows is not None:
        assert n_valid_rows <= input_token_ids(tokens).shape[0], (
            n_valid_rows,
            input_token_ids(tokens).shape,
        )
        valid_row_mask = (jnp.arange(input_token_ids(tokens).shape[0]) < n_valid_rows).astype(
            jnp.float32
        )
    return PreparedLMBatch(
        tokens=tokens,
        clean=clean,
        prepared_weights=prepared_weights,
        conditioning=clean_forward_result.conditioning,
        ci_lower=ci_lower,
        valid_row_mask=valid_row_mask,
    )


def _compute_masked_output[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    batch: PreparedLMBatch[TargetIn, PreparedT, Conditioning],
    masking: MaterializedMasking,
    mesh: Mesh | None,
    capture_keys: CaptureKeys,
) -> LMOutput:
    masked_forward_result = model.masked_forward(
        batch.prepared_weights,
        batch.conditioning,
        masking=model.model.prepare_masking(masking),
        routes=None,
        capture_keys=capture_keys,
        remat=False,
    )
    return model.pin_output_batch(masked_forward_result.output, mesh)


def _kl[TargetIn: LMBatch | LMBatchWithDocuments, PreparedT: ComponentActivations, Conditioning](
    batch: PreparedLMBatch[TargetIn, PreparedT, Conditioning], output: LMOutput
) -> Array:
    match (output, batch.clean.output):
        case (StreamedLinearOutput(), StreamedLinearOutput()):
            position_kl = streamed_position_kl(output, batch.clean.output)
            if batch.valid_row_mask is None:
                return position_kl.mean()
            return _row_masked_mean(position_kl, batch.valid_row_mask)
        case (jax.Array(), jax.Array()):
            if batch.valid_row_mask is None:
                return kl_per_position(output, batch.clean.output)
            return _row_masked_kl(output, batch.clean.output, batch.valid_row_mask)
        case _:
            raise AssertionError(
                f"mixed model-output edges: {type(output).__name__} vs "
                f"{type(batch.clean.output).__name__}"
            )


def _ce[TargetIn: LMBatch | LMBatchWithDocuments, PreparedT: ComponentActivations, Conditioning](
    batch: PreparedLMBatch[TargetIn, PreparedT, Conditioning], output: LMOutput
) -> Array:
    match output:
        case StreamedLinearOutput():
            position_ce = streamed_position_next_token_ce(output, input_token_ids(batch.tokens))
            weights = (
                jnp.ones(input_token_ids(batch.tokens).shape[0])
                if batch.valid_row_mask is None
                else batch.valid_row_mask
            )
            return _next_token_mean(position_ce, batch.tokens, weights)
        case jax.Array():
            if batch.valid_row_mask is None:
                return next_token_cross_entropy(output, batch.tokens)
            return _row_masked_cross_entropy(output, batch.tokens, batch.valid_row_mask)


def make_ce_kl_scorer[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model_static: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    rounding_threshold: Probability,
    mesh: Mesh | None = None,
) -> ScalarScorer[TargetIn, PreparedT, Conditioning, PreparedMaskingT]:
    """Build the pure CE/KL scorer over a prepared batch."""
    assert model_static.model.has_position_axis, (
        "CEandKLLosses is LM-only and requires a position axis"
    )

    def score(
        model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
        batch: PreparedLMBatch[TargetIn, PreparedT, Conditioning],
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        stoch_key, random_key, _ = random.split(key, 3)
        stochastic_masks: dict[str, SiteCI] = {}
        stochastic_deltas: dict[str, Array] = {}
        for site_idx, site in enumerate(model.model.site_names):
            ci = batch.ci_lower[site]
            source_key = random.fold_in(stoch_key, site_idx)
            stochastic_masks[site] = sample_component_mask(ci, source_key)
            stochastic_deltas[site] = sample_delta_mask(
                ci, random.fold_in(stoch_key, len(model.model.site_names) + site_idx)
            )
        # Every variant is pointwise on the CI values, so narrow sites' masks inherit
        # the bundle (indices carried through to the masked forward).
        variants = {
            "ci_masked": MaterializedMasking(component_masks=batch.ci_lower),
            "unmasked": MaterializedMasking(
                component_masks={
                    site: map_site_ci(jnp.ones_like, batch.ci_lower[site])
                    for site in model.model.site_names
                },
            ),
            "stoch_masked": MaterializedMasking(
                component_masks=stochastic_masks, weight_delta_masks=stochastic_deltas
            ),
            "random_masked": MaterializedMasking(
                component_masks={
                    site: sample_component_mask(
                        map_site_ci(jnp.zeros_like, batch.ci_lower[site]),
                        random.fold_in(random_key, site_idx),
                    )
                    for site_idx, site in enumerate(model.model.site_names)
                },
            ),
            "rounded_masked": MaterializedMasking(
                component_masks={
                    site: map_site_ci(
                        lambda v: (v > rounding_threshold).astype(COMPUTE_DT),
                        batch.ci_lower[site],
                    )
                    for site in model.model.site_names
                },
            ),
            "zero_masked": MaterializedMasking(
                component_masks={
                    site: map_site_ci(jnp.zeros_like, batch.ci_lower[site])
                    for site in model.model.site_names
                },
            ),
        }
        variant_outputs = {
            name: _compute_masked_output(model, batch, masking, mesh, frozenset())
            for name, masking in variants.items()
        }
        target_ce = _ce(batch, batch.clean.output)
        metrics = {
            f"ce_kl/kl_{name}": _kl(batch, output) for name, output in variant_outputs.items()
        }
        metrics.update(
            {
                f"ce_kl/ce_difference_{name}": _ce(batch, variant_outputs[name]) - target_ce
                for name in variants
                if name != "zero_masked"
            }
        )
        return metrics

    return score


def make_ce_kl_step[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model_static: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    ci_capture_keys: CaptureKeys,
    rounding_threshold: Probability,
    mesh: Mesh | None = None,
    *,
    n_valid_rows: int | None = None,
) -> ScalarStep[TargetIn, PreparedT, Conditioning, PreparedMaskingT]:
    """The fused CE/KL evaluator (own preparation) for probes and parity tests."""
    score = make_ce_kl_scorer(model_static, rounding_threshold, mesh)

    def eval_step(
        model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
        components: ComponentStacks,
        ci_fn: CIFn[Conditioning],
        token_ids: TargetIn,
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        batch = prepare_lm_batch(
            model, components, ci_fn, token_ids, mesh, n_valid_rows, ci_capture_keys
        )
        return score(model, batch, key)

    return eval_step


def make_ci_l0_scorer[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model_static: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    ci_alive_threshold: Probability,
    groups: dict[str, tuple[str, ...]] | None,
) -> ScalarScorer[TargetIn, PreparedT, Conditioning, PreparedMaskingT]:
    """Bind the generic `CI_L0` arithmetic (`core.ci_l0_eval`) to a prepared LM batch
    (row-masked mean for the padded arithmetic probes)."""
    assert model_static.model.has_position_axis, "CI_L0 is LM-only and requires a position axis"
    resolved_groups = resolve_site_groups(model_static.model.site_names, groups)

    def score(
        model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
        batch: PreparedLMBatch[TargetIn, PreparedT, Conditioning],
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        del key

        def mean(value: Array) -> Array:
            if batch.valid_row_mask is None:
                return value.mean()
            return _row_masked_mean(value, batch.valid_row_mask)

        return ci_l0_scalars(
            batch.ci_lower, model.model.site_names, ci_alive_threshold, resolved_groups, mean
        )

    return score


def make_ci_l0_step[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model_static: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    ci_capture_keys: CaptureKeys,
    ci_alive_threshold: Probability,
    groups: dict[str, tuple[str, ...]] | None,
    mesh: Mesh | None = None,
    *,
    n_valid_rows: int | None = None,
) -> ScalarStep[TargetIn, PreparedT, Conditioning, PreparedMaskingT]:
    """The fused CI-L0 evaluator (own preparation) for probes and parity tests."""
    score = make_ci_l0_scorer(model_static, ci_alive_threshold, groups)

    def eval_step(
        model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
        components: ComponentStacks,
        ci_fn: CIFn[Conditioning],
        token_ids: TargetIn,
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        batch = prepare_lm_batch(
            model, components, ci_fn, token_ids, mesh, n_valid_rows, ci_capture_keys
        )
        return score(model, batch, key)

    return eval_step


def make_fresh_pgd_scorer[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model_static: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    fresh_pgd: FreshPGDReconEval,
    mesh: Mesh | None = None,
) -> ScalarScorer[TargetIn, PreparedT, Conditioning, PreparedMaskingT]:
    """Build the fresh-PGD scorer over a prepared batch carrying all probe captures."""
    assert model_static.model.has_position_axis, "LM PGDReconLoss requires a position axis"
    reconstruction_capture_keys = fresh_pgd.reconstruction_capture_keys

    def score(
        model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
        batch: PreparedLMBatch[TargetIn, PreparedT, Conditioning],
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        _, _, pgd_key = random.split(key, 3)

        def end_to_end_error(masked: LMOutput, clean_value: LMOutput) -> Array:
            if batch.valid_row_mask is None:
                return model.recon_loss_fn(masked, clean_value)
            match (masked, clean_value):
                case (StreamedLinearOutput(), StreamedLinearOutput()):
                    return _row_masked_mean(
                        streamed_position_kl(masked, clean_value), batch.valid_row_mask
                    )
                case (jax.Array(), jax.Array()):
                    return _row_masked_kl(masked, clean_value, batch.valid_row_mask)
                case _:
                    raise AssertionError(
                        f"mixed model-output edges: {type(masked).__name__} vs "
                        f"{type(clean_value).__name__}"
                    )

        def objective_with_breakdown(masking: MaterializedMasking) -> ReconstructionLoss:
            masked_forward_result = model.masked_forward(
                batch.prepared_weights,
                batch.conditioning,
                masking=model.model.prepare_masking(masking),
                routes=None,
                capture_keys=reconstruction_capture_keys,
                remat=False,
            )
            masked = reconstruction_observations(
                masked_forward_result,
                model.pin_output_batch,
                capture_keys=reconstruction_capture_keys,
                mesh=mesh,
            )
            return reconstruction_loss(
                end_to_end_error,
                masked=masked,
                clean=batch.clean,
                reconstruction=fresh_pgd.reconstruction,
                valid_row_mask=batch.valid_row_mask,
            )

        def loss_at_masking(masking: MaterializedMasking) -> Array:
            return objective_with_breakdown(masking).total

        sources = fresh_pgd_recon_sources(
            model.model.sites,
            batch.ci_lower,
            input_token_ids(batch.tokens).shape,
            pgd_key,
            fresh_pgd,
            loss_at_masking,
        )
        masking = materialize_masking(source_masking(batch.ci_lower, sources))
        breakdown = objective_with_breakdown(masking)
        prefix = f"loss/{fresh_pgd.name}"
        metrics = {prefix: breakdown.total}
        dict_safe_update_(
            metrics,
            {
                f"{prefix}/{suffix}": value
                for suffix, value in reconstruction_loss_metrics(breakdown).items()
            },
        )
        return metrics

    return score


def make_fresh_pgd_step[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model_static: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    ci_capture_keys: CaptureKeys,
    fresh_pgd: FreshPGDReconEval,
    mesh: Mesh | None = None,
    *,
    n_valid_rows: int | None = None,
) -> ScalarStep[TargetIn, PreparedT, Conditioning, PreparedMaskingT]:
    """The fused fresh-PGD evaluator (own preparation) for probes and parity tests."""
    score = make_fresh_pgd_scorer(model_static, fresh_pgd, mesh)

    def eval_step(
        model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
        components: ComponentStacks,
        ci_fn: CIFn[Conditioning],
        token_ids: TargetIn,
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        batch = prepare_lm_batch(
            model,
            components,
            ci_fn,
            token_ids,
            mesh,
            n_valid_rows,
            ci_capture_keys,
            fresh_pgd.reconstruction_capture_keys,
        )
        return score(model, batch, key)

    return eval_step


def make_eval_step[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model_static: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    ci_capture_keys: CaptureKeys,
    rounding_threshold: Probability,
    ci_alive_threshold: Probability,
    l0_group_patterns: dict[str, tuple[str, ...]] | None,
    fresh_pgd: FreshPGDReconEval | None,
    mesh: Mesh | None = None,
    *,
    n_valid_rows: int | None,
) -> ScalarStep[TargetIn, PreparedT, Conditioning, PreparedMaskingT]:
    """Compose independent metric kernels for arithmetic probes and parity tests."""
    ce_kl = make_ce_kl_step(
        model_static,
        ci_capture_keys,
        rounding_threshold,
        mesh,
        n_valid_rows=n_valid_rows,
    )
    ci_l0 = make_ci_l0_step(
        model_static,
        ci_capture_keys,
        ci_alive_threshold,
        l0_group_patterns,
        mesh,
        n_valid_rows=n_valid_rows,
    )
    pgd = (
        make_fresh_pgd_step(
            model_static,
            ci_capture_keys,
            fresh_pgd,
            mesh,
            n_valid_rows=n_valid_rows,
        )
        if fresh_pgd is not None
        else None
    )

    def evaluate(
        model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
        components: ComponentStacks,
        ci_fn: CIFn[Conditioning],
        token_ids: TargetIn,
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        record = dict(ce_kl(model, components, ci_fn, token_ids, key))
        record.update(ci_l0(model, components, ci_fn, token_ids, key))
        if pgd is not None:
            record.update(pgd(model, components, ci_fn, token_ids, key))
        return record

    return evaluate
