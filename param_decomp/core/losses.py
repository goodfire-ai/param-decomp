"""Pure loss calculations and frequency estimators with fp32 reductions."""

import math
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import NamedTuple

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Array, Float, Float32, Int32, jaxtyped

from param_decomp.core.components import (
    ComponentStacks,
    SelectedCI,
    SiteCI,
    SiteSpec,
    aligned_component_vectors,
    block_selection_counts,
    selected_component_sums,
    site_ci_leading,
)
from param_decomp.core.configs import (
    CaptureReconstruction,
    FrequencyMinimalityConfig,
    ImportanceMinimalityLossConfig,
)
from param_decomp.core.nonlinearity import (
    DeltaNetHeads,
    KVHeads,
    Neurons,
    NonlinearityAlignment,
    NonlinearityPartition,
    NonlinearityUnitKind,
    QueryHeads,
)
from param_decomp.core.recon import (
    AuxiliaryReconstructionSpec,
    ForwardObservations,
)


@jaxtyped(typechecker=beartype)
def relative_squared_error(
    masked: Float[Array, "batch *positions d"],
    clean: Float[Array, "batch *positions d"],
    *,
    valid_row_mask: Float[Array, " batch"] | None = None,
) -> Float[Array, ""]:
    """`Σ(masked−clean)² / Σ(clean²)` at ONE measurement point, in fp32.

    Per point, not over a stacked point axis: points need not share a width, and each
    divides by its own clean scale. Callers stack the resulting scalars, never the
    activations."""
    masked_f32 = masked.astype(jnp.float32)
    clean_f32 = clean.astype(jnp.float32)
    squared_error = (masked_f32 - clean_f32) ** 2
    squared_clean = clean_f32**2
    if valid_row_mask is not None:
        mask = valid_row_mask.reshape(valid_row_mask.shape[0], *((1,) * (clean.ndim - 1)))
        squared_error = squared_error * mask
        squared_clean = squared_clean * mask
    return jnp.sum(squared_error) / jnp.sum(squared_clean)


@jaxtyped(typechecker=beartype)
def categorical_kl_from_logits(
    masked_logits: Float[Array, "batch *positions categories"],
    clean_logits: Float[Array, "batch *positions categories"],
    *,
    valid_row_mask: Float[Array, " batch"] | None = None,
) -> Float[Array, ""]:
    """KL(clean || masked), averaged over positions, with a frozen clean target."""
    clean_logp = jax.nn.log_softmax(
        jax.lax.stop_gradient(clean_logits).astype(jnp.float32), axis=-1
    )
    masked_logp = jax.nn.log_softmax(masked_logits.astype(jnp.float32), axis=-1)
    per_token = jnp.sum(jnp.exp(clean_logp) * (clean_logp - masked_logp), axis=-1)
    if valid_row_mask is None:
        return jnp.mean(per_token)
    weights = valid_row_mask.reshape(valid_row_mask.shape[0], *((1,) * (per_token.ndim - 1)))
    token_count = jnp.sum(valid_row_mask) * math.prod(per_token.shape[1:])
    return jnp.sum(jnp.where(weights > 0, per_token * weights, 0)) / jnp.maximum(token_count, 1)


def _capture_reconstruction_loss(
    comparison: CaptureReconstruction,
    masked: Array,
    clean: Array,
    valid_row_mask: Float[Array, " batch"] | None,
) -> Float[Array, ""]:
    match comparison.distance:
        case "relative_squared_error":
            return relative_squared_error(masked, clean, valid_row_mask=valid_row_mask)
        case "categorical_kl_from_logits":
            return categorical_kl_from_logits(masked, clean, valid_row_mask=valid_row_mask)


class ReconstructionLoss(NamedTuple):
    total: Array
    output: Array
    auxiliaries: dict[str, dict[str, Array]]


def reconstruction_loss[Out](
    recon_loss_fn: Callable[[Out, Out], Array],
    *,
    masked: ForwardObservations[Out],
    clean: ForwardObservations[Out],
    reconstruction: AuxiliaryReconstructionSpec,
    valid_row_mask: Float[Array, " batch"] | None = None,
) -> ReconstructionLoss:
    output_loss = recon_loss_fn(masked.output, clean.output)
    total = output_loss
    auxiliary_losses: dict[str, dict[str, Array]] = {}
    for auxiliary in reconstruction:
        values = {
            comparison.capture: _capture_reconstruction_loss(
                comparison,
                masked.captures[comparison.capture],
                clean.captures[comparison.capture],
                valid_row_mask,
            )
            for comparison in auxiliary.comparisons
        }
        auxiliary_losses[auxiliary.name] = values
        total = total + auxiliary.coeff * jnp.mean(jnp.stack(tuple(values.values())))
    return ReconstructionLoss(total, output_loss, auxiliary_losses)


def reconstruction_loss_metrics(loss: ReconstructionLoss) -> dict[str, Array]:
    metrics: dict[str, Array] = {}
    for name, values in loss.auxiliaries.items():
        metrics["e2e"] = loss.output
        metrics[name] = jnp.mean(jnp.stack(tuple(values.values())))
        metrics.update({f"{name}/{point}": value for point, value in values.items()})
    return metrics


def unit_squared_norms(
    vectors: Float[Array, "*components d"], partition: NonlinearityPartition
) -> Float[Array, "*components U"]:
    """Per-unit sums of squared coordinates over the partition's output-axis blocks.

    The per-head sum is a matmul against a constant block indicator, not a
    reshape-and-sum: `d` may be sharded with shard boundaries splitting heads
    (under tp), where a reshape forces GSPMD to all-gather the full width while
    a contraction over `d` keeps partial [C, U] sums shard-local (one tiny
    all-reduce), in the backward too.
    """
    squares = vectors * vectors
    match partition:
        case Neurons():
            return squares
        case (
            QueryHeads(head_count=head_count)
            | KVHeads(head_count=head_count)
            | DeltaNetHeads(head_count=head_count)
        ):
            d = vectors.shape[-1]
            assert d % head_count == 0, (d, head_count)
            head_of_column = jnp.repeat(
                jnp.eye(head_count, dtype=vectors.dtype), d // head_count, axis=0
            )
            return squares @ head_of_column


def nonlinearity_unit_squared_norm_fractions(
    vectors: Float[Array, "*components d"], partition: NonlinearityPartition
) -> Float[Array, "*components U"]:
    """Each unit's fraction of its component's squared write-vector norm.

    For component `c` and unit `u`, returns
    `Σ_{j in u} vectors[c,j]² / Σ_j vectors[c,j]²`.

    Stop-gradient max normalization prevents fp32 underflow without changing the result.
    Exact-zero rows return zero; an epsilon floor would break scale invariance.
    """
    vectors = vectors.astype(jnp.float32)
    scale = jax.lax.stop_gradient(jnp.max(jnp.abs(vectors), axis=-1, keepdims=True))
    vectors = vectors / jnp.where(scale > 0.0, scale, 1.0)
    unit_sq = unit_squared_norms(vectors, partition)
    total_sq = unit_sq.sum(-1, keepdims=True)
    alive = total_sq > 0.0
    return jnp.where(alive, unit_sq / jnp.where(alive, total_sq, 1.0), 0.0)


def soft_unit_count(
    fractions: Float[Array, "*components U"],
    relative_threshold: Float[Array, ""] | float,
    *,
    normalize_at_one: bool,
) -> Float[Array, "*components"]:
    """Per-component soft unit count.

    The base count is `Σ_u f_u / (f_u + relative_threshold / U)`. Normalization
    rescales it by `1 + relative_threshold / U`, making a one-hot unit fraction
    contribute exactly one throughout a threshold schedule.
    """
    unit_count = fractions.shape[-1]
    one_unit_normalizer = 1.0 + relative_threshold / unit_count if normalize_at_one else 1.0
    return one_unit_normalizer * (fractions / (fractions + relative_threshold / unit_count)).sum(-1)


class _NonlinearityGroupTerm(NamedTuple):
    kind: NonlinearityUnitKind
    masked_count_sum: Float[Array, ""]
    n_components: int


@jaxtyped(typechecker=beartype)
def nonlinearity_loss(
    components: ComponentStacks,
    alignments: Mapping[str, NonlinearityAlignment],
    relative_threshold: Float[Array, ""],
    kind_coefficients: Mapping[NonlinearityUnitKind, float],
    *,
    normalize_at_one: bool,
) -> tuple[Float[Array, ""], dict[NonlinearityUnitKind, Float[Array, ""]]]:
    """Return the kind-weighted nonlinearity penalty and its unweighted per-kind means
    of soft uses per component. Callers exclude a kind by omitting its sites
    AND its coefficient — an excluded kind is never computed, so weight 0.0 is not a
    state here."""
    assert alignments, "nonlinearity loss needs at least one partitioned site"
    assert {a.partition.unit_kind for a in alignments.values()} == kind_coefficients.keys(), (
        alignments,
        kind_coefficients,
    )
    grouped: defaultdict[tuple[str, NonlinearityAlignment], list[int]] = defaultdict(list)
    for name, alignment in alignments.items():
        group, index = components.stack_index_of(name)
        grouped[group, alignment].append(index)

    terms: list[_NonlinearityGroupTerm] = []
    for (group, alignment), stack_indices in grouped.items():
        partition = alignment.partition
        vectors = aligned_component_vectors(components.stacks[group], alignment.side)
        # Uses, not blocks: each block is consumed by `use_multiplicity` nonlinearities
        # (GQA query heads per kv head, value-head recurrences per DeltaNet key head), so
        # the per-block soft count scales by that factor.
        counts = partition.use_multiplicity * soft_unit_count(
            nonlinearity_unit_squared_norm_fractions(vectors, partition),
            relative_threshold,
            normalize_at_one=normalize_at_one,
        )
        # Reduce the full resident stack under a constant stack-index mask — never gather
        # the stack axis by index: it is owner-partitioned across nodes, and a stack-axis
        # gather forces cross-node resharding. The mask is numpy so it bakes into the
        # graph as a constant rather than a scatter.
        mask = np.zeros((vectors.shape[0],) + (1,) * (counts.ndim - 1), np.float32)
        mask[stack_indices] = 1.0
        terms.append(
            _NonlinearityGroupTerm(
                partition.unit_kind,
                (counts * mask).sum(),
                len(stack_indices) * math.prod(vectors.shape[1:-1]),
            )
        )

    kinds: tuple[NonlinearityUnitKind, ...] = tuple(dict.fromkeys(term.kind for term in terms))
    by_kind: dict[NonlinearityUnitKind, Float[Array, ""]] = {
        kind: sum(
            (term.masked_count_sum for term in terms if term.kind == kind),
            start=jnp.zeros((), jnp.float32),
        )
        / sum(term.n_components for term in terms if term.kind == kind)
        for kind in kinds
    }
    total = sum(
        (kind_coefficients[kind] * mean for kind, mean in by_kind.items()),
        start=jnp.zeros((), jnp.float32),
    )
    return total, by_kind


def _selected_site_frequencies(
    ci: SelectedCI, per_value_penalty: Callable[[Array], Array]
) -> Float[Array, " C"]:
    """Compute full-width component frequencies from selected CI values.

    Scatter selected penalties to their global components and include `psi(0)`
    for every unselected token/component pair, all divided by the same token count.
    The global reduction precedes the nonlinear frequency penalty; averaging
    per-shard penalties instead would introduce Jensen bias."""
    n = math.prod(site_ci_leading(ci))
    selected_sums = selected_component_sums(ci, per_value_penalty(ci.values.astype(jnp.float32)))
    selection_counts = block_selection_counts(ci)
    psi_zero = per_value_penalty(jnp.zeros((), jnp.float32))
    unselected = jnp.repeat(n - selection_counts, ci.c_per_block) * psi_zero
    return (selected_sums + unselected) / n


def _site_frequencies(
    ci: SiteCI, per_value_penalty: Callable[[Array], Array]
) -> Float[Array, " _"]:
    match ci:
        case SelectedCI():
            return _selected_site_frequencies(ci, per_value_penalty)
        case jax.Array():
            return einops.reduce(per_value_penalty(ci.astype(jnp.float32)), "... c -> c", "mean")


def _per_component_frequencies(
    ci_upper: Mapping[str, SiteCI],
    per_value_penalty: Callable[[Float[Array, "*leading _"]], Float[Array, "*leading _"]],
) -> dict[str, Float[Array, " _"]]:
    """Per-site firing frequencies `f_c = mean_{b,t} psi(c)` for any per-value penalty
    `psi`. Under GSPMD the leading axes are the global batch, so the reduction
    IS the exact global per-component mean — XLA reduces across shards inside the graph,
    so `f_c` is the true full-batch frequency inside the convex `log2` (a per-shard
    `f_c` would give a Jensen bias). Selected-emitting sites take the exact scatter arm
    (`_selected_site_frequencies`); the full `[C]` vector exists only HERE, as a
    reduction, never as a per-token tensor."""
    return {name: _site_frequencies(ci, per_value_penalty) for name, ci in ci_upper.items()}


def _site_activity(ci: SiteCI, per_value_penalty: Callable[[Array], Array]) -> Float[Array, ""]:
    """One site's activity contribution `Σ_c f_c`, computed WITHOUT the `[C]` accumulator:
    a selected-emitting site's sum over the selected axis is exact under the same `B·T`
    denominator (zeros contribute nothing to a sum — never a mean over the last axis,
    which would corrupt the denominator), plus the per-site `(C − k·c)·psi(0)` constant —
    zero for smooth-L0's psi, kept so the helper is exact for any per-value penalty."""
    match ci:
        case SelectedCI():
            n = math.prod(site_ci_leading(ci))
            selected = jnp.sum(per_value_penalty(ci.values.astype(jnp.float32))) / n
            psi_zero = per_value_penalty(jnp.zeros((), jnp.float32))
            return selected + (ci.C - ci.values.shape[-1]) * psi_zero
        case jax.Array():
            return jnp.sum(
                einops.reduce(per_value_penalty(ci.astype(jnp.float32)), "... c -> c", "mean")
            )


def _frequency_curve(f: Float[Array, " _"], reference_datapoint_count: int) -> Float[Array, " _"]:
    """`Φ(f) = f · log2(1 + a'·f)`, the per-component frequency penalty."""
    return f * jnp.log2(1.0 + reference_datapoint_count * f)


def _frequency_curve_slope(
    f: Float[Array, " _"], reference_datapoint_count: int
) -> Float[Array, " _"]:
    """`Φ'(f) = log2(1 + a'·f) + a'·f / ((1 + a'·f)·ln 2)`."""
    af = reference_datapoint_count * f
    return jnp.log2(1.0 + af) + af / ((1.0 + af) * math.log(2.0))


def _smooth_l0_psi(
    gamma: Float[Array, ""], *, normalize_at_one: bool
) -> Callable[[Float[Array, "*leading _"]], Float[Array, "*leading _"]]:
    gamma_sq = gamma * gamma
    if normalize_at_one:
        return lambda ci: (1.0 + gamma_sq) * ci**2 / (ci**2 + gamma_sq)
    return lambda ci: ci**2 / (ci**2 + gamma_sq)


def activity_sum(frequencies: dict[str, Float[Array, " _"]]) -> Float[Array, ""]:
    """`Σ_s Σ_c f_c` — the linear importance term."""
    return sum((jnp.sum(f) for f in frequencies.values()), start=jnp.zeros((), jnp.float32))


def activity_sum_from_ci(
    ci_upper: Mapping[str, SiteCI], gamma: Array, *, normalize_at_one: bool
) -> Float[Array, ""]:
    """The activity term directly from the CI values — the reader for steps with NO
    frequency penalty, whose selected arm needs no `[C]` machinery at all."""
    per_value_penalty = _smooth_l0_psi(gamma, normalize_at_one=normalize_at_one)
    return sum(
        (_site_activity(ci, per_value_penalty) for ci in ci_upper.values()),
        start=jnp.zeros((), jnp.float32),
    )


def _frequency_penalty(
    frequencies: dict[str, Float[Array, " _"]], reference_datapoint_count: int
) -> Float[Array, ""]:
    """Sum the frequency penalty `f * log2(1 + reference_datapoint_count * f)` over all
    components."""
    return sum(
        (jnp.sum(_frequency_curve(f, reference_datapoint_count)) for f in frequencies.values()),
        start=jnp.zeros((), jnp.float32),
    )


def ema_frequency_penalty(
    frequencies: dict[str, Float[Array, " _"]],
    ema: dict[str, Float[Array, " _"]],
    count_f32: Float32[Array, ""],
    halflife_steps: float,
    reference_datapoint_count: int,
) -> tuple[Float[Array, ""], dict[str, Float[Array, " _"]]]:
    """`(freq, new_ema)`: the frequency penalty at a debiased EMA of `f_c`.

    `new_ema = decay·ema + (1-decay)·sg(f_batch)`, debiased `f̂ = new_ema/(1-decay^(count+1))`
    where `count` is the number of prior observations. The current batch is included,
    so the first observation reproduces the un-smoothed penalty exactly.
    The value is `Σ Φ(f̂)`; the first-order surrogate keeps the gradient at the un-smoothed
    penalty's scale, `Φ'(f̂)·∂f_batch/∂θ`, with the estimate stop-gradded.

    `decay = 2^(-1/halflife)` exists only in log space: formed directly it rounds to 1
    (past `h ~ 1e16` even in f64), the subtractive `1-decay` forms cancel to 0, and the
    debias division returns NaN; the direct `-ln(2)/h` with `expm1` stays finite for
    every admitted halflife. The config's `1e6` halflife cap bounds fp32 rounding drift
    in the recurrence (pinned by `test_ema_long_scan_rounding_bounded`)."""
    log_decay = -math.log(2.0) / halflife_steps
    alpha = -math.expm1(log_decay)  # 1 - decay
    debias = -jnp.expm1(log_decay * (count_f32 + 1.0))  # 1 - decay^(count+1)
    freq = jnp.zeros((), jnp.float32)
    new_ema: dict[str, Float[Array, " _"]] = {}
    for name, f_batch in frequencies.items():
        f_sg = jax.lax.stop_gradient(f_batch)
        new_ema[name] = ema[name] + alpha * (f_sg - ema[name])
        f_hat = new_ema[name] / debias
        surrogate = _frequency_curve_slope(f_hat, reference_datapoint_count) * (
            f_batch - jax.lax.stop_gradient(f_batch)
        )
        freq = freq + jnp.sum(_frequency_curve(f_hat, reference_datapoint_count) + surrogate)
    return freq, new_ema


@jaxtyped(typechecker=beartype)
def importance_minimality_terms(
    ci_upper: Mapping[str, SiteCI],
    gamma: Float[Array, ""],
    reference_datapoint_count: int | None,
    *,
    normalize_at_one: bool,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Geman–McClure smooth-L0 imp-min terms: per-value penalty `c^2 / (c^2 + gamma^2)`.
    Flat at the origin (`phi'(0)=0`) and bounded (`|phi'| <= 0.65/gamma`) — no singularity,
    no `eps` floor. Approaches the true `L_0` count as `gamma -> 0`. `normalize_at_one`
    switches to `(1 + gamma^2) c^2 / (c^2 + gamma^2)`, so `c = 1` contributes exactly 1.

    Without a frequency penalty (`reference_datapoint_count is None`), the
    activity reads the CI values directly (no `[C]` accumulator — a selected-emitting
    site's sum is exact as-is) and `freq = 0.0`; with one, both terms read the same
    per-component frequencies."""
    if reference_datapoint_count is None:
        return activity_sum_from_ci(ci_upper, gamma, normalize_at_one=normalize_at_one), jnp.zeros(
            (), jnp.float32
        )
    frequencies = _per_component_frequencies(
        ci_upper, _smooth_l0_psi(gamma, normalize_at_one=normalize_at_one)
    )
    return activity_sum(frequencies), _frequency_penalty(frequencies, reference_datapoint_count)


def per_component_frequencies(
    ci_upper: Mapping[str, SiteCI], gamma: Array, *, normalize_at_one: bool
) -> dict[str, Float[Array, " _"]]:
    """Return per-site component frequencies under smooth-L0 at the supplied `gamma`."""
    return _per_component_frequencies(
        ci_upper, _smooth_l0_psi(gamma, normalize_at_one=normalize_at_one)
    )


class BatchFrequencyPenalty(NamedTuple):
    value: Float32[Array, ""]


class EmaFrequencyPenalty(NamedTuple):
    value: Float32[Array, ""]
    batch_value: Float32[Array, ""]


FrequencyPenalty = BatchFrequencyPenalty | EmaFrequencyPenalty


class BatchFrequency(eqx.Module):
    """A frequency estimate from the current batch."""

    def evaluate(
        self,
        frequencies: dict[str, Float32[Array, " _"]],
        reference_datapoint_count: int,
    ) -> tuple[BatchFrequencyPenalty, "BatchFrequency"]:
        return BatchFrequencyPenalty(
            _frequency_penalty(frequencies, reference_datapoint_count)
        ), self


class EmaFrequency(eqx.Module):
    """A running frequency estimate, debiased when evaluating its penalty."""

    halflife_steps: float = eqx.field(static=True)
    estimate: dict[str, Float32[Array, " _"]]
    count: Int32[Array, ""]

    def evaluate(
        self,
        frequencies: dict[str, Float32[Array, " _"]],
        reference_datapoint_count: int,
    ) -> tuple[EmaFrequencyPenalty, "EmaFrequency"]:
        value, estimate = ema_frequency_penalty(
            frequencies,
            self.estimate,
            jnp.asarray(self.count, jnp.float32),
            self.halflife_steps,
            reference_datapoint_count,
        )
        return EmaFrequencyPenalty(
            value, _frequency_penalty(frequencies, reference_datapoint_count)
        ), replace(self, estimate=estimate, count=self.count + 1)


FrequencyEstimator = BatchFrequency | EmaFrequency


def init_frequency_estimator(
    cfg: FrequencyMinimalityConfig | None, sites: tuple[SiteSpec, ...]
) -> FrequencyEstimator:
    match cfg:
        case None:
            return BatchFrequency()
        case FrequencyMinimalityConfig():
            match cfg.ema_halflife_steps:
                case None:
                    return BatchFrequency()
                case int() | float() as halflife_steps:
                    return EmaFrequency(
                        halflife_steps,
                        {site.name: jnp.zeros((site.C,), jnp.float32) for site in sites},
                        jnp.zeros((), jnp.int32),
                    )


def imp_min_terms(
    ci_upper: Mapping[str, SiteCI],
    cfg: ImportanceMinimalityLossConfig,
    gamma: Array,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """`(activity, freq)` from the single-batch estimate — the reader for steps without
    EMA state, such as evaluations. EMA-aware training composes
    `per_component_frequencies` + `activity_sum` + the frequency estimator's `evaluate`
    instead."""
    ref = cfg.frequency.reference_datapoint_count if cfg.frequency is not None else None
    return importance_minimality_terms(ci_upper, gamma, ref, normalize_at_one=cfg.normalize_at_one)
