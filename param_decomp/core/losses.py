"""The pure loss terms (SPEC §2) and their schedules — fp32 reductions, no state."""

import math
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import einops
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from param_decomp.core.components import (
    ComponentStacks,
    SiteSpec,
)
from param_decomp.core.configs import (
    FrequencyMinimalityConfig,
    HiddenActsReconstruction,
    ImportanceMinimalityLossConfig,
    LossCoeff,
)
from param_decomp.core.nonlinearity import (
    KVHeads,
    Neurons,
    NonlinearityPartition,
    NonlinearityUnitKind,
    QueryHeads,
)
from param_decomp.core.recon import (
    ForwardObservations,
    OutputAndHiddenActsReconstruction,
    OutputOnlyReconstruction,
    ReconstructionSpec,
)
from param_decomp.core.schedule import Knot, ScheduleConfig


def _interval_frac_traced(prev: Knot, knot: Knot, t: Array) -> Array:
    u = (t - prev.at) / (knot.at - prev.at)
    match knot.interp:
        case "linear":
            return prev.frac + (knot.frac - prev.frac) * u
        case "cosine":
            return prev.frac + (knot.frac - prev.frac) * 0.5 * (1 - jnp.cos(jnp.pi * u))
        case "hold":
            return jnp.where(u >= 1.0, knot.frac, prev.frac)


def scheduled_value_at(train_frac: Array, config: ScheduleConfig) -> Array:
    """Evaluate a schedule at one traced fraction of the run."""
    assert train_frac.shape == (), train_frac.shape
    points = config.points
    t = train_frac
    frac = _interval_frac_traced(points[0], points[1], t)
    for prev, knot in zip(points[1:], points[2:], strict=False):
        frac = jnp.where(t >= prev.at, _interval_frac_traced(prev, knot, t), frac)
    return jnp.asarray(config.max_val * frac, jnp.float32)


def train_frac_at(step: Array, total_steps: int) -> Array:
    """Map update ``0 .. total_steps - 1`` to fraction-time ``0 .. 1`` (SPEC S20)."""
    assert total_steps > 0, f"total_steps must be positive, got {total_steps}"
    if total_steps == 1:
        return jnp.zeros((), jnp.float32)
    return step.astype(jnp.float32) / jnp.asarray(total_steps - 1, jnp.float32)


def scheduled_value_traced(step_f32: Array, total_steps: int, config: ScheduleConfig) -> Array:
    """Compatibility boundary for optax and host probes that naturally own a count."""
    train_frac = jnp.minimum(train_frac_at(step_f32, total_steps), 1.0)
    return scheduled_value_at(train_frac, config)


def coeff_at(train_frac: Array, coeff: LossCoeff) -> Float[Array, ""] | float:
    """Resolve one loss coefficient from the step's fraction-time scalar."""
    match coeff:
        case ScheduleConfig():
            return scheduled_value_at(train_frac, coeff)
        case float() | int():
            return coeff


def reconstruction_spec_at(
    hidden_acts_reconstruction: HiddenActsReconstruction | None,
    train_frac: Array,
) -> ReconstructionSpec:
    """Resolve the S35 hidden-activation reconstruction at this fraction-time."""
    if hidden_acts_reconstruction is None:
        return OutputOnlyReconstruction()
    return OutputAndHiddenActsReconstruction(
        coeff_at(train_frac, hidden_acts_reconstruction.coeff),
        hidden_acts_reconstruction.points,
    )


@jaxtyped(typechecker=beartype)
def relative_squared_error(
    masked: Float[Array, "*leading d"],
    clean: Float[Array, "*leading d"],
    *,
    valid_row_mask: Float[Array, " batch"] | None = None,
) -> Float[Array, ""]:
    """`Σ(masked−clean)² / Σ(clean²)` at ONE measurement point, in fp32 (SPEC S35).

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


class OutputOnlyReconstructionLoss(NamedTuple):
    total: Array


class OutputAndHiddenActsReconstructionLoss(NamedTuple):
    total: Array
    output: Array
    hidden_acts_by_point: dict[str, Array]


type ReconstructionLoss = OutputOnlyReconstructionLoss | OutputAndHiddenActsReconstructionLoss


def reconstruction_loss(
    recon_loss_fn: Callable[[Any, Any], Array],
    *,
    masked: ForwardObservations,
    clean: ForwardObservations,
    reconstruction: ReconstructionSpec,
    valid_row_mask: Array | None = None,
) -> ReconstructionLoss:
    """The closed forms of one recon comparison (SPEC S35)."""
    output_loss = recon_loss_fn(masked.output, clean.output)
    match reconstruction:
        case OutputOnlyReconstruction():
            return OutputOnlyReconstructionLoss(output_loss)
        case OutputAndHiddenActsReconstruction(coeff=coeff, points=points):
            per_point = {
                point: relative_squared_error(
                    masked.hidden_acts_by_point[point],
                    clean.hidden_acts_by_point[point],
                    valid_row_mask=valid_row_mask,
                )
                for point in points
            }
            aggregate = jnp.mean(jnp.stack(tuple(per_point.values())))
            return OutputAndHiddenActsReconstructionLoss(
                output_loss + coeff * aggregate, output_loss, per_point
            )


def reconstruction_loss_metrics(loss: ReconstructionLoss) -> dict[str, Array]:
    """Metric suffixes contributed by one reconstruction-loss result."""
    match loss:
        case OutputOnlyReconstructionLoss():
            return {}
        case OutputAndHiddenActsReconstructionLoss(
            output=output, hidden_acts_by_point=hidden_acts_by_point
        ):
            return {
                "e2e": output,
                "hidden_acts_reconstruction": jnp.mean(
                    jnp.stack(tuple(hidden_acts_by_point.values()))
                ),
                **{
                    f"hidden_acts_reconstruction/{point}": value
                    for point, value in hidden_acts_by_point.items()
                },
            }


def mean_reconstruction_losses[T](values: tuple[T, ...]) -> T:
    """Mean every fp32 scalar leaf across structurally identical pytrees."""

    def scalar_mean(*scalars: Array) -> Array:
        return sum(scalars, start=jnp.zeros((), jnp.float32)) / len(scalars)

    return jax.tree.map(scalar_mean, *values)


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
        case QueryHeads(head_count=head_count) | KVHeads(head_count=head_count):
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
    fractions: Float[Array, "*components U"], relative_threshold: Float[Array, ""] | float
) -> Float[Array, "*components"]:
    """Per-component soft count `Σ_u f_u / (f_u + relative_threshold / U)` (SPEC S36)."""
    unit_count = fractions.shape[-1]
    return (fractions / (fractions + relative_threshold / unit_count)).sum(-1)


class _NonlinearityGroupTerm(NamedTuple):
    kind: NonlinearityUnitKind
    masked_count_sum: Float[Array, ""]
    n_components: int


@jaxtyped(typechecker=beartype)
def nonlinearity_loss(
    components: ComponentStacks,
    partitions: Mapping[str, NonlinearityPartition],
    relative_threshold: Float[Array, ""],
    kind_coefficients: Mapping[NonlinearityUnitKind, float],
) -> tuple[Float[Array, ""], dict[NonlinearityUnitKind, Float[Array, ""]]]:
    """Return the kind-weighted nonlinearity penalty and its unweighted per-kind means
    of soft uses per component (SPEC S36). Callers exclude a kind by omitting its sites
    AND its coefficient — an excluded kind is never computed, so weight 0.0 is not a
    state here."""
    assert partitions, "nonlinearity loss needs at least one partitioned site"
    assert {p.unit_kind for p in partitions.values()} == kind_coefficients.keys(), (
        partitions,
        kind_coefficients,
    )
    grouped: defaultdict[tuple[str, NonlinearityPartition], list[int]] = defaultdict(list)
    for name, partition in partitions.items():
        group, slot = components.slot_of(name)
        grouped[group, partition].append(slot)

    terms: list[_NonlinearityGroupTerm] = []
    for (group, partition), slots in grouped.items():
        us = components.stacks[group][1]
        # Uses, not blocks: each block is consumed by `use_multiplicity` attention
        # nonlinearities, so the per-block soft count scales by that factor (SPEC S36).
        counts = partition.use_multiplicity * soft_unit_count(
            nonlinearity_unit_squared_norm_fractions(us, partition), relative_threshold
        )
        # Reduce the full resident stack under a constant slot mask — never gather the
        # stack axis by slots: it is owner-partitioned across nodes, and a stack-axis
        # gather forces cross-node resharding. The mask is numpy so it bakes into the
        # graph as a constant rather than a scatter.
        mask = np.zeros((us.shape[0], 1), np.float32)
        mask[slots] = 1.0
        terms.append(
            _NonlinearityGroupTerm(
                partition.unit_kind, (counts * mask).sum(), len(slots) * us.shape[1]
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


def _per_component_frequencies(
    ci_upper: dict[str, Float[Array, "*leading _"]],
    per_value_penalty: Callable[[Float[Array, "*leading _"]], Float[Array, "*leading _"]],
) -> dict[str, Float[Array, " _"]]:
    """Per-site firing frequencies `f_c = mean_{b,t} psi(c)` for any per-value penalty
    `psi` (SPEC S8). Under GSPMD the leading axes are the global batch, so the reduction
    IS the exact global per-component mean — XLA reduces across shards inside the graph,
    so `f_c` is the true full-batch frequency inside the convex `log2` (a per-shard
    `f_c` would give a Jensen bias)."""
    return {
        name: einops.reduce(per_value_penalty(ci.astype(jnp.float32)), "... c -> c", "mean")
        for name, ci in ci_upper.items()
    }


def _frequency_curve(f: Float[Array, " _"], reference_datapoint_count: int) -> Float[Array, " _"]:
    """`Φ(f) = f · log2(1 + a'·f)`, the per-component frequency penalty (SPEC S8)."""
    return f * jnp.log2(1.0 + reference_datapoint_count * f)


def _frequency_curve_slope(
    f: Float[Array, " _"], reference_datapoint_count: int
) -> Float[Array, " _"]:
    """`Φ'(f) = log2(1 + a'·f) + a'·f / ((1 + a'·f)·ln 2)`."""
    af = reference_datapoint_count * f
    return jnp.log2(1.0 + af) + af / ((1.0 + af) * math.log(2.0))


def _smooth_l0_psi(
    gamma: Float[Array, ""],
) -> Callable[[Float[Array, "*leading _"]], Float[Array, "*leading _"]]:
    gamma_sq = gamma * gamma
    return lambda ci: ci**2 / (ci**2 + gamma_sq)


def activity_sum(frequencies: dict[str, Float[Array, " _"]]) -> Float[Array, ""]:
    """`Σ_s Σ_c f_c` — the linear importance term (SPEC S8)."""
    return sum((jnp.sum(f) for f in frequencies.values()), start=jnp.zeros((), jnp.float32))


def _frequency_penalty(
    frequencies: dict[str, Float[Array, " _"]], reference_datapoint_count: int
) -> Float[Array, ""]:
    """`freq = Σ_s Σ_c Φ(f_c)` (SPEC S7/S8)."""
    return sum(
        (jnp.sum(_frequency_curve(f, reference_datapoint_count)) for f in frequencies.values()),
        start=jnp.zeros((), jnp.float32),
    )


def _activity_and_freq(
    frequencies: dict[str, Float[Array, " _"]], reference_datapoint_count: int | None
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """`(activity, freq)` from one set of per-site frequencies; `freq = 0.0` when no
    frequency penalty is configured (`reference_datapoint_count is None`, SPEC S8')."""
    freq = (
        _frequency_penalty(frequencies, reference_datapoint_count)
        if reference_datapoint_count is not None
        else jnp.zeros((), jnp.float32)
    )
    return activity_sum(frequencies), freq


def ema_frequency_penalty(
    frequencies: dict[str, Float[Array, " _"]],
    ema: dict[str, Float[Array, " _"]],
    step_f32: Array,
    halflife_steps: float,
    reference_datapoint_count: int,
) -> tuple[Float[Array, ""], dict[str, Float[Array, " _"]]]:
    """`(freq, new_ema)`: the frequency penalty at a debiased EMA of `f_c` (SPEC S8'').

    `new_ema = decay·ema + (1-decay)·sg(f_batch)`, debiased `f̂ = new_ema/(1-decay^(step+1))`
    — the current batch is included, so step 0 reproduces the un-smoothed penalty exactly.
    The value is `Σ Φ(f̂)`; the first-order surrogate keeps the gradient at the un-smoothed
    penalty's scale, `Φ'(f̂)·∂f_batch/∂θ`, with the estimate stop-gradded (both S8'').

    `decay = 2^(-1/halflife)` exists only in log space: formed directly it rounds to 1
    (past `h ~ 1e16` even in f64), the subtractive `1-decay` forms cancel to 0, and the
    debias division returns NaN; the direct `-ln(2)/h` with `expm1` stays finite for
    every admitted halflife. The config's `1e6` halflife cap bounds fp32 rounding drift
    in the recurrence (pinned by `test_ema_long_scan_rounding_bounded`)."""
    log_decay = -math.log(2.0) / halflife_steps
    alpha = -math.expm1(log_decay)  # 1 - decay
    debias = -jnp.expm1(log_decay * (step_f32 + 1.0))  # 1 - decay^(step+1)
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
    ci_upper: dict[str, Float[Array, "*leading _"]],
    gamma: Float[Array, ""],
    reference_datapoint_count: int | None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Geman–McClure smooth-L0 imp-min terms: per-value penalty `c^2 / (c^2 + gamma^2)`.
    Flat at the origin (`phi'(0)=0`) and bounded (`|phi'| <= 0.65/gamma`) — no singularity,
    no `eps` floor. Approaches the true `L_0` count as `gamma -> 0`."""
    return _activity_and_freq(
        _per_component_frequencies(ci_upper, _smooth_l0_psi(gamma)), reference_datapoint_count
    )


def per_component_frequencies(
    ci_upper: dict[str, Float[Array, "*leading _"]], gamma: Array
) -> dict[str, Float[Array, " _"]]:
    """The per-site `f_c` vectors both imp-min readouts consume (SPEC S8), under the
    smooth-L0 penalty at its annealed width (SPEC S9)."""
    return _per_component_frequencies(ci_upper, _smooth_l0_psi(gamma))


class BatchFrequencyTerm(NamedTuple):
    """The frequency penalty at the single-batch `f_c` (SPEC S8/S8')."""

    freq: Float[Array, ""]


class EmaFrequencyTerm(NamedTuple):
    """The frequency penalty at the debiased EMA of `f_c` (SPEC S8''). `freq_batch` is
    the un-smoothed diagnostic logged alongside; `new_freq_ema` is the per-site `(C,)`
    EMA state to carry forward."""

    freq: Float[Array, ""]
    freq_batch: Float[Array, ""]
    new_freq_ema: dict[str, Float[Array, " _"]]


FrequencyTerm = BatchFrequencyTerm | EmaFrequencyTerm


@dataclass(frozen=True)
class BatchFrequency:
    """The single-batch frequency-penalty mode (SPEC S8'), owning no cross-step state."""

    coeff: LossCoeff
    reference_datapoint_count: int

    def term(self, frequencies: dict[str, Float[Array, " _"]]) -> BatchFrequencyTerm:
        return BatchFrequencyTerm(_frequency_penalty(frequencies, self.reference_datapoint_count))


@dataclass(frozen=True)
class EmaFrequency:
    """The debiased-EMA frequency-penalty mode (SPEC S8'') — the only mode that owns
    EMA state."""

    coeff: LossCoeff
    reference_datapoint_count: int
    halflife_steps: float

    def initial_state(self, sites: tuple[SiteSpec, ...]) -> dict[str, Array]:
        """Zero-init per-site `(C,)` fp32 EMA of the per-component firing frequencies."""
        return {spec.name: jnp.zeros((spec.C,), jnp.float32) for spec in sites}

    def term(
        self,
        frequencies: dict[str, Float[Array, " _"]],
        freq_ema: dict[str, Float[Array, " _"]] | None,
        step_f32: Array,
    ) -> EmaFrequencyTerm:
        assert freq_ema is not None, "ema_halflife_steps set but no freq_ema state (S8'')"
        freq, new_freq_ema = ema_frequency_penalty(
            frequencies, freq_ema, step_f32, self.halflife_steps, self.reference_datapoint_count
        )
        return EmaFrequencyTerm(
            freq, _frequency_penalty(frequencies, self.reference_datapoint_count), new_freq_ema
        )


FrequencyRole = BatchFrequency | EmaFrequency


def resolve_frequency(cfg: FrequencyMinimalityConfig) -> FrequencyRole:
    """The config's `ema_halflife_steps` optional decided into one mode noun, once, at
    step-build time — the runtime consumers match on the role. Whether a frequency
    penalty exists at all is the caller's match on the config's presence."""
    match cfg.ema_halflife_steps:
        case None:
            return BatchFrequency(cfg.coeff, cfg.reference_datapoint_count)
        case halflife_steps:
            return EmaFrequency(cfg.coeff, cfg.reference_datapoint_count, halflife_steps)


def imp_min_terms(
    ci_upper: dict[str, Float[Array, "*leading _"]],
    cfg: ImportanceMinimalityLossConfig,
    gamma: Array,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """`(activity, freq)` from the single-batch estimate — the reader for steps without
    EMA state (the targeted objective, evals). The EMA-aware train step composes
    `per_component_frequencies` + `activity_sum` + the resolved `FrequencyRole`'s `term`
    instead (SPEC S8'')."""
    ref = cfg.frequency.reference_datapoint_count if cfg.frequency is not None else None
    return _activity_and_freq(per_component_frequencies(ci_upper, gamma), ref)
