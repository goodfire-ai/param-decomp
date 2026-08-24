"""JAX-native slow (plot-type) eval metrics — a LIBRARY for the in-loop slow tier (`run.py`)
and the toy eval functions (`param_decomp/experiments/{tms,resid_mlp}`). Slow eval is
IN-LOOP ONLY; there is no offline/retrospective CLI.

`eval.py` runs the FAST scalar tier in-loop (CE/KL, CI-L0, the fresh-PGD probe). The
SLOW tier is the heavy plot metrics: `CIHistograms`, `ComponentActivationDensity`,
`CIMeanPerComponent` (the torch eval-metric classes of the same names). Every one of them
is a reduction over the per-site causal-importance arrays from a masked-free forward, then
a numpy/matplotlib plot. The forward + reduction is JAX; the plotting is framework-agnostic
(it mirrors the torch `param_decomp/eval_metrics/plotting.py` reductions on numpy
arrays, no torch). `accumulate_site_reductions` / `render_slow_eval_figures` / `accumulate_position_ci` /
`render_permutation_figures` are what the LM slow-tier operations bind
(`experiments/lm/diagnostic_eval_operations.py`); the toys use only the UV figure helpers
(`render_uv_figure` / `plot_uv_matrices`).

The slow tier runs IN-LOOP on `eval.slow_every` next to the fast pass (`run.py`,
SPEC S28/S29), reusing the fast pass's eval batches and logging `slow_eval/*` on the live
`_step` axis from a rank-0 background thread.

Cross-batch reductions are exact under micro-batching: density/mean accumulate
SUM-over-positions + a position count, divided once at the end (token-weighted mean,
uniform `(B, T)` makes it the plain mean). `CIHistograms` is the exception — its two value
histograms bin against each batch's own min/max, and counts on different edges do not sum,
so they require `eval.n_steps=1`. It
ALSO opts into a per-token CI density heatmap (`density_heatmap_n_bins`): a per-component
on-device bincount into log-spaced `[1e-9, 1]` bands over the same forward's `lower`,
accumulated over EVERY batch — a small `(C, n_bins + 1)` reduction, so unlike the value
histograms it costs nothing to carry across batches. Rendered as
`figures/ci_density_heatmap` (`plot_ci_density_heatmap`).

It also computes the two SCALAR hidden-acts recon eval metrics (`CIHiddenActsReconLoss`,
`StochasticHiddenActsReconLoss`) natively — per decomposed site, the summed MSE between
the masked-model and target-model site OUTPUT activations, divided once by the element
count (`hidden_acts_eval.py`). Those request the same canonical output keys from clean and masked forwards (SPEC S31)
and are emitted as scalars under the torch log keys (`<ClassName>/<site>` + a combined `<ClassName>`).

The three CONFIG-GATED permutation metrics (`PermutedCIPlots`, `UVPlots`,
`IdentityCIError`) are recomputed natively too, off the run's `eval.metrics` block
from the resolved domain eval plan. They share one column permutation per site — identity (scipy
`linear_sum_assignment` on `-CI`) or dense (by column mass) — derived from a per-site
upper-leaky CI matrix. `PermutedCIPlots` and `IdentityCIError` use the LM batch-mean
`(position, C)` matrix (`make_position_ci_step` / `accumulate_position_ci`) and are LM-only
(they need the position axis). `UVPlots` reorders the V/U columns by the same kind of
permutation and is the one figure metric usable for ANY decomposition: the toys feed it
their probe CI as the permutation source (`render_uv_figure`), the LM in-loop tier feeds it
the position-CI upper matrix. The LM in-loop UVPlots does a NAIVE host gather of the
C-sharded V/U — cheap for the toys (small, replicated, already on host) but it OOMs / breaks
at production C BY DESIGN: no special handling, the gather is the cost.
"""

import fnmatch
import io
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from param_decomp.core.ci_fn import (
    PlacedCIFn,
    ci_preactivations,
    lower_leaky_hard_sigmoid,
    upper_leaky_hard_sigmoid,
)
from param_decomp.core.configs import (
    DenseCITargetSpec,
    IdentityCIErrorConfig,
    IdentityCITargetSpec,
    PermutedCIPlotsConfig,
    UVPlotsConfig,
)
from param_decomp.core.jit_util import filter_jit
from param_decomp.core.model import CaptureKeys, PlacedModel

IDENTITY_CI_ERROR_TOLERANCE = 0.1
"""Torch `IdentityCIPattern.distance_from` / `compute_target_metrics` default tolerance —
avoids sensitivity to small CI values from inactive components."""


VALUE_HISTOGRAM_N_BINS = 100
"""Bins in each `CIHistograms` value histogram (torch `plot_ci_values_histograms`)."""


@dataclass(frozen=True)
class ValueHistogram:
    """One `CIHistograms` histogram: `ax.hist(values, bins=VALUE_HISTOGRAM_N_BINS)` binned on
    device, so the `(*leading, C)` values never cross to the host.

    `lo`/`hi` span every value, as matplotlib's own edges would. They must NOT be taken over
    a subsample to save the transfer: they are order statistics, so a sample renders a
    narrower x-axis and empties bins holding a rare-but-real fraction of the mass — on a log
    y-axis, what the figure is read for."""

    counts: np.ndarray
    lo: float
    hi: float

    @property
    def edges(self) -> np.ndarray:
        return np.linspace(self.lo, self.hi, len(self.counts) + 1)


@dataclass(frozen=True)
class SiteReduction:
    """Per-site accumulators across the eval pass (all `(C,)` or scalar / small histogram).

    `density_counts[c]` = #(positions where `lower_leaky > threshold`); `ci_sums[c]` =
    Σ positions `lower_leaky`; `n_positions` = total positions seen (shared count for
    both means). `value_histograms` is the `(lower, preactivations)` pair the two
    `CIHistograms` figures plot — `None` for a metric that renders neither, which then
    pays no host transfer at all. `density_hist` is the
    opt-in per-token CI density histogram `(C, n_bins + 1)`: column 0 = underflow (CI below
    `CI_DENSITY_HEATMAP_FLOOR`, including exact-zero inactive tokens), columns `1..n_bins` =
    counts in the `n_bins` log-spaced `[FLOOR, 1]` bands. It accumulates over EVERY eval batch
    since it is a small on-device reduction; `None` when the metric doesn't opt in."""

    density_counts: np.ndarray
    ci_sums: np.ndarray
    n_positions: int
    value_histograms: tuple[ValueHistogram, ValueHistogram] | None
    density_hist: np.ndarray | None


BinnedValues = tuple[Array, Array, Array]
"""`(counts, lo, hi)` — one on-device value histogram, `(n_bins,)` plus its two edges."""


SlowEvalStep = Callable[
    [PlacedModel, Any, Float[Array, "*leading d"]],
    tuple[
        dict[str, Array],
        dict[str, Array],
        Array,
        dict[str, BinnedValues],
        dict[str, BinnedValues],
        dict[str, Array],
    ],
]
"""`(model, placed_ci_fn, residual) -> (density_counts, ci_sums, n_positions, binned_lower,
binned_preactivations, density_hist)` — the per-batch reduction, pre-reduced over positions.
`density_hist` maps site -> `(C, n_bins + 1)` counts (empty when the density heatmap is off);
the two binned dicts are empty when the caller asked for no value histogram.
The slow plot metrics read only the CI arrays, so V/U (`components`) is not an input. `model`
(frozen-weight-bearing) is the jit ARG."""


CI_DENSITY_HEATMAP_FLOOR = 1e-9
"""Lower edge of the log-spaced CI bands: CI below this (including exact 0) falls in the
underflow column 0. Equal to the sampling floor of the mean-CI tail."""


def _count_ge(values: Array, edges: Array) -> Array:
    """`counts[k] = #{v >= edges[k]}`, reduced over every `values` axis — the searchsorted
    that survives a sharded operand. `jnp.histogram`/`bincount`/`digitize` all lower
    through scatter or `select` ops that have no explicit-sharding rule and refuse the
    dp-sharded CI arrays; a broadcast-compare fuses into its reduction instead, and the
    counts land replicated. Exact: `#{v >= e_k}` IS `searchsorted(e, v, side='right')`
    summed per edge."""
    return (values[..., None] >= edges).sum(tuple(range(values.ndim)))


def _binned_values(values: Array, n_bins: int) -> BinnedValues:
    """`ax.hist(values, bins=n_bins)` as a device reduction, the data's own min/max as the
    outer edges. fp32 throughout: the values are bf16, and matplotlib would have upcast
    them before binning."""
    v = values.astype(jnp.float32).reshape(-1, values.shape[-1])
    lo, hi = v.min(), v.max()
    edges = jnp.linspace(lo, hi, n_bins + 1)
    # Bin b is [e_b, e_{b+1}), the last closed at hi (numpy's convention): every value
    # sits in [lo, hi], so counts[b] = count_ge[b] - count_ge[b+1] with the final
    # subtrahend dropped.
    count_ge = _count_ge(v, edges[:-1])
    counts = count_ge - jnp.concatenate([count_ge[1:], jnp.zeros((1,), count_ge.dtype)])
    return counts, lo, hi


def _to_value_histogram(binned: BinnedValues) -> ValueHistogram:
    """All three are reductions over the dp-sharded batch axis, hence replicated: a bare
    `np.asarray` is addressable on every process, no `process_allgather` needed."""
    counts, lo, hi = binned
    return ValueHistogram(counts=np.asarray(counts), lo=float(lo), hi=float(hi))


def _per_component_ci_hist(lower: Array, n_bins: int) -> Array:
    """Per-component per-token CI histogram `(C, n_bins + 1)` from `lower (*, C)`: column 0
    counts underflow tokens (CI < `CI_DENSITY_HEATMAP_FLOOR`, including exact-zero inactive
    ones), columns `1..n_bins` the `n_bins` log-spaced bands over `[FLOOR, 1]` (the top band
    includes CI = 1). Band membership as cumulative `>=`-edge counts differenced per band
    (`bincount`'s scatter has no explicit-sharding rule — see `_count_ge`), reduced over
    tokens only so the counts keep the C axis (and its sharding)."""
    c = lower.shape[-1]
    v = lower.astype(jnp.float32).reshape(-1, c)
    edges = jnp.logspace(math.log10(CI_DENSITY_HEATMAP_FLOOR), 0.0, n_bins + 1)
    # count_ge[t, c, k] summed over tokens: (C, n_bins + 1) with count_ge[:, 0] = #tokens
    # at or above the floor; band j >= 1 is [e_{j-1}, e_j) except the top band, closed at
    # 1 (every CI <= 1, so the final subtrahend is 0); column 0 is the underflow
    # complement.
    count_ge = (v[:, :, None] >= edges[:-1]).sum(0)
    n_tokens = jnp.asarray(v.shape[0], count_ge.dtype)
    bands = count_ge - jnp.concatenate([count_ge[:, 1:], jnp.zeros_like(count_ge[:, :1])], axis=1)
    underflow = n_tokens - count_ge[:, 0]
    return jnp.concatenate([underflow[:, None], bands], axis=1)


def make_slow_eval_step(
    model_static: PlacedModel,
    ci_capture_keys: CaptureKeys,
    ci_alive_threshold: float,
    density_heatmap_n_bins: int | None,
    value_histogram_n_bins: int | None,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> SlowEvalStep:
    """Build the jit'd per-batch reduction `slow_eval_step(model, placed_ci_fn, residual) ->
    ({site: density_counts}, {site: ci_sums}, n_positions, {site: binned lower},
    {site: binned preactivations}, {site: density_hist})`. Counts/sums are pre-reduced over
    positions. `value_histogram_n_bins` opts into the `lower`/`preactivations` histograms the
    two `CIHistograms` figures plot, binned ON DEVICE so only counts cross to the host
    (empty dicts when None — a metric reading neither figure pays no transfer).
    `density_heatmap_n_bins` opts into the per-component CI density histogram (empty dict
    when None); it shares this forward's `lower`, adding only an on-device bincount."""
    site_names = model_static.site_names

    def slow_eval_step(
        model: PlacedModel, placed_ci_fn: PlacedCIFn, residual: Float[Array, "*leading d"]
    ) -> tuple[
        dict[str, Array],
        dict[str, Array],
        Array,
        dict[str, BinnedValues],
        dict[str, BinnedValues],
        dict[str, Array],
    ]:
        # Read the CI fn in training precision (bf16), like train.py / eval.py: the readout
        # reflects the deployed model, and cuDNN flash attention rejects fp32.
        preactivations = ci_preactivations(
            placed_ci_fn,
            model.clean_forward(residual, ci_capture_keys).captures,
            remat=False,
        )
        lower = {s: lower_leaky_hard_sigmoid(preactivations[s]) for s in site_names}

        density_counts = {
            s: (lower[s] > ci_alive_threshold)
            .astype(jnp.float32)
            .reshape(-1, lower[s].shape[-1])
            .sum(0)
            for s in site_names
        }
        ci_sums = {s: lower[s].reshape(-1, lower[s].shape[-1]).sum(0) for s in site_names}
        first = lower[site_names[0]]
        n_positions = jnp.asarray(math.prod(first.shape[:-1]), jnp.int32)
        binned_lower = (
            {}
            if value_histogram_n_bins is None
            else {s: _binned_values(lower[s], value_histogram_n_bins) for s in site_names}
        )
        binned_preactivations = (
            {}
            if value_histogram_n_bins is None
            else {s: _binned_values(preactivations[s], value_histogram_n_bins) for s in site_names}
        )
        density_hist = (
            {s: _per_component_ci_hist(lower[s], density_heatmap_n_bins) for s in site_names}
            if density_heatmap_n_bins is not None
            else {}
        )
        return (
            density_counts,
            ci_sums,
            n_positions,
            binned_lower,
            binned_preactivations,
            density_hist,
        )

    return filter_jit(slow_eval_step, compiler_options=compiler_options)


def accumulate_site_reductions(
    slow_eval_step: SlowEvalStep,
    model: PlacedModel,
    placed_ci_fn: PlacedCIFn,
    residual_batches: list[Float[Array, "*leading d"]],
) -> dict[str, SiteReduction]:
    """Drive `slow_eval_step` over the eval batches and fold the per-batch reductions
    into one `SiteReduction` per site. The `(C,)` reductions and the opt-in `density_hist`
    accumulate over EVERY batch; the value histograms cannot, and a step emitting them
    requires exactly one."""
    assert residual_batches, "slow eval needs at least one batch"
    density: dict[str, np.ndarray] = {}
    sums: dict[str, np.ndarray] = {}
    hist: dict[str, np.ndarray] = {}
    value_histograms: dict[str, tuple[ValueHistogram, ValueHistogram]] = {}
    total_positions = 0
    for batch_idx, residual in enumerate(residual_batches):
        d, s, n_pos, binned_lower, binned_preactivations, density_hist = slow_eval_step(
            model, placed_ci_fn, residual
        )
        assert not (binned_lower and len(residual_batches) > 1), (
            "the CIHistograms value histograms bin against each batch's own min/max, so "
            f"counts from {len(residual_batches)} batches cannot be summed: run them at "
            f"eval.n_steps=1, or drop CIHistograms from eval.metrics"
        )
        total_positions += int(n_pos)
        for site in d:
            counts, ci_sum = np.asarray(d[site]), np.asarray(s[site])
            density[site] = counts if batch_idx == 0 else density[site] + counts
            sums[site] = ci_sum if batch_idx == 0 else sums[site] + ci_sum
            if density_hist:
                h = np.asarray(density_hist[site])
                hist[site] = h if batch_idx == 0 else hist[site] + h
            if binned_lower:
                value_histograms[site] = (
                    _to_value_histogram(binned_lower[site]),
                    _to_value_histogram(binned_preactivations[site]),
                )

    return {
        site: SiteReduction(
            density_counts=density[site],
            ci_sums=sums[site],
            n_positions=total_positions,
            value_histograms=value_histograms.get(site),
            density_hist=hist.get(site),
        )
        for site in density
    }


PositionCIStep = Callable[
    [PlacedModel, Any, Float[Array, "*leading d"]],
    tuple[dict[str, Array], dict[str, Array], Array],
]
"""`(model, placed_ci_fn, residual) -> ({site: lower (T, C)}, {site: upper (T, C)}, n_batch)` —
the per-batch CI summed over the batch leading axis, position axis kept. Pairs with
`accumulate_position_ci` to form a batch-mean `(T, C)` CI matrix per site. `model`
(frozen-weight-bearing) is the jit ARG."""


def make_position_ci_step(
    model_static: PlacedModel,
    ci_capture_keys: CaptureKeys,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> PositionCIStep:
    """Per-batch CI reduction that KEEPS the position axis (the `(T, C)` matrix the
    permutation/heatmap metrics plot), summing only over the batch leading axis. LM-only:
    the residual is `(B, T, d)` and CI is `(B, T, C)`."""
    site_names = model_static.site_names

    def position_ci_step(
        model: PlacedModel, placed_ci_fn: PlacedCIFn, residual: Float[Array, "*leading d"]
    ) -> tuple[dict[str, Array], dict[str, Array], Array]:
        # Training precision (bf16) readout — see make_slow_eval_step; preactivations upcast to fp32.
        preactivations = ci_preactivations(
            placed_ci_fn,
            model.clean_forward(residual, ci_capture_keys).captures,
            remat=False,
        )
        lower = {s: lower_leaky_hard_sigmoid(preactivations[s]) for s in site_names}
        upper = {s: upper_leaky_hard_sigmoid(preactivations[s]) for s in site_names}
        first = lower[site_names[0]]
        assert first.ndim == 3, f"position CI metrics are LM-only ((B, T, C)); got {first.shape}"
        n_batch = jnp.asarray(first.shape[0], jnp.int32)
        lower_sum = {s: lower[s].sum(0) for s in site_names}  # (T, C)
        upper_sum = {s: upper[s].sum(0) for s in site_names}
        return lower_sum, upper_sum, n_batch

    return filter_jit(position_ci_step, compiler_options=compiler_options)


@dataclass(frozen=True)
class PositionCI:
    """Batch-mean CI matrices for one site, position axis kept (`(T, C)`)."""

    lower: np.ndarray
    upper: np.ndarray


def accumulate_position_ci(
    position_ci_step: PositionCIStep,
    model: PlacedModel,
    placed_ci_fn: PlacedCIFn,
    residual_batches: list[Float[Array, "*leading d"]],
) -> dict[str, PositionCI]:
    """Fold `position_ci_step` over the eval batches into a batch-mean `(T, C)` CI matrix
    per site (token-weighted mean over batch elements; uniform batch makes it the plain
    mean). All batches must share one `(B, T)` shape."""
    assert residual_batches, "position CI accumulation needs at least one batch"
    lower: dict[str, np.ndarray] = {}
    upper: dict[str, np.ndarray] = {}
    total_batch = 0
    for batch_idx, residual in enumerate(residual_batches):
        lo, hi, n_batch = position_ci_step(model, placed_ci_fn, residual)
        total_batch += int(n_batch)
        for site in lo:
            lo_np, hi_np = np.asarray(lo[site]), np.asarray(hi[site])
            lower[site] = lo_np if batch_idx == 0 else lower[site] + lo_np
            upper[site] = hi_np if batch_idx == 0 else upper[site] + hi_np
    assert total_batch > 0
    return {
        site: PositionCI(lower=lower[site] / total_batch, upper=upper[site] / total_batch)
        for site in lower
    }


def permute_to_identity(ci_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column permutation toward identity via Hungarian on `-ci` over the `min(shape)`
    square block, with unassigned columns appended in order. Returns
    `(permuted (rows, C), perm_indices (C,))`. Mirrors torch `permute_to_identity_hungarian`
    / the toy `identity_ci_error`'s permutation."""
    from scipy.optimize import linear_sum_assignment

    assert ci_vals.ndim == 2, ci_vals.shape
    rows, C = ci_vals.shape
    size = min(rows, C)
    _, col_indices = linear_sum_assignment(-ci_vals[:size])
    assigned = set(col_indices.tolist())
    remaining = [c for c in range(C) if c not in assigned]
    perm = np.array(list(col_indices) + remaining, dtype=np.int64)
    return ci_vals[:, perm], perm


def permute_to_dense(ci_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column permutation by total mass, densest first (torch `permute_to_dense`).
    Returns `(permuted (rows, C), perm_indices (C,))`."""
    assert ci_vals.ndim == 2, ci_vals.shape
    perm = np.argsort(-ci_vals.sum(axis=0))
    return ci_vals[:, perm], perm


def identity_ci_error(ci_vals: np.ndarray, tolerance: float) -> int:
    """Discrete identity-CI distance (torch `IdentityCIPattern.distance_from`,
    generalizing the toy `tms`/`resid_mlp` `identity_ci_error`): permute columns toward
    identity, then over the FULL matrix minus the `min(shape)` block diagonal count entries
    `> tolerance` plus on-diagonal entries `< 1 - tolerance` (torch parity — trailing
    overcomplete columns/rows count as off-diagonal errors)."""
    ci = ci_vals.astype(np.float64)
    permuted, _ = permute_to_identity(ci)
    size = min(permuted.shape)
    off_diag = np.ones(permuted.shape, dtype=bool)
    off_diag[:size, :size] &= ~np.eye(size, dtype=bool)
    off_diag_errors = int((permuted[off_diag] > tolerance).sum())
    on_diag_errors = int((np.diagonal(permuted[:size, :size]) < (1 - tolerance)).sum())
    return off_diag_errors + on_diag_errors


def dense_ci_error(ci_vals: np.ndarray, k: int, tolerance: float, min_entries: int = 1) -> int:
    """Discrete dense-CI distance (torch `DenseCIPattern.distance_from`): sort columns by
    total mass, then over the first `k` columns count one error per column with fewer than
    `min_entries` strong activations (`>= 1 - tolerance`), and over the rest one error per
    weak activation (`> tolerance`)."""
    ci = ci_vals.astype(np.float64)
    C = ci.shape[1]
    assert k <= C, f"expected at least {k} columns, got {C}"
    sorted_ci, _ = permute_to_dense(ci)
    strong = (sorted_ci >= 1 - tolerance).sum(axis=0)
    missing_strong = np.clip(min_entries - strong, a_min=0, a_max=None)
    first_k_error = int(missing_strong[:k].sum())
    weak = (sorted_ci > tolerance).sum(axis=0)
    inactive_error = int(weak[k:].sum())
    return first_k_error + inactive_error


@dataclass(frozen=True)
class PermutationMetricSpec:
    """The permutation-plot / identity-error metrics resolved against the run's sites.

    `permutation` records, per matched site, which target shape (`identity` / `dense`)
    governs its column permutation — driving both the `PermutedCIPlots` heatmaps and the
    `UVPlots` V/U column reorder. `identity_targets` / `dense_targets` add the
    `IdentityCIError` discrete distances (per-site, by fnmatch pattern over site names).
    Empty maps mean the corresponding metric is not configured."""

    permutation: dict[str, "Literal['identity', 'dense']"]
    identity_targets: dict[str, int]
    dense_targets: dict[str, int]
    want_uv_plots: bool

    @property
    def any_plots(self) -> bool:
        return bool(self.permutation)

    @property
    def any_identity_error(self) -> bool:
        return bool(self.identity_targets) or bool(self.dense_targets)


def _resolve_permutation(
    site_names: tuple[str, ...],
    identity_patterns: list[str] | None,
    dense_patterns: list[str] | None,
) -> dict[str, "Literal['identity', 'dense']"]:
    """Map each site to its permutation target (torch `plot_causal_importance_vals`:
    identity patterns win, then dense, else default identity)."""
    resolved: dict[str, Literal["identity", "dense"]] = {}
    for name in site_names:
        if identity_patterns and any(fnmatch.fnmatch(name, p) for p in identity_patterns):
            resolved[name] = "identity"
        elif dense_patterns and any(fnmatch.fnmatch(name, p) for p in dense_patterns):
            resolved[name] = "dense"
        else:
            resolved[name] = "identity"
    return resolved


def resolve_permutation_metrics(
    site_names: tuple[str, ...], metrics: list[Any]
) -> PermutationMetricSpec:
    """Build the `PermutationMetricSpec` from the run config's typed `eval.metrics` entries
    (`UVPlots` / `PermutedCIPlots` / `IdentityCIError`). The two plot metrics share one
    column permutation; `UVPlots` additionally reorders V/U. Permutation is only computed
    when at least one plot metric is configured (both reuse it)."""
    plot_cfgs = [m for m in metrics if isinstance(m, (PermutedCIPlotsConfig, UVPlotsConfig))]
    want_uv = any(isinstance(m, UVPlotsConfig) for m in metrics)
    permutation: dict[str, Literal["identity", "dense"]] = {}
    if plot_cfgs:
        identity_patterns: list[str] = []
        dense_patterns: list[str] = []
        for cfg in plot_cfgs:
            identity_patterns += cfg.identity_patterns or []
            dense_patterns += cfg.dense_patterns or []
        permutation = _resolve_permutation(site_names, identity_patterns, dense_patterns)

    identity_targets: dict[str, int] = {}
    dense_targets: dict[str, int] = {}
    for metric in metrics:
        if not isinstance(metric, IdentityCIErrorConfig):
            continue
        for spec in metric.identity_ci or []:
            assert isinstance(spec, IdentityCITargetSpec)
            for name in site_names:
                if fnmatch.fnmatch(name, spec.layer_pattern):
                    identity_targets[name] = spec.n_features
        for spec in metric.dense_ci or []:
            assert isinstance(spec, DenseCITargetSpec)
            for name in site_names:
                if fnmatch.fnmatch(name, spec.layer_pattern):
                    dense_targets[name] = spec.k
    return PermutationMetricSpec(
        permutation=permutation,
        identity_targets=identity_targets,
        dense_targets=dense_targets,
        want_uv_plots=want_uv,
    )


def _render_figure(fig: Figure) -> bytes:
    """Encode a standalone `Figure` to PNG bytes.

    Every figure here is built with the object-oriented `Figure` API, never `pyplot`: these
    renders run on `BackgroundRenderer`'s worker thread, and pyplot's global figure registry
    is both unsynchronized (two figure tiers can render concurrently) and backed by whatever
    interactive backend the host resolves — a GUI backend refuses to build a figure manager
    off the main thread. A canvas-less `Figure` sidesteps both: `savefig` picks the Agg
    writer from the format, and the figure is garbage — not registry — collected."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf.getvalue()


def _grid_dims(n: int, max_rows: int = 6) -> tuple[int, int]:
    n_cols = (n + max_rows - 1) // max_rows
    n_rows = min(n, max_rows)
    return n_rows, n_cols


def plot_ci_value_histograms(histograms: dict[str, ValueHistogram]) -> bytes:
    """Per-site histogram of flattened CI values (torch `plot_ci_values_histograms`), drawn
    from counts binned on device — `ax.stairs` over what `ax.hist` would have computed."""
    n_rows, n_cols = _grid_dims(len(histograms))
    fig = Figure(figsize=(6 * n_cols, 5 * n_rows))
    axs = fig.subplots(n_rows, n_cols, squeeze=False)
    flat_axes = axs.T.ravel()
    for ax in flat_axes[len(histograms) :]:
        ax.set_visible(False)
    for ax, (name, histogram) in zip(flat_axes, histograms.items(), strict=False):
        ax.stairs(histogram.counts, histogram.edges, fill=True)
        ax.set_yscale("log")
        ax.set_title(f"Causal importances for {name.replace('.', '_')}")
        ax.set_xlabel("Causal importance value")
        ax.set_ylabel("Frequency")
    fig.tight_layout()
    return _render_figure(fig)


def plot_component_activation_density(densities: dict[str, np.ndarray], bins: int = 100) -> bytes:
    """Per-site histogram of per-component activation density (torch
    `plot_component_activation_density`)."""
    n_rows, n_cols = _grid_dims(len(densities))
    fig = Figure(figsize=(5 * n_cols, 5 * n_rows))
    axs = fig.subplots(n_rows, n_cols, squeeze=False)
    flat_axes = axs.T.ravel()
    for ax in flat_axes[len(densities) :]:
        ax.set_visible(False)
    for ax, (name, density) in zip(flat_axes, densities.items(), strict=False):
        ax.hist(density, bins=bins)
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("Activation density")
        ax.set_ylabel("Frequency")
    fig.tight_layout()
    return _render_figure(fig)


def plot_mean_component_cis_both_scales(
    mean_cis: dict[str, np.ndarray],
) -> tuple[bytes, bytes]:
    """Sorted-descending mean-CI scatter, linear and log y (torch
    `plot_mean_component_cis_both_scales`)."""
    sorted_data = {name: np.sort(v)[::-1] for name, v in mean_cis.items()}
    n_rows, n_cols = _grid_dims(len(sorted_data))
    images: list[bytes] = []
    for log_y in (False, True):
        fig = Figure(figsize=(8 * n_cols, 3 * n_rows))
        axs = fig.subplots(n_rows, n_cols, squeeze=False)
        flat_axes = axs.T.ravel()
        for ax in flat_axes[len(sorted_data) :]:
            ax.set_visible(False)
        for ax, (name, sorted_components) in zip(flat_axes, sorted_data.items(), strict=False):
            if log_y:
                ax.set_yscale("log")
            ax.scatter(range(len(sorted_components)), sorted_components, marker="x", s=10)
            ax.set_xlabel("Component")
            ax.set_ylabel("mean CI")
            ax.set_title(name, fontsize=10)
        fig.tight_layout()
        images.append(_render_figure(fig))
    return images[0], images[1]


def _plot_ci_matrices(matrices: dict[str, np.ndarray], colormap: str, title_prefix: str) -> bytes:
    """Per-site `(rows, C)` CI heatmaps stacked vertically with a shared colorbar (torch
    `_plot_causal_importances_figure`). `rows` is the position axis for the LM path."""
    n = len(matrices)
    fig = Figure(figsize=(5, 5 * n), layout="constrained")
    axs = fig.subplots(n, 1, squeeze=False)
    flat_axes = axs[:, 0]
    vmin = min(float(m.min()) for m in matrices.values())
    vmax = max(float(m.max()) for m in matrices.values())
    norm = Normalize(vmin=vmin, vmax=vmax)
    images = []
    for ax, (name, matrix) in zip(flat_axes, matrices.items(), strict=True):
        im = ax.matshow(matrix, aspect="auto", cmap=colormap, norm=norm)
        images.append(im)
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position("bottom")
        ax.set_xlabel("Subcomponent index")
        ax.set_ylabel("Position index")
        ax.set_title(name)
    fig.colorbar(images[0], ax=axs.ravel().tolist())
    fig.suptitle(title_prefix)
    return _render_figure(fig)


def plot_permuted_ci_heatmaps(
    position_ci: dict[str, PositionCI], permutation: dict[str, "Literal['identity', 'dense']"]
) -> tuple[bytes, bytes]:
    """The `PermutedCIPlots` figures: per-site `(position, C)` CI heatmaps with columns
    permuted toward each site's target shape (identity / dense). The lower-leaky (`Blues`) and
    upper-leaky (`Reds`) views are each permuted by their OWN-derived permutation (torch parity:
    `plot_causal_importance_vals` permutes the lower plot by a lower-derived perm, the upper by
    an upper-derived one). Returns `(lower_png, upper_png)`."""
    assert set(permutation) <= set(position_ci), "permutation sites must be a subset of CI sites"
    lower_permuted: dict[str, np.ndarray] = {}
    upper_permuted: dict[str, np.ndarray] = {}
    for name, target in permutation.items():
        pci = position_ci[name]
        permute = permute_to_identity if target == "identity" else permute_to_dense
        lower_permuted[name], _ = permute(pci.lower)
        upper_permuted[name], _ = permute(pci.upper)
    lower_png = _plot_ci_matrices(lower_permuted, "Blues", "Importance values lower leaky relu")
    upper_png = _plot_ci_matrices(upper_permuted, "Reds", "Importance values")
    return lower_png, upper_png


def uv_permutation_indices(
    permutation: dict[str, "Literal['identity', 'dense']"],
    permutation_source_ci: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Per-site column-permutation indices (C,) for the V/U reorder, derived from each
    site's `(rows, C)` upper-leaky CI matrix (identity via Hungarian, dense by column mass).
    `rows` is the position axis for the LM (`position_ci[name].upper`) or the probe-feature
    axis for the toys — the permutation is the same either way."""
    perms: dict[str, np.ndarray] = {}
    for name, target in permutation.items():
        permute = permute_to_identity if target == "identity" else permute_to_dense
        perms[name] = permute(permutation_source_ci[name])[1]
    return perms


def plot_uv_matrices(
    components: dict[str, tuple[np.ndarray, np.ndarray]],
    perms: dict[str, np.ndarray],
) -> bytes:
    """The `UVPlots` figure: per-site V `(d_in, C)` and U `(C, d_out)` heatmaps with the
    component axis reordered by `perms` (the shared identity/dense permutation, computed by
    `uv_permutation_indices`; torch `plot_UV_matrices`). One row per site, V left / U right,
    shared colorbar."""
    names = sorted(components)
    n = len(names)
    fig = Figure(figsize=(10, 5 * n), layout="constrained")
    axs = fig.subplots(n, 2, squeeze=False)
    all_vals = [m for name in names for m in components[name]]
    norm = Normalize(
        vmin=min(float(m.min()) for m in all_vals), vmax=max(float(m.max()) for m in all_vals)
    )
    images = []
    for row, name in enumerate(names):
        V, U = components[name]
        v_im = axs[row, 0].matshow(V[:, perms[name]], aspect="auto", cmap="coolwarm", norm=norm)
        axs[row, 0].set_ylabel("d_in index")
        axs[row, 0].set_xlabel("Component index")
        axs[row, 0].set_title(f"{name} (V matrix)")
        u_im = axs[row, 1].matshow(U[perms[name], :], aspect="auto", cmap="coolwarm", norm=norm)
        axs[row, 1].set_ylabel("Component index")
        axs[row, 1].set_xlabel("d_out index")
        axs[row, 1].set_title(f"{name} (U matrix)")
        images += [v_im, u_im]
    fig.colorbar(images[0], ax=axs.ravel().tolist())
    return _render_figure(fig)


def render_permutation_figures(
    spec: PermutationMetricSpec,
    position_ci: dict[str, PositionCI],
    components: dict[str, tuple[np.ndarray, np.ndarray]] | None,
) -> dict[str, bytes]:
    """The config-driven LM permutation plots (`PermutedCIPlots`, `UVPlots`) as
    `{figures/<key>: png}`, keyed as torch logs them under `slow_eval/`. Empty when neither
    plot metric is configured.

    The CI heatmaps come from `position_ci` (cheap). `components` is the host-gathered
    C-sharded V/U: pass it (and have `UVPlots` configured) to render `UVPlots`, `None` to
    skip it. The gather is NAIVE — it OOMs / breaks at production C BY DESIGN, so
    the caller only gathers when `spec.want_uv_plots` and accepts the failure at scale; the
    UV column order reuses the position-CI permutation."""
    figures: dict[str, bytes] = {}
    if not spec.any_plots:
        return figures
    lower_png, upper_png = plot_permuted_ci_heatmaps(position_ci, spec.permutation)
    figures["figures/causal_importances"] = lower_png
    figures["figures/causal_importances_upper_leaky"] = upper_png
    if spec.want_uv_plots and components is not None:
        present = {name: components[name] for name in spec.permutation}
        perms = uv_permutation_indices(
            spec.permutation, {name: position_ci[name].upper for name in spec.permutation}
        )
        figures["figures/uv_matrices"] = plot_uv_matrices(present, perms)
    return figures


def render_uv_figure(
    spec: PermutationMetricSpec,
    components: dict[str, tuple[np.ndarray, np.ndarray]],
    permutation_source_ci: dict[str, np.ndarray],
) -> dict[str, bytes]:
    """The `UVPlots` figure alone (`{figures/uv_matrices: png}`), for the positionless toys
    (TMS / ResidMLP) — they have no `(T, C)` position axis, so they drive the V/U column
    order off a per-site `(rows, C)` probe CI matrix instead. Empty unless the config names
    `UVPlots`. Toy V/U is small / replicated / already on host, so this is cheap."""
    if not spec.want_uv_plots:
        return {}
    present = {name: components[name] for name in spec.permutation}
    perms = uv_permutation_indices(spec.permutation, permutation_source_ci)
    return {"figures/uv_matrices": plot_uv_matrices(present, perms)}


def compute_identity_ci_errors(
    spec: PermutationMetricSpec, position_ci: dict[str, PositionCI], tolerance: float
) -> dict[str, float]:
    """The `IdentityCIError` discrete distances per configured site (torch
    `compute_target_metrics`), keyed `IdentityCIError/<site>` plus a summed
    `IdentityCIError` total. Empty when not configured. Operates on the batch-mean
    upper-leaky `(position, C)` CI matrix."""
    if not spec.any_identity_error:
        return {}
    per_site: dict[str, float] = {}
    for name, n_features in spec.identity_targets.items():
        matrix = position_ci[name].upper
        assert matrix.shape[1] >= n_features, (
            f"{name}: IdentityCIError expects >= {n_features} components, got {matrix.shape[1]}"
        )
        per_site[f"IdentityCIError/{name}"] = float(identity_ci_error(matrix, tolerance))
    for name, k in spec.dense_targets.items():
        per_site[f"IdentityCIError/{name}"] = float(
            dense_ci_error(position_ci[name].upper, k, tolerance)
        )
    per_site["IdentityCIError"] = float(sum(per_site.values()))
    return per_site


CI_DENSITY_HEATMAP_N_COLUMNS = 600
"""Component-axis resolution of the density heatmap: the C components (sorted desc by mean
CI) are summed into up to this many equal rank-blocks — dense left, sparse right (fewer
blocks when C is smaller, so no block is ever empty)."""
CI_DENSITY_HEATMAP_Y_DISPLAY_FLOOR = 1e-6
"""Lower limit of the (log) per-token-CI y axis. Bands span down to `CI_DENSITY_HEATMAP_FLOOR`
(1e-9), but the near-empty 1e-9..1e-6 continuum is clipped out of view; the per-column-max
colour norm is taken over the VISIBLE bands only so an off-screen band can't set the scale."""


def plot_ci_density_heatmap(
    density_hists: dict[str, np.ndarray], mean_cis: dict[str, np.ndarray]
) -> bytes:
    """The opt-in per-token CI density heatmap (one row per site). Components are sorted
    descending by mean CI and summed into up to `CI_DENSITY_HEATMAP_N_COLUMNS` equal
    rank-blocks (x); the y axis is the `n_bins` log-spaced `[FLOOR, 1]` CI bands on a LOG
    scale, ACTIVE-conditional (the underflow column is dropped, so each column's mass is
    CI ≥ FLOOR only). Color is per-column density rescaled so each column's VISIBLE max = 1.
    The sorted per-component mean CI is overlaid on a twin log axis. `density_hists[s]` is
    `(C, n_bins + 1)` (column 0 = underflow)."""
    names = list(density_hists)
    fig = Figure(figsize=(9, 3.6 * len(names)), layout="constrained")
    axs = fig.subplots(len(names), 1, squeeze=False)
    mesh = None
    for ax, name in zip(axs[:, 0], names, strict=True):
        hist = density_hists[name]
        c, n_bins = hist.shape[0], hist.shape[1] - 1
        n_cols = min(CI_DENSITY_HEATMAP_N_COLUMNS, c)
        order = np.argsort(mean_cis[name])[::-1]
        edges = np.linspace(0, c, n_cols + 1).astype(int)
        active = hist[order, 1:].astype(np.float64)  # drop underflow column
        col = np.stack([active[edges[i] : edges[i + 1]].sum(0) for i in range(n_cols)])
        col_total = col.sum(1, keepdims=True)
        density = np.divide(col, col_total, out=np.zeros_like(col), where=col_total > 0)
        y_edges = np.logspace(math.log10(CI_DENSITY_HEATMAP_FLOOR), 0.0, n_bins + 1)
        visible_band = y_edges[1:] > CI_DENSITY_HEATMAP_Y_DISPLAY_FLOOR
        col_max = np.where(visible_band, density, 0.0).max(axis=1, keepdims=True)
        plot_density = np.divide(density, col_max, out=np.zeros_like(density), where=col_max > 0)
        x_edges = np.linspace(0, c, n_cols + 1)
        cmap = colormaps["magma"].copy()
        cmap.set_bad(cmap(0.0))
        masked = np.ma.masked_where(plot_density <= 0, plot_density)
        mesh = ax.pcolormesh(
            x_edges, y_edges, masked.T, cmap=cmap, norm=Normalize(0.0, 1.0), shading="flat"
        )
        ax.set_yscale("log")
        ax.set_ylim(CI_DENSITY_HEATMAP_Y_DISPLAY_FLOOR, 1.0)
        ax.set_xlabel("Component (sorted desc by mean CI)")
        ax.set_ylabel("per-token CI")
        ax.set_title(name, fontsize=10)
        mean_sorted = mean_cis[name][order]
        block_mean = np.array([mean_sorted[edges[i] : edges[i + 1]].mean() for i in range(n_cols)])
        xc = 0.5 * (x_edges[:-1] + x_edges[1:])
        tw = ax.twinx()
        tw.plot(xc, block_mean, color="#34d8eb", lw=1.0)
        tw.set_yscale("log")
        tw.set_ylim(CI_DENSITY_HEATMAP_FLOOR, 1.0)
        tw.set_ylabel("mean CI (sorted)", color="#34d8eb", fontsize=8)
        tw.tick_params(labelsize=7, colors="#34d8eb")
    assert mesh is not None, "density_hists must be non-empty"
    fig.colorbar(mesh, ax=axs[:, 0], label="per-column density (visible col max = 1)", shrink=0.6)
    fig.suptitle("per-token CI density (active-conditional, log bins)", fontsize=11)
    return _render_figure(fig)


def render_slow_eval_figures(
    reductions: dict[str, SiteReduction],
) -> dict[str, bytes]:
    """The slow plot metrics as `{log_key: png_bytes}`, keyed exactly as torch logs them
    under `slow_eval/` (`figures/<key>` from each metric's `compute()`). The two value
    histograms appear only when the reductions carry one; a metric that renders neither
    bins nothing. When the run opts
    into the per-token CI density heatmap (`density_hist` present), it is added under
    `figures/ci_density_heatmap`."""
    assert all(r.n_positions > 0 for r in reductions.values())
    densities = {s: r.density_counts / r.n_positions for s, r in reductions.items()}
    mean_cis = {s: r.ci_sums / r.n_positions for s, r in reductions.items()}
    mean_linear, mean_log = plot_mean_component_cis_both_scales(mean_cis)
    binned = {s: r.value_histograms for s, r in reductions.items() if r.value_histograms}
    figures: dict[str, bytes] = {}
    if binned:
        assert len(binned) == len(reductions), "value_histograms must be all-sites or none"
        figures["figures/causal_importance_values"] = plot_ci_value_histograms(
            {s: lower for s, (lower, _) in binned.items()}
        )
        figures["figures/causal_importance_values_pre_sigmoid"] = plot_ci_value_histograms(
            {s: preactivations for s, (_, preactivations) in binned.items()}
        )
    figures["figures/component_activation_density"] = plot_component_activation_density(densities)
    figures["figures/ci_mean_per_component"] = mean_linear
    figures["figures/ci_mean_per_component_log"] = mean_log
    density_hists = {s: r.density_hist for s, r in reductions.items() if r.density_hist is not None}
    if density_hists:
        assert len(density_hists) == len(reductions), "density_hist must be all-sites or none"
        figures["figures/ci_density_heatmap"] = plot_ci_density_heatmap(density_hists, mean_cis)
    return figures
