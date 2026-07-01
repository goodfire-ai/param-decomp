"""JAX-native slow (plot-type) eval metrics — a LIBRARY for the in-loop slow tier (`run.py`)
and the toy eval functions (`param_decomp_lab/experiments/{tms,resid_mlp}`). Slow eval is
IN-LOOP ONLY; there is no offline/retrospective CLI.

`eval.py` runs the FAST scalar tier in-loop (CE/KL, CI-L0, the fresh-PGD probe). The
SLOW tier is the heavy plot metrics: `CIHistograms`, `ComponentActivationDensity`,
`CIMeanPerComponent` (the torch eval-metric classes of the same names). Every one of them
is a reduction over the per-site causal-importance arrays from a masked-free forward, then
a numpy/matplotlib plot. The forward + reduction is JAX; the plotting is framework-agnostic
(it mirrors the torch `param_decomp_lab/eval_metrics/plotting.py` reductions on numpy
arrays, no torch). `accumulate_site_reductions` / `render_slow_eval_figures` /
`compute_hidden_acts_metrics` are the SHARED interface both callers use.

The slow tier runs IN-LOOP on `eval.slow_every` next to the fast pass (`run.py`,
SPEC S28/S29), reusing the fast pass's eval batches and logging `slow_eval/*` on the live
`_step` axis from a rank-0 background thread.

Cross-batch reductions are exact under micro-batching: density/mean accumulate
SUM-over-positions + a position count, divided once at the end (token-weighted mean,
uniform `(B, T)` makes it the plain mean). `CIHistograms` caps its raw-value sample at
`n_batches_accum` batches, matching the torch metric's `n_batches_accum` early-stop.

It also computes the two SCALAR hidden-acts recon eval metrics (`CIHiddenActsReconLoss`,
`StochasticHiddenActsReconLoss`) natively — per decomposed site, the summed MSE between
the masked-model and target-model site OUTPUT activations, divided once by the element
count (`hidden_acts_eval.py`). Those ride the `masked_site_outputs` model seam (SPEC S31,
amended 2026-06-16 from keep-on-bridge) and are emitted as scalars under the torch log
keys (`<ClassName>/<site>` + a combined `<ClassName>`).

The three CONFIG-GATED permutation metrics (`PermutedCIPlots`, `UVPlots`,
`IdentityCIError`) are recomputed natively too, off the run's `eval.metrics` block
(re-validated from `config.yaml`, since the trainer's `EvalConfig` drops the raw metric
list). They share one column permutation per site — identity (scipy
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
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from jax import random
from jax.experimental import multihost_utils
from jaxtyping import Array, Float, Int
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from param_decomp.ci_fn import lower_leaky_hard_sigmoid, upper_leaky_hard_sigmoid
from param_decomp.configs import (
    DenseCITargetSpec,
    IdentityCIErrorConfig,
    IdentityCITargetSpec,
    PermutedCIPlotsConfig,
    UVPlotsConfig,
)
from param_decomp.hidden_acts_eval import (
    accumulate_hidden_acts,
    hidden_acts_log_entries,
    make_ci_hidden_acts_step,
    make_stochastic_hidden_acts_step,
)
from param_decomp.jit_util import filter_jit
from param_decomp.lm import DecomposedModel
from param_decomp.train import COMPUTE_DT, cast_floating

IDENTITY_CI_ERROR_TOLERANCE = 0.1
"""Torch `IdentityCIPattern.distance_from` / `compute_target_metrics` default tolerance —
avoids sensitivity to small CI values from inactive components."""


@dataclass(frozen=True)
class SiteReduction:
    """Per-site accumulators across the eval pass (all `(C,)` or scalar / capped sample).

    `density_counts[c]` = #(positions where `lower_leaky > threshold`); `ci_sums[c]` =
    Σ positions `lower_leaky`; `n_positions` = total positions seen (shared count for
    both means). `lower_sample` / `logits_sample` are flattened raw values from the first
    `n_batches_accum` batches, for the two `CIHistograms` histograms."""

    density_counts: np.ndarray
    ci_sums: np.ndarray
    n_positions: int
    lower_sample: np.ndarray
    logits_sample: np.ndarray


SlowEvalStep = Callable[
    [DecomposedModel, Any, Float[Array, "*leading d"]],
    tuple[dict[str, Array], dict[str, Array], Array, dict[str, Array], dict[str, Array]],
]
"""`(model, ci_fn, residual) -> (density_counts, ci_sums, n_positions, flat_lower,
flat_logits)` — the per-batch reduction, pre-reduced over positions. The slow plot
metrics read only the CI arrays, so V/U (`components`) is not an input. `model`
(frozen-weight-bearing) is the jit ARG."""


def make_slow_eval_step(
    lm: DecomposedModel,
    ci_alive_threshold: float,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> SlowEvalStep:
    """Build the jit'd per-batch reduction `slow_eval_step(model, ci_fn, residual) ->
    ({site: density_counts}, {site: ci_sums}, n_positions, {site: flat lower},
    {site: flat logits})`. `lower`/`logits` are returned whole (the host caps the
    histogram sample); counts/sums are pre-reduced over positions."""
    site_names = lm.site_names

    def slow_eval_step(
        model: DecomposedModel, ci_fn: Any, residual: Float[Array, "*leading d"]
    ) -> tuple[dict[str, Array], dict[str, Array], Array, dict[str, Array], dict[str, Array]]:
        # Read the CI fn in training precision (bf16), like train.py / eval.py: the readout
        # reflects the deployed model, and cuDNN flash attention rejects fp32.
        ci_fn = cast_floating(ci_fn, COMPUTE_DT)
        taps = {
            k: x.astype(COMPUTE_DT)
            for k, x in model.read_activations(residual, ci_fn.input_names).items()
        }
        logits = {s: v.astype(jnp.float32) for s, v in ci_fn(taps, remat=False).logits.items()}
        lower = {s: lower_leaky_hard_sigmoid(logits[s]) for s in site_names}

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
        flat_lower = {s: lower[s].reshape(-1) for s in site_names}
        flat_logits = {s: logits[s].reshape(-1) for s in site_names}
        return density_counts, ci_sums, n_positions, flat_lower, flat_logits

    return filter_jit(slow_eval_step, compiler_options=compiler_options)


def accumulate_site_reductions(
    slow_eval_step: SlowEvalStep,
    model: DecomposedModel,
    ci_fn: Any,
    residual_batches: list[Float[Array, "*leading d"]],
    n_batches_accum: int | None,
) -> dict[str, SiteReduction]:
    """Drive `slow_eval_step` over the eval batches and fold the per-batch reductions
    into one `SiteReduction` per site. `n_batches_accum` caps how many batches feed the
    `CIHistograms` raw-value sample (torch `n_batches_accum`); None keeps all."""
    assert residual_batches, "slow eval needs at least one batch"
    density: dict[str, np.ndarray] = {}
    sums: dict[str, np.ndarray] = {}
    lower_chunks: dict[str, list[np.ndarray]] = {}
    logits_chunks: dict[str, list[np.ndarray]] = {}
    total_positions = 0
    for batch_idx, residual in enumerate(residual_batches):
        d, s, n_pos, flat_lower, flat_logits = slow_eval_step(model, ci_fn, residual)
        total_positions += int(n_pos)
        keep_sample = n_batches_accum is None or batch_idx < n_batches_accum
        for site in d:
            counts, ci_sum = np.asarray(d[site]), np.asarray(s[site])
            density[site] = counts if batch_idx == 0 else density[site] + counts
            sums[site] = ci_sum if batch_idx == 0 else sums[site] + ci_sum
            if keep_sample:
                # The sample keeps the dp-sharded batch axis; on >1 process a bare np.asarray
                # spans non-addressable devices, so gather it (counts/sums are already reduced).
                lower_chunks.setdefault(site, []).append(
                    np.asarray(multihost_utils.process_allgather(flat_lower[site], tiled=True))
                )
                logits_chunks.setdefault(site, []).append(
                    np.asarray(multihost_utils.process_allgather(flat_logits[site], tiled=True))
                )

    return {
        site: SiteReduction(
            density_counts=density[site],
            ci_sums=sums[site],
            n_positions=total_positions,
            lower_sample=np.concatenate(lower_chunks[site]),
            logits_sample=np.concatenate(logits_chunks[site]),
        )
        for site in density
    }


PositionCIStep = Callable[
    [DecomposedModel, Any, Float[Array, "*leading d"]],
    tuple[dict[str, Array], dict[str, Array], Array],
]
"""`(model, ci_fn, residual) -> ({site: lower (T, C)}, {site: upper (T, C)}, n_batch)` —
the per-batch CI summed over the batch leading axis, position axis kept. Pairs with
`accumulate_position_ci` to form a batch-mean `(T, C)` CI matrix per site. `model`
(frozen-weight-bearing) is the jit ARG."""


def make_position_ci_step(
    lm: DecomposedModel, compiler_options: dict[str, bool | int | str] | None = None
) -> PositionCIStep:
    """Per-batch CI reduction that KEEPS the position axis (the `(T, C)` matrix the
    permutation/heatmap metrics plot), summing only over the batch leading axis. LM-only:
    the residual is `(B, T, d)` and CI is `(B, T, C)`."""
    site_names = lm.site_names

    def position_ci_step(
        model: DecomposedModel, ci_fn: Any, residual: Float[Array, "*leading d"]
    ) -> tuple[dict[str, Array], dict[str, Array], Array]:
        # Training precision (bf16) readout — see make_slow_eval_step; logits upcast to fp32.
        ci_fn = cast_floating(ci_fn, COMPUTE_DT)
        taps = {
            k: x.astype(COMPUTE_DT)
            for k, x in model.read_activations(residual, ci_fn.input_names).items()
        }
        logits = {s: v.astype(jnp.float32) for s, v in ci_fn(taps, remat=False).logits.items()}
        lower = {s: lower_leaky_hard_sigmoid(logits[s]) for s in site_names}
        upper = {s: upper_leaky_hard_sigmoid(logits[s]) for s in site_names}
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
    model: DecomposedModel,
    ci_fn: Any,
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
        lo, hi, n_batch = position_ci_step(model, ci_fn, residual)
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
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _grid_dims(n: int, max_rows: int = 6) -> tuple[int, int]:
    n_cols = (n + max_rows - 1) // max_rows
    n_rows = min(n, max_rows)
    return n_rows, n_cols


def plot_ci_value_histograms(samples: dict[str, np.ndarray], bins: int = 100) -> bytes:
    """Per-site histogram of flattened CI values (torch `plot_ci_values_histograms`)."""
    n_rows, n_cols = _grid_dims(len(samples))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    flat_axes = axs.T.ravel()
    for ax in flat_axes[len(samples) :]:
        ax.set_visible(False)
    for ax, (name, values) in zip(flat_axes, samples.items(), strict=False):
        ax.hist(values, bins=bins)
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
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
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
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 3 * n_rows), squeeze=False)
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
    fig, axs = plt.subplots(n, 1, figsize=(5, 5 * n), constrained_layout=True, squeeze=False)
    flat_axes = axs[:, 0]
    vmin = min(float(m.min()) for m in matrices.values())
    vmax = max(float(m.max()) for m in matrices.values())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
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
    fig, axs = plt.subplots(n, 2, figsize=(10, 5 * n), constrained_layout=True, squeeze=False)
    all_vals = [m for name in names for m in components[name]]
    norm = plt.Normalize(
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


def render_slow_eval_figures(
    reductions: dict[str, SiteReduction],
) -> dict[str, bytes]:
    """The three slow plot metrics as `{log_key: png_bytes}`, keyed exactly as torch
    logs them under `slow_eval/` (`figures/<key>` from each metric's `compute()`)."""
    lower_hist = plot_ci_value_histograms({s: r.lower_sample for s, r in reductions.items()})
    logits_hist = plot_ci_value_histograms({s: r.logits_sample for s, r in reductions.items()})
    assert all(r.n_positions > 0 for r in reductions.values())
    densities = {s: r.density_counts / r.n_positions for s, r in reductions.items()}
    mean_cis = {s: r.ci_sums / r.n_positions for s, r in reductions.items()}
    density_fig = plot_component_activation_density(densities)
    mean_linear, mean_log = plot_mean_component_cis_both_scales(mean_cis)
    return {
        "figures/causal_importance_values": lower_hist,
        "figures/causal_importance_values_pre_sigmoid": logits_hist,
        "figures/component_activation_density": density_fig,
        "figures/ci_mean_per_component": mean_linear,
        "figures/ci_mean_per_component_log": mean_log,
    }


def compute_hidden_acts_metrics(
    model: DecomposedModel,
    state: Any,
    token_batches: list[Int[Array, "*leading"]],
    n_mask_samples: int,
    base_key: Array,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> dict[str, float]:
    """Both hidden-acts recon eval metrics over the eval batches, keyed by the torch
    `<ClassName>[/<site>]` log keys. `state.components`/`state.ci_fn` are the restored
    trajectory; `base_key` seeds the stochastic variant's per-batch draws."""
    ci_key, stoch_key = random.split(base_key)
    ci_step = make_ci_hidden_acts_step(model, compiler_options)
    ci_reductions = accumulate_hidden_acts(
        ci_step, model, state.components, state.ci_fn, token_batches, ci_key
    )
    stoch_step = make_stochastic_hidden_acts_step(model, n_mask_samples, compiler_options)
    stoch_reductions = accumulate_hidden_acts(
        stoch_step, model, state.components, state.ci_fn, token_batches, stoch_key
    )
    return {
        **hidden_acts_log_entries("CIHiddenActsReconLoss", ci_reductions),
        **hidden_acts_log_entries("StochasticHiddenActsReconLoss", stoch_reductions),
    }


def eval_metrics_from_run_dir(run_dir: Path) -> list[Any]:
    """The typed `eval.metrics` configs from the run's `config.yaml`. The trainer's
    `EvalConfig` keeps only scalar-tier fields, so the plot/permutation metric configs are
    re-validated here from the raw block. The in-loop slow tier (`run.py`) reads the metric
    list this way to resolve the config-gated permutation/UV/identity metrics."""
    import yaml
    from pydantic import TypeAdapter

    from param_decomp.configs import AnyEvalMetricConfig

    raw = yaml.safe_load((run_dir / "config.yaml").read_text())
    adapter = TypeAdapter(AnyEvalMetricConfig)
    return [adapter.validate_python(m) for m in raw["eval"]["metrics"]]


def stochastic_hidden_acts_n_mask_samples(eval_metrics: list[Any]) -> int:
    """The `n_mask_samples` for the stochastic hidden-acts slow-eval metric, read off its
    `StochasticHiddenActsReconLossConfig` in the run's `eval.metrics` (1 when the config
    omits it). The slow tier always computes hidden-acts; absent the metric, the count is
    moot but defaults to 1."""
    from param_decomp.configs import StochasticHiddenActsReconLossConfig

    matches = [m for m in eval_metrics if isinstance(m, StochasticHiddenActsReconLossConfig)]
    assert len(matches) <= 1, f"multiple StochasticHiddenActsReconLoss eval metrics: {matches}"
    return matches[0].n_mask_samples if matches else 1
