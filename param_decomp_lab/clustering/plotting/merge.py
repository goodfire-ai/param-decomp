"""Numpy plotting for clustering diagnostics: per-iteration heatmaps, per-run cluster
sizes, and the ensemble stability plot."""

from typing import Literal

import numpy as np
from jaxtyping import Float, Int
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from param_decomp_lab.clustering.formatting import format_scientific_latex
from param_decomp_lab.clustering.merge_history import MergeHistory
from param_decomp_lab.clustering.types import ClusterCoactivationShaped, DistancesArray


def plot_coact_and_costs(
    current_coact: ClusterCoactivationShaped,
    costs: ClusterCoactivationShaped,
    iteration: int,
    tick_spacing: int = 5,
    figsize: tuple[int, int] = (12, 6),
) -> Figure:
    """Side-by-side coactivation and cost heatmaps at a single merge iteration.

    Caller is responsible for closing the returned figure with `plt.close(fig)`.
    """
    assert current_coact.shape[0] == current_coact.shape[1], "coactivation must be square"
    assert costs.shape[0] == costs.shape[1], "costs must be square"

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    coact_min: float = float(np.nanmin(current_coact))
    coact_max: float = float(np.nanmax(current_coact))
    coact_nan_diag = current_coact.copy()
    np.fill_diagonal(coact_nan_diag, np.nan)
    axs[0].matshow(coact_nan_diag, aspect="equal")
    axs[0].set_title(
        f"Coactivations\n[{format_scientific_latex(coact_min)}, "
        f"{format_scientific_latex(coact_max)}]"
    )

    costs_min: float = float(np.nanmin(costs))
    costs_max: float = float(np.nanmax(costs))
    costs_nan_diag = costs.copy()
    np.fill_diagonal(costs_nan_diag, np.nan)
    axs[1].matshow(costs_nan_diag, aspect="equal")
    axs[1].set_title(
        f"Costs\n[{format_scientific_latex(costs_min)}, {format_scientific_latex(costs_max)}]"
    )

    for ax, mat in zip(axs, (coact_nan_diag, costs_nan_diag), strict=True):
        ticks = list(range(0, mat.shape[0], tick_spacing))
        ax.set_yticks(ticks)
        ax.set_xticks(ticks)
        ax.set_xticklabels([])

    fig.suptitle(f"Iteration {iteration}")
    fig.tight_layout()
    return fig


def plot_merge_history_cluster_sizes(
    history: MergeHistory,
    figsize: tuple[int, int] = (10, 5),
) -> Figure:
    """Scatter of every cluster's size against the iteration it appears at (log y).

    Caller is responsible for closing the returned figure with `plt.close(fig)`.
    """
    n_iters: int = history.n_iters_current
    assert n_iters > 0, "history has no populated iterations"

    group_idxs: Int[np.ndarray, " n_iters n_components"] = history.merges.group_idxs[:n_iters]
    max_k: int = int(history.merges.k_groups[:n_iters].max())

    counts: Int[np.ndarray, " n_iters max_k"] = np.stack(
        [np.bincount(row[row >= 0], minlength=max_k) for row in group_idxs],
        axis=0,
    )
    iters_idx, _group_idx = np.nonzero(counts > 0)
    sizes: Int[np.ndarray, " n_points"] = counts[counts > 0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(iters_idx, sizes, "bo", markersize=3, alpha=0.15, markeredgewidth=0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cluster size")
    ax.set_yscale("log")
    ax.set_title("Distribution of cluster sizes over time")
    return fig


def plot_dists_distribution(
    distances: DistancesArray,
    mode: Literal["points", "dist"] = "points",
    label: str | None = None,
    ax: Axes | None = None,
    use_symlog: bool = True,
    linthresh: float = 1.0,
) -> Axes:
    """Plot the per-iteration distribution of pairwise ensemble distances.

    `points` overlays every pairwise distance; `dist` draws min/median/mean/max with a
    shaded inter-quartile band.
    """
    n_iters: int = distances.shape[0]
    n_ens: int = distances.shape[1]
    assert distances.shape[2] == n_ens, "distances must be square in the ensemble axes"

    dists_flat: Float[np.ndarray, " n_iters n_ens*n_ens"] = distances.reshape(n_iters, -1)

    if ax is None:
        _fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    match mode:
        case "points":
            n_pairs: int = dists_flat.shape[1]
            for i in range(n_iters):
                ax.plot(
                    np.full(n_pairs, i),
                    dists_flat[i],
                    marker="o",
                    linestyle="",
                    color="blue",
                    alpha=min(1.0, 10 / (n_ens * n_ens)),
                    markersize=5,
                    markeredgewidth=0,
                )
        case "dist":
            iterations: Int[np.ndarray, " n_iters"] = np.arange(n_iters)
            valid = [row[~np.isnan(row)] for row in dists_flat]
            mins = np.array([float(np.min(v)) if v.size else np.nan for v in valid])
            maxs = np.array([float(np.max(v)) if v.size else np.nan for v in valid])
            means = np.array([float(np.mean(v)) if v.size else np.nan for v in valid])
            medians = np.array([float(np.median(v)) if v.size else np.nan for v in valid])
            q1s = np.array([float(np.percentile(v, 25)) if v.size else np.nan for v in valid])
            q3s = np.array([float(np.percentile(v, 75)) if v.size else np.nan for v in valid])

            color = np.random.rand(3)
            ax.plot(iterations, mins, "-", color=color, alpha=0.5)
            ax.plot(iterations, maxs, "-", color=color, alpha=0.5)
            ax.plot(iterations, means, "-", color=color, linewidth=2, label=label)
            ax.plot(iterations, medians, "--", color=color, linewidth=2)
            ax.plot(iterations, q1s, ":", color=color, alpha=0.7)
            ax.plot(iterations, q3s, ":", color=color, alpha=0.7)
            ax.fill_between(iterations, q1s, q3s, color=color, alpha=0.2)

    ax.set_xlabel("Iteration #")
    ax.set_ylabel("distance")
    ax.set_title("Distribution of pairwise distances between group merges in an ensemble")

    if use_symlog:
        ax.set_yscale("symlog", linthresh=linthresh, linscale=0.2)

        def custom_format(y: float, _pos: int) -> str:
            if abs(y) < linthresh:
                return f"{y:.1f}"
            if abs(y) == 1:
                return "1"
            if abs(y) == 10:
                return "10"
            exponent = int(np.log10(abs(y)))
            return f"$10^{{{exponent}}}$"

        ax.yaxis.set_major_formatter(FuncFormatter(custom_format))
        ax.axhspan(0, linthresh, alpha=0.05, color="gray", zorder=-10)
        ax.axhline(linthresh, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    return ax
