"""Plotting functions for merge visualizations."""

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from param_decomp.clustering.consts import (
    ClusterCoactivationShaped,
    ComponentLabels,
    DistancesArray,
)
from param_decomp.clustering.math.merge_matrix import GroupMerge
from param_decomp.clustering.merge_history import MergeHistory
from param_decomp.clustering.util import format_scientific_latex

DEFAULT_PLOT_CONFIG: dict[str, Any] = dict(
    figsize=(16, 10),
    tick_spacing=5,
    save_pdf=False,
    figure_prefix="merge_iteration",
)


def plot_merge_matrix(
    merge_matrix: Bool[Tensor, "k_groups n_components"],
    show: bool = True,
    figsize: tuple[int, int] = (10, 3),
    show_row_sums: bool | None = None,
    ax: "plt.Axes | None" = None,
    component_labels: ComponentLabels | None = None,
) -> None:
    import matplotlib.pyplot as plt

    k_groups: int
    k_groups, _ = merge_matrix.shape
    group_sizes: Int[Tensor, " k_groups"] = merge_matrix.sum(dim=1)

    if show_row_sums is None:
        show_row_sums = k_groups <= 20

    ax_lbl: plt.Axes | None = None
    if ax is not None:
        show_row_sums = False  # don't show row sums if we have an ax to plot on
        ax_mat = ax
        assert not show_row_sums
    else:
        if show_row_sums:
            _fig, (ax_mat, ax_lbl) = plt.subplots(
                1, 2, figsize=figsize, gridspec_kw={"width_ratios": [10, 1]}
            )
        else:
            _fig, ax_mat = plt.subplots(figsize=figsize)

    ax_mat.matshow(merge_matrix.cpu(), aspect="auto", cmap="Blues", interpolation="nearest")
    ax_mat.set_xlabel("Components")
    ax_mat.set_ylabel("Groups")
    ax_mat.set_title("Merge Matrix")

    # Add component labeling if component labels are provided
    if component_labels is not None:
        # Import the function here to avoid circular imports
        from param_decomp.clustering.plotting.activations import add_component_labeling

        add_component_labeling(ax_mat, component_labels, axis="x")

    if show_row_sums:
        assert ax_lbl is not None
        ax_lbl.set_xlim(0, 1)
        ax_lbl.set_ylim(-0.5, k_groups - 0.5)
        ax_lbl.invert_yaxis()
        ax_lbl.set_title("Row Sums")
        ax_lbl.axis("off")
        for i, size in enumerate(group_sizes):
            ax_lbl.text(0.5, i, str(size.item()), va="center", ha="center", fontsize=12)

    plt.tight_layout()
    if show:
        plt.show()


def plot_merge_iteration(
    current_merge: GroupMerge,
    current_coact: ClusterCoactivationShaped,
    costs: ClusterCoactivationShaped,
    # pair_cost: float,
    iteration: int,
    component_labels: ComponentLabels | None = None,
    plot_config: dict[str, Any] | None = None,
    nan_diag: bool = True,
    show: bool = False,
) -> plt.Figure:
    """Plot merge iteration results with merge tree, coactivations, and costs.

    Args:
            current_merge: Current merge state
            current_coact: Current coactivation matrix
            costs: Current cost matrix
            pair_cost: Cost of selected merge pair
            iteration: Current iteration number
            component_labels: Component labels for axis labeling
            plot_config: Plot configuration settings
            nan_diag: Whether to set diagonal to NaN for visualization
            show: Whether to display the plot (default: False)

    Returns:
            The matplotlib figure object

    Note:
            Caller is responsible for closing the returned figure with plt.close(fig)
            to prevent memory leaks.
    """
    plot_config_: dict[str, Any] = {
        **DEFAULT_PLOT_CONFIG,
        **(plot_config or {}),
    }
    axs: list[plt.Axes]
    fig, axs = plt.subplots(
        1, 3, figsize=plot_config_["figsize"], sharey=True, gridspec_kw={"width_ratios": [2, 1, 1]}
    )

    # Merge plot
    plot_merge_matrix(
        current_merge.to_matrix(),
        ax=axs[0],
        show=False,
        component_labels=component_labels,
    )

    axs[0].set_title("Merge")

    # Coactivations plot
    coact_min: float = current_coact.min().item()
    coact_max: float = current_coact.max().item()
    if nan_diag:
        current_coact = current_coact.clone()
        current_coact.fill_diagonal_(np.nan)
    axs[1].matshow(current_coact.cpu().numpy(), aspect="equal")
    coact_min_str: str = format_scientific_latex(coact_min)
    coact_max_str: str = format_scientific_latex(coact_max)
    axs[1].set_title(f"Coactivations\n[{coact_min_str}, {coact_max_str}]")

    # Setup ticks for coactivations
    k_groups: int = current_coact.shape[0]
    minor_ticks: list[int] = list(range(0, k_groups, plot_config_["tick_spacing"]))
    axs[1].set_yticks(minor_ticks)
    axs[1].set_xticks(minor_ticks)
    axs[1].set_xticklabels([])  # Remove x-axis tick labels but keep ticks

    # Costs plot
    costs_min: float = costs.min().item()
    costs_max: float = costs.max().item()
    if nan_diag:
        costs = costs.clone()
        costs.fill_diagonal_(np.nan)
    axs[2].matshow(costs.cpu().numpy(), aspect="equal")
    costs_min_str: str = format_scientific_latex(costs_min)
    costs_max_str: str = format_scientific_latex(costs_max)
    axs[2].set_title(f"Costs\n[{costs_min_str}, {costs_max_str}]")

    # Setup ticks for costs
    axs[2].set_yticks(minor_ticks)
    axs[2].set_xticks(minor_ticks)
    axs[2].set_xticklabels([])  # Remove x-axis tick labels but keep ticks

    # fig.suptitle(f"Iteration {iteration} with cost {pair_cost:.4f}")
    fig.suptitle(f"Iteration {iteration}")
    plt.tight_layout()

    if plot_config_["save_pdf"]:
        fig.savefig(
            f"{plot_config_['figure_prefix']}_iter_{iteration:03d}.pdf",
            bbox_inches="tight",
            dpi=300,
        )

    if show:
        plt.show()

    return fig


def plot_dists_distribution(
    distances: DistancesArray,
    mode: Literal["points", "dist"] = "points",
    label: str | None = None,
    ax: plt.Axes | None = None,
    kwargs_fig: dict[str, Any] | None = None,
    kwargs_plot: dict[str, Any] | None = None,
    use_symlog: bool = True,
    linthresh: float = 1.0,
) -> plt.Axes:
    n_iters: int = distances.shape[0]
    n_ens: int = distances.shape[1]
    assert distances.shape[2] == n_ens, "Distances must be square"

    # Ensure ax and kwargs_fig are not both provided
    if ax is not None and kwargs_fig is not None:
        raise ValueError("Cannot provide both ax and kwargs_fig")

    dists_flat: Float[np.ndarray, " n_iters n_ens*n_ens"] = distances.reshape(
        distances.shape[0], -1
    )

    # Create figure if ax not provided
    if ax is None:
        _fig, ax_ = plt.subplots(  # pyright: ignore[reportCallIssue]
            1,
            1,
            **dict(
                figsize=(8, 5),  # pyright: ignore[reportArgumentType]
                **(kwargs_fig or {}),
            ),
        )
    else:
        ax_ = ax

    if mode == "points":
        # Original points mode
        n_samples: int = dists_flat.shape[1]
        for i in range(n_iters):
            ax_.plot(
                np.full((n_samples), i),
                dists_flat[i],
                **dict(  # pyright: ignore[reportArgumentType]
                    marker="o",
                    linestyle="",
                    color="blue",
                    alpha=min(1, 10 / (n_ens * n_ens)),
                    markersize=5,
                    markeredgewidth=0,
                    **(kwargs_plot or {}),
                ),
            )
    elif mode == "dist":
        # Distribution statistics mode
        # Generate a random color for this plot
        color: Float[np.ndarray, " 3"] = np.random.rand(3)

        # Calculate statistics for each iteration
        mins: list[float] = []
        maxs: list[float] = []
        means: list[float] = []
        medians: list[float] = []
        q1s: list[float] = []
        q3s: list[float] = []

        for i in range(n_iters):
            # Filter out NaN values (diagonal and upper triangle)
            valid_dists: Float[np.ndarray, " n_valid"] = dists_flat[i][~np.isnan(dists_flat[i])]
            if len(valid_dists) > 0:
                mins.append(np.min(valid_dists))
                maxs.append(np.max(valid_dists))
                means.append(float(np.mean(valid_dists)))
                medians.append(float(np.median(valid_dists)))
                q1s.append(float(np.percentile(valid_dists, 25)))
                q3s.append(float(np.percentile(valid_dists, 75)))
            else:
                # Handle case with no valid distances
                mins.append(np.nan)
                maxs.append(np.nan)
                means.append(np.nan)
                medians.append(np.nan)
                q1s.append(np.nan)
                q3s.append(np.nan)

        iterations: Int[np.ndarray, " n_iters"] = np.arange(n_iters)

        # Plot statistics
        ax_.plot(iterations, mins, "-", color=color, alpha=0.5)
        ax_.plot(iterations, maxs, "-", color=color, alpha=0.5)
        ax_.plot(iterations, means, "-", color=color, linewidth=2, label=label)
        ax_.plot(iterations, medians, "--", color=color, linewidth=2)
        ax_.plot(iterations, q1s, ":", color=color, alpha=0.7)
        ax_.plot(iterations, q3s, ":", color=color, alpha=0.7)

        # Shade between quartiles
        ax_.fill_between(iterations, q1s, q3s, color=color, alpha=0.2)

    ax_.set_xlabel("Iteration #")
    ax_.set_ylabel("distance")
    ax_.set_title("Distribution of pairwise distances between group merges in an ensemble")

    if use_symlog:
        from matplotlib.ticker import FuncFormatter

        ax_.set_yscale("symlog", linthresh=linthresh, linscale=0.2)

        # Custom formatter for y-axis ticks
        def custom_format(y: float, _pos: int) -> str:
            if abs(y) < linthresh:
                # Show exact values in the linear range
                return f"{y:.1f}"
            elif abs(y) == 1:
                return "1"
            elif abs(y) == 10:
                return "10"
            else:
                # Use scientific notation for larger values
                exponent = int(np.log10(abs(y)))
                return f"$10^{{{exponent}}}$"

        ax_.yaxis.set_major_formatter(FuncFormatter(custom_format))

        # Add a visual indicator for the linear region (0 to linthresh)
        ax_.axhspan(0, linthresh, alpha=0.05, color="gray", zorder=-10)
        # Add subtle lines at linthresh boundaries
        ax_.axhline(linthresh, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
        if linthresh > 0:
            ax_.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    return ax_


def plot_merge_history_cluster_sizes(
    history: MergeHistory,
    figsize: tuple[int, int] = (10, 5),
    fmt: str = "png",
    file_prefix: str | None = None,
) -> plt.Figure:
    """Plot cluster sizes over iterations.

    Note:
        Caller is responsible for closing the returned figure with plt.close(fig)
        to prevent memory leaks.
    """
    k_groups_t: Int[Tensor, " n_iters"] = history.merges.k_groups
    valid_mask: Bool[Tensor, " n_iters"] = k_groups_t.ne(-1)
    has_data: bool = bool(valid_mask.any().item())
    if not has_data:
        raise ValueError("No populated iterations in history.k_groups")

    group_idxs_all: Int[Tensor, " n_iters n_components"] = history.merges.group_idxs[valid_mask]
    k_groups_all: Int[Tensor, " n_iters"] = k_groups_t[valid_mask]
    max_k: int = int(k_groups_all.max().item())

    counts_list: list[Int[Tensor, " max_k"]] = [
        torch.bincount(row[row.ge(0)], minlength=max_k)  # per-iteration cluster sizes
        for row in group_idxs_all
    ]
    counts: Int[Tensor, " n_iters max_k"] = torch.stack(counts_list, dim=0)

    mask_pos: Bool[Tensor, " n_iters max_k"] = counts.gt(0)
    it_idx_t, grp_idx_t = torch.nonzero(mask_pos, as_tuple=True)
    xs_t: Float[Tensor, " n_points"] = it_idx_t.to(torch.float32)
    sizes_t: Float[Tensor, " n_points"] = counts[it_idx_t, grp_idx_t].to(torch.float32)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        xs_t.cpu().numpy(), sizes_t.cpu().numpy(), "bo", markersize=3, alpha=0.15, markeredgewidth=0
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cluster size")
    ax.set_yscale("log")
    ax.set_title("Distribution of cluster sizes over time")

    if file_prefix is not None:
        fig.savefig(f"{file_prefix}_cluster_sizes.{fmt}", bbox_inches="tight", dpi=300)

    return fig
