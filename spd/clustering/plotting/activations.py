"""Plotting functions for activation visualizations."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import wandb.sdk.wandb_run
from jaxtyping import Float, Int
from torch import Tensor

from spd.clustering.activations import ProcessedActivations, compute_coactivatons
from spd.clustering.consts import ActivationsTensor, ClusterCoactivationShaped, ComponentLabels


def plot_activations(
    processed_activations: ProcessedActivations,
    save_dir: Path | None,
    n_samples_max: int,
    figure_prefix: str = "activations",
    figsize_raw: tuple[int, int] = (12, 4),
    figsize_concat: tuple[int, int] = (12, 2),
    figsize_coact: tuple[int, int] = (8, 6),
    hist_scales: tuple[str, str] = ("lin", "log"),
    hist_bins: int = 100,
    do_sorted_samples: bool = False,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> None:
    """Plot activation visualizations including raw, concatenated, sorted, and coactivations.

    Args:
        activations: Dictionary of raw activations by module
        act_concat: Concatenated activations tensor
        coact: Coactivation matrix
        labels: Component labels
        save_dir: The directory to save the plots to (None to skip saving to disk)
        figure_prefix: Prefix for PDF filenames
        figsize_raw: Figure size for raw activations
        figsize_concat: Figure size for concatenated activations
        figsize_coact: Figure size for coactivations
        hist_scales: Tuple of (x_scale, y_scale) where each is "lin" or "log"
        hist_bins: Number of bins for histograms
    """
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    act_concat: ActivationsTensor = processed_activations.activations
    coact: ClusterCoactivationShaped = compute_coactivatons(act_concat)
    labels: ComponentLabels = ComponentLabels(processed_activations.labels)
    n_samples: int = act_concat.shape[0]

    # trim the activations if n_samples_max is specified
    # clone here so we don't modify the original tensor
    act_concat = act_concat[:n_samples_max].clone()

    # Reconstruct per-module views (alive components only), truncated to n_samples_max
    act_dict: dict[str, ActivationsTensor] = {
        key: act[:n_samples_max]
        for key, act in processed_activations.get_module_activations().items()
    }

    # Update n_samples to reflect the truncated size
    n_samples = act_concat.shape[0]

    # Raw activations
    axs_act: Sequence[plt.Axes]
    _fig1: plt.Figure
    _fig1, axs_act = plt.subplots(len(act_dict), 1, figsize=figsize_raw)
    if len(act_dict) == 1:
        assert isinstance(axs_act, plt.Axes)
        axs_act = [axs_act]
    for i, (key, act) in enumerate(act_dict.items()):
        act_raw_data: np.ndarray = act.T.cpu().numpy()
        axs_act[i].matshow(
            act_raw_data, aspect="auto", vmin=act_raw_data.min(), vmax=act_raw_data.max()
        )
        axs_act[i].set_ylabel(f"components\n{key}")
        axs_act[i].set_title(f"Raw Activations: {key} (shape: {act_raw_data.shape})")

    if save_dir is not None:
        fig1_fname = save_dir / f"{figure_prefix}_raw.pdf"
        _fig1.savefig(fig1_fname, bbox_inches="tight", dpi=300)

    # Log to WandB if available
    if wandb_run is not None:
        wandb_run.log({"plots/activations/raw": wandb.Image(_fig1)}, step=0)

    # Close figure to free memory
    plt.close(_fig1)

    # Concatenated activations
    fig2: plt.Figure
    ax2: plt.Axes
    fig2, ax2 = plt.subplots(figsize=figsize_concat)
    act_data: np.ndarray = act_concat.T.cpu().numpy()
    im2 = ax2.matshow(act_data, aspect="auto", vmin=act_data.min(), vmax=act_data.max())
    ax2.set_title("Concatenated Activations")

    # Add component labeling on y-axis
    add_component_labeling(ax2, labels, axis="y")

    plt.colorbar(im2)

    if save_dir is not None:
        fig2_fname: Path = save_dir / f"{figure_prefix}_concatenated.pdf"
        fig2.savefig(fig2_fname, bbox_inches="tight", dpi=300)

    # Log to WandB if available
    if wandb_run is not None:
        wandb_run.log({"plots/activations/concatenated": wandb.Image(fig2)}, step=0)

    # Close figure to free memory
    plt.close(fig2)

    # Concatenated activations, sorted samples
    if do_sorted_samples:
        # TODO: move sample sorting logic to its own function, see
        # https://github.com/goodfire-ai/spd/pull/172/files#r2387275601
        fig3: plt.Figure
        ax3: plt.Axes
        fig3, ax3 = plt.subplots(figsize=figsize_concat)

        # Compute gram matrix (sample similarity) and sort samples using greedy ordering
        gram_matrix: Float[Tensor, "samples samples"] = act_concat @ act_concat.T

        # Normalize gram matrix to get cosine similarity
        norms: Float[Tensor, "samples 1"] = torch.norm(act_concat, dim=1, keepdim=True)
        norms = torch.where(norms > 1e-8, norms, torch.ones_like(norms))
        similarity_matrix: Float[Tensor, "samples samples"] = gram_matrix / (norms @ norms.T)

        # Greedy ordering: start with sample most similar to all others
        avg_similarity: Float[Tensor, " samples"] = similarity_matrix.mean(dim=1)
        start_idx: int = int(torch.argmax(avg_similarity).item())

        # Build ordering greedily
        ordered_indices: list[int] = [start_idx]
        remaining: set[int] = set(range(n_samples))
        remaining.remove(start_idx)

        # Greedily add the nearest unvisited sample
        current_idx: int = start_idx
        while remaining:
            # Find the unvisited sample most similar to current
            best_similarity: float = -1
            best_idx: int = -1
            for idx in remaining:
                sim: float = similarity_matrix[current_idx, idx].item()
                if sim > best_similarity:
                    best_similarity = sim
                    best_idx = idx

            ordered_indices.append(best_idx)
            remaining.remove(best_idx)
            current_idx = best_idx

        sorted_indices: Int[Tensor, " samples"] = torch.tensor(
            ordered_indices, dtype=torch.long, device=act_concat.device
        )
        act_concat_sorted: ActivationsTensor = act_concat[sorted_indices]

        # Handle log10 properly - add small epsilon to avoid log(0)
        act_sorted_data: np.ndarray = act_concat_sorted.T.cpu().numpy()
        act_sorted_log: np.ndarray = np.log10(act_sorted_data + 1e-10)
        im3 = ax3.matshow(
            act_sorted_log, aspect="auto", vmin=act_sorted_log.min(), vmax=act_sorted_log.max()
        )
        ax3.set_title("Concatenated Activations $\\log_{10}$, Sorted Samples")

        # Add component labeling on y-axis
        add_component_labeling(ax3, labels, axis="y")

        plt.colorbar(im3)

        if save_dir is not None:
            fig3_fname: Path = save_dir / f"{figure_prefix}_concatenated_sorted.pdf"
            fig3.savefig(fig3_fname, bbox_inches="tight", dpi=300)

        # Log to WandB if available
        if wandb_run is not None:
            wandb_run.log({"plots/activations/concatenated_sorted": wandb.Image(fig3)}, step=0)

        # Close figure to free memory
        plt.close(fig3)

    # Coactivations
    fig4: plt.Figure
    ax4: plt.Axes
    fig4, ax4 = plt.subplots(figsize=figsize_coact)
    coact_data: np.ndarray = coact.cpu().numpy()
    im4 = ax4.matshow(coact_data, aspect="auto", vmin=coact_data.min(), vmax=coact_data.max())
    ax4.set_title("Coactivations")

    # Add component labeling on both axes
    add_component_labeling(ax4, labels, axis="x")
    add_component_labeling(ax4, labels, axis="y")

    plt.colorbar(im4)

    if save_dir is not None:
        fig4_fname: Path = save_dir / f"{figure_prefix}_coactivations.pdf"
        fig4.savefig(fig4_fname, bbox_inches="tight", dpi=300)

    # Log to WandB if available
    if wandb_run is not None:
        wandb_run.log({"plots/activations/coactivations": wandb.Image(fig4)}, step=0)

    # Close figure to free memory
    plt.close(fig4)

    # log coactivations
    fig4_log: plt.Figure
    ax4_log: plt.Axes
    fig4_log, ax4_log = plt.subplots(figsize=figsize_coact)
    # assert np.all(coact_data >= 0) # TODO: why are coacts negative? :/
    coact_log_data: np.ndarray = np.log10(coact_data + 1e-6 + coact_data.min())
    im4_log = ax4_log.matshow(
        coact_log_data, aspect="auto", vmin=coact_log_data.min(), vmax=coact_log_data.max()
    )
    ax4_log.set_title("Coactivations $\\log_{10}$")
    # Add component labeling on both axes
    add_component_labeling(ax4_log, labels, axis="x")
    add_component_labeling(ax4_log, labels, axis="y")
    plt.colorbar(im4_log)
    if save_dir is not None:
        fig4_log_fname: Path = save_dir / f"{figure_prefix}_coactivations_log.pdf"
        fig4_log.savefig(fig4_log_fname, bbox_inches="tight", dpi=300)

    # Log to WandB if available
    if wandb_run is not None:
        wandb_run.log({"plots/activations/coactivations_log": wandb.Image(fig4_log)}, step=0)

    # Close figure to free memory
    plt.close(fig4_log)

    # Activation histograms
    fig5: plt.Figure
    ax5a: plt.Axes
    ax5b: plt.Axes
    ax5c: plt.Axes
    fig5, (ax5a, ax5b, ax5c) = plt.subplots(1, 3, figsize=(15, 4))

    x_scale: str
    y_scale: str
    x_scale, y_scale = hist_scales

    # Histogram 1: All activations
    all_activations: Float[Tensor, " samples*n_components"] = act_concat.flatten()
    all_vals: np.ndarray = all_activations.cpu().numpy()
    hist_counts: np.ndarray
    bin_edges: np.ndarray
    hist_counts, bin_edges = np.histogram(all_vals, bins=hist_bins)
    bin_centers: np.ndarray = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax5a.plot(bin_centers, hist_counts, color="blue", linewidth=2)
    ax5a.set_title("All Activations")
    ax5a.set_xlabel("Activation Value")
    ax5a.set_ylabel("Count")
    if x_scale == "log":
        ax5a.set_xscale("log")
    if y_scale == "log":
        ax5a.set_yscale("log")
    ax5a.grid(True, alpha=0.3)

    # Histogram 2: Activations per component
    n_components: int = act_concat.shape[1]

    # Common bin edges for all component histograms
    all_min: float = float(all_vals.min())
    all_max: float = float(all_vals.max())
    common_bins: np.ndarray = np.linspace(all_min, all_max, hist_bins)
    common_centers: np.ndarray = (common_bins[:-1] + common_bins[1:]) / 2

    # Get unique label prefixes and assign colors
    label_prefixes: list[str] = [label.split(":")[0] for label in labels]
    unique_prefixes: list[str] = list(dict.fromkeys(label_prefixes))  # Preserve order
    colors: Sequence[tuple[int, int, int]] = mpl.colormaps["tab10"](
        np.linspace(0, 1, len(unique_prefixes))
    )  # pyright: ignore[reportAssignmentType]
    prefix_colors: dict[str, tuple[int, int, int]] = {
        prefix: colors[i] for i, prefix in enumerate(unique_prefixes)
    }

    for comp_idx in range(n_components):
        component_activations: Float[Tensor, " n_samples"] = act_concat[:, comp_idx]
        comp_vals: np.ndarray = component_activations.cpu().numpy()
        hist_counts, _ = np.histogram(comp_vals, bins=common_bins, density=True)

        # Get color based on label prefix
        prefix: str = label_prefixes[comp_idx]
        color: tuple[int, int, int] = prefix_colors[prefix]

        ax5b.plot(common_centers, hist_counts, color=color, alpha=0.1, linewidth=1)

    ax5b.set_title(f"Per Component ({n_components} components)")
    ax5b.set_xlabel("Activation Value")
    ax5b.set_ylabel("Density")
    if x_scale == "log":
        ax5b.set_xscale("log")
    if y_scale == "log":
        ax5b.set_yscale("log")
    ax5b.grid(True, alpha=0.3)

    # Histogram 3: Activations per sample
    for sample_idx in range(n_samples):
        sample_activations: Float[Tensor, " n_components"] = act_concat[sample_idx, :]
        sample_vals: np.ndarray = sample_activations.cpu().numpy()
        hist_counts, _ = np.histogram(sample_vals, bins=common_bins, density=True)
        ax5c.plot(common_centers, hist_counts, color="blue", alpha=0.1, linewidth=1)

    ax5c.set_title(f"Per Sample ({n_samples} samples)")
    ax5c.set_xlabel("Activation Value")
    ax5c.set_ylabel("Density")
    if x_scale == "log":
        ax5c.set_xscale("log")
    if y_scale == "log":
        ax5c.set_yscale("log")
    ax5c.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir is not None:
        fig5_fname: Path = save_dir / f"{figure_prefix}_histograms.pdf"
        fig5.savefig(fig5_fname, bbox_inches="tight", dpi=300)

    # Log to WandB if available
    if wandb_run is not None:
        wandb_run.log({"plots/activations/histograms": wandb.Image(fig5)}, step=0)

    # Close figure to free memory
    plt.close(fig5)


def add_component_labeling(
    ax: plt.Axes, component_labels: ComponentLabels, axis: str = "x"
) -> None:
    """Add component labeling using major/minor ticks to show module boundaries.

    Args:
            ax: Matplotlib axis to modify
            component_labels: List of component labels in format "module:index"
            axis: Which axis to label ('x' or 'y')
    """
    if not component_labels:
        return

    # Extract module information
    module_changes: list[int] = []
    current_module: str = component_labels[0].split(":")[0]
    module_labels: list[str] = []

    for i, label in enumerate(component_labels):
        module: str = label.split(":")[0]
        if module != current_module:
            module_changes.append(i)
            module_labels.append(current_module)
            current_module = module
    module_labels.append(current_module)

    # Set up major and minor ticks
    # Minor ticks: every 10 components
    minor_ticks: list[int] = list(range(0, len(component_labels), 10))

    # Major ticks: module boundaries (start of each module)
    major_ticks: list[int] = [0] + module_changes
    major_labels: list[str] = module_labels

    if axis == "x":
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_xticks(major_ticks)
        ax.set_xticklabels(major_labels)
        ax.set_xlim(-0.5, len(component_labels) - 0.5)
        # Style the ticks
        ax.tick_params(axis="x", which="minor", length=2, width=0.5)
        ax.tick_params(axis="x", which="major", length=6, width=1.5)
        for x in major_ticks:
            ax.axvline(x - 0.5, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    else:
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticklabels(major_labels)
        ax.set_ylim(-0.5, len(component_labels) - 0.5)
        # Style the ticks
        ax.tick_params(axis="y", which="minor", length=2, width=0.5)
        ax.tick_params(axis="y", which="major", length=6, width=1.5)
        for y in major_ticks:
            ax.axhline(y - 0.5, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
