from pathlib import Path
from typing import Any, cast

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from spd.experiments.resid_mlp.models import (
    MLP,
    ResidMLP,
)
from spd.experiments.tms.models import TMSModel
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import Components
from spd.plotting import plot_causal_importance_vals
from spd.registry import EXPERIMENT_REGISTRY
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import set_seed
from spd.utils.run_utils import ExecutionStamp


def extract_ci_val_figures(
    run_id: str, input_magnitude: float = 0.75, device: str = "cuda"
) -> dict[str, Any]:
    """Extract causal importances from a single run.

    Args:
        run_id: Wandb run ID to load model from
        input_magnitude: Magnitude of input features for causal importances plotting
        device: Device to use for model

    Returns:
        Dictionary containing causal importances data and metadata
    """
    run_info = SPDRunInfo.from_path(run_id)
    model = ComponentModel.from_run_info(run_info)
    model.to(device)

    config = run_info.config
    assert isinstance(model.target_model, ResidMLP | TMSModel), (
        "model must be a ResidMLP or TMSModel"
    )
    n_features = model.target_model.config.n_features

    # Assume no position dimension
    batch_shape = (1, n_features)

    # Get mask values without plotting regular masks
    figures, all_perm_indices_ci_vals = plot_causal_importance_vals(
        model=model,
        batch_shape=batch_shape,
        input_magnitude=input_magnitude,
        plot_raw_cis=False,
        sampling=config.sampling,
    )

    return {
        "figures": figures,
        "all_perm_indices_ci_vals": all_perm_indices_ci_vals,
        "config": config,
        "components": model.components,
        "n_features": n_features,
    }


def feature_contribution_plot(
    ax: plt.Axes,
    relu_conns: Float[Tensor, "n_layers n_features d_mlp"],
    n_layers: int,
    n_features: int,
    d_mlp: int,
    pre_labelled_neurons: dict[int, list[int]] | None = None,
    legend: bool = True,
) -> dict[int, list[int]]:
    diag_relu_conns: Float[Tensor, "n_layers n_features d_mlp"] = relu_conns.cpu().detach()

    # Define colors for different layers
    assert n_layers in [1, 2, 3]
    layer_colors = ["blue", "red", "green"]  # Always use same colors regardless of n_layers

    distinct_colors = [
        "#E41A1C",  # red
        "#377EB8",  # blue
        "#4DAF4A",  # green
        "#984EA3",  # purple
        "#FF7F00",  # orange
        "#A65628",  # brown
        "#F781BF",  # pink
        "#1B9E77",  # teal
        "#D95F02",  # dark orange
        "#7570B3",  # slate blue
        "#66A61E",  # lime green
    ]

    # Add legend if there are two layers
    if n_layers == 2 and legend:
        # Create dummy scatter plots for legend
        ax.scatter([], [], c="blue", alpha=0.3, marker=".", label="First MLP")
        ax.scatter([], [], c="red", alpha=0.3, marker=".", label="Second MLP")
        ax.legend(loc="upper right")
    # Add legend if there are three layers
    if n_layers == 3 and legend:
        # Create dummy scatter plots for legend
        ax.scatter([], [], c="blue", alpha=0.3, marker=".", label="First MLP")
        ax.scatter([], [], c="red", alpha=0.3, marker=".", label="Second MLP")
        ax.scatter([], [], c="green", alpha=0.3, marker=".", label="Third MLP")
        ax.legend(loc="upper right")
    labelled_neurons: dict[int, list[int]] = {i: [] for i in range(n_features)}

    ax.axvline(-0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
    for i in range(n_features):
        # Split points by layer and plot separately
        for layer in range(n_layers):
            ax.scatter(
                [i] * d_mlp,
                diag_relu_conns[layer, i, :],
                alpha=0.3,
                marker=".",
                c=layer_colors[layer],
            )
        ax.axvline(i + 0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
        for layer in range(n_layers):
            for j in range(d_mlp):
                # Label the neuron if it's in the pre-labelled set or if no pre-labelled set is provided
                # and the neuron has a connection strength greater than 0.1
                if (
                    pre_labelled_neurons is not None
                    and layer * d_mlp + j in pre_labelled_neurons[i]
                ) or (pre_labelled_neurons is None and diag_relu_conns[layer, i, j].item() > 0.1):
                    color_idx = j % len(distinct_colors)
                    # Make the neuron label alternate between left and right (-0.1, 0.1)
                    # Add 0.05 or -0.05 to the x coordinate to shift the label left or right
                    ax.text(
                        i,
                        diag_relu_conns[layer, i, j].item(),
                        str(layer * d_mlp + j),
                        color=distinct_colors[color_idx],
                        ha="left" if (len(labelled_neurons[i]) + 1) % 2 == 0 else "right",
                    )
                    labelled_neurons[i].append(layer * d_mlp + j)
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlim(-0.5, n_features - 0.5)
    ax.set_xlabel("Features")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return labelled_neurons


def compute_patched_weight_neuron_contributions(
    model: ResidMLP, n_features: int | None = None
) -> Float[Tensor, "n_layers n_features d_mlp"]:
    """Compute per-neuron contribution strengths for a *trained* ResidMLP.

    The returned tensor has shape ``(n_layers, n_features, d_mlp)`` recording - for
    every hidden layer and every input feature - the *virtual* weight connecting
    that feature to each neuron after the ReLU (i.e. the product ``W_in * W_out``)
    as described in the original script.

    Args:
        model: The ResidMLP model
        n_features: The number of features to keep. If None, all features are kept.

    Returns:
        A tensor of shape ``(n_layers, n_features, d_mlp)`` recording the neuron contributions.
    """

    n_features = model.config.n_features if n_features is None else n_features

    W_E: Float[Tensor, "n_features d_embed"] = model.W_E
    assert torch.equal(W_E, model.W_U.T)

    W_in_weights = []
    W_out_weights = []
    for layer in model.layers:
        layer = cast(MLP, layer)
        W_in_weights.append(layer.mlp_in.weight)
        W_out_weights.append(layer.mlp_out.weight)
    assert all(W_in_weights), "All W_in_weights must be non-None"
    assert all(W_out_weights), "All W_out_weights must be non-None"
    # Stack mlp_in / mlp_out weights across layers so that einsums can broadcast
    W_in: Float[Tensor, "n_layers d_mlp d_embed"] = torch.stack(W_in_weights, dim=0)
    W_out: Float[Tensor, "n_layers d_embed d_mlp"] = torch.stack(W_out_weights, dim=0)

    # Compute connection strengths
    in_conns: Float[Tensor, "n_layers n_features d_mlp"] = einops.einsum(
        W_E,
        W_in,
        "n_features d_embed, n_layers d_mlp d_embed -> n_layers n_features d_mlp",
    )
    out_conns: Float[Tensor, "n_layers d_mlp n_features"] = einops.einsum(
        W_out,
        W_E,
        "n_layers d_embed d_mlp, n_features d_embed -> n_layers d_mlp n_features",
    )
    relu_conns: Float[Tensor, "n_layers n_features d_mlp"] = einops.einsum(
        in_conns,
        out_conns,
        "n_layers n_features d_mlp, n_layers d_mlp n_features -> n_layers n_features d_mlp",
    )

    # Truncate to the first *n_features* for visualisation
    return relu_conns[:, :n_features, :]


def compute_spd_weight_neuron_contributions(
    model: ResidMLP,
    components: dict[str, Components],
    n_features: int | None = None,
) -> Float[Tensor, "n_layers n_features C d_mlp"]:
    """Compute per-neuron contribution strengths for the *SPD* factorisation.

    Returns a tensor of shape ``(n_layers, n_features, C, d_mlp)`` where *C* is
    the number of sub-components in the SPD decomposition.
    """

    n_layers: int = model.config.n_layers
    n_features = model.config.n_features if n_features is None else n_features

    W_E: Float[Tensor, "n_features d_embed"] = model.W_E

    # Build the *virtual* input weight matrices (V @ U) for every layer
    W_in_spd: Float[Tensor, "n_layers d_embed C d_mlp"] = torch.stack(
        [
            einops.einsum(
                components[f"layers.{i}.mlp_in"].V,
                components[f"layers.{i}.mlp_in"].U,
                "d_embed C, C d_mlp -> d_embed C d_mlp",
            )
            for i in range(n_layers)
        ],
        dim=0,
    )

    # Output weights for every layer
    W_out_spd: Float[Tensor, "n_layers d_embed d_mlp"] = torch.stack(
        [components[f"layers.{i}.mlp_out"].weight for i in range(n_layers)],
        dim=0,
    )

    # Connection strengths
    in_conns_spd: Float[Tensor, "n_layers n_features C d_mlp"] = einops.einsum(
        W_E,
        W_in_spd,
        "n_features d_embed, n_layers d_embed C d_mlp -> n_layers n_features C d_mlp",
    )
    out_conns_spd: Float[Tensor, "n_layers d_mlp n_features"] = einops.einsum(
        W_out_spd,
        W_E,
        "n_layers d_embed d_mlp, n_features d_embed -> n_layers d_mlp n_features",
    )
    relu_conns_spd: Float[Tensor, "n_layers n_features C d_mlp"] = einops.einsum(
        in_conns_spd,
        out_conns_spd,
        "n_layers n_features C d_mlp, n_layers d_mlp n_features -> n_layers n_features C d_mlp",
    )

    return relu_conns_spd[:, :n_features, :, :]


def plot_spd_feature_contributions_truncated(
    model: ResidMLP,
    components: dict[str, Components],
    n_features: int | None = 50,
):
    n_layers = model.config.n_layers
    n_features = model.config.n_features if n_features is None else n_features
    d_mlp = model.config.d_mlp

    # Assert that there are no biases
    assert not model.config.in_bias and not model.config.out_bias, (
        "Biases are not supported for these plots"
    )

    # --- Compute neuron contribution tensors ---
    relu_conns: Float[Tensor, "n_layers n_features d_mlp"] = (
        compute_patched_weight_neuron_contributions(
            model=model,
            n_features=n_features,
        )
    )

    relu_conns_spd: Float[Tensor, "n_layers n_features C d_mlp"] = (
        compute_spd_weight_neuron_contributions(
            model=model,
            components=components,
            n_features=n_features,
        )
    )

    max_component_indices = []
    for i in range(n_layers):
        # For each feature, find the C component with the largest max value over d_mlp
        max_component_indices.append(relu_conns_spd[i].max(dim=-1).values.argmax(dim=-1))
    # For each feature, use the C values based on the max_component_indices
    max_component_contributions: Float[Tensor, "n_layers n_features d_mlp"] = torch.stack(
        [
            relu_conns_spd[i, torch.arange(n_features), max_component_indices[i], :]
            for i in range(n_layers)
        ],
        dim=0,
    )

    n_rows = 2
    fig1, axes1 = plt.subplots(n_rows, 1, figsize=(10, 7), constrained_layout=True)
    axes1 = np.atleast_1d(axes1)

    labelled_neurons = feature_contribution_plot(
        ax=axes1[0],
        relu_conns=relu_conns,
        n_layers=n_layers,
        n_features=n_features,
        d_mlp=d_mlp,
        legend=True,
    )
    axes1[0].set_ylabel("Neuron contribution")
    axes1[0].set_xlabel(f"Input feature index (first {n_features} shown)")
    axes1[0].set_title("Target model")
    axes1[0].set_xticks(range(n_features))  # Ensure all xticks have labels

    feature_contribution_plot(
        ax=axes1[1],
        relu_conns=max_component_contributions,
        n_layers=n_layers,
        n_features=n_features,
        d_mlp=d_mlp,
        pre_labelled_neurons=labelled_neurons,
        legend=False,
    )
    axes1[1].set_ylabel("Neuron contribution")
    axes1[1].set_xlabel("Subcomponent index")
    axes1[1].set_title("Individual SPD subcomponents")
    axes1[1].set_xticks(range(n_features))

    # Set the same y-axis limits for both plots
    y_min = min(axes1[0].get_ylim()[0], axes1[1].get_ylim()[0])
    y_max = max(axes1[0].get_ylim()[1], axes1[1].get_ylim()[1])
    axes1[0].set_ylim(y_min, y_max)
    axes1[1].set_ylim(y_min, y_max)

    # Label the x axis with the subnets that have the largest neuron for each feature
    axes1[1].set_xticklabels(max_component_indices[0].tolist())  # Labels are the subnet indices

    return fig1


def plot_neuron_contribution_pairs(
    model: ResidMLP,
    components: dict[str, Components],
    n_features: int | None = 50,
) -> plt.Figure:
    """Create a scatter plot comparing target model and SPD component neuron contributions.

    Each point represents a (component, input_feature, neuron) combination across all layers.
    X-axis: neuron contribution from the target model
    Y-axis: neuron contribution from the SPD component
    """
    n_layers = model.config.n_layers
    n_features = model.config.n_features if n_features is None else n_features

    # Assert that there are no biases
    assert not model.config.in_bias and not model.config.out_bias, (
        "Biases are not supported for these plots"
    )

    # Compute neuron contribution tensors
    relu_conns: Float[Tensor, "n_layers n_features d_mlp"] = (
        compute_patched_weight_neuron_contributions(
            model=model,
            n_features=n_features,
        )
    )

    relu_conns_spd: Float[Tensor, "n_layers n_features C d_mlp"] = (
        compute_spd_weight_neuron_contributions(
            model=model,
            components=components,
            n_features=n_features,
        )
    )

    # For each layer and feature, find the component with the largest max value over d_mlp
    max_component_indices = []
    for i in range(n_layers):
        # For each feature, find the C component with the largest max value over d_mlp
        max_component_indices.append(relu_conns_spd[i].max(dim=-1).values.argmax(dim=-1))

    # For each feature, use the C values based on the max_component_indices
    max_component_contributions: Float[Tensor, "n_layers n_features d_mlp"] = torch.stack(
        [
            relu_conns_spd[i, torch.arange(n_features), max_component_indices[i], :]
            for i in range(n_layers)
        ],
        dim=0,
    )

    # Define colors for different layers (same as in plot_spd_feature_contributions_truncated)
    assert n_layers in [1, 2, 3]
    layer_colors = ["blue", "red", "green"]  # Always use same colors regardless of n_layers

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot points separately for each layer with different colors
    for layer in range(n_layers):
        x_values = relu_conns[layer].flatten().cpu().detach().numpy()
        y_values = max_component_contributions[layer].flatten().cpu().detach().numpy()

        layer_label = {0: "First MLP", 1: "Second MLP", 2: "Third MLP"}.get(layer, f"Layer {layer}")

        ax.scatter(
            x_values,
            y_values,
            alpha=0.3,
            s=10,
            color=layer_colors[layer],
            edgecolors="none",
            label=layer_label if n_layers > 1 else None,
        )

    # Add y=x reference line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, "k--", alpha=0.2, zorder=0, label="y=x")

    # Labels and title
    ax.set_xlabel("Target Model Neuron Contribution", fontsize=12)
    ax.set_ylabel("Subcomponent Neuron Contribution (Max Subcomponent)", fontsize=12)
    ax.set_title(
        f"{n_features} input features, {n_layers} layer{'s' if n_layers != 1 else ''}", fontsize=12
    )

    # Make axes equal and square
    ax.set_aspect("equal", adjustable="box")

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend if there are multiple layers
    if n_layers > 1:
        ax.legend(loc="lower right")

    # Add some statistics to the plot
    # Calculate correlation for all points combined
    all_x = relu_conns.flatten().cpu().detach().numpy()
    all_y = max_component_contributions.flatten().cpu().detach().numpy()
    correlation = np.corrcoef(all_x, all_y)[0, 1]
    ax.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    return fig


def main(out_dir: Path, device: str):
    canonical_runs = [
        EXPERIMENT_REGISTRY["resid_mlp1"].canonical_run,
        EXPERIMENT_REGISTRY["resid_mlp2"].canonical_run,
        EXPERIMENT_REGISTRY["resid_mlp3"].canonical_run,
    ]

    for path in canonical_runs:
        assert path is not None
        wandb_id = path.split("/")[-1]

        run_info = SPDRunInfo.from_path(path)
        model = ComponentModel.from_run_info(run_info)
        config = run_info.config
        assert isinstance(model.target_model, ResidMLP)
        model.target_model.to(device)

        n_layers = model.target_model.config.n_layers

        fig = plot_spd_feature_contributions_truncated(
            model.target_model, model.components, n_features=10
        )
        fname_weights = out_dir / f"resid_mlp_weights_{n_layers}layers_{wandb_id}.png"
        fig.savefig(
            fname_weights,
            bbox_inches="tight",
            dpi=500,
        )
        logger.info(f"Saved figure to {fname_weights}")

        # Generate and save neuron contribution pairs plot
        fig_pairs = plot_neuron_contribution_pairs(
            model.target_model,
            model.components,
            n_features=None,  # Using same number of features as above
        )
        fname_pairs = out_dir / f"neuron_contribution_pairs_{n_layers}layers_{wandb_id}.png"
        fig_pairs.savefig(
            fname_pairs,
            bbox_inches="tight",
            dpi=500,
        )
        logger.info(f"Saved figure to {fname_pairs}")

        # Define a title formatter for ResidMLP component names
        def format_resid_mlp_title(mask_name: str) -> str:
            """Convert 'layers.X.mlp_in/out' to 'Layer Y - $W_{in/out}$' with LaTeX formatting."""
            parts = mask_name.split(".")
            if len(parts) == 3 and parts[0] == "layers":
                layer_idx = int(parts[1]) + 1  # Convert to 1-based indexing
                weight_type = parts[2]
                if weight_type == "mlp_in":
                    return f"Layer {layer_idx} - $W_{{in}}$"
                elif weight_type == "mlp_out":
                    return f"Layer {layer_idx} - $W_{{out}}$"
            return mask_name  # Fallback to original if pattern doesn't match

        batch_shape = (1, model.target_model.config.n_features)
        figs_causal: dict[str, Image.Image] = plot_causal_importance_vals(
            model=model,
            batch_shape=batch_shape,
            input_magnitude=0.75,
            sampling=config.sampling,
            plot_raw_cis=False,
            title_formatter=format_resid_mlp_title,
        )[0]

        fname_importances = (
            out_dir / f"causal_importance_upper_leaky_{n_layers}layers_{wandb_id}.png"
        )
        figs_causal["causal_importances_upper_leaky"].save(fname_importances)
        logger.info(f"Saved figure to {fname_importances}")


if __name__ == "__main__":
    execution_stamp = ExecutionStamp.create(run_type="spd", create_snapshot=False)
    out_dir = execution_stamp.out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(0)
    device = get_device()

    main(out_dir, device)

    # NOTE: We used to plot the varying importance minimality coeff runs (Figure 8 in SPD paper) by
    # hackily plotting each run separately and then combining the figures. Now that out causal
    # importance plots return Image.Image objects, we can't do this.
    # We've removed this figure, but it could be supported in the future by doing the sensible thing
    # of calculating causal importances and using a custom plotting function for plotting them all
    # side-by-side.
