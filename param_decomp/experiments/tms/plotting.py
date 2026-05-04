"""Plotting utilities for TMS experiments.

This module provides visualization functions for analyzing TMS models and their
sparse decompositions, including vector plots, network diagrams, and weight heatmaps.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import zip_longest
from typing import cast

import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from torch import Tensor

from param_decomp.experiments.tms.models import TMSModel
from param_decomp.log import logger
from param_decomp.models.component_model import ComponentModel
from param_decomp.models.components import Components
from param_decomp.settings import REPO_ROOT


@dataclass
class PlotConfig:
    """Configuration for plot styling and parameters."""

    # Figure sizes
    vector_plot_size: tuple[float, float] = (3, 6)
    network_plot_size: tuple[float, float] = (3, 6)
    heatmap_plot_size: tuple[float, float] = (3.4, 3)

    # Thresholds
    subnet_norm_threshold: float = 0.017
    hidden_layer_threshold: float = 0.009

    # Styling
    colormap_vectors: str = "viridis"
    colormap_weights: str = "gray_r"
    colormap_heatmap: str = "bwr"

    # Layout
    vector_plot_limits: float = 1.3
    network_box_alpha: float = 0.33
    node_size: int = 200

    # Output
    dpi: int = 400


class TMSAnalyzer:
    """Analyzer for TMS model decompositions."""

    def __init__(
        self,
        model: TMSModel,
        components: dict[str, Components],
        config: PlotConfig | None = None,
    ):
        self.model = model
        self.components = components
        self.config = config or PlotConfig()

    def extract_subnets(self) -> Float[Tensor, "n_subnets n_features n_hidden"]:
        """Extract subnet weights from the component model."""
        linear1_components = self.components["linear1"]

        Vs = linear1_components.V.detach().cpu()  # (n_features, C)
        Us = linear1_components.U.detach().cpu()  # (C, n_hidden)

        # Calculate subnets: (n_features, C) x (C, n_hidden) -> (C, n_features, n_hidden)
        subnets = torch.einsum("f C, C h -> C f h", Vs, Us)
        return subnets

    def compute_cosine_similarities(
        self, eps: float = 1e-12
    ) -> tuple[
        Float[Tensor, "n_subnets n_features"],
        Float[Tensor, " n_features"],
        Float[Tensor, "n_features n_hidden"],
    ]:
        """Compute cosine similarities between subnets and target model."""
        subnets = self.extract_subnets()
        target_weights = self.model.linear1.weight.T  # (n_features, n_hidden)

        # Normalize weights
        subnets_norm = subnets / (torch.norm(subnets, dim=-1, keepdim=True) + eps)
        target_norm = target_weights / (torch.norm(target_weights, dim=-1, keepdim=True) + eps)

        # Compute cosine similarities
        cosine_sims = torch.einsum("C f h, f h -> C f", subnets_norm, target_norm)
        max_cosine_sim = cosine_sims.max(dim=0).values

        # Get subnet weights at max cosine similarity
        max_indices = cosine_sims.max(dim=0).indices
        subnet_weights_at_max = subnets[max_indices, torch.arange(self.model.config.n_features)]

        return cosine_sims, max_cosine_sim, subnet_weights_at_max

    def filter_significant_subnets(
        self, subnets: Float[Tensor, "n_subnets n_features n_hidden"]
    ) -> tuple[Float[Tensor, "n_subnets n_features n_hidden"], npt.NDArray[np.int32], int]:
        """Filter subnets based on norm threshold."""
        # Calculate norms and sum across features dimension
        subnet_feature_norms = subnets.norm(dim=-1).sum(-1)
        subnet_feature_norms_order = subnet_feature_norms.argsort(descending=True)

        # Reorder subnets by norm
        subnets = subnets[subnet_feature_norms_order]
        subnet_feature_norms = subnet_feature_norms[subnet_feature_norms_order]

        # Apply threshold
        mask = subnet_feature_norms > self.config.subnet_norm_threshold
        n_significant = int(mask.sum().item())

        # Filter subnets
        filtered_subnets = subnets[mask]

        subnets_indices = subnet_feature_norms_order[:n_significant].cpu().numpy()

        return filtered_subnets, subnets_indices, n_significant


class VectorPlotter:
    """Handles 2D vector plotting for subnetworks."""

    def __init__(self, config: PlotConfig):
        self.config = config

    def plot(
        self,
        subnets: Float[Tensor, "n_subnets n_features n_hidden"],
        axs: npt.NDArray[np.object_],
        subnets_indices: npt.NDArray[np.int32],
    ) -> None:
        """Create 2D polygon plots of subnetworks."""
        n_subnets, n_features, _ = subnets.shape

        # Use different colors for each feature
        color_vals = np.linspace(0, 1, n_features)
        colors = plt.colormaps[self.config.colormap_vectors](color_vals)

        for subnet_idx in range(n_subnets):
            ax = axs[subnet_idx]
            self._plot_single_vector(ax, subnets[subnet_idx].cpu().detach().numpy(), colors)
            self._style_axis(ax)

            ax.set_title(
                self._get_subnet_label(subnet_idx, subnets_indices),
                pad=10,
                fontsize="large",
            )

    def _plot_single_vector(
        self, ax: Axes, vectors: npt.NDArray[np.float64], colors: npt.NDArray[np.float64]
    ) -> None:
        """Plot vectors for a single subnet."""
        n_features = vectors.shape[0]

        for j in range(n_features):
            # Plot points
            ax.scatter(vectors[j, 0], vectors[j, 1], color=colors[j])
            # Plot lines from origin
            ax.add_collection(
                mc.LineCollection([[(0, 0), (vectors[j, 0], vectors[j, 1])]], colors=[colors[j]])
            )

    def _style_axis(self, ax: Axes) -> None:
        """Apply consistent styling to axis."""
        ax.set_aspect("equal")
        ax.set_facecolor("#f6f6f6")
        ax.set_xlim((-self.config.vector_plot_limits, self.config.vector_plot_limits))
        ax.set_ylim((-self.config.vector_plot_limits, self.config.vector_plot_limits))
        ax.tick_params(left=True, right=False, labelleft=False, labelbottom=False, bottom=True)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_position("center")

    @staticmethod
    def _get_subnet_label(subnet_idx: int, subnets_indices: npt.NDArray[np.int32]) -> str:
        """Get appropriate label for subnet."""
        if subnet_idx == 0:
            return "Target model"
        elif subnet_idx == 1:
            return "Sum of components"
        else:
            return f"Subcomponent {subnets_indices[subnet_idx - 2]}"


class NetworkDiagramPlotter:
    """Handles neural network diagram plotting."""

    def __init__(self, config: PlotConfig):
        self.config = config

    def plot(
        self,
        subnets: Float[Tensor, "n_subnets n_features n_hidden"],
        axs: npt.NDArray[np.object_],
    ) -> None:
        """Plot neural network diagrams for models without hidden layers.

        This shows the decomposition of the first linear layer (input → hidden)
        and its transpose (hidden → output).
        """
        n_subnets, n_features, n_hidden = subnets.shape

        # Take absolute values for visualization
        subnets_abs = subnets.abs()

        axs = np.atleast_1d(np.array(axs))
        self._add_labels(axs[0])

        cmap = plt.colormaps[self.config.colormap_weights]

        for subnet_idx in range(n_subnets):
            ax = axs[subnet_idx]
            self._plot_single_network(
                ax,
                subnets_abs[subnet_idx].cpu().detach().numpy(),
                subnets_abs.max().item(),
                n_features,
                n_hidden,
                cmap,
            )
            self._style_network_axis(ax)

    def _add_labels(self, ax: Axes) -> None:
        """Add input/output labels to first axis."""
        ax.text(
            0.05,
            0.05,
            "Outputs (before bias & ReLU)",
            ha="left",
            va="center",
            transform=ax.transAxes,
        )
        ax.text(0.05, 0.95, "Inputs", ha="left", va="center", transform=ax.transAxes)

    def _plot_single_network(
        self,
        ax: Axes,
        weights: npt.NDArray[np.float64],
        max_weight: float,
        n_features: int,
        n_hidden: int,
        cmap: Colormap,
    ) -> None:
        """Plot a single network diagram."""
        # Define node positions
        y_input, y_hidden, y_output = 0, -1, -2
        x_input = np.linspace(0.05, 0.95, n_features).astype(np.float64)
        x_hidden = np.linspace(0.25, 0.75, n_hidden).astype(np.float64)
        x_output = np.linspace(0.05, 0.95, n_features).astype(np.float64)

        # Add hidden layer background
        self._add_hidden_layer_box(ax, y_hidden)

        # Plot nodes
        self._plot_nodes(
            ax, x_input, y_input, x_hidden, y_hidden, x_output, y_output, n_features, n_hidden
        )

        # Plot edges
        self._plot_edges(
            ax,
            weights,
            max_weight,
            x_input,
            y_input,
            x_hidden,
            y_hidden,
            x_output,
            y_output,
            n_features,
            n_hidden,
            cmap,
        )

    def _add_hidden_layer_box(self, ax: Axes, y_hidden: float) -> None:
        """Add background box for hidden layer."""
        box = plt.Rectangle(
            (0.1, y_hidden - 0.2),
            0.8,
            0.4,
            fill=True,
            facecolor="#e4e4e4",
            edgecolor="none",
            alpha=self.config.network_box_alpha,
            transform=ax.transData,
        )
        ax.add_patch(box)

    def _plot_nodes(
        self,
        ax: Axes,
        x_input: npt.NDArray[np.float64],
        y_input: float,
        x_hidden: npt.NDArray[np.float64],
        y_hidden: float,
        x_output: npt.NDArray[np.float64],
        y_output: float,
        n_features: int,
        n_hidden: int,
    ) -> None:
        """Plot network nodes."""
        ax.scatter(
            x_input,
            [y_input] * n_features,
            s=self.config.node_size,
            color="grey",
            edgecolors="k",
            zorder=3,
        )
        ax.scatter(
            x_hidden,
            [y_hidden] * n_hidden,
            s=self.config.node_size,
            color="grey",
            edgecolors="k",
            zorder=3,
        )
        ax.scatter(
            x_output,
            [y_output] * n_features,
            s=self.config.node_size,
            color="grey",
            edgecolors="k",
            zorder=3,
        )

    def _plot_edges(
        self,
        ax: Axes,
        weights: npt.NDArray[np.float64],
        max_weight: float,
        x_input: npt.NDArray[np.float64],
        y_input: float,
        x_hidden: npt.NDArray[np.float64],
        y_hidden: float,
        x_output: npt.NDArray[np.float64],
        y_output: float,
        n_features: int,
        n_hidden: int,
        cmap: Colormap,
    ) -> None:
        """Plot network edges with weight-based coloring."""
        # Ensure max_weight is never zero
        max_weight = max_weight if max_weight > 0 else 1

        # Input to hidden
        for i in range(n_features):
            for h in range(n_hidden):
                weight = weights[i, h]
                normalized_weight = weight / max_weight
                color = cmap(normalized_weight)
                ax.plot(
                    [x_input[i], x_hidden[h]],
                    [y_input, y_hidden],
                    color=color,
                    linewidth=0.5 + 1.5 * normalized_weight,
                    alpha=0.3 + 0.7 * normalized_weight,
                )

        # Hidden to output (transpose for W^T)
        for h in range(n_hidden):
            for o in range(n_features):
                weight = weights[o, h]
                normalized_weight = weight / max_weight
                color = cmap(normalized_weight)
                ax.plot(
                    [x_hidden[h], x_output[o]],
                    [y_hidden, y_output],
                    color=color,
                    linewidth=0.5 + 1.5 * normalized_weight,
                    alpha=0.3 + 0.7 * normalized_weight,
                )

    def _style_network_axis(self, ax: Axes) -> None:
        """Style network diagram axis."""
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-2.5, 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(False)


class FullNetworkDiagramPlotter:
    """Handles full neural network diagram plotting including hidden layers."""

    def __init__(self, config: PlotConfig):
        self.config = config

    def plot(
        self,
        model: TMSModel,
        components: dict[str, Components],
    ) -> Figure:
        """Plot full network architecture with all layers."""
        # Extract all layer weights
        config = model.config

        # Get subnet decompositions for linear1
        linear1_components = components["linear1"]
        assert isinstance(linear1_components, Components)
        Vs = linear1_components.V.detach().cpu()
        Us = linear1_components.U.detach().cpu()
        linear1_subnets = torch.einsum("f C, C h -> C f h", Vs, Us)

        # Get hidden layer decompositions if they exist
        hidden_layer_components = None
        if config.n_hidden_layers > 0:
            hidden_layer_components = []
            for i in range(config.n_hidden_layers):
                hidden_comp_name = f"hidden_layers-{i}"
                hidden_comp = components[hidden_comp_name]
                assert isinstance(hidden_comp, Components)
                hidden_V = hidden_comp.V.detach().cpu()
                hidden_U = hidden_comp.U.detach().cpu()
                hidden_weights = torch.einsum("h C, C j -> C h j", hidden_V, hidden_U)
                hidden_layer_components.append(hidden_weights)

        # Determine which components are significant in linear1 vs hidden layers
        linear1_norms = linear1_subnets.norm(dim=(1, 2))
        hidden_norms = None
        if hidden_layer_components:
            # Sum norms across all hidden layers for each component
            hidden_norms = torch.zeros(linear1_norms.shape[0])
            for hw in hidden_layer_components:
                hidden_norms += hw.norm(dim=(1, 2))

        # Classify components as either "linear" or "hidden" based on where they have larger norms
        component_types = []
        for c_idx in range(linear1_norms.shape[0]):
            if hidden_norms is None:
                component_types.append("linear")
            else:
                if linear1_norms[c_idx] > hidden_norms[c_idx]:
                    component_types.append("linear")
                else:
                    component_types.append("hidden")

        # Filter significant components overall
        total_norms = linear1_norms.clone()
        if hidden_norms is not None:
            total_norms += hidden_norms

        significant_mask = total_norms > self.config.subnet_norm_threshold
        significant_indices = torch.where(significant_mask)[0]

        # Prepare data for plotting
        plot_configs = []

        # Target model
        plot_configs.append(
            {
                "title": "Target model",
                "linear1_weights": model.linear1.weight.T.detach().cpu().numpy(),
                "hidden_weights": [
                    cast(torch.nn.Linear, model.hidden_layers[i]).weight.T.detach().cpu().numpy()
                    for i in range(config.n_hidden_layers)
                ]
                if config.n_hidden_layers > 0 and model.hidden_layers is not None
                else None,
                "component_type": "full",
            }
        )

        # Sum of components
        sum_linear1 = linear1_subnets.sum(dim=0).numpy()
        sum_hidden = None
        if hidden_layer_components:
            sum_hidden = [hw.sum(dim=0).numpy() for hw in hidden_layer_components]
        plot_configs.append(
            {
                "title": "Sum of components",
                "linear1_weights": sum_linear1,
                "hidden_weights": sum_hidden,
                "component_type": "full",
            }
        )

        # Individual significant components
        for idx in significant_indices:
            comp_type = component_types[idx]
            if comp_type == "linear":
                # Linear component: show weights in linear1/2, zeros in hidden
                linear_weights = linear1_subnets[idx].numpy()
                hidden_weights = None
                if config.n_hidden_layers > 0 and model.hidden_layers is not None:
                    # Show zeros for hidden layers (not identity)
                    hidden_weights = [
                        np.zeros((config.n_hidden, config.n_hidden))
                        for _ in range(config.n_hidden_layers)
                    ]
            else:
                # Hidden component: show zeros in linear1/2, actual weights in hidden
                linear_weights = np.zeros((config.n_features, config.n_hidden))
                hidden_weights = None
                if hidden_layer_components is not None:
                    hidden_weights = [hw[idx].numpy() for hw in hidden_layer_components]

            plot_configs.append(
                {
                    "title": f"Subcomponent {idx.item()}",
                    "linear1_weights": linear_weights,
                    "hidden_weights": hidden_weights,
                    "component_type": comp_type,
                }
            )

        # Create figure
        n_plots = len(plot_configs)
        fig, axs = plt.subplots(
            nrows=1,
            ncols=n_plots,
            figsize=(4 * n_plots, 6 + 2 * config.n_hidden_layers),
        )

        # Ensure axs is always iterable
        axs_array = [axs] if n_plots == 1 else np.array(axs).flatten()

        # Plot each configuration
        for _, (ax, plot_config) in enumerate(zip_longest(axs_array, plot_configs)):
            if ax is None or plot_config is None:
                break
            assert isinstance(ax, Axes)
            self._plot_full_network(
                ax,
                linear1_weights=plot_config["linear1_weights"],
                hidden_weights=plot_config["hidden_weights"],
                component_type=plot_config["component_type"],
                n_features=config.n_features,
                n_hidden=config.n_hidden,
                n_hidden_layers=config.n_hidden_layers,
            )
            ax.set_title(plot_config["title"], pad=10, fontsize="large")

        return fig

    def _plot_full_network(
        self,
        ax: Axes,
        linear1_weights: npt.NDArray[np.float64],
        hidden_weights: list[npt.NDArray[np.float64]] | None,
        component_type: str,
        n_features: int,
        n_hidden: int,
        n_hidden_layers: int,
    ) -> None:
        """Plot a complete network architecture."""
        # Calculate positions
        total_positions = 3 + n_hidden_layers
        y_positions = np.linspace(0, -(total_positions - 1), total_positions)

        # Node x positions
        x_input = np.linspace(0.1, 0.9, n_features).astype(np.float64)
        x_hidden = np.linspace(0.2, 0.8, n_hidden).astype(np.float64)
        x_output = np.linspace(0.1, 0.9, n_features).astype(np.float64)

        # Plot nodes

        # Input nodes
        ax.scatter(
            x_input,
            [y_positions[0]] * n_features,
            s=self.config.node_size,
            color="grey",
            edgecolors="k",
            zorder=3,
        )

        # All hidden layers
        for layer_idx in range(1 + n_hidden_layers):
            y = y_positions[layer_idx + 1]
            ax.scatter(
                x_hidden,
                [y] * n_hidden,
                s=self.config.node_size,
                color="grey",
                edgecolors="k",
                zorder=3,
            )
            # Add background box
            box = plt.Rectangle(
                (0.15, y - 0.15),
                0.7,
                0.3,
                fill=True,
                facecolor="#e4e4e4",
                edgecolor="none",
                alpha=self.config.network_box_alpha,
                transform=ax.transData,
            )
            ax.add_patch(box)

        # Output nodes
        ax.scatter(
            x_output,
            [y_positions[-1]] * n_features,
            s=self.config.node_size,
            color="grey",
            edgecolors="k",
            zorder=3,
        )

        # Plot edges
        cmap = plt.colormaps[self.config.colormap_weights]

        # Determine if this component uses linear weights
        show_linear_weights = component_type in ["full", "linear"]

        # Input to first hidden (linear1)
        weights_abs = np.abs(linear1_weights)
        max_weight = weights_abs.max() if weights_abs.max() > 0 else 1

        if show_linear_weights:
            # Show actual weights
            for i in range(n_features):
                for h in range(n_hidden):
                    weight = weights_abs[i, h]
                    normalized_weight = weight / max_weight if max_weight > 0 else 0
                    color = cmap(normalized_weight)
                    ax.plot(
                        [x_input[i], x_hidden[h]],
                        [y_positions[0], y_positions[1]],
                        color=color,
                        linewidth=0.5 + 1.5 * normalized_weight,
                        alpha=0.3 + 0.7 * normalized_weight,
                    )
        # If not showing linear weights, draw nothing at all

        # Hidden to hidden layers
        if hidden_weights and n_hidden_layers > 0:
            for layer_idx, hw in enumerate(hidden_weights):
                hw_abs = np.abs(hw)
                max_hw = hw_abs.max() if hw_abs.max() > 0 else 1
                from_y = y_positions[layer_idx + 1]
                to_y = y_positions[layer_idx + 2]

                # Only draw connections if there are non-zero weights
                if np.any(hw_abs > 0.01):  # Threshold for visibility
                    for h1 in range(n_hidden):
                        for h2 in range(n_hidden):
                            weight = hw_abs[h1, h2]
                            normalized_weight = weight / max_hw if max_hw > 0 else 0
                            if normalized_weight > 0.01:  # Only draw visible connections
                                color = cmap(normalized_weight)
                                ax.plot(
                                    [x_hidden[h1], x_hidden[h2]],
                                    [from_y, to_y],
                                    color=color,
                                    linewidth=0.5 + 1.5 * normalized_weight,
                                    alpha=0.3 + 0.7 * normalized_weight,
                                )
                # If weights are all near zero, draw nothing

        # Last hidden to output (transpose of linear1)
        if show_linear_weights:
            linear1_T_abs = weights_abs.T
            last_hidden_idx = 1 + n_hidden_layers
            for h in range(n_hidden):
                for o in range(n_features):
                    weight = linear1_T_abs[h, o]
                    normalized_weight = weight / max_weight if max_weight > 0 else 0
                    color = cmap(normalized_weight)
                    ax.plot(
                        [x_hidden[h], x_output[o]],
                        [y_positions[last_hidden_idx], y_positions[-1]],
                        color=color,
                        linewidth=0.5 + 1.5 * normalized_weight,
                        alpha=0.3 + 0.7 * normalized_weight,
                    )
        # If not showing linear weights, draw nothing at all

        # Add layer labels
        ax.text(0.0, y_positions[0], "Input", ha="right", va="center", fontsize="medium")
        ax.text(0.05, y_positions[1], "Hidden 1", ha="right", va="center", fontsize="medium")
        for i in range(n_hidden_layers):
            ax.text(
                0.05,
                y_positions[i + 2],
                f"Hidden {i + 2}",
                ha="right",
                va="center",
                fontsize="medium",
            )
        ax.text(0.0, y_positions[-1], "Output", ha="right", va="center", fontsize="medium")

        # Style axis
        ax.set_xlim(-0.2, 1.05)
        ax.set_ylim(y_positions[-1] - 0.5, y_positions[0] + 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(False)


class HiddenLayerPlotter:
    """Handles hidden layer weight heatmap plotting."""

    def __init__(self, config: PlotConfig):
        self.config = config

    def plot(self, model: TMSModel, components: dict[str, Components]) -> Figure:
        """Plot hidden layer weights as heatmaps."""
        # Extract weights
        hidden_weights, target_weights, subnets_order = self._extract_hidden_weights(
            model, components
        )

        # Filter by threshold
        hidden_weights_norm = hidden_weights.norm(dim=(-1, -2))
        n_significant = int((hidden_weights_norm > self.config.hidden_layer_threshold).sum().item())
        n_subnets = n_significant + 2  # Add target and sum

        # Prepare data for plotting
        sum_weights = hidden_weights.sum(dim=0, keepdim=True)
        all_weights = torch.cat([target_weights, sum_weights, hidden_weights], dim=0)

        # Create figure
        fig, axs = plt.subplots(
            1,
            n_subnets,
            figsize=(
                self.config.heatmap_plot_size[0] * n_subnets,
                self.config.heatmap_plot_size[1],
            ),
        )

        # Ensure axs is iterable even for single subplot
        from matplotlib.axes import Axes as AxesType

        axs_list: list[AxesType] = [axs] if isinstance(axs, AxesType) else list(axs)

        # Plot heatmaps
        self._plot_heatmaps(fig, axs_list, all_weights, subnets_order, n_subnets)

        return fig

    def _extract_hidden_weights(
        self, model: TMSModel, components: dict[str, Components]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Extract and sort hidden layer weights."""
        if model.hidden_layers is None:
            raise ValueError("model must have hidden layers")

        hidden_comp_name = "hidden_layers-0"
        hidden_components = components[hidden_comp_name]
        assert isinstance(hidden_components, Components)

        hidden_V = hidden_components.V.detach().cpu()
        hidden_U = hidden_components.U.detach().cpu()
        hidden_weights = torch.einsum("f C, C h -> C f h", hidden_V, hidden_U)

        # Sort by norm
        weights_norm = hidden_weights.norm(dim=(-1, -2))
        order = weights_norm.argsort(dim=0, descending=True)
        hidden_weights = hidden_weights[order]

        # Get target weights
        target_weights = (
            cast(torch.nn.Linear, model.hidden_layers[0]).weight.T.unsqueeze(0).detach().cpu()
        )

        return hidden_weights, target_weights, order

    def _plot_heatmaps(
        self,
        fig: Figure,
        axs: Sequence[Axes],
        weights: Tensor,
        subnets_order: Tensor,
        n_subnets: int,
    ) -> None:
        """Plot weight heatmaps with consistent colormap."""
        cmap = plt.colormaps[self.config.colormap_heatmap]
        vmax = float(torch.max(torch.abs(weights.min()), torch.abs(weights.max())).item())
        vmin = -vmax

        for idx in range(n_subnets):
            ax = axs[idx]
            ax.imshow(weights[idx].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)

            # Set title
            if idx == 0:
                title = "Target model"
            elif idx == 1:
                title = "Sum of components"
            else:
                title = f"Subcomponent {subnets_order[idx - 2].item()}"
            ax.set_title(title, pad=10, fontsize="large")

            # Style axis
            ax.set_xticks([])
            ax.set_yticks([])

        # Add colorbar
        self._add_colorbar(fig, cmap, vmin, vmax)

    def _add_colorbar(self, fig: Figure, cmap: Colormap, vmin: float, vmax: float) -> None:
        """Add colorbar to figure."""
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # pyright: ignore[reportCallIssue,reportArgumentType]
        fig.colorbar(ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax)), cax=cbar_ax)
        cbar_ax.set_ylabel("Weight magnitude", fontsize="large")
        cbar_ax.tick_params(labelsize="large")


class TMSPlotter:
    """Main plotting interface for TMS experiments."""

    def __init__(
        self,
        model: TMSModel,
        components: dict[str, Components],
        config: PlotConfig | None = None,
    ):
        self.config = config or PlotConfig()
        self.analyzer = TMSAnalyzer(model, components, self.config)
        self.vector_plotter = VectorPlotter(self.config)
        self.network_plotter = NetworkDiagramPlotter(self.config)
        self.full_network_plotter = FullNetworkDiagramPlotter(self.config)
        self.hidden_plotter = HiddenLayerPlotter(self.config)

    def plot_combined_diagram(self) -> Figure:
        """Create combined vector and network diagram figure.

        Note: Only works for models without hidden layers.
        For models with hidden layers, use plot_vectors() and plot_full_network() separately.
        """
        if self.analyzer.model.config.n_hidden_layers > 0:
            raise ValueError(
                "Combined diagram not supported for models with hidden layers. "
                "Use plot_vectors() and plot_full_network() separately."
            )

        # Extract and prepare data
        subnets = self.analyzer.extract_subnets()
        target_weights = self.analyzer.model.linear1.weight.T.detach().cpu()

        # Filter significant subnets
        filtered_subnets, subnets_indices, n_significant = self.analyzer.filter_significant_subnets(
            subnets
        )

        # Add target and sum panels
        target_subnet = target_weights.unsqueeze(0)
        summed_subnet = filtered_subnets.sum(dim=0, keepdim=True)
        all_subnets = torch.cat([target_subnet, summed_subnet, filtered_subnets], dim=0)
        n_subnets = n_significant + 2

        # Create figure
        fig, axs = plt.subplots(
            nrows=2,
            ncols=n_subnets,
            figsize=(
                self.config.vector_plot_size[0] * n_subnets,
                self.config.vector_plot_size[1],
            ),
        )
        plt.subplots_adjust(hspace=0)

        axs = np.atleast_2d(np.array(axs))

        # Plot vectors and networks
        self.vector_plotter.plot(all_subnets, axs[0, :], subnets_indices)
        self.network_plotter.plot(all_subnets, axs[1, :])

        return fig

    def plot_vectors(self) -> Figure:
        """Create figure with only vector diagrams."""
        # Extract and prepare data
        subnets = self.analyzer.extract_subnets()
        target_weights = self.analyzer.model.linear1.weight.T.detach().cpu()

        # Filter significant subnets
        filtered_subnets, subnets_indices, n_significant = self.analyzer.filter_significant_subnets(
            subnets
        )

        # Add target and sum panels
        target_subnet = target_weights.unsqueeze(0)
        summed_subnet = filtered_subnets.sum(dim=0, keepdim=True)
        all_subnets = torch.cat([target_subnet, summed_subnet, filtered_subnets], dim=0)
        n_subnets = n_significant + 2

        # Create figure
        fig, axs = plt.subplots(
            nrows=1,
            ncols=n_subnets,
            figsize=(
                self.config.vector_plot_size[0] * n_subnets,
                self.config.vector_plot_size[1],
            ),
        )

        axs = np.atleast_1d(np.array(axs))

        # Plot vectors
        self.vector_plotter.plot(all_subnets, axs, subnets_indices)

        return fig

    def plot_full_network(self) -> Figure:
        """Create full network diagram showing all layers."""
        return self.full_network_plotter.plot(self.analyzer.model, self.analyzer.components)

    def plot_cosine_similarity_analysis(self) -> Figure:
        """Plot cosine similarity analysis."""
        _, max_cosine_sim, _ = self.analyzer.compute_cosine_similarities()

        fig, ax = plt.subplots()
        ax.bar(range(max_cosine_sim.shape[0]), max_cosine_sim.cpu().detach().numpy())
        ax.axhline(1, color="grey", linestyle="--")
        ax.set_xlabel("Input feature index")
        ax.set_ylabel("Max cosine similarity")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return fig

    def plot_hidden_layers(self) -> Figure | None:
        """Plot hidden layer weights if model has hidden layers."""
        if self.analyzer.model.config.n_hidden_layers > 0:
            return self.hidden_plotter.plot(self.analyzer.model, self.analyzer.components)
        return None

    def get_analysis_summary(self) -> dict[str, float]:
        """Print analysis summary statistics."""
        output: dict[str, float] = {}
        _, max_cosine_sim, subnet_weights_at_max = self.analyzer.compute_cosine_similarities()

        output["Max cosine similarity"] = max_cosine_sim.item()
        output["Mean max cosine similarity"] = max_cosine_sim.mean().item()
        output["Std max cosine similarity"] = max_cosine_sim.std().item()

        # L2 ratio analysis
        target_weights = self.analyzer.model.linear1.weight.T.detach().cpu()
        target_norm = torch.norm(target_weights, dim=-1, keepdim=True)
        subnet_norm = torch.norm(subnet_weights_at_max, dim=-1, keepdim=True)
        l2_ratio = subnet_norm / target_norm

        output["Mean L2 ratio"] = l2_ratio.mean().item()
        output["Std L2 ratio"] = l2_ratio.std().item()

        return output


def main():
    """Main execution function."""

    # Define run configurations with custom PlotConfig for each
    run_configs = {
        # "wandb:goodfire/spd-tms/runs/f63itpo1": {"config": PlotConfig(), "name": "5-2"},
        "wandb:goodfire/spd-tms/runs/8bxfjeu5": {
            "config": PlotConfig(subnet_norm_threshold=0.03, hidden_layer_threshold=0.0115),
            "name": "5-2-identity",
        },
        # "wandb:goodfire/spd-tms/runs/xq1ivc6b": {"config": PlotConfig(), "name": "40-10"},
        # "wandb:goodfire/spd-tms/runs/xyq22lbc": {"config": PlotConfig(), "name": "40-10-identity"},
    }

    for run_id, run_info in run_configs.items():
        run_id_stem = run_id.split("/")[-1]

        # Setup output directory
        out_dir = REPO_ROOT / "param_decomp/experiments/tms/out/figures" / run_id_stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load models
        model = ComponentModel.from_pretrained(run_id)
        assert isinstance(model.target_model, TMSModel)

        # Get custom config and name for this run
        plot_config = run_info["config"]
        assert isinstance(plot_config, PlotConfig)
        run_name = run_info["name"]

        # Create plotter with custom config
        plotter = TMSPlotter(
            model=model.target_model, components=model.components, config=plot_config
        )

        # log analysis
        logger.section(f"TMS Analysis Summary - {run_name}")
        logger.values(plotter.get_analysis_summary())

        # Generate plots based on model architecture
        if model.target_model.config.n_hidden == 2:
            if model.target_model.config.n_hidden_layers == 0:
                # Model without hidden layers - use combined plot
                fig = plotter.plot_combined_diagram()
                filename = f"tms_combined_diagram_{run_name}.png"
                fig.savefig(
                    out_dir / filename,
                    bbox_inches="tight",
                    dpi=plotter.config.dpi,
                )
                logger.info(f"Saved combined diagram to {out_dir / filename}")
            else:
                # Model with hidden layers - use separate plots
                # Vector plot
                fig = plotter.plot_vectors()
                filename = f"tms_vectors_{run_name}.png"
                fig.savefig(out_dir / filename, bbox_inches="tight", dpi=plotter.config.dpi)
                logger.info(f"Saved vectors plot to {out_dir / filename}")

                # Full network plot
                fig = plotter.plot_full_network()
                filename = f"tms_full_network_{run_name}.png"
                fig.savefig(out_dir / filename, bbox_inches="tight", dpi=plotter.config.dpi)
                logger.info(f"Saved full network diagram to {out_dir / filename}")

        # Hidden layer heatmaps (if applicable)
        if model.target_model.config.n_hidden_layers > 0:
            fig = plotter.plot_hidden_layers()
            if fig:
                filename = f"tms_hidden_layers_{run_name}.png"
                fig.savefig(out_dir / filename, bbox_inches="tight", dpi=plotter.config.dpi)
                logger.info(f"Saved hidden layers plot to {out_dir / filename}")

        # Plot cosine similarity analysis
        fig = plotter.plot_cosine_similarity_analysis()
        filename = f"cosine_similarity_analysis_{run_name}.png"
        fig.savefig(out_dir / filename, bbox_inches="tight", dpi=plotter.config.dpi)
        logger.info(f"Saved cosine similarity analysis to {out_dir / filename}")


if __name__ == "__main__":
    main()
