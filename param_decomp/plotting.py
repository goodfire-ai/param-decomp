import fnmatch
import io
from collections.abc import Callable

import numpy as np
import torch
from jaxtyping import Float
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor

from param_decomp.configs import SamplingType
from param_decomp.models.component_model import CIOutputs, ComponentModel
from param_decomp.models.components import Components
from param_decomp.utils.general_utils import get_obj_device
from param_decomp.utils.target_ci_solutions import permute_to_dense, permute_to_identity


def _render_figure(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    buf.close()
    return img


def _plot_causal_importances_figure(
    ci_vals: dict[str, Float[Tensor, "... C"]],
    title_prefix: str,
    colormap: str,
    input_magnitude: float,
    has_pos_dim: bool,
    title_formatter: Callable[[str], str] | None = None,
) -> Image.Image:
    """Plot causal importances for components stacked vertically.

    Args:
        ci_vals: Dictionary of causal importances (or causal importances upper leaky relu) to plot
        title_prefix: String to prepend to the title (e.g., "causal importances" or
            "causal importances upper leaky relu")
        colormap: Matplotlib colormap name
        input_magnitude: Input magnitude value for the title
        has_pos_dim: Whether the masks have a position dimension
        title_formatter: Optional callable to format subplot titles. Takes mask_name as input.

    Returns:
        The matplotlib figure
    """
    figsize = (5, 5 * len(ci_vals))
    fig, axs = plt.subplots(
        len(ci_vals),
        1,
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
        dpi=300,
    )
    axs = np.array(axs)

    images = []
    for j, (mask_name, mask) in enumerate(ci_vals.items()):
        # mask has shape (batch, C) or (batch, pos, C)
        mask_data = mask.detach().cpu().numpy()
        if has_pos_dim:
            assert mask_data.ndim == 3
            mask_data = mask_data[:, 0, :]
        ax = axs[j, 0]
        im = ax.matshow(mask_data, aspect="auto", cmap=colormap)
        images.append(im)

        # Move x-axis ticks to bottom
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position("bottom")
        ax.set_xlabel("Subcomponent index")
        ax.set_ylabel("Input feature index")

        # Apply custom title formatting if provided
        title = title_formatter(mask_name) if title_formatter is not None else mask_name
        ax.set_title(title)

    # Add unified colorbar
    norm = plt.Normalize(
        vmin=min(mask.min().item() for mask in ci_vals.values()),
        vmax=max(mask.max().item() for mask in ci_vals.values()),
    )
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs.ravel().tolist())

    # Capitalize first letter of title prefix for the figure title
    fig.suptitle(f"{title_prefix.capitalize()} - Input magnitude: {input_magnitude}")

    img = _render_figure(fig)
    plt.close(fig)

    return img


def plot_mean_component_cis_both_scales(
    mean_component_cis: dict[str, Float[Tensor, " C"]],
) -> tuple[Image.Image, Image.Image]:
    """
    Efficiently plot mean CI per component with both linear and log scales.

    This function optimizes the plotting by pre-processing data once and
    reusing it for both plots.

    Args:
        mean_component_cis: Dictionary mapping module names to mean CI tensors

    Returns:
        Tuple of (linear_scale_image, log_scale_image)
    """
    n_modules = len(mean_component_cis)
    max_rows = 6

    # Calculate grid dimensions once
    n_cols = (n_modules + max_rows - 1) // max_rows  # Ceiling division
    n_rows = min(n_modules, max_rows)

    # Adjust figure size based on grid dimensions
    fig_width = 8 * n_cols
    fig_height = 3 * n_rows

    # Pre-process data once
    processed_data = []
    for module_name, mean_component_ci in mean_component_cis.items():
        sorted_components = torch.sort(mean_component_ci, descending=True)[0]
        processed_data.append((module_name, sorted_components.detach().cpu().numpy()))

    # Create both figures
    images = []
    for log_y in [False, True]:
        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_width, fig_height),
            dpi=200,
            squeeze=False,
        )
        axs = np.array(axs)

        # Ensure axs is always 2D array for consistent indexing
        if axs.ndim == 1:
            axs = axs.reshape(n_rows, n_cols)

        # Hide unused subplots
        for i in range(n_modules, n_rows * n_cols):
            row = i % n_rows
            col = i // n_rows
            axs[row, col].set_visible(False)

        for i, (module_name, sorted_components_np) in enumerate(processed_data):
            # Calculate position in grid (fill column by column)
            row = i % n_rows
            col = i // n_rows
            ax = axs[row, col]

            if log_y:
                ax.set_yscale("log")

            ax.scatter(
                range(len(sorted_components_np)),
                sorted_components_np,
                marker="x",
                s=10,
            )

            # Only add x-label to bottom row of each column
            if row == n_rows - 1 or i == n_modules - 1:
                ax.set_xlabel("Component")
            ax.set_ylabel("mean CI")
            ax.set_title(module_name, fontsize=10)

        fig.tight_layout()
        img = _render_figure(fig)
        plt.close(fig)
        images.append(img)

    return images[0], images[1]


def get_single_feature_causal_importances(
    model: ComponentModel,
    batch_shape: tuple[int, ...],
    input_magnitude: float,
    sampling: SamplingType,
) -> CIOutputs:
    """Compute causal importance arrays for single active features.

    Args:
        model: The ComponentModel
        batch_shape: Shape of the batch
        input_magnitude: Magnitude of input features

    Returns:
        Tuple of (ci_raw, ci_upper_leaky_raw) dictionaries of causal importance arrays (2D tensors)
    """
    device = get_obj_device(model)
    # Create a batch of inputs with single active features
    has_pos_dim = len(batch_shape) == 3
    n_features = batch_shape[-1]
    batch = torch.eye(n_features, device=device) * input_magnitude
    if has_pos_dim:
        # NOTE: For now, we only use the first pos dim
        batch = batch.unsqueeze(1)

    pre_weight_acts = model(batch, cache_type="input").cache

    return model.calc_causal_importances(
        pre_weight_acts=pre_weight_acts,
        detach_inputs=False,
        sampling=sampling,
    )


def plot_causal_importance_vals(
    model: ComponentModel,
    batch_shape: tuple[int, ...],
    input_magnitude: float,
    sampling: SamplingType,
    identity_patterns: list[str] | None = None,
    dense_patterns: list[str] | None = None,
    plot_raw_cis: bool = True,
    title_formatter: Callable[[str], str] | None = None,
) -> tuple[dict[str, Image.Image], dict[str, Float[Tensor, " C"]]]:
    """Plot the values of the causal importances for a batch of inputs with single active features.

    Args:
        model: The ComponentModel
        batch_shape: Shape of the batch
        input_magnitude: Magnitude of input features
        sampling: Sampling method to use
        plot_raw_cis: Whether to plot the raw causal importances (blue plots)
        title_formatter: Optional callable to format subplot titles. Takes mask_name as input.
        identity_patterns: List of patterns to match for identity permutation
        dense_patterns: List of patterns to match for dense permutation
        plot_raw_cis: Whether to plot the raw causal importances (blue plots)
        title_formatter: Optional callable to format subplot titles. Takes mask_name as input.

    Returns:
        Tuple of:
            - Dictionary of figures with keys 'causal_importances' (if plot_raw_cis=True) and 'causal_importances_upper_leaky'
            - Dictionary of permutation indices for causal importances
    """
    # Get the causal importance arrays
    ci_output = get_single_feature_causal_importances(
        model=model,
        batch_shape=batch_shape,
        input_magnitude=input_magnitude,
        sampling=sampling,
    )

    ci: dict[str, Float[Tensor, "... C"]] = {}
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]] = {}
    all_perm_indices: dict[str, Float[Tensor, " C"]] = {}
    for k in ci_output.lower_leaky:
        # Determine permutation strategy based on patterns
        if identity_patterns and any(fnmatch.fnmatch(k, pattern) for pattern in identity_patterns):
            ci[k], _ = permute_to_identity(ci_vals=ci_output.lower_leaky[k])
            ci_upper_leaky[k], all_perm_indices[k] = permute_to_identity(
                ci_vals=ci_output.upper_leaky[k]
            )
        elif dense_patterns and any(fnmatch.fnmatch(k, pattern) for pattern in dense_patterns):
            ci[k], _ = permute_to_dense(ci_vals=ci_output.lower_leaky[k])
            ci_upper_leaky[k], all_perm_indices[k] = permute_to_dense(
                ci_vals=ci_output.upper_leaky[k]
            )
        else:
            # Default: identity permutation
            ci[k], _ = permute_to_identity(ci_vals=ci_output.lower_leaky[k])
            ci_upper_leaky[k], all_perm_indices[k] = permute_to_identity(
                ci_vals=ci_output.upper_leaky[k]
            )

    # Create figures dictionary
    figures: dict[str, Image.Image] = {}

    # TODO: Need to handle this differently for e.g. convolutional tasks
    has_pos_dim = len(batch_shape) == 3
    if plot_raw_cis:
        ci_fig = _plot_causal_importances_figure(
            ci_vals=ci,
            title_prefix="importance values lower leaky relu",
            colormap="Blues",
            input_magnitude=input_magnitude,
            has_pos_dim=has_pos_dim,
            title_formatter=title_formatter,
        )
        figures["causal_importances"] = ci_fig

    ci_upper_leaky_fig = _plot_causal_importances_figure(
        ci_vals=ci_upper_leaky,
        title_prefix="importance values",
        colormap="Reds",
        input_magnitude=input_magnitude,
        has_pos_dim=has_pos_dim,
        title_formatter=title_formatter,
    )
    figures["causal_importances_upper_leaky"] = ci_upper_leaky_fig

    return figures, all_perm_indices


def plot_UV_matrices(
    components: dict[str, Components],
    all_perm_indices: dict[str, Float[Tensor, " C"]] | None = None,
) -> Image.Image:
    """Plot V and U matrices for each instance, grouped by layer."""
    n_layers = len(components)

    # Create figure for plotting - 2 rows per layer (V and U)
    fig, axs = plt.subplots(
        n_layers,
        2,  # U, V
        figsize=(5 * 2, 5 * n_layers),
        constrained_layout=True,
        squeeze=False,
    )
    axs = np.array(axs)

    images = []

    # Plot V and U matrices for each layer
    for j, (name, component) in enumerate(sorted(components.items())):
        # Plot V matrix
        V = component.V if all_perm_indices is None else component.V[:, all_perm_indices[name]]
        V_np = V.detach().cpu().numpy()
        im = axs[j, 0].matshow(V_np, aspect="auto", cmap="coolwarm")
        axs[j, 0].set_ylabel("d_in index")
        axs[j, 0].set_xlabel("Component index")
        axs[j, 0].set_title(f"{name} (V matrix)")
        images.append(im)

        # Plot U matrix
        U = component.U if all_perm_indices is None else component.U[all_perm_indices[name], :]
        U_np = U.detach().cpu().numpy()
        im = axs[j, 1].matshow(U_np, aspect="auto", cmap="coolwarm")
        axs[j, 1].set_ylabel("Component index")
        axs[j, 1].set_xlabel("d_out index")
        axs[j, 1].set_title(f"{name} (U matrix)")
        images.append(im)

    # Add unified colorbar
    all_matrices = [c.V for c in components.values()] + [c.U for c in components.values()]
    norm = plt.Normalize(
        vmin=min(m.min().item() for m in all_matrices),
        vmax=max(m.max().item() for m in all_matrices),
    )
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs.ravel().tolist())

    fig_img = _render_figure(fig)
    plt.close(fig)

    return fig_img


def plot_component_activation_density(
    component_activation_density: dict[str, Float[Tensor, " C"]],
    bins: int = 100,
) -> Image.Image:
    """Plot the activation density of each component as a histogram in a grid layout."""

    n_modules = len(component_activation_density)
    max_rows = 6

    # Calculate grid dimensions
    n_cols = (n_modules + max_rows - 1) // max_rows  # Ceiling division
    n_rows = min(n_modules, max_rows)

    # Adjust figure size based on grid dimensions
    fig_width = 5 * n_cols
    fig_height = 5 * n_rows

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    axs = np.array(axs)

    # Ensure axs is always 2D array for consistent indexing
    if axs.ndim == 1:
        axs = axs.reshape(n_rows, n_cols)

    # Hide unused subplots
    for i in range(n_modules, n_rows * n_cols):
        row = i % n_rows
        col = i // n_rows
        axs[row, col].set_visible(False)

    # Iterate through modules and plot each histogram on its corresponding axis
    for i, (module_name, density) in enumerate(component_activation_density.items()):
        # Calculate position in grid (fill column by column)
        row = i % n_rows
        col = i // n_rows
        ax = axs[row, col]

        data = density.detach().cpu().numpy()
        ax.hist(data, bins=bins)
        ax.set_yscale("log")  # Beware, memory leak unless gc.collect() is called after eval loop
        ax.set_title(module_name)  # Add module name as title to each subplot

        # Only add x-label to bottom row of each column
        if row == n_rows - 1 or i == n_modules - 1:
            ax.set_xlabel("Activation density")
        ax.set_ylabel("Frequency")

    fig.tight_layout()

    fig_img = _render_figure(fig)
    plt.close(fig)

    return fig_img


def plot_ci_values_histograms(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    bins: int = 100,
) -> Image.Image:
    """Plot histograms of mask values for all layers in a grid layout.

    Args:
        causal_importances: Dictionary of causal importances for each component.
        bins: Number of bins for the histogram.

    Returns:
        Single figure with subplots for each layer.
    """
    assert len(causal_importances) > 0, "No causal importances to plot"
    n_layers = len(causal_importances)
    max_rows = 6

    # Calculate grid dimensions
    n_cols = (n_layers + max_rows - 1) // max_rows  # Ceiling division
    n_rows = min(n_layers, max_rows)

    # Adjust figure size based on grid dimensions
    fig_width = 6 * n_cols
    fig_height = 5 * n_rows

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    axs = np.array(axs)

    # Ensure axs is always 2D array for consistent indexing
    if axs.ndim == 1:
        axs = axs.reshape(n_rows, n_cols)

    # Hide unused subplots
    for i in range(n_layers, n_rows * n_cols):
        row = i % n_rows
        col = i // n_rows
        axs[row, col].set_visible(False)

    for i, (layer_name_raw, layer_ci) in enumerate(causal_importances.items()):
        layer_name = layer_name_raw.replace(".", "_")

        # Calculate position in grid (fill column by column)
        row = i % n_rows
        col = i // n_rows
        ax = axs[row, col]

        data = layer_ci.flatten().cpu().numpy()
        ax.hist(data, bins=bins)
        ax.set_yscale("log")  # Beware, memory leak unless gc.collect() is called after eval loop
        ax.set_title(f"Causal importances for {layer_name}")

        # Only add x-label to bottom row of each column
        if row == n_rows - 1 or i == n_layers - 1:
            ax.set_xlabel("Causal importance value")
        ax.set_ylabel("Frequency")

    fig.tight_layout()

    fig_img = _render_figure(fig)
    plt.close(fig)

    return fig_img
