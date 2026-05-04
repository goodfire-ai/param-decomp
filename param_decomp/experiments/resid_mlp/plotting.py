from collections.abc import Callable
from typing import Literal

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor

from param_decomp.experiments.resid_mlp.configs import ResidMLPModelConfig


def plot_individual_feature_response(
    model_fn: Callable[[Tensor], Tensor],
    device: str,
    model_config: ResidMLPModelConfig,
    sweep: bool = False,
    subtract_inputs: bool = True,
    plot_type: Literal["line", "scatter"] = "scatter",
    ax: plt.Axes | None = None,
    cbar: bool = True,
):
    """Plot the response of the model to a single feature being active.

    If sweep is False then the amplitude of the active feature is 1.
    If sweep is True then the amplitude of the active feature is swept from -1 to 1. This is an
    arbitrary choice (choosing feature 0 to be the one where we test x=-1 etc) made for convenience.
    """

    assert plot_type in ["line", "scatter"], "Unknown plot_type"

    n_features = model_config.n_features
    batch_size = model_config.n_features
    batch = torch.zeros(batch_size, n_features, device=device)
    inputs = torch.ones(n_features) if not sweep else torch.linspace(-1, 1, n_features)
    batch[torch.arange(n_features), torch.arange(n_features)] = inputs.to(device)
    out = model_fn(batch)

    cmap_viridis = plt.get_cmap("viridis")
    fig, ax = plt.subplots(constrained_layout=True) if ax is None else (ax.figure, ax)
    sweep_str = "set to 1" if not sweep else "between -1 and 1"
    title = (
        f"Feature response with one active feature {sweep_str}\n"
        f"n_features={model_config.n_features}, "
        f"d_embed={model_config.d_embed}, "
        f"d_mlp={model_config.d_mlp}"
    )
    ax.set_title(title)
    if subtract_inputs:
        out = out - batch
    for f in range(n_features):
        x = torch.arange(n_features)
        y = out[f, :].detach().cpu()
        if plot_type == "line":
            ax.plot(x, y, color=cmap_viridis(f / n_features))
        elif plot_type == "scatter":
            s = torch.ones_like(x)
            alpha = torch.ones_like(x) * 0.33
            s[f] = 20
            alpha[f] = 1
            # Permute order to make zorder random
            order = torch.randperm(n_features)
            ax.scatter(
                x[order],
                y[order],
                color=cmap_viridis(f / n_features),
                marker=".",
                s=s[order],
                alpha=alpha[order].numpy(),  # pyright: ignore[reportArgumentType]
                # According to the announcement, alpha is allowed to be an iterable since v3.4.0,
                # but the docs & type annotations seem to be wrong. Here's the announcement:
                # https://matplotlib.org/stable/users/prev_whats_new/whats_new_3.4.0.html#transparency-alpha-can-be-set-as-an-array-in-collections
            )
    # Plot labels
    label_fn = F.relu if model_config.act_fn_name == "relu" else F.gelu
    inputs = batch[torch.arange(n_features), torch.arange(n_features)].detach().cpu()
    targets = label_fn(inputs) if subtract_inputs else inputs + label_fn(inputs)
    baseline = torch.zeros(n_features) if subtract_inputs else inputs
    if plot_type == "line":
        ax.plot(
            torch.arange(n_features),
            targets.cpu().detach(),
            color="red",
            label=r"Label ($x+\mathrm{ReLU}(x)$)",
        )
        ax.plot(
            torch.arange(n_features),
            baseline,
            color="red",
            linestyle=":",
            label="Baseline (Identity)",
        )
    else:
        ax.scatter(
            torch.arange(n_features),
            targets.cpu().detach(),
            color="red",
            label=r"Label ($x+\mathrm{ReLU}(x)$)",
            marker="x",
            s=5,
        )

    ax.legend()
    if cbar:
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_viridis, norm=plt.Normalize(0, n_features))
        sm.set_array([])
        bar = plt.colorbar(sm, ax=ax, orientation="vertical")
        bar.set_label("Active input feature index")
    ax.set_xlabel("Output index")
    ax.set_ylabel("Output values $x̂_i$")

    ax.set_xticks([0, n_features])
    ax.set_xticklabels(["0", str(n_features)])
    return fig


def plot_single_feature_response(
    model_fn: Callable[[Tensor], Tensor],
    device: str,
    model_config: ResidMLPModelConfig,
    subtract_inputs: bool = True,
    feature_idx: int = 15,
    plot_type: Literal["line", "scatter"] = "scatter",
    ax: plt.Axes | None = None,
):
    """Plot the response of the model to a single feature being active.

    If sweep is False then the amplitude of the active feature is 1.
    If sweep is True then the amplitude of the active feature is swept from -1 to 1. This is an
    arbitrary choice (choosing feature 0 to be the one where we test x=-1 etc) made for convenience.
    """
    assert plot_type in ["line", "scatter"], "Unknown plot_type"

    n_features = model_config.n_features
    batch_size = 1
    batch_idx = 0
    batch = torch.zeros(batch_size, n_features, device=device)
    batch[batch_idx, feature_idx] = 1
    out = model_fn(batch)

    cmap_viridis = plt.get_cmap("viridis")
    fig, ax = plt.subplots(constrained_layout=True) if ax is None else (ax.figure, ax)
    if subtract_inputs:
        out = out - batch
    x = torch.arange(n_features)
    y = out[batch_idx, :].detach().cpu()
    inputs = batch[batch_idx, :].detach().cpu()
    label_fn = F.relu if model_config.act_fn_name == "relu" else F.gelu
    targets = label_fn(inputs) if subtract_inputs else inputs + label_fn(inputs)
    if plot_type == "line":
        ax.plot(x, y, color=cmap_viridis(feature_idx / n_features), label="Target Model")
        ax.plot(
            torch.arange(n_features),
            targets.cpu().detach(),
            color="red",
            label=r"Label ($x+\mathrm{ReLU}(x)$)",
        )
    else:
        ax.scatter(
            x,
            y,
            color=cmap_viridis(feature_idx / n_features),
            label="Target Model",
            marker=".",
            s=20,
        )
        ax.scatter(
            torch.arange(n_features),
            targets.cpu().detach(),
            color="red",
            label=r"Label ($x+\mathrm{ReLU}(x)$)",
            marker="x",
            s=5,
        )

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    ax.set_xlabel("Output index")
    ax.set_ylabel(f"Output value $x̂_{{{feature_idx}}}$")
    ax.set_title(f"Output for a single input $x_{{{feature_idx}}}=1$")

    # Only need feature indices 0, feature_idx, n_features.
    ax.set_xticks([0, feature_idx, n_features])
    ax.set_xticklabels(["0", str(feature_idx), str(n_features)])

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def plot_single_relu_curve(
    model_fn: Callable[[Tensor], Tensor],
    device: str,
    model_config: ResidMLPModelConfig,
    subtract_inputs: bool = True,
    feature_idx: int = 15,
    ax: plt.Axes | None = None,
    label: bool = True,
):
    n_features = model_config.n_features
    batch_size = 1000
    x = torch.linspace(-1, 1, batch_size)
    batch = torch.zeros(batch_size, n_features, device=device)
    batch[:, feature_idx] = x
    out = model_fn(batch)
    cmap_viridis = plt.get_cmap("viridis")
    fig, ax = plt.subplots(constrained_layout=True) if ax is None else (ax.figure, ax)
    if subtract_inputs:
        out = out - batch

    y = out[:, feature_idx].detach().cpu()
    label_fn = F.relu if model_config.act_fn_name == "relu" else F.gelu
    targets = label_fn(x) if subtract_inputs else x + label_fn(x)
    ax.plot(
        x, y, color=cmap_viridis(feature_idx / n_features), label="Target Model" if label else None
    )
    ax.plot(
        x,
        targets.cpu().detach(),
        color="red",
        label=r"Label ($x+\mathrm{ReLU}(x)$)" if label else None,
        ls="--",
    )
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    ax.set_xlabel(f"Input value $x_{{{feature_idx}}}$")
    ax.set_ylabel(f"Output value $x̂_{{{feature_idx}}}$")
    ax.set_title(f"Input-output response for input feature {feature_idx}")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def plot_all_relu_curves(
    model_fn: Callable[[Tensor], Tensor],
    device: str,
    model_config: ResidMLPModelConfig,
    ax: plt.Axes,
    subtract_inputs: bool = True,
):
    n_features = model_config.n_features
    fig = ax.figure
    for feature_idx in range(n_features):
        plot_single_relu_curve(
            model_fn=model_fn,
            device=device,
            model_config=model_config,
            subtract_inputs=subtract_inputs,
            feature_idx=feature_idx,
            ax=ax,
            label=False,
        )
    ax.set_title(f"Input-output response for all {n_features} input features")
    ax.set_xlabel("Input values $x_i$")
    ax.set_ylabel("Output values $x̂_i$")
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig
