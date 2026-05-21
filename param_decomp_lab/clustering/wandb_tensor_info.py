"""Minimal WandB tensor logging utilities."""

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import wandb
import wandb.sdk.wandb_run
from torch import Tensor


def _array_info(arr: Tensor | np.ndarray) -> dict[str, Any]:
    """Get basic statistics about an array or tensor."""
    arr_np = arr.detach().cpu().numpy() if isinstance(arr, Tensor) else arr

    if arr_np.size == 0:
        return {
            "status": "empty",
            "size": 0,
            "shape": arr_np.shape,
            "dtype": str(arr.dtype) if isinstance(arr, Tensor) else str(arr_np.dtype),
            "has_nans": False,
            "nan_percent": None,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }

    has_nans = bool(np.isnan(arr_np).any())
    nan_count = int(np.isnan(arr_np).sum())
    nan_percent = 100.0 * nan_count / arr_np.size if arr_np.size > 0 else 0.0

    # Compute stats ignoring NaNs
    return {
        "status": "ok",
        "size": arr_np.size,
        "shape": arr_np.shape,
        "dtype": str(arr.dtype) if isinstance(arr, Tensor) else str(arr_np.dtype),
        "has_nans": has_nans,
        "nan_percent": nan_percent,
        "mean": float(np.nanmean(arr_np)),
        "median": float(np.nanmedian(arr_np)),
        "std": float(np.nanstd(arr_np)),
        "min": float(np.nanmin(arr_np)),
        "max": float(np.nanmax(arr_np)),
    }


def wandb_log_tensor(
    run: wandb.sdk.wandb_run.Run,
    data: Tensor | dict[str, Tensor],
    name: str,
    step: int,
    single: bool = False,
) -> None:
    """Log tensor(s) with stats to WandB as metrics and histograms.

    Args:
        run: Current WandB run (None if WandB disabled)
        data: Either a Tensor or dict[str, Tensor]
        name: Name for logging
        step: WandB step
        single: True if this tensor is only logged once (component activations)
    """
    try:
        if isinstance(data, dict):
            # Handle dict of tensors
            for key, tensor in data.items():
                full_name: str = f"{name}.{key}"
                _log_one(run, tensor, full_name, step, single=single)
        else:
            # Handle single tensor
            _log_one(run, data, name, step, single=single)
    except Exception as e:
        warnings.warn(f"Failed to log tensor {name}: {e}")  # noqa: B028
        raise e


def _create_histogram(
    info: dict[str, Any], tensor: Tensor, name: str, logy: bool = True
) -> plt.Figure:
    """Create matplotlib histogram with stats markers."""
    # sanity check
    if info["status"] != "ok" or info["size"] == 0:
        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"{info['status']}", ha="center", va="center")
        ax.set_title(f"{name} - {info['status']}")
        return fig

    # make basic hist
    values: np.ndarray = tensor.flatten().detach().cpu().numpy()
    if info["has_nans"]:
        values = values[~np.isnan(values)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)

    # Add stat lines
    mean_val: float = info["mean"] or float("nan")
    median_val: float = info["median"] or float("nan")
    std_val: float = info["std"] or float("nan")

    if info["mean"] is not None:
        ax.axvline(
            mean_val,
            color="red",
            linestyle="-",
            linewidth=2,
            label="$\\mu$",
        )
        ax.axvline(
            median_val,
            color="blue",
            linestyle="-",
            linewidth=2,
            label="$\\tilde{x}$",
        )
        if std_val:
            ax.axvline(
                mean_val + std_val,
                color="orange",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label="$\\mu+\\sigma$",
            )
            ax.axvline(
                mean_val - std_val,
                color="orange",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label="$\\mu-\\sigma$",
            )

    # Build informative title with tensor stats
    shape_str: str = str(tuple(info["shape"])) if "shape" in info else "unknown"
    dtype_str: str = str(info.get("dtype", "unknown")).replace("torch.", "")

    title_line1: str = f"{name}"
    title_line2: str = f"shape={shape_str}, dtype={dtype_str}"
    title_line3: str = (
        f"range=[{info['min']:.3g}, {info['max']:.3g}], "
        f"$\\mu$={mean_val:.3g}, $\\tilde{{x}}$={median_val:.3g}, $\\sigma$={std_val:.3g}"
    )

    # Combine into multi-line title
    full_title: str = f"{title_line1}\n{title_line2}\n{title_line3}"
    ax.set_title(full_title, fontsize=10)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if logy:
        ax.set_yscale("log")

    plt.tight_layout()
    return fig


def _log_one(
    run: wandb.sdk.wandb_run.Run,
    tensor_: Tensor,
    name: str,
    step: int,
    single: bool = False,
    # use_log_counts: bool = True,
) -> None:
    """Log a single tensor."""
    info: dict[str, Any] = _array_info(tensor_)

    if single:
        # For single-use logging, log a single histogram as a figure
        hist_fig: plt.Figure = _create_histogram(info=info, tensor=tensor_, name=name)
        histogram_key: str = f"single_hists/{name}"
        run.log({histogram_key: wandb.Image(hist_fig)}, step=step)
        plt.close(hist_fig)  # Close figure to free memory
    else:
        # Log numeric stats as metrics (viewable like loss) using dict comprehension
        stats_to_log: dict[str, float | wandb.Histogram] = {
            f"tensor_metrics/{name}/{key}": info[key]
            for key in ["mean", "std", "median", "min", "max"]
            if key in info and info[key] is not None
        }

        # For regular logging, use wandb.Histogram directly
        hist_key: str = f"tensor_histograms/{name}"
        stats_to_log[hist_key] = wandb.Histogram(tensor_.flatten().cpu().numpy())  # pyright: ignore[reportArgumentType]

        # Add nan_percent if present
        nan_percent: float | None = info["nan_percent"]
        if nan_percent is None:
            nan_percent = float("nan")
        if nan_percent > 0:
            stats_to_log[f"tensor_metrics/{name}/nan_percent"] = nan_percent

        if stats_to_log:
            run.log(stats_to_log, step=step)
