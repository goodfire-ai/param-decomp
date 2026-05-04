"""Interaction matrix analysis: I = G × H.

Computes the paper's interaction matrices:
- G_{c,c'} = (Σ_i |U_{i,c}| · |U_{i,c'}|) / (Σ_i |U_{i,c}|²)  [from weights]
- H_{c,c'} = (Σ_{b,t} |g·a|_c · |g·a|_{c'}) / (Σ_{b,t} |g·a|_c²)  [from harvest]
- I_{c,c'} = G_{c,c'} · H_{c,c'}  [element-wise product]

Generates heatmaps, scatter plots, row summaries, and statistical tests.

Usage:
    python param_decomp/scripts/geometric_interaction/interaction_analysis.py config.yaml
    python param_decomp/scripts/geometric_interaction/interaction_analysis.py \\
        --model_path="wandb:goodfire/spd/runs/s-55ea3f9b"
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import einops
import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from matplotlib.gridspec import GridSpec
from pydantic import Field
from scipy import stats
from torch import Tensor

from param_decomp.base_config import BaseConfig
from param_decomp.log import logger

SCRIPT_DIR = Path(__file__).parent

# Modules where nonlinearities matter (paper's focus)
NONLINEAR_MODULE_PATTERNS = ["down_proj", "o_proj"]


class InteractionAnalysisConfig(BaseConfig):
    model_path: str = Field(..., description="wandb: or local path to PD model")
    harvest_id: str | None = Field(None, description="CI harvest ID for alive filtering")
    alive_density_threshold: float = Field(default=0.001)
    module_filter: list[str] = Field(
        default=NONLINEAR_MODULE_PATTERNS,
        description="Only analyse modules matching these substrings",
    )
    output_dir: str | None = None


def _extract_run_id(path: str) -> str:
    return path.rstrip("/").split("/")[-1]


# ── G matrix (weight-only) ───────────────────────────────────────────────────


def compute_G_matrices(
    uv_by_module: dict[str, tuple[Float[Tensor, "C d_out"], Float[Tensor, "d_in C"]]],
) -> dict[str, Float[Tensor, "C C"]]:
    """Paper's G: geometric interaction strength from raw |U| vectors.

    G_{c,c'} = (Σ_i |U_{i,c}| · |U_{i,c'}|) / (Σ_i |U_{i,c}|²)
    """
    g_matrices: dict[str, Float[Tensor, "C C"]] = {}
    for module_name, (U, _V) in sorted(uv_by_module.items()):
        abs_U = U.abs()
        norms_sq = (abs_U**2).sum(dim=1)
        inner = einops.einsum(abs_U, abs_U, "C1 d, C2 d -> C1 C2")
        g = inner / norms_sq.unsqueeze(1)
        g = torch.nan_to_num(g, nan=0.0)
        g_matrices[module_name] = g
    return g_matrices


# ── H matrix (from harvest) ──────────────────────────────────────────────────


def compute_H_matrices(
    sum_ga_cross: dict[str, Tensor],
    alive_inds: dict[str, Tensor],
) -> dict[str, Float[Tensor, "C_alive C_alive"]]:
    """Compute H from harvest cross-products, restricted to alive components.

    H_{c,c'} = sum_ga_cross[c,c'] / sum_ga_cross[c,c]
    """
    h_matrices: dict[str, Float[Tensor, "C_alive C_alive"]] = {}
    for module_name, cross_full in sum_ga_cross.items():
        if module_name not in alive_inds:
            continue
        idx = alive_inds[module_name]
        cross = cross_full[idx][:, idx].double()
        diag = cross.diag()
        h_mat = torch.zeros_like(cross)
        nonzero = diag > 0
        h_mat[nonzero] = cross[nonzero] / diag[nonzero].unsqueeze(1)
        h_matrices[module_name] = h_mat.float()
    return h_matrices


# ── Alive component selection ─────────────────────────────────────────────────


def get_alive_inds(
    activation_density: dict[str, Float[Tensor, " C"]],
    threshold: float,
) -> dict[str, Tensor]:
    """Return sorted alive indices per module."""
    result: dict[str, Tensor] = {}
    for module_name, density in activation_density.items():
        sorted_inds = torch.argsort(density, descending=True)
        alive = sorted_inds[density[sorted_inds] > threshold]
        result[module_name] = alive
    return result


# ── Plotting ──────────────────────────────────────────────────────────────────


def _off_diag(matrix: Tensor) -> np.ndarray:
    n = matrix.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    return matrix[mask].cpu().numpy()


def plot_heatmaps(
    matrices: dict[str, Float[Tensor, "C C"]],
    output_path: Path,
    cmap: str,
    cbar_label: str,
    title_suffix: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    module_names = sorted(matrices.keys())
    n_modules = len(module_names)
    if n_modules == 0:
        return

    fig = plt.figure(figsize=(8, 7 * n_modules))
    gs: GridSpec = fig.add_gridspec(n_modules, 2, width_ratios=[17, 1], wspace=0.1)

    norm = plt.Normalize(vmin=vmin, vmax=vmax) if vmin is not None and vmax is not None else None

    images = []
    for i, name in enumerate(module_names):
        mat = matrices[name].cpu().numpy()
        ax = fig.add_subplot(gs[i, 0])
        im = ax.matshow(mat, aspect="auto", cmap=cmap, norm=norm)
        images.append(im)
        n = mat.shape[0]
        ax.set_title(f"{name} ({n} alive) — {title_suffix}")
        ax.set_xlabel("Component (sorted by density)")
        ax.set_ylabel("Component (sorted by density)")

    if images:
        cbar_ax = fig.add_subplot(gs[:, 1])
        fig.colorbar(images[0], cax=cbar_ax, label=cbar_label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  {output_path}")


def plot_scatter_G_vs_H(
    g_matrices: dict[str, Tensor],
    h_matrices: dict[str, Tensor],
    output_dir: Path,
) -> None:
    """Scatter plot of off-diagonal G vs H per module."""
    scatter_dir = output_dir / "scatter"
    scatter_dir.mkdir(parents=True, exist_ok=True)

    for name in sorted(g_matrices):
        if name not in h_matrices:
            continue
        g_flat = _off_diag(g_matrices[name])
        h_flat = _off_diag(h_matrices[name])

        sr = stats.spearmanr(g_flat, h_flat)
        rho = float(sr.statistic)  # pyright: ignore[reportAttributeAccessIssue]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(g_flat, h_flat, alpha=0.15, s=1.5, linewidths=0)
        ax.set_xlabel("G (geometric)")
        ax.set_ylabel("H (coactivation)")
        ax.set_title(f"{name} — G vs H (Spearman ρ = {rho:.4f})")
        ax.grid(alpha=0.3)

        safe = name.replace(".", "_")
        fig.savefig(scatter_dir / f"G_vs_H_{safe}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"  G vs H scatter → {scatter_dir}")


def plot_I_distribution(
    i_matrices: dict[str, Tensor],
    output_dir: Path,
) -> None:
    """Histogram of off-diagonal I values per module."""
    hist_dir = output_dir / "histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)

    for name in sorted(i_matrices):
        off = _off_diag(i_matrices[name])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(off, bins=100, edgecolor="none", alpha=0.7)
        ax.set_xlabel("I (interaction strength)")
        ax.set_ylabel("Count")
        ax.set_yscale("log")
        ax.set_title(f"{name} — off-diagonal I distribution")
        ax.axvline(1.0, color="red", linestyle="--", alpha=0.5, label="I = 1")
        ax.legend()

        safe = name.replace(".", "_")
        fig.savefig(hist_dir / f"I_dist_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"  I distribution → {hist_dir}")


def plot_row_summaries(
    i_matrices: dict[str, Tensor],
    output_dir: Path,
) -> None:
    """For each component, sum of off-diagonal I in its row = total interaction burden."""
    summary_dir = output_dir / "row_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    for name in sorted(i_matrices):
        I_mat = i_matrices[name]
        n = I_mat.shape[0]
        # Sum off-diagonal entries per row
        row_sums = I_mat.sum(dim=1) - I_mat.diag()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(n), row_sums.cpu().numpy(), width=1.0, edgecolor="none", alpha=0.7)
        ax.set_xlabel("Component index (sorted by density)")
        ax.set_ylabel("Σ I_{c,c'} (off-diagonal)")
        ax.set_title(f"{name} — total interaction per component")
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.5)

        safe = name.replace(".", "_")
        fig.savefig(summary_dir / f"row_sum_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"  Row summaries → {summary_dir}")


def plot_per_layer_summary(
    i_matrices: dict[str, Tensor],
    g_matrices: dict[str, Tensor],
    h_matrices: dict[str, Tensor],
    output_dir: Path,
) -> None:
    """Per-layer summary of mean off-diagonal values for G, H, I."""
    layer_data: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"G": [], "H": [], "I": []})

    for name in sorted(i_matrices):
        parts = name.split(".")
        layer = next((int(p) for p in parts if p.isdigit()), None)
        if layer is None:
            continue
        layer_data[layer]["G"].append(float(_off_diag(g_matrices[name]).mean()))
        layer_data[layer]["H"].append(float(_off_diag(h_matrices[name]).mean()))
        layer_data[layer]["I"].append(float(_off_diag(i_matrices[name]).mean()))

    if not layer_data:
        return

    layers = sorted(layer_data.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["G", "H", "I"], strict=True):
        for layer in layers:
            vals = layer_data[layer][metric]
            ax.scatter([layer] * len(vals), vals, alpha=0.6, s=40)
        means = [np.mean(layer_data[ly][metric]) for ly in layers]
        ax.plot(layers, means, "o-", color="red", markersize=8, linewidth=2)
        ax.set_xlabel("Layer")
        ax.set_ylabel(f"Mean off-diagonal {metric}")
        ax.set_title(metric)
        ax.set_xticks(layers)
        ax.grid(alpha=0.3)

    fig.suptitle("Per-layer interaction summary (nonlinear modules only)", fontsize=13)
    fig.tight_layout()
    path = output_dir / "per_layer_summary.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Per-layer summary → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main(config_path: Path | str | None = None, **overrides: Any) -> None:
    if config_path is not None:
        config = InteractionAnalysisConfig.from_file(config_path)
    else:
        config = InteractionAnalysisConfig(**overrides)

    run_id = _extract_run_id(config.model_path)

    if config.output_dir is not None:
        output_dir = Path(config.output_dir)
    else:
        output_dir = SCRIPT_DIR / "out" / run_id / "interaction_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load U/V for G ────────────────────────────────────────────────────
    from param_decomp.scripts.geometric_interaction.geometric_interaction import load_component_uv

    logger.info(f"Loading component weights from {config.model_path}")
    uv_by_module = load_component_uv(config.model_path)

    # ── Load interaction harvest for H ────────────────────────────────────
    harvest_path = SCRIPT_DIR / "out" / run_id / "interaction_harvest" / "data.pt"
    assert harvest_path.exists(), (
        f"No interaction harvest at {harvest_path}. Run harvest_interaction.py first."
    )
    logger.info(f"Loading interaction harvest from {harvest_path}")
    harvest_data = torch.load(harvest_path, map_location="cpu", weights_only=False)
    sum_ga_cross = harvest_data["sum_ga_cross"]

    # ── Load CI harvest for alive filtering ───────────────────────────────
    from param_decomp.scripts.geometric_interaction.geometric_interaction import (
        compute_per_module_coactivation,
        load_harvest_coactivation,
    )

    logger.info(f"Loading CI harvest for alive filtering (run {run_id})")
    component_keys, count_i, count_ij, count_total = load_harvest_coactivation(
        run_id, config.harvest_id
    )
    activation_density, _, _ = compute_per_module_coactivation(
        component_keys, count_i, count_ij, count_total
    )

    # ── Filter to modules of interest ─────────────────────────────────────
    def matches_filter(name: str) -> bool:
        return any(pat in name for pat in config.module_filter)

    filtered_modules = [m for m in sorted(uv_by_module) if matches_filter(m)]
    logger.info(f"Filtered to {len(filtered_modules)} modules: {filtered_modules}")

    # ── Compute alive indices ─────────────────────────────────────────────
    alive_inds = get_alive_inds(activation_density, config.alive_density_threshold)
    for name in filtered_modules:
        n_alive = len(alive_inds.get(name, []))
        n_total = uv_by_module[name][0].shape[0]
        logger.info(f"  {name}: {n_alive}/{n_total} alive")

    # ── Compute G (all modules, then filter) ──────────────────────────────
    logger.info("Computing G matrices...")
    all_G = compute_G_matrices(uv_by_module)
    # Crop to alive
    G_alive: dict[str, Tensor] = {}
    for name in filtered_modules:
        idx = alive_inds[name]
        G_alive[name] = all_G[name][idx][:, idx]

    # ── Compute H (from harvest, alive only) ──────────────────────────────
    logger.info("Computing H matrices...")
    H_alive = compute_H_matrices(
        {k: v for k, v in sum_ga_cross.items() if k in filtered_modules},
        alive_inds,
    )

    # ── Compute I = G × H ─────────────────────────────────────────────────
    logger.info("Computing I = G × H...")
    I_alive: dict[str, Tensor] = {}
    for name in filtered_modules:
        if name in G_alive and name in H_alive:
            I_alive[name] = G_alive[name] * H_alive[name]

    # ── Statistics ─────────────────────────────────────────────────────────
    logger.info("Computing statistics...")
    stats_results: dict[str, dict[str, Any]] = {}
    for name in sorted(I_alive):
        g_off = _off_diag(G_alive[name])
        h_off = _off_diag(H_alive[name])
        i_off = _off_diag(I_alive[name])

        sr = stats.spearmanr(g_off, h_off)
        rho_gh = float(sr.statistic)  # pyright: ignore[reportAttributeAccessIssue]

        # Row sums of off-diagonal I
        I_mat = I_alive[name]
        row_sums = (I_mat.sum(dim=1) - I_mat.diag()).cpu().numpy()

        stats_results[name] = {
            "G_off_diag_mean": float(g_off.mean()),
            "H_off_diag_mean": float(h_off.mean()),
            "I_off_diag_mean": float(i_off.mean()),
            "I_off_diag_median": float(np.median(i_off)),
            "I_off_diag_max": float(i_off.max()),
            "I_off_diag_gt_0.1_frac": float((i_off > 0.1).mean()),
            "I_off_diag_gt_0.5_frac": float((i_off > 0.5).mean()),
            "I_off_diag_gt_1.0_frac": float((i_off > 1.0).mean()),
            "spearman_G_H": rho_gh,
            "row_sum_mean": float(row_sums.mean()),
            "row_sum_max": float(row_sums.max()),
            "row_sum_median": float(np.median(row_sums)),
            "n_alive": int(I_mat.shape[0]),
        }
        logger.info(
            f"  {name}: I_mean={i_off.mean():.4f}, I_max={i_off.max():.2f}, "
            f"I>0.1={100 * (i_off > 0.1).mean():.1f}%, rho(G,H)={rho_gh:.4f}"
        )

    # ── Plots ─────────────────────────────────────────────────────────────
    logger.info("Generating plots...")

    plot_heatmaps(
        G_alive, output_dir / "heatmaps" / "G.png", "Reds", "G", "Geometric", vmin=0, vmax=1
    )
    plot_heatmaps(
        H_alive, output_dir / "heatmaps" / "H.png", "Blues", "H", "Coactivation", vmin=0, vmax=1
    )
    plot_heatmaps(
        I_alive,
        output_dir / "heatmaps" / "I.png",
        "Purples",
        "I = G×H",
        "Interaction",
        vmin=0,
        vmax=1,
    )

    plot_scatter_G_vs_H(G_alive, H_alive, output_dir)
    plot_I_distribution(I_alive, output_dir)
    plot_row_summaries(I_alive, output_dir)
    plot_per_layer_summary(I_alive, G_alive, H_alive, output_dir)

    # ── Save ──────────────────────────────────────────────────────────────
    torch.save(
        {
            "G_matrices": G_alive,
            "H_matrices": H_alive,
            "I_matrices": I_alive,
            "stats": stats_results,
            "run_id": run_id,
            "config": config.model_dump(),
        },
        output_dir / "data.pt",
    )

    with open(output_dir / "results.json", "w") as f:
        json.dump(stats_results, f, indent=2)

    logger.info(f"Saved → {output_dir}")

    # ── Summary table ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 95)
    logger.info(
        f"{'Module':<30} {'n':>5} {'I_mean':>8} {'I_max':>8} {'I>0.1%':>8} "
        f"{'I>1.0%':>8} {'ρ(G,H)':>8} {'RowΣ':>8}"
    )
    logger.info("-" * 95)
    for name in sorted(stats_results):
        s = stats_results[name]
        logger.info(
            f"  {name:<28} {s['n_alive']:>5} {s['I_off_diag_mean']:>8.4f} "
            f"{s['I_off_diag_max']:>8.2f} {100 * s['I_off_diag_gt_0.1_frac']:>7.1f}% "
            f"{100 * s['I_off_diag_gt_1.0_frac']:>7.1f}% {s['spearman_G_H']:>8.4f} "
            f"{s['row_sum_mean']:>8.2f}"
        )
    logger.info("=" * 95)


if __name__ == "__main__":
    fire.Fire(main)
