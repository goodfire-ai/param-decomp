"""Compare intruder detection scores across decomposition methods.

Loads intruder scores from harvest DBs and produces:
1. Coherence vs firing density curves (binned means)
2. Score distribution violin plot by method group
3. Summary table (N, mean, median, p25, p75)

Usage:
    python -m spd.harvest.scripts.compare_intruder_scores config.json [--out-dir ./plots]

Config JSON format:
    {
        "models": {
            "CLT k8 (local)": ["clt-1dbdaa40", "h-20260323_163757"],
            "VPD (CI>0.1)":   ["s-55ea3f9b",   "h-20260319_121635"]
        },
        "groups": {
            "CLT (local)": ["CLT k8 (local)", "CLT k16 (local)"],
            "VPD":         ["VPD (CI>0.1)"]
        },
        "colors": {
            "CLT k8 (local)": "#1f77b4",
            "VPD (CI>0.1)":   "#2c2c2c"
        },
        "group_colors": {
            "CLT (local)": "#1f77b4",
            "VPD":         "#2c2c2c"
        }
    }
"""

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spd.settings import SPD_OUT_DIR

HARVEST_ROOT = SPD_OUT_DIR / "harvest"


def load_scores(decomp_id: str, subrun: str) -> tuple[np.ndarray, np.ndarray]:
    db_path = HARVEST_ROOT / decomp_id / subrun / "harvest.db"
    assert db_path.exists(), f"No harvest DB at {db_path}"
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        """
        SELECT c.firing_density, s.score
        FROM scores s
        JOIN components c ON s.component_key = c.component_key
        WHERE s.score_type = 'intruder'
        ORDER BY c.firing_density
        """
    ).fetchall()
    conn.close()
    assert rows, f"No intruder scores in {db_path}"
    densities = np.array([r[0] for r in rows])
    scores = np.array([r[1] for r in rows])
    return densities, scores


def plot_coherence_vs_density(
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    colors: dict[str, str],
    out_path: Path,
    n_bins: int = 25,
) -> None:
    all_log_d = [np.log10(np.clip(d, 1e-8, None)) for d, _ in data.values()]
    global_min = min(a.min() for a in all_log_d)
    global_max = max(a.max() for a in all_log_d)
    shared_edges = np.linspace(global_min, global_max, n_bins + 1)
    shared_centers = (shared_edges[:-1] + shared_edges[1:]) / 2

    fig, (ax_line, ax_hist, ax_norm) = plt.subplots(
        3, 1, figsize=(12, 10), height_ratios=[2, 1, 1], sharex=True
    )

    for label, (densities, scores) in data.items():
        log_d = np.log10(np.clip(densities, 1e-8, None))
        centers, means = [], []
        for i in range(n_bins):
            mask = (log_d >= shared_edges[i]) & (log_d < shared_edges[i + 1])
            if mask.sum() < 10:
                continue
            centers.append(shared_centers[i])
            means.append(scores[mask].mean())
        color = colors.get(label, "#333333")
        ax_line.plot(centers, means, label=label, color=color, linewidth=1.5)

    ax_line.axhline(0.2, color="gray", linestyle=":", alpha=0.5, label="Random (1/5)")
    ax_line.set_ylabel("Mean intruder score", fontsize=12)
    ax_line.set_title("Activation Coherence vs Firing Density", fontsize=14)
    ax_line.legend(fontsize=7, ncol=3, loc="lower left")
    ax_line.set_ylim(0, 1.05)
    ax_line.grid(alpha=0.2)

    for label, (densities, _) in data.items():
        log_d = np.log10(np.clip(densities, 1e-8, None))
        color = colors.get(label, "#333333")
        ax_hist.hist(
            log_d, bins=shared_edges, color=color, alpha=0.1, edgecolor=color, histtype="stepfilled"
        )
        ax_hist.hist(log_d, bins=shared_edges, color=color, histtype="step")
    ax_hist.set_ylabel("Component count", fontsize=12)
    ax_hist.grid(alpha=0.2)

    for label, (densities, _) in data.items():
        log_d = np.log10(np.clip(densities, 1e-8, None))
        color = colors.get(label, "#333333")
        ax_norm.hist(
            log_d,
            bins=shared_edges,
            color=color,
            alpha=0.1,
            edgecolor=color,
            histtype="stepfilled",
            density=True,
        )
        ax_norm.hist(log_d, bins=shared_edges, color=color, histtype="step", density=True)
    ax_norm.set_xlabel("log₁₀(firing density)", fontsize=12)
    ax_norm.set_ylabel("Density (normalised)", fontsize=12)
    ax_norm.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_violins(
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    groups: dict[str, list[str]],
    group_colors: dict[str, str],
    out_path: Path,
) -> None:
    score_levels = np.round(np.arange(0, 1.1, 0.1), 1)
    max_half_width = 0.4

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (group_label, members) in enumerate(groups.items()):
        present = [m for m in members if m in data]
        if not present:
            continue
        all_scores = np.concatenate([data[m][1] for m in present])
        color = group_colors.get(group_label, "#333333")
        n = len(all_scores)

        rounded = np.round(all_scores, 1)
        counts = np.array([np.sum(rounded == level) for level in score_levels])
        widths = counts / counts.max() * max_half_width

        for y, w in zip(score_levels, widths, strict=True):
            if w > 0:
                ax.barh(
                    y,
                    width=2 * w,
                    left=i - w,
                    height=0.08,
                    color=color,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.3,
                )

        ax.plot(
            i,
            all_scores.mean(),
            "D",
            color="white",
            markeredgecolor="black",
            markersize=5,
            zorder=5,
        )
        ax.text(i, -0.08, f"n={n}\nμ={all_scores.mean():.2f}", ha="center", va="top", fontsize=7)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups.keys(), fontsize=10)
    ax.set_ylabel("Intruder score", fontsize=12)
    ax.set_title("Score Distribution by Decomposition Method", fontsize=14)
    ax.axhline(0.2, color="gray", linestyle=":", alpha=0.5, label="Random (1/5)")
    ax.set_ylim(-0.15, 1.05)
    ax.grid(alpha=0.2, axis="y")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_bars(
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    groups: dict[str, list[str]],
    group_colors: dict[str, str],
    out_path: Path,
) -> None:
    labels: list[str] = []
    means: list[float] = []
    bar_colors: list[str] = []

    for group_label, members in groups.items():
        present = [m for m in members if m in data]
        if not present:
            continue
        all_scores = np.concatenate([data[m][1] for m in present])
        labels.append(group_label)
        means.append(all_scores.mean())
        bar_colors.append(group_colors.get(group_label, "#333333"))

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
    x = np.arange(len(labels))
    ax.bar(x, means, color=bar_colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(0.2, color="gray", linestyle=":", alpha=0.5, label="Random (1/5)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean intruder score", fontsize=12)
    ax.set_title("Intruder Score by Decomposition Method", fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2, axis="y")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def print_summary(data: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    print(f"\n{'Model':28s}  {'N':>6s}  {'Mean':>6s}  {'Median':>6s}  {'p25':>6s}  {'p75':>6s}")
    print("-" * 64)
    for label, (_, scores) in data.items():
        print(
            f"{label:28s}  {len(scores):6d}  {scores.mean():6.3f}  "
            f"{np.median(scores):6.2f}  {np.percentile(scores, 25):6.2f}  "
            f"{np.percentile(scores, 75):6.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare intruder scores across decompositions")
    parser.add_argument("config", type=Path, help="JSON config file (see module docstring)")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("."), help="Output directory for plots"
    )
    args = parser.parse_args()

    assert args.config.exists(), f"Config not found: {args.config}"
    with open(args.config) as f:
        cfg = json.load(f)

    models: dict[str, list[str]] = cfg["models"]
    colors: dict[str, str] = cfg.get("colors", {})
    groups: dict[str, list[str]] = cfg.get("groups", {})
    group_colors: dict[str, str] = cfg.get("group_colors", {})

    args.out_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, (decomp_id, subrun) in models.items():
        data[label] = load_scores(decomp_id, subrun)
        print(f"{label:28s}: {len(data[label][0]):6d} components")

    print_summary(data)

    plot_coherence_vs_density(data, colors, args.out_dir / "coherence_vs_density.png")

    if groups:
        plot_violins(data, groups, group_colors, args.out_dir / "score_distribution.png")
        plot_bars(data, groups, group_colors, args.out_dir / "score_bars.png")


if __name__ == "__main__":
    main()
