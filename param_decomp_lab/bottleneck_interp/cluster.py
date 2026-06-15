"""Hierarchical clustering of bottleneck code dimensions by co-activation.

Simple alternative to the Ising-topology pipeline: correlate the D code dims across
harvested positions, then agglomeratively cluster (1 - correlation distance). Outputs a
dendrogram, a correlation heatmap reordered by the clustering, and flat cluster
assignments at a few cut thresholds.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from param_decomp_lab.bottleneck_interp.geometry import load_codes


def code_correlation(codes: torch.Tensor) -> np.ndarray:
    """Pearson correlation between code dims across positions, as a (D, D) matrix."""
    x = codes.float()
    x = x - x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp(min=1e-8)
    x = x / std
    corr = (x.T @ x) / x.shape[0]
    return corr.clamp(-1.0, 1.0).numpy()


def run(code_dir: Path, out_dir: Path, n_samples: int, cut_thresholds: list[float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    codes = load_codes(code_dir, n_samples)[:n_samples]
    print(f"loaded codes {tuple(codes.shape)}")

    corr = code_correlation(codes)
    dim = corr.shape[0]

    # distance = 1 - corr, condensed for linkage (average linkage on dim similarity)
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    dist = 0.5 * (dist + dist.T)
    Z = linkage(squareform(dist, checks=False), method="average")

    order = leaves_list(Z)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_title("code-dim correlation (hierarchical-clustering order)")
    plt.colorbar(im, ax=ax, label="Pearson r")
    fig.savefig(out_dir / "corr_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4))
    dendrogram(Z, no_labels=True, color_threshold=cut_thresholds[len(cut_thresholds) // 2], ax=ax)
    ax.set_title("code-dim dendrogram (1 - correlation, average linkage)")
    ax.set_ylabel("merge distance")
    fig.savefig(out_dir / "dendrogram.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    summary: dict[str, object] = {"code_dir": str(code_dir), "dim": dim, "cuts": {}}
    for t in cut_thresholds:
        labels = fcluster(Z, t=t, criterion="distance")
        sizes = np.bincount(labels)[1:]
        n_clusters = int(labels.max())
        big = sorted((int(s) for s in sizes), reverse=True)[:10]
        summary["cuts"][f"{t}"] = {  # pyright: ignore[reportIndexIssue]
            "n_clusters": n_clusters,
            "n_singletons": int((sizes == 1).sum()),
            "largest_sizes": big,
            "labels": labels.tolist(),
        }
        print(
            f"cut {t}: {n_clusters} clusters, {int((sizes == 1).sum())} singletons, top sizes {big}"
        )

    (out_dir / "clusters.json").write_text(json.dumps(summary, indent=2))
    np.save(out_dir / "linkage.npy", Z)
    np.save(out_dir / "corr.npy", corr)
    print(f"wrote {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n_samples", type=int, default=200_000)
    ap.add_argument("--cuts", type=float, nargs="+", default=[0.7, 0.85, 0.95])
    args = ap.parse_args()
    run(args.codes, args.out, args.n_samples, args.cuts)


if __name__ == "__main__":
    main()
