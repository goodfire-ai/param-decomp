"""Statistical analysis of the GIS–coactivation relationship.

Loads the data.pt output from geometric_interaction.py and runs:
1. Spearman rank correlation per module
2. Pearson correlation on log-transformed GIS
3. Binned coactivation by GIS quantile (decile plot)
4. GIS distribution conditioned on coactivation threshold
5. Permutation test (null model)
6. Mutual information (discretised)
7. Logistic regression: can GIS predict top-k% coactivation? (AUC)
8. Per-layer comparison of the GIS–coactivation relationship

Usage:
    python spd/scripts/geometric_interaction/statistical_analysis.py <data.pt path>
    python spd/scripts/geometric_interaction/statistical_analysis.py  # uses latest out/
"""

import json
from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression  # pyright: ignore[reportMissingImports]
from sklearn.metrics import roc_auc_score  # pyright: ignore[reportMissingImports]

from spd.log import logger


def _unpack_scipy(res: Any) -> tuple[float, float]:
    """Extract (statistic, pvalue) from a scipy stats result, handling stubs."""
    return (float(res[0]), float(res[1]))


def load_data(data_path: Path) -> dict[str, Any]:
    return torch.load(data_path, map_location="cpu", weights_only=False)


def _off_diag_flat(matrix: torch.Tensor) -> np.ndarray:
    """Extract off-diagonal elements as a flat numpy array."""
    n = matrix.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    return matrix[mask].numpy()


def _parse_layer(module_name: str) -> int | None:
    """Extract layer number from module name like 'h.2.attn.o_proj' -> 2."""
    parts = module_name.split(".")
    for p in parts:
        if p.isdigit():
            return int(p)
    return None


# ── 1. Spearman rank correlation ──────────────────────────────────────────────


def spearman_per_module(
    gis: dict[str, torch.Tensor],
    coact: dict[str, torch.Tensor],
) -> dict[str, tuple[float, float]]:
    results: dict[str, tuple[float, float]] = {}
    for name in sorted(gis):
        g = _off_diag_flat(gis[name])
        c = _off_diag_flat(coact[name])
        results[name] = _unpack_scipy(stats.spearmanr(g, c))
    return results


# ── 2. Pearson on log-transformed GIS ─────────────────────────────────────────


def pearson_log_gis_per_module(
    gis: dict[str, torch.Tensor],
    coact: dict[str, torch.Tensor],
) -> dict[str, tuple[float, float]]:
    results: dict[str, tuple[float, float]] = {}
    for name in sorted(gis):
        g = _off_diag_flat(gis[name])
        c = _off_diag_flat(coact[name])
        log_g = np.log1p(g)
        results[name] = _unpack_scipy(stats.pearsonr(log_g, c))
    return results


# ── 3. Binned coactivation by GIS decile ──────────────────────────────────────


def binned_coact_by_gis_decile(
    gis: dict[str, torch.Tensor],
    coact: dict[str, torch.Tensor],
    n_bins: int = 10,
) -> dict[str, dict[str, list[float]]]:
    """For each module, bin pairs by GIS decile and compute mean/median coactivation per bin."""
    results: dict[str, dict[str, list[float]]] = {}
    for name in sorted(gis):
        g = _off_diag_flat(gis[name])
        c = _off_diag_flat(coact[name])

        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(g, percentiles)
        bin_indices = np.digitize(g, bin_edges[1:-1])

        means = []
        medians = []
        bin_centers = []
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() == 0:
                means.append(float("nan"))
                medians.append(float("nan"))
            else:
                means.append(float(np.mean(c[mask])))
                medians.append(float(np.median(c[mask])))
            bin_centers.append(float((bin_edges[b] + bin_edges[b + 1]) / 2))

        results[name] = {
            "bin_centers": bin_centers,
            "means": means,
            "medians": medians,
        }
    return results


def plot_decile_curves(
    binned: dict[str, dict[str, list[float]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in sorted(binned.items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(data["bin_centers"], data["means"], "o-", label="Mean", markersize=5)
        ax.plot(data["bin_centers"], data["medians"], "s--", label="Median", markersize=5)
        ax.set_xlabel("GIS (decile bin center)")
        ax.set_ylabel("Coactivation Fraction")
        ax.set_title(f"{name} — Coactivation by GIS decile")
        ax.legend()
        ax.grid(alpha=0.3)

        safe = name.replace(".", "_")
        path = output_dir / f"decile_{safe}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  {path}")


# ── 4. GIS conditioned on coactivation threshold ──────────────────────────────


def gis_conditioned_on_coact(
    gis: dict[str, torch.Tensor],
    coact: dict[str, torch.Tensor],
    threshold: float = 0.1,
) -> dict[str, dict[str, float]]:
    """Compare GIS distribution for high-coact pairs vs low-coact pairs."""
    results: dict[str, dict[str, float]] = {}
    for name in sorted(gis):
        g = _off_diag_flat(gis[name])
        c = _off_diag_flat(coact[name])

        high = g[c >= threshold]
        low = g[c < threshold]

        if len(high) > 0 and len(low) > 0:
            ks_stat, ks_pval = _unpack_scipy(stats.ks_2samp(high, low))
        else:
            ks_stat, ks_pval = 0.0, 1.0

        results[name] = {
            "n_high": int(len(high)),
            "n_low": int(len(low)),
            "mean_gis_high": float(np.mean(high)) if len(high) > 0 else float("nan"),
            "mean_gis_low": float(np.mean(low)) if len(low) > 0 else float("nan"),
            "median_gis_high": float(np.median(high)) if len(high) > 0 else float("nan"),
            "median_gis_low": float(np.median(low)) if len(low) > 0 else float("nan"),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
        }
    return results


def plot_conditional_gis_histograms(
    gis: dict[str, torch.Tensor],
    coact: dict[str, torch.Tensor],
    threshold: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(gis):
        g = _off_diag_flat(gis[name])
        c = _off_diag_flat(coact[name])

        high = g[c >= threshold]
        low = g[c < threshold]
        if len(high) == 0 or len(low) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(0, max(g.max(), 1.0), 60)
        ax.hist(low, bins=bins, alpha=0.5, label=f"Coact < {threshold}", density=True)
        ax.hist(high, bins=bins, alpha=0.5, label=f"Coact >= {threshold}", density=True)
        ax.set_xlabel("Geometric Interaction Strength")
        ax.set_ylabel("Density")
        ax.set_title(f"{name} — GIS conditioned on coactivation")
        ax.legend()
        ax.grid(alpha=0.3)

        safe = name.replace(".", "_")
        path = output_dir / f"cond_gis_{safe}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  {path}")


# ── 5. Permutation test ──────────────────────────────────────────────────────


def permutation_test(
    gis: dict[str, torch.Tensor],
    coact: dict[str, torch.Tensor],
    n_permutations: int = 200,
    max_pairs: int = 50_000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Test whether observed Spearman correlation exceeds what's expected from
    shuffling component indices within each module.

    For modules with more than max_pairs off-diagonal elements, subsamples
    pairs for the spearmanr calls to keep runtime manageable.
    """
    rng = np.random.default_rng(seed)
    results: dict[str, dict[str, float]] = {}

    for name in sorted(gis):
        g_mat = gis[name].numpy()
        c_mat = coact[name].numpy()
        n = g_mat.shape[0]
        mask = ~np.eye(n, dtype=bool)

        g_flat = g_mat[mask]
        c_flat = c_mat[mask]

        # Subsample if too many pairs
        if len(g_flat) > max_pairs:
            sub_idx = rng.choice(len(g_flat), max_pairs, replace=False)
            g_sub, c_sub = g_flat[sub_idx], c_flat[sub_idx]
        else:
            g_sub, c_sub = g_flat, c_flat
            sub_idx = None

        observed_rho = _unpack_scipy(stats.spearmanr(g_sub, c_sub))[0]

        null_rhos = np.empty(n_permutations)
        for i in range(n_permutations):
            perm = rng.permutation(n)
            g_perm = g_mat[np.ix_(perm, perm)][mask]
            if sub_idx is not None:
                g_perm = g_perm[sub_idx]
            null_rhos[i] = _unpack_scipy(stats.spearmanr(g_perm, c_sub))[0]

        p_value = float(np.mean(np.abs(null_rhos) >= np.abs(observed_rho)))

        results[name] = {
            "observed_rho": float(observed_rho),
            "null_mean": float(np.mean(null_rhos)),
            "null_std": float(np.std(null_rhos)),
            "p_value": p_value,
        }
        logger.info(
            f"  {name}: rho={observed_rho:.4f}, null={np.mean(null_rhos):.4f}±{np.std(null_rhos):.4f}, p={p_value:.4f}"
        )

    return results


# ── 6. Mutual information ────────────────────────────────────────────────────


def mutual_information_per_module(
    gis: dict[str, torch.Tensor],
    coact: dict[str, torch.Tensor],
    n_bins: int = 20,
) -> dict[str, float]:
    """Estimate mutual information between discretised GIS and coactivation."""
    results: dict[str, float] = {}
    for name in sorted(gis):
        g = _off_diag_flat(gis[name])
        c = _off_diag_flat(coact[name])

        # Discretise into equal-frequency bins
        g_bins = np.clip(
            np.digitize(g, np.percentile(g, np.linspace(0, 100, n_bins + 1)[1:-1])),
            0,
            n_bins - 1,
        )
        c_bins = np.clip(
            np.digitize(c, np.percentile(c, np.linspace(0, 100, n_bins + 1)[1:-1])),
            0,
            n_bins - 1,
        )

        # Joint and marginal distributions
        joint = np.zeros((n_bins, n_bins))
        for gi, ci in zip(g_bins, c_bins, strict=True):
            joint[gi, ci] += 1
        joint /= joint.sum()

        p_g = joint.sum(axis=1)
        p_c = joint.sum(axis=0)

        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint[i, j] > 0 and p_g[i] > 0 and p_c[j] > 0:
                    mi += joint[i, j] * np.log2(joint[i, j] / (p_g[i] * p_c[j]))

        results[name] = float(mi)
    return results


# ── 7. Logistic regression: GIS predicting top-k% coactivation ───────────────


def logistic_regression_auc(
    gis: dict[str, torch.Tensor],
    coact: dict[str, torch.Tensor],
    top_k_pct: float = 10.0,
) -> dict[str, dict[str, float]]:
    """Train logistic regression: can GIS predict whether a pair is in the top-k% of coactivation?"""
    results: dict[str, dict[str, float]] = {}
    for name in sorted(gis):
        g = _off_diag_flat(gis[name])
        c = _off_diag_flat(coact[name])

        threshold = np.percentile(c, 100 - top_k_pct)
        y = (c >= threshold).astype(int)

        # Need both classes
        if y.sum() == 0 or y.sum() == len(y):
            results[name] = {
                "auc": float("nan"),
                "n_positive": int(y.sum()),
                "threshold": float(threshold),
            }
            continue

        X = np.log1p(g).reshape(-1, 1)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        y_prob = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)

        results[name] = {
            "auc": float(auc),
            "coef": float(clf.coef_[0, 0]),
            "n_positive": int(y.sum()),
            "threshold": float(threshold),
        }
    return results


# ── 8. Per-layer comparison ──────────────────────────────────────────────────


def per_layer_summary(
    spearman: dict[str, tuple[float, float]],
    pearson_log: dict[str, tuple[float, float]],
    mi: dict[str, float],
    logreg: dict[str, dict[str, float]],
) -> dict[int, dict[str, dict[str, float]]]:
    """Aggregate statistics by layer number."""
    layers: dict[int, dict[str, dict[str, float]]] = {}
    for name in spearman:
        layer = _parse_layer(name)
        if layer is None:
            continue
        if layer not in layers:
            layers[layer] = {}
        layers[layer][name] = {
            "spearman_rho": spearman[name][0],
            "pearson_log_r": pearson_log[name][0],
            "mi_bits": mi[name],
            "logreg_auc": logreg[name]["auc"],
        }
    return dict(sorted(layers.items()))


def plot_per_layer_comparison(
    layer_summary: dict[int, dict[str, dict[str, float]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["spearman_rho", "pearson_log_r", "mi_bits", "logreg_auc"]
    titles = [
        "Spearman rho",
        "Pearson r (log GIS)",
        "Mutual Information (bits)",
        "Logistic Regression AUC",
    ]

    # Aggregate: mean per layer across modules
    layer_nums = sorted(layer_summary.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, metric, title in zip(axes.flat, metrics, titles, strict=True):
        # Individual module points
        for layer in layer_nums:
            module_vals = [
                v[metric] for v in layer_summary[layer].values() if not np.isnan(v[metric])
            ]
            ax.scatter([layer] * len(module_vals), module_vals, alpha=0.5, s=30, color="#0173B2")

        # Layer means
        layer_means = []
        for layer in layer_nums:
            vals = [v[metric] for v in layer_summary[layer].values() if not np.isnan(v[metric])]
            layer_means.append(np.mean(vals) if vals else float("nan"))
        ax.plot(
            layer_nums,
            layer_means,
            "o-",
            color="#D55E00",
            markersize=8,
            linewidth=2,
            label="Layer mean",
        )

        ax.set_xlabel("Layer")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(layer_nums)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("GIS–Coactivation relationship across layers", fontsize=14)
    fig.tight_layout()
    path = output_dir / "per_layer_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main(data_path: str | None = None) -> None:
    if data_path is None:
        # Find latest output
        out_dir = Path(__file__).parent / "out"
        candidates = sorted(out_dir.iterdir()) if out_dir.exists() else []
        assert candidates, f"No output directories found in {out_dir}"
        data_path_resolved = candidates[-1] / "data.pt"
    else:
        data_path_resolved = Path(data_path)

    assert data_path_resolved.exists(), f"Data file not found: {data_path_resolved}"
    logger.info(f"Loading data from {data_path_resolved}")
    data = load_data(data_path_resolved)

    gis = data["gis_matrices"]
    coact = data["coactivation_fractions"]
    run_id = data.get("run_id", "unknown")

    output_dir = data_path_resolved.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"run_id": run_id}

    # ── 1. Spearman ───────────────────────────────────────────────────────
    logger.info("1. Spearman rank correlation per module")
    spearman = spearman_per_module(gis, coact)
    results["spearman"] = {name: {"rho": r, "pvalue": p} for name, (r, p) in spearman.items()}
    for name, (rho, pval) in sorted(spearman.items()):
        logger.info(f"  {name}: rho={rho:.4f}, p={pval:.2e}")

    # ── 2. Pearson on log GIS ─────────────────────────────────────────────
    logger.info("2. Pearson correlation on log(1+GIS)")
    pearson_log = pearson_log_gis_per_module(gis, coact)
    results["pearson_log_gis"] = {
        name: {"r": r, "pvalue": p} for name, (r, p) in pearson_log.items()
    }
    for name, (r, pval) in sorted(pearson_log.items()):
        logger.info(f"  {name}: r={r:.4f}, p={pval:.2e}")

    # ── 3. Binned coactivation by GIS decile ──────────────────────────────
    logger.info("3. Binned coactivation by GIS decile")
    binned = binned_coact_by_gis_decile(gis, coact)
    results["decile_bins"] = binned
    plot_decile_curves(binned, output_dir / "decile_plots")

    # ── 4. GIS conditioned on coactivation ────────────────────────────────
    coact_threshold = 0.1
    logger.info(f"4. GIS conditioned on coactivation (threshold={coact_threshold})")
    cond = gis_conditioned_on_coact(gis, coact, threshold=coact_threshold)
    results["conditional_gis"] = cond
    for name, info in sorted(cond.items()):
        logger.info(
            f"  {name}: high_mean={info['mean_gis_high']:.3f}, low_mean={info['mean_gis_low']:.3f}, "
            f"KS={info['ks_statistic']:.3f}, p={info['ks_pvalue']:.2e}"
        )
    plot_conditional_gis_histograms(
        gis, coact, coact_threshold, output_dir / "conditional_histograms"
    )

    # ── 5. Permutation test ───────────────────────────────────────────────
    logger.info("5. Permutation test (1000 permutations)")
    perm = permutation_test(gis, coact)
    results["permutation_test"] = perm

    # ── 6. Mutual information ─────────────────────────────────────────────
    logger.info("6. Mutual information (20 bins)")
    mi = mutual_information_per_module(gis, coact)
    results["mutual_information_bits"] = mi
    for name, mi_val in sorted(mi.items()):
        logger.info(f"  {name}: MI={mi_val:.4f} bits")

    # ── 7. Logistic regression AUC ────────────────────────────────────────
    logger.info("7. Logistic regression: GIS predicting top-10% coactivation")
    logreg = logistic_regression_auc(gis, coact)
    results["logistic_regression"] = logreg
    for name, info in sorted(logreg.items()):
        logger.info(f"  {name}: AUC={info['auc']:.4f}")

    # ── 8. Per-layer comparison ───────────────────────────────────────────
    logger.info("8. Per-layer comparison")
    layer_summary = per_layer_summary(spearman, pearson_log, mi, logreg)
    results["per_layer"] = {str(k): v for k, v in layer_summary.items()}
    plot_per_layer_comparison(layer_summary, output_dir)

    # ── Save results ──────────────────────────────────────────────────────
    results_path = output_dir / "results.json"

    # Convert any remaining non-serialisable types
    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not JSON serialisable: {type(obj)}")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    logger.info(f"Saved results → {results_path}")

    # ── Print summary table ───────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 90)
    logger.info(
        f"{'Module':<30} {'Spearman':>10} {'Pearson(log)':>13} {'MI(bits)':>10} {'AUC':>8} {'Perm p':>8}"
    )
    logger.info("-" * 90)
    for name in sorted(gis.keys()):
        rho = spearman[name][0]
        r = pearson_log[name][0]
        mi_val = mi[name]
        auc = logreg[name]["auc"]
        perm_p = perm[name]["p_value"]
        logger.info(
            f"  {name:<28} {rho:>10.4f} {r:>13.4f} {mi_val:>10.4f} {auc:>8.4f} {perm_p:>8.4f}"
        )
    logger.info("=" * 90)


if __name__ == "__main__":
    fire.Fire(main)
