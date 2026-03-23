"""Analyze merge quality at specific iterations.

For a given iteration, reconstructs the group-level coactivation matrix from
the raw memberships + group assignments, computes the full MDL cost matrix,
and produces a scatter plot of (MDL cost, Jaccard) for all candidate merges.

Usage:
    python -m spd.clustering.scripts.analyze_merge_quality \
        --run_dir /path/to/clustering/runs/c-XXXXX \
        --iterations 3242,3677,4340,4515,4579,4599,4790,5066
"""

import json
from pathlib import Path

import fire
import numpy as np
import torch
from scipy import sparse

from spd.clustering.compute_costs import compute_merge_costs
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.membership_snapshot import load_membership_snapshot
from spd.clustering.merge_history import MergeHistory
from spd.log import logger
from spd.settings import SPD_OUT_DIR


def build_group_coactivation(
    csc: sparse.csc_matrix,
    group_idxs: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Compute group-level coactivation matrix via GPU if available.

    Returns (n_groups, n_groups) int32 array where [i,j] = samples where both group i and j fire.
    """
    n_components = csc.shape[1]
    n_samples = csc.shape[0]

    # Build sparse mapping: (n_components, n_groups)
    rows = np.arange(n_components)
    cols = group_idxs
    mapping = sparse.csc_matrix(
        (np.ones(n_components, dtype=np.float32), (rows, cols)),
        shape=(n_components, n_groups),
    )

    # Group activation: (n_samples, n_groups) via sparse matmul on CPU, then GPU for coact
    group_act_sparse = csc.astype(np.float32) @ mapping
    group_act_sparse.data[:] = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Computing coactivation on {device} ({n_samples} samples, {n_groups} groups)")

    # Convert to dense torch tensor in chunks to avoid OOM
    CHUNK = 500_000
    n_chunks = (n_samples + CHUNK - 1) // CHUNK
    coact = torch.zeros((n_groups, n_groups), dtype=torch.int32, device=device)
    for i, start in enumerate(range(0, n_samples, CHUNK)):
        end = min(start + CHUNK, n_samples)
        logger.info(f"  Chunk {i+1}/{n_chunks}: samples {start}-{end}")
        chunk_dense = torch.from_numpy(group_act_sparse[start:end].toarray()).to(device=device, dtype=torch.float32)
        chunk_dense = (chunk_dense > 0).float()
        coact += (chunk_dense.T @ chunk_dense).int()

    return coact.cpu().numpy()


def analyze_iteration(
    csc: sparse.csc_matrix,
    history: MergeHistory,
    iteration: int,
    n_samples: int,
) -> dict:
    """Analyze all candidate merges at a specific iteration."""
    prev_iter = iteration - 1
    merge_state = history.merges[prev_iter]
    group_idxs = merge_state.group_idxs.numpy()
    k_groups = int(merge_state.k_groups)
    selected_pair = history.selected_pairs[iteration].tolist()

    logger.info(f"Iteration {iteration}: {k_groups} groups, computing coactivation...")
    coact = build_group_coactivation(csc, group_idxs, k_groups).astype(np.float32)
    coact_tensor = torch.from_numpy(coact)

    # Firing counts per group
    diag = np.diag(coact)

    # Normalize for MDL cost (as merge loop does)
    coact_norm = coact_tensor / n_samples

    # Compute MDL costs
    logger.info("Computing merge costs...")
    costs = compute_merge_costs(
        coact=coact_norm,
        merges=merge_state,
        alpha=history.merge_config.alpha,
    )
    costs_np = costs.numpy()

    # Vectorized pair extraction
    logger.info("Extracting pair metrics (vectorized)...")
    row_idx, col_idx = np.triu_indices(k_groups, k=1)
    all_costs = costs_np[row_idx, col_idx]
    all_coact = coact[row_idx, col_idx]
    all_si = diag[row_idx]
    all_sj = diag[col_idx]
    all_union = all_si + all_sj - all_coact
    with np.errstate(divide="ignore", invalid="ignore"):
        all_jaccard = np.where(all_union > 0, all_coact / all_union, 0.0)

    has_coact = all_coact > 0

    # Range sampler band
    min_cost = float(all_costs.min())
    max_cost = float(all_costs.max())
    threshold = history.merge_config.merge_pair_sampling_kwargs.get("threshold", 0.05)
    range_upper = min_cost + threshold * (max_cost - min_cost)

    in_range = all_costs <= range_upper
    n_in_range = int(in_range.sum())
    n_coact_in_range = int((in_range & has_coact).sum())
    n_zero_in_range = n_in_range - n_coact_in_range

    logger.info(
        f"  Range band: [{min_cost:.6f}, {range_upper:.6f}] "
        f"({n_in_range} pairs: {n_coact_in_range} co-firing, {n_zero_in_range} zero-coact)"
    )

    # Selected pair
    sel_a, sel_b = min(selected_pair), max(selected_pair)
    sel_mask = (row_idx == sel_a) & (col_idx == sel_b)
    assert sel_mask.sum() == 1, f"Selected pair {selected_pair} not found in upper triangle"
    sel_idx = np.where(sel_mask)[0][0]

    # Optimal pair
    opt_idx = int(np.argmin(all_costs))

    # Best co-firing pair
    coact_mask = has_coact
    best_coact_idx = int(np.where(coact_mask, all_costs, np.inf).argmin()) if coact_mask.any() else None

    def _pair_dict(idx: int, is_selected: bool = False) -> dict:
        c, si, sj = all_coact[idx], all_si[idx], all_sj[idx]
        pmi = float(np.log(c * n_samples / (si * sj))) if c > 0 and si > 0 and sj > 0 else None
        return {
            "i": int(row_idx[idx]), "j": int(col_idx[idx]),
            "cost": float(all_costs[idx]), "jaccard": float(all_jaccard[idx]),
            "pmi": pmi,
            "coact": int(c), "si": int(si), "sj": int(sj),
            "is_selected": is_selected,
        }

    selected = _pair_dict(sel_idx, is_selected=True)
    optimal = _pair_dict(opt_idx)
    best_coact = _pair_dict(best_coact_idx) if best_coact_idx is not None else None

    # Build pairs list for scatter plot (only keep a manageable sample)
    coact_indices = np.where(has_coact)[0]
    zero_indices = np.where(~has_coact)[0]
    # Sample up to 50k zero-coact pairs for plotting
    if len(zero_indices) > 50000:
        zero_sample = np.random.default_rng(0).choice(zero_indices, 50000, replace=False)
    else:
        zero_sample = zero_indices
    plot_indices = np.concatenate([coact_indices, zero_sample])

    pairs = []
    for idx in plot_indices:
        p = _pair_dict(int(idx), is_selected=(int(idx) == sel_idx))
        # PMI for co-firing pairs
        if all_coact[idx] > 0 and all_si[idx] > 0 and all_sj[idx] > 0:
            p["pmi"] = float(np.log(all_coact[idx] * n_samples / (all_si[idx] * all_sj[idx])))
        else:
            p["pmi"] = None
        pairs.append(p)

    logger.info(f"  Selected: cost={selected['cost']:.6f}, jaccard={selected['jaccard']:.4f}, coact={selected['coact']}")
    logger.info(f"  Optimal:  cost={optimal['cost']:.6f}, jaccard={optimal['jaccard']:.4f}, coact={optimal['coact']}")
    if best_coact:
        logger.info(f"  Best co-firing: cost={best_coact['cost']:.6f}, jaccard={best_coact['jaccard']:.4f}, coact={best_coact['coact']}")

    return {
        "iteration": iteration,
        "k_groups": k_groups,
        "n_samples": n_samples,
        "selected_pair": selected_pair,
        "min_cost": min_cost,
        "max_cost": max_cost,
        "range_upper": range_upper,
        "range_threshold": threshold,
        "n_in_range": n_in_range,
        "n_coact_in_range": n_coact_in_range,
        "n_zero_in_range": n_zero_in_range,
        "selected": selected,
        "optimal": optimal,
        "best_coact": best_coact,
        "pairs": pairs,
    }


def render_scatter(result: dict, output_path: Path) -> None:
    """Render density heatmap of MDL cost vs Jaccard, plus cost histogram showing band composition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    pairs = result["pairs"]
    all_costs = np.array([p["cost"] for p in pairs])
    all_jaccard = np.array([p["jaccard"] for p in pairs])
    all_coact = np.array([p["coact"] for p in pairs])

    has_coact = all_coact > 0
    n_zero = int((~has_coact).sum())
    n_cofire = int(has_coact.sum())

    cost_range = result["range_upper"] - result["min_cost"]
    x_lo = result["min_cost"] - cost_range * 0.5
    x_hi = result["min_cost"] + cost_range * 10

    fig, (ax_scatter, ax_hist) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1], sharex=True)

    # --- Top: Jaccard vs Cost density ---
    valid = np.isfinite(all_jaccard) & (all_costs >= x_lo) & (all_costs <= x_hi)
    cofire_valid = valid & has_coact
    if cofire_valid.sum() > 10:
        ax_scatter.hexbin(
            all_costs[cofire_valid], all_jaccard[cofire_valid],
            gridsize=80, cmap="Blues", norm=LogNorm(), mincnt=1, zorder=1,
        )

    # Range band
    ax_scatter.axvspan(result["min_cost"], result["range_upper"], alpha=0.15, color="gold", zorder=0)

    # Optimal merge
    opt = result["optimal"]
    ax_scatter.scatter([opt["cost"]], [opt["jaccard"]], c="lime", s=250, marker="D", zorder=10,
                       edgecolors="black", linewidths=1, label=f"Optimal (j={opt['jaccard']:.3f})")

    ax_scatter.set_ylabel("Jaccard", fontsize=12)
    ax_scatter.set_ylim(-0.02, 1.05)
    ax_scatter.set_title(
        f"Iteration {result['iteration']} — {result['k_groups']} groups, {result['n_samples']:,} samples\n"
        f"{n_cofire:,} co-firing pairs, {n_zero:,} zero-coact pairs",
        fontsize=11,
    )
    ax_scatter.legend(fontsize=9)

    # --- Bottom: stacked cost histogram (co-firing vs zero-coact) ---
    bins = np.linspace(x_lo, x_hi, 200)

    cofire_costs = all_costs[valid & has_coact]
    zero_costs = all_costs[valid & ~has_coact]

    ax_hist.hist(
        [cofire_costs, zero_costs], bins=bins, stacked=True,
        color=["steelblue", "indianred"], alpha=0.8,
        label=[f"co-firing ({len(cofire_costs):,})", f"zero coact ({len(zero_costs):,})"],
    )

    # Range band
    ax_hist.axvspan(result["min_cost"], result["range_upper"], alpha=0.2, color="gold",
                    label=f"range band ({result['n_in_range']:,} pairs)")

    # Annotate band composition
    zero_pct = result["n_zero_in_range"] / max(result["n_in_range"], 1) * 100
    ax_hist.annotate(
        f"In band: {result['n_coact_in_range']:,} co-fire, {result['n_zero_in_range']:,} zero ({zero_pct:.0f}%)",
        xy=(result["min_cost"], ax_hist.get_ylim()[1] * 0.85),
        fontsize=9, ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="goldenrod", alpha=0.9),
    )

    ax_hist.set_xlabel("MDL merge cost", fontsize=12)
    ax_hist.set_ylabel("Count", fontsize=12)
    ax_hist.set_yscale("log")
    ax_hist.legend(fontsize=9)
    ax_hist.set_xlim(x_lo, x_hi)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot: {output_path}")


def compute(
    run_dir: str,
    iterations: str,
    output_dir: str | None = None,
    max_samples: int = 500_000,
) -> None:
    """Compute merge quality data and save to disk. Slow (GPU coactivation)."""
    import pickle

    run_path = Path(run_dir)
    if isinstance(iterations, int):
        iter_list = [iterations]
    elif isinstance(iterations, str):
        iter_list = [int(x.strip()) for x in iterations.split(",")]
    else:
        iter_list = [int(x) for x in iterations]

    history = MergeHistory.read(run_path / "history.zip")
    config = history.merge_config

    merge_config_path = run_path / "merge_config.json"
    assert merge_config_path.exists()
    with open(merge_config_path) as f:
        mc = json.load(f)
    snapshot = load_membership_snapshot(Path(mc["snapshot_path"]))
    csc = snapshot.matrix_csc
    n_samples = snapshot.n_samples

    if n_samples > max_samples:
        logger.info(f"Subsampling {n_samples} -> {max_samples} samples")
        csc = csc[:max_samples]
        n_samples = max_samples

    out_dir = Path(output_dir) if output_dir else SPD_OUT_DIR / "www" / "merge_analysis" / run_path.name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Config: alpha={config.alpha}, threshold={config.merge_pair_sampling_kwargs}")
    logger.info(f"Membership: {csc.shape}, {n_samples} samples")

    for iteration in iter_list:
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing iteration {iteration}")
        logger.info(f"{'='*60}")
        result = analyze_iteration(csc, history, iteration, n_samples)

        data_path = out_dir / f"iter_{iteration}.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(result, f)
        logger.info(f"Saved data: {data_path}")

    logger.info(f"\nData saved to {out_dir}/iter_*.pkl")
    logger.info("Run 'render' subcommand to generate plots.")


def render(
    data_dir: str,
    iterations: str | None = None,
) -> None:
    """Render scatter plots from pre-computed data. Fast (no GPU needed)."""
    import glob
    import pickle

    data_path = Path(data_dir)
    assert data_path.exists(), f"Data dir not found: {data_path}"

    if iterations is not None:
        if isinstance(iterations, int):
            iter_list = [iterations]
        elif isinstance(iterations, str):
            iter_list = [int(x.strip()) for x in iterations.split(",")]
        else:
            iter_list = [int(x) for x in iterations]
        pkl_files = [data_path / f"iter_{i}.pkl" for i in iter_list]
    else:
        pkl_files = sorted(data_path.glob("iter_*.pkl"))

    assert pkl_files, f"No .pkl files found in {data_path}"

    for pkl_path in pkl_files:
        assert pkl_path.exists(), f"Not found: {pkl_path}"
        logger.info(f"Loading {pkl_path.name}...")
        with open(pkl_path, "rb") as f:
            result = pickle.load(f)
        out_path = pkl_path.with_suffix(".png")
        render_scatter(result, out_path)

    logger.info("Done.")


if __name__ == "__main__":
    fire.Fire({"compute": compute, "render": render})
