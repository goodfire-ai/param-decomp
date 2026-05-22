"""Stacked-bar visualization of predicted pool A memory breakdown vs observed.

Uses the MVP cost model (mem_model.py) to compute per-contributor predicted GB
for every measurement. Plots each row as a stacked bar; observed peak shown as
a marker on top. Variable-term coefficients (c_act, c_ci) and overhead refit
from the same data inline.

Run:
    python -m param_decomp.scripts.two_pool_benchmark.plot_mem_breakdown
Output: ~/param_decomp_out/two_pool_gen/mem_breakdown.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from param_decomp.scripts.two_pool_benchmark.mem_model import (
    MEASUREMENTS,
    MemMeasurement,
    predict_pool_a_components,
    x_features,
)

OUT = Path("/mnt/home/oli/param_decomp_out/two_pool_gen/mem_breakdown.png")


def fit_coeffs() -> tuple[float, float, float]:
    """Fit c_act, c_ci, k_overhead from MEASUREMENTS (same as mem_model.fit_and_report)."""
    constants = []
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    for m in MEASUREMENTS:
        comps = predict_pool_a_components(
            m.batch,
            m.seq,
            m.bpg,
            m.ddp,
            m.ci_d,
            m.ci_n,
            m.use_fused_kl,
        )
        constants.append(sum(comps.values()))
        x_act, x_ci = x_features(m.batch, m.seq, m.ddp, m.ci_d, m.ci_n)
        x_rows.append([x_act, x_ci, 1.0])
        y_rows.append(m.observed_pool_a_gb - constants[-1])
    x_arr = np.array(x_rows)
    y_arr = np.array(y_rows)
    coeffs, *_ = np.linalg.lstsq(x_arr, y_arr, rcond=None)
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])


def label_for(m: MemMeasurement) -> str:
    fkl = "fkl" if m.use_fused_kl else "nofkl"
    blk = f"{m.bpg}block" if m.bpg > 1 else "1block"
    return f"{blk}-{m.ddp}ddp\nb={m.batch} s={m.seq}\nci=d{m.ci_d}n{m.ci_n} {fkl}"


def main() -> None:
    c_act, c_ci, k_overhead = fit_coeffs()
    print(f"c_act={c_act:.2f}  c_ci={c_ci:.4f}  k_overhead={k_overhead:.2f} GB")

    # Sort by observed (smallest → largest) for a left-to-right "memory ladder".
    sorted_m = sorted(MEASUREMENTS, key=lambda m: m.observed_pool_a_gb)

    # Collect per-config segments (in stacking order, bottom → top).
    seg_names = [
        "target_model",
        "vu_state",
        "ci_params",
        "cached_hidden",
        "per_iter_predgrad",
        "kl_work",
        "target_acts (c_act)",
        "ci_acts (c_ci)",
        "k_overhead",
    ]
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(seg_names)))

    rows: list[dict[str, float]] = []
    observed: list[float] = []
    for m in sorted_m:
        comps = predict_pool_a_components(
            m.batch,
            m.seq,
            m.bpg,
            m.ddp,
            m.ci_d,
            m.ci_n,
            m.use_fused_kl,
        )
        x_act, x_ci = x_features(m.batch, m.seq, m.ddp, m.ci_d, m.ci_n)
        rows.append(
            {
                "target_model": comps["target_model"],
                "vu_state": comps["vu_state"],
                "ci_params": comps["ci_params"],
                "cached_hidden": comps["cached_hidden"],
                "per_iter_predgrad": comps["per_iter_predgrad"],
                "kl_work": comps["kl_work"],
                "target_acts (c_act)": c_act * x_act,
                "ci_acts (c_ci)": c_ci * x_ci,
                "k_overhead": k_overhead,
            }
        )
        observed.append(m.observed_pool_a_gb)

    # Plot
    n = len(rows)
    _fig, ax = plt.subplots(figsize=(max(14, n * 0.85), 8))
    x_idx = np.arange(n)
    bottom = np.zeros(n)
    for i, name in enumerate(seg_names):
        heights = np.array([r[name] for r in rows])
        ax.bar(x_idx, heights, bottom=bottom, label=name, color=colors[i], width=0.7)
        bottom = bottom + heights

    # Observed peaks as dark markers
    ax.scatter(
        x_idx,
        observed,
        marker="_",
        s=300,
        color="black",
        linewidths=3,
        zorder=10,
        label="observed peak",
    )

    # B200 cap line
    ax.axhline(178, color="red", linestyle="--", alpha=0.5, label="B200 cap (178 GB)")

    ax.set_xticks(x_idx)
    ax.set_xticklabels([label_for(m) for m in sorted_m], rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Pool A memory (GB)")
    ax.set_title(
        f"Pool A memory breakdown — predicted (stacked) vs observed (—)\n"
        f"Fitted coefficients: c_act={c_act:.1f} (acts/layer/token), "
        f"c_ci={c_ci:.3f}, k_overhead={k_overhead:.1f} GB"
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=120, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
