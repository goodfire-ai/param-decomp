"""Reproduce the 3-pool topology screen for the big512 production regime.

This is a config-specific RECORD, not a generic tool — see `scripts/topology_search.py`
for the model and `docs/3pool_topology_calibration_2026-06-03.md` for the findings.

Calibration provenance (2026-06-03, current code: vendored ComponentGPT2, LW+CI
torch.compile, activation checkpointing):
  * per-pool COMPUTE from the rebalance-6site torch.profiler trace (job 38431,
    112 ranks LW64/CI16/PPGD32, B=256) via `scripts/analyze_3pool_trace.py`. Its
    per-rank batch_local (lw 64 / ci 16 / ppgd 8) is IDENTICAL to big512 production,
    so the per-rank compute carries over exactly.
  * step WALL from big512 production itself (p-b6505e9c, 224 ranks, ~2358 ms) so
    OVERHEAD reflects the 224-rank cross-pool cost (the 112-rank trace's own wall is
    ~2138 ms; overhead grows with rank count).

Usage:
    python scripts/repro_big512_topology_search.py            # print the search
    python scripts/repro_big512_topology_search.py --plots DIR # also (re)write figures
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.topology_search import Calibration, Topo, report  # noqa: E402

CALIBRATION = Calibration.from_measurements(
    n_sites=96,
    ci=(579.1, 16),  # (compute_ms, batch_local)
    ppgd=(1140.2, 8),
    lw=(1243.5, 64),
    lw_sites_per_block=6,
    step_wall_ms=2358.0,  # big512 production step_ms @ 224 ranks
)
BIG512 = Topo(n_ci=32, n_ppgd=64, n_blocks=16, n_per_block=8, batch=512)

# Historical LW per-(site·sample) compute at the old bl_lw=4 calibration (job 34379,
# pre-compile): 632 ms / (4 sites/block · 4 samples). Compared to the current bl_lw=64
# point to show the sublinearity. (compute_ms, sites_per_block, batch_local)
LW_OLD_BL4 = (632.0, 4, 4)

# Where each calibration INPUT comes from (the repro trusts these; verify with the
# commands shown). Printed so running this script is self-evident about its sources.
PROVENANCE = """\
inputs (verify these — the model below is derived from them):
  per-pool compute  ← scripts/analyze_3pool_trace.py on the rebalance-6site trace
                      (job 38431, 112 ranks LW64/CI16/PPGD32, B=256; per-rank
                      batch_local lw64/ci16/ppgd8 == big512, so compute carries over)
  step wall 2358 ms ← big512 production p-b6505e9c, logged train/perf/step_ms @224 ranks
  LW old point 632ms@(spb4,bl4) ← job 34379 (pre-compile), for the sublinearity figure"""


def make_plots(out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cream, ink, oxblood, hair, muted = "#f6f0e2", "#2b2b2b", "#6e1423", "#cabfa6", "#b9a06b"
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino Linotype", "Palatino", "Georgia", "DejaVu Serif"],
            "figure.facecolor": cream,
            "axes.facecolor": cream,
            "savefig.facecolor": cream,
            "text.color": ink,
            "axes.labelcolor": ink,
            "xtick.color": ink,
            "ytick.color": ink,
            "axes.edgecolor": hair,
            "axes.linewidth": 0.8,
            "axes.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fig 1 — per-pool compute vs the step wall; the gap above max-compute is overhead.
    cal, t = CALIBRATION, BIG512
    pools = [("CI", t.compute_ci(cal)), ("PPGD", t.compute_ppgd(cal)), ("LW", t.compute_lw(cal))]
    step = t.step_ms(cal)
    pole = max(c for _, c in pools)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axvspan(pole, step, color=oxblood, alpha=0.10)
    ax.axvline(step, color=oxblood, ls="--", lw=1.0)
    for i, (_name, c) in enumerate(pools):
        ax.barh(i, c, color=oxblood if c == pole else muted, height=0.58)
        ax.text(c + 25, i, f"{c:.0f} ms", va="center", fontsize=10, color=ink)
    ax.set_yticks(range(len(pools)))
    ax.set_yticklabels([n for n, _ in pools])
    ax.set_xlabel("per-rank compute (ms / step)")
    ax.set_xlim(0, step * 1.04)
    ax.set_xticks([0, 500, 1000, 1500, 2000, round(step)])
    ax.text(
        (pole + step) / 2,
        1,
        f"non-overlapped\noverhead\n{cal.overhead:.0f} ms\n({100 * cal.overhead / step:.0f}% of step)",
        ha="center",
        va="center",
        fontsize=9.5,
        color=oxblood,
    )
    ax.set_title("3-pool step at the big512 regime (224 ranks) — LW is the pole")
    fig.tight_layout()
    fig.savefig(out_dir / "step_breakdown.png", dpi=150)
    plt.close(fig)

    # Fig 2 — LW per-(site·sample) compute vs per-rank batch (the sublinearity caveat).
    old_ms, old_spb, old_bl = LW_OLD_BL4
    old_per = old_ms / (old_spb * old_bl)
    new_per = cal.k_lw_total / cal.n_sites
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    xs = ["bl_lw = 4\n(pre-compile)", "bl_lw = 64\n(big512)"]
    ys = [old_per, new_per]
    ax.bar(xs, ys, color=[muted, oxblood], width=0.6)
    for xi, y in enumerate(ys):
        ax.text(xi, y + 0.8, f"{y:.1f}", ha="center", fontsize=10, color=ink)
    ax.annotate(
        f"≈{old_per / new_per:.0f}× drop\n(fixed per-site overhead)",
        xy=(1, new_per),
        xytext=(0.55, old_per * 0.6),
        fontsize=9.5,
        color=oxblood,
        arrowprops=dict(arrowstyle="->", color=oxblood, lw=1.0),
    )
    ax.set_ylabel("LW compute per (site · sample), ms")
    ax.set_title("LW is sublinear in per-rank batch")
    fig.tight_layout()
    fig.savefig(out_dir / "lw_sublinearity.png", dpi=150)
    plt.close(fig)
    print(f"wrote figures → {out_dir}/step_breakdown.png, lw_sublinearity.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plots", type=Path, default=None, help="dir to (re)write the figures into")
    args = ap.parse_args()
    if args.plots is not None:
        make_plots(args.plots)
        return
    print(PROVENANCE + "\n")
    report(
        CALIBRATION,
        budget=224,
        baseline=BIG512,
        batch_groups=[
            ([512], "B=512 (current production batch)"),
            ([256, 512, 768, 1024], "B free (perf-only; changes opt dynamics)"),
        ],
    )


if __name__ == "__main__":
    main()
