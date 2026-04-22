"""Pull post_warmup_component_scale sweep runs from WandB and plot recon/L0/target-error vs alpha."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

from spd.settings import SPD_OUT_DIR

WANDB_ENTITY_PROJECT = "goodfire/spd"

METRICS = {
    "eval/loss/StochasticReconLoss": "Stochastic recon loss",
    "l0_sum": "L0 (sum across layers, CI>0.1)",
    "eval/target_solution_error/total": "Target solution error",
}


def pull_runs(experiment: str, since_iso: str | None = None) -> list[dict]:
    api = wandb.Api()
    filters: dict = {
        "display_name": {"$regex": f"^{experiment}-post_warmup_component_scale-"},
        "state": "finished",
    }
    if since_iso is not None:
        filters["created_at"] = {"$gte": since_iso}
    runs = api.runs(WANDB_ENTITY_PROJECT, filters=filters)

    rows = []
    for run in runs:
        scale = run.config.get("post_warmup_component_scale")
        seed = run.config["seed"]

        hist = list(run.scan_history())
        if not hist:
            continue

        def tail_mean(key: str) -> float | None:
            vals = [h[key] for h in hist if key in h and h[key] is not None]
            if not vals:
                return None
            n_tail = max(1, len(vals) // 5)
            return float(np.mean(vals[-n_tail:]))

        l0_vals = [tail_mean(f"eval/l0/0.1_{layer}") for layer in ("linear1", "linear2")]
        l0_sum = sum(l0_vals) if all(v is not None for v in l0_vals) else None

        rows.append({
            "name": run.name,
            "scale": scale,
            "seed": seed,
            "eval/loss/StochasticReconLoss": tail_mean("eval/loss/StochasticReconLoss"),
            "l0_sum": l0_sum,
            "eval/target_solution_error/total": tail_mean("eval/target_solution_error/total"),
        })

    return rows


def aggregate(rows: list[dict], metric: str) -> dict:
    groups: dict = {}
    for r in rows:
        v = r[metric]
        if v is None:
            continue
        groups.setdefault(r["scale"], []).append(v)
    return {k: (float(np.mean(vs)), float(np.std(vs)), len(vs)) for k, vs in groups.items()}


def make_plot(rows: list[dict], experiment: str, out_path: Path) -> None:
    scales_sorted = sorted({r["scale"] for r in rows if r["scale"] is not None})

    fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 4.5))
    n_runs = len(rows)
    n_none = sum(1 for r in rows if r["scale"] is None)

    fig.suptitle(
        f"{experiment} — post_warmup_component_scale sweep (asymmetric, U-only)\n"
        f"{n_runs} finished runs ({n_none} None-baseline, {n_runs - n_none} scaled). "
        f"Red dashed = None baseline mean. Error bars = std across seeds.",
        fontsize=10,
    )

    for ax, (metric, label) in zip(axes, METRICS.items(), strict=True):
        agg = aggregate(rows, metric)

        seeds_x = [r["scale"] for r in rows if r["scale"] is not None and r[metric] is not None]
        seeds_y = [r[metric] for r in rows if r["scale"] is not None and r[metric] is not None]
        ax.scatter(seeds_x, seeds_y, s=30, alpha=0.35, color="tab:blue", label="per-seed")

        x_present = [s for s in scales_sorted if s in agg]
        means = np.array([agg[s][0] for s in x_present])
        stds = np.array([agg[s][1] for s in x_present])
        ax.errorbar(
            x_present, means, yerr=stds,
            fmt="o-", color="navy", linewidth=2, markersize=8,
            capsize=4, label="mean ± std",
        )

        if None in agg:
            none_mean, none_std, _ = agg[None]
            ax.axhline(none_mean, color="tab:red", linestyle="--", linewidth=2, label=f"None: {none_mean:.4g}")
            ax.axhspan(none_mean - none_std, none_mean + none_std, color="tab:red", alpha=0.1)

        ax.set_xscale("log")
        ax.set_xlabel("post_warmup_component_scale (α)")
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", help="e.g. tms_5-2, resid_mlp1")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--since", default=None, help="ISO8601 UTC cutoff for run created_at (e.g. 2026-04-22T12:29:00Z)")
    args = p.parse_args()

    out_path = args.out or (SPD_OUT_DIR / "www" / f"post_warmup_scale_sweep_{args.experiment}.png")

    rows = pull_runs(args.experiment, since_iso=args.since)
    print(f"pulled {len(rows)} finished runs for {args.experiment}")
    if not rows:
        print("no runs found, exiting")
        return
    for scale in sorted({r["scale"] for r in rows}, key=lambda s: (s is not None, s or 0)):
        seeds = [r["seed"] for r in rows if r["scale"] == scale]
        print(f"  scale={scale}: {len(seeds)} seeds")
    make_plot(rows, args.experiment, out_path)


if __name__ == "__main__":
    main()
