"""Plot post_warmup_component_scale × StochasticReconLoss.coeff sweep results."""

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


import re

_COEFF_RE = re.compile(r"StochRecon-coeff-([0-9.]+)")


def find_recon_coeff(run_name: str) -> float | None:
    m = _COEFF_RE.search(run_name)
    return float(m.group(1)) if m else None


def pull_runs(experiment: str, since_iso: str) -> list[dict]:
    api = wandb.Api()
    runs = api.runs(
        WANDB_ENTITY_PROJECT,
        filters={
            "display_name": {"$regex": f"^{experiment}-"},
            "state": "finished",
            "created_at": {"$gte": since_iso},
        },
    )

    rows = []
    for run in runs:
        scale = run.config.get("post_warmup_component_scale")
        coeff = find_recon_coeff(run.name)
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
            "coeff": coeff,
            "seed": seed,
            "eval/loss/StochasticReconLoss": tail_mean("eval/loss/StochasticReconLoss"),
            "l0_sum": l0_sum,
            "eval/target_solution_error/total": tail_mean("eval/target_solution_error/total"),
        })

    return rows


def make_plot(rows: list[dict], experiment: str, out_path: Path, none_baseline: dict | None = None) -> None:
    scales = sorted({r["scale"] for r in rows if r["scale"] is not None})
    coeffs = sorted({r["coeff"] for r in rows if r["coeff"] is not None})

    fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 4.8))
    fig.suptitle(
        f"{experiment} — post_warmup_component_scale × StochasticReconLoss.coeff "
        f"(asymmetric, U-only, delta component on)\n"
        f"{len(rows)} finished runs; scales={scales}; coeffs={coeffs}; "
        f"lines = per-scale mean over seeds; red dashed = None baseline (coeff=1.0)",
        fontsize=9,
    )

    cmap = plt.get_cmap("viridis")
    color_for_scale = {s: cmap(0.15 + 0.7 * i / max(1, len(scales) - 1)) for i, s in enumerate(scales)}

    for ax, (metric, label) in zip(axes, METRICS.items(), strict=True):
        for scale in scales:
            sub = [r for r in rows if r["scale"] == scale and r[metric] is not None]
            if not sub:
                continue
            by_coeff: dict[float, list[float]] = {}
            for r in sub:
                by_coeff.setdefault(r["coeff"], []).append(r[metric])
            cs = sorted(by_coeff.keys())
            means = np.array([np.mean(by_coeff[c]) for c in cs])
            stds = np.array([np.std(by_coeff[c]) for c in cs])

            ax.errorbar(
                cs, means, yerr=stds,
                fmt="o-", color=color_for_scale[scale], linewidth=2, markersize=7,
                capsize=4, label=f"α={scale:g}",
            )

            scatter_x = [r["coeff"] for r in sub]
            scatter_y = [r[metric] for r in sub]
            ax.scatter(scatter_x, scatter_y, s=20, color=color_for_scale[scale], alpha=0.25)

        if none_baseline is not None and metric in none_baseline:
            v = none_baseline[metric]
            ax.axhline(v, color="tab:red", linestyle="--", linewidth=2, label=f"None: {v:.4g}")

        ax.set_xscale("log", base=3)
        ax.set_xlabel("StochasticReconLoss.coeff (log₃)")
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment")
    p.add_argument("--since", default="2026-04-22T13:19:00Z", help="ISO cutoff for sweep-478317 runs")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    out_path = args.out or (SPD_OUT_DIR / "www" / f"scale_x_recon_coeff_{args.experiment}.png")

    rows = pull_runs(args.experiment, since_iso=args.since)
    print(f"pulled {len(rows)} runs since {args.since}")
    if not rows:
        return

    api = wandb.Api()
    baseline_runs = api.runs(
        WANDB_ENTITY_PROJECT,
        filters={
            "display_name": {"$regex": f"^{args.experiment}-post_warmup_component_scale-None-"},
            "state": "finished",
            "created_at": {"$gte": "2026-04-22T12:29:00Z"},
        },
    )
    none_rows: list[dict] = []
    for run in baseline_runs:
        hist = list(run.scan_history())
        if not hist:
            continue
        def tm(k: str) -> float | None:
            vs = [h[k] for h in hist if k in h and h[k] is not None]
            return float(np.mean(vs[-max(1, len(vs) // 5):])) if vs else None
        l0_vals = [tm(f"eval/l0/0.1_{layer}") for layer in ("linear1", "linear2")]
        none_rows.append({
            "eval/loss/StochasticReconLoss": tm("eval/loss/StochasticReconLoss"),
            "l0_sum": sum(l0_vals) if all(v is not None for v in l0_vals) else None,
            "eval/target_solution_error/total": tm("eval/target_solution_error/total"),
        })
    if none_rows:
        none_baseline = {
            m: float(np.mean([r[m] for r in none_rows if r.get(m) is not None]))
            for m in METRICS
        }
        print(f"None baseline from previous sweep: {none_baseline}")
    else:
        none_baseline = None

    make_plot(rows, args.experiment, out_path, none_baseline=none_baseline)


if __name__ == "__main__":
    main()
