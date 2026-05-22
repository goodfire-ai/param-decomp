"""Summarize 2-pool Qwen3-1.7B sweep results from W&B.

Pulls runs named with the `q17b-*` prefix from the param-decomp project, reads
their final-window perf/step_ms + mem/pool_a_peak_gb + mem/pool_b_peak_gb, and
emits a TSV-formatted table keyed by sweep axes (batch, seq, ci_d, ci_n,
topology). Useful for at-a-glance: which configs fit, which OOM'd, which are
fastest.

Runs without our naming convention are skipped. Runs missing the perf/mem
metrics (e.g. crashed before logging) show as "—".
"""

import argparse
from dataclasses import dataclass
from typing import Any

import wandb


@dataclass(order=True, frozen=True)
class RowSortKey:
    """Tuple-equivalent sort key for the result table, in field order."""

    topology: str
    seq: int
    ci_d: int
    ci_n: int
    batch: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="goodfire")
    parser.add_argument("--project", default="param-decomp")
    parser.add_argument("--name-prefix", default="q17b-")
    parser.add_argument(
        "--last-n",
        type=int,
        default=10,
        help="Window of last N logged steps to summarize (median).",
    )
    args = parser.parse_args()

    api = wandb.Api()
    runs = list(
        api.runs(
            f"{args.entity}/{args.project}",
            filters={"display_name": {"$regex": f"^{args.name_prefix}"}},
        )
    )
    if not runs:
        print(f"No runs found matching {args.name_prefix}*")
        return

    rows: list[dict[str, Any]] = []
    for run in runs:
        # view_meta keys are flattened to "view_meta/<key>" in wandb.config
        batch = run.config.get("view_meta/batch") or run.config.get("pd/batch_size", "?")
        seq = run.config.get("view_meta/seq", "?")
        ci_d = run.config.get("view_meta/ci_d", "?")
        ci_n = run.config.get("view_meta/ci_n", "?")
        topology = run.config.get("view_meta/topology", "?")

        # History keys are prefixed with the sink section, "train/...".
        keys = ["train/perf/step_ms", "train/mem/pool_a_peak_gb", "train/mem/pool_b_peak_gb"]
        history = run.history(keys=keys, samples=10_000, pandas=False)
        rows_with_perf = [h for h in history if h.get("train/perf/step_ms") is not None]
        if not rows_with_perf:
            step_ms = pool_a = pool_b = None
        else:
            tail = rows_with_perf[-args.last_n :]

            def med(key: str, tail_rows: list[dict[str, Any]] = tail) -> float | None:
                vals = sorted([h[key] for h in tail_rows if h.get(key) is not None])
                if not vals:
                    return None
                return vals[len(vals) // 2]

            step_ms = med("train/perf/step_ms")
            pool_a = med("train/mem/pool_a_peak_gb")
            pool_b = med("train/mem/pool_b_peak_gb")

        rows.append(
            {
                "name": run.name,
                "state": run.state,
                "batch": batch,
                "seq": seq,
                "ci_d": ci_d,
                "ci_n": ci_n,
                "topology": topology,
                "step_ms": step_ms,
                "pool_a_gb": pool_a,
                "pool_b_gb": pool_b,
                "url": run.url,
            }
        )

    def sort_key(r: dict[str, Any]) -> RowSortKey:
        return RowSortKey(
            topology=str(r["topology"]),
            seq=r["seq"] if isinstance(r["seq"], int) else 0,
            ci_d=r["ci_d"] if isinstance(r["ci_d"], int) else 0,
            ci_n=r["ci_n"] if isinstance(r["ci_n"], int) else 0,
            batch=r["batch"] if isinstance(r["batch"], int) else 0,
        )

    rows.sort(key=sort_key)

    cols = [
        "topology",
        "batch",
        "seq",
        "ci_d",
        "ci_n",
        "state",
        "step_ms",
        "pool_a_gb",
        "pool_b_gb",
        "name",
    ]

    def fmt(x: Any) -> str:
        if x is None:
            return "—"
        if isinstance(x, float):
            return f"{x:.1f}"
        return str(x)

    widths = {c: max(len(c), max((len(fmt(r[c])) for r in rows), default=1)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(fmt(r[c]).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    main()
