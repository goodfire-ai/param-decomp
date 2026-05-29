"""Extract steady-state train/perf/step_ms from a 3-pool run's metrics.jsonl.

Discards the step-0 init spike; reports mean/median over steps >= min_step.
"""

import json
import statistics
import sys
from pathlib import Path

PARAM_DECOMP_OUT_DIR = Path("/mnt/data/artifacts/mechanisms/param-decomp")


def extract(run_id: str, min_step: int) -> None:
    path = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id / "metrics.jsonl"
    assert path.exists(), f"no metrics.jsonl at {path}"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    pts = [
        (r["step"], r["train/perf/step_ms"])
        for r in rows
        if "train/perf/step_ms" in r and r.get("step", 0) >= min_step
    ]
    assert pts, f"no step_ms rows >= step {min_step} in {path}"
    steps = [s for s, _ in pts]
    ms = [m for _, m in pts]
    print(f"run {run_id}: steps {min(steps)}..{max(steps)} (n={len(ms)})")
    print(f"  step_ms mean={statistics.mean(ms):.1f}  median={statistics.median(ms):.1f}")
    print(f"  per-step: " + " ".join(f"{s}:{m:.0f}" for s, m in pts))


if __name__ == "__main__":
    run_id = sys.argv[1]
    min_step = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    extract(run_id, min_step)
