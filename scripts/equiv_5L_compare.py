"""Compare loss curves from the 3 equivalence-test runs.

Reads ``metrics.jsonl`` from three run dirs and prints / plots per-step
losses (faithfulness, importance-minimality, stoch-recon, ppgd-recon, total)
side by side.

Usage:
    python -m scripts.equiv_5L_compare <run_dir_1pool> <run_dir_2pool> <run_dir_3pool>

The run_dirs land under PARAM_DECOMP_OUT_DIR/decompositions/<run_id>/ —
each fresh-run's `_fresh_main` logs the path on startup.
"""

import json
import sys
from pathlib import Path


def _load_metrics(run_dir: Path) -> list[dict]:
    """Read newline-delimited metrics from ``{run_dir}/metrics.jsonl``."""
    path = run_dir / "metrics.jsonl"
    assert path.is_file(), f"no metrics.jsonl at {path}"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _key(metrics_row: dict, suffix: str) -> float | None:
    """Pick the loss key matching ``suffix`` (different strategies use slightly
    different key shapes; this finds the first match)."""
    for k, v in metrics_row.items():
        if k.endswith(suffix) and isinstance(v, int | float):
            return float(v)
    return None


def main(*run_dirs: str) -> None:
    assert len(run_dirs) == 3, "expected 3 run dirs: 1pool, 2pool, 3pool"
    labels = ["1pool", "2pool", "3pool"]
    runs = {label: _load_metrics(Path(d)) for label, d in zip(labels, run_dirs, strict=True)}

    # Show shape.
    for label, rows in runs.items():
        steps = sorted({r["step"] for r in rows})
        print(f"{label}: {len(rows)} log rows, steps {steps[:3]}...{steps[-3:]}")

    # For each strategy, print loss/* at a few representative steps.
    print()
    keys_of_interest = [
        "loss/total",
        "loss/FaithfulnessLoss",
        "loss/StochasticReconLayerwiseLoss",
        "loss/ImportanceMinimalityLoss",
        "loss/PersistentPGDReconLoss",
    ]
    sample_steps = [0, 50, 100, 150, 199]
    for step in sample_steps:
        print(f"=== step {step} ===")
        for label, rows in runs.items():
            row = next((r for r in rows if r["step"] == step), None)
            if row is None:
                print(f"  {label}: (no row at step {step})")
                continue
            vals = {k: _key(row, k.split("/")[-1]) for k in keys_of_interest}
            vals_str = " ".join(
                f"{k.split('/')[-1]}={v:.4g}" if v is not None else f"{k.split('/')[-1]}=MISS"
                for k, v in vals.items()
            )
            print(f"  {label}: {vals_str}")
        print()


if __name__ == "__main__":
    main(*sys.argv[1:])
