"""Diff the Track-2 quality bundle between a baseline run and one or more variants.

Reads each run's `metrics.jsonl` and compares the final logged value of every
`QUALITY_BUNDLE` metric (see `quality_bundle.py`). Emits one markdown table per
variant. This only *reads* artifacts — it never touches the eval harness or baselines.

    python -m param_decomp_lab.speedup.compare_runs <baseline> <variant> [<variant> ...] \
        [--tol_pct 2.0] [--out report.md]

Each run arg is a run_id (resolved to PARAM_DECOMP_OUT_DIR/runs/<id>/metrics.jsonl), a
run directory, or a direct path to a metrics.jsonl.
"""

import json
from pathlib import Path

import fire

from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.speedup.quality_bundle import QUALITY_BUNDLE


def _resolve_metrics_path(run: str) -> Path:
    p = Path(run)
    if p.is_file():
        return p
    if p.is_dir():
        return p / "metrics.jsonl"
    candidate = PARAM_DECOMP_OUT_DIR / "runs" / run / "metrics.jsonl"
    assert candidate.is_file(), f"no metrics.jsonl for run {run!r} (looked at {candidate})"
    return candidate


def _final_values(metrics_path: Path, keys: list[str]) -> dict[str, float]:
    """Last logged value for each requested key (keys are logged sparsely across steps)."""
    wanted = set(keys)
    latest: dict[str, float] = {}
    with open(metrics_path) as f:
        for line in f:
            row = json.loads(line)
            for k in wanted & row.keys():
                v = row[k]
                if isinstance(v, (int, float)):
                    latest[k] = float(v)
    return latest


def _verdict(
    base: float | None, var: float | None, *, lower_is_better: bool, tol_pct: float
) -> str:
    if base is None or var is None:
        return "missing"
    if base == 0:
        return "base=0"
    pct = 100.0 * (var - base) / abs(base)
    improved = pct < 0 if lower_is_better else pct > 0
    if abs(pct) <= tol_pct:
        return "within-band"
    return "improved" if improved else "REGRESSED"


def _table(base: dict[str, float], var: dict[str, float], *, tol_pct: float) -> str:
    header = "| metric | baseline | variant | Δ% | verdict |\n|---|---|---|---|---|\n"
    rows: list[str] = []
    for m in QUALITY_BUNDLE:
        b = base.get(m.key)
        v = var.get(m.key)
        pct = "n/a" if (b is None or v is None or b == 0) else f"{100.0 * (v - b) / abs(b):+.2f}%"
        verdict = _verdict(b, v, lower_is_better=m.lower_is_better, tol_pct=tol_pct)
        b_s = "—" if b is None else f"{b:.5g}"
        v_s = "—" if v is None else f"{v:.5g}"
        rows.append(f"| {m.label} | {b_s} | {v_s} | {pct} | {verdict} |")
    return header + "\n".join(rows) + "\n"


def compare(
    baseline: str,
    *variants: str,
    tol_pct: float = 2.0,
    out: str | None = None,
) -> None:
    assert variants, "pass at least one variant run after the baseline"
    base_path = _resolve_metrics_path(baseline)
    base_vals = _final_values(base_path, [m.key for m in QUALITY_BUNDLE])

    sections = [f"# Quality-bundle comparison\n\nBaseline: `{baseline}` (`{base_path}`)\n"]
    sections.append(
        f"Within-band tolerance: ±{tol_pct:.1f}% (replace with baseline seed spread once locked)\n"
    )
    for variant in variants:
        var_path = _resolve_metrics_path(variant)
        var_vals = _final_values(var_path, [m.key for m in QUALITY_BUNDLE])
        sections.append(
            f"## Variant: `{variant}`\n\n`{var_path}`\n\n"
            + _table(base_vals, var_vals, tol_pct=tol_pct)
        )

    report = "\n".join(sections)
    print(report)
    if out is not None:
        Path(out).write_text(report)
        print(f"\nWrote report to {out}")


def cli() -> None:
    fire.Fire(compare)


if __name__ == "__main__":
    cli()
