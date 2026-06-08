"""Diff the Track-2 quality bundle between a baseline run and one or more variants.

Reads each run's `metrics.jsonl` and compares every `QUALITY_BUNDLE` metric (see
`quality_bundle.py`), grouped by tier (primary / secondary / guardrail). This only *reads*
artifacts — it never touches the eval harness or baselines.

    python -m param_decomp_lab.speedup.compare_runs <baseline> <variant> [<variant> ...] \
        [--tol_pct 2.0] [--at_step N] [--out report.md]

By default each metric is taken at its **last** logged step. Pass `--at_step N` to compare
both runs at the step closest to N instead — the key to judging an experiment *early*
(e.g. partway through the 4L run) rather than waiting for full training, using the
baseline's own trajectory at the same step. Slow eval metrics only appear on slow-eval
steps, so pick an `at_step` that lands on one (a multiple of the config's `slow_every`).

Each run arg is a run_id (resolved to PARAM_DECOMP_OUT_DIR/runs/<id>/metrics.jsonl), a
run directory, or a direct path to a metrics.jsonl.
"""

import json
from pathlib import Path

import fire

from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.speedup.quality_bundle import QUALITY_BUNDLE, Tier


def _resolve_metrics_path(run: str) -> Path:
    p = Path(run)
    if p.is_file():
        return p
    if p.is_dir():
        return p / "metrics.jsonl"
    candidate = PARAM_DECOMP_OUT_DIR / "runs" / run / "metrics.jsonl"
    assert candidate.is_file(), f"no metrics.jsonl for run {run!r} (looked at {candidate})"
    return candidate


def _values(metrics_path: Path, keys: list[str], *, at_step: int | None) -> dict[str, float]:
    """Value for each key, either at the last logged step (default) or nearest `at_step`.

    Train and eval log as separate lines at the same step, so for `at_step` we merge all
    rows at the chosen step. Keys are logged sparsely, so each key resolves to its own
    nearest step.
    """
    wanted = set(keys)
    with open(metrics_path) as f:
        rows = [json.loads(line) for line in f]
    if at_step is None:
        latest: dict[str, float] = {}
        for row in rows:
            for k in wanted & row.keys():
                if isinstance(row[k], (int, float)):
                    latest[k] = float(row[k])
        return latest
    out: dict[str, float] = {}
    for k in wanted:
        steps = [
            r["step"]
            for r in rows
            if r.get("step") is not None and isinstance(r.get(k), (int, float))
        ]
        if not steps:
            continue
        chosen = min(steps, key=lambda s: abs(s - at_step))
        out[k] = float(
            next(
                r[k] for r in rows if r.get("step") == chosen and isinstance(r.get(k), (int, float))
            )
        )
    return out


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
    header = "| tier | metric | baseline | variant | Δ% | verdict |\n|---|---|---|---|---|---|\n"
    rows: list[str] = []
    tiers: list[Tier] = ["primary", "secondary", "guardrail"]
    for tier in tiers:
        for m in (m for m in QUALITY_BUNDLE if m.tier == tier):
            b = base.get(m.key)
            v = var.get(m.key)
            pct = (
                "n/a" if (b is None or v is None or b == 0) else f"{100.0 * (v - b) / abs(b):+.2f}%"
            )
            verdict = _verdict(b, v, lower_is_better=m.lower_is_better, tol_pct=tol_pct)
            b_s = "—" if b is None else f"{b:.5g}"
            v_s = "—" if v is None else f"{v:.5g}"
            rows.append(f"| {tier} | {m.label} | {b_s} | {v_s} | {pct} | {verdict} |")
    return header + "\n".join(rows) + "\n"


def compare(
    baseline: str,
    *variants: str,
    tol_pct: float = 2.0,
    at_step: int | None = None,
    out: str | None = None,
) -> None:
    assert variants, "pass at least one variant run after the baseline"
    keys = [m.key for m in QUALITY_BUNDLE]
    base_path = _resolve_metrics_path(baseline)
    base_vals = _values(base_path, keys, at_step=at_step)

    when = "final step" if at_step is None else f"step ≈ {at_step}"
    sections = [
        f"# Quality-bundle comparison ({when})\n\nBaseline: `{baseline}` (`{base_path}`)\n",
        f"Within-band tolerance: ±{tol_pct:.1f}% (fixed; baselines are single-seed — no spread).\n",
    ]
    for variant in variants:
        var_path = _resolve_metrics_path(variant)
        var_vals = _values(var_path, keys, at_step=at_step)
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
