"""Diff the Track-2 quality bundle between the baseline(s) and one or more variants.

Reads each run's `metrics.jsonl` and compares every `QUALITY_BUNDLE` metric (see
`quality_bundle.py`), grouped by tier, then emits a faithfulness-gated overall verdict.
This only *reads* artifacts — it never touches the eval harness or baselines.

    python -m param_decomp_lab.speedup.compare_runs <baseline> <variant> [<variant> ...] \
        [--tol_pct 2.0] [--at_step N] [--out report.md]

The **baseline may be several seed runs** (comma-separated, e.g. `p-aaa,p-bbb,p-ccc`): the
per-metric band is then the **observed seed spread** (widened to at least `tol_pct`), which
is the noise floor. With a single baseline there's no spread, so the band is just
`±tol_pct`.

Pass `--at_step N` to compare both at the step closest to N — judge an experiment *early*
(partway through training) against the baseline at the same step, rather than waiting for
full convergence. Slow eval metrics only appear on slow-eval steps, so pick an `at_step`
on one (a multiple of the config's `slow_every`).

Decision rule (see track2/README.md): faithfulness (guardrail tier) is a gate — if any
guardrail metric regresses past the band, it's a FAIL regardless of the rest. Otherwise the
**primary** tier drives the verdict (WIN if a primary improves with none regressed).
Secondary metrics are informational.

Each run arg is a run_id (resolved to PARAM_DECOMP_OUT_DIR/runs/<id>/metrics.jsonl), a
run directory, or a direct path to a metrics.jsonl.
"""

import json
import statistics
from pathlib import Path

import fire
import yaml

from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.speedup.quality_bundle import QUALITY_BUNDLE, BundleMetric, Tier


def _resolve_run_dir(run: str) -> Path | None:
    p = Path(run)
    if p.is_file():
        return p.parent
    if p.is_dir():
        return p
    candidate = PARAM_DECOMP_OUT_DIR / "runs" / run
    return candidate if candidate.is_dir() else None


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
    """Value for each key, at the last logged step (default) or nearest `at_step`.

    Train and eval log as separate lines at the same step, so each key resolves to its own
    nearest step independently.
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
        cands = [
            r for r in rows if r.get("step") is not None and isinstance(r.get(k), (int, float))
        ]
        if not cands:
            continue
        chosen = min(cands, key=lambda r: abs(r["step"] - at_step))
        out[k] = float(chosen[k])
    return out


def _band(seed_values: list[float], *, tol_pct: float) -> tuple[float, float, float]:
    """(center, lo, hi). With ≥2 seeds the half-band is the seed half-range, floored at
    tol_pct; with one value it's just ±tol_pct."""
    center = statistics.fmean(seed_values)
    floor = tol_pct / 100.0 * abs(center)
    half = (
        max((max(seed_values) - min(seed_values)) / 2.0, floor) if len(seed_values) >= 2 else floor
    )
    return center, center - half, center + half


def _verdict(variant: float, lo: float, hi: float, *, lower_is_better: bool) -> str:
    if lo <= variant <= hi:
        return "within-band"
    better = variant < lo if lower_is_better else variant > hi
    return "improved" if better else "REGRESSED"


def _overall(verdicts: dict[str, str]) -> str:
    """Faithfulness-gated, primary-driven overall call."""
    by_tier: dict[Tier, list[str]] = {"primary": [], "secondary": [], "guardrail": []}
    for m in QUALITY_BUNDLE:
        if m.key in verdicts:
            by_tier[m.tier].append(verdicts[m.key])
    if "REGRESSED" in by_tier["guardrail"]:
        return "FAIL — faithfulness (guardrail) regressed"
    if "REGRESSED" in by_tier["primary"]:
        return "FAIL — primary (PPGD/L0) regressed"
    note = " (secondary regressed — check)" if "REGRESSED" in by_tier["secondary"] else ""
    if "improved" in by_tier["primary"]:
        return f"WIN — primary improved, faithfulness held{note}"
    return f"NEUTRAL — within band{note}"


def _table(
    seeds: list[dict[str, float]], var: dict[str, float], *, tol_pct: float
) -> tuple[str, dict[str, str]]:
    header = "| tier | metric | baseline | variant | Δ% | verdict |\n|---|---|---|---|---|---|\n"
    rows: list[str] = []
    verdicts: dict[str, str] = {}
    ordered: list[tuple[Tier, BundleMetric]] = [
        (t, m) for t in ("primary", "secondary", "guardrail") for m in QUALITY_BUNDLE if m.tier == t
    ]
    for tier, m in ordered:
        sv = [s[m.key] for s in seeds if m.key in s]
        v = var.get(m.key)
        if not sv or v is None:
            rows.append(f"| {tier} | {m.label} | — | — | n/a | missing |")
            continue
        center, lo, hi = _band(sv, tol_pct=tol_pct)
        verdict = _verdict(v, lo, hi, lower_is_better=m.lower_is_better)
        verdicts[m.key] = verdict
        pct = "n/a" if center == 0 else f"{100.0 * (v - center) / abs(center):+.2f}%"
        base_s = f"{center:.5g}" + (f" [{min(sv):.4g},{max(sv):.4g}]" if len(sv) > 1 else "")
        rows.append(f"| {tier} | {m.label} | {base_s} | {v:.5g} | {pct} | {verdict} |")
    return header + "\n".join(rows) + "\n", verdicts


def _eval_config(run: str) -> object | None:
    d = _resolve_run_dir(run)
    if d is None:
        return None
    cfg = d / "experiment_config.yaml"
    if not cfg.is_file():
        return None
    loaded = yaml.safe_load(cfg.read_text())
    return loaded.get("eval") if isinstance(loaded, dict) else None


def _eval_parity_note(baseline_ref: str, variant: str) -> str:
    """Warn if the variant's eval block differs from the baseline's — the eval config (incl.
    the PGD-attack strength) is part of the ruler; a mismatch invalidates the comparison."""
    b, v = _eval_config(baseline_ref), _eval_config(variant)
    if b is None or v is None:
        return "_eval-config parity: not checked (a config is unavailable — e.g. a W&B-cached baseline)._\n"
    if b == v:
        return "_eval-config parity: OK (variant matches baseline)._\n"
    return (
        "**⚠ eval-config MISMATCH vs baseline — comparison may be invalid; the `eval:` block "
        "(incl. PGD-attack n_steps/step_size, eval batch, cadence) is part of the ruler and must "
        "match.**\n"
    )


def compare(
    baseline: str,
    *variants: str,
    tol_pct: float = 2.0,
    at_step: int | None = None,
    out: str | None = None,
) -> None:
    assert variants, "pass at least one variant run after the baseline"
    keys = [m.key for m in QUALITY_BUNDLE]
    baseline_runs = [b for b in baseline.split(",") if b]
    seeds = [_values(_resolve_metrics_path(b), keys, at_step=at_step) for b in baseline_runs]

    when = "final step" if at_step is None else f"step ≈ {at_step}"
    band_src = (
        f"{len(seeds)}-seed spread (floor ±{tol_pct:.1f}%)"
        if len(seeds) > 1
        else f"±{tol_pct:.1f}% (single baseline — no spread)"
    )
    sections = [
        f"# Quality-bundle comparison ({when})\n",
        f"Baseline: `{baseline}` | band: {band_src}\n",
    ]
    for variant in variants:
        var_vals = _values(_resolve_metrics_path(variant), keys, at_step=at_step)
        table, verdicts = _table(seeds, var_vals, tol_pct=tol_pct)
        sections.append(
            f"## Variant: `{variant}`\n\n"
            + _eval_parity_note(baseline_runs[0], variant)
            + f"\n**Overall: {_overall(verdicts)}**\n\n"
            + table
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
