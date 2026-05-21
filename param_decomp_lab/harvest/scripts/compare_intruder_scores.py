"""Export intruder detection scores across decomposition methods.

Reads harvest DBs for each model in `scripts/intruder_comparison.json` and writes:
- `data.json`: per-component scores + densities, per-model summary stats, group/colour spec
- `README.md`: description of the data, summary table, and intended bar chart

Output goes to `PARAM_DECOMP_OUT_DIR/intruder_comparison/`.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np

from param_decomp.settings import PARAM_DECOMP_OUT_DIR

HARVEST_ROOT = PARAM_DECOMP_OUT_DIR / "harvest"
OUT_DIR = PARAM_DECOMP_OUT_DIR / "intruder_comparison"
CONFIG_PATH = Path(__file__).resolve().parents[3] / "scripts" / "intruder_comparison.json"

EMBER = "#B17039"
GREY = "#B8B3A6"
OBSIDIAN = "#2C2B2C"


def load_scores(decomp_id: str, subrun: str) -> tuple[np.ndarray, np.ndarray]:
    db_path = HARVEST_ROOT / decomp_id / subrun / "harvest.db"
    assert db_path.exists(), f"No harvest DB at {db_path}"
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        """
        SELECT c.firing_density, s.score
        FROM scores s
        JOIN components c ON s.component_key = c.component_key
        WHERE s.score_type = 'intruder'
        ORDER BY c.firing_density
        """
    ).fetchall()
    conn.close()
    assert rows, f"No intruder scores in {db_path}"
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


def summarize(scores: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(scores)),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "p25": float(np.percentile(scores, 25)),
        "p75": float(np.percentile(scores, 75)),
    }


def group_mean(data: dict[str, np.ndarray], members: list[str]) -> float | None:
    present = [m for m in members if m in data]
    if not present:
        return None
    return float(np.concatenate([data[m] for m in present]).mean())


def write_readme(
    path: Path,
    models: dict[str, list[str]],
    per_model: dict[str, dict[str, float]],
    group_means: dict[str, float],
    group_colors: dict[str, str],
) -> None:
    lines: list[str] = []
    lines.append("# Intruder Score Comparison\n")
    lines.append(
        "Per-component intruder detection scores from harvest DBs, grouped by "
        "decomposition method (CLT / TC / VPD across configurations).\n"
    )
    lines.append("## Files\n")
    lines.append("- `data.json` — raw per-component data + summary stats + plot spec")
    lines.append("- `README.md` — this file\n")

    lines.append("## `data.json` schema\n")
    lines.append("```")
    lines.append("{")
    lines.append('  "models": {')
    lines.append('    "<label>": {')
    lines.append('      "decomposition_id": str,')
    lines.append('      "harvest_subrun_id": str,')
    lines.append('      "densities": [float, ...],   # firing density per component')
    lines.append('      "scores":    [float, ...],   # intruder score per component (0..1)')
    lines.append('      "summary":   {n, mean, median, p25, p75}')
    lines.append("    }, ...")
    lines.append("  },")
    lines.append('  "groups": {')
    lines.append('    "<group_label>": {')
    lines.append('      "members": [<model label>, ...],   # subset of models')
    lines.append('      "color":   str,                    # hex')
    lines.append('      "mean":    float                   # mean over concatenated members')
    lines.append("    }, ...")
    lines.append("  },")
    lines.append('  "palette": {"ember": str, "grey": str, "obsidian": str}')
    lines.append("}")
    lines.append("```\n")

    lines.append("## Per-model summary\n")
    lines.append(
        f"| {'Model':<28} | {'N':>6} | {'Mean':>6} | {'Median':>6} | {'p25':>6} | {'p75':>6} |"
    )
    lines.append(f"| {'-' * 28} | {'-' * 6} | {'-' * 6} | {'-' * 6} | {'-' * 6} | {'-' * 6} |")
    for label in models:
        s = per_model[label]
        lines.append(
            f"| {label:<28} | {int(s['n']):>6d} | {s['mean']:>6.3f} | "
            f"{s['median']:>6.3f} | {s['p25']:>6.3f} | {s['p75']:>6.3f} |"
        )
    lines.append("")

    lines.append("## Per-group means\n")
    lines.append(f"| {'Group':<14} | {'Mean':>6} | Color |")
    lines.append(f"| {'-' * 14} | {'-' * 6} | {'-' * 7} |")
    for g, mean in group_means.items():
        single_line_label = g.replace("\n", " ")
        lines.append(f"| {single_line_label:<14} | {mean:>6.3f} | `{group_colors[g]}` |")
    lines.append("")

    lines.append("## Intended plot\n")
    lines.append(
        "Grouped bar chart: one bar per group (x-axis), height = mean intruder score over all "
        "components in that group's models. Bars use the colours under `groups[*].color` "
        "(VPD groups in **ember**, all others in **grey**), with an obsidian edge. "
        'Add a horizontal dotted line at 0.2 labelled "Random (1/5)". Y-axis: 0 to 1, '
        'labelled "Mean intruder score". Title: "Intruder Score by Decomposition Method". '
        "Group labels can use `\\n` for two-line x-tick labels."
    )

    path.write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    models: dict[str, list[str]] = cfg["models"]
    groups: dict[str, list[str]] = cfg["groups"]

    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, (decomp_id, subrun) in models.items():
        raw[label] = load_scores(decomp_id, subrun)
        print(f"{label:28s}: {len(raw[label][0]):6d} components")

    per_model: dict[str, dict[str, float]] = {
        label: summarize(scores) for label, (_, scores) in raw.items()
    }

    score_only = {label: scores for label, (_, scores) in raw.items()}
    group_colors = {g: EMBER if "VPD" in g else GREY for g in groups}
    group_means: dict[str, float] = {}
    for g, members in groups.items():
        m = group_mean(score_only, members)
        if m is not None:
            group_means[g] = m

    out: dict[str, object] = {
        "models": {
            label: {
                "decomposition_id": models[label][0],
                "harvest_subrun_id": models[label][1],
                "densities": densities.tolist(),
                "scores": scores.tolist(),
                "summary": per_model[label],
            }
            for label, (densities, scores) in raw.items()
        },
        "groups": {
            g: {
                "members": [m for m in members if m in raw],
                "color": group_colors[g],
                "mean": group_means[g],
            }
            for g, members in groups.items()
            if g in group_means
        },
        "palette": {"ember": EMBER, "grey": GREY, "obsidian": OBSIDIAN},
    }

    data_path = OUT_DIR / "data.json"
    with open(data_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {data_path}")

    readme_path = OUT_DIR / "README.md"
    write_readme(readme_path, models, per_model, group_means, group_colors)
    print(f"Saved {readme_path}")


if __name__ == "__main__":
    main()
