"""Generate a markdown summary report from a set of WandB sweep runs.

Usage:
    python spd/scripts/sweep_summary/sweep_summary_stats.py s-e8bde534 s-73d2385c ...
    python spd/scripts/sweep_summary/sweep_summary_stats.py s-e8bde534 ... --name my_sweep
    python spd/scripts/sweep_summary/sweep_summary_stats.py s-e8bde534 ... --stdout

Results are saved to spd/scripts/sweep_summary/out/<name>/.
"""

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import wandb

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

CE_KL_KEYS = [
    "eval/ce_kl/kl_unmasked",
    "eval/ce_kl/kl_stoch_masked",
    "eval/ce_kl/kl_ci_masked",
    "eval/ce_kl/kl_rounded_masked",
    "eval/ce_kl/kl_random_masked",
    "eval/ce_kl/kl_zero_masked",
    "eval/ce_kl/ce_difference_unmasked",
    "eval/ce_kl/ce_difference_stoch_masked",
    "eval/ce_kl/ce_difference_ci_masked",
    "eval/ce_kl/ce_difference_rounded_masked",
    "eval/ce_kl/ce_difference_random_masked",
    "eval/ce_kl/ce_unrecovered_unmasked",
    "eval/ce_kl/ce_unrecovered_stoch_masked",
    "eval/ce_kl/ce_unrecovered_ci_masked",
    "eval/ce_kl/ce_unrecovered_rounded_masked",
    "eval/ce_kl/ce_unrecovered_random_masked",
]

EVAL_LOSS_KEYS = [
    "eval/loss/StochasticReconSubsetLoss",
    "eval/loss/PGDReconLoss",
    "eval/loss/StochasticHiddenActsReconLoss",
    "eval/loss/CIHiddenActsReconLoss",
    "eval/loss/FaithfulnessLoss",
    "eval/loss/ImportanceMinimalityLoss",
]

L0_LAYER_KEYS = [
    "eval/l0/0.0_layer_0",
    "eval/l0/0.0_layer_1",
    "eval/l0/0.0_layer_2",
    "eval/l0/0.0_layer_3",
    "eval/l0/0.0_total",
]

TRAIN_LOSS_KEYS = [
    "train/loss/total",
    "train/loss/FaithfulnessLoss",
    "train/loss/ImportanceMinimalityLoss",
    "train/loss/StochasticReconSubsetLoss",
    "train/loss/PersistentPGDReconLoss",
]

MODULES = [
    "h.0.attn.q_proj",
    "h.0.attn.k_proj",
    "h.0.attn.v_proj",
    "h.0.attn.o_proj",
    "h.0.mlp.c_fc",
    "h.0.mlp.down_proj",
    "h.1.attn.q_proj",
    "h.1.attn.k_proj",
    "h.1.attn.v_proj",
    "h.1.attn.o_proj",
    "h.1.mlp.c_fc",
    "h.1.mlp.down_proj",
    "h.2.attn.q_proj",
    "h.2.attn.k_proj",
    "h.2.attn.v_proj",
    "h.2.attn.o_proj",
    "h.2.mlp.c_fc",
    "h.2.mlp.down_proj",
    "h.3.attn.q_proj",
    "h.3.attn.k_proj",
    "h.3.attn.v_proj",
    "h.3.attn.o_proj",
    "h.3.mlp.c_fc",
    "h.3.mlp.down_proj",
]

LAYER_PATTERN = re.compile(r"h\.(\d+)\.")


def _layer_of(module: str) -> int:
    m = LAYER_PATTERN.search(module)
    assert m, f"Cannot extract layer from {module}"
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _fmt(v: float | np.floating) -> str:
    if v == 0:
        return "0"
    abs_v = abs(v)
    if abs_v < 1e-10:
        return f"{v:.14f}"
    if abs_v < 1e-6:
        return f"{v:.10f}"
    if abs_v < 0.0001:
        return f"{v:.8f}"
    if abs_v < 0.01:
        return f"{v:.6f}"
    if abs_v < 1:
        return f"{v:.4f}"
    if abs_v < 100:
        return f"{v:.2f}"
    return f"{v:.1f}"


def _short(key: str) -> str:
    """Extract a short display name for table column headers."""
    for prefix in [
        "eval/ce_kl/",
        "eval/loss/",
        "eval/l0/0.0_",
        "train/loss/",
        "train/l0/",
    ]:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _summary_name(key: str) -> str:
    """Extract a descriptive name for the plain-text summary list."""
    if key.startswith("eval/l0/0.0_"):
        return "L0 " + key[len("eval/l0/0.0_") :]
    if key.startswith("train/loss/"):
        return "train " + key[len("train/loss/") :]
    for prefix in ["eval/ce_kl/", "eval/loss/"]:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _raw_table(
    seeds: list[int],
    keys: list[str],
    data: dict[int, dict[str, float]],
) -> str:
    headers = ["seed"] + [_short(k) for k in keys]
    rows = []
    for s in seeds:
        row = [str(s)]
        for k in keys:
            v = data[s].get(k)
            row.append(_fmt(v) if v is not None else "—")
        rows.append(row)
    return _md_table(headers, rows)


def _summary_table(
    seeds: list[int],
    keys: list[str],
    data: dict[int, dict[str, float]],
) -> str:
    headers = ["stat"] + [_short(k) for k in keys]
    mean_row = ["mean"]
    std_row = ["std"]
    for k in keys:
        vals = [data[s][k] for s in seeds if data[s].get(k) is not None]
        if vals:
            mean_row.append(_fmt(np.mean(vals)))
            std_row.append(_fmt(np.std(vals)))
        else:
            mean_row.append("—")
            std_row.append("—")
    return _md_table(headers, [mean_row, std_row])


def _cross_layer_summary(
    seeds: list[int],
    module_keys: list[str],
    data: dict[int, dict[str, float]],
) -> str:
    """Mean across modules within each layer, then mean/std across seeds."""
    # Extract module name from each key for layer grouping
    key_to_module = {}
    for k in module_keys:
        for m in MODULES:
            if k.endswith(m):
                key_to_module[k] = m
                break

    layers = sorted(set(_layer_of(key_to_module[k]) for k in module_keys if k in key_to_module))
    layer_keys: dict[int, list[str]] = defaultdict(list)
    for k in module_keys:
        if k in key_to_module:
            layer_keys[_layer_of(key_to_module[k])].append(k)

    headers = ["stat"] + [f"layer {ly}" for ly in layers] + ["all"]
    per_seed_layer: dict[int, dict[int, float]] = {}
    per_seed_all: dict[int, float] = {}
    for s in seeds:
        per_seed_layer[s] = {}
        all_vals = []
        for ly in layers:
            vals = [data[s][k] for k in layer_keys[ly] if data[s].get(k) is not None]
            per_seed_layer[s][ly] = float(np.mean(vals)) if vals else float("nan")
            all_vals.extend(vals)
        per_seed_all[s] = float(np.mean(all_vals)) if all_vals else float("nan")

    mean_row = ["mean"]
    std_row = ["std"]
    for ly in layers:
        vals = [per_seed_layer[s][ly] for s in seeds]
        mean_row.append(_fmt(np.mean(vals)))
        std_row.append(_fmt(np.std(vals)))
    all_vals = [per_seed_all[s] for s in seeds]
    mean_row.append(_fmt(np.mean(all_vals)))
    std_row.append(_fmt(np.std(all_vals)))

    return _md_table(headers, [mean_row, std_row])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@dataclass
class TargetModelInfo:
    wandb_path: str
    model_class: str
    architecture: dict[str, object]
    tokenizer: str
    train_loss: float
    val_loss: float
    train_dataset: str
    train_steps: int
    val_loss_curve: list[tuple[int, float]]  # (step, val_loss) sorted by step


def _fetch_target_model_info(
    pretrained_model_name: str,
    pretrained_model_class: str,
    tokenizer_name: str,
    project: str,
) -> TargetModelInfo:
    api = wandb.Api()
    run = api.run(f"{project}/runs/{pretrained_model_name.split('/')[-1]}")
    history = list(run.scan_history(keys=["val_loss", "_step"], page_size=10000))
    val_loss_curve = sorted(
        {(int(h["_step"]), float(h["val_loss"])) for h in history if h.get("val_loss") is not None}
    )
    return TargetModelInfo(
        wandb_path=pretrained_model_name,
        model_class=pretrained_model_class,
        architecture=run.config.get("model", {}),
        tokenizer=tokenizer_name,
        train_loss=float(run.summary.get("train_loss", float("nan"))),
        val_loss=float(run.summary.get("val_loss", float("nan"))),
        train_dataset=run.config.get("train_dataset_config", {}).get("name", "unknown"),
        train_steps=int(run.summary.get("_step", 0)),
        val_loss_curve=val_loss_curve,
    )


def _smooth_val_curve(
    curve: list[tuple[int, float]],
    alpha: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth val loss data with a forward-only EMA.

    Args:
        alpha: EMA smoothing factor. Higher = less smoothing, more responsive.
            0.3 gives a ~3-point effective window, smoothing noise without
            flattening the steep early drop.
    """
    steps = np.array([s for s, _ in curve], dtype=np.float64)
    losses = np.array([v for _, v in curve], dtype=np.float64)

    # Run EMA forward, then backward, and average. This eliminates the
    # initialisation lag that makes a forward-only EMA overshoot early points
    # (where the curve drops steeply).
    fwd = np.empty_like(losses)
    fwd[0] = losses[0]
    for i in range(1, len(losses)):
        fwd[i] = alpha * losses[i] + (1 - alpha) * fwd[i - 1]

    bwd = np.empty_like(losses)
    bwd[-1] = losses[-1]
    for i in range(len(losses) - 2, -1, -1):
        bwd[i] = alpha * losses[i] + (1 - alpha) * bwd[i + 1]

    smoothed = (fwd + bwd) / 2
    return steps, smoothed


def _compute_recovered_pct(target_info: TargetModelInfo, spd_ce: float) -> float | None:
    """Find what % through target training had the same val loss as spd_ce.

    Smooths the target model's val loss history with a bidirectional EMA, then
    interpolates to find the step where loss equals spd_ce.
    Returns None if the SPD CE is worse than the earliest logged val loss.
    """
    curve = target_info.val_loss_curve
    assert len(curve) >= 2
    total_steps = target_info.train_steps

    steps, smoothed = _smooth_val_curve(curve)

    # If SPD is better than final target, return 100%
    if spd_ce <= smoothed[-1]:
        return 100.0

    # If SPD is worse than the very first checkpoint, return None
    if spd_ce >= smoothed[0]:
        return None

    # Interpolate on the smoothed monotone curve
    for i in range(len(steps) - 1):
        if smoothed[i] >= spd_ce >= smoothed[i + 1]:
            if smoothed[i] == smoothed[i + 1]:
                step = float(steps[i])
            else:
                frac = (smoothed[i] - spd_ce) / (smoothed[i] - smoothed[i + 1])
                step = float(steps[i] + frac * (steps[i + 1] - steps[i]))
            return step / total_steps * 100.0

    return None


@dataclass
class SPDConfig:
    """Relevant SPD run config fields."""

    module_info: list[dict[str, Any]]  # [{module_pattern, C}, ...]

    def layer_component_counts(self) -> dict[int, int]:
        """Total C per layer, expanding wildcard patterns across 4 layers."""
        layer_c: dict[int, int] = defaultdict(int)
        for mi in self.module_info:
            pattern = str(mi["module_pattern"])
            c = int(mi["C"])
            if "*" in pattern:
                for layer_idx in range(4):
                    layer_c[layer_idx] += c
            else:
                m = LAYER_PATTERN.search(pattern)
                assert m, f"Cannot extract layer from {pattern}"
                layer_c[int(m.group(1))] += c
        return dict(layer_c)


def _fetch_n_alive_from_harvest(run_id: str) -> dict[str, int] | None:
    """Try to get per-module alive counts from the harvest DB. Returns None if unavailable."""
    from spd.settings import SPD_OUT_DIR

    harvest_parent = SPD_OUT_DIR / "harvest" / run_id
    if not harvest_parent.exists():
        return None
    # Find the harvest subdirectory
    subdirs = [d for d in harvest_parent.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    db_path = subdirs[0] / "harvest.db"
    if not db_path.exists():
        return None

    import sqlite3

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT layer, COUNT(*), "
        "SUM(CASE WHEN firing_density > 0 THEN 1 ELSE 0 END) "
        "FROM components GROUP BY layer"
    ).fetchall()
    conn.close()

    return {layer: int(alive) for layer, _total, alive in rows}


def fetch_runs(
    run_ids: list[str], project: str
) -> tuple[
    list[int], dict[int, dict[str, float]], TargetModelInfo, SPDConfig, dict[str, int] | None
]:
    api = wandb.Api()
    seeds: list[int] = []
    data: dict[int, dict[str, float]] = {}
    first_config: dict[str, Any] | None = None
    first_run_id: str | None = None
    for rid in run_ids:
        run = api.run(f"{project}/runs/{rid}")
        if first_config is None:
            first_config = run.config
            first_run_id = rid
        seed = run.config["seed"]
        seeds.append(seed)
        summary = {}
        for k, v in run.summary.items():
            if isinstance(v, (int, float)):
                summary[k] = float(v)
        data[seed] = summary
    seeds.sort()
    assert first_config is not None
    assert first_run_id is not None
    target_info = _fetch_target_model_info(
        pretrained_model_name=str(first_config["pretrained_model_name"]),
        pretrained_model_class=str(first_config["pretrained_model_class"]),
        tokenizer_name=str(first_config["tokenizer_name"]),
        project=project,
    )
    spd_config = SPDConfig(module_info=first_config["module_info"])
    n_alive = _fetch_n_alive_from_harvest(first_run_id)
    return seeds, data, target_info, spd_config, n_alive


def _per_module_keys(prefix: str) -> list[str]:
    return [f"{prefix}/{m}" for m in MODULES]


LATEX_MODE_LABELS = {
    "unmasked": r"Unmasked (All masks$=$1)",
    "stoch_masked": "Stochastic masks",
    "ci_masked": "CIs used as masks",
    "rounded_masked": r"Rounded masks (mask$=$1 if CI$>$0)",
}


def _render_latex_summary(
    seeds: list[int],
    data: dict[int, dict[str, float]],
    target_info: TargetModelInfo,
    per_mode_pcts: dict[str, list[float]],
    spd_config: SPDConfig,
    n_alive: dict[str, int] | None,
) -> str:
    modes = ["unmasked", "stoch_masked", "ci_masked", "rounded_masked"]
    lines = [
        "\n## LaTeX Summary Table\n",
        "```latex",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{SPD decomposition quality by masking mode.}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Masking mode & CE loss & Training compute recovered (\%) \\",
        r"\midrule",
    ]

    target_val_ce = target_info.val_loss
    for mode in modes:
        label = LATEX_MODE_LABELS[mode]
        ce_diffs = [data[s].get(f"eval/ce_kl/ce_difference_{mode}") for s in seeds]
        ce_vals = [target_val_ce + d for d in ce_diffs if d is not None]
        pcts = per_mode_pcts.get(mode, [])

        ce_mean = _fmt(np.mean(ce_vals)) if ce_vals else "—"
        pct_mean = f"{np.mean(pcts):.1f}" if pcts else "—"

        lines.append(f"{label} & ${ce_mean}$ & ${pct_mean}$ \\\\")

    lines.extend(
        [
            r"\midrule",
            f"Target model & ${_fmt(target_val_ce)}$ & $100.0$ \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "```",
        ]
    )

    # Eval reconstruction losses table
    recon_keys = [
        ("StochasticReconSubsetLoss", "eval/loss/StochasticReconSubsetLoss"),
        ("PGDReconLoss", "eval/loss/PGDReconLoss"),
        ("StochasticHiddenActsReconLoss", "eval/loss/StochasticHiddenActsReconLoss"),
        ("CIHiddenActsReconLoss", "eval/loss/CIHiddenActsReconLoss"),
    ]
    lines.append("")
    lines.append("```latex")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Eval reconstruction losses.}")
    lines.append(r"\begin{tabular}{lc}")
    lines.append(r"\toprule")
    lines.append(r"Loss & Value \\")
    lines.append(r"\midrule")
    for label, key in recon_keys:
        vals = [data[s][key] for s in seeds if data[s].get(key) is not None]
        if vals:
            lines.append(f"{label} & ${_fmt(np.mean(vals))}$ \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("```")

    # L0 and component counts table
    layer_c = spd_config.layer_component_counts()
    has_alive = n_alive is not None
    # Group n_alive by layer
    layer_alive: dict[int, int] = defaultdict(int)
    if n_alive is not None:
        for mod, count in n_alive.items():
            m = LAYER_PATTERN.search(mod)
            if m:
                layer_alive[int(m.group(1))] += count

    if has_alive:
        col_spec = r"\begin{tabular}{lcccc}"
        header = r"Layer & $C$ & Alive & Mean L0 & L0 / $C$ (\%) \\"
    else:
        col_spec = r"\begin{tabular}{lccc}"
        header = r"Layer & $C$ & Mean L0 & L0 / $C$ (\%) \\"

    lines.append("")
    lines.append("```latex")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Sparsity: component counts and CI-L0 per layer.}")
    lines.append(col_spec)
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")

    for layer_idx in sorted(layer_c.keys()):
        total_c = layer_c[layer_idx]
        l0_key = f"eval/l0/0.0_layer_{layer_idx}"
        l0_vals = [data[s][l0_key] for s in seeds if data[s].get(l0_key) is not None]
        if l0_vals:
            mean_l0 = float(np.mean(l0_vals))
            pct = mean_l0 / total_c * 100
            if has_alive:
                alive_count = layer_alive.get(layer_idx, 0)
                lines.append(
                    f"Layer {layer_idx} & ${total_c}$ & ${alive_count}$ "
                    f"& ${_fmt(mean_l0)}$ & ${pct:.1f}$ \\\\"
                )
            else:
                lines.append(
                    f"Layer {layer_idx} & ${total_c}$ & ${_fmt(mean_l0)}$ & ${pct:.1f}$ \\\\"
                )

    # Total row
    total_c_all = sum(layer_c.values())
    total_l0_vals = [
        data[s]["eval/l0/0.0_total"] for s in seeds if data[s].get("eval/l0/0.0_total") is not None
    ]
    if total_l0_vals:
        mean_total_l0 = float(np.mean(total_l0_vals))
        pct_total = mean_total_l0 / total_c_all * 100
        lines.append(r"\midrule")
        if has_alive:
            total_alive = sum(layer_alive.values())
            lines.append(
                f"Total & ${total_c_all}$ & ${total_alive}$ "
                f"& ${_fmt(mean_total_l0)}$ & ${pct_total:.1f}$ \\\\"
            )
        else:
            lines.append(
                f"Total & ${total_c_all}$ & ${_fmt(mean_total_l0)}$ & ${pct_total:.1f}$ \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("```")

    # Training losses table
    train_keys = [
        ("Total", "train/loss/total"),
        ("FaithfulnessLoss", "train/loss/FaithfulnessLoss"),
        ("StochasticReconSubsetLoss", "train/loss/StochasticReconSubsetLoss"),
        ("PersistentPGDReconLoss", "train/loss/PersistentPGDReconLoss"),
        ("ImportanceMinimalityLoss", "train/loss/ImportanceMinimalityLoss"),
    ]
    lines.append("")
    lines.append("```latex")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Training losses (final step).}")
    lines.append(r"\begin{tabular}{lc}")
    lines.append(r"\toprule")
    lines.append(r"Loss & Value \\")
    lines.append(r"\midrule")
    for label, key in train_keys:
        vals = [data[s][key] for s in seeds if data[s].get(key) is not None]
        if vals:
            lines.append(f"{label} & ${_fmt(np.mean(vals))}$ \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("```")

    return "\n".join(lines)


def _render_target_model_section(info: TargetModelInfo) -> str:
    arch = info.architecture
    lines = [
        "## Target Model",
        "",
        f"WandB: `{info.wandb_path}`",
        f"Class: `{info.model_class}`",
        f"Tokenizer: `{info.tokenizer}`",
        f"Training dataset: `{info.train_dataset}`",
        f"Training steps: {info.train_steps:,}",
        f"Train CE loss: {_fmt(info.train_loss)}",
        f"Val CE loss: {_fmt(info.val_loss)}",
        "",
        "Architecture:",
        f"Layers: {arch.get('n_layer')}",
        f"Hidden dim: {arch.get('n_embd')}",
        f"Heads: {arch.get('n_head')}",
        f"MLP intermediate: {arch.get('n_intermediate')}",
        f"Context length: {arch.get('n_ctx')}",
        f"Vocab size: {arch.get('vocab_size')}",
    ]
    return "\n\n".join(lines)


def generate_report(
    seeds: list[int],
    data: dict[int, dict[str, float]],
    target_info: TargetModelInfo,
    spd_config: SPDConfig,
    n_alive: dict[str, int] | None,
) -> str:
    sections: list[str] = []

    sections.append(f"# Sweep Summary Report\n\n**Seeds**: {seeds}\n")
    sections.append(_render_target_model_section(target_info))

    # 1. CE/KL
    sections.append("## Output Quality (CE/KL)\n")
    sections.append("### Raw values\n")
    sections.append(_raw_table(seeds, CE_KL_KEYS, data))
    sections.append("\n### Summary\n")
    sections.append(_summary_table(seeds, CE_KL_KEYS, data))

    # 2. Eval losses (aggregate)
    sections.append("\n## Eval Reconstruction Losses (aggregate)\n")
    sections.append("### Raw values\n")
    sections.append(_raw_table(seeds, EVAL_LOSS_KEYS, data))
    sections.append("\n### Summary\n")
    sections.append(_summary_table(seeds, EVAL_LOSS_KEYS, data))

    # 3. Hidden acts recon per module
    for loss_name, prefix in [
        ("StochasticHiddenActsReconLoss", "eval/loss/StochasticHiddenActsReconLoss"),
        ("CIHiddenActsReconLoss", "eval/loss/CIHiddenActsReconLoss"),
    ]:
        mod_keys = _per_module_keys(prefix)
        sections.append(f"\n## {loss_name} (per module)\n")
        sections.append("### Raw values\n")
        sections.append(_raw_table(seeds, mod_keys, data))
        sections.append("\n### Summary across seeds\n")
        sections.append(_summary_table(seeds, mod_keys, data))
        sections.append("\n### Summary across modules per layer\n")
        sections.append(_cross_layer_summary(seeds, mod_keys, data))

    # 4. Sparsity
    sections.append("\n## Sparsity (CI-L0)\n")
    sections.append("### Per-layer (raw)\n")
    sections.append(_raw_table(seeds, L0_LAYER_KEYS, data))
    sections.append("\n### Per-layer (summary)\n")
    sections.append(_summary_table(seeds, L0_LAYER_KEYS, data))

    l0_mod_keys = [f"eval/l0/0.0_{m}" for m in MODULES]
    sections.append("\n### Per-module (raw)\n")
    sections.append(_raw_table(seeds, l0_mod_keys, data))
    sections.append("\n### Per-module (summary across seeds)\n")
    sections.append(_summary_table(seeds, l0_mod_keys, data))
    sections.append("\n### Mean L0 per layer across seeds\n")
    sections.append(_cross_layer_summary(seeds, l0_mod_keys, data))

    # 5. Training losses
    sections.append("\n## Training Losses (final step)\n")
    sections.append("### Raw values\n")
    sections.append(_raw_table(seeds, TRAIN_LOSS_KEYS, data))
    sections.append("\n### Summary\n")
    sections.append(_summary_table(seeds, TRAIN_LOSS_KEYS, data))

    # 6. Absolute CE by masking mode
    MASKING_MODES = ["unmasked", "stoch_masked", "ci_masked", "rounded_masked"]
    sections.append("\n## Absolute CE by Masking Mode\n")
    sections.append(f"Target model val CE: {_fmt(target_info.val_loss)}\n\n")
    ce_headers = ["seed"] + MASKING_MODES
    ce_rows: list[list[str]] = []
    per_mode_ces: dict[str, list[float]] = {m: [] for m in MASKING_MODES}
    for s in seeds:
        row = [str(s)]
        for mode in MASKING_MODES:
            ce_diff = data[s].get(f"eval/ce_kl/ce_difference_{mode}")
            if ce_diff is not None:
                spd_ce = target_info.val_loss + ce_diff
                row.append(_fmt(spd_ce))
                per_mode_ces[mode].append(spd_ce)
            else:
                row.append("—")
        ce_rows.append(row)
    sections.append(_md_table(ce_headers, ce_rows))

    ce_summary_headers = ["stat"] + MASKING_MODES
    ce_mean_row = ["mean"]
    ce_std_row = ["std"]
    for mode in MASKING_MODES:
        vals = per_mode_ces[mode]
        if vals:
            ce_mean_row.append(_fmt(np.mean(vals)))
            ce_std_row.append(_fmt(np.std(vals)))
        else:
            ce_mean_row.append("—")
            ce_std_row.append("—")
    sections.append("\n### Summary\n")
    sections.append(_md_table(ce_summary_headers, [ce_mean_row, ce_std_row]))

    # 7. Training compute recovered
    sections.append("\n## Training Compute Recovered\n")
    sections.append(
        "Percentage through target model training where target val loss equals SPD model CE.\n"
    )
    recovered_headers = ["seed"] + MASKING_MODES
    recovered_rows: list[list[str]] = []
    per_mode_pcts: dict[str, list[float]] = {m: [] for m in MASKING_MODES}
    for s in seeds:
        row = [str(s)]
        for mode in MASKING_MODES:
            ce_diff = data[s].get(f"eval/ce_kl/ce_difference_{mode}")
            if ce_diff is not None:
                spd_ce = target_info.val_loss + ce_diff
                pct = _compute_recovered_pct(target_info, spd_ce)
                if pct is not None:
                    row.append(f"{pct:.1f}%")
                    per_mode_pcts[mode].append(pct)
                else:
                    row.append("< 0%")
            else:
                row.append("—")
        recovered_rows.append(row)
    sections.append(_md_table(recovered_headers, recovered_rows))

    summary_headers = ["stat"] + MASKING_MODES
    mean_row = ["mean"]
    std_row = ["std"]
    for mode in MASKING_MODES:
        vals = per_mode_pcts[mode]
        if vals:
            mean_row.append(f"{np.mean(vals):.1f}%")
            std_row.append(f"{np.std(vals):.1f}%")
        else:
            mean_row.append("—")
            std_row.append("—")
    sections.append("\n### Summary\n")
    sections.append(_md_table(summary_headers, [mean_row, std_row]))

    # 8. Plain-text summary list
    summary_groups: list[tuple[str, list[str]]] = [
        ("Output Quality (CE/KL)", CE_KL_KEYS),
        ("Eval Reconstruction Losses", EVAL_LOSS_KEYS),
        ("Sparsity (CI-L0)", L0_LAYER_KEYS),
        ("Training Losses", TRAIN_LOSS_KEYS),
    ]
    sections.append("\n## All Summary Statistics\n")
    for group_name, keys in summary_groups:
        lines = [f"**{group_name}**"]
        for k in keys:
            vals = [data[s][k] for s in seeds if data[s].get(k) is not None]
            if vals:
                lines.append(
                    f"{_summary_name(k)}: {_fmt(np.mean(vals))} (std: {_fmt(np.std(vals))})"
                )
        sections.append("\n\n" + "\n\n".join(lines))

    # 9. LaTeX summary table
    sections.append(
        _render_latex_summary(seeds, data, target_info, per_mode_pcts, spd_config, n_alive)
    )

    return "\n".join(sections) + "\n"


def plot_val_loss_curve(target_info: TargetModelInfo, out_path: Path) -> None:
    """Plot the raw and isotonic-fitted target model val loss curve."""
    import matplotlib.pyplot as plt

    curve = target_info.val_loss_curve
    steps_raw = np.array([s for s, _ in curve], dtype=np.float64)
    losses_raw = np.array([v for _, v in curve], dtype=np.float64)
    steps_fit, losses_fit = _smooth_val_curve(curve)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps_raw / 1000, losses_raw, "o", markersize=3, alpha=0.5, color="C0", label="Raw")
    ax.plot(steps_fit / 1000, losses_fit, "-", linewidth=2, color="C1", label="EMA smoothed")
    ax.set_xlabel("Training step (k)")
    ax.set_ylabel("Val loss (CE)")
    ax.set_title("Target model val loss curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out_path}", file=sys.stderr)


_SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sweep summary report")
    parser.add_argument("run_ids", nargs="+", help="WandB run IDs")
    parser.add_argument("--project", default="goodfire/spd")
    parser.add_argument(
        "--name",
        default=None,
        help="Name for the output directory (default: first run ID). "
        "Results are saved to spd/scripts/sweep_summary/out/<name>/",
    )
    parser.add_argument(
        "--harvest-run",
        default=None,
        help="Run ID to use for n_alive harvest data (if different from sweep runs)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print report to stdout instead of saving to file",
    )
    args = parser.parse_args()

    seeds, data, target_info, spd_config, n_alive = fetch_runs(args.run_ids, args.project)
    if args.harvest_run:
        n_alive = _fetch_n_alive_from_harvest(args.harvest_run)
    report = generate_report(seeds, data, target_info, spd_config, n_alive)

    if args.stdout:
        print(report)
        return

    name = args.name or args.run_ids[0]
    out_dir = _SCRIPT_DIR / "out" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "report.md"
    report_path.write_text(report)
    plot_val_loss_curve(target_info, out_dir / "target_val_loss_curve.png")
    print(f"Results saved to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
