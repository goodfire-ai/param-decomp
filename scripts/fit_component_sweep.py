"""Launch a sweep of fit_component.py runs as a SLURM array job.

Generates one array task per parameter combination. Each task gets its own
GPU and writes its output to a separate file. A dependent summary job collects
all outputs into a single results file with a LaTeX table.

Usage:
    python scripts/fit_component_sweep.py \
        --config scripts/fit_component_config.yaml \
        --n_gpus 8 \
        --sweep hidden_width=0,1,2,4,8,16,32,64 interact_width=0

Each --sweep argument is key=val1,val2,...  Multiple keys create independent
variations (not a Cartesian product) unless --product is passed.
"""

import argparse
import datetime
import itertools
import json
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import yaml

from spd.log import logger
from spd.settings import SPD_OUT_DIR
from spd.utils.slurm import (
    SlurmArrayConfig,
    SlurmConfig,
    generate_array_script,
    generate_script,
    submit_slurm_job,
)

SWEEP_OUT_DIR = SPD_OUT_DIR / "fit_component_sweeps"


def _get_default_partition() -> str:
    """Query SLURM for the default partition (marked with *)."""
    result = subprocess.run(["sinfo", "--format=%P", "--noheader"], capture_output=True, text=True)
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line.endswith("*"):
            return line[:-1]
    raise RuntimeError("No default SLURM partition found")


def load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_sweep_arg(s: str) -> tuple[str, list[str]]:
    """Parse 'key=v1,v2,v3' into (key, [v1, v2, v3])."""
    key, _, vals = s.partition("=")
    assert vals, f"Sweep arg must be key=val1,val2,...  Got: {s!r}"
    return key, vals.split(",")


def cast_value(key: str, val: str, base_config: dict[str, Any]) -> Any:
    """Cast a string value to match the type of the same key in base_config."""
    if key in base_config:
        target_type = type(base_config[key])
        if target_type is bool:
            return val.lower() in ("true", "1", "yes")
        return target_type(val)
    # Best-effort: try int, then float, then keep as string
    for t in (int, float):
        try:
            return t(val)
        except ValueError:
            continue
    return val


def build_combinations(
    sweep_specs: list[tuple[str, list[str]]],
    base_config: dict[str, Any],
    product: bool,
) -> list[dict[str, Any]]:
    """Build list of override dicts from sweep specs.

    If product=True, returns the Cartesian product of all sweep dimensions.
    Otherwise, zips them (all sweep lists must have the same length).
    """
    if product:
        keys = [k for k, _ in sweep_specs]
        val_lists = [vs for _, vs in sweep_specs]
        combos = []
        for vals in itertools.product(*val_lists):
            combos.append(
                {k: cast_value(k, v, base_config) for k, v in zip(keys, vals, strict=True)}
            )
        return combos

    # Zip mode: single-value specs are broadcast, multi-value specs must agree in length
    lengths = {k: len(vs) for k, vs in sweep_specs if len(vs) > 1}
    unique_lengths = set(lengths.values())
    assert len(unique_lengths) <= 1, (
        f"In zip mode all multi-value sweep lists must have the same length, got {lengths}. "
        "Use --product for Cartesian product."
    )
    n = unique_lengths.pop() if unique_lengths else 1
    combos = []
    for i in range(n):
        combo = {}
        for k, vs in sweep_specs:
            v = vs[0] if len(vs) == 1 else vs[i]
            combo[k] = cast_value(k, v, base_config)
        combos.append(combo)
    return combos


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch fit_component sweep on SLURM")
    parser.add_argument("--config", type=Path, required=True, help="Base config YAML")
    parser.add_argument("--n_gpus", type=int, required=True, help="Number of GPUs to use")
    parser.add_argument(
        "--sweep",
        nargs="+",
        required=True,
        help="Sweep specs: key=v1,v2,...  (one per swept parameter)",
    )
    parser.add_argument(
        "--product",
        action="store_true",
        help="Take Cartesian product of sweep dimensions (default: zip)",
    )
    parser.add_argument("--time", default="4:00:00", help="SLURM time limit (default: 4:00:00)")
    args = parser.parse_args()

    base_config = load_config(args.config)
    sweep_specs = [parse_sweep_arg(s) for s in args.sweep]
    combos = build_combinations(sweep_specs, base_config, args.product)
    n_tasks = len(combos)
    logger.info(f"Sweep has {n_tasks} configurations, deploying on {args.n_gpus} GPUs")

    # Create output directory for this sweep
    SWEEP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = SWEEP_OUT_DIR / sweep_id
    sweep_dir.mkdir()

    # Write per-task configs and build commands
    commands: list[str] = []
    task_labels: list[str] = []
    for i, overrides in enumerate(combos):
        task_config = {**base_config, **overrides}

        config_path = sweep_dir / f"config_{i}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(task_config, f, default_flow_style=False)

        output_path = sweep_dir / f"output_{i}.txt"
        overrides_json = json.dumps(overrides)
        cmd = (
            f"python scripts/fit_component.py --config {config_path} "
            f"> {output_path} 2>&1 ; "
            f'echo "SWEEP_OVERRIDES: {overrides_json}" >> {output_path}'
        )
        commands.append(cmd)

        label = " ".join(f"{k}={v}" for k, v in overrides.items())
        task_labels.append(label)

    # Save sweep metadata
    metadata = {
        "base_config": str(args.config),
        "sweep_specs": {k: vs for k, vs in sweep_specs},
        "combinations": combos,
        "sweep_dir": str(sweep_dir),
    }
    with open(sweep_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Write the summary script that runs after all tasks complete
    summary_script_path = sweep_dir / "summarize.py"
    # Embed sweep_dir and combos into the summary script
    summary_script_content = textwrap.dedent(f"""\
        import json, re, sys
        from pathlib import Path

        sweep_dir = Path("{sweep_dir}")
        combos = {json.dumps(combos)}
        n_tasks = len(combos)

        results_path = sweep_dir / "results.txt"
        with open(results_path, "w") as out:
            # Collect per-task outputs
            for i in range(n_tasks):
                output_path = sweep_dir / f"output_{{i}}.txt"
                label = " ".join(f"{{k}}={{v}}" for k, v in combos[i].items())
                out.write("=" * 80 + "\\n")
                out.write(f"CONFIG: {{label}}\\n")
                out.write("=" * 80 + "\\n")
                if output_path.exists():
                    out.write(output_path.read_text())
                else:
                    out.write("ERROR: output file not found\\n")
                out.write("\\n\\n")

            # Parse results and build tables
            rows = []
            for i in range(n_tasks):
                output_path = sweep_dir / f"output_{{i}}.txt"
                if not output_path.exists():
                    continue
                text = output_path.read_text()
                train_m = re.search(r"Last training step:\\s+MSE=([\\d.]+),\\s+R.=([\\d.+-]+)", text)
                eval_m = re.search(r"Evaluation \\(\\d+ batches\\):\\s+MSE=([\\d.]+),\\s+R.=([\\d.+-]+)", text)
                actual_m = re.search(r"Eval vs actual target:\\s+MSE=([\\d.]+),\\s+R.=([\\d.+-]+)", text)
                ci_m = re.search(r"Eval pred vs CI-masked:\\s+MSE=([\\d.]+),\\s+R.=([\\d.+-]+)", text)
                if not (train_m and eval_m and actual_m):
                    print(f"WARNING: could not parse results from task {{i}}", file=sys.stderr)
                    continue
                row = {{
                    "overrides": combos[i],
                    "train_mse": float(train_m.group(1)),
                    "train_r2": float(train_m.group(2)),
                    "eval_mse": float(eval_m.group(1)),
                    "eval_r2": float(eval_m.group(2)),
                    "actual_mse": float(actual_m.group(1)),
                    "actual_r2": float(actual_m.group(2)),
                }}
                if ci_m:
                    row["ci_mse"] = float(ci_m.group(1))
                    row["ci_r2"] = float(ci_m.group(2))
                rows.append(row)

            # Plain text summary
            out.write("=" * 80 + "\\n")
            out.write("SUMMARY\\n")
            out.write("=" * 80 + "\\n\\n")

            # Determine sweep keys for column headers
            sweep_keys = list(combos[0].keys()) if combos else []
            has_ci = any("ci_mse" in r for r in rows)
            key_headers = "  ".join(f"{{k:>12}}" for k in sweep_keys)
            ci_header = f"  {{'CI MSE':>12}}  {{'CI R²':>10}}" if has_ci else ""
            header = f"{{key_headers}}  {{'Train MSE':>12}}  {{'Train R²':>10}}  {{'Eval MSE':>12}}  {{'Eval R²':>10}}  {{'Actual MSE':>12}}  {{'Actual R²':>10}}{{ci_header}}"
            out.write(header + "\\n")
            out.write("-" * len(header) + "\\n")
            for r in rows:
                key_vals = "  ".join(f"{{str(r['overrides'][k]):>12}}" for k in sweep_keys)
                ci_cols = (
                    f"  {{r['ci_mse']:>12.6f}}  {{r['ci_r2']:>10.4f}}"
                    if "ci_mse" in r else ""
                )
                out.write(
                    f"{{key_vals}}  {{r['train_mse']:>12.6f}}  {{r['train_r2']:>10.4f}}"
                    f"  {{r['eval_mse']:>12.6f}}  {{r['eval_r2']:>10.4f}}"
                    f"  {{r['actual_mse']:>12.6f}}  {{r['actual_r2']:>10.4f}}{{ci_cols}}\\n"
                )

            # LaTeX table
            n_key_cols = len(sweep_keys)
            ci_col_spec = " r r" if has_ci else ""
            col_spec = "r " * n_key_cols + "r r r r r r" + ci_col_spec
            latex_key_header = " & ".join(f"\\\\texttt{{{{{{k}}}}}}" for k in sweep_keys)
            ci_latex_header = " & CI MSE & CI $R^2$" if has_ci else ""

            out.write("\\n\\nLaTeX table (paste into Overleaf):\\n\\n")
            out.write("\\\\begin{{table}}[h]\\n")
            out.write("\\\\centering\\n")
            out.write("\\\\caption{{Fit quality sweep results}}\\n")
            out.write("\\\\label{{tab:fit-sweep}}\\n")
            out.write(f"\\\\begin{{{{tabular}}}}{{{{{{col_spec}}}}}}\\n")
            out.write("\\\\toprule\\n")
            out.write(
                f"{{latex_key_header}} & Train MSE & Train $R^2$ & Eval MSE & Eval $R^2$"
                f" & Actual MSE & Actual $R^2${{ci_latex_header}} \\\\\\\\\\n"
            )
            out.write("\\\\midrule\\n")
            for r in rows:
                key_vals = " & ".join(str(r["overrides"][k]) for k in sweep_keys)
                ci_latex_cols = (
                    f" & {{r['ci_mse']:.6f}} & {{r['ci_r2']:.4f}}"
                    if "ci_mse" in r else ""
                )
                out.write(
                    f"{{key_vals}} & {{r['train_mse']:.6f}} & {{r['train_r2']:.4f}}"
                    f" & {{r['eval_mse']:.6f}} & {{r['eval_r2']:.4f}}"
                    f" & {{r['actual_mse']:.6f}} & {{r['actual_r2']:.4f}}{{ci_latex_cols}} \\\\\\\\\\n"
                )
            out.write("\\\\bottomrule\\n")
            out.write("\\\\end{{tabular}}\\n")
            out.write("\\\\end{{table}}\\n")

        print(f"Results written to {{results_path}}")
    """)
    with open(summary_script_path, "w") as f:
        f.write(summary_script_content)

    # Submit SLURM array job
    partition = _get_default_partition()
    array_config = SlurmArrayConfig(
        job_name="fit-sweep",
        partition=partition,
        n_gpus=1,
        time=args.time,
        max_concurrent_tasks=args.n_gpus,
    )
    array_script = generate_array_script(
        array_config,
        commands,
        per_task_comments=task_labels,
    )
    array_result = submit_slurm_job(
        array_script,
        script_name_prefix="fit_sweep",
        is_array=True,
        n_array_tasks=n_tasks,
    )
    logger.info(f"Submitted array job {array_result.job_id} ({n_tasks} tasks)")
    logger.info(f"Logs: {array_result.log_pattern}")

    # Submit dependent summary job
    summary_config = SlurmConfig(
        job_name="fit-sweep-summary",
        partition=partition,
        n_gpus=0,
        time="00:10:00",
        dependency_job_id=array_result.job_id,
    )
    summary_cmd = f"python {summary_script_path}"
    summary_script = generate_script(summary_config, summary_cmd)
    summary_result = submit_slurm_job(
        summary_script,
        script_name_prefix="fit_sweep_summary",
    )
    logger.info(f"Submitted summary job {summary_result.job_id} (depends on {array_result.job_id})")
    logger.info(f"Results will be at: {sweep_dir / 'results.txt'}")


if __name__ == "__main__":
    main()
