"""Unified geometric consistency comparison runner.

Orchestrates all comparison types (SPD subcomponents, SPD at init, SPD clusters,
transcoders, CLTs) and generates an HTML summary table.

Usage:
    python spd/scripts/compare_models/run_all_comparisons.py run <config.yaml>
    python spd/scripts/compare_models/run_all_comparisons.py collect <config.yaml>
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import fire
import torch
import torch.nn.functional as F
import yaml

from spd.log import logger
from spd.scripts.compare_models.compare_models import resolve_output_dir
from spd.settings import SPD_OUT_DIR


def _load_config(config_path: Path | str) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


# ---------------------------------------------------------------------------
# Alive mask computation for TC/CLT
# ---------------------------------------------------------------------------


def _compute_alive_masks(
    wandb_project: str,
    run_ids: list[str],
    model_type: str,
    alive_dir: Path,
) -> None:
    """Compute alive masks for transcoder/CLT runs by running encoders on real data."""
    prefix = "tc" if model_type == "transcoder" else "clt"
    needed = [rid for rid in run_ids if not (alive_dir / f"{prefix}_{rid}.pt").exists()]
    if not needed:
        logger.info(f"All alive masks exist for {model_type} runs")
        return

    logger.info(f"Computing alive masks for {len(needed)} {model_type} runs...")

    from spd.data import DatasetConfig, create_data_loader
    from spd.models.component_model import ComponentModel
    from spd.scripts.compare_models.compare_transcoders import (
        _download_run_artifacts,
    )

    # Load target model to get MLP inputs
    logger.info("Loading target model for MLP inputs...")
    model = ComponentModel.from_pretrained("wandb:goodfire/spd/runs/s-55ea3f9b")
    model.eval()
    target = model.target_model

    dataset_config = DatasetConfig(
        name="danbraunai/pile-uncopyrighted-tok-shuffled",
        hf_tokenizer_path="EleutherAI/gpt-neox-20b",
        split="val",
        n_ctx=512,
        is_tokenized=True,
        streaming=True,
        column_name="input_ids",
        shuffle_each_epoch=False,
        seed=None,
    )
    loader, _ = create_data_loader(
        dataset_config=dataset_config, batch_size=16, buffer_size=1000, global_seed=42
    )

    mlp_input_lists: dict[int, list[torch.Tensor]] = {i: [] for i in range(4)}
    hooks = []
    for li in range(4):

        def hook_fn(_module: Any, input: Any, _output: Any, li: int = li) -> None:
            mlp_input_lists[li].append(input[0].detach().float())

        hooks.append(target.h[li].mlp.c_fc.register_forward_hook(hook_fn))  # pyright: ignore[reportAttributeAccessIssue,reportIndexIssue]

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= 50:
                break
            target(batch["input_ids"])
    for h in hooks:
        h.remove()
    mlp_inputs: dict[int, torch.Tensor] = {}
    for i in range(4):
        mlp_inputs[i] = torch.cat(mlp_input_lists[i]).reshape(-1, 768)

    del model, target

    def compute_alive(W_enc: torch.Tensor, b_enc: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        n_feat = W_enc.shape[1]
        ever_active = torch.zeros(n_feat, dtype=torch.bool)
        for s in range(0, x.shape[0], 4096):
            batch = x[s : s + 4096]
            pre = F.relu(batch @ W_enc + b_enc)
            k = min(16 * batch.shape[0], pre.numel())
            _, idxs = torch.topk(pre.flatten(), k)
            ever_active[idxs.unique() % n_feat] = True
        return ever_active

    alive_dir.mkdir(parents=True, exist_ok=True)

    for run_id in needed:
        artifacts = _download_run_artifacts(wandb_project, run_id)
        alive_per_layer: dict[int, torch.Tensor] = {}

        if model_type == "transcoder":
            for name, path in artifacts:
                if "checkpoint" not in name or "history" in name:
                    continue
                for part in name.split("_"):
                    if part.startswith("layer"):
                        layer = int(part.replace("layer", ""))
                        sd = torch.load(path / "encoder.pt", map_location="cpu", weights_only=True)
                        alive_per_layer[layer] = compute_alive(
                            sd["W_enc"].float(), sd["b_enc"].float(), mlp_inputs[layer]
                        )
                        break
        else:
            model_arts = [(n, p) for n, p in artifacts if "checkpoint" in n and "history" not in n]
            assert len(model_arts) == 1
            _, path = model_arts[0]
            sd = torch.load(path / "encoder.pt", map_location="cpu", weights_only=True)
            for layer in range(4):
                alive_per_layer[layer] = compute_alive(
                    sd[f"W_enc.{layer}"].float(), sd[f"b_enc.{layer}"].float(), mlp_inputs[layer]
                )

        torch.save(alive_per_layer, alive_dir / f"{prefix}_{run_id}.pt")
        n_alive = sum(m.sum().item() for m in alive_per_layer.values())
        n_total = sum(m.shape[0] for m in alive_per_layer.values())
        logger.info(f"  {run_id}: {int(n_alive)}/{n_total} alive")


# ---------------------------------------------------------------------------
# SLURM submission helpers
# ---------------------------------------------------------------------------

WORKTREE = Path(__file__).resolve().parents[3]
VENV = SPD_OUT_DIR.parent.parent / "home" / "lee" / "spd" / ".venv"


def _submit_slurm(
    job_name: str,
    script_path: str,
    config_path: str,
    gpu: bool,
    mem: str = "64G",
    time: str = "4:00:00",
) -> int:
    """Submit a SLURM job and return the job ID."""
    gpu_arg = "--gres=gpu:1" if gpu else ""
    cmd = f"""sbatch --job-name={job_name} {gpu_arg} --mem={mem} --time={time} \
--output=$HOME/slurm_logs/{job_name}-%j.out <<'SBATCH'
#!/bin/bash
source {VENV}/bin/activate
export PYTHONPATH={WORKTREE}:$PYTHONPATH
cd {WORKTREE}
python {script_path} run {config_path}
SBATCH"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    assert "Submitted batch job" in result.stdout, f"SLURM submission failed: {result.stderr}"
    job_id = int(result.stdout.strip().split()[-1])
    logger.info(f"Submitted {job_name}: job {job_id}")
    return job_id


# ---------------------------------------------------------------------------
# Run: submit jobs and run CPU work
# ---------------------------------------------------------------------------


def run(config_path: Path | str) -> None:
    config = _load_config(config_path)
    output_dir = resolve_output_dir(config.get("output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    job_ids: dict[str, int] = {}

    # --- SPD trained (GPU) ---
    if "spd_trained" in config:
        spd_cfg = config["spd_trained"]
        cfg_path = output_dir / "spd_trained_config.yaml"
        _write_yaml({**spd_cfg, "output_dir": str(output_dir / "spd_trained")}, cfg_path)
        job_ids["spd_trained"] = _submit_slurm(
            "geom-spd-trained",
            "spd/scripts/compare_models/compare_multi.py",
            str(cfg_path),
            gpu=True,
        )

    # --- SPD init (GPU) ---
    if "spd_init" in config:
        spd_cfg = config["spd_init"]
        cfg_path = output_dir / "spd_init_config.yaml"
        _write_yaml({**spd_cfg, "output_dir": str(output_dir / "spd_init")}, cfg_path)
        job_ids["spd_init"] = _submit_slurm(
            "geom-spd-init",
            "spd/scripts/compare_models/compare_multi.py",
            str(cfg_path),
            gpu=True,
        )

    # --- SPD clusters (CPU, high memory) ---
    if "spd_clusters" in config:
        cluster_cfg = config["spd_clusters"]
        cfg_path = output_dir / "spd_clusters_config.yaml"
        _write_yaml({**cluster_cfg, "output_dir": str(output_dir / "spd_clusters")}, cfg_path)
        job_ids["spd_clusters"] = _submit_slurm(
            "geom-clusters",
            "spd/scripts/compare_models/compare_clusters.py",
            str(cfg_path),
            gpu=False,
            mem="128G",
        )

    # --- Alive masks (CPU, needed before TC/CLT comparisons) ---
    alive_dir = output_dir / "alive_masks"
    all_tc_clt = list(config.get("transcoders", [])) + list(config.get("clts", []))
    for group in all_tc_clt:
        _compute_alive_masks(
            group["wandb_project"], group["run_ids"], group["model_type"], alive_dir
        )

    # --- Transcoders (CPU) ---
    for group in config.get("transcoders", []):
        label = group["label"].replace(" ", "_").lower()
        cfg_path = output_dir / f"tc_{label}_config.yaml"
        _write_yaml(
            {
                "model_type": group["model_type"],
                "wandb_project": group["wandb_project"],
                "run_ids": group["run_ids"],
                "alive_masks_dir": str(alive_dir),
                "output_dir": str(output_dir / f"tc_{label}"),
                "label": label,
            },
            cfg_path,
        )
        # Run directly (CPU, fast enough)
        logger.info(f"Running transcoder comparison: {group['label']}")
        subprocess.run(
            [
                sys.executable,
                "spd/scripts/compare_models/compare_transcoders.py",
                "run",
                str(cfg_path),
            ],
            check=True,
            env={
                **__import__("os").environ,
                "PYTHONPATH": f"{WORKTREE}:{__import__('os').environ.get('PYTHONPATH', '')}",
            },
        )

    # --- CLTs (CPU) ---
    for group in config.get("clts", []):
        label = group["label"].replace(" ", "_").lower()
        cfg_path = output_dir / f"clt_{label}_config.yaml"
        _write_yaml(
            {
                "model_type": group["model_type"],
                "wandb_project": group["wandb_project"],
                "run_ids": group["run_ids"],
                "alive_masks_dir": str(alive_dir),
                "output_dir": str(output_dir / f"clt_{label}"),
                "label": label,
            },
            cfg_path,
        )
        logger.info(f"Running CLT comparison: {group['label']}")
        subprocess.run(
            [
                sys.executable,
                "spd/scripts/compare_models/compare_transcoders.py",
                "run",
                str(cfg_path),
            ],
            check=True,
            env={
                **__import__("os").environ,
                "PYTHONPATH": f"{WORKTREE}:{__import__('os').environ.get('PYTHONPATH', '')}",
            },
        )

    # Save job IDs for collect
    if job_ids:
        with open(output_dir / "slurm_jobs.json", "w") as f:
            json.dump(job_ids, f)
        logger.info(f"SLURM jobs submitted: {job_ids}")
        logger.info("Run 'collect' after GPU jobs complete to generate the HTML table.")
    else:
        logger.info("No SLURM jobs needed. Run 'collect' to generate the HTML table.")


# ---------------------------------------------------------------------------
# Collect: read results and generate HTML table
# ---------------------------------------------------------------------------


def _read_spd_results(results_dir: Path) -> dict[str, float] | None:
    """Read multi_summary.json from an SPD comparison directory."""
    summary_path = results_dir / "multi_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        data = json.load(f)
    # Average across all pairs
    pairwise = data["pairwise"]
    if not pairwise:
        return None
    all_keys: set[str] = set()
    for r in pairwise.values():
        all_keys.update(r.keys())
    averaged: dict[str, float] = {}
    for key in all_keys:
        vals = [r[key] for r in pairwise.values() if key in r]
        if vals:
            averaged[key] = sum(vals) / len(vals)
    return averaged


def _read_tc_clt_results(results_dir: Path) -> float | None:
    """Read the overall mean max-match from a TC/CLT multi_summary.json."""
    summary_path = results_dir / "multi_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        data = json.load(f)
    pairwise = data.get("pairwise", {})
    if not pairwise:
        return None
    means = [r["a_to_b_mean"] for r in pairwise.values() if "a_to_b_mean" in r]
    return sum(means) / len(means) if means else None


def _read_cluster_results(results_dir: Path) -> float | None:
    """Read cluster comparison results."""
    # Find the single results.json in the subdirectory
    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            rpath = subdir / "results.json"
            if rpath.exists():
                with open(rpath) as f:
                    data = json.load(f)
                return data.get("a_to_b_mean")
    return None


def collect(config_path: Path | str) -> None:
    config = _load_config(config_path)
    output_dir = resolve_output_dir(config.get("output_dir"))

    rows: list[tuple[str, str]] = []  # (label, value_str)

    # --- SPD trained ---
    spd_trained = _read_spd_results(output_dir / "spd_trained")
    if spd_trained:
        for prefix, label in [
            ("rank1", "SPD rank-1 (V@U)"),
            ("u", "SPD U vectors"),
            ("v", "SPD V vectors"),
        ]:
            key = f"{prefix}_cosine_mean/all_layers"
            val = spd_trained.get(key)
            rows.append((label, f"{val:.4f}" if val is not None else "—"))
    else:
        rows.append(("SPD rank-1 (V@U)", "pending"))
        rows.append(("SPD U vectors", "pending"))
        rows.append(("SPD V vectors", "pending"))

    # --- SPD init ---
    spd_init = _read_spd_results(output_dir / "spd_init")
    if spd_init:
        for prefix, label in [
            ("rank1", "SPD at init rank-1 (V@U)"),
            ("u", "SPD at init U vectors"),
            ("v", "SPD at init V vectors"),
        ]:
            key = f"{prefix}_cosine_mean/all_layers"
            val = spd_init.get(key)
            rows.append((label, f"{val:.4f}" if val is not None else "—"))
    else:
        rows.append(("SPD at init rank-1 (V@U)", "pending"))
        rows.append(("SPD at init U vectors", "pending"))
        rows.append(("SPD at init V vectors", "pending"))

    # --- SPD clusters ---
    cluster_val = _read_cluster_results(output_dir / "spd_clusters")
    rows.append(
        (
            "SPD clusters (cross-model, iter 4423)",
            f"{cluster_val:.4f}" if cluster_val is not None else "pending",
        )
    )

    # --- Transcoders ---
    for group in config.get("transcoders", []):
        label_key = group["label"].replace(" ", "_").lower()
        val = _read_tc_clt_results(output_dir / f"tc_{label_key}")
        rows.append((group["label"], f"{val:.4f}" if val is not None else "pending"))

    # --- CLTs ---
    for group in config.get("clts", []):
        label_key = group["label"].replace(" ", "_").lower()
        val = _read_tc_clt_results(output_dir / f"clt_{label_key}")
        rows.append((group["label"], f"{val:.4f}" if val is not None else "pending"))

    # Generate HTML table
    html_lines = [
        "<table>",
        "<tr><th>Method</th><th>Mean max-match cos sim</th></tr>",
    ]
    for label, val in rows:
        html_lines.append(f"<tr><td>{label}</td><td>{val}</td></tr>")
    html_lines.append("</table>")
    html = "\n".join(html_lines)

    html_path = output_dir / "results_table.html"
    html_path.write_text(html)
    logger.info(f"HTML table written to {html_path}")
    print(html)


if __name__ == "__main__":
    fire.Fire({"run": run, "collect": collect})
