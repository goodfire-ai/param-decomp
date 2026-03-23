"""Backfill merge-only clustering runs into WandB from SLURM logs."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb

from spd.clustering.math.semilog import semilog
from spd.clustering.merge_history import MergeHistory
from spd.settings import DEFAULT_PROJECT_NAME
from spd.utils.wandb_utils import get_wandb_entity

os.environ["WANDB_QUIET"] = "true"

SLURM_LOG_DIR = Path("/mnt/polished-lake/artifacts/mechanisms/spd/slurm_logs")
STATE_FILENAME = "wandb_backfill_state.json"

ITER_METRIC_RE = re.compile(
    r"k=(?P<k>\d+), mdl=(?P<mdl>[0-9.eE+-]+), pair=(?P<pair>[0-9.eE+-]+):"
    r"[^\r\n]*?\|\s*(?P<iter>\d+)/(?P<total>\d+)\s*\["
)
PROGRESS_RE = re.compile(
    r"Compressed merge progress: iter=(?P<iter>\d+)/(?P<total>\d+), "
    r"elapsed=(?P<elapsed>[0-9.]+)s, sec_per_iter=(?P<sec>[0-9.]+), "
    r"k_groups=(?P<groups>\d+)"
)
LOAD_RE = re.compile(r"Loaded: (?P<comps>\d+) components, (?P<samples>\d+) samples")


@dataclass(frozen=True)
class IterMetric:
    step: int
    total_iters: int
    k_groups: int
    merge_pair_cost: float
    mdl_loss_norm: float
    mdl_loss: float
    elapsed_s: float | None = None
    sec_per_iter_avg: float | None = None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def find_log_path(run_dir: Path) -> Path:
    needle = f"Output: {run_dir}"
    result = subprocess.run(
        [
            "rg",
            "-lF",
            "-g",
            "slurm-*.out",
            needle,
            str(SLURM_LOG_DIR),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        matches = [Path(line) for line in result.stdout.splitlines() if line.strip()]
        if matches:
            return sorted(matches)[0]
    raise FileNotFoundError(f"Could not find SLURM log containing {needle!r}")


def parse_iteration_metrics(log_path: Path, n_samples: int) -> tuple[list[IterMetric], dict[str, Any]]:
    text = log_path.read_text(errors="replace")
    chunks = [chunk for chunk in re.split(r"[\r\n]+", text) if chunk.strip()]

    progress_by_step: dict[int, tuple[float, float]] = {}
    run_meta: dict[str, Any] = {}

    for chunk in chunks:
        progress_match = PROGRESS_RE.search(chunk)
        if progress_match:
            step = int(progress_match.group("iter")) - 1
            progress_by_step[step] = (
                float(progress_match.group("elapsed")),
                float(progress_match.group("sec")),
            )
            continue

        load_match = LOAD_RE.search(chunk)
        if load_match and "n_components" not in run_meta:
            run_meta["n_components"] = int(load_match.group("comps"))
            run_meta["n_samples"] = int(load_match.group("samples"))

    metrics_by_step: dict[int, IterMetric] = {}
    for chunk in chunks:
        metric_match = ITER_METRIC_RE.search(chunk)
        if metric_match is None:
            continue

        total_iters = int(metric_match.group("total"))
        iter_num = int(metric_match.group("iter"))
        step = iter_num - 1
        mdl_loss_norm = float(metric_match.group("mdl"))
        elapsed_s, sec_per_iter_avg = progress_by_step.get(step, (None, None))

        metrics_by_step[step] = IterMetric(
            step=step,
            total_iters=total_iters,
            k_groups=int(metric_match.group("k")),
            merge_pair_cost=float(metric_match.group("pair")),
            mdl_loss_norm=mdl_loss_norm,
            mdl_loss=mdl_loss_norm * n_samples,
            elapsed_s=elapsed_s,
            sec_per_iter_avg=sec_per_iter_avg,
        )

    return [metrics_by_step[step] for step in sorted(metrics_by_step)], run_meta


def build_wandb_config(run_dir: Path) -> tuple[dict[str, Any], Path]:
    merge_config_path = run_dir / "merge_config.json"
    merge_payload = load_json(merge_config_path)

    snapshot_path = Path(merge_payload["snapshot_path"])
    harvest_config_path = snapshot_path / "harvest_config.json"
    harvest_config = load_json(harvest_config_path)

    config = {
        **harvest_config,
        "merge_config": merge_payload["merge_config"],
        "snapshot_path": str(snapshot_path),
        "backfilled_from_merge_only": True,
        "backfill_source_run_dir": str(run_dir),
    }
    return config, snapshot_path


def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {"last_logged_step": -1, "artifact_logged": False}
    return load_json(state_path)


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.write_text(json.dumps(state, indent=2))


def maybe_log_history_artifact(run: wandb.sdk.wandb_run.Run, run_dir: Path, state: dict[str, Any]) -> bool:
    if state.get("artifact_logged", False):
        return False

    history_path = run_dir / "history.zip"
    if not history_path.exists():
        return False

    history = MergeHistory.read(history_path)
    artifact = wandb.Artifact(
        name="merge_history",
        type="merge_history",
        description="Merge history backfilled from merge-only run",
        metadata={"n_iters_current": history.n_iters_current, "filename": str(history_path)},
    )
    artifact.add_file(str(history_path))
    run.log_artifact(artifact)
    state["artifact_logged"] = True
    return True


def backfill_run(
    run_dir: Path,
    *,
    project: str,
    entity: str,
    log_path: Path | None,
    dry_run: bool,
) -> tuple[str, int, int]:
    config, snapshot_path = build_wandb_config(run_dir)
    if log_path is None:
        log_path = find_log_path(run_dir)

    n_samples = int(config["n_tokens"] if config.get("n_tokens") is not None else config["n_samples"])
    metrics, log_meta = parse_iteration_metrics(log_path, n_samples=n_samples)
    state_path = run_dir / STATE_FILENAME
    state = load_state(state_path)

    run_id = run_dir.name
    pending_metrics = [metric for metric in metrics if metric.step > int(state["last_logged_step"])]

    if dry_run:
        return run_id, len(pending_metrics), len(metrics)

    tags = ["clustering", "merge-only", f"snapshot:{snapshot_path.name}"]
    if (alpha := config["merge_config"].get("alpha")) is not None:
        tags.append(f"alpha:{alpha}")

    run = wandb.init(
        id=run_id,
        entity=entity,
        project=project,
        resume="allow",
        config=config,
        tags=tags,
    )
    assert run is not None

    run.summary["snapshot_path"] = str(snapshot_path)
    run.summary["slurm_log_path"] = str(log_path)
    if "n_components" in log_meta:
        run.summary["n_components"] = log_meta["n_components"]
    if "n_samples" in log_meta:
        run.summary["n_samples"] = log_meta["n_samples"]

    for metric in pending_metrics:
        payload = {
            "k_groups": metric.k_groups,
            "merge_pair_cost": metric.merge_pair_cost,
            "merge_pair_cost_semilog[1e-3]": semilog(metric.merge_pair_cost, epsilon=1e-3),
            "mdl_loss": metric.mdl_loss,
            "mdl_loss_norm": metric.mdl_loss_norm,
        }
        if metric.elapsed_s is not None:
            payload["backfill/elapsed_s"] = metric.elapsed_s
        if metric.sec_per_iter_avg is not None:
            payload["backfill/sec_per_iter_avg"] = metric.sec_per_iter_avg
        run.log(payload, step=metric.step)
        state["last_logged_step"] = metric.step

    maybe_log_history_artifact(run, run_dir, state)
    run.summary["last_logged_step"] = state["last_logged_step"]
    run.summary["backfill/total_logged_steps"] = max(int(state["last_logged_step"]) + 1, 0)
    run.summary["backfill/source"] = "merge-only slurm log"
    run.finish()

    save_state(state_path, state)
    return run_id, len(pending_metrics), len(metrics)


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="spd-cluster-backfill-wandb",
        description="Backfill merge-only clustering runs into WandB from SLURM logs.",
    )
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Merge run directories.")
    parser.add_argument("--project", default=DEFAULT_PROJECT_NAME, help="WandB project name.")
    parser.add_argument(
        "--entity",
        default=None,
        help="WandB entity. Defaults to WANDB_ENTITY or authenticated default.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional explicit SLURM log path. Only valid with one run dir.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse and report without logging.")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Poll and append new metrics until interrupted.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Polling interval in seconds for --watch.",
    )
    args = parser.parse_args()

    if args.log is not None and len(args.run_dirs) != 1:
        raise SystemExit("--log can only be used with a single run dir")

    entity = get_wandb_entity() if args.entity is None else args.entity

    while True:
        for idx, run_dir in enumerate(args.run_dirs):
            log_path = args.log if idx == 0 else None
            run_id, logged, total = backfill_run(
                run_dir=run_dir,
                project=args.project,
                entity=entity,
                log_path=log_path,
                dry_run=args.dry_run,
            )
            mode = "parsed" if args.dry_run else "logged"
            print(
                f"{run_id}: {mode} {logged} new steps ({total} total seen)",
                file=sys.stderr,
            )

        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    cli()
