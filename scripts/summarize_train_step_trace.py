"""Summarize per-stage GPU and NCCL communication time from train-step traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dirs", type=Path, nargs="+")
    return parser.parse_args()


def interval_union_us(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


def is_comm_event(event: dict[str, Any]) -> bool:
    name = str(event.get("name", "")).lower()
    args = event.get("args", {})
    return (
        "nccl" in name
        or "allreduce" in name
        or "all_reduce" in name
        or "reduce_scatter" in name
        or "all_gather" in name
        or "broadcast" in name
        or "collective name" in {str(k).lower() for k in args}
        or "process group name" in {str(k).lower() for k in args}
    )


def load_stage_names(profile_dir: Path) -> list[str]:
    result = json.loads((profile_dir / "result.json").read_text())
    return list(result["summary"]["phases"].keys())


def summarize_rank(trace_path: Path, stage_names: set[str]) -> dict[str, dict[str, float | int]]:
    events = json.loads(trace_path.read_text())["traceEvents"]
    stage_events = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "user_annotation"
        and event.get("name") in stage_names
    ]
    gpu_events = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("cat") in GPU_CATEGORIES
    ]

    rank_summary: dict[str, dict[str, float | int]] = {}
    for stage in stage_events:
        name = str(stage["name"])
        start = float(stage["ts"])
        end = start + float(stage["dur"])

        gpu_intervals: list[tuple[float, float]] = []
        comm_intervals: list[tuple[float, float]] = []
        gpu_sum_us = 0.0
        comm_sum_us = 0.0
        comm_count = 0

        for event in gpu_events:
            event_start = float(event["ts"])
            event_end = event_start + float(event["dur"])
            overlap_start = max(start, event_start)
            overlap_end = min(end, event_end)
            if overlap_end <= overlap_start:
                continue

            overlap_us = overlap_end - overlap_start
            gpu_intervals.append((overlap_start, overlap_end))
            gpu_sum_us += overlap_us
            if is_comm_event(event):
                comm_intervals.append((overlap_start, overlap_end))
                comm_sum_us += overlap_us
                comm_count += 1

        rank_summary[name] = {
            "wall_ms": float(stage["dur"]) / 1000.0,
            "gpu_active_ms": interval_union_us(gpu_intervals) / 1000.0,
            "gpu_sum_ms": gpu_sum_us / 1000.0,
            "comm_ms": interval_union_us(comm_intervals) / 1000.0,
            "comm_sum_ms": comm_sum_us / 1000.0,
            "comm_count": comm_count,
        }

    return rank_summary


def aggregate(profile_dir: Path) -> dict[str, Any]:
    stage_names = set(load_stage_names(profile_dir))
    traces = sorted(profile_dir.glob("trace_rank*.json"))
    if not traces:
        raise FileNotFoundError(f"no trace_rank*.json files under {profile_dir}")

    rank_summaries = {
        int(trace.stem.removeprefix("trace_rank")): summarize_rank(trace, stage_names)
        for trace in traces
    }

    phase_summary: dict[str, Any] = {}
    for stage_name in sorted(stage_names):
        per_rank = [
            summary[stage_name]
            for _rank, summary in sorted(rank_summaries.items())
            if stage_name in summary
        ]
        if not per_rank:
            continue
        phase_summary[stage_name] = {
            "wall_ms_max": max(float(row["wall_ms"]) for row in per_rank),
            "wall_ms_avg": sum(float(row["wall_ms"]) for row in per_rank) / len(per_rank),
            "gpu_active_ms_max": max(float(row["gpu_active_ms"]) for row in per_rank),
            "gpu_active_ms_avg": sum(float(row["gpu_active_ms"]) for row in per_rank)
            / len(per_rank),
            "gpu_sum_ms_max": max(float(row["gpu_sum_ms"]) for row in per_rank),
            "comm_ms_max": max(float(row["comm_ms"]) for row in per_rank),
            "comm_ms_avg": sum(float(row["comm_ms"]) for row in per_rank) / len(per_rank),
            "comm_sum_ms_max": max(float(row["comm_sum_ms"]) for row in per_rank),
            "comm_count_max": max(int(row["comm_count"]) for row in per_rank),
            "per_rank": per_rank,
        }

    result = json.loads((profile_dir / "result.json").read_text())
    return {
        "profile_dir": str(profile_dir),
        "world_size": result["world_size"],
        "rank_batch_size": result["rank_batch_size"],
        "global_batch_size": result["global_batch_size"],
        "trace_count": len(traces),
        "phases": phase_summary,
    }


def write_outputs(profile_dir: Path, summary: dict[str, Any]) -> None:
    json_path = profile_dir / "trace_stage_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    tsv_path = profile_dir / "trace_stage_summary.tsv"
    fields = [
        "phase",
        "wall_ms_max",
        "gpu_active_ms_max",
        "gpu_sum_ms_max",
        "comm_ms_max",
        "comm_sum_ms_max",
        "comm_count_max",
        "comm_frac_of_wall_pct",
    ]
    rows = []
    for phase, stats in sorted(
        summary["phases"].items(),
        key=lambda item: float(item[1]["wall_ms_max"]),
        reverse=True,
    ):
        wall = float(stats["wall_ms_max"])
        comm = float(stats["comm_ms_max"])
        rows.append(
            {
                "phase": phase,
                "wall_ms_max": f"{wall:.3f}",
                "gpu_active_ms_max": f"{float(stats['gpu_active_ms_max']):.3f}",
                "gpu_sum_ms_max": f"{float(stats['gpu_sum_ms_max']):.3f}",
                "comm_ms_max": f"{comm:.3f}",
                "comm_sum_ms_max": f"{float(stats['comm_sum_ms_max']):.3f}",
                "comm_count_max": str(stats["comm_count_max"]),
                "comm_frac_of_wall_pct": f"{100 * comm / wall:.1f}" if wall > 0 else "0.0",
            }
        )

    with tsv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {json_path}")
    print(f"wrote {tsv_path}")


def main() -> None:
    args = parse_args()
    for profile_dir in args.profile_dirs:
        write_outputs(profile_dir, aggregate(profile_dir))


if __name__ == "__main__":
    main()
