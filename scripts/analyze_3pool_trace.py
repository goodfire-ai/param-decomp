"""Quantify per-pool compute / comm / idle from 3-pool torch.profiler traces.

The 3-pool trainer profiles one rank per pool (LW block-0 leader, CI leader,
PPGD leader). Steps run in lockstep, so within one ProfilerStep window the pool
with the most GPU compute-busy is the bottleneck and the pool with the most GPU
idle is the one stalling on cross-pool comms.

Usage:
    python scripts/analyze_3pool_trace.py <trace_dir>
    python scripts/analyze_3pool_trace.py trace_ci_rank96.json trace_ppgd_rank100.json ...

Each Chrome trace is large (100-400 MB); loading is the slow part.
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

Interval = tuple[float, float]


def merged_length(intervals: list[Interval]) -> float:
    """Total length covered by the union of (start, end) intervals (microseconds)."""
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    total += cur_end - cur_start
    return total


def clip(intervals: list[Interval], window: Interval) -> list[Interval]:
    w0, w1 = window
    out = []
    for s, e in intervals:
        s, e = max(s, w0), min(e, w1)
        if e > s:
            out.append((s, e))
    return out


@dataclass(frozen=True)
class StepBreakdown:
    step_name: str
    wall_us: float
    compute_us: float  # union of non-nccl GPU kernels
    nccl_us: float  # union of nccl GPU kernels
    gpu_busy_us: float  # union of ALL gpu work
    idle_us: float  # wall - gpu_busy
    nccl_cpu_by_op: dict[str, float]  # cpu-side nccl annotation time by op kind


@dataclass(frozen=True)
class TraceSummary:
    path: str
    pool: str
    rank: int
    steps: list[StepBreakdown]


def _is_gpu_kernel(e: dict) -> bool:
    return e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")


def _is_nccl(name: str) -> bool:
    return "nccl" in name.lower()


def _nccl_op_kind(name: str) -> str:
    # names like "nccl:send 96->0", "nccl:recv 96<-100", "nccl:all_reduce"
    body = name.split(":", 1)[1] if ":" in name else name
    return body.split()[0]


def analyze(path: Path) -> TraceSummary:
    d = json.loads(path.read_text())
    ev = d["traceEvents"]
    dist = d["distributedInfo"]
    rank = dist["rank"]
    pool = path.stem.replace("trace_", "").split("_rank")[0]

    step_events = [e for e in ev if e.get("ph") == "X" and "ProfilerStep" in str(e.get("name", ""))]
    # Each step appears once per thread; keep the longest (CPU launch thread).
    by_name: dict[str, dict] = {}
    for e in step_events:
        if e["name"] not in by_name or e["dur"] > by_name[e["name"]]["dur"]:
            by_name[e["name"]] = e
    steps_meta = sorted(by_name.values(), key=lambda e: e["ts"])
    assert steps_meta, f"no ProfilerStep events in {path}"

    gpu_compute: list[Interval] = []
    gpu_nccl: list[Interval] = []
    for e in ev:
        if not _is_gpu_kernel(e):
            continue
        iv = (e["ts"], e["ts"] + e.get("dur", 0.0))
        if _is_nccl(str(e.get("name", ""))):
            gpu_nccl.append(iv)
        else:
            gpu_compute.append(iv)

    nccl_cpu = [
        e for e in ev if e.get("cat") == "user_annotation" and _is_nccl(str(e.get("name", "")))
    ]

    breakdowns = []
    for sm in steps_meta:
        window = (sm["ts"], sm["ts"] + sm["dur"])
        comp = clip(gpu_compute, window)
        nccl = clip(gpu_nccl, window)
        compute_us = merged_length(comp)
        nccl_us = merged_length(nccl)
        gpu_busy_us = merged_length(comp + nccl)
        wall_us = sm["dur"]

        op_time: dict[str, float] = defaultdict(float)
        for e in nccl_cpu:
            if window[0] <= e["ts"] < window[1]:
                op_time[_nccl_op_kind(str(e["name"]))] += e.get("dur", 0.0)

        breakdowns.append(
            StepBreakdown(
                step_name=sm["name"],
                wall_us=wall_us,
                compute_us=compute_us,
                nccl_us=nccl_us,
                gpu_busy_us=gpu_busy_us,
                idle_us=wall_us - gpu_busy_us,
                nccl_cpu_by_op=dict(op_time),
            )
        )
    return TraceSummary(str(path), pool, rank, breakdowns)


def _fmt_ms(us: float) -> str:
    return f"{us / 1000:7.1f}"


def print_report(summaries: list[TraceSummary]) -> None:
    print("\n" + "=" * 92)
    print("PER-POOL STEP BREAKDOWN  (ms, GPU-side union unless noted)")
    print("=" * 92)
    header = f"{'pool':<11}{'rank':>5}{'step':>14}{'wall':>9}{'compute':>9}{'nccl':>9}{'busy':>9}{'idle':>9}{'idle%':>7}"
    print(header)
    print("-" * 92)
    for s in summaries:
        for b in s.steps:
            idle_pct = 100 * b.idle_us / b.wall_us
            print(
                f"{s.pool:<11}{s.rank:>5}{b.step_name:>14}"
                f"{_fmt_ms(b.wall_us)}{_fmt_ms(b.compute_us)}{_fmt_ms(b.nccl_us)}"
                f"{_fmt_ms(b.gpu_busy_us)}{_fmt_ms(b.idle_us)}{idle_pct:>6.1f}%"
            )
        print("-" * 92)

    print("\nMEANS ACROSS STEPS")
    print(f"{'pool':<11}{'wall':>9}{'compute':>9}{'nccl':>9}{'idle':>9}{'idle%':>7}{'compute%':>9}")
    print("-" * 70)
    for s in summaries:
        n = len(s.steps)
        wall = sum(b.wall_us for b in s.steps) / n
        comp = sum(b.compute_us for b in s.steps) / n
        nccl = sum(b.nccl_us for b in s.steps) / n
        idle = sum(b.idle_us for b in s.steps) / n
        print(
            f"{s.pool:<11}{_fmt_ms(wall)}{_fmt_ms(comp)}{_fmt_ms(nccl)}{_fmt_ms(idle)}"
            f"{100 * idle / wall:>6.1f}%{100 * comp / wall:>8.1f}%"
        )

    print("\nCPU-SIDE NCCL TIME BY OP KIND (ms, mean/step — includes blocking wait)")
    print(f"{'pool':<11}{'op':>14}{'ms/step':>10}")
    print("-" * 40)
    for s in summaries:
        agg: dict[str, float] = defaultdict(float)
        for b in s.steps:
            for op, t in b.nccl_cpu_by_op.items():
                agg[op] += t
        for op, t in sorted(agg.items(), key=lambda kv: -kv[1]):
            print(f"{s.pool:<11}{op:>14}{_fmt_ms(t / len(s.steps)):>10}")


def main() -> None:
    args = [Path(a) for a in sys.argv[1:]]
    assert args, "pass a trace dir or one+ trace json files"
    paths = sorted(args[0].glob("trace_*.json")) if len(args) == 1 and args[0].is_dir() else args
    assert paths, "no trace files found"

    summaries = []
    for p in paths:
        print(f"loading {p.name} ({p.stat().st_size / 1e6:.0f} MB) ...", flush=True)
        summaries.append(analyze(p))
    print_report(summaries)


if __name__ == "__main__":
    main()
