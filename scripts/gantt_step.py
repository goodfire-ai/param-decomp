"""ASCII Gantt of one step across LW/CI/PPGD pools, from a slurm trace log.

Reads ``[trace rank=R +Tms] phase: NAME (cur=...|end ...)`` lines emitted under
``PD_PHASE_TRACE=1`` plus the per-pool ``Trainer.run: step N: (start|done)``
markers. For a given step, renders one horizontal bar per rank, each phase
labelled with its short name + duration. ``perf_counter`` clocks are nearly
aligned across ranks (slurm spawns processes within ~tens of ms), so the +Tms
values can be used as a shared timeline.

Defaults assume the production smoke layout: rank 0 = LW, rank 96 = CI,
rank 104 = PPGD. Pass ``--ranks`` to override.

Usage:
    python scripts/gantt_step.py /path/to/slurm-NNN.out --step 3
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

PHASE_RE = re.compile(
    r"^\[trace rank=(\d+) \+\s*([\d.]+)ms\] phase: (\S+?)(\s+end\s.*|\s*cur=.*)?$"
)
STEP_RE = re.compile(
    r"^\[trace rank=(\d+) \+\s*([\d.]+)ms\] Trainer\.run: step (\d+): (start|done)"
)

POOL_FOR_PREFIX = {"ci": "CI", "lw": "LW", "pgd": "PPGD"}

# Phases known to force CPU↔GPU sync (CPU wall ≈ real wait), vs. async enqueue
# (CPU wall ≈ kernel-launch time, real GPU work may overflow into next phase).
SYNC_HINTS: dict[str, str] = {
    "ci/1_ci_fn_fwd": "S(asserts)",
    "ci/3_imp_min": "S(nccl)",
    "ci/5_recv_g_ci_from_lw": "S(nccl)",
    "ci/6_recv_g_ci_from_ppgd": "S(nccl)",
    "ci/8b_bwd_imp_min_only": "S(nccl)",
    "ci/9_in_pool_allreduce": "S(nccl)",
    "ci/11_wait_sends": "S(nccl)",
    "lw/D2_wait_ci_recv": "S(nccl)",
    "pgd/D2_wait_ci_recv": "S(nccl)",
}


@dataclass
class PhaseEvent:
    rank: int
    t_ms: float
    name: str
    is_end: bool


@dataclass
class StepEvent:
    rank: int
    t_ms: float
    step: int
    is_done: bool


def parse_log(path: Path) -> tuple[list[PhaseEvent], list[StepEvent]]:
    phases: list[PhaseEvent] = []
    steps: list[StepEvent] = []
    with open(path) as f:
        for line in f:
            if m := STEP_RE.match(line):
                rank, t, step, kind = m.groups()
                steps.append(StepEvent(int(rank), float(t), int(step), kind == "done"))
                continue
            if m := PHASE_RE.match(line):
                rank, t, name, tail = m.groups()
                is_end = bool(tail and tail.lstrip().startswith("end"))
                phases.append(PhaseEvent(int(rank), float(t), name, is_end))
    return phases, steps


def step_window_per_rank(
    steps: list[StepEvent], rank: int, target_step: int
) -> tuple[float, float] | None:
    start: float | None = None
    done: float | None = None
    for ev in steps:
        if ev.rank != rank or ev.step != target_step:
            continue
        if ev.is_done:
            done = ev.t_ms
        else:
            start = ev.t_ms
    if start is None or done is None:
        return None
    return start, done


def per_rank_phases_in_window(
    phases: list[PhaseEvent], rank: int, t_start: float, t_end: float
) -> list[tuple[float, float, str]]:
    """Return list of (start_ms, end_ms, name) for completed phases in window."""
    open_starts: dict[str, float] = {}
    out: list[tuple[float, float, str]] = []
    for ev in phases:
        if ev.rank != rank:
            continue
        if ev.t_ms < t_start - 1 or ev.t_ms > t_end + 1:
            continue
        if not ev.is_end:
            open_starts[ev.name] = ev.t_ms
        else:
            s = open_starts.pop(ev.name, None)
            if s is not None:
                out.append((s, ev.t_ms, ev.name))
    return sorted(out, key=lambda x: x[0])


def short_name(name: str) -> str:
    """Strip pool prefix and trim trailing words. ci/8a_bwd_lower_leaky_only → 8a_bwd_lower."""
    if "/" in name:
        name = name.split("/", 1)[1]
    parts = name.split("_")
    if len(parts) > 4:
        parts = parts[:4]
    return "_".join(parts)


def render_bar(
    phases: list[tuple[float, float, str]],
    origin: float,
    span_ms: float,
    width: int,
) -> str:
    """Render one rank's phases as a fixed-width bar with phase boundaries.

    Each column ≈ ``span_ms/width`` ms. Phase boundary marks at start of each
    phase; phase name + duration printed below the bar in a legend.
    """
    cells = [" "] * width
    boundary_marks: list[tuple[int, str]] = []
    for s_ms, e_ms, name in phases:
        c_start = int((s_ms - origin) / span_ms * width)
        c_end = int((e_ms - origin) / span_ms * width)
        c_start = max(0, min(width - 1, c_start))
        c_end = max(c_start + 1, min(width, c_end))
        for c in range(c_start, c_end):
            cells[c] = "─"
        cells[c_start] = "┤"
        boundary_marks.append((c_start, name))
    return "".join(cells)


def legend(
    phases: list[tuple[float, float, str]],
    width: int,
    span_ms: float,
    origin: float,
    *,
    threshold_ms: float,
) -> list[str]:
    """Return list of legend lines: 'col_start name dur_ms hint'."""
    lines: list[str] = []
    for s_ms, e_ms, name in phases:
        dur = e_ms - s_ms
        if dur < threshold_ms:
            continue
        c_start = int((s_ms - origin) / span_ms * width)
        sync = SYNC_HINTS.get(name, "A")
        lines.append(
            f"  c{c_start:>3} +{s_ms - origin:>5.0f}ms  {dur:>5.0f}ms  [{sync:>10}]  {name}"
        )
    return lines


def render(
    log_path: Path,
    *,
    target_step: int,
    rank_labels: list[tuple[int, str]],
    width: int,
    legend_threshold_ms: float,
) -> str:
    phases, steps = parse_log(log_path)
    windows: dict[int, tuple[float, float]] = {}
    for rank, _ in rank_labels:
        w = step_window_per_rank(steps, rank, target_step)
        if w is None:
            return f"step {target_step} not found for rank {rank}"
        windows[rank] = w
    origin = min(s for s, _ in windows.values())
    end = max(e for _, e in windows.values())
    span = end - origin
    if span <= 0:
        return f"step {target_step} has non-positive span"

    out: list[str] = []
    out.append(
        f"step {target_step}: origin={origin:.0f}ms, span={span:.1f}ms, width={width}cols, "
        f"1col≈{span / width:.1f}ms"
    )
    out.append("")
    # tick line
    tick = []
    for c in range(width):
        ms = c * span / width
        if c % 10 == 0:
            tick.append("|")
        else:
            tick.append("·")
    out.append("        " + "".join(tick))
    # ms labels
    label_line = [" "] * (width + 8)
    for c in range(0, width, 10):
        ms = c * span / width
        s = f"{ms:.0f}"
        for i, ch in enumerate(s):
            if c + i < width:
                label_line[8 + c + i] = ch
    out.append("".join(label_line))
    # per-rank bars
    per_rank_data: list[tuple[str, list[tuple[float, float, str]]]] = []
    for rank, label in rank_labels:
        ps = per_rank_phases_in_window(phases, rank, *windows[rank])
        bar = render_bar(ps, origin, span, width)
        out.append(f"{label:<7} {bar}")
        per_rank_data.append((label, ps))
    out.append("")
    # legends per rank
    out.append(
        f"phases ≥ {legend_threshold_ms:.0f}ms  (S = forces CPU↔GPU sync, A = async enqueue)"
    )
    for label, ps in per_rank_data:
        out.append(f"-- {label} --")
        for ln in legend(ps, width, span, origin, threshold_ms=legend_threshold_ms):
            out.append(ln)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path, help="slurm log path")
    ap.add_argument("--step", type=int, default=3, help="step number to render (default: 3)")
    ap.add_argument("--width", type=int, default=100, help="bar width in columns")
    ap.add_argument(
        "--ranks",
        type=str,
        default="0:LW,96:CI,104:PPGD",
        help="comma-separated rank:label pairs",
    )
    ap.add_argument(
        "--legend-threshold-ms",
        type=float,
        default=5.0,
        help="only list phases ≥ this duration in the legend",
    )
    args = ap.parse_args()

    rank_labels = [
        (int(rank), label) for rank, label in (s.split(":") for s in args.ranks.split(","))
    ]
    print(
        render(
            args.log,
            target_step=args.step,
            rank_labels=rank_labels,
            width=args.width,
            legend_threshold_ms=args.legend_threshold_ms,
        )
    )


if __name__ == "__main__":
    main()
