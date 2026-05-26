"""Per-phase wall-clock breakdown from a slurm log's PD_PHASE_TRACE output.

Parses ``[trace rank=R +Tms] phase: NAME ...`` lines emitted by
``PhaseProfiler.phase`` when ``PD_PHASE_TRACE=1``. For each (rank, phase)
pair, computes the time between consecutive entry traces (= time spent
inside that phase + whatever immediately follows up to the next phase
boundary, since exit traces share the same line as the next entry's "cur=").

Outputs per-rank phase breakdowns: total, mean, percent-of-step. Useful for
finding the dominant cost in a step ("we spend 60% of every step in
lw/D3_layerwise", "ci/2_async_send_ci is 15% on every CI step", etc.) so we
know what to optimize for throughput.

Usage:
    python scripts/analyze_step_times.py /path/to/slurm-NNNNN.out
    python scripts/analyze_step_times.py /path/to/slurm-NNNNN.out --rank=0
    python scripts/analyze_step_times.py /path/to/slurm-NNNNN.out --skip-warmup=1
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import fire

# Match only phase ENTRY traces ("phase: NAME cur=..."), not the "end peak="
# exit traces — otherwise each phase gets counted twice per step and the
# "duration" alternates between (real phase work) and (inter-phase gap).
_PHASE_RE = re.compile(r"\[trace rank=(\d+) \+\s*([0-9.]+)ms\] phase: (\S+) cur=")
# Phase EXIT trace with the new cpu=/gpu=/wait= columns. The exit line also
# carries peak/end/delta, but we only need the timing here. Older logs lacking
# these columns simply won't match — analyzer falls back to phase-entry timing.
_PHASE_EXIT_RE = re.compile(
    r"\[trace rank=(\d+) \+\s*[0-9.]+ms\] phase: (\S+) end .*"
    r"cpu=([0-9.]+)ms gpu=([0-9.]+)ms wait=([+-]?[0-9.]+)ms"
)
# [trace rank=R +Tms] Trainer.run: step N: done in Yms
_STEP_DONE_RE = re.compile(
    r"\[trace rank=(\d+) \+\s*([0-9.]+)ms\] Trainer\.run: step (\d+): done in ([0-9.]+)ms"
)
# [trace rank=R +Tms] Trainer.run: step N: start ...
_STEP_START_RE = re.compile(
    r"\[trace rank=(\d+) \+\s*([0-9.]+)ms\] Trainer\.run: step (\d+): start"
)


def _bar(frac: float, width: int = 30) -> str:
    n = round(frac * width)
    return "█" * n + "·" * (width - n)


def main(log_path: str, rank: int | None = None, skip_warmup: int = 1) -> None:
    """Break down step time by phase.

    Args:
        log_path: Path to the slurm-NNN.out file.
        rank: If set, only analyze this rank. Otherwise pick one rank per pool
            from the traces found in the log.
        skip_warmup: Skip first N steps from per-phase aggregation (step 0 is
            usually cold-start dominated and not representative).
    """
    path = Path(log_path)
    assert path.exists(), f"log not found: {path}"

    # Parse all phase + step events per rank.
    # Per rank: ordered list of (timestamp_ms, kind, payload)
    events: dict[int, list[tuple[float, str, str]]] = defaultdict(list)
    step_times: dict[int, dict[int, float]] = defaultdict(dict)
    step_starts: dict[int, dict[int, float]] = defaultdict(dict)
    # Per (rank, phase): list of (cpu_ms, gpu_ms, wait_ms) from exit traces.
    # We track which step each exit belongs to so we can honour ``skip_warmup``.
    gpu_samples: dict[int, dict[str, list[tuple[int, float, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    # Track the most recent step seen per rank so exit traces can attribute
    # themselves to a step. The exit trace itself has no step number.
    current_step_per_rank: dict[int, int] = defaultdict(lambda: -1)

    with open(path) as f:
        for line in f:
            if m := _STEP_DONE_RE.search(line):
                r, t, step, dur = int(m[1]), float(m[2]), int(m[3]), float(m[4])
                step_times[r][step] = dur
                events[r].append((t, "step_done", str(step)))
                current_step_per_rank[r] = -1
            elif m := _STEP_START_RE.search(line):
                r, t, step = int(m[1]), float(m[2]), int(m[3])
                step_starts[r][step] = t
                events[r].append((t, "step_start", str(step)))
                current_step_per_rank[r] = step
            elif m := _PHASE_EXIT_RE.search(line):
                r, name = int(m[1]), m[2]
                cpu_ms, gpu_ms, wait_ms = float(m[3]), float(m[4]), float(m[5])
                step = current_step_per_rank[r]
                gpu_samples[r][name].append((step, cpu_ms, gpu_ms, wait_ms))
            elif m := _PHASE_RE.search(line):
                r, t, name = int(m[1]), float(m[2]), m[3]
                events[r].append((t, "phase", name))

    if rank is None:
        # Pick one rank per pool — first rank we see by name prefix.
        ranks_by_prefix: dict[str, int] = {}
        for r, evs in events.items():
            for _, kind, name in evs:
                if kind == "phase":
                    prefix = name.split("/", 1)[0]
                    if prefix not in ranks_by_prefix:
                        ranks_by_prefix[prefix] = r
                    break
        ranks_to_show = sorted(ranks_by_prefix.values())
    else:
        ranks_to_show = [rank]

    for r in ranks_to_show:
        evs = events[r]
        if not evs:
            print(f"rank={r}: no traces found")
            continue

        # Compute per-phase durations: time from one phase entry to the next.
        phase_durations: dict[str, list[float]] = defaultdict(list)
        current_step = -1
        current_phase: tuple[str, float] | None = None  # (name, start_ms)

        for t, kind, payload in evs:
            if kind == "step_start":
                current_step = int(payload)
                current_phase = None
            elif kind == "step_done":
                if current_phase is not None and current_step >= skip_warmup:
                    name, start_ms = current_phase
                    phase_durations[name].append(t - start_ms)
                current_phase = None
                current_step = -1
            elif kind == "phase":
                if current_phase is not None and current_step >= skip_warmup:
                    name, start_ms = current_phase
                    phase_durations[name].append(t - start_ms)
                current_phase = (payload, t)

        if not phase_durations:
            print(f"rank={r}: no phase data after skip_warmup={skip_warmup}")
            continue

        # Mean per phase, then % of total mean step.
        means = {name: sum(ds) / len(ds) for name, ds in phase_durations.items()}
        step_mean = sum(d for s, d in step_times[r].items() if s >= skip_warmup) / max(
            1, len([s for s in step_times[r] if s >= skip_warmup])
        )

        # Per-phase GPU/wait means from exit traces (post-warmup samples only).
        # Empty dict if the log predates the cpu=/gpu=/wait= format.
        gpu_means: dict[str, tuple[float, float]] = {}  # name -> (gpu_ms, wait_ms)
        for name, samples in gpu_samples[r].items():
            post_warmup = [(g, w) for s, _c, g, w in samples if s >= skip_warmup]
            if not post_warmup:
                continue
            gpu_mean = sum(g for g, _ in post_warmup) / len(post_warmup)
            wait_mean = sum(w for _, w in post_warmup) / len(post_warmup)
            gpu_means[name] = (gpu_mean, wait_mean)
        has_gpu = bool(gpu_means)

        # Group by pool prefix (lw/, ci/, pgd/) for tidy output.
        pool = next(iter(means)).split("/", 1)[0]
        print(f"\n=== rank={r} (pool={pool}) — mean step {step_mean:.1f}ms ===")
        if has_gpu:
            print(
                f"{'phase':40s} {'cpu_ms':>8s} {'gpu_ms':>8s} {'wait_ms':>8s} "
                f"{'count':>5s} {'%step':>6s}"
            )
            print("-" * 90)
        else:
            print(f"{'phase':40s} {'mean_ms':>8s} {'count':>5s} {'%step':>6s}")
            print("-" * 72)
        sum_phase = 0.0
        for name, mean_ms in sorted(means.items(), key=lambda kv: -kv[1]):
            count = len(phase_durations[name])
            pct = 100 * mean_ms / step_mean if step_mean > 0 else 0
            sum_phase += mean_ms
            bar = _bar(mean_ms / step_mean if step_mean > 0 else 0)
            if has_gpu:
                gpu_ms, wait_ms = gpu_means.get(name, (float("nan"), float("nan")))
                print(
                    f"{name:40s} {mean_ms:8.1f} {gpu_ms:8.1f} {wait_ms:+8.1f} "
                    f"{count:5d} {pct:5.1f}%  {bar}"
                )
            else:
                print(f"{name:40s} {mean_ms:8.1f} {count:5d} {pct:5.1f}%  {bar}")
        sep_width = 90 if has_gpu else 72
        print("-" * sep_width)
        overhead = step_mean - sum_phase
        print(f"{'(phase sum)':40s} {sum_phase:8.1f} {'':5s} {100 * sum_phase / step_mean:5.1f}%")
        print(
            f"{'(unaccounted: sync/log/io)':40s} {overhead:8.1f} "
            f"{'':5s} {100 * overhead / step_mean if step_mean > 0 else 0:5.1f}%"
        )


if __name__ == "__main__":
    fire.Fire(main)
