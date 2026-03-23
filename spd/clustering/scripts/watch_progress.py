"""Watch clustering harvest and merge progress from SLURM jobs and logs."""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

SLURM_LOG_DIR = Path("/mnt/polished-lake/artifacts/mechanisms/spd/slurm_logs")
DEFAULT_NAME_PREFIX = "jose_"

SQUEUE_FORMAT = "%i|%T|%R|%M|%S|%j"
SACCT_FORMAT = "JobIDRaw,State,Reason,Elapsed,Start,JobName%40"

MERGE_PROGRESS_RE = re.compile(
    r"Compressed merge progress: iter=(?P<iter>\d+)/(?P<total>\d+), "
    r"elapsed=(?P<elapsed>[0-9.]+)s, sec_per_iter=(?P<sec>[0-9.]+), "
    r"k_groups=(?P<groups>\d+)"
)
CSR_RE = re.compile(r"Built component activity CSR in (?P<secs>[0-9.]+)s")
COACT_RE = re.compile(r"Built coactivation matrix in (?P<secs>[0-9.]+)s")
LOAD_RE = re.compile(r"Loaded: (?P<comps>\d+) components, (?P<samples>\d+) samples")
COLLECT_RE = re.compile(
    r"Collected (?P<collected>\d+) token activations \(requested (?P<requested>\d+)\)"
)
SAVE_RE = re.compile(
    r"Saving snapshot: (?P<comps>\d+) alive components, (?P<samples>\d+) samples"
)


@dataclass
class SchedulerInfo:
    job_id: str
    state: str
    reason: str
    elapsed: str
    start_time: str
    name: str


@dataclass
class LogSummary:
    stage: str
    detail: str
    progress: float | None = None
    eta: str | None = None


@dataclass
class JobStatus:
    scheduler: SchedulerInfo
    log: LogSummary


def _infer_progress(stage: str, *, ratio: float | None = None) -> float | None:
    if stage == "done":
        return 1.0
    if stage == "collect" and ratio is not None:
        return 0.02 + 0.90 * ratio
    if stage == "save":
        return 0.94
    if stage == "load-model":
        return 0.01
    if stage == "load":
        return 0.02
    if stage == "loaded":
        return 0.04
    if stage == "merge-start":
        return 0.05
    if stage == "csr-done":
        return 0.10
    if stage == "coact":
        return 0.12
    if stage == "coact-done":
        return 0.15
    if stage == "merge" and ratio is not None:
        return 0.15 + 0.85 * ratio
    return None


def _job_kind(name: str) -> str:
    if "harvest" in name:
        return "harvest"
    if "merge" in name:
        return "merge"
    return "other"


def _run_lines(cmd: list[str]) -> list[str]:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def expand_job_tokens(job_tokens: list[str]) -> list[str]:
    expanded: list[str] = []
    for token in job_tokens:
        for part in token.split(","):
            part = part.strip()
            if not part:
                continue
            if "_" in part:
                expanded.append(part)
                continue
            if "-" in part:
                start_str, end_str = part.split("-", maxsplit=1)
                start, end = int(start_str), int(end_str)
                step = 1 if end >= start else -1
                expanded.extend(str(job_id) for job_id in range(start, end + step, step))
                continue
            expanded.append(part)
    return expanded


def get_squeue_jobs(job_ids: list[str] | None = None, name_prefix: str | None = None) -> dict[str, SchedulerInfo]:
    cmd = ["squeue", "-h", "-o", SQUEUE_FORMAT]
    if job_ids:
        cmd.extend(["-j", ",".join(job_ids)])
    lines = _run_lines(cmd)
    jobs: dict[str, SchedulerInfo] = {}
    for line in lines:
        try:
            job_id, state, reason, elapsed, start_time, name = line.split("|", maxsplit=5)
        except ValueError:
            continue
        if name_prefix and not name.startswith(name_prefix):
            continue
        jobs[job_id] = SchedulerInfo(
            job_id=job_id,
            state=state,
            reason=reason,
            elapsed=elapsed,
            start_time=start_time,
            name=name,
        )
    return jobs


def get_sacct_jobs(job_ids: list[str]) -> dict[str, SchedulerInfo]:
    if not job_ids:
        return {}
    lines = _run_lines(
        [
            "sacct",
            "-n",
            "-P",
            "-X",
            "-j",
            ",".join(job_ids),
            "--format",
            SACCT_FORMAT,
        ]
    )
    jobs: dict[str, SchedulerInfo] = {}
    for line in lines:
        try:
            job_id, state, reason, elapsed, start_time, name = line.split("|", maxsplit=5)
        except ValueError:
            continue
        if "." in job_id:
            continue
        jobs[job_id] = SchedulerInfo(
            job_id=job_id,
            state=state,
            reason=reason or "-",
            elapsed=elapsed or "-",
            start_time=start_time or "-",
            name=name or "-",
        )
    return jobs


def discover_job_ids(name_prefix: str) -> list[str]:
    return sorted(get_squeue_jobs(name_prefix=name_prefix).keys(), key=_job_sort_key)


def _job_sort_key(job_id: str) -> tuple[int, str]:
    job_num = job_id.split("_", maxsplit=1)[0]
    return (int(job_num), job_id)


def find_log_path(job_id: str) -> Path | None:
    direct = SLURM_LOG_DIR / f"slurm-{job_id}.out"
    if direct.exists():
        return direct

    matches = sorted(glob.glob(str(SLURM_LOG_DIR / f"slurm-{job_id}_*.out")))
    if matches:
        return Path(matches[0])
    return None


def summarize_log(log_path: Path | None) -> LogSummary:
    if log_path is None or not log_path.exists():
        return LogSummary(stage="no-log", detail="-")

    try:
        text = log_path.read_text(errors="replace")
    except OSError as exc:
        return LogSummary(stage="log-error", detail=str(exc))

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return LogSummary(stage="empty-log", detail="-")

    for line in reversed(lines):
        if "History saved to" in line:
            return LogSummary(stage="done", detail="history saved", progress=1.0)

        match = MERGE_PROGRESS_RE.search(line)
        if match:
            iter_idx = int(match.group("iter"))
            total = int(match.group("total"))
            sec_per_iter = float(match.group("sec"))
            remaining = max(total - iter_idx, 0)
            eta = _format_seconds(remaining * sec_per_iter)
            ratio = iter_idx / max(total, 1)
            detail = (
                f"{iter_idx}/{total} iters, {sec_per_iter:.3f}s/it, "
                f"groups={match.group('groups')}, eta={eta}"
            )
            return LogSummary(
                stage="merge",
                detail=detail,
                progress=_infer_progress("merge", ratio=ratio),
                eta=eta,
            )

        if "Building coactivation matrix from compressed memberships" in line:
            return LogSummary(
                stage="coact",
                detail="building coactivation matrix",
                progress=_infer_progress("coact"),
            )

        match = COACT_RE.search(line)
        if match:
            return LogSummary(
                stage="coact-done",
                detail=f"coact in {match.group('secs')}s",
                progress=_infer_progress("coact-done"),
            )

        match = CSR_RE.search(line)
        if match:
            return LogSummary(
                stage="csr-done",
                detail=f"csr in {match.group('secs')}s",
                progress=_infer_progress("csr-done"),
            )

        match = LOAD_RE.search(line)
        if match:
            return LogSummary(
                stage="loaded",
                detail=f"{match.group('comps')} comps, {match.group('samples')} samples",
                progress=_infer_progress("loaded"),
            )

        if "Loading snapshot from" in line:
            return LogSummary(stage="load", detail="loading snapshot", progress=_infer_progress("load"))

        if "Starting merge" in line:
            return LogSummary(
                stage="merge-start",
                detail=line.split("Starting merge ", maxsplit=1)[-1],
                progress=_infer_progress("merge-start"),
            )

        if "Harvest complete:" in line:
            return LogSummary(stage="done", detail="harvest saved", progress=1.0)

        match = SAVE_RE.search(line)
        if match:
            return LogSummary(
                stage="save",
                detail=f"{match.group('samples')} samples, {match.group('comps')} comps",
                progress=_infer_progress("save"),
            )

        match = COLLECT_RE.search(line)
        if match:
            collected = int(match.group("collected"))
            requested = int(match.group("requested"))
            pct = 100.0 * collected / max(requested, 1)
            return LogSummary(
                stage="collect",
                detail=f"{collected}/{requested} tokens ({pct:.1f}%)",
                progress=_infer_progress("collect", ratio=collected / max(requested, 1)),
            )

        if "Collecting memberships" in line:
            return LogSummary(stage="collect", detail="collecting memberships")

        if "Loading model" in line:
            return LogSummary(
                stage="load-model",
                detail="loading model",
                progress=_infer_progress("load-model"),
            )

    return LogSummary(stage="started", detail=lines[-1][:80])


def _format_seconds(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def build_rows(job_ids: list[str], name_prefix: str | None) -> list[JobStatus]:
    live = get_squeue_jobs(job_ids=job_ids or None, name_prefix=name_prefix)
    requested_ids = job_ids or sorted(live.keys(), key=_job_sort_key)
    historic = get_sacct_jobs([job_id for job_id in requested_ids if job_id not in live])

    rows: list[JobStatus] = []
    for job_id in sorted(set(requested_ids) | set(live) | set(historic), key=_job_sort_key):
        info = live.get(job_id) or historic.get(job_id)
        if info is None:
            continue
        log_summary = summarize_log(find_log_path(job_id))
        rows.append(JobStatus(scheduler=info, log=log_summary))
    return rows


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = min(max(widths[idx], len(cell)), 80)

    def fmt_row(row: list[str]) -> str:
        clipped = []
        for idx, cell in enumerate(row):
            width = widths[idx]
            clipped.append(cell if len(cell) <= width else f"{cell[: width - 1]}…")
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(clipped))

    lines = [fmt_row(headers), fmt_row(["-" * width for width in widths])]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def _progress_bar(progress: float | None, width: int) -> str:
    if progress is None:
        return "[" + "." * width + "]"
    clamped = max(0.0, min(progress, 1.0))
    filled = int(round(clamped * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _render_section(title: str, jobs: list[JobStatus], width: int) -> str:
    if not jobs:
        return ""

    bar_width = max(12, min(28, width // 5))
    name_width = max(16, min(24, width // 6))
    state_width = 10
    elapsed_width = 8
    stage_width = 10

    lines = [title]
    for job in jobs:
        sched = job.scheduler
        log = job.log
        progress_pct = "  ??.?%" if log.progress is None else f"{100 * log.progress:6.1f}%"
        name = sched.name[:name_width].ljust(name_width)
        state = sched.state[:state_width].ljust(state_width)
        elapsed = sched.elapsed[:elapsed_width].ljust(elapsed_width)
        stage = log.stage[:stage_width].ljust(stage_width)
        detail_width = max(20, width - (len(job.scheduler.job_id) + name_width + state_width + elapsed_width + stage_width + bar_width + 18))
        detail = log.detail[:detail_width]
        lines.append(
            f"{sched.job_id}  {name}  {state}  {elapsed}  {stage}  "
            f"{_progress_bar(log.progress, bar_width)} {progress_pct}  {detail}"
        )
    return "\n".join(lines)


def _render_summary(jobs: list[JobStatus]) -> list[str]:
    total = len(jobs)
    running = sum(1 for job in jobs if job.scheduler.state == "RUNNING")
    pending = sum(1 for job in jobs if job.scheduler.state == "PENDING")
    done = sum(1 for job in jobs if job.scheduler.state in {"COMPLETED", "COMPLETING"} or job.log.stage == "done")
    failed = sum(1 for job in jobs if job.scheduler.state in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"} or "oom" in job.scheduler.state.lower())
    known = [job.log.progress for job in jobs if job.log.progress is not None]
    avg_progress = sum(known) / len(known) if known else 0.0
    return [
        f"Jobs: {total}  Running: {running}  Pending: {pending}  Done: {done}  Failed: {failed}",
        f"Overall progress: {_progress_bar(avg_progress, 32)} {100 * avg_progress:5.1f}%",
    ]


def render(job_ids: list[str], name_prefix: str | None) -> str:
    rows = build_rows(job_ids=job_ids, name_prefix=name_prefix)
    title = f"Cluster watch at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
    if not rows:
        return f"{title}\nNo matching jobs found."
    width = shutil.get_terminal_size((160, 40)).columns
    lines = [title, *_render_summary(rows), ""]

    running_harvests = [job for job in rows if job.scheduler.state == "RUNNING" and _job_kind(job.scheduler.name) == "harvest"]
    running_merges = [job for job in rows if job.scheduler.state == "RUNNING" and _job_kind(job.scheduler.name) == "merge"]
    pending = [job for job in rows if job.scheduler.state == "PENDING"]
    finished = [
        job
        for job in rows
        if job.scheduler.state not in {"RUNNING", "PENDING"}
    ]

    sections = [
        _render_section("Running Merges", running_merges, width),
        _render_section("Running Harvests", running_harvests, width),
        _render_section("Pending", pending, width),
        _render_section("Finished", finished, width),
    ]
    lines.extend(section for section in sections if section)
    return "\n".join(lines)


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="spd-cluster-watch",
        description="Watch clustering harvest and merge progress from SLURM jobs and logs.",
    )
    parser.add_argument(
        "job_ids",
        nargs="*",
        help="Job IDs, comma-separated lists, or numeric ranges like 376819-376832.",
    )
    parser.add_argument(
        "--name-prefix",
        default=None,
        help=f"If no job IDs are given, watch active jobs starting with this prefix. Defaults to {DEFAULT_NAME_PREFIX!r}.",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=10.0,
        help="Refresh interval in seconds for watch mode.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one snapshot and exit.",
    )
    args = parser.parse_args()

    job_ids = expand_job_tokens(args.job_ids)
    name_prefix = args.name_prefix
    if not job_ids:
        name_prefix = DEFAULT_NAME_PREFIX if name_prefix is None else name_prefix
        job_ids = discover_job_ids(name_prefix)

    if args.once:
        print(render(job_ids=job_ids, name_prefix=name_prefix))
        return

    while True:
        if os.environ.get("TERM"):
            print("\033[2J\033[H", end="")
        print(render(job_ids=job_ids, name_prefix=name_prefix))
        time.sleep(args.refresh)


if __name__ == "__main__":
    cli()
