"""Garbage-collect early-crashed runs — finds runs that are safely deletable.

A run dir (runs/<id> or jax_runs/<id> under $PARAM_DECOMP_OUT_DIR) is deletable
iff ALL of:
  - the invoking user owns it
  - it has checkpoints (numeric ckpts/<step> entries) — that's where the bytes are
  - its max checkpoint step is below --max-step: it crashed early, so the
    checkpoints are useless for analysis or resumption
  - no live SLURM job references it in a stdout log under <out_dir>/slurm_logs/.
    A healthy run passes through low steps early in training, so "early" alone
    is NOT watertight — this guard is what makes it so
  - its top-level entries were untouched for --quiet-minutes (race guard for
    jobs between submission and their first stdout lines)

Dry-run by default: prints the sized candidate list. --delete removes them.

    python -m param_decomp.tools.gc_dead_runs [--max-step 5000] [--delete]
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

RUN_BASES = ("runs", "jax_runs")
LIVE_JOB_STATES = ("RUNNING", "PENDING", "COMPLETING", "CONFIGURING", "REQUEUED")


@dataclass(frozen=True)
class Candidate:
    run_dir: Path
    rel: str
    max_step: int


def max_ckpt_step(run_dir: Path) -> int | None:
    ckpts = run_dir / "ckpts"
    if not ckpts.is_dir():
        return None
    steps = [int(p.name) for p in ckpts.iterdir() if p.name.isdigit()]
    return max(steps) if steps else None


def last_activity(run_dir: Path) -> float:
    """Newest mtime among the run dir, its top-level entries, and ckpt step dirs.
    Training touches logs/metrics constantly and checkpoint saves rename into
    ckpts/<step>, so top-two-levels mtime is a faithful liveness signal."""
    paths = [run_dir, *run_dir.iterdir()]
    ckpts = run_dir / "ckpts"
    if ckpts.is_dir():
        paths += list(ckpts.iterdir())
    return max(p.stat().st_mtime for p in paths)


def runs_referenced_by_live_jobs(out_dir: Path) -> set[str]:
    """Relative run paths (e.g. 'runs/p-8e2380e2') mentioned in the stdout of any
    live SLURM job that logs under <out_dir>/slurm_logs/ (the pd launcher
    convention). Jobs logging elsewhere are covered by the quiet-period guard."""
    scontrol = subprocess.run(
        ["scontrol", "show", "jobs", "-o"], capture_output=True, text=True, check=True
    )
    stdout_logs = []
    for line in scontrol.stdout.splitlines():
        state = re.search(r"JobState=(\S+)", line)
        stdout = re.search(r"StdOut=(\S+)", line)
        if not (state and stdout and state.group(1) in LIVE_JOB_STATES):
            continue
        log = Path(stdout.group(1))
        if log.is_relative_to(out_dir / "slurm_logs") and log.is_file():
            stdout_logs.append(log)
    if not stdout_logs:
        return set()
    grep = subprocess.run(
        ["grep", "-oahE", r"(runs|jax_runs)/[A-Za-z0-9_.-]+", *map(str, stdout_logs)],
        capture_output=True,
        text=True,
    )
    assert grep.returncode in (0, 1), grep.stderr
    return set(grep.stdout.splitlines())


def find_candidates(out_dir: Path, max_step: int, quiet_minutes: int) -> list[Candidate]:
    live_refs = runs_referenced_by_live_jobs(out_dir)
    quiet_cutoff = time.time() - quiet_minutes * 60
    candidates = []
    for base in RUN_BASES:
        for run_dir in sorted((out_dir / base).iterdir()):
            rel = f"{base}/{run_dir.name}"
            if run_dir.stat().st_uid != os.getuid():
                continue
            step = max_ckpt_step(run_dir)
            if step is None or step >= max_step:
                continue
            if rel in live_refs:
                print(
                    f"  skipping {rel} (max step {step}): referenced by a live job", file=sys.stderr
                )
                continue
            if last_activity(run_dir) > quiet_cutoff:
                print(
                    f"  skipping {rel} (max step {step}): active in the last {quiet_minutes} min",
                    file=sys.stderr,
                )
                continue
            candidates.append(Candidate(run_dir, rel, step))
    return candidates


def size_bytes(path: Path) -> int:
    du = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True, check=True)
    return int(du.stdout.split()[0])


def main() -> None:
    assert __doc__ is not None
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--max-step",
        type=int,
        default=5000,
        help="runs whose newest checkpoint is below this crashed early (default 5000)",
    )
    parser.add_argument(
        "--quiet-minutes",
        type=int,
        default=60,
        help="skip runs touched more recently than this (default 60)",
    )
    parser.add_argument(
        "--delete", action="store_true", help="actually delete (default: dry-run listing)"
    )
    args = parser.parse_args()

    out_dir = Path(os.environ["PARAM_DECOMP_OUT_DIR"])
    assert out_dir.is_dir(), f"PARAM_DECOMP_OUT_DIR does not exist: {out_dir}"

    candidates = find_candidates(out_dir, args.max_step, args.quiet_minutes)
    if not candidates:
        print("Nothing to collect.")
        return

    with ThreadPoolExecutor(max_workers=8) as pool:
        sizes = list(pool.map(lambda c: size_bytes(c.run_dir), candidates))

    verb = "deleting" if args.delete else "would delete"
    total = 0
    for cand, size in sorted(zip(candidates, sizes, strict=True), key=lambda cs: -cs[1]):
        total += size
        print(f"{size / 2**40:8.2f} TiB  {cand.rel}  (max ckpt step {cand.max_step}) — {verb}")
        if args.delete:
            shutil.rmtree(cand.run_dir)
    action = "Deleted" if args.delete else "Would delete"
    print(f"\n{action} {len(candidates)} runs, {total / 2**40:.2f} TiB total.")


if __name__ == "__main__":
    main()
