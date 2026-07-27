"""Reclaim the trainer-only half of checkpoints that will never be resumed.

A PD checkpoint step holds two orbax items (see `param_decomp/checkpoint.py`):
`decomposition` — the trained product every consumer reads — and `training` — the
optimizer moments, persistent adversaries and step counter that only trainer resume
touches. On the LM runs `training` is 80-90% of the bytes, and a run keeps one copy
per retained step, so a `keep_last_n_checkpoints: 40` job carries ~40x the trajectory
tail nobody will ever read. Three cluster-wide storage sweeps have gone at this by
hand; this is that operation, committed.

Thinning removes `ckpts/<step>/training` and nothing else. The step keeps its
`decomposition`, so harvest, autointerp, clustering, CI statistics and the offline
PGD anatomy all still load it (`param_decomp_lab.experiments.lm.load_run.open_jax_run`
restores the `decomposition` item alone). What a thinned step loses is resumability:
`restore_latest` asks for both items and raises on the missing one.

That loss is why `--keep-newest` exists and defaults to `RESUME_WINDOW`. A running
trainer holds its `CheckpointManager` in memory and never re-reads the directory, so
thinning older steps underneath it is invisible to it; what would bite is a SLURM
requeue landing on a newest step whose `training` is gone. Runs referenced by a live
job are therefore floored at `RESUME_WINDOW` however low `--keep-newest` is set.

    python -m param_decomp_lab.tools.thin_training_state                # dry run, whole out dir
    python -m param_decomp_lab.tools.thin_training_state --delete
    python -m param_decomp_lab.tools.thin_training_state --keep_newest=0 --run_ids=p-abc12345

`_CHECKPOINT_METADATA` is deliberately left listing both items: it is read only by an
argument-less composite restore, which nothing in this repo issues, and rewriting a
file inside a live run's checkpoint tree buys nothing.
"""

import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import fire

from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR, SLURM_LOGS_DIR

RESUME_WINDOW = 2
"""Newest checkpoints a run keeps resumable: the newest, plus one for the requeue that
lands while a save is in flight."""

RUN_ID = re.compile(rb"p-[0-9a-f]{8}")
LIVE_STATES = ("RUNNING", "PENDING", "COMPLETING", "CONFIGURING", "REQUEUED")
TRASH_DIR = ".thin_trash"


@dataclass(frozen=True)
class Checkpoint:
    step: int
    training: Path
    nbytes: int


@dataclass(frozen=True)
class Run:
    run_id: str
    live: bool
    thinnable: tuple[Checkpoint, ...]
    """Ascending by step; steps already missing their `training` item are absent."""


@dataclass(frozen=True)
class Thin:
    run: Run
    victims: tuple[Checkpoint, ...]


@dataclass(frozen=True)
class Skip:
    run_id: str
    reason: Literal["not-owned", "no-training-item", "inside-resume-window"]


RunPlan = Thin | Skip


# ---------------------------------------------------------------- pure core


def plan_run(run: Run, keep_newest: int) -> RunPlan:
    """Which of a run's `training` items are safe to drop."""
    keep = max(keep_newest, RESUME_WINDOW) if run.live else keep_newest
    victims = run.thinnable[: len(run.thinnable) - keep]
    if not victims:
        return Skip(run.run_id, "inside-resume-window")
    return Thin(run, tuple(victims))


def reclaimed(plans: list[RunPlan]) -> int:
    return sum(c.nbytes for p in plans if isinstance(p, Thin) for c in p.victims)


def terabytes(nbytes: int) -> str:
    return f"{nbytes / 1e12:.3f} TB"


# ---------------------------------------------------------------- the filesystem edge


def tree_bytes(root: Path) -> int:
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())


def read_run(run_dir: Path, live_run_ids: frozenset[str]) -> Run | Skip:
    if run_dir.stat().st_uid != os.getuid():
        return Skip(run_dir.name, "not-owned")
    thinnable = sorted(
        (
            Checkpoint(int(step_dir.name), training, tree_bytes(training))
            for step_dir in run_dir.glob("ckpts/*")
            if step_dir.name.isdigit()
            if (training := step_dir / "training").is_dir()
        ),
        key=lambda c: c.step,
    )
    if not thinnable:
        return Skip(run_dir.name, "no-training-item")
    return Run(run_dir.name, run_dir.name in live_run_ids, tuple(thinnable))


def live_run_ids() -> frozenset[str]:
    """Run ids named in the stdout of every SLURM job that is not finished.

    The trainer announces its run id into its own log, so the log is the one honest
    job -> run link (`scontrol`'s comment field is free-form and often absent). A live
    job whose log we cannot read fails the sweep rather than silently freeing its run.
    """
    jobs = subprocess.run(
        ["scontrol", "show", "jobs", "-o"], capture_output=True, text=True, check=True
    ).stdout
    logs = {
        Path(m.group(1))
        for line in jobs.splitlines()
        if any(f"JobState={state}" in line for state in LIVE_STATES)
        if (m := re.search(r"StdOut=(\S+)", line))
        if m.group(1).startswith(str(SLURM_LOGS_DIR))
    }
    return frozenset(
        m.group().decode()
        for log in logs
        if log.exists()
        for m in RUN_ID.finditer(log.read_bytes())
    )


def sweep(out_dir: Path, run_ids: tuple[str, ...] | None, keep_newest: int) -> list[RunPlan]:
    live = live_run_ids()
    run_dirs = (
        sorted((out_dir / "runs").glob("p-*"))
        if run_ids is None
        else [out_dir / "runs" / run_id for run_id in run_ids]
    )
    return [
        read if isinstance(read := read_run(d, live), Skip) else plan_run(read, keep_newest)
        for d in run_dirs
    ]


def execute(plan: Thin, trash: Path) -> Iterator[Checkpoint]:
    """Rename each victim out of the checkpoint tree before deleting it, so a concurrent
    reader sees the item either whole or absent — never half-deleted."""
    for victim in plan.victims:
        staged = trash / f"{plan.run.run_id}-{victim.step}-training"
        victim.training.rename(staged)
        shutil.rmtree(staged)
        yield victim


# ---------------------------------------------------------------- CLI


def main(
    out_dir: str | Path = PARAM_DECOMP_OUT_DIR,
    run_ids: str | tuple[str, ...] | None = None,
    keep_newest: int = RESUME_WINDOW,
    delete: bool = False,
) -> None:
    """Report (or, with --delete, drop) thinnable `training` items.

    out_dir: run root; defaults to $PARAM_DECOMP_OUT_DIR.
    run_ids: restrict to these runs; default sweeps every run in `out_dir/runs`.
    keep_newest: newest checkpoints per run left resumable. Live runs are floored at
        RESUME_WINDOW. 0 thins every checkpoint of every non-live run — that forfeits
        resumption for good, so pass it deliberately.
    delete: actually delete. Default reports the plan and touches nothing.
    """
    root = Path(out_dir)
    only = (run_ids,) if isinstance(run_ids, str) else run_ids
    plans = sweep(root, only, keep_newest)

    for plan in sorted(
        (p for p in plans if isinstance(p, Thin)),
        key=lambda p: -sum(c.nbytes for c in p.victims),
    ):
        steps = [c.step for c in plan.victims]
        live = " LIVE" if plan.run.live else ""
        size = terabytes(sum(c.nbytes for c in plan.victims))
        print(f"{plan.run.run_id}{live}  {size}  {len(steps)} steps {steps[0]}..{steps[-1]}")

    total = reclaimed(plans)
    print(f"\n{sum(isinstance(p, Thin) for p in plans)} runs, {terabytes(total)} reclaimable")
    if not delete:
        print("dry run — pass --delete to remove")
        return

    trash = root / TRASH_DIR
    trash.mkdir(exist_ok=True)
    freed = sum(
        victim.nbytes for plan in plans if isinstance(plan, Thin) for victim in execute(plan, trash)
    )
    trash.rmdir()
    print(f"freed {terabytes(freed)}")


if __name__ == "__main__":
    fire.Fire(main)
