"""Unit tests for scratch cleanup — the two complementary mechanisms.

CPU-only, no torch: pure filesystem logic.

  * ``_prune_scratch_through``: consolidation drops everything at/below the step it
    just finished — so a completed run's scratch goes fully to zero.
  * ``prune_old_scratch``: the train-loop backstop keeps only the newest N snapshots,
    bounding scratch when consolidation never runs at all.
"""

from pathlib import Path

import pytest

from param_decomp_lab.three_pool.consolidate import (
    SNAPSHOT_SCRATCH_DIRNAME,
    _prune_scratch_through,
    prune_old_scratch,
)


def _make_scratch(tmp_path: Path, steps: list[int]) -> Path:
    scratch = tmp_path / SNAPSHOT_SCRATCH_DIRNAME
    for s in steps:
        (scratch / f"step_{s}").mkdir(parents=True)
    return scratch


def _steps(scratch: Path) -> set[int]:
    return {int(d.name.removeprefix("step_")) for d in scratch.glob("step_*")}


def test_through_drops_at_or_below_step_keeps_newer(tmp_path: Path) -> None:
    scratch = _make_scratch(tmp_path, [20, 40, 60])
    _prune_scratch_through(scratch, step=40)
    assert _steps(scratch) == {60}  # 20, 40 consolidated/superseded; 60 still pending


def test_through_final_step_clears_a_finished_run(tmp_path: Path) -> None:
    # The highest step's consolidation sweeps everything — even a backlog of earlier,
    # never-consolidated steps — so a completed run leaves zero scratch (not N).
    scratch = _make_scratch(tmp_path, [20, 40, 60])
    _prune_scratch_through(scratch, step=60)
    assert _steps(scratch) == set()


def test_old_keeps_newest_n_drops_the_rest(tmp_path: Path) -> None:
    scratch = _make_scratch(tmp_path, [20, 40, 60])
    prune_old_scratch(scratch, keep_last_n=2)
    assert _steps(scratch) == {40, 60}  # ordered by step number, not name/mtime


def test_old_noop_when_at_or_under_limit(tmp_path: Path) -> None:
    scratch = _make_scratch(tmp_path, [10])
    prune_old_scratch(scratch, keep_last_n=2)
    assert _steps(scratch) == {10}


def test_noop_when_scratch_absent(tmp_path: Path) -> None:
    absent = tmp_path / SNAPSHOT_SCRATCH_DIRNAME
    prune_old_scratch(absent)  # glob on a missing dir is empty — must not raise
    _prune_scratch_through(absent, step=10)


def test_old_keep_last_n_must_be_at_least_one(tmp_path: Path) -> None:
    # keep_last_n=0 would delete the just-written step out from under its consolidation.
    with pytest.raises(AssertionError):
        prune_old_scratch(_make_scratch(tmp_path, [10]), keep_last_n=0)
