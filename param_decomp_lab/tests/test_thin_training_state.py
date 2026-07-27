"""The checkpoint thinner: which `training` items a sweep drops, and that dropping one
leaves the step's `decomposition` and the rest of the run untouched."""

from pathlib import Path

import pytest
from pytest import MonkeyPatch

from param_decomp_lab.tools.thin_training_state import (
    RESUME_WINDOW,
    Checkpoint,
    Run,
    Skip,
    Thin,
    execute,
    plan_run,
    read_run,
)


def _run(run_id: str, steps: list[int], live: bool) -> Run:
    return Run(
        run_id,
        live,
        tuple(Checkpoint(step, Path(f"/{run_id}/{step}/training"), 1) for step in steps),
    )


def _write_ckpts(run_dir: Path, steps: list[int], with_training: bool = True) -> None:
    for step in steps:
        step_dir = run_dir / "ckpts" / str(step)
        (step_dir / "decomposition" / "d").mkdir(parents=True)
        (step_dir / "decomposition" / "d" / "chunk").write_bytes(b"product")
        (step_dir / "_CHECKPOINT_METADATA").write_text("{}")
        if with_training:
            (step_dir / "training" / "d").mkdir(parents=True)
            (step_dir / "training" / "d" / "chunk").write_bytes(b"trajectory tail")


def test_keeps_the_resume_window():
    plan = plan_run(_run("p-aaaaaaaa", [1, 2, 3, 4], live=False), keep_newest=2)
    assert isinstance(plan, Thin)
    assert [c.step for c in plan.victims] == [1, 2]


def test_zero_keep_thins_everything_on_a_dead_run():
    plan = plan_run(_run("p-aaaaaaaa", [1, 2], live=False), keep_newest=0)
    assert isinstance(plan, Thin)
    assert [c.step for c in plan.victims] == [1, 2]


def test_live_run_is_floored_at_the_resume_window():
    plan = plan_run(_run("p-aaaaaaaa", list(range(10)), live=True), keep_newest=0)
    assert isinstance(plan, Thin)
    assert [c.step for c in plan.victims] == list(range(10 - RESUME_WINDOW))


def test_run_shorter_than_the_window_is_skipped():
    assert plan_run(_run("p-aaaaaaaa", [7], live=True), keep_newest=0) == Skip(
        "p-aaaaaaaa", "inside-resume-window"
    )


def test_pre_split_run_has_no_training_item(tmp_path: Path):
    run_dir = tmp_path / "p-aaaaaaaa"
    _write_ckpts(run_dir, [100], with_training=False)
    assert read_run(run_dir, frozenset()) == Skip("p-aaaaaaaa", "no-training-item")


def test_read_run_sizes_and_orders_thinnable_steps(tmp_path: Path):
    run_dir = tmp_path / "p-bbbbbbbb"
    _write_ckpts(run_dir, [2000, 1000])
    run = read_run(run_dir, frozenset({"p-bbbbbbbb"}))
    assert isinstance(run, Run)
    assert run.live
    assert [c.step for c in run.thinnable] == [1000, 2000]
    assert {c.nbytes for c in run.thinnable} == {len(b"trajectory tail")}


def test_execute_drops_training_and_spares_the_decomposition(tmp_path: Path):
    run_dir = tmp_path / "p-cccccccc"
    _write_ckpts(run_dir, [1000, 2000, 3000])
    trash = tmp_path / "trash"
    trash.mkdir()

    run = read_run(run_dir, frozenset())
    assert isinstance(run, Run)
    plan = plan_run(run, keep_newest=1)
    assert isinstance(plan, Thin)
    assert [c.step for c in list(execute(plan, trash))] == [1000, 2000]

    ckpts = run_dir / "ckpts"
    assert not (ckpts / "1000" / "training").exists()
    assert not (ckpts / "2000" / "training").exists()
    assert (ckpts / "3000" / "training" / "d" / "chunk").exists()
    for step in (1000, 2000, 3000):
        assert (ckpts / str(step) / "decomposition" / "d" / "chunk").read_bytes() == b"product"
        assert (ckpts / str(step) / "_CHECKPOINT_METADATA").exists()
    assert list(trash.iterdir()) == []


def test_a_second_sweep_finds_nothing_left(tmp_path: Path):
    run_dir = tmp_path / "p-dddddddd"
    _write_ckpts(run_dir, [1000, 2000])
    trash = tmp_path / "trash"
    trash.mkdir()

    run = read_run(run_dir, frozenset())
    assert isinstance(run, Run)
    plan = plan_run(run, keep_newest=1)
    assert isinstance(plan, Thin)
    list(execute(plan, trash))

    again = read_run(run_dir, frozenset())
    assert isinstance(again, Run)
    assert plan_run(again, keep_newest=1) == Skip("p-dddddddd", "inside-resume-window")


def test_a_run_owned_by_someone_else_is_never_touched(tmp_path: Path, monkeypatch: MonkeyPatch):
    run_dir = tmp_path / "p-eeeeeeee"
    _write_ckpts(run_dir, [1000, 2000])
    monkeypatch.setattr("os.getuid", lambda: run_dir.stat().st_uid + 1)
    assert read_run(run_dir, frozenset()) == Skip("p-eeeeeeee", "not-owned")


@pytest.mark.parametrize("keep_newest", [0, 1, RESUME_WINDOW, 5])
def test_a_live_run_always_keeps_a_resumable_step(keep_newest: int):
    run = _run("p-ffffffff", list(range(6)), live=True)
    plan = plan_run(run, keep_newest)
    kept = (
        set(range(6)) - {c.step for c in plan.victims} if isinstance(plan, Thin) else set(range(6))
    )
    assert max(kept) == 5
    assert len(kept) >= RESUME_WINDOW
