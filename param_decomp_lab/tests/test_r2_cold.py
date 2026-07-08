"""pd-cold push/pull unit tests — the run-dir inspection, completeness gating, remote
verification parsing, and the end-of-training edge hook's non-fatality. The R2 shell layer
(`_r2_bash`) is monkeypatched throughout: real transfers need cluster creds."""

import getpass
import json
import subprocess
from pathlib import Path

import pytest

from param_decomp_lab.tools import r2_cold


def _make_run_dir(tmp_path: Path, run_id: str, steps: int, ckpt_steps: list[int]) -> Path:
    run = tmp_path / "runs" / run_id
    for step in ckpt_steps:
        item = run / "ckpts" / str(step) / "decomposition"
        item.mkdir(parents=True)
        (item / "array.bin").write_bytes(b"x" * 128)
        (run / "ckpts" / str(step) / "_CHECKPOINT_METADATA").write_text("{}")
    (run / "launch_config.yaml").write_text(f"pd:\n  steps: {steps}\n")
    return run


def test_latest_decomposition_step_ignores_pre_split(tmp_path: Path):
    run = _make_run_dir(tmp_path, "p-aaaaaaaa", steps=100, ckpt_steps=[50])
    pre_split = run / "ckpts" / "75" / "default"
    pre_split.mkdir(parents=True)
    assert r2_cold.latest_decomposition_step(run) == 50
    assert r2_cold.latest_decomposition_step(tmp_path) is None  # no ckpts dir at all


def test_push_refuses_incomplete_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run = _make_run_dir(tmp_path, "p-bbbbbbbb", steps=100, ckpt_steps=[50])
    monkeypatch.setattr(r2_cold, "_r2_bash", lambda *a, **k: pytest.fail("must not touch R2"))
    with pytest.raises(AssertionError, match="not a finished run"):
        r2_cold.push(str(run))


def test_push_uploads_payload_then_manifest_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run = _make_run_dir(tmp_path, "p-cccccccc", steps=100, ckpt_steps=[100])
    scripts: list[str] = []

    def fake_r2_bash(script: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
        _ = capture
        scripts.append(script)
        if script.startswith("r2_ls"):
            # payload = array.bin (128 B) + _CHECKPOINT_METADATA (2 B) + launch_config (17 B)
            out = "Total Objects: 3\n   Total Size: 147\n"
            return subprocess.CompletedProcess([], 0, stdout=out, stderr="")
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr(r2_cold, "_r2_bash", fake_r2_bash)
    monkeypatch.setattr(r2_cold, "_cluster_name", lambda: "TEST")
    monkeypatch.setattr(getpass, "getuser", lambda: "alice")

    prefix = r2_cold.push(str(run))

    assert prefix == "alice/pd-cold/p-cccccccc"
    assert "training" not in "".join(scripts), "must never upload the training item"
    assert "manifest.json" in scripts[-1], "manifest is the LAST upload (completion marker)"
    assert "decomposition" in scripts[0] and "launch_config.yaml" in scripts[0]


def test_push_fails_on_remote_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run = _make_run_dir(tmp_path, "p-dddddddd", steps=100, ckpt_steps=[100])

    def fake_r2_bash(script: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
        _ = capture
        if script.startswith("r2_ls"):
            out = "Total Objects: 1\n   Total Size: 7\n"
            return subprocess.CompletedProcess([], 0, stdout=out, stderr="")
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr(r2_cold, "_r2_bash", fake_r2_bash)
    monkeypatch.setattr(r2_cold, "_cluster_name", lambda: "TEST")
    with pytest.raises(AssertionError, match="post-upload verification failed"):
        r2_cold.push(str(run))


def test_remote_stats_excludes_previous_manifest(monkeypatch: pytest.MonkeyPatch):
    out = (
        "2026-07-08 12:00:00        128 alice/pd-cold/p-x/ckpts/100/decomposition/array.bin\n"
        "2026-07-08 12:00:00         42 alice/pd-cold/p-x/manifest.json\n"
        "Total Objects: 2\n   Total Size: 170\n"
    )
    monkeypatch.setattr(
        r2_cold,
        "_r2_bash",
        lambda *a, **k: subprocess.CompletedProcess([], 0, stdout=out, stderr=""),
    )
    assert r2_cold._remote_stats("alice/pd-cold/p-x") == (1, 128)


def test_pull_refuses_to_overlay_live_training_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dest = tmp_path / "runs" / "p-eeeeeeee"
    (dest / "ckpts" / "50" / "training").mkdir(parents=True)
    monkeypatch.setattr(
        r2_cold,
        "_r2_bash",
        lambda *a, **k: subprocess.CompletedProcess([], 0, stdout="manifest.json", stderr=""),
    )
    with pytest.raises(AssertionError, match="resumable training run"):
        r2_cold.pull("p-eeeeeeee", user="alice", dest=str(dest))


def test_pull_syncs_into_run_dir_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dest = tmp_path / "runs" / "p-ffffffff"
    manifest = {
        "format": "pd-cold-v1",
        "run_id": "p-ffffffff",
        "step": 100,
        "bytes": 128,
        "source_cluster": "TEST",
    }

    def fake_r2_bash(script: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
        _ = capture
        if script.startswith("r2_sync_down"):
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "manifest.json").write_text(json.dumps(manifest))
        return subprocess.CompletedProcess([], 0, stdout="manifest.json", stderr="")

    monkeypatch.setattr(r2_cold, "_r2_bash", fake_r2_bash)
    assert r2_cold.pull("p-ffffffff", user="alice", dest=str(dest)) == dest


def test_maybe_push_skips_unfinished_and_survives_push_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    run = _make_run_dir(tmp_path, "p-gggggggg", steps=100, ckpt_steps=[50])
    monkeypatch.setattr(
        r2_cold, "push", lambda *a, **k: pytest.fail("must not push an unfinished run")
    )
    r2_cold.maybe_push_completed_run(run, pd_steps=100)
    assert "not archiving" in capsys.readouterr().out

    def failing_push(run_dir: str) -> str:
        _ = run_dir
        raise RuntimeError("R2 exploded")

    monkeypatch.setattr(r2_cold, "push", failing_push)
    r2_cold.maybe_push_completed_run(run, pd_steps=50)  # 50 IS the final step here
    err = capsys.readouterr().err
    assert "WARNING" in err and "retry manually" in err and "R2 exploded" in err
