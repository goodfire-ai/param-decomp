"""CPU tests for the lab-side arithmetic-probe glue: loading the offline artifact (with the
row-major order guard + missing-file fail-fast) and sharding it like a batch. The full
in-loop run is exercised by the GPU smoke; here we pin the IO that's easy to get wrong.
"""

import json
from pathlib import Path

import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from jax.sharding import Mesh

from param_decomp_lab.experiments.lm.run import (
    _arithmetic_probe_global,
    _ensure_arithmetic_probe,
    _load_arithmetic_probe,
)

N_A, N_B, T = 3, 4, 5


def _write_artifact(out: Path, *, scramble: bool = False) -> None:
    a_values, b_values = list(range(1, N_A + 1)), list(range(1, N_B + 1))
    rows = [(a, b) for a in a_values for b in b_values]
    if scramble:
        rows = rows[::-1]  # break row-major (a, b) order
    table = pa.table(
        {
            "a": pa.array([a for a, _ in rows], pa.int32()),
            "b": pa.array([b for _, b in rows], pa.int32()),
            "answer_id": pa.array([a + b for a, b in rows], pa.int32()),
            "input_ids": pa.array([[1, a, 2, b, 3] for a, b in rows], pa.list_(pa.int32())),
        }
    )
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out / "grid.parquet")
    (out / "meta.json").write_text(json.dumps({
        "operation": "add", "symbol": "+", "seq_len": T, "answer_position": T - 1,
        "n_a": N_A, "n_b": N_B, "a_values": a_values, "b_values": b_values,
        "n_prompts": N_A * N_B,
    }))  # fmt: skip


def test_load_arithmetic_probe(tmp_path: Path):
    _write_artifact(tmp_path)
    tokens, grid, answer_pos, n_prompts = _load_arithmetic_probe(tmp_path)
    assert tokens.shape == (N_A * N_B, T)
    assert answer_pos == T - 1 and n_prompts == N_A * N_B
    assert grid.n_a == N_A and grid.n_b == N_B and grid.symbol == "+"
    # row 0 is (a=1, b=1); reshape lands it at grid[0, 0]
    assert grid.to_grid(tokens)[0, 0].tolist() == [1, 1, 2, 1, 3]


def test_load_arithmetic_probe_rejects_non_row_major(tmp_path: Path):
    _write_artifact(tmp_path, scramble=True)
    with pytest.raises(AssertionError, match="row-major"):
        _load_arithmetic_probe(tmp_path)


def test_load_arithmetic_probe_missing_artifact(tmp_path: Path):
    with pytest.raises(AssertionError, match="prestage_arithmetic"):
        _load_arithmetic_probe(tmp_path / "nonexistent")


def test_arithmetic_probe_global_preserves_grid(tmp_path: Path):
    _write_artifact(tmp_path)
    tokens, _, _, _ = _load_arithmetic_probe(tmp_path)
    devices = np.asarray(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(devices, ("replicate", "fsdp"))
    sharded = _arithmetic_probe_global(tokens, mesh, n_proc=1)
    # one device: no padding, rows preserved verbatim (the eval trims pad via n_prompts anyway)
    assert sharded.shape == (N_A * N_B, T)
    np.testing.assert_array_equal(np.asarray(sharded), tokens)


def test_ensure_arithmetic_probe_generates_when_missing_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # rank 0 generates the probe once if artifact_dir is empty, then no-ops when it exists.
    # (prestage is mocked; the real generator is covered by prestage_arithmetic itself.)
    calls: list[str] = []

    def fake_prestage(*, out_dir: str) -> None:
        calls.append(out_dir)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "meta.json").write_text("{}")

    import param_decomp_lab.experiments.lm.prestage_arithmetic as prestage_mod

    monkeypatch.setattr(prestage_mod, "prestage", fake_prestage)
    probe = tmp_path / "probe"
    _ensure_arithmetic_probe(probe, is_main=True, n_proc=1)
    assert calls == [str(probe)]  # generated
    _ensure_arithmetic_probe(probe, is_main=True, n_proc=1)
    assert calls == [str(probe)]  # exists -> no-op, no second generation


def test_ensure_arithmetic_probe_noop_for_non_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # non-rank-0 never generates (rank 0 writes; single-proc has no barrier to join here).
    import param_decomp_lab.experiments.lm.prestage_arithmetic as prestage_mod

    monkeypatch.setattr(
        prestage_mod, "prestage", lambda **_: pytest.fail("non-main must not prestage")
    )
    _ensure_arithmetic_probe(tmp_path / "probe", is_main=False, n_proc=1)
