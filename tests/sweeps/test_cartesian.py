"""Tests for the built-in CartesianGridSweep generator."""

from pathlib import Path

import pytest
import yaml

from param_decomp.sweeps import CartesianGridSweep, SweepSpec


def _write_grid(tmp_path: Path, description: str, grid: dict[str, list[object]]) -> Path:
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(yaml.dump({"description": description, "grid": grid}, sort_keys=False))
    return grid_path


def test_grid_cartesian_product(tmp_path: Path) -> None:
    grid_path = _write_grid(
        tmp_path,
        "two axes",
        {"pd.seed": [0, 1, 2], "pd.steps": [10, 20]},
    )
    base = {"pd": {"seed": -1, "steps": -1, "extra": "kept"}}
    spec = CartesianGridSweep(str(grid_path))(base)
    assert isinstance(spec, SweepSpec)
    assert spec.description == "two axes"
    assert len(spec.runs) == 6

    # Each axis combination appears exactly once.
    combos = {(r.config["pd"]["seed"], r.config["pd"]["steps"]) for r in spec.runs}
    assert combos == {(s, t) for s in (0, 1, 2) for t in (10, 20)}

    # Base config fields not in the grid are preserved.
    assert all(r.config["pd"]["extra"] == "kept" for r in spec.runs)


def test_view_meta_records_axes(tmp_path: Path) -> None:
    grid_path = _write_grid(tmp_path, "tiny", {"pd.seed": [0, 1]})
    spec = CartesianGridSweep(str(grid_path))({"pd": {"seed": -1}})
    for run in spec.runs:
        assert "pd.seed" in run.view_meta
        assert run.view_meta["pd.seed"] == run.config["pd"]["seed"]


def test_run_names_encode_axis_values(tmp_path: Path) -> None:
    grid_path = _write_grid(
        tmp_path,
        "named",
        {"pd.seed": [0, 1], "pd.lr_ratio": [0.5]},
    )
    spec = CartesianGridSweep(str(grid_path))({"pd": {"seed": -1, "lr_ratio": -1.0}})
    names = {r.name for r in spec.runs}
    # Last segment of the dotted axis, value compactly rendered, axes joined by _
    assert names == {"seed=0_lr_ratio=0.5", "seed=1_lr_ratio=0.5"}


def test_missing_arg_rejected() -> None:
    with pytest.raises(AssertionError, match="requires a yaml path"):
        CartesianGridSweep()


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="sweep grid not found"):
        CartesianGridSweep(str(tmp_path / "does_not_exist.yaml"))


def test_empty_grid_rejected(tmp_path: Path) -> None:
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(yaml.dump({"description": "empty", "grid": {}}))
    with pytest.raises(AssertionError, match="must be non-empty"):
        CartesianGridSweep(str(grid_path))({})


def test_default_description_when_unset(tmp_path: Path) -> None:
    grid_path = tmp_path / "named_grid.yaml"
    grid_path.write_text(yaml.dump({"grid": {"pd.seed": [0]}}))
    spec = CartesianGridSweep(str(grid_path))({"pd": {"seed": -1}})
    assert grid_path.name in spec.description
