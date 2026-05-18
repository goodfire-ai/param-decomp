"""Tests for sweep generator auto-discovery and CLI-string resolution."""

from pathlib import Path

import pytest
import yaml

from param_decomp.sweeps import CartesianGridSweep
from param_decomp.sweeps.discovery import discover_sweeps, resolve_sweep


def test_cartesian_is_discovered() -> None:
    registry = discover_sweeps()
    assert "cartesian" in registry
    assert registry["cartesian"] is CartesianGridSweep


def test_resolve_by_short_name(tmp_path: Path) -> None:
    grid_path = tmp_path / "g.yaml"
    grid_path.write_text(yaml.dump({"grid": {"pd.seed": [0]}}))
    gen = resolve_sweep(f"cartesian:{grid_path}")
    assert isinstance(gen, CartesianGridSweep)


def test_resolve_yaml_path_shorthand(tmp_path: Path) -> None:
    grid_path = tmp_path / "g.yaml"
    grid_path.write_text(yaml.dump({"grid": {"pd.seed": [0]}}))
    gen = resolve_sweep(str(grid_path))
    assert isinstance(gen, CartesianGridSweep)


def test_resolve_unknown_rejected() -> None:
    with pytest.raises(AssertionError, match="not a known sweep name"):
        resolve_sweep("does_not_exist")
