"""Tests for ``load_sweep_generator`` (path-based sweep import)."""

from pathlib import Path

import pytest

from param_decomp.sweeps import SweepSpec, load_sweep_generator


def _write_sweep_module(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "sweep_mod.py"
    path.write_text(body)
    return path


def test_load_and_call(tmp_path: Path) -> None:
    path = _write_sweep_module(
        tmp_path,
        "from param_decomp import RunSpec\n"
        "from param_decomp.sweeps import SweepSpec\n"
        "def my_sweep():\n"
        "    return SweepSpec(\n"
        '        description="tiny",\n'
        "        runs=[\n"
        '            RunSpec(driver="pkg.mod:Driver", config={"a": 1}, wandb_run_name="r"),\n'
        "        ],\n"
        "    )\n",
    )
    gen = load_sweep_generator(f"{path}:my_sweep")
    spec = gen()
    assert isinstance(spec, SweepSpec)
    assert spec.driver == "pkg.mod:Driver"
    assert len(spec.runs) == 1
    assert spec.runs[0].config == {"a": 1}


def test_non_absolute_path_rejected() -> None:
    with pytest.raises(AssertionError, match="must be absolute"):
        load_sweep_generator("relative/file.py:f")


def test_missing_file_rejected(tmp_path: Path) -> None:
    bogus = tmp_path / "does_not_exist.py"
    with pytest.raises(AssertionError, match="file not found"):
        load_sweep_generator(f"{bogus}:f")


def test_non_py_path_rejected(tmp_path: Path) -> None:
    path = tmp_path / "foo.txt"
    path.write_text("hi")
    with pytest.raises(AssertionError, match="must end in .py"):
        load_sweep_generator(f"{path}:f")


def test_missing_function_rejected(tmp_path: Path) -> None:
    path = _write_sweep_module(tmp_path, "def other(): pass\n")
    with pytest.raises(AssertionError, match="no function named 'my_sweep'"):
        load_sweep_generator(f"{path}:my_sweep")


def test_missing_func_name_rejected(tmp_path: Path) -> None:
    path = _write_sweep_module(tmp_path, "def f(): pass\n")
    with pytest.raises(AssertionError, match="expected '/abs/path"):
        load_sweep_generator(str(path))
