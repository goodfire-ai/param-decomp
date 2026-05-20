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
        "from param_decomp.run import RunConfig\n"
        "from param_decomp.settings import REPO_ROOT\n"
        "from param_decomp.sweeps import SweepSpec\n"
        "from param_decomp.sweeps.cartesian import cartesian_product\n"
        "import yaml\n"
        "DRIVER = 'param_decomp.experiments.tms.experiment:Driver'\n"
        "def my_sweep():\n"
        '    with open(REPO_ROOT / "param_decomp" / "experiments" / "tms" / "tms_5-2_config.yaml") as f:\n'
        "        config = yaml.safe_load(f)\n"
        "    base_run = RunConfig.model_validate(config)\n"
        "    return cartesian_product(\n"
        "        base_config=base_run,\n"
        '        grid={"pd.seed": [0]},\n'
        "        n_agents=1,\n"
        '        description="tiny",\n'
        "        driver_path=DRIVER,\n"
        "    )\n",
    )
    gen = load_sweep_generator(f"{path}:my_sweep")
    spec = gen()
    assert isinstance(spec, SweepSpec)
    assert spec.driver_path == "param_decomp.experiments.tms.experiment:Driver"
    assert len(spec.swept_datas) == 1
    assert spec.swept_datas[0].pd_config.seed == 0


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
