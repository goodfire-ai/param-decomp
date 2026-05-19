from pathlib import Path
from typing import Any

import pytest
import yaml

from param_decomp.experiments.discovery import discover_experiments
from param_decomp.experiments.driver import load_driver
from param_decomp.experiments.runner import _resolve_source
from param_decomp.run import RUN_METADATA_FILENAME
from param_decomp.settings import REPO_ROOT


def _write_yaml(path: Path, data: dict[str, object]) -> Path:
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return path


def _builtin(name: str) -> tuple[str, dict[str, Any]]:
    discovered = discover_experiments()
    exp = discovered[name]
    with open(REPO_ROOT / exp.config_path) as f:
        return exp.driver_path, yaml.safe_load(f)


def test_at_most_one_source(tmp_path: Path) -> None:
    config_path = _write_yaml(tmp_path / "config.yaml", {"pd": {}})

    with pytest.raises(AssertionError, match="exactly one"):
        _resolve_source(
            experiment="tms_5-2",
            config_path=config_path,
            driver="param_decomp.experiments.tms.experiment:Driver",
            rerun=None,
        )


def test_config_path_requires_driver(tmp_path: Path) -> None:
    config_path = _write_yaml(tmp_path / "config.yaml", {"pd": {}})

    with pytest.raises(AssertionError, match="--config_path requires --driver"):
        _resolve_source(
            experiment=None,
            config_path=config_path,
            driver=None,
            rerun=None,
        )


def test_config_path_with_driver_resolves(tmp_path: Path) -> None:
    driver, config_data = _builtin("tms_5-2")
    config_data["pd"]["seed"] = 123
    config_path = _write_yaml(tmp_path / "my_config.yaml", config_data)

    name, driver_path, raw = _resolve_source(
        experiment=None,
        config_path=config_path,
        driver=driver,
        rerun=None,
    )

    assert name == "my_config"
    assert driver_path == driver
    assert raw["pd"]["seed"] == 123


def test_rerun_loads_driver_from_run(tmp_path: Path) -> None:
    driver, config_data = _builtin("tms_5-2")
    config_data["pd"]["seed"] = 123
    run_path = tmp_path / RUN_METADATA_FILENAME
    run = load_driver(driver).config_type.model_validate({**config_data, "driver_path": driver})
    run.write(run_path)

    name, driver_path, raw = _resolve_source(
        experiment=None,
        config_path=None,
        driver=None,
        rerun=str(run_path.parent),
    )

    assert name == "rerun"
    assert driver_path == driver
    assert raw["pd"]["seed"] == 123


def test_rerun_rejects_driver_override(tmp_path: Path) -> None:
    driver, config_data = _builtin("tms_5-2")
    run_path = tmp_path / RUN_METADATA_FILENAME
    run = load_driver(driver).config_type.model_validate({**config_data, "driver_path": driver})
    run.write(run_path)

    with pytest.raises(AssertionError, match="--driver is implied by --rerun"):
        _resolve_source(
            experiment=None,
            config_path=None,
            driver="other:Driver",
            rerun=str(run_path.parent),
        )
