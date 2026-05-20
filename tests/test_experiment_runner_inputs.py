from pathlib import Path
from typing import Any

import pytest
import yaml

from param_decomp.experiments.discovery import discover_experiments
from param_decomp.experiments.runner import _resolve_source
from param_decomp.run import RUN_CONFIG_FILENAME, RunConfig
from param_decomp.settings import REPO_ROOT


def _write_yaml(path: Path, data: dict[str, object]) -> Path:
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return path


def _builtin(name: str) -> dict[str, Any]:
    discovered = discover_experiments()
    with open(REPO_ROOT / discovered[name].config_path) as f:
        return yaml.safe_load(f)


def test_at_most_one_source(tmp_path: Path) -> None:
    config_path = _write_yaml(tmp_path / "config.yaml", {"pd": {}})

    with pytest.raises(AssertionError, match="exactly one"):
        _resolve_source(experiment="tms_5-2", config_path=config_path, rerun=None)


def test_config_path_requires_recipe_in_yaml(tmp_path: Path) -> None:
    config_path = _write_yaml(tmp_path / "config.yaml", {"pd": {}})

    with pytest.raises(AssertionError, match="missing a top-level `recipe:` block"):
        _resolve_source(experiment=None, config_path=config_path, rerun=None)


def test_config_path_resolves(tmp_path: Path) -> None:
    config_data = _builtin("tms_5-2")
    config_data["pd"]["seed"] = 123
    config_path = _write_yaml(tmp_path / "my_config.yaml", config_data)

    raw = _resolve_source(experiment=None, config_path=config_path, rerun=None)

    assert raw["name"] == "my_config"
    assert raw["recipe"] == config_data["recipe"]
    assert raw["pd"]["seed"] == 123


def test_rerun_loads_recipe_from_run(tmp_path: Path) -> None:
    config_data = _builtin("tms_5-2")
    config_data["pd"]["seed"] = 123
    run_path = tmp_path / RUN_CONFIG_FILENAME
    recipe = config_data["recipe"]

    run = RunConfig.from_dict(config_data)
    run.write(run_path)

    raw = _resolve_source(experiment=None, config_path=None, rerun=str(run_path.parent))

    assert raw["name"] == "rerun"
    assert raw["recipe"] == recipe
    assert raw["pd"]["seed"] == 123


def test_rerun_drops_saved_run_id(tmp_path: Path) -> None:
    config_data = _builtin("tms_5-2")

    run = RunConfig.from_dict(config_data)
    run.write(tmp_path / RUN_CONFIG_FILENAME)
    original_run_id = run.run_id

    raw = _resolve_source(experiment=None, config_path=None, rerun=str(tmp_path))

    new_run = RunConfig.from_dict(raw)

    assert "run_id" not in raw
    assert new_run.run_id != original_run_id
