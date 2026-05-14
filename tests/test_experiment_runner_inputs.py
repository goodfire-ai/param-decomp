from pathlib import Path

import pytest
import yaml

from param_decomp.experiments.runner import _resolve_inputs
from param_decomp.run_metadata import RunMetadata


def _write_yaml(path: Path, data: dict[str, object]) -> Path:
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return path


def test_builtin_experiment_rejects_explicit_config_options(tmp_path: Path) -> None:
    config_path = _write_yaml(tmp_path / "config.yaml", {"pd": {}})

    with pytest.raises(ValueError, match="Choose one pd-run input mode"):
        _resolve_inputs(
            experiment="tms_5-2",
            config_path=config_path,
            config_json=None,
            driver="param_decomp.experiments.tms.experiment:Driver",
        )


def test_raw_config_path_requires_driver(tmp_path: Path) -> None:
    config_path = _write_yaml(tmp_path / "config.yaml", {"pd": {}})

    with pytest.raises(ValueError, match="Raw experiment configs require --driver"):
        _resolve_inputs(
            experiment=None,
            config_path=config_path,
            config_json=None,
            driver=None,
        )


def test_config_path_with_driver_resolves_raw_config(tmp_path: Path) -> None:
    config_path = _write_yaml(tmp_path / "config.yaml", {"pd": {"seed": 123}})

    driver, config = _resolve_inputs(
        experiment=None,
        config_path=config_path,
        config_json=None,
        driver="param_decomp.experiments.tms.experiment:Driver",
    )

    assert driver == "param_decomp.experiments.tms.experiment:Driver"
    assert config == {"pd": {"seed": 123}}


def test_run_metadata_config_path_supplies_driver(tmp_path: Path) -> None:
    metadata_path = tmp_path / "run_metadata.yaml"
    RunMetadata(
        driver="param_decomp.experiments.tms.experiment:Driver",
        config={"pd": {"seed": 123}},
    ).write(metadata_path)

    driver, config = _resolve_inputs(
        experiment=None,
        config_path=metadata_path,
        config_json=None,
        driver=None,
    )

    assert driver == "param_decomp.experiments.tms.experiment:Driver"
    assert config == {"pd": {"seed": 123}}


def test_missing_input_reports_supported_modes() -> None:
    with pytest.raises(ValueError, match="No run input provided"):
        _resolve_inputs(
            experiment=None,
            config_path=None,
            config_json=None,
            driver=None,
        )
