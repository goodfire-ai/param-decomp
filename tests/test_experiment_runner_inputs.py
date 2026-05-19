from pathlib import Path

import pytest
import yaml

from param_decomp.experiments.runner import _resolve_source
from param_decomp.run_spec import RunSpec


def _write_yaml(path: Path, data: dict[str, object]) -> Path:
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return path


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
    config_path = _write_yaml(tmp_path / "my_config.yaml", {"pd": {"seed": 123}})

    name, driver_path, config = _resolve_source(
        experiment=None,
        config_path=config_path,
        driver="param_decomp.experiments.tms.experiment:Driver",
        rerun=None,
    )

    assert name == "my_config"
    assert driver_path == "param_decomp.experiments.tms.experiment:Driver"
    assert config == {"pd": {"seed": 123}}


def test_rerun_loads_driver_from_spec(tmp_path: Path) -> None:
    spec_path = tmp_path / "run_metadata.yaml"
    RunSpec(
        driver="param_decomp.experiments.tms.experiment:Driver",
        config={"pd": {"seed": 123}},
    ).write(spec_path)

    name, driver_path, config = _resolve_source(
        experiment=None,
        config_path=None,
        driver=None,
        rerun=str(spec_path.parent),
    )

    assert name == "rerun"
    assert driver_path == "param_decomp.experiments.tms.experiment:Driver"
    assert config == {"pd": {"seed": 123}}


def test_rerun_rejects_driver_override(tmp_path: Path) -> None:
    spec_path = tmp_path / "run_metadata.yaml"
    RunSpec(
        driver="param_decomp.experiments.tms.experiment:Driver",
        config={"pd": {}},
    ).write(spec_path)

    with pytest.raises(AssertionError, match="--driver is implied by --rerun"):
        _resolve_source(
            experiment=None,
            config_path=None,
            driver="other:Driver",
            rerun=str(spec_path.parent),
        )
