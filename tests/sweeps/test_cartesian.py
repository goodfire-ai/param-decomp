"""Tests for the built-in cartesian helper and example sweep."""

from pathlib import Path

import yaml

from param_decomp.settings import REPO_ROOT
from param_decomp.sweeps import SweepSpec
from param_decomp.sweeps.cartesian import cartesian_product, example_cartesian_sweep

TMS_DRIVER_PATH = "param_decomp.experiments.tms.experiment:Driver"


def _base_config() -> dict[str, object]:
    with open(REPO_ROOT / "param_decomp" / "experiments" / "tms" / "tms_5-2_config.yaml") as f:
        return yaml.safe_load(f)


def test_cartesian_product_basic() -> None:
    spec = cartesian_product(
        base_config=_base_config(),
        grid={"pd.seed": [0, 1, 2], "pd.steps": [10, 20]},
        description="two axes",
        driver_path=TMS_DRIVER_PATH,
    )
    assert isinstance(spec, SweepSpec)
    assert spec.description == "two axes"
    assert spec.driver_path == TMS_DRIVER_PATH
    assert all(r.driver_path == TMS_DRIVER_PATH for r in spec.runs)
    assert len(spec.runs) == 6

    combos = {(r.pd.seed, r.pd.steps) for r in spec.runs}
    assert combos == {(s, t) for s in (0, 1, 2) for t in (10, 20)}
    assert all(r.pd.batch_size == 4096 for r in spec.runs)
    assert len({r.run_id for r in spec.runs}) == len(spec.runs)


def test_cartesian_view_meta_records_axes() -> None:
    spec = cartesian_product(
        base_config=_base_config(),
        grid={"pd.seed": [0, 1]},
        description="tiny",
        driver_path=TMS_DRIVER_PATH,
    )
    for run in spec.runs:
        assert run.logging.view_meta["pd.seed"] == run.pd.seed


def test_cartesian_run_names_encode_axis_values() -> None:
    spec = cartesian_product(
        base_config=_base_config(),
        grid={"pd.seed": [0, 1], "pd.faithfulness_warmup_lr": [0.5]},
        description="named",
        driver_path=TMS_DRIVER_PATH,
    )
    names = {r.logging.wandb_run_name for r in spec.runs}
    assert names == {
        "seed=0_faithfulness_warmup_lr=0.5",
        "seed=1_faithfulness_warmup_lr=0.5",
    }


def test_cartesian_product_strips_run_id_from_base_config() -> None:
    base_config = {**_base_config(), "run_id": "p-deadbeef"}
    spec = cartesian_product(
        base_config=base_config,
        grid={"pd.seed": [0, 1]},
        description="strips run_id",
        driver_path=TMS_DRIVER_PATH,
    )

    run_ids = {r.run_id for r in spec.runs}
    assert len(run_ids) == 2
    assert "p-deadbeef" not in run_ids


def test_example_cartesian_sweep_smoke() -> None:
    spec = example_cartesian_sweep()
    assert isinstance(spec, SweepSpec)
    assert spec.driver_path == "param_decomp.experiments.tms.experiment:Driver"
    assert len(spec.runs) == 3
    for run in spec.runs:
        assert "pd.seed" in run.logging.view_meta


def test_sweep_spec_write_serializes_plain_yaml(tmp_path: Path) -> None:
    spec = cartesian_product(
        base_config=_base_config(),
        grid={"pd.seed": [0]},
        description="serializable",
        driver_path=TMS_DRIVER_PATH,
    )
    path = tmp_path / "spec.yaml"
    spec.write(path)

    data = yaml.safe_load(path.read_text())
    assert data["runs"][0]["pd"]["seed"] == 0
