"""Tests for the built-in cartesian helper and example sweep."""

from pathlib import Path

import yaml

from param_decomp.run import RunConfig
from param_decomp.settings import REPO_ROOT
from param_decomp.sweeps import SweepSpec
from param_decomp.sweeps.cartesian import cartesian_product, example_cartesian_sweep

TMS_DRIVER_PATH = "param_decomp.experiments.tms.experiment:Driver"


def _base_config() -> RunConfig:
    with open(REPO_ROOT / "param_decomp" / "experiments" / "tms" / "tms_5-2_config.yaml") as f:
        return RunConfig.from_dict(yaml.safe_load(f))


def test_cartesian_product_basic() -> None:
    spec = cartesian_product(
        base_config=_base_config(),
        grid={"pd.seed": [0, 1, 2], "pd.steps": [10, 20]},
        n_agents=2,
        description="two axes",
        driver_path=TMS_DRIVER_PATH,
    )
    assert isinstance(spec, SweepSpec)
    assert spec.description == "two axes"
    assert spec.driver_path == TMS_DRIVER_PATH
    assert len(spec.swept_datas) == 6

    combos = {(d.pd_config.seed, d.pd_config.steps) for d in spec.swept_datas}
    assert combos == {(s, t) for s in (0, 1, 2) for t in (10, 20)}
    assert all(d.pd_config.batch_size == 4096 for d in spec.swept_datas)

    run_cfgs = spec.run_cfgs()
    assert len({r.run_id for r in run_cfgs}) == len(run_cfgs)


def test_cartesian_view_meta_records_axes() -> None:
    spec = cartesian_product(
        base_config=_base_config(),
        grid={"pd.seed": [0, 1]},
        n_agents=2,
        description="tiny",
        driver_path=TMS_DRIVER_PATH,
    )
    for sweep_data in spec.swept_datas:
        assert sweep_data.view_meta["pd.seed"] == sweep_data.pd_config.seed


def test_cartesian_run_names_encode_axis_values() -> None:
    spec = cartesian_product(
        base_config=_base_config(),
        grid={"pd.seed": [0, 1], "pd.faithfulness_warmup_lr": [0.5]},
        n_agents=2,
        description="named",
        driver_path=TMS_DRIVER_PATH,
    )
    names = {d.name for d in spec.swept_datas}
    assert names == {
        "seed=0_faithfulness_warmup_lr=0.5",
        "seed=1_faithfulness_warmup_lr=0.5",
    }


def test_cartesian_product_generates_new_run_ids_from_run_template() -> None:
    base_run = _base_config()
    spec = cartesian_product(
        base_config=base_run,
        grid={"pd.seed": [0, 1]},
        n_agents=2,
        description="from run",
        driver_path=TMS_DRIVER_PATH,
    )

    run_ids = {r.run_id for r in spec.run_cfgs()}
    assert len(run_ids) == 2
    assert base_run.run_id not in run_ids


def test_example_cartesian_sweep_smoke() -> None:
    spec = example_cartesian_sweep()
    assert isinstance(spec, SweepSpec)
    assert spec.driver_path == "param_decomp.experiments.tms.experiment:Driver"
    assert len(spec.swept_datas) == 3
    for sweep_data in spec.swept_datas:
        assert "pd.seed" in sweep_data.view_meta


def test_sweep_spec_write_serializes_plain_yaml(tmp_path: Path) -> None:
    spec = cartesian_product(
        base_config=_base_config(),
        grid={"pd.seed": [0]},
        n_agents=1,
        description="serializable",
        driver_path=TMS_DRIVER_PATH,
    )
    path = tmp_path / "spec.yaml"
    spec.write(path)

    data = yaml.safe_load(path.read_text())
    assert data["swept_data"][0]["pd_config"]["seed"] == 0
