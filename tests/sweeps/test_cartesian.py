"""Tests for the built-in cartesian helper and example sweep."""

from pathlib import Path

import yaml

from param_decomp.run import RunConfig
from param_decomp.settings import REPO_ROOT
from param_decomp.sweeps import SweepSpec
from param_decomp.sweeps.cartesian import cartesian_product, example_cartesian_sweep


def _base_config() -> RunConfig:
    with open(REPO_ROOT / "param_decomp" / "experiments" / "tms" / "tms_5-2_config.yaml") as f:
        return RunConfig.from_dict(yaml.safe_load(f))


def test_cartesian_product_basic() -> None:
    spec = cartesian_product(
        base_config=_base_config(),
        grid={"pd.seed": [0, 1, 2], "pd.steps": [10, 20]},
        n_agents=2,
        description="two axes",
    )
    assert isinstance(spec, SweepSpec)
    assert spec.description == "two axes"
    assert len(spec.runs) == 6

    combos = {(r.pd.seed, r.pd.steps) for r in spec.runs}
    assert combos == {(s, t) for s in (0, 1, 2) for t in (10, 20)}
    assert all(r.pd.batch_size == 4096 for r in spec.runs)

    run_cfgs = spec.run_cfgs()
    assert len({r.run_id for r in run_cfgs}) == len(run_cfgs)


def test_cartesian_view_meta_records_axes() -> None:
    spec = cartesian_product(
        base_config=_base_config(),
        grid={"pd.seed": [0, 1]},
        n_agents=2,
        description="tiny",
    )
    for run in spec.runs:
        assert run.view_meta["pd.seed"] == run.pd.seed


def test_cartesian_run_names_encode_axis_values() -> None:
    spec = cartesian_product(
        base_config=_base_config(),
        grid={"pd.seed": [0, 1], "pd.faithfulness_warmup_lr": [0.5]},
        n_agents=2,
        description="named",
    )
    names = {r.name for r in spec.runs}
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
    )

    run_ids = {r.run_id for r in spec.run_cfgs()}
    assert len(run_ids) == 2
    assert base_run.run_id not in run_ids


def test_example_cartesian_sweep_smoke() -> None:
    spec = example_cartesian_sweep()
    assert isinstance(spec, SweepSpec)
    assert len(spec.runs) == 3
    for run in spec.runs:
        assert "pd.seed" in run.view_meta
        assert run.recipe.path == "param_decomp.experiments.tms.experiment:Recipe"


def test_sweep_spec_write_serializes_plain_yaml(tmp_path: Path) -> None:
    spec = cartesian_product(
        base_config=_base_config(),
        grid={"pd.seed": [0]},
        n_agents=1,
        description="serializable",
    )
    path = tmp_path / "spec.yaml"
    spec.write(path)

    data = yaml.safe_load(path.read_text())
    assert data["runs"][0]["pd"]["seed"] == 0
    assert data["runs"][0]["recipe"]["path"] == "param_decomp.experiments.tms.experiment:Recipe"
