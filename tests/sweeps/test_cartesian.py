"""Tests for the built-in cartesian helper and example sweep."""

from param_decomp.sweeps import SweepSpec
from param_decomp.sweeps.cartesian import cartesian_product, example_cartesian_sweep


def test_cartesian_product_basic() -> None:
    base = {"pd": {"seed": -1, "steps": -1, "extra": "kept"}}
    spec = cartesian_product(
        base_config=base,
        grid={"pd.seed": [0, 1, 2], "pd.steps": [10, 20]},
        description="two axes",
        driver_path="pkg.mod:Driver",
    )
    assert isinstance(spec, SweepSpec)
    assert spec.description == "two axes"
    assert spec.driver == "pkg.mod:Driver"
    assert all(r.driver == "pkg.mod:Driver" for r in spec.runs)
    assert len(spec.runs) == 6

    combos = {(r.config["pd"]["seed"], r.config["pd"]["steps"]) for r in spec.runs}
    assert combos == {(s, t) for s in (0, 1, 2) for t in (10, 20)}
    assert all(r.config["pd"]["extra"] == "kept" for r in spec.runs)


def test_cartesian_view_meta_records_axes() -> None:
    spec = cartesian_product(
        base_config={"pd": {"seed": -1}},
        grid={"pd.seed": [0, 1]},
        description="tiny",
        driver_path="pkg.mod:Driver",
    )
    for run in spec.runs:
        assert run.view_meta["pd.seed"] == run.config["pd"]["seed"]


def test_cartesian_run_names_encode_axis_values() -> None:
    spec = cartesian_product(
        base_config={"pd": {"seed": -1, "lr_ratio": -1.0}},
        grid={"pd.seed": [0, 1], "pd.lr_ratio": [0.5]},
        description="named",
        driver_path="pkg.mod:Driver",
    )
    names = {r.wandb_run_name for r in spec.runs}
    assert names == {"seed=0_lr_ratio=0.5", "seed=1_lr_ratio=0.5"}


def test_example_cartesian_sweep_smoke() -> None:
    spec = example_cartesian_sweep()
    assert isinstance(spec, SweepSpec)
    assert spec.driver == "param_decomp.experiments.tms.experiment:Driver"
    assert len(spec.runs) == 3
    for run in spec.runs:
        assert "pd.seed" in run.view_meta
