"""Built-in example sweep: a small Cartesian grid over ``tms_5-2``.

User sweep files don't have to live here — pass ``--sweep /abs/path/to/file.py:func``
to ``pd-run`` and any zero-arg function returning a ``SweepSpec`` works. This module
ships ``example_cartesian_sweep`` as a working reference and exposes
``cartesian_product`` as a reusable helper for users who want the Cartesian-grid
pattern without writing the product loop themselves.
"""

import itertools
from typing import Any

import yaml

from param_decomp.experiments.driver import load_driver
from param_decomp.run import Run
from param_decomp.settings import REPO_ROOT
from param_decomp.sweeps.spec import SweepSpec
from param_decomp.utils.run_utils import apply_nested_updates


def cartesian_product(
    base_config: Run | dict[str, Any],
    grid: dict[str, list[Any]],
    *,
    description: str,
    driver_path: str,
) -> SweepSpec:
    """Cartesian product of dot-pathed axes over a base config.

    Each axis key is a dotted path into ``base_config`` (e.g.
    ``"pd.loss_metrics.importance_minimality.coeff"``). Axis values are
    recorded in each run's ``logging.view_meta`` so W&B can group/color by them.
    """
    assert grid, "cartesian_product requires a non-empty grid"
    for axis, values in grid.items():
        assert isinstance(values, list) and values, (
            f"grid['{axis}'] must be a non-empty list, got {values!r}"
        )

    axes = list(grid.keys())
    value_lists = [grid[a] for a in axes]
    base_config_data = _config_data(base_config)
    config_type = load_driver(driver_path).config_type
    runs: list[Run] = []
    for combo in itertools.product(*value_lists):
        updates = dict(zip(axes, combo, strict=True))
        config_data = apply_nested_updates(base_config_data, updates)
        name = "_".join(f"{_short_axis(a)}={_short_value(v)}" for a, v in updates.items())
        logging_data = {
            **config_data.get("logging", {}),
            "wandb_run_name": name,
            "view_meta": dict(updates),
        }
        run = config_type.model_validate(
            {**config_data, "driver_path": driver_path, "logging": logging_data}
        )
        runs.append(run)
    return SweepSpec(description=description, runs=runs)


def example_cartesian_sweep() -> SweepSpec:
    """Reference sweep: TMS 5-2 across three seeds.

    Demonstrates the zero-arg ``SweepGenerator`` shape — loads its own base
    config, defines its grid inline, and declares the driver in the returned
    spec. Copy this file as a starting point for real sweeps.
    """
    base_config_path = REPO_ROOT / "param_decomp" / "experiments" / "tms" / "tms_5-2_config.yaml"
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)
    return cartesian_product(
        base_config=base_config,
        grid={"pd.seed": [0, 1, 2]},
        description="Example: tms_5-2 seed sweep",
        driver_path="param_decomp.experiments.tms.experiment:Driver",
    )


def _short_axis(dotted: str) -> str:
    """Last segment of a dot-path; good enough for run names."""
    return dotted.rsplit(".", 1)[-1]


def _short_value(v: Any) -> str:
    """Compact, filename-safe rendering of a scalar/list value."""
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, list):
        return "-".join(_short_value(x) for x in v)
    return str(v).replace("/", "_")


def _config_data(config: Run | dict[str, Any]) -> dict[str, Any]:
    data = config.model_dump(mode="json") if isinstance(config, Run) else dict(config)
    data.pop("run_id", None)
    return data
