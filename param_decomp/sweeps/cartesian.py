"""Built-in example sweep: a small Cartesian grid over ``tms_5-2``.

User sweep files don't have to live here — pass
``--sweep_generator_path /abs/path/to/file.py:func`` to ``pd-run`` and any
zero-arg function returning a ``SweepSpec`` works. This module ships
``example_cartesian_sweep`` as a working reference and exposes
``cartesian_product`` as a reusable helper for users who want the Cartesian-grid
pattern without writing the product loop themselves.
"""

import itertools
from typing import Any

import yaml

from param_decomp.configs import PDConfig
from param_decomp.run import RunConfig
from param_decomp.settings import REPO_ROOT
from param_decomp.sweeps.spec import SweepData, SweepSpec
from param_decomp.utils.run_utils import apply_nested_updates

_PD_PREFIX = "pd."


def cartesian_product(
    base_config: RunConfig,
    grid: dict[str, list[Any]],
    *,
    n_agents: int,
    description: str,
    driver_path: str,
) -> SweepSpec:
    """Cartesian product of dot-pathed axes over a base config.

    Each axis key is a dotted path into ``base_config.pd`` (e.g.
    ``"pd.loss_metrics.ImportanceMinimalityLoss.coeff"``). Only ``pd.*`` keys are
    supported — ``logging`` and ``runtime`` are hoisted to the ``SweepSpec``
    and shared across runs. Axis values are recorded in each run's
    ``view_meta`` so W&B can group/color by them.
    """
    assert grid, "cartesian_product requires a non-empty grid"
    for axis, values in grid.items():
        assert axis.startswith(_PD_PREFIX), (
            f"grid keys must start with 'pd.' (logging/runtime are shared across runs "
            f"and cannot be swept); got {axis!r}"
        )
        assert isinstance(values, list) and values, (
            f"grid['{axis}'] must be a non-empty list, got {values!r}"
        )

    axes = list(grid.keys())
    value_lists = [grid[a] for a in axes]

    base_config_data = base_config.model_dump(mode="json")

    swept_datas: list[SweepData] = []

    for combo in itertools.product(*value_lists):
        updates = dict(zip(axes, combo, strict=True))
        config_data = apply_nested_updates(base_config_data, updates)

        sweep_data = SweepData(
            name="_".join(f"{_short_axis(a)}={_short_value(v)}" for a, v in updates.items()),
            pd_config=PDConfig.model_validate(config_data["pd"]),
            view_meta=dict(updates),
        )

        swept_datas.append(sweep_data)

    return SweepSpec(
        description=description,
        driver_path=driver_path,
        logging=base_config.logging,
        runtime=base_config.runtime,
        target=base_config.target,
        data=base_config.data,
        n_agents=n_agents,
        swept_datas=swept_datas,
    )


def example_cartesian_sweep() -> SweepSpec:
    """Reference sweep: TMS 5-2 across three seeds.

    Demonstrates the zero-arg ``SweepGenerator`` shape — loads its own base
    config, defines its grid inline, and declares the driver in the returned
    spec. Copy this file as a starting point for real sweeps.
    """
    base_config_path = REPO_ROOT / "param_decomp" / "experiments" / "tms" / "tms_5-2_config.yaml"
    with open(base_config_path) as f:
        base_config = RunConfig.model_validate(yaml.safe_load(f))
    return cartesian_product(
        base_config=base_config,
        grid={"pd.seed": [0, 1, 2]},
        n_agents=3,
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
