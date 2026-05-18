"""Built-in Cartesian grid sweep generator.

Reads a yaml file of the form::

    description: "lr x recon coeff sweep"
    grid:
      pd.seed: [0, 1, 2]
      pd.loss_metrics.importance_minimality.coeff: [0.1, 0.2, 0.5]

and produces one ``SweepRun`` per Cartesian-product combination. The varying
axes are recorded in each run's ``view_meta`` so W&B can group/color by them.
"""

import itertools
from pathlib import Path
from typing import Any, ClassVar, override

import yaml

from param_decomp.sweeps.spec import SweepGenerator, SweepRun, SweepSpec
from param_decomp.utils.run_utils import apply_nested_updates


class CartesianGridSweep(SweepGenerator):
    """The 80% case: dot-pathed parameter grid → Cartesian product of runs."""

    name: ClassVar[str] = "cartesian"

    @override
    def __init__(self, arg: str | None = None) -> None:
        assert arg is not None, (
            "cartesian sweep requires a yaml path: --sweep cartesian:my_grid.yaml"
        )
        super().__init__(arg=None)  # don't trigger base assertion; we accept this arg
        self.grid_path = Path(arg)
        assert self.grid_path.exists(), f"sweep grid not found: {self.grid_path}"

    @override
    def __call__(self, base_config: dict[str, Any]) -> SweepSpec:
        with open(self.grid_path) as f:
            grid_spec = yaml.safe_load(f)
        assert isinstance(grid_spec, dict), f"{self.grid_path}: expected a YAML mapping"
        description = grid_spec.get("description", f"Cartesian grid from {self.grid_path.name}")
        grid: dict[str, list[Any]] = grid_spec["grid"]
        assert grid, f"{self.grid_path}: 'grid' must be non-empty"
        for axis, values in grid.items():
            assert isinstance(values, list) and values, (
                f"{self.grid_path}: grid['{axis}'] must be a non-empty list, got {values!r}"
            )

        axes = list(grid.keys())
        value_lists = [grid[a] for a in axes]
        runs: list[SweepRun] = []
        for combo in itertools.product(*value_lists):
            updates = dict(zip(axes, combo, strict=True))
            config = apply_nested_updates(base_config, updates)
            name = "_".join(f"{_short_axis(a)}={_short_value(v)}" for a, v in updates.items())
            runs.append(SweepRun(name=name, config=config, view_meta=dict(updates)))
        return SweepSpec(description=description, runs=runs)


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
