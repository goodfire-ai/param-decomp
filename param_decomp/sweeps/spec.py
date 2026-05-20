"""Launch data model: ``SweepSpec`` (many runs sharing a driver and substrate).

A single PD launch is just a ``Run`` — the same type the worker writes
to disk. A sweep is a ``SweepSpec`` carrying ``list[Run]``.

A ``SweepGenerator`` is any zero-arg callable returning a ``SweepSpec``. The
sweep is fully self-contained — it loads whatever base config it wants and
each ``Run`` declares its own driver.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from param_decomp.run import RunConfig


@dataclass(frozen=True)
class SweepSpec:
    """A complete sweep: description and the list of runs to launch.

    All runs in a sweep must share one driver and one ``runtime:`` block
    (single SLURM array allocation, one substrate). Both invariants are
    asserted at construction. The W&B project is supplied to the launcher
    separately (``--project``) and is not part of the spec.

    Serialized to ``PARAM_DECOMP_OUT_DIR/sweeps/<launch_id>/spec.yaml`` on
    submit so reproducing the sweep doesn't require re-running the generator.
    """

    description: str
    runs: list[RunConfig]

    def __post_init__(self) -> None:
        assert self.runs, "SweepSpec.runs must be non-empty"
        head_driver = self.runs[0].driver_path
        head_runtime = self.runs[0].runtime
        assert head_driver is not None, (
            "SweepSpec runs must declare a driver_path; got driver_path=None on the first run"
        )
        for run in self.runs:
            assert run.driver_path == head_driver, (
                f"sweep run {run.logging.wandb_run_name!r} declares driver_path="
                f"{run.driver_path!r}, but the first run uses {head_driver!r}; all runs in "
                "one sweep must share a driver"
            )
            assert run.runtime == head_runtime, (
                f"sweep run {run.logging.wandb_run_name!r} has a different runtime block than "
                "the first run; all runs in one sweep must share the same substrate"
            )

    @property
    def driver_path(self) -> str:
        head = self.runs[0].driver_path
        assert head is not None  # enforced by __post_init__
        return head

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "runs": [run.model_dump(mode="json") for run in self.runs],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


@runtime_checkable
class SweepGenerator(Protocol):
    """Zero-arg callable returning a ``SweepSpec``.

    User sweep files expose one or more such functions; ``pd-run
    --sweep_generator_path /abs/path/file.py:func_name`` imports the file and
    calls the function.
    """

    def __call__(self) -> SweepSpec: ...
