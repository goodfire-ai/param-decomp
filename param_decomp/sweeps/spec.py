"""Launch data model: ``SweepSpec`` (many runs sharing a driver and substrate).

A single PD launch is just a ``RunConfig`` — the same type the worker writes
to disk. A sweep is a ``SweepSpec`` carrying shared driver/logging/runtime
plus a ``list[SweepData]`` of per-run varying ``pd``/``name``/``view_meta``.

A ``SweepGenerator`` is any zero-arg callable returning a ``SweepSpec``. The
sweep is fully self-contained — it loads whatever base config it wants and
the spec declares the driver shared across all runs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from param_decomp import PDConfig
from param_decomp.configs import LoggingConfig, RuntimeConfig
from param_decomp.run import RunConfig


@dataclass(frozen=True)
class SweepData:
    name: str
    pd_config: PDConfig
    view_meta: dict[str, Any]


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
    driver_path: str | None
    logging: LoggingConfig
    runtime: RuntimeConfig
    n_agents: int

    swept_datas: list[SweepData]

    def run_cfgs(self) -> list[RunConfig]:
        return [
            RunConfig(
                name=sweep_data.name,
                driver_path=self.driver_path,
                pd=sweep_data.pd_config,
                logging=self.logging,
                runtime=self.runtime,
                view_meta=sweep_data.view_meta,
            )
            for sweep_data in self.swept_datas
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "driver_path": self.driver_path,
            "logging": self.logging.model_dump(mode="json"),
            "runtime": self.runtime.model_dump(mode="json"),
            "n_agents": self.n_agents,
            "swept_data": [
                {
                    "name": sweep_data.name,
                    "pd_config": sweep_data.pd_config.model_dump(mode="json"),
                    "view_meta": sweep_data.view_meta,
                }
                for sweep_data in self.swept_datas
            ],
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
