"""Launch data model: ``SweepSpec`` (many complete runs).

A single PD launch is a ``RunConfig``. A sweep is a list of complete
``RunConfig`` objects plus a concurrency cap. Sweep generators are ordinary
zero-arg Python functions returning a ``SweepSpec``.
"""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml
from pydantic import PositiveInt

from param_decomp.base_config import BaseConfig
from param_decomp.run import RunConfig


class SweepSpec(BaseConfig):
    """A complete sweep: description and the exact runs to launch.

    Serialized to ``PARAM_DECOMP_OUT_DIR/sweeps/<launch_id>/spec.yaml`` on
    submit so reproducing the sweep doesn't require re-running the generator.
    """

    description: str
    n_agents: PositiveInt
    runs: list[RunConfig]

    def run_cfgs(self) -> list[RunConfig]:
        return self.runs

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "n_agents": self.n_agents,
            "runs": [run.to_dict() for run in self.runs],
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
