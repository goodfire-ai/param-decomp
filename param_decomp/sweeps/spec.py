"""Sweep data model and generator protocol.

A ``SweepGenerator`` is anything callable as ``(base_config) -> SweepSpec``. The
recommended pattern is to subclass ``SweepGenerator`` (the ABC defined here) so
built-in discovery can find it by ``name``, but the runner accepts any callable
with the right shape.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import yaml


@dataclass(frozen=True)
class SweepRun:
    """One concrete run produced by a sweep generator.

    - ``name`` is a short identifier the runner uses for the W&B run name and
      for human-readable logging. It does NOT need to encode the axes — that's
      what ``view_meta`` is for.
    - ``config`` is the materialized experiment config dict (driver-validated by
      the runner before submission).
    - ``view_meta`` carries researcher-facing labels (e.g. ``{"lr_ratio": 0.1,
      "size": "medium"}``) that get surfaced to W&B under a ``view_meta/`` prefix.
      Values should be JSON scalars (str | int | float | bool) so W&B can group
      and color by them.
    """

    name: str
    config: dict[str, Any]
    view_meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SweepSpec:
    """A complete sweep: a human-facing description plus the list of runs to launch.

    Serialized to ``PARAM_DECOMP_OUT_DIR/sweeps/<launch_id>/spec.yaml`` on submit
    so reproducing the sweep doesn't require re-running the generator code.
    """

    description: str
    runs: list[SweepRun]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


class SweepGenerator(ABC):
    """Base class for sweep generators.

    Subclasses must set ``name`` (used for auto-discovery / short CLI refs) and
    implement ``__call__(base_config)``. The constructor may accept an optional
    string argument (e.g. a yaml path) — the CLI surface is ``--sweep <name>:<arg>``.

    Custom generators don't have to subclass this — anything callable with the
    right signature works — but subclassing gives you auto-discovery for free.
    """

    name: ClassVar[str]

    def __init__(self, arg: str | None = None) -> None:
        """Default: reject args. Override if your generator takes one."""
        assert arg is None, f"{type(self).__name__} does not accept a CLI argument"

    @abstractmethod
    def __call__(self, base_config: dict[str, Any]) -> SweepSpec: ...
