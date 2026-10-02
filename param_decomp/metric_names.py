"""Two-level metric names, independent of a logging backend."""

import re
from dataclasses import dataclass


def _validate_name(name: str) -> None:
    if re.fullmatch(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", name) is None:
        raise ValueError(f"Expected a lowercase name with dash-separated words: {name!r}")


@dataclass(frozen=True)
class MetricKey:
    group: str
    chart: str

    def __post_init__(self) -> None:
        _validate_name(self.group)
        _validate_name(self.chart)

    @property
    def path(self) -> str:
        return f"{self.group}/{self.chart}"


@dataclass(frozen=True)
class MetricGroup:
    name: str
    charts: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if not self.charts or len(set(self.charts)) != len(self.charts):
            raise ValueError(f"A metric group needs a nonempty, unique chart list: {self.name}")
        for chart in self.charts:
            _validate_name(chart)

    def key(self, chart: str) -> MetricKey:
        if chart not in self.charts:
            raise ValueError(f"Unknown chart {chart!r} in group {self.name!r}")
        return MetricKey(self.name, chart)


@dataclass(frozen=True)
class MetricGrid:
    rows: tuple[str, ...]
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        for axis in (self.rows, self.columns):
            if not axis or len(set(axis)) != len(axis):
                raise ValueError("Grid axes must be nonempty and unique")
            for name in axis:
                _validate_name(name)
        if len(set(self.charts)) != len(self.charts):
            raise ValueError("Grid coordinates produce colliding chart names")

    @property
    def charts(self) -> tuple[str, ...]:
        return tuple(f"{row}-{column}" for row in self.rows for column in self.columns)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.columns), len(self.rows)
