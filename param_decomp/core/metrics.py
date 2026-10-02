"""Typed, transport-independent metric records."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BarChart:
    rows: tuple[tuple[str, float], ...]
    x_label: str
    y_label: str
    title: str


@dataclass(frozen=True)
class LineChart:
    """Named series over one shared x axis."""

    xs: np.ndarray
    series: tuple[tuple[str, np.ndarray], ...]
    x_label: str
    title: str

    def __post_init__(self) -> None:
        # W&B zips each series against xs non-strictly, silently dropping unmatched values.
        assert self.xs.ndim == 1 and all(ys.shape == self.xs.shape for _, ys in self.series), (
            [ys.shape for _, ys in self.series],
            self.xs.shape,
        )


@dataclass(frozen=True)
class PNGImage:
    encoded: bytes


type MetricValue = float | BarChart | LineChart | PNGImage
type LogRecord = Mapping[str, MetricValue]
