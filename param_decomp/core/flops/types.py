"""Logical work and parameter shapes shared by architecture descriptions and accounting.

These values contain no weights, placement, optimizer policy or training objectives.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ForwardBackwardFlops:
    forward: int
    backward: int

    def __post_init__(self) -> None:
        if self.forward < 0 or self.backward < 0:
            raise ValueError("FLOP counts must be nonnegative")

    @property
    def total(self) -> int:
        return self.forward + self.backward


@dataclass(frozen=True)
class MatrixParameters:
    n_rows: int
    n_columns: int
    n_matrices: int

    def __post_init__(self) -> None:
        if min(self.n_rows, self.n_columns, self.n_matrices) <= 0:
            raise ValueError("Matrix dimensions and counts must be positive")

    @property
    def n_parameters(self) -> int:
        return self.n_matrices * self.n_rows * self.n_columns


@dataclass(frozen=True)
class ParameterCensus:
    matrices: tuple[MatrixParameters, ...]
    n_vector_parameters: int

    def __post_init__(self) -> None:
        if self.n_vector_parameters < 0:
            raise ValueError("Vector parameter counts must be nonnegative")

    @property
    def n_parameters(self) -> int:
        return sum(matrix.n_parameters for matrix in self.matrices) + self.n_vector_parameters


@dataclass(frozen=True)
class UsefulFlops:
    model: float
    optimizer: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) and value >= 0 for value in (self.model, self.optimizer)):
            raise ValueError("Useful FLOPs must be finite and nonnegative")


type StepFlops = Callable[[int], UsefulFlops]
