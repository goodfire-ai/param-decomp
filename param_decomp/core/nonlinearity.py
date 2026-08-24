"""JAX-free vocabulary for nonlinearity-facing output partitions (SPEC S36)."""

from dataclasses import dataclass
from typing import ClassVar, Literal

NonlinearityUnitKind = Literal["neuron", "attention_head"]


@dataclass(frozen=True)
class Neurons:
    """One nonlinearity unit per output coordinate (SPEC S36)."""

    unit_kind: ClassVar[NonlinearityUnitKind] = "neuron"


@dataclass(frozen=True)
class AttentionHeads:
    """Equal contiguous per-head blocks of a site's output axis (SPEC S36)."""

    head_count: int

    unit_kind: ClassVar[NonlinearityUnitKind] = "attention_head"

    def __post_init__(self) -> None:
        assert self.head_count >= 1, f"head_count must be positive: {self.head_count}"


NonlinearityPartition = Neurons | AttentionHeads
