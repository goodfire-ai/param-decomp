"""JAX-free vocabulary for nonlinearity-facing output partitions (SPEC S36)."""

from dataclasses import dataclass
from typing import ClassVar, Literal

NonlinearityUnitKind = Literal["neuron", "attention_head"]


@dataclass(frozen=True)
class Neurons:
    """One nonlinearity unit per output coordinate, each used once (SPEC S36)."""

    unit_kind: ClassVar[NonlinearityUnitKind] = "neuron"
    use_multiplicity: ClassVar[int] = 1


@dataclass(frozen=True)
class QueryHeads:
    """Equal contiguous blocks of a q projection's output axis, one per query head
    (SPEC S36). Each block is used in exactly its own attention nonlinearity."""

    head_count: int

    unit_kind: ClassVar[NonlinearityUnitKind] = "attention_head"
    use_multiplicity: ClassVar[int] = 1

    def __post_init__(self) -> None:
        assert self.head_count >= 1, f"head_count must be positive: {self.head_count}"


@dataclass(frozen=True)
class KVHeads:
    """Equal contiguous blocks of a k/v projection's output axis, one per kv head
    (SPEC S36).

    Under GQA a kv block is written once but consumed by `n_head / n_kv_head`
    query-attention nonlinearities, and the soft count measures uses (SPEC S36) —
    so this is the one partition where `use_multiplicity` is a real field.
    """

    head_count: int
    use_multiplicity: int

    unit_kind: ClassVar[NonlinearityUnitKind] = "attention_head"

    def __post_init__(self) -> None:
        assert self.head_count >= 1, f"head_count must be positive: {self.head_count}"
        assert self.use_multiplicity >= 1, (
            f"use_multiplicity must be positive: {self.use_multiplicity}"
        )


NonlinearityPartition = Neurons | QueryHeads | KVHeads
