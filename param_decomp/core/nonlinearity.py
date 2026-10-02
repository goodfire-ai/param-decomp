"""JAX-free vocabulary for nonlinearity-facing component alignment."""

from dataclasses import dataclass
from typing import ClassVar, Literal

NonlinearityUnitKind = Literal["neuron", "attention_head", "deltanet_head"]


@dataclass(frozen=True)
class Neurons:
    """One nonlinearity unit per aligned coordinate, each used once."""

    unit_kind: ClassVar[NonlinearityUnitKind] = "neuron"
    use_multiplicity: ClassVar[int] = 1


@dataclass(frozen=True)
class QueryHeads:
    """One contiguous block per query head on a q projection's output or an attention
    output projection's input. Each block belongs to one attention head."""

    head_count: int

    unit_kind: ClassVar[NonlinearityUnitKind] = "attention_head"
    use_multiplicity: ClassVar[int] = 1

    def __post_init__(self) -> None:
        assert self.head_count >= 1, f"head_count must be positive: {self.head_count}"


@dataclass(frozen=True)
class KVHeads:
    """Equal contiguous blocks of a k/v projection's aligned axis, one per kv head.

    Under GQA a kv block is written once but consumed by `n_head / n_kv_head`
    query-attention nonlinearities, and the soft count measures uses —
    so `use_multiplicity` is an explicit positive count.
    """

    head_count: int
    use_multiplicity: int

    unit_kind: ClassVar[NonlinearityUnitKind] = "attention_head"

    def __post_init__(self) -> None:
        assert self.head_count >= 1, f"head_count must be positive: {self.head_count}"
        assert self.use_multiplicity >= 1, (
            f"use_multiplicity must be positive: {self.use_multiplicity}"
        )


@dataclass(frozen=True)
class DeltaNetHeads:
    """Equal contiguous blocks of a gated-DeltaNet projection's aligned axis, one per
    head: the per-head l2norm and delta-rule recurrence for a key head, the
    per-head gated RMS norm for a value head. A key head is read by
    `linear_num_value_heads / linear_num_key_heads` value-head recurrences (the q/k heads
    repeat to the value-head count), so like `KVHeads` its `use_multiplicity` is a real
    field; a value head is used once."""

    head_count: int
    use_multiplicity: int

    unit_kind: ClassVar[NonlinearityUnitKind] = "deltanet_head"

    def __post_init__(self) -> None:
        assert self.head_count >= 1, f"head_count must be positive: {self.head_count}"
        assert self.use_multiplicity >= 1, (
            f"use_multiplicity must be positive: {self.use_multiplicity}"
        )


NonlinearityPartition = Neurons | QueryHeads | KVHeads | DeltaNetHeads


ComponentSide = Literal["input", "output"]


@dataclass(frozen=True)
class NonlinearityAlignment:
    """A matrix's nonlinearity-facing side and its partition into units.

    Initialization and locality independently consume this architectural geometry.
    """

    side: ComponentSide
    partition: NonlinearityPartition
