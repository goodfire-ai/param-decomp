"""Dictionary composition with disjoint keys, and an immutable mapping hashable by value."""

from collections.abc import Hashable, Iterator, Mapping
from typing import override


def dict_safe_update_[K, V](target: dict[K, V], additions: Mapping[K, V]) -> None:
    """Add disjoint entries; reject every collision before changing the target."""
    overlap = tuple(key for key in additions if key in target)
    if overlap:
        raise ValueError(f"dictionary keys already present: {overlap!r}")
    target.update(additions)


class FrozenMapping[K: Hashable, V: Hashable](Mapping[K, V]):
    """An immutable mapping that hashes by its entries, so a frozen dataclass holding one
    can be a jit static argument."""

    __slots__ = ("_entries", "_hash")

    def __init__(self, entries: Mapping[K, V]) -> None:
        self._entries = dict(entries)
        self._hash = hash(frozenset(self._entries.items()))

    @override
    def __getitem__(self, key: K) -> V:
        return self._entries[key]

    @override
    def __iter__(self) -> Iterator[K]:
        return iter(self._entries)

    @override
    def __len__(self) -> int:
        return len(self._entries)

    @override
    def __hash__(self) -> int:
        return self._hash

    @override
    def __repr__(self) -> str:
        return f"FrozenMapping({self._entries!r})"
