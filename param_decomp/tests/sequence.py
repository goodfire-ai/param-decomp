"""Sequence layouts for fixtures with no internal document boundaries."""

import jax.numpy as jnp
from jaxtyping import Array

from param_decomp.sequence import SequenceLayout


def unsegmented_sequence_layout(taps: dict[str, Array]) -> SequenceLayout | None:
    sample = next(iter(taps.values()))
    match sample.ndim:
        case 2:
            return None
        case 3:
            return SequenceLayout(jnp.zeros_like(sample[..., 0], dtype=jnp.int32))
        case other:
            raise AssertionError(f"unexpected fixture rank: {other}")
