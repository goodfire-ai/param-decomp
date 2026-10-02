"""Chunk-stack padding shared by the independent chunkwise implementations.

A placement whose stack cut does not tile the chunk count pads the persist stack with
trailing zero chunks. These pads ride the persist→compute gather and leave only after it
(`real_chunks`), unlike the V/U stacks, which strip theirs before gathering
(`placement.materialize_reduced_weights`). CI weights are small and the pad is at most
(stack-sharding degree − 1) chunks, so gathering the pads is cheap and each CI leaf
enters compute in one reshard."""

import equinox as eqx
import jax
import jax.numpy as jnp

from param_decomp.core.placement import StackCensus


def unpadded_census(n_chunks: int) -> StackCensus:
    """The real chunks alone: an unplaced fn's stack, which no placement cuts."""
    return StackCensus(stack_len=n_chunks, stack_pad=0)


def validated_census(census: StackCensus, n_chunks: int) -> StackCensus:
    """A placement's resolved census, checked against the chunks a fn routes."""
    assert census.stack_len == n_chunks, (
        f"placement expects a {census.stack_len}-chunk CI fn; this fn routes {n_chunks} chunks"
    )
    return census


def pad_chunk_stack[Chunks: eqx.Module](chunks: Chunks, census: StackCensus) -> Chunks:
    """Append the census's persist pad to real chunk leaves."""
    pad = census.stack_pad
    if pad == 0:
        return chunks
    return jax.tree.map(
        lambda leaf: jnp.concatenate([leaf, jnp.zeros((pad, *leaf.shape[1:]), leaf.dtype)]), chunks
    )


def validate_chunk_leaves(chunks: eqx.Module, census: StackCensus) -> None:
    """Chunk leaves, stored or compute, hold exactly the census's real chunks plus its pad."""
    for leaf in jax.tree.leaves(chunks):
        assert leaf.shape[0] == census.padded_stack_len, (
            "chunk leaf disagrees with the placement's padded extent",
            leaf.shape,
            census.padded_stack_len,
        )


def real_chunks[Chunks: eqx.Module](chunks: Chunks, census: StackCensus) -> Chunks:
    """The leading real chunks of a stack whose axis rests whole, so the slice is local.
    Its transpose writes exact zeros into the pad cotangents."""
    if census.stack_pad == 0:
        return chunks
    return jax.tree.map(
        lambda leaf: jax.lax.slice_in_dim(leaf, 0, census.stack_len, axis=0), chunks
    )
