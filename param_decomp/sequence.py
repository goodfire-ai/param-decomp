"""Document boundaries within batch sequences, shared by models, kernels, and CI functions."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, Shaped


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SequenceLayout:
    """Contiguous document labels within each batch sequence; -1 denotes right padding."""

    document_ids: Int[Array, "b t"]

    @classmethod
    def unsegmented_sequences_like(cls, reference: Shaped[Array, "b t"]) -> "SequenceLayout":
        """Use the reference shape and sharding, without boundaries inside each sequence."""
        return cls(jnp.zeros_like(reference, dtype=jnp.int32))

    def attention_mask(self) -> Bool[Array, "b t t"]:
        ids = self.document_ids
        return ids[:, :, None] == ids[:, None, :]

    def position_resets(self) -> Bool[Array, "b t"]:
        """Reset at sequence starts and document-ID changes, including the start of padding."""
        ids = self.document_ids
        return jnp.concatenate(
            (jnp.ones_like(ids[:, :1], dtype=bool), ids[:, 1:] != ids[:, :-1]), axis=1
        )

    def position_ids(self) -> Int[Array, "b t"]:
        positions = jnp.arange(self.document_ids.shape[1], dtype=jnp.int32)[None, :]
        starts = jnp.where(self.position_resets(), positions, 0)
        offsets = jax.lax.associative_scan(jnp.maximum, starts, axis=1)
        return positions - offsets

    def next_token_mask(self) -> Bool[Array, "b t_minus_one"]:
        ids = self.document_ids
        return (ids[:, :-1] == ids[:, 1:]) & (ids[:, :-1] >= 0)
