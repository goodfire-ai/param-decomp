"""LM token sequences, with independently composed document layout and routing."""

from dataclasses import dataclass

import jax
from jaxtyping import Array, Int

from param_decomp.core.components import BlockSelection
from param_decomp.sequence import SequenceLayout


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMBatch:
    """Token IDs for fixed-length sequences; each row is one batch element."""

    token_ids: Int[Array, "b t"]

    def validate_shapes(self) -> None:
        assert self.token_ids.ndim == 2, self.token_ids.shape


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMBatchWithDocuments:
    batch: LMBatch
    sequence: SequenceLayout

    def validate_shapes(self) -> None:
        self.batch.validate_shapes()
        assert self.sequence.document_ids.shape == self.batch.token_ids.shape, (
            self.sequence.document_ids.shape,
            self.batch.token_ids.shape,
        )

    @classmethod
    def from_unsegmented_sequences(cls, token_ids: Int[Array, "b t"]) -> "LMBatchWithDocuments":
        """Allow attention across each entire sequence, without internal document boundaries."""
        return cls(LMBatch(token_ids), SequenceLayout.unsegmented_sequences_like(token_ids))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMBatchWithRouting[Batch]:
    """Retain an input batch alongside its clean expert selection."""

    batch: Batch
    selection: BlockSelection
