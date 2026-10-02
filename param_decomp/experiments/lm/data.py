"""Pack encoded documents while preserving their attention boundaries."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PackedDocuments:
    token_ids: NDArray[np.int32]
    document_ids: NDArray[np.int32]


def pack_documents(documents: Sequence[Sequence[int]], *, max_length: int) -> PackedDocuments:
    """Each output row is a sequence, which can contain multiple documents or a fragment.
    Documents may span sequences; the incomplete final sequence is dropped.

    IDs are local to this batch, since attention never crosses rows. Boundary
    tokens must already belong to their documents before packing.
    """
    assert max_length > 0, "max_length must be positive"
    max_token_id = np.iinfo(np.int32).max
    token_ids: list[int] = []
    document_ids: list[int] = []
    for document_id, tokens in enumerate(documents):
        assert tokens, "encoded documents must contain at least one token"
        assert all(0 <= token <= max_token_id for token in tokens), (
            "token IDs must be nonnegative int32 values"
        )
        token_ids.extend(tokens)
        document_ids.extend([document_id] * len(tokens))
    n_tokens = len(token_ids) // max_length * max_length
    return PackedDocuments(
        token_ids=np.asarray(token_ids[:n_tokens], dtype=np.int32).reshape(-1, max_length),
        document_ids=np.asarray(document_ids[:n_tokens], dtype=np.int32).reshape(-1, max_length),
    )
