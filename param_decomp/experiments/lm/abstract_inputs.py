"""Abstract token and document-aware inputs for LM trace and memory checks."""

from typing import Any, cast

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from param_decomp.core.placement import batch_axes
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.sequence import SequenceLayout


def abstract_lm_batch_with_documents(
    batch_size: int, seq_len: int, mesh: Mesh
) -> LMBatchWithDocuments:
    batch = abstract_lm_batch(batch_size, seq_len, mesh)
    return LMBatchWithDocuments(
        batch, SequenceLayout(_abstract_token_ids(batch_size, seq_len, mesh))
    )


def abstract_lm_batch(batch_size: int, seq_len: int, mesh: Mesh) -> LMBatch:
    return LMBatch(_abstract_token_ids(batch_size, seq_len, mesh))


def _abstract_token_ids(batch_size: int, seq_len: int, mesh: Mesh) -> Array:
    # AOT inputs occupy Array positions but carry shapes and concrete entry shardings.
    return cast(
        Any,
        jax.ShapeDtypeStruct(
            (batch_size, seq_len),
            np.int32,
            sharding=NamedSharding(mesh, P(batch_axes(mesh), None)),
        ),
    )
