"""Read tokens and valid labels from the supported LM input records."""

import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments


def input_token_ids(inputs: LMBatch | LMBatchWithDocuments) -> Int[Array, "b t"]:
    match inputs:
        case (
            LMBatch(token_ids=token_ids) | LMBatchWithDocuments(batch=LMBatch(token_ids=token_ids))
        ):
            return token_ids


def input_next_token_mask(inputs: LMBatch | LMBatchWithDocuments) -> Bool[Array, "b t_minus_one"]:
    match inputs:
        case LMBatchWithDocuments(sequence=sequence):
            return sequence.next_token_mask()
        case LMBatch(token_ids=token_ids):
            return jnp.ones_like(token_ids[:, 1:], dtype=jnp.bool_)
