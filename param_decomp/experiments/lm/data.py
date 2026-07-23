"""Language-model HuggingFace tokenization helper for the offline pre-staging tool.

`prestage_tokenized.py` is the only consumer: it tokenizes + concatenates text into
fixed-length int sequences and writes int32 parquet shards. Run-time data loading is the
JAX trainer's own `ShardServer` (pre-tokenized parquet, never streamed from HF)."""

from typing import Any

import numpy as np
from datasets import Dataset, IterableDataset
from numpy.typing import NDArray
from transformers import PreTrainedTokenizer


def _keep_single_column(
    dataset: Dataset | IterableDataset, col_name: str
) -> Dataset | IterableDataset:
    """Remove all HuggingFace dataset columns except `col_name`."""
    features = dataset.features
    assert features is not None, "Dataset features must be known to drop unused columns."
    for key in features:
        if key != col_name:
            dataset = dataset.remove_columns(key)
    return dataset


def tokenize_and_concatenate(
    dataset: Dataset | IterableDataset,
    tokenizer: PreTrainedTokenizer,
    column_name: str,
    max_length: int = 1024,
    add_bos_token: bool = False,
    num_proc: int = 10,
    to_lower: bool = False,
) -> Dataset | IterableDataset:
    """Tokenize text, concatenate documents, and chunk into fixed-length token sequences.

    Adapted from TransformerLens' tokenizer helper, with support for streaming datasets.
    """
    dataset = _keep_single_column(dataset, column_name)
    seq_len = max_length - 1 if add_bos_token else max_length

    def tokenize_function(
        examples: dict[str, list[str]],
    ) -> dict[
        str,
        NDArray[np.signedinteger[Any]],
    ]:
        text = examples[column_name]
        assert hasattr(tokenizer, "eos_token") and isinstance(tokenizer.eos_token, str)
        full_text = tokenizer.eos_token.join(text)

        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)]

        if to_lower:
            chunks = [
                chunk.replace(tokenizer.eos_token.lower(), tokenizer.eos_token) for chunk in chunks
            ]
        tokens = [tokenizer.encode(chunk, add_special_tokens=False) for chunk in chunks]
        tokens = np.concatenate(tokens)

        num_tokens = len(tokens)
        num_batches = num_tokens // seq_len
        tokens = tokens[: seq_len * num_batches]
        tokens = tokens.reshape((num_batches, seq_len))

        if add_bos_token:
            assert hasattr(tokenizer, "bos_token_id")
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)

        return {"input_ids": tokens}

    if isinstance(dataset, IterableDataset):
        return dataset.map(tokenize_function, batched=True, remove_columns=[column_name])
    return dataset.map(
        tokenize_function, batched=True, remove_columns=[column_name], num_proc=num_proc
    )
