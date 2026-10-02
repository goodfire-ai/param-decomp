"""Llama 3 base-text preprocessing with document-disjoint attention.

Each source row becomes BOS, unmodified literal text, EOS, all under one document
ID. Empty text produces BOS and EOS. Embedded special-token spellings remain text.
Long documents continue across packed rows without extra boundary tokens. Each
1000-document map batch drops its incomplete final row; worker partitioning can
therefore change row cuts and discarded tails. Streaming applies the same policy.
"""

from dataclasses import dataclass
from functools import partial

import numpy as np
from datasets import Dataset, IterableDataset, Value
from numpy.typing import NDArray
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from param_decomp.experiments.lm.data import pack_documents

LLAMA3_PREPROCESSING_NAME = "llama3-bos-eos-v1"
LLAMA3_TOKENIZERS = frozenset(
    {
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-70B",
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-70B",
        "meta-llama/Llama-3.1-405B",
    }
)


@dataclass(frozen=True)
class Llama3Tokenizer:
    tokenizer: PreTrainedTokenizerFast

    def __post_init__(self) -> None:
        assert self.tokenizer.bos_token_id == 128000, "Llama 3 requires BOS token ID 128000"
        assert self.tokenizer.eos_token_id == 128001, "Llama 3 requires EOS token ID 128001"
        assert len(self.tokenizer) == 128256, "Llama 3 requires a 128256-token vocabulary"
        vocabulary = self.tokenizer.get_vocab()
        for spelling, token_id in (
            ("<|begin_of_text|>", 128000),
            ("<|end_of_text|>", 128001),
            ("<|eot_id|>", 128009),
        ):
            assert vocabulary.get(spelling) == token_id, f"unexpected Llama 3 token: {spelling}"
        assert self.tokenizer.backend_tokenizer.normalizer is None, (
            "Llama 3 raw text must not be normalized"
        )

    def encode_document(self, text: str) -> tuple[int, ...]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)
        assert all(0 <= token < 128000 for token in tokens), (
            "Llama 3 document text must encode entirely as ordinary tokens"
        )
        return (128000, *tokens, 128001)


def load_llama3_tokenizer(name: str) -> Llama3Tokenizer:
    assert name in LLAMA3_TOKENIZERS, f"unsupported Llama 3 base tokenizer: {name}"
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    assert isinstance(tokenizer, PreTrainedTokenizerFast), "Llama 3 requires a fast tokenizer"
    return Llama3Tokenizer(tokenizer)


def _pack_llama3_texts(
    texts: list[str], *, tokenizer: Llama3Tokenizer, max_length: int
) -> dict[str, NDArray[np.int32]]:
    packed = pack_documents(
        [tokenizer.encode_document(text) for text in texts], max_length=max_length
    )
    return {"input_ids": packed.token_ids, "document_ids": packed.document_ids}


def _text_columns(dataset: Dataset | IterableDataset, column_name: str) -> list[str]:
    features = dataset.features
    assert features is not None, "dataset features must be known"
    assert column_name in features, f"missing text column: {column_name}"
    feature = features[column_name]
    assert isinstance(feature, Value) and feature.dtype == "string", (
        "Llama 3 requires a string text column"
    )
    return list(features)


def preprocess_llama3(
    dataset: Dataset,
    tokenizer: Llama3Tokenizer,
    *,
    column_name: str,
    max_length: int,
    num_proc: int,
) -> Dataset:
    """Pack Llama 3 documents using worker processes; each map batch drops its tail."""
    assert max_length > 0, "max_length must be positive"
    assert num_proc > 0, "num_proc must be positive"
    return dataset.map(
        partial(_pack_llama3_texts, tokenizer=tokenizer, max_length=max_length),
        input_columns=column_name,
        batched=True,
        batch_size=1000,
        remove_columns=_text_columns(dataset, column_name),
        num_proc=num_proc,
    )


def preprocess_llama3_streaming(
    dataset: IterableDataset,
    tokenizer: Llama3Tokenizer,
    *,
    column_name: str,
    max_length: int,
) -> IterableDataset:
    """Lazily pack Llama 3 documents; each map batch drops its incomplete final row."""
    assert max_length > 0, "max_length must be positive"
    return dataset.map(
        partial(_pack_llama3_texts, tokenizer=tokenizer, max_length=max_length),
        input_columns=column_name,
        batched=True,
        batch_size=1000,
        remove_columns=_text_columns(dataset, column_name),
    )
