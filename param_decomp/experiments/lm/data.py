"""Language-model HuggingFace dataset loading."""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from datasets import Dataset, IterableDataset, load_dataset
from numpy.typing import NDArray
from pydantic import Field, PositiveInt
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, PreTrainedTokenizer

from param_decomp.base_config import BaseConfig
from param_decomp.log import logger
from param_decomp.utils.distributed_utils import DistributedState


class LMDataConfig(BaseConfig):
    """LM experiment dataset / dataloader settings."""

    dataset_name: str = Field(..., description="HuggingFace dataset id")
    tokenizer_name: str = Field(..., description="HF tokenizer id or path")
    column_name: str = Field(default="text", description="Dataset column with the text/tokens")
    max_seq_len: PositiveInt = Field(default=512, description="Max sequence length")
    train_split: str = Field(default="train")
    eval_split: str = Field(default="test")
    is_tokenized: bool = Field(default=False)
    streaming: bool = Field(default=False)
    buffer_size: PositiveInt = Field(default=1000)
    shuffle_each_epoch: bool = Field(default=True)
    dataset_shuffle_seed: int = Field(
        default=0,
        description="Dataset shuffle seed",
    )


class LMDataLoaderConfig(BaseConfig):
    """Split-specific LM dataloader config.

    This exists for pretraining configs, which specify separate train/validation dataset configs.
    LM experiments should normally use `LMDataConfig` plus `build_lm_dataloaders`.
    """

    name: str
    is_tokenized: bool
    hf_tokenizer_path: str
    streaming: bool
    split: str
    n_ctx: int
    """Must be model n_ctx + 1 to provide room for next-token label indexing."""
    seed: int | None = None
    column_name: str
    """Dataset column containing text or token ids."""
    shuffle_each_epoch: bool = True


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
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[column_name],
        )
    else:
        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=[column_name], num_proc=num_proc
        )

    return tokenized_dataset.with_format("torch")


def _prepare_lm_dataset(
    dataset: Dataset | IterableDataset,
    *,
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    column_name: str,
    max_seq_len: int,
    is_tokenized: bool,
) -> Dataset | IterableDataset:
    if is_tokenized:
        torch_dataset = dataset.with_format("torch")
        sample = next(iter(torch_dataset))[column_name]
        assert isinstance(sample, Tensor) and sample.ndim == 1, (
            f"Expected the dataset to be tokenized. Got type {type(sample)}"
        )
        tokenized_len = len(sample)
        assert max_seq_len <= tokenized_len, (
            f"max_seq_len ({max_seq_len}) is larger than the tokenized length ({tokenized_len})."
        )
        if max_seq_len < tokenized_len:
            torch_dataset = dataset.map(lambda x: {column_name: x[column_name][:max_seq_len]})
            torch_dataset = torch_dataset.with_format("torch")
        return torch_dataset

    to_lower = "SimpleStories" in dataset_name
    return tokenize_and_concatenate(
        dataset,
        tokenizer,
        max_length=max_seq_len,
        column_name=column_name,
        add_bos_token=False,
        to_lower=to_lower,
    )


def create_lm_data_loader(
    *,
    dataset_name: str,
    tokenizer_name: str,
    split: str,
    max_seq_len: int,
    is_tokenized: bool,
    streaming: bool,
    column_name: str,
    batch_size: int,
    buffer_size: int,
    seed: int,
    shuffle_each_epoch: bool = True,
    dist_state: DistributedState | None = None,
    collate_fn: Callable[..., Any] | None = None,
) -> tuple[DataLoader[Any], PreTrainedTokenizer]:
    """Create an LM token dataloader from a HuggingFace dataset split."""
    dataset = load_dataset(
        dataset_name,
        streaming=streaming,
        split=split,
        trust_remote_code=False,
    )
    assert isinstance(dataset, Dataset | IterableDataset)

    if streaming:
        assert isinstance(dataset, IterableDataset)
        if dist_state is not None:
            ds_num_shards = getattr(dataset, "num_shards", None)
            if isinstance(ds_num_shards, int) and ds_num_shards >= dist_state.world_size:
                dataset = dataset.shard(num_shards=dist_state.world_size, index=dist_state.rank)
            else:
                dataset = dataset.filter(
                    lambda _ex, idx: idx % dist_state.world_size == dist_state.rank,
                    with_indices=True,
                )
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    else:
        assert isinstance(dataset, Dataset)
        logger.info("Shuffling dataset (len=%d)", len(dataset))
        dataset = dataset.shuffle(seed=seed)
        logger.info("Shuffled dataset")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    torch_dataset = _prepare_lm_dataset(
        dataset,
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        column_name=column_name,
        max_seq_len=max_seq_len,
        is_tokenized=is_tokenized,
    )

    sampler = None
    if not streaming and dist_state is not None:
        sampler = DistributedSampler(
            torch_dataset,  # pyright: ignore[reportArgumentType]
            num_replicas=dist_state.world_size,
            rank=dist_state.rank,
            shuffle=shuffle_each_epoch,
            seed=seed,
            drop_last=True,
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    loader = DataLoader[Dataset | IterableDataset](
        torch_dataset,  # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and shuffle_each_epoch and not streaming),
        drop_last=True,
        generator=generator,
        collate_fn=collate_fn,
    )
    return loader, tokenizer


def _rank_batch_size(batch_size: int, dist_state: DistributedState | None, *, label: str) -> int:
    if dist_state is None:
        return batch_size

    world_size = dist_state.world_size
    assert batch_size % world_size == 0 and batch_size > 0, (
        f"{label} {batch_size} not divisible by world size {world_size}"
    )
    return batch_size // world_size


def build_lm_dataloaders(
    data_cfg: LMDataConfig,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    dist_state: DistributedState | None,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Build train/eval dataloaders from total batch sizes."""
    train_batch_size = _rank_batch_size(train_batch_size, dist_state, label="train_batch_size")
    eval_batch_size = _rank_batch_size(eval_batch_size, dist_state, label="eval_batch_size")
    collate_column = data_cfg.column_name if data_cfg.is_tokenized else "input_ids"

    data_seed = data_cfg.dataset_shuffle_seed

    def collate_token_column(batch: list[dict[str, Tensor]]) -> Tensor:
        return torch.stack([item[collate_column] for item in batch])

    train_loader, _ = create_lm_data_loader(
        dataset_name=data_cfg.dataset_name,
        tokenizer_name=data_cfg.tokenizer_name,
        split=data_cfg.train_split,
        max_seq_len=data_cfg.max_seq_len,
        is_tokenized=data_cfg.is_tokenized,
        streaming=data_cfg.streaming,
        column_name=data_cfg.column_name,
        batch_size=train_batch_size,
        buffer_size=data_cfg.buffer_size,
        seed=data_seed,
        shuffle_each_epoch=data_cfg.shuffle_each_epoch,
        dist_state=dist_state,
        collate_fn=collate_token_column,
    )
    eval_loader, _ = create_lm_data_loader(
        dataset_name=data_cfg.dataset_name,
        tokenizer_name=data_cfg.tokenizer_name,
        split=data_cfg.eval_split,
        max_seq_len=data_cfg.max_seq_len,
        is_tokenized=data_cfg.is_tokenized,
        streaming=data_cfg.streaming,
        column_name=data_cfg.column_name,
        batch_size=eval_batch_size,
        buffer_size=data_cfg.buffer_size,
        seed=data_seed + 1,
        shuffle_each_epoch=data_cfg.shuffle_each_epoch,
        dist_state=dist_state,
        collate_fn=collate_token_column,
    )
    return train_loader, eval_loader
