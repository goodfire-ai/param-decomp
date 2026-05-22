"""Language-model HuggingFace dataset loading."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, override

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
    is_random: bool = Field(
        default=False,
        description="Synthesize uniform-random token ids for perf characterization. "
        "Skips HF dataset loading and tokenization. Requires random_vocab_size.",
    )
    random_vocab_size: PositiveInt | None = Field(
        default=None,
        description="Vocab range for random tokens. Required iff is_random=True.",
    )


class _RandomTokenIterable(torch.utils.data.IterableDataset[dict[str, Tensor]]):
    """Yields {'input_ids': randint(0, vocab_size, (seq_len,))} forever.

    Seeded per (rank, dataloader-worker) so ranks see disjoint streams.
    """

    def __init__(self, vocab_size: int, seq_len: int, seed: int, rank: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.seed = seed
        self.rank = rank

    @override
    def __iter__(self):
        g = torch.Generator()
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        g.manual_seed(self.seed + self.rank * 100003 + worker_id * 1009)
        while True:
            yield {
                "input_ids": torch.randint(
                    0, self.vocab_size, (self.seq_len,), generator=g, dtype=torch.long
                )
            }


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


def _tokenize_and_concatenate(
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
    return _tokenize_and_concatenate(
        dataset,
        tokenizer,
        max_length=max_seq_len,
        column_name=column_name,
        add_bos_token=False,
        to_lower=to_lower,
    )


def create_lm_data_loader(
    cfg: LMDataConfig,
    *,
    split: str,
    batch_size: int,
    seed: int,
    dist_state: DistributedState | None = None,
    collate_fn: Callable[..., Any] | None = None,
) -> tuple[DataLoader[Any], PreTrainedTokenizer]:
    """Create an LM token dataloader from a HuggingFace dataset split.

    ``cfg.dataset_name`` may be either an HF hub id (passed to ``load_dataset``)
    or a local directory previously produced by ``Dataset.save_to_disk``
    (e.g. via ``param_decomp/scripts/pretokenize_pile.py``). Local paths are
    detected by directory existence + ``streaming=False``.

    When ``cfg.is_random=True``, all HF dataset machinery is bypassed and the
    loader yields uniform-random token ids — for perf/memory characterization
    where loss values are not meaningful but step time and memory are.
    """
    if cfg.is_random:
        assert cfg.random_vocab_size is not None, "is_random=True requires random_vocab_size"
        rank = dist_state.rank if dist_state is not None else 0
        dataset_random = _RandomTokenIterable(
            vocab_size=cfg.random_vocab_size,
            seq_len=cfg.max_seq_len,
            seed=seed,
            rank=rank,
        )
        loader_random = DataLoader[Any](
            dataset_random,
            batch_size=batch_size,
            drop_last=True,
            collate_fn=collate_fn,
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        return loader_random, tokenizer

    dataset_path = Path(cfg.dataset_name)
    if dataset_path.is_dir():
        assert not cfg.streaming, (
            f"Local saved-to-disk datasets cannot be streamed; set streaming=false. "
            f"Got dataset_name={cfg.dataset_name} (a directory)."
        )
        from datasets import load_from_disk

        loaded = load_from_disk(cfg.dataset_name)
        dataset = loaded if isinstance(loaded, Dataset) else loaded[split]
    else:
        dataset = load_dataset(
            cfg.dataset_name,
            streaming=cfg.streaming,
            split=split,
            trust_remote_code=False,
        )
    assert isinstance(dataset, Dataset | IterableDataset)

    if cfg.streaming:
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
        dataset = dataset.shuffle(seed=seed, buffer_size=cfg.buffer_size)
    else:
        assert isinstance(dataset, Dataset)
        logger.info("Shuffling dataset (len=%d)", len(dataset))
        # keep_in_memory=True avoids HF writing a fingerprint cache file in
        # the dataset dir — under multi-rank launches (2-pool with 64+ procs),
        # the race on that cache write triggers SIGBUS from mmap.
        dataset = dataset.shuffle(seed=seed, keep_in_memory=True)
        logger.info("Shuffled dataset")

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    torch_dataset = _prepare_lm_dataset(
        dataset,
        dataset_name=cfg.dataset_name,
        tokenizer=tokenizer,
        column_name=cfg.column_name,
        max_seq_len=cfg.max_seq_len,
        is_tokenized=cfg.is_tokenized,
    )

    sampler = None
    if not cfg.streaming and dist_state is not None:
        sampler = DistributedSampler(
            torch_dataset,  # pyright: ignore[reportArgumentType]
            num_replicas=dist_state.world_size,
            rank=dist_state.rank,
            shuffle=cfg.shuffle_each_epoch,
            seed=seed,
            drop_last=True,
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    loader = DataLoader[Dataset | IterableDataset](
        torch_dataset,  # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and cfg.shuffle_each_epoch and not cfg.streaming),
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


def _collate_fn_for(data_cfg: LMDataConfig):
    collate_column = data_cfg.column_name if data_cfg.is_tokenized else "input_ids"

    def collate_token_column(batch: list[dict[str, Tensor]]) -> Tensor:
        return torch.stack([item[collate_column] for item in batch])

    return collate_token_column


def build_lm_train_loader(
    data_cfg: LMDataConfig,
    *,
    batch_size: int,
    dist_state: DistributedState | None,
    seed: int,
) -> DataLoader[Any]:
    """Build the LM train dataloader."""
    batch_size = _rank_batch_size(batch_size, dist_state, label="train_batch_size")
    train_loader, _ = create_lm_data_loader(
        data_cfg,
        split=data_cfg.train_split,
        batch_size=batch_size,
        seed=seed,
        dist_state=dist_state,
        collate_fn=_collate_fn_for(data_cfg),
    )
    return train_loader


def build_lm_eval_loader(
    data_cfg: LMDataConfig,
    *,
    batch_size: int,
    dist_state: DistributedState | None,
    seed: int,
) -> DataLoader[Any]:
    """Build the LM eval dataloader.

    Seed is offset by 1 so the eval split shuffles differently from the train split
    when both are constructed from the same ``run_cfg.pd.seed``.
    """
    batch_size = _rank_batch_size(batch_size, dist_state, label="eval_batch_size")
    eval_loader, _ = create_lm_data_loader(
        data_cfg,
        split=data_cfg.eval_split,
        batch_size=batch_size,
        seed=seed + 1,
        dist_state=dist_state,
        collate_fn=_collate_fn_for(data_cfg),
    )
    return eval_loader
