# pyright: reportAttributeAccessIssue=false, reportCallIssue=false
"""Tokenize ONE physical shard of pile-uncopyrighted.

Designed for SLURM array launches: one task per `train/{NN}.jsonl.zst` file.
Each task downloads + tokenizes + packs into seq_len windows + writes one
HF Dataset to `<shards_dir>/shard-{NN}/`. A separate merge step concatenates
all shards into the final usable dataset.

Speed levers:
  - hf_transfer (set HF_HUB_ENABLE_HF_TRANSFER=1 in sbatch) for fast download
  - Fast tokenizer's parallel encode_batch (uses all CPU cores in the
    allocated --cpus-per-task)

Usage:
    python -m param_decomp_lab.scripts.pile.pretokenize_pile_shard \\
        --shard_idx 7 \\
        --tokenizer Qwen/Qwen3-0.6B-Base \\
        --seq_len 2048 \\
        --shards_dir /mnt/polished-lake/home/oli/datasets/pile-qwen-tok-2048-shards
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from datasets import (
    Dataset,
    Features,
    IterableDataset,
    Sequence,
    Value,
    load_dataset,
)
from transformers import AutoTokenizer, PreTrainedTokenizer

PILE_REPO = "monology/pile-uncopyrighted"
N_PILE_SHARDS = 30  # train/00.jsonl.zst .. train/29.jsonl.zst


def _generate_packed_sequences(
    dataset: IterableDataset,
    tokenizer: PreTrainedTokenizer,
    seq_len: int,
    encode_batch: int,
):
    """Stream docs, batch-tokenize with fast tokenizer, pack into seq_len windows."""
    assert tokenizer.eos_token_id is not None, "tokenizer must have an EOS token"
    assert getattr(tokenizer, "is_fast", False), (
        "Need a fast (Rust-backed) tokenizer for parallel encode_batch. "
        f"Got is_fast={getattr(tokenizer, 'is_fast', None)} for {tokenizer.__class__.__name__}."
    )
    eos_id = tokenizer.eos_token_id
    buf: list[int] = []
    chunk_docs: list[str] = []

    for ex in dataset:
        chunk_docs.append(ex["text"])
        if len(chunk_docs) < encode_batch:
            continue
        encoded = tokenizer.batch_encode_plus(
            chunk_docs,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        chunk_docs = []
        for doc_ids in encoded:
            buf.extend(doc_ids)
            buf.append(eos_id)
        while len(buf) >= seq_len:
            yield {"input_ids": np.asarray(buf[:seq_len], dtype=np.int32)}
            buf = buf[seq_len:]

    # Flush remaining docs.
    if chunk_docs:
        encoded = tokenizer.batch_encode_plus(
            chunk_docs,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        for doc_ids in encoded:
            buf.extend(doc_ids)
            buf.append(eos_id)
        while len(buf) >= seq_len:
            yield {"input_ids": np.asarray(buf[:seq_len], dtype=np.int32)}
            buf = buf[seq_len:]
    # Drop the trailing partial — not enough for a full window.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard_idx",
        type=int,
        required=True,
        help=f"Pile shard index in [0, {N_PILE_SHARDS}). Reads train/{{:02d}}.jsonl.zst.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-0.6B-Base",
        help="HF tokenizer name. Must be fast (Rust-backed).",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Tokens per packed sequence.",
    )
    parser.add_argument(
        "--shards_dir",
        type=Path,
        required=True,
        help="Output root. Writes to {shards_dir}/shard-{shard_idx:02d}/.",
    )
    parser.add_argument(
        "--encode_batch",
        type=int,
        default=1024,
        help="Docs per encode_batch() call (Rust parallelism granularity).",
    )
    parser.add_argument(
        "--writer_batch_size",
        type=int,
        default=2048,
        help="HF Dataset Arrow writer batch.",
    )
    args = parser.parse_args()

    assert 0 <= args.shard_idx < N_PILE_SHARDS, (
        f"shard_idx must be in [0, {N_PILE_SHARDS}); got {args.shard_idx}"
    )
    out_dir = args.shards_dir / f"shard-{args.shard_idx:02d}"
    if (out_dir / "dataset_info.json").exists():
        print(f"[shard {args.shard_idx}] already done at {out_dir}, skipping.", flush=True)
        return
    # If the dir exists but isn't a complete dataset, nuke it for a clean retry.
    if out_dir.exists():
        import shutil

        print(
            f"[shard {args.shard_idx}] partial dir at {out_dir}, removing for fresh start.",
            flush=True,
        )
        shutil.rmtree(out_dir)

    print(f"Loading tokenizer: {args.tokenizer}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    file_path = f"train/{args.shard_idx:02d}.jsonl.zst"
    print(f"Loading streaming pile file: {file_path}", flush=True)
    dataset: Any = load_dataset(
        PILE_REPO,
        data_files=file_path,
        split="train",
        streaming=True,
    )
    assert isinstance(dataset, IterableDataset)

    features = Features({"input_ids": Sequence(Value("int32"), length=args.seq_len)})
    args.shards_dir.mkdir(parents=True, exist_ok=True)

    def gen() -> Any:
        emitted = 0
        for i, ex in enumerate(
            _generate_packed_sequences(dataset, tokenizer, args.seq_len, args.encode_batch)
        ):
            emitted = i + 1
            if emitted % 10000 == 0:
                print(f"  [shard {args.shard_idx}] emitted {emitted:,} sequences", flush=True)
            yield ex
        print(f"  [shard {args.shard_idx}] DONE emitting {emitted:,} sequences", flush=True)

    cache_root = args.shards_dir / f".cache-shard-{args.shard_idx:02d}"
    ds = Dataset.from_generator(
        gen,
        features=features,
        writer_batch_size=args.writer_batch_size,
        cache_dir=str(cache_root),
    )
    assert isinstance(ds, Dataset)
    n_tokens = len(ds) * args.seq_len
    print(
        f"[shard {args.shard_idx}] tokenized: {len(ds):,} sequences "
        f"(~{n_tokens / 1e9:.2f}B tokens). Saving to {out_dir}",
        flush=True,
    )
    ds.save_to_disk(str(out_dir))
    # Cache dir is large (intermediate Arrow); remove it.
    if cache_root.exists():
        import shutil

        shutil.rmtree(cache_root)
    print(f"[shard {args.shard_idx}] done.", flush=True)


if __name__ == "__main__":
    main()
