# pyright: reportAttributeAccessIssue=false, reportCallIssue=false
"""Pre-tokenize pile-uncopyrighted with a HuggingFace tokenizer and save to disk.

We pre-tokenize because the LM driver's streaming tokenize path
(`_tokenize_and_concatenate` over an IterableDataset) trips HF datasets'
`assert features is not None, "Dataset features must be known..."` when
`remove_columns=` is used. The simplest sidestep is to materialize a
pre-tokenized dataset on disk; downstream code then sets ``is_tokenized: true``
and points ``dataset_name`` at the resulting directory.

Output schema (matches the existing pre-tokenized variant
``danbraunai/pile-uncopyrighted-tok-shuffled``):

  Features({"input_ids": Sequence(Value("int32"), length=seq_len)})

Streams the raw text from ``monology/pile-uncopyrighted``, joins documents
with the tokenizer's EOS, then packs into fixed-length sequences. The
joining-then-packing strategy matches ``_tokenize_and_concatenate`` so the
distributional content lines up with what Lucius's runs trained on
(modulo tokenizer choice).

Usage:
    python -m param_decomp_lab.scripts.pile.pretokenize_pile \\
        --tokenizer Qwen/Qwen3-0.6B-Base \\
        --output_dir /mnt/polished-lake/home/oli/datasets/pile-uncopyrighted-qwen-tok-1024 \\
        --seq_len 1024 \\
        --n_tokens 200_000_000
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

PILE_DATASET = "monology/pile-uncopyrighted"


def _generate_packed_sequences(
    dataset: IterableDataset,
    tokenizer: PreTrainedTokenizer,
    seq_len: int,
    n_tokens: int,
    batch_size: int = 1024,
):
    """Stream pile docs, tokenize in batches, pack into seq_len windows.

    Uses the Rust-backed ``tokenizer.encode_batch`` (when available) so a
    single Python process saturates multiple CPU cores for tokenization —
    ~3-5× faster than single-doc ``encode`` calls.
    """
    assert tokenizer.eos_token is not None, "tokenizer must have an EOS token"
    eos_id = tokenizer.eos_token_id
    use_fast = getattr(tokenizer, "is_fast", False)
    buf: list[int] = []
    produced = 0
    chunk_docs: list[str] = []

    def flush_chunk() -> None:
        nonlocal chunk_docs
        if use_fast:
            # Fast tokenizers (Rust) parallelize encode_batch across cores.
            encoded = tokenizer(
                chunk_docs,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            for doc_ids in encoded:
                buf.extend(doc_ids)
                buf.append(eos_id)
        else:
            text = tokenizer.eos_token.join(chunk_docs)
            buf.extend(tokenizer.encode(text, add_special_tokens=False))
        chunk_docs = []

    for ex in dataset:
        chunk_docs.append(ex["text"])
        if len(chunk_docs) < batch_size:
            continue
        flush_chunk()
        while len(buf) >= seq_len:
            yield {"input_ids": np.asarray(buf[:seq_len], dtype=np.int32)}
            produced += seq_len
            buf = buf[seq_len:]
            if produced >= n_tokens:
                return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-0.6B-Base",
        help="HF tokenizer name.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Where to write the dataset. A directory; will hold Arrow shards + dataset_info.json.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=1024,
        help="Tokens per sequence.",
    )
    parser.add_argument(
        "--n_tokens",
        type=int,
        default=200_000_000,
        help="Approximate total number of tokens to produce.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Pile split to read from.",
    )
    parser.add_argument(
        "--writer_batch_size",
        type=int,
        default=1024,
        help="HF Dataset writer batch size (controls Arrow shard sizing).",
    )
    args = parser.parse_args()

    assert not args.output_dir.exists(), (
        f"Output dir already exists: {args.output_dir}. Refusing to overwrite."
    )

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Loading streaming dataset: {PILE_DATASET} (split={args.split})")
    dataset: Any = load_dataset(PILE_DATASET, split=args.split, streaming=True)
    assert isinstance(dataset, IterableDataset)

    features = Features({"input_ids": Sequence(Value("int32"), length=args.seq_len)})
    n_expected = args.n_tokens // args.seq_len
    print(
        f"Producing ~{n_expected:,} sequences of length {args.seq_len} (~{args.n_tokens:,} tokens)"
    )

    def gen() -> Any:
        for i, ex in enumerate(
            _generate_packed_sequences(dataset, tokenizer, args.seq_len, args.n_tokens)
        ):
            if i % 5000 == 0 and i > 0:
                print(f"  generated {i:,} / {n_expected:,} sequences", flush=True)
            yield ex

    ds = Dataset.from_generator(
        gen,
        features=features,
        writer_batch_size=args.writer_batch_size,
        cache_dir=str(args.output_dir.parent / f".{args.output_dir.name}_cache"),
    )
    assert isinstance(ds, Dataset)
    print(f"Tokenized dataset: {len(ds):,} sequences. Saving to {args.output_dir}")
    ds.save_to_disk(str(args.output_dir))
    print("Done.")


if __name__ == "__main__":
    main()
