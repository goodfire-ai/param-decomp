"""Concatenate per-shard tokenized pile datasets into one HF Dataset.

Designed to run AFTER `pretokenize_pile_shard.py` has produced all 30
`{shards_dir}/shard-NN/` directories. Loads each shard via `load_from_disk`,
concatenates them, writes a single combined HF Dataset, and optionally
pushes to the Hub.

Usage:
    python -m param_decomp_lab.scripts.pile.merge_pile_shards \\
        --shards_dir /mnt/polished-lake/home/oli/datasets/pile-qwen-tok-2048-shards \\
        --output_dir /mnt/polished-lake/home/oli/datasets/pile-qwen-tok-2048 \\
        --push_to_hub goodfire/pile-qwen-tok-2048    # optional
"""

import argparse
import shutil
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk

N_PILE_SHARDS = 30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards_dir",
        type=Path,
        required=True,
        help="Root containing shard-{NN}/ subdirs produced by pretokenize_pile_shard.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Where to write the merged Dataset. Must not exist.",
    )
    parser.add_argument(
        "--n_shards",
        type=int,
        default=N_PILE_SHARDS,
        help="Expected number of physical shards.",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="Optional HF Hub repo id (e.g. goodfire/pile-qwen-tok-2048). Requires HF auth.",
    )
    parser.add_argument(
        "--hub_private",
        action="store_true",
        help="If --push_to_hub is set, create as a private repo.",
    )
    parser.add_argument(
        "--delete_shards_after",
        action="store_true",
        help="Remove the per-shard dirs after a successful merge.",
    )
    args = parser.parse_args()

    assert not args.output_dir.exists(), f"Output already exists: {args.output_dir}"
    shard_dirs = sorted(args.shards_dir.glob("shard-*"))
    assert len(shard_dirs) == args.n_shards, (
        f"Expected {args.n_shards} shards under {args.shards_dir}, found {len(shard_dirs)}: "
        f"{[d.name for d in shard_dirs]}"
    )

    print(f"Loading {len(shard_dirs)} shards from {args.shards_dir}", flush=True)
    shards: list[Dataset] = []
    for d in shard_dirs:
        ds = load_from_disk(str(d))
        assert isinstance(ds, Dataset)
        shards.append(ds)
        print(f"  {d.name}: {len(ds):,} sequences", flush=True)

    combined = concatenate_datasets(shards)
    total = len(combined)
    seq_len = len(combined[0]["input_ids"])
    print(
        f"Concatenated: {total:,} sequences × {seq_len} tokens "
        f"(~{total * seq_len / 1e9:.2f}B tokens). Writing to {args.output_dir}",
        flush=True,
    )
    combined.save_to_disk(str(args.output_dir))

    if args.push_to_hub is not None:
        print(f"Pushing to hub: {args.push_to_hub} (private={args.hub_private})", flush=True)
        combined.push_to_hub(args.push_to_hub, private=args.hub_private)
        print("Push complete.", flush=True)

    if args.delete_shards_after:
        for d in shard_dirs:
            print(f"  deleting {d}", flush=True)
            shutil.rmtree(d)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
