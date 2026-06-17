"""Pre-stage + pre-tokenize a portion of a HF text dataset to local int32 parquet shards.

One-shot offline tool so training never streams or tokenizes from HF at run time. Streaming
the dataset at launch makes every rank hit HF Hub for parquet shards; at 80 ranks that
thunderherd read-times-out / RemoteDisconnects, stranding a rank before the `build_two_world`
collective and hanging startup. Pre-tokenizing to local Arrow/parquet removes the runtime HF
dependency AND the per-rank tokenization cost.

Processes source parquet files one at a time (download -> tokenize with `num_proc` ->
write one int32 output shard -> delete the raw file) so peak disk stays ~one raw file plus
the growing output, and is resumable: an output shard that already exists is skipped, so a
requeued job continues where it left off.

Output: `<out_dir>/shard_<NNNNN>.parquet`, each row an `input_ids` list of length `seq_len`
(int32). Point `LMDataConfig` at it with `dataset_name: parquet`,
`data_files: <out_dir>/*.parquet`, `column_name: input_ids`, `is_tokenized: true`,
`streaming: false`.

Run: `python -m param_decomp_lab.experiments.lm.prestage_tokenized --out_dir <abs> [...]`
"""

from pathlib import Path

import fire
from datasets import Dataset, Sequence, Value, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from param_decomp.log import logger
from param_decomp_lab.experiments.lm.data import tokenize_and_concatenate


def _shard_token_count(path: Path, seq_len: int) -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows * seq_len


def prestage(
    *,
    out_dir: str,
    num_files: int = 366,
    task_id: int = 0,
    num_tasks: int = 1,
    dataset_repo: str = "HuggingFaceFW/fineweb",
    subdir: str = "sample/350BT",
    revision: str = "9bb295ddab0e05d785b879661af7260fed5140fc",
    tokenizer_name: str = "meta-llama/Llama-3.1-8B",
    seq_len: int = 2048,
    column_name: str = "text",
    num_proc: int = 96,
) -> None:
    """Tokenize the first `num_files` source parquet files into int32 shards.

    Fan-out: task `task_id` of `num_tasks` processes the strided slice
    `range(task_id, num_files, num_tasks)`; shards are named by GLOBAL file index so
    tasks never collide. ~366 files of `sample/350BT` ≈ 256B tokens ≈ 512GB on disk
    (int32 with ~2x parquet compression).
    Interruption-safe (scavenge): writes are atomic (`.tmp` + rename) and resume skips
    any already-complete shard, so a preempted+requeued task continues cleanly.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)

    api = HfApi()
    files = sorted(
        f
        for f in api.list_repo_files(dataset_repo, repo_type="dataset", revision=revision)
        if f.startswith(f"{subdir}/") and f.endswith(".parquet")
    )
    assert files, f"no parquet files under {subdir} in {dataset_repo}@{revision}"
    num_files = min(num_files, len(files))
    my_indices = list(range(task_id, num_files, num_tasks))
    logger.info(
        f"task {task_id}/{num_tasks}: {len(files)} files available, processing "
        f"{len(my_indices)} of the first {num_files} (indices {my_indices[:3]}...)"
    )

    for i in my_indices:
        shard = out / f"shard_{i:05d}.parquet"
        if shard.exists():
            logger.info(f"shard_{i:05d} exists; skip")
            continue
        tmp = out / f"shard_{i:05d}.parquet.tmp"  # atomic: write tmp, rename on success

        local = Path(
            hf_hub_download(dataset_repo, files[i], repo_type="dataset", revision=revision)
        )
        raw = load_dataset("parquet", data_files=str(local), split="train")
        assert isinstance(raw, Dataset)
        tokenized = tokenize_and_concatenate(
            raw, tokenizer, column_name=column_name, max_length=seq_len, num_proc=num_proc
        )
        assert isinstance(tokenized, Dataset)
        tokenized = tokenized.cast_column("input_ids", Sequence(Value("int32")))  # pyright: ignore[reportArgumentType]
        tokenized.to_parquet(str(tmp))
        tmp.rename(shard)

        n = len(tokenized) * seq_len
        logger.info(f"[{i}] {files[i]}: {len(tokenized)} seqs / {n / 1e9:.2f}B tok -> {shard.name}")
        local.unlink(missing_ok=True)  # bound peak disk to ~one raw file + the output

    staged = sum(_shard_token_count(p, seq_len) for p in sorted(out.glob("shard_*.parquet")))
    logger.info(
        f"task {task_id} DONE; total staged across all tasks so far: {staged / 1e9:.1f}B tokens"
    )


def cli() -> None:
    fire.Fire(prestage)


if __name__ == "__main__":
    cli()
