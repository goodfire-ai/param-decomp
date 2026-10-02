"""Pre-stage a HF text dataset with the Llama 3 base-text policy to local int32 parquet shards.

One-shot offline tool so training never streams or tokenizes at run time: the trainer's
deterministic schedule reads a fixed local shard set, startup keeps third-party services
out of the N-rank collective's critical path, and no rank pays tokenization CPU.

Processes source parquet files one at a time (download -> tokenize with `num_proc` ->
write one int32 output shard -> delete the raw file) so peak disk stays ~one raw file plus
the growing output, and is resumable: an output shard that already exists is skipped, so an
interrupted invocation can continue where it left off.

Output: `<out_dir>/shard_<NNNNN>.parquet`, each row paired `input_ids` and `document_ids`
lists of length `seq_len` (int32), plus `meta.json` (`infra.dataset_store.DatasetMeta`: version,
seq_len, tokenizer, preprocessing_name —
the dir is self-describing). Publish the dir into the dataset store under a new
immutable name and reference it as `data: {kind: name, name: <name>}`,
or point `data: {kind: dir, dir: <out_dir>}` at it directly.

Run: `python -m param_decomp.experiments.lm.llama3.prestage_tokenized --out_dir <abs> [...]`
"""

from pathlib import Path

import fire
from datasets import Dataset, Sequence, Value, load_dataset
from huggingface_hub import HfApi, hf_hub_download

from param_decomp.core.log import logger
from param_decomp.experiments.lm.llama3.data import (
    LLAMA3_PREPROCESSING_NAME,
    load_llama3_tokenizer,
    preprocess_llama3,
)
from param_decomp.infra.dataset_store import (
    DATASET_META_FILENAME,
    DatasetMeta,
    write_dataset_meta,
)


def _shard_token_count(path: Path, seq_len: int) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    assert set(schema.names) == {"input_ids", "document_ids"}, (
        f"{path}: shards require input_ids and document_ids; tokenize older datasets again"
    )
    for field in schema:
        assert field.type == pa.list_(pa.int32(), seq_len), (
            f"{path}: {field.name} must be a fixed-size int32 list of length {seq_len}"
        )
    return parquet.metadata.num_rows * seq_len


def _prepare_output(out: Path, meta: DatasetMeta) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # Another worker can publish metadata between the directory listing and this check.
    assert not any(out.iterdir()) or (out / DATASET_META_FILENAME).exists(), (
        f"{out}: refusing to resume a directory without document-aware metadata; "
        "tokenize into a new directory"
    )
    write_dataset_meta(out, meta)
    for shard in sorted(out.glob("shard_*.parquet")):
        _shard_token_count(shard, meta.seq_len)


def prestage(
    *,
    out_dir: str,
    num_files: int,
    skip_files: int,
    task_id: int,
    num_tasks: int,
    dataset_repo: str,
    subdir: str,
    revision: str,
    tokenizer_name: str,
    seq_len: int,
    column_name: str,
    num_proc: int,
) -> None:
    """Tokenize `num_files` source parquet files, starting at `skip_files`, into int32
    shards. A disjoint eval split is `skip_files` set past the training split's file
    range, into the same `dataset_repo`.

    Fan-out: task `task_id` of `num_tasks` processes the strided slice
    `range(task_id, num_files, num_tasks)`; shards are named by GLOBAL file index so
    tasks never collide. For scale: 366 files of fineweb `sample/350BT` ≈ 256B tokens ≈ 512GB on disk
    (int32 with ~2x parquet compression).
    Interruption-safe (scavenge): writes are atomic (`.tmp` + rename) and resume skips
    any already-complete shard, so a preempted+requeued task continues cleanly.
    """
    assert num_files > 0, "num_files must be positive"
    assert num_tasks > 0, "num_tasks must be positive"
    assert 0 <= task_id < num_tasks, "task_id must be within [0, num_tasks)"
    assert skip_files >= 0, "skip_files must be nonnegative"
    assert seq_len > 0, "seq_len must be positive"
    assert num_proc > 0, "num_proc must be positive"
    tokenizer = load_llama3_tokenizer(tokenizer_name)
    out = Path(out_dir)
    _prepare_output(
        out,
        DatasetMeta(
            format_version=2,
            seq_len=seq_len,
            tokenizer_name=tokenizer_name,
            preprocessing_name=LLAMA3_PREPROCESSING_NAME,
        ),
    )

    api = HfApi()
    files = sorted(
        f
        for f in api.list_repo_files(dataset_repo, repo_type="dataset", revision=revision)
        if f.startswith(f"{subdir}/") and f.endswith(".parquet")
    )
    assert files, f"no parquet files under {subdir} in {dataset_repo}@{revision}"
    assert skip_files < len(files), f"skip_files={skip_files} >= {len(files)} available"
    num_files = min(num_files, len(files) - skip_files)
    my_indices = list(range(skip_files + task_id, skip_files + num_files, num_tasks))
    logger.info(
        f"task {task_id}/{num_tasks}: {len(files)} files available, processing "
        f"{len(my_indices)} of files [{skip_files}, {skip_files + num_files}) "
        f"(indices {my_indices[:3]}...)"
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
        tokenized = preprocess_llama3(
            raw,
            tokenizer,
            column_name=column_name,
            max_length=seq_len,
            num_proc=num_proc,
        )
        tokenized = tokenized.cast_column("input_ids", Sequence(Value("int32"), length=seq_len))  # pyright: ignore[reportArgumentType]
        tokenized = tokenized.cast_column("document_ids", Sequence(Value("int32"), length=seq_len))  # pyright: ignore[reportArgumentType]
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
