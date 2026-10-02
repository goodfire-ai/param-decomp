"""How an LM target reads its corpus: the row reader it opens and the batch it builds.

The target family owns the choice; training and every offline consumer share it."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import jax
from jax.sharding import Mesh

from param_decomp.core.sharding import local_data_parallel_size
from param_decomp.infra.dataset_store import read_dataset_identity, read_dataset_meta
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.lm.batch_data import (
    HostLMBatch,
    ShardServer,
    TokenShardServer,
    global_lm_batch,
    global_token_batch,
    scan_shards,
)
from param_decomp.lm.batch_schedule import BatchRowSchedule, BatchSchedule
from param_decomp.targets.qwen3 import Qwen3FrozenAttn
from param_decomp.targets.transformer import (
    FrozenAttn,
    GatedMLP,
    PlainMLP,
    TransformerDecomposedModel,
)


class RowReader[TargetIn](Protocol):
    """This process's scheduled rows, as the target's input."""

    @property
    def seq_len(self) -> int:
        """The corpus's row length, as its own metadata records it."""
        ...

    @property
    def per_process(self) -> int: ...

    def device_batch(self, step: int, vocab_size: int) -> TargetIn:
        """This process's rows on its default device."""
        ...

    def global_batch(self, step: int, mesh: Mesh, global_batch: int, vocab_size: int) -> TargetIn:
        """Every process's rows, sharded over the mesh's batch axes."""
        ...


class InputFormat[TargetIn](Protocol):
    """How a target family reads a corpus directory."""

    def reader(
        self,
        directory: Path,
        schedule: BatchRowSchedule,
        process_index: int,
        process_count: int,
    ) -> RowReader[TargetIn]:
        """`schedule`'s rows of the corpus at `directory`; every scheduled shard lives there."""
        ...


@dataclass(frozen=True)
class DocumentRows:
    """Document-aware shards: packed rows with their document identity."""

    server: ShardServer

    @property
    def seq_len(self) -> int:
        return self.server.seq_len

    @property
    def per_process(self) -> int:
        return self.server.per_process

    def device_batch(self, step: int, vocab_size: int) -> LMBatchWithDocuments:
        return self.server.local_batch(step).to_device(vocab_size)

    def global_batch(
        self, step: int, mesh: Mesh, global_batch: int, vocab_size: int
    ) -> LMBatchWithDocuments:
        return global_lm_batch(self.server.local_batch(step), mesh, global_batch, vocab_size)


@dataclass(frozen=True)
class UnsegmentedTokenRows:
    """Token-only shards read as document-aware rows, each one unpadded document."""

    server: TokenShardServer

    def _host_rows(self, step: int) -> HostLMBatch:
        return HostLMBatch.from_unsegmented_sequences(self.server.local_batch(step).token_ids)

    @property
    def seq_len(self) -> int:
        return self.server.seq_len

    @property
    def per_process(self) -> int:
        return self.server.per_process

    def device_batch(self, step: int, vocab_size: int) -> LMBatchWithDocuments:
        return self._host_rows(step).to_device(vocab_size)

    def global_batch(
        self, step: int, mesh: Mesh, global_batch: int, vocab_size: int
    ) -> LMBatchWithDocuments:
        return global_lm_batch(self._host_rows(step), mesh, global_batch, vocab_size)


@dataclass(frozen=True)
class TokenRows:
    """Token-only shards read as token rows."""

    server: TokenShardServer

    @property
    def seq_len(self) -> int:
        return self.server.seq_len

    @property
    def per_process(self) -> int:
        return self.server.per_process

    def device_batch(self, step: int, vocab_size: int) -> LMBatch:
        return self.server.local_batch(step).to_device(vocab_size)

    def global_batch(self, step: int, mesh: Mesh, global_batch: int, vocab_size: int) -> LMBatch:
        return global_token_batch(self.server.local_batch(step), mesh, global_batch, vocab_size)


def _assert_scheduled_from(directory: Path, schedule: BatchRowSchedule) -> None:
    stray = [shard.path for shard in schedule.shards if shard.path.parent != directory]
    assert not stray, f"schedule reads shards outside {directory}: {stray}"


def _token_server(
    directory: Path, schedule: BatchRowSchedule, process_index: int, process_count: int
) -> TokenShardServer:
    _assert_scheduled_from(directory, schedule)
    seq_len = read_dataset_identity(directory).seq_len
    return TokenShardServer(schedule, seq_len, process_index, process_count)


@dataclass(frozen=True)
class DocumentInput:
    """Versioned document-aware corpora."""

    def reader(
        self,
        directory: Path,
        schedule: BatchRowSchedule,
        process_index: int,
        process_count: int,
    ) -> DocumentRows:
        _assert_scheduled_from(directory, schedule)
        seq_len = read_dataset_meta(directory).seq_len
        return DocumentRows(ShardServer(schedule, seq_len, process_index, process_count))


@dataclass(frozen=True)
class UnsegmentedTokenInput:
    """Token-only corpora, each row one unpadded document."""

    def reader(
        self,
        directory: Path,
        schedule: BatchRowSchedule,
        process_index: int,
        process_count: int,
    ) -> UnsegmentedTokenRows:
        return UnsegmentedTokenRows(
            _token_server(directory, schedule, process_index, process_count)
        )


@dataclass(frozen=True)
class TokenInput:
    """Token-only corpora read as token rows."""

    def reader(
        self,
        directory: Path,
        schedule: BatchRowSchedule,
        process_index: int,
        process_count: int,
    ) -> TokenRows:
        return TokenRows(_token_server(directory, schedule, process_index, process_count))


def transformer_input_format(
    model: TransformerDecomposedModel,
) -> DocumentInput | UnsegmentedTokenInput:
    """The corpus rows each shared-transformer architecture trains on.

    Inferred from the model's layer types, which only correlate with the answer: whether a
    target reads document-aware rows is a property of the target variant (the corpus it was
    trained on and whether its attention honours document boundaries), not of its MLP. The
    sound form has each variant declare its input format where the variant is defined, so
    this function and its match disappear."""
    match model.stacked.attn, model.stacked.mlp:
        case FrozenAttn(), PlainMLP():
            return UnsegmentedTokenInput()
        case ((FrozenAttn() | Qwen3FrozenAttn()), GatedMLP()):
            return DocumentInput()


def input_sampler[TargetIn](
    input_format: InputFormat[TargetIn],
    directory: Path,
    batch_size: int,
    seed: int,
    mesh: Mesh,
    vocab_size: int,
) -> Callable[[int], TargetIn]:
    """The training batch for each step, drawn across every process."""
    schedule = BatchSchedule(scan_shards(directory), batch_size, seed)
    reader = input_format.reader(directory, schedule, jax.process_index(), jax.process_count())
    assert reader.per_process % local_data_parallel_size(mesh) == 0
    return lambda step: reader.global_batch(step, mesh, batch_size, vocab_size)
