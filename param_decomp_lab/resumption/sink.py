"""``ResumableRunSink`` — wraps a base :class:`RunSink` and additionally
writes per-rank resume shards on every checkpoint.

Construct on all ranks (unlike :class:`param_decomp_lab.run_sink.RunSink`,
which is silent-noop off the main rank). The wrapped base sink retains its
own rank gating: on rank 0 it writes the consumable model + logs as before;
on non-main ranks it's a no-op. The shard write happens on every rank.
"""

from pathlib import Path
from typing import Any

from param_decomp.run_sink import RunSink
from param_decomp.trainer_snapshot import TrainerSnapshot
from param_decomp_lab.resumption.shards import save_shard


class ResumableRunSink:
    """Adds per-rank shard writes to any base :class:`RunSink`.

    Attributes mirror the :class:`RunSink` Protocol; the wrapper composes
    rather than subclasses so any sink implementation can be promoted to
    resumable.
    """

    def __init__(self, base: RunSink, *, run_dir: Path, rank: int) -> None:
        self._base = base
        self._run_dir = run_dir
        self._rank = rank

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self._base.log(metrics, step)

    def console(self, *lines: str) -> None:
        self._base.console(*lines)

    def checkpoint(self, snapshot: TrainerSnapshot) -> None:
        """Write this rank's resume shard, then delegate to the base sink
        (which writes the consumable model on rank 0 and is a no-op elsewhere)."""
        save_shard(snapshot.resume, self._run_dir, snapshot.step, self._rank)
        self._base.checkpoint(snapshot)

    def finish(self) -> None:
        finish = getattr(self._base, "finish", None)
        if callable(finish):
            finish()
