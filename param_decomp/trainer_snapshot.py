"""``TrainerSnapshot``: an atomic point-in-time view of a trainer.

Lives in its own module so both :mod:`param_decomp.optimize` (where Trainer
produces it) and :mod:`param_decomp.run_sink` (where the Protocol consumes
it) can import it without a cycle.
"""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TrainerSnapshot:
    """An atomic point-in-time view of a trainer.

    ``step`` identifies which training step this snapshot was taken at
    (also recoverable from ``resume["state"]["step"]`` but lifted to the
    top level for sinks that need it for naming).

    ``resume`` is the rank-local cfg+state dict needed to reconstruct the
    trainer (every rank has a populated one).

    ``consumable`` is the full gathered model state dict in the form
    downstream tools consume (e.g. ``SavedLMRun.load_model``). For 1-pool
    every rank's snapshot has this populated (DDP replicates); for sharded
    pools only rank 0 has it after the gather, and other ranks have ``None``.

    The three pieces travel together so a caller can't accidentally pair a
    rank-local state with the wrong gathered model file: they come from a
    single :meth:`Trainer.snapshot` call. Persistence is the caller's
    concern — typically the consumable is saved as one model file on rank 0
    and the resume half is saved as per-rank shards by a resume-aware sink.
    """

    step: int
    resume: dict[str, Any]
    consumable: dict[str, torch.Tensor] | None
