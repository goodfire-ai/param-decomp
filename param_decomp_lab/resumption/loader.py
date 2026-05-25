"""Load a per-rank shard from a parent run into a ``TrainerSnapshot``.

The lab's blessed atomic-load path: experiment-specific resume entrypoints
(e.g. ``lm/run.py::_resume_main``) construct the right concrete trainer via
``Trainer.from_snapshot(snapshot, ...)`` with the snapshot returned here.
"""

from typing import Any

from param_decomp.trainer_snapshot import TrainerSnapshot
from param_decomp_lab.distributed import get_device
from param_decomp_lab.resumption.config import ResumeConfig
from param_decomp_lab.resumption.shards import load_shard, resolve_step


def read_resume_snapshot(
    resume_cfg: ResumeConfig,
    *,
    rank: int,
    current_device: str | None = None,
) -> TrainerSnapshot:
    """Read this rank's per-rank shard from the parent run.

    The saved ``runtime_config.device`` is replaced with ``current_device``
    (defaulting to :func:`get_device`) — runtime environment is not part of
    the persisted training state, and a saved cluster device string typically
    won't match the resume environment's device.

    Returns a :class:`TrainerSnapshot` whose ``consumable`` half is ``None``;
    the resume-only blob suffices for ``Trainer.from_snapshot(...)``.
    """
    device = current_device if current_device is not None else get_device()
    parent_run_dir = resume_cfg.from_run
    resolved_step = resolve_step(parent_run_dir, resume_cfg.step)
    resume_dict: dict[str, Any] = load_shard(parent_run_dir, resolved_step, rank)
    resume_dict["runtime_config"]["device"] = device
    return TrainerSnapshot(
        step=resume_dict["state"]["step"],
        resume=resume_dict,
        consumable=None,
    )
