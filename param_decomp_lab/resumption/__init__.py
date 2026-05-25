"""Resumption support: continue a prior PD run from its latest (or a specific) checkpoint.

A resumption is a separate top-level concept from a fresh run, expressed via its
own ``ResumeConfig`` YAML schema and dispatched from the ``pd-lm --resume <path>``
CLI flag. Resumption is *continuous*: the resumed run extends the parent's step
axis, inheriting its scaffolding cfg from ``run_meta.yaml`` and its training
state from per-rank shards under ``<parent_run_dir>/resume/step_<N>/``.

See ``ResumeConfig`` for the schema and :class:`ResumableRunSink` for the
save-side wrapper that writes per-rank shards alongside the consumable model.
"""

from param_decomp_lab.resumption.config import ResumeConfig, ResumeOverrides
from param_decomp_lab.resumption.loader import read_resume_snapshot
from param_decomp_lab.resumption.provenance import (
    RESUME_PROVENANCE_FILENAME,
    ResumeProvenance,
    read_provenance,
    write_provenance,
)
from param_decomp_lab.resumption.shards import (
    list_resume_steps,
    load_shard,
    resolve_step,
    save_shard,
    shard_path,
)
from param_decomp_lab.resumption.sink import ResumableRunSink

__all__ = [
    "RESUME_PROVENANCE_FILENAME",
    "ResumableRunSink",
    "ResumeConfig",
    "ResumeOverrides",
    "ResumeProvenance",
    "list_resume_steps",
    "load_shard",
    "read_provenance",
    "read_resume_snapshot",
    "resolve_step",
    "save_shard",
    "shard_path",
    "write_provenance",
]
