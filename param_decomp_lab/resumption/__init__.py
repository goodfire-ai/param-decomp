"""Resumption: continue a prior PD run from one of its `training_<step>.pth` checkpoints.

A resumption is a separate top-level concept from a fresh run, expressed via its own
`ResumeConfig` YAML schema and dispatched from the `pd-lm --resume <path>` CLI flag.
Resumption is *continuous*: the resumed run extends the parent's step axis, inheriting
its config from `experiment_config.yaml` and its training state from `training_<step>.pth`.

The training state is canonical and topology-independent — a single file written by
rank 0 carries everything needed to reconstruct the trainer for any compatible
topology.
"""

from param_decomp_lab.resumption.config import (
    ResumeConfig,
    read_training_snapshot,
    resolve_step,
)
from param_decomp_lab.resumption.provenance import (
    RESUME_PROVENANCE_FILENAME,
    ResumeProvenance,
    read_provenance,
    write_provenance,
)

__all__ = [
    "RESUME_PROVENANCE_FILENAME",
    "ResumeConfig",
    "ResumeProvenance",
    "read_provenance",
    "read_training_snapshot",
    "resolve_step",
    "write_provenance",
]
