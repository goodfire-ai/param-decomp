"""Resume provenance: a small YAML sibling of ``experiment_config.yaml`` recording
which run a resumed run was forked from.

Resumed runs get their own ``run_id`` and own ``experiment_config.yaml`` — provenance
is what makes them traceable back to the parent. A future reader can inspect
``resume_provenance.yaml`` to find the parent run dir + the step it was
resumed from.

A run without this file is a fresh run.
"""

from pathlib import Path

from param_decomp.base_config import BaseConfig

RESUME_PROVENANCE_FILENAME = "resume_provenance.yaml"


class ResumeProvenance(BaseConfig):
    """Sibling of ``experiment_config.yaml`` recording where this resumed run came from."""

    parent_run_dir: Path
    """Path to the parent run's directory."""

    parent_step: int
    """The step at which we resumed (i.e. the step number in the parent's
    ``resume/step_<N>/`` snapshot we loaded from)."""


def write_provenance(out_dir: Path, provenance: ResumeProvenance) -> None:
    """Persist ``provenance`` to ``{out_dir}/resume_provenance.yaml``."""
    provenance.to_file(out_dir / RESUME_PROVENANCE_FILENAME)


def read_provenance(run_dir: Path) -> ResumeProvenance | None:
    """Read provenance from ``run_dir``. Returns ``None`` if the run is fresh
    (no ``resume_provenance.yaml`` file present)."""
    path = run_dir / RESUME_PROVENANCE_FILENAME
    if not path.is_file():
        return None
    return ResumeProvenance.from_file(path)
