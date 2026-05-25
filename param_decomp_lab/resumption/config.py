"""YAML schema for resuming a prior PD run.

A resume YAML is a separate top-level config — distinct from the run YAML —
because resumption is a different operation: it doesn't define a run, it
points at one. The schema is small on purpose: ``from_run`` + ``step`` +
narrow ``overrides``.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import PositiveInt

from param_decomp.base_config import BaseConfig


class ResumeOverrides(BaseConfig):
    """Narrow set of fields that may be overridden on resume.

    Resumption is continuous with the parent's step axis, so most config
    fields are inherited verbatim. The fields here are explicitly the ones we
    know are safe to change mid-trajectory.
    """

    extend_to_step: PositiveInt | None = None
    """Extend ``pd_config.steps`` so the resumed run trains further than the
    parent's original target. Must be > ``parent.pd_config.steps``; otherwise
    use the no-override resume to finish the original."""

    def to_pd_config_patch(self) -> dict[str, Any]:
        """Convert to a flat dict patch applied to the saved ``pd_config``
        dict before pydantic re-validates it inside ``Trainer.from_blob``."""
        patch: dict[str, Any] = {}
        if self.extend_to_step is not None:
            patch["steps"] = self.extend_to_step
        return patch


class ResumeConfig(BaseConfig):
    """A resumption YAML: which run to resume, which checkpoint, what to override."""

    from_run: Path
    """Path to the parent run directory (the one with ``run_meta.yaml`` and
    ``resume/step_<N>/`` snapshots)."""

    step: int | Literal["latest"] = "latest"
    """Which resume snapshot to load. ``"latest"`` picks the highest-numbered
    ``resume/step_<N>/`` under ``from_run``."""

    overrides: ResumeOverrides | None = None
    """Optional narrow overrides applied to the saved ``pd_config`` before
    constructing the resumed trainer."""
