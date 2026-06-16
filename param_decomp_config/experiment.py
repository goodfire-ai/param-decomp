"""Shared config schema for in-repo experiment YAMLs.

Each experiment subclasses `ExperimentConfig` to fix the concrete `target` / `data`
types.
"""

from pathlib import Path
from typing import Self

from pydantic import Field, PositiveInt, model_validator

from param_decomp_config.base import BaseConfig
from param_decomp_config.eval_metrics import AnyEvalMetricConfig
from param_decomp_config.pd import Cadence, PDConfig, RuntimeConfig


class WandbConfig(BaseConfig):
    """Wandb logging settings. Presence on `ExperimentConfig` opts in; omit to skip wandb."""

    project: str
    entity: str | None = None


class EvalConfig(BaseConfig):
    """Eval-pass settings consumed by `EvalLoop`. `slow_every` must be a multiple of `every`."""

    batch_size: PositiveInt
    n_steps: PositiveInt
    every: PositiveInt
    slow_every: PositiveInt
    slow_on_first_step: bool = True
    metrics: list[AnyEvalMetricConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_slow_every_multiple_of_every(self) -> Self:
        assert self.slow_every % self.every == 0, (
            f"slow_every ({self.slow_every}) must be a multiple of every ({self.every})"
        )
        return self


class ResumeProvenance(BaseConfig):
    """Records where a resumed run came from. Lives on `ExperimentConfig`.

    Resumed runs get their own `run_id` and own `experiment_config.yaml`. Provenance is
    what makes them traceable back to the parent: the resume composition root sets
    `ExperimentConfig.resume_provenance` on the effective config it hands to
    `init_pd_run`, so the lineage is dumped into `experiment_config.yaml` and surfaced
    in `wandb.config` (and thus the wandb UI). A run with `resume_provenance is None`
    is a fresh run.
    """

    parent_run_dir: Path
    """Path to the parent run's directory."""

    parent_step: int
    """The step at which we resumed (i.e. the step number of the parent's orbax
    `ckpts/<step>/` checkpoint we loaded from)."""


class ExperimentConfig[T: BaseConfig, D: BaseConfig](BaseConfig):
    """Full YAML schema for an in-repo experiment.

    Subclass with concrete `target` / `data` types per experiment:

        class LMExperimentConfig(ExperimentConfig[LMTargetConfig, LMDataConfig]):
            pass

    Omit the `eval:` block to skip eval entirely; omit `wandb:` to skip wandb (the run
    still writes `experiment_config.yaml` + checkpoints locally).
    """

    pd: PDConfig
    runtime: RuntimeConfig
    cadence: Cadence
    target: T
    data: D
    eval: EvalConfig | None = None
    wandb: WandbConfig | None = None
    resume_provenance: ResumeProvenance | None = None
    """Set on resumed runs (parent run dir + step); `None` for fresh runs. Lives on the
    config so it flows into `experiment_config.yaml` and `wandb.config` via `init_pd_run`,
    making a resumed run's lineage visible in the wandb UI."""
