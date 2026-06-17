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
    group: str | None = None
    """Wandb UI group (`pd-jax-lm --group`); None = ungrouped."""
    tags: list[str] = Field(default_factory=list)
    """Wandb tags (`pd-jax-lm --tags a,b,c`, comma-split); empty = untagged."""


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
    """Fine-tune lineage: a fresh run initialized from a PARENT decomposition. Lives on
    `ExperimentConfig`.

    A fine-tune run gets its own `run_id` / `config.yaml` / `ckpts/`; this records the
    parent it forked from. The JAX trainer (`run.py::train`, SPEC S33) loads the parent
    checkpoint's V/U + ci_fn onto a fresh reference state and trains a clean schedule from
    step 0 (fresh optimizer / sources) under the new config — only when the run's own
    `ckpts/` is empty (a subsequent SLURM requeue resumes from the run's own dir, ignoring
    provenance). The structure (sites / C / ci-fn arch) must match the parent; only
    LR / coeffs / eps / seq / batch / steps may change. Provenance flows into
    `config.yaml` and `wandb.config` so the lineage is visible in the wandb UI. A run with
    `resume_provenance is None` is a fresh-from-init run.
    """

    parent_run_dir: Path
    """Path to the parent run's directory (the dir that contains `ckpts/<parent_step>/`)."""

    parent_step: int
    """The parent's orbax `ckpts/<step>/` checkpoint step to initialize V/U + ci_fn from."""


class ExperimentConfig[T: BaseConfig, D: BaseConfig](BaseConfig):
    """Full YAML schema for an in-repo experiment.

    Subclass with concrete `target` / `data` types per experiment:

        class LMExperimentConfig(ExperimentConfig[LMTargetConfig, LMDataConfig]):
            pass

    Omit the `eval:` block to skip eval entirely; omit `wandb:` to skip wandb (the run
    still writes `config.yaml` + checkpoints locally).

    `run_id` / `out_dir` are minted by `pd-jax-lm` at submit time (both `None` in a
    hand-authored config); the stamped workspace copy carries them, and `jsp-train`
    resumes by byte-comparing that pinned copy.
    """

    run_name: str
    """Human-readable display name (the wandb run NAME)."""
    run_id: str | None = None
    """Canonical `p-<8hex>` id (wandb run ID + run-dir name). `None` in a hand-authored
    config; minted + stamped by `pd-jax-lm` at submit time."""
    out_dir: Path | None = None
    """Run-output root (the run dir is `out_dir / run_id`). `None` lets `pd-jax-lm` mint
    `PARAM_DECOMP_OUT_DIR/runs`; set it to override (the llama8b configs use `jax_runs`)."""

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
