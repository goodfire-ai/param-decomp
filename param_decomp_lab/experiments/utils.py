"""Shared config schema for in-repo experiment YAMLs.

Each experiment subclasses `ExperimentConfig` to fix the concrete `target` / `data` types
and parses its YAML with `<Experiment>Config.from_file(path)`. The resolved config is
persisted as ``run_meta.yaml`` via `BaseConfig.to_file` and rebuilt on reload by the
matching per-experiment ``SavedXRun`` class.
"""

from pydantic import Field, PositiveInt

from param_decomp.base_config import BaseConfig
from param_decomp.configs import Cadence, PDConfig, RuntimeConfig
from param_decomp_lab.eval_metrics import AnyEvalMetricConfig

RUN_META_FILENAME = "run_meta.yaml"


class EvalConfig(BaseConfig):
    """Eval-pass settings consumed by `EvalLoop`.

    Attributes:
        batch_size: Loader batch size for the eval split.
        n_steps: Number of batches to consume per eval tick.
        every: Run eval every N optimizer steps.
        slow_every: Run the slow-eval subset every N optimizer steps; must be a multiple
            of `every`.
        slow_on_first_step: If True, also run slow-eval metrics on the first step.
        metrics: Discriminated-union eval metric configs to instantiate.
    """

    batch_size: PositiveInt
    n_steps: PositiveInt
    every: PositiveInt
    slow_every: PositiveInt
    slow_on_first_step: bool = True
    metrics: list[AnyEvalMetricConfig] = Field(default_factory=list)


class ExperimentConfig[T: BaseConfig, D: BaseConfig](BaseConfig):
    """Full YAML schema for an in-repo experiment.

    Subclass with concrete `target` / `data` types per experiment, e.g.::

        class LMExperimentConfig(ExperimentConfig[LMTargetConfig, LMDataConfig]):
            pass

    Omit the `eval:` block to skip eval entirely.

    Attributes:
        pd: PD algorithm config.
        runtime: Compute-substrate config (autocast, device, DP).
        cadence: Train-log + checkpoint cadence.
        target: Per-experiment target-model config.
        data: Per-experiment data config.
        eval: Optional eval-pass config; `None` skips eval entirely.
    """

    pd: PDConfig
    runtime: RuntimeConfig
    cadence: Cadence
    target: T
    data: D
    eval: EvalConfig | None = None
