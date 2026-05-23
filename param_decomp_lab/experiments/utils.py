"""Shared config schema for in-repo experiment YAMLs.

Each experiment subclasses `ExperimentConfig` to fix the concrete `target` / `data` types
and parses its YAML with `<Experiment>Config.from_file(path)`. `save_run_meta` persists
the resolved config under `run_meta.yaml`; `SavedRun` reads it back and dispatches via
the experiment's `Reloader` class FQN.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, PositiveInt

from param_decomp.base_config import BaseConfig
from param_decomp.configs import Cadence, PDConfig, RuntimeConfig
from param_decomp_lab.eval_metrics import AnyEvalMetricConfig

RUN_META_FILENAME = "run_meta.yaml"


class EvalConfig(BaseConfig):
    """Eval-pass settings: loader batch size, how many batches to consume per eval tick,
    when eval fires, and the list of eval `Metric` configs to instantiate."""

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
    """

    pd: PDConfig
    runtime: RuntimeConfig
    cadence: Cadence
    target: T
    data: D
    eval: EvalConfig | None = None


def save_run_meta(
    out_dir: Path | None,
    *,
    reloader_class: type,
    cfg: ExperimentConfig[Any, Any],
) -> None:
    """Write `{out_dir}/run_meta.yaml`: the resolved `ExperimentConfig` plus the
    `reloader_class` FQN used by `SavedRun` to rebuild target/loaders/run_batch.

    Skipped when `out_dir` is None (non-main ranks / silent sinks).
    """
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fqn = f"{reloader_class.__module__}:{reloader_class.__qualname__}"
    payload = {"reloader_class": fqn, **cfg.model_dump(mode="json")}
    with open(out_dir / RUN_META_FILENAME, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
