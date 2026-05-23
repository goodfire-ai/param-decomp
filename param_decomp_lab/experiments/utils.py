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
    """Eval-loader batch size + the list of eval `Metric` configs to instantiate.

    Note: how many batches per eval call lives on `Cadence.n_eval_steps` since the
    trainer owns that loop count.
    """

    batch_size: PositiveInt
    metrics: list[AnyEvalMetricConfig] = Field(default_factory=list)


class ExperimentConfig[T: BaseConfig, D: BaseConfig](BaseConfig):
    """Full YAML schema for an in-repo experiment.

    Subclass with concrete `target` / `data` types per experiment, e.g.::

        class LMExperimentConfig(ExperimentConfig[LMTargetConfig, LMDataConfig]):
            pass
    """

    pd: PDConfig
    runtime: RuntimeConfig
    cadence: Cadence
    target: T
    data: D
    eval: EvalConfig


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
