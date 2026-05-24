"""Shared config schema for in-repo experiment YAMLs.

Each experiment subclasses `ExperimentConfig` to fix the concrete `target` / `data` types
and parses its YAML with `<Experiment>Config.from_file(path)`. `save_run_meta` persists
the resolved config under `run_meta.yaml`; `SavedRun` reads it back and dispatches via
the `experiment_kind` field to the matching `experiments/<kind>/run.py` module.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, PositiveInt

from param_decomp.base_config import BaseConfig
from param_decomp.configs import Cadence, PDConfig, RuntimeConfig
from param_decomp_lab.eval_metrics import AnyEvalMetricConfig

RUN_META_FILENAME = "run_meta.yaml"

RunKind = Literal["lm", "tms", "resid_mlp"]
"""Discriminator literal naming the in-repo experiment kinds dispatched on by `SavedRun`."""


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


def save_run_meta(
    out_dir: Path | None,
    *,
    kind: RunKind,
    cfg: ExperimentConfig[Any, Any],
) -> None:
    """Write `{out_dir}/run_meta.yaml` with the resolved config + `experiment_kind`.

    The `experiment_kind` literal lets `SavedRun` dispatch to the matching
    `experiments/<kind>/run.py` module when rebuilding target / loaders / run_batch.
    No-op when `out_dir` is None (non-main ranks / silent sinks).

    Args:
        out_dir: Destination directory; the file is written at
            `out_dir / RUN_META_FILENAME`. Pass `None` to skip writing.
        kind: Experiment kind literal saved under `experiment_kind`.
        cfg: Resolved `ExperimentConfig` to serialize.
    """
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"experiment_kind": kind, **cfg.model_dump(mode="json")}
    with open(out_dir / RUN_META_FILENAME, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
