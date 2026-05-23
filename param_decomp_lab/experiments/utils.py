"""Shared helpers used by the in-repo experiment scripts.

These bridge YAML config -> the objects `optimize()` expects, without imposing inheritance
on experiment scripts.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter

from param_decomp.base_config import BaseConfig
from param_decomp.configs import Cadence, PDConfig, RuntimeConfig
from param_decomp.metrics.base import Metric
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES, AnyEvalMetricConfig
from param_decomp_lab.run_sink import RunSink

RUN_META_FILENAME = "run_meta.yaml"

_EVAL_METRIC_LIST_ADAPTER = TypeAdapter(list[AnyEvalMetricConfig])


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Read a YAML file and assert that it parses to a mapping."""
    with open(path) as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"config must be a YAML mapping: {path}"
    return data


def build_eval_metrics(eval_metrics_list: list[dict[str, Any]] | None) -> list[Metric[BaseConfig]]:
    """Turn a list of eval-metric configs (from YAML) into instantiated eval `Metric` objects.

    Each entry is a dict with a `type: "<ClassName>"` discriminator plus the metric's config
    fields, mirroring the loss-metrics YAML pattern. The list is validated by pydantic as
    `list[AnyEvalMetricConfig]` and each entry is dispatched to its `Metric` class via
    `EVAL_METRIC_CLASSES`.

    The metrics are not yet bound to a `ComponentModel` — `optimize()` binds them after
    constructing the model.
    """
    if not eval_metrics_list:
        return []
    configs = _EVAL_METRIC_LIST_ADAPTER.validate_python(eval_metrics_list)
    return [EVAL_METRIC_CLASSES[cfg.type](cfg) for cfg in configs]


def cadence_from_logging_block(logging_block: dict[str, Any]) -> Cadence:
    """Build a `Cadence` from the YAML `logging:` block."""
    return Cadence(
        train_log_every=logging_block["train_log_freq"],
        eval_every=logging_block["eval_freq"],
        slow_eval_every=logging_block["slow_eval_freq"],
        n_eval_steps=logging_block["n_eval_steps"],
        save_every=logging_block.get("save_freq"),
        slow_eval_on_first_step=logging_block.get("slow_eval_on_first_step", True),
    )


def run_sink_from_logging_block(
    out_dir: Path | None,
    *,
    wandb_project: str | None = None,
    wandb_run_id: str | None = None,
    wandb_name: str | None = None,
    wandb_tags: list[str] | None = None,
    wandb_configs: dict[str, Any] | None = None,
) -> RunSink:
    """Build a `RunSink` from the YAML `logging:` block.

    Pass `out_dir=None` to get a `RunSink.silent()`. Pass `wandb_project` to use
    `RunSink.with_wandb(...)`; otherwise `RunSink.local(...)`.
    """
    if out_dir is None:
        return RunSink.silent()
    if wandb_project is None:
        return RunSink.local(out_dir)
    assert wandb_run_id is not None, "wandb_run_id required when wandb_project is set"
    return RunSink.with_wandb(
        out_dir,
        project=wandb_project,
        run_id=wandb_run_id,
        name=wandb_name,
        tags=wandb_tags,
        configs=wandb_configs,
    )


def save_run_meta(
    out_dir: Path | None,
    *,
    experiment_name: str,
    pd_config: PDConfig,
    runtime_config: RuntimeConfig,
    target_dict: dict[str, Any],
    data_dict: dict[str, Any],
) -> None:
    """Write `{out_dir}/run_meta.yaml` describing how to reload this run.

    Schema:
        experiment: tms                       # EXPERIMENTS registry key (module dispatch)
        pd: { ... PDConfig.model_dump ... }
        runtime: { ... RuntimeConfig.model_dump ... }
        target: { ... raw experiment target_cfg ... }
        data: { ... raw experiment data_cfg ... }

    Skipped when ``out_dir`` is None (non-main ranks / silent sinks).
    """
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": experiment_name,
        "pd": pd_config.model_dump(mode="json"),
        "runtime": runtime_config.model_dump(mode="json"),
        "target": target_dict,
        "data": data_dict,
    }
    with open(out_dir / RUN_META_FILENAME, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
