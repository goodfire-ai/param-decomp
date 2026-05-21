"""Shared helpers used by the in-repo experiment scripts.

These bridge YAML config -> the objects `optimize()` expects, without imposing inheritance
on experiment scripts.
"""

from pathlib import Path
from typing import Any

import yaml

from param_decomp.configs import PDConfig, RuntimeConfig
from param_decomp.metrics import METRIC_REGISTRY, discover_metrics
from param_decomp.metrics.base import Metric, MetricConfig
from param_decomp.run_sink import RunSink

RUN_META_FILENAME = "run_meta.yaml"


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Read a YAML file and assert that it parses to a mapping."""
    with open(path) as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"config must be a YAML mapping: {path}"
    return data


def build_eval_metrics(eval_metrics_dict: dict[str, Any] | None) -> list[Metric[MetricConfig]]:
    """Turn a dict-of-config (from YAML) into instantiated eval `Metric` objects.

    The metrics are not yet bound to a `ComponentModel` — `optimize()` binds them after
    constructing the model.
    """
    discover_metrics()
    if not eval_metrics_dict:
        return []
    instances: list[Metric[MetricConfig]] = []
    for metric_name, raw_cfg in eval_metrics_dict.items():
        assert metric_name in METRIC_REGISTRY, (
            f"unknown metric {metric_name!r} (registered: {sorted(METRIC_REGISTRY)})"
        )
        cls = METRIC_REGISTRY[metric_name]
        cfg = (
            raw_cfg
            if isinstance(raw_cfg, MetricConfig)
            else cls.config_type.model_validate(raw_cfg or {})
        )
        instances.append(cls(cfg))
    return instances


def run_sink_from_logging_block(
    out_dir: Path | None,
    logging_block: dict[str, Any],
    *,
    wandb_project: str | None = None,
    wandb_run_id: str | None = None,
    wandb_name: str | None = None,
    wandb_tags: list[str] | None = None,
    wandb_configs: dict[str, Any] | None = None,
) -> RunSink:
    """Build a `RunSink` from the YAML `logging:` block.

    The YAML block carries cadence (train_log_freq, eval_freq, slow_eval_freq, n_eval_steps,
    slow_eval_on_first_step, save_freq) and historically also a dict-of-eval-metrics-configs.
    This helper consumes only the cadence keys; the eval-metrics dict is the responsibility of
    `build_eval_metrics(...)`.

    Pass `out_dir=None` to get a `RunSink.silent(...)`. Pass `wandb_project` to use
    `RunSink.with_wandb(...)`; otherwise `RunSink.local(...)`.
    """
    kwargs = dict(
        train_log_freq=logging_block["train_log_freq"],
        eval_freq=logging_block["eval_freq"],
        slow_eval_freq=logging_block["slow_eval_freq"],
        n_eval_steps=logging_block["n_eval_steps"],
        save_freq=logging_block.get("save_freq"),
        slow_eval_on_first_step=logging_block.get("slow_eval_on_first_step", True),
    )
    if out_dir is None:
        return RunSink.silent(**kwargs)
    if wandb_project is None:
        return RunSink.local(out_dir, **kwargs)
    assert wandb_run_id is not None, "wandb_run_id required when wandb_project is set"
    return RunSink.with_wandb(
        out_dir,
        project=wandb_project,
        run_id=wandb_run_id,
        name=wandb_name,
        tags=wandb_tags,
        configs=wandb_configs,
        **kwargs,
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
        experiment: tms                       # ExperimentSpec name (dispatch key)
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
