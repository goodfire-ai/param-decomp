"""Shared config schema for in-repo experiment YAMLs.

Each experiment subclasses `ExperimentConfig` to fix the concrete `target` / `data`
types.
"""

from typing import Self

import wandb
from pydantic import Field, PositiveInt, model_validator

from param_decomp.base_config import BaseConfig
from param_decomp.configs import Cadence, PDConfig, RuntimeConfig
from param_decomp.distributed import is_main_process
from param_decomp.metrics.faithfulness import FaithfulnessLossConfig
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES, AnyEvalMetricConfig
from param_decomp_lab.eval_metrics.targeted_ci_heatmap import TargetedCIHeatmapConfig
from param_decomp_lab.infra.run_files import generate_run_id, write_run_metadata_start
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.infra.wandb import try_wandb
from param_decomp_lab.run_sink import RunSink
from param_decomp_lab.targeted import NontargetConfig

EXPERIMENT_CONFIG_FILENAME = "experiment_config.yaml"


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


def _assert_heatmap_probe_matches_data(cfg: TargetedCIHeatmapConfig, data: BaseConfig) -> None:
    """The heatmap's target probes ARE the target distribution, so its probe spec must
    match `data` (the single source of truth) rather than being re-specified and drifting."""
    if cfg.active_indices is not None:
        data_indices = getattr(data, "active_indices", None)
        assert data_indices == cfg.active_indices, (
            f"TargetedCIHeatmap.active_indices {cfg.active_indices} must match "
            f"data.active_indices {data_indices}: the heatmap probes are the target distribution"
        )
    if cfg.prompts_file is not None:
        for field in ("prompts_file", "tokenizer_name", "max_seq_len"):
            data_val = getattr(data, field, None)
            cfg_val = getattr(cfg, field)
            assert data_val == cfg_val, (
                f"TargetedCIHeatmap.{field}={cfg_val!r} must match data.{field}={data_val!r}: "
                "the heatmap probes are the target distribution"
            )


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
    label: str | None = None
    eval: EvalConfig | None = None
    wandb: WandbConfig | None = None
    nontarget: NontargetConfig[D] | None = None

    @model_validator(mode="after")
    def validate_targeted(self) -> Self:
        eval_metrics = self.eval.metrics if self.eval is not None else []
        nontarget_only = [
            c.type
            for c in eval_metrics
            if EVAL_METRIC_CLASSES[c.type].eval_distribution == "nontarget"
        ]
        if self.nontarget is None:
            # These metrics are routed to the nontarget eval loop, which only exists when
            # a `nontarget:` block is configured. Without one they would silently run on
            # the target distribution and log misleading numbers — reject the combination.
            assert not nontarget_only, (
                f"eval metrics {nontarget_only} consume the nontarget distribution but no "
                "`nontarget:` block is configured; add one or remove these metrics"
            )
            return self
        assert self.pd.use_delta_component, (
            "targeted decomposition forces the delta component on; set pd.use_delta_component=true"
        )
        assert not any(isinstance(c, FaithfulnessLossConfig) for c in self.pd.loss_metrics), (
            "FaithfulnessLoss drives the delta to zero; targeted runs need the delta to carry "
            "nontarget behavior — remove it from pd.loss_metrics"
        )
        assert self.pd.faithfulness_warmup_steps == 0, (
            "faithfulness warmup zeroes the delta right before targeted training; set "
            "pd.faithfulness_warmup_steps=0"
        )
        for c in eval_metrics:
            if isinstance(c, TargetedCIHeatmapConfig):
                _assert_heatmap_probe_matches_data(c, self.data)
        return self


def init_pd_run[T: BaseConfig, D: BaseConfig](
    cfg: ExperimentConfig[T, D],
    *,
    group: str | None,
    tags: str | None,
    run_id: str | None = None,
) -> RunSink:
    """Allocate `run_id` + `out_dir`, write `experiment_config.yaml`, return a sink.

    Local-only when `cfg.wandb is None`, else wandb-backed. Non-main DDP ranks get a
    silent no-op sink without touching disk or wandb. `group` is a "launched together"
    id; `tags` is a comma-separated string of orthogonal labels.
    """
    if not is_main_process():
        return RunSink.silent()
    run_id = run_id or generate_run_id("param_decomp")
    out_dir = PARAM_DECOMP_OUT_DIR / "runs" / run_id
    cfg_path = out_dir / EXPERIMENT_CONFIG_FILENAME
    cfg.to_file(cfg_path)
    write_run_metadata_start(out_dir)
    keep_last_n = cfg.cadence.keep_last_n_checkpoints
    if cfg.wandb is None:
        return RunSink.local(out_dir, keep_last_n_checkpoints=keep_last_n)
    parsed_tags = [s.strip() for s in tags.split(",") if s.strip()] if tags else None
    sink = RunSink.with_wandb(
        out_dir,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        run_id=run_id,
        config=cfg,
        group=group,
        tags=parsed_tags,
        keep_last_n_checkpoints=keep_last_n,
    )
    try_wandb(wandb.save, str(cfg_path), base_path=str(out_dir), policy="now")
    return sink
