"""Shared config schema for in-repo experiment YAMLs.

Each experiment subclasses `ExperimentConfig` to fix the concrete `target` / `data`
types.
"""

import wandb
from pydantic import Field, PositiveInt

from param_decomp.base_config import BaseConfig
from param_decomp.configs import Cadence, PDConfig, RuntimeConfig
from param_decomp.distributed import is_main_process
from param_decomp_lab.eval_metrics import AnyEvalMetricConfig
from param_decomp_lab.infra.run_files import generate_run_id
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.infra.wandb import try_wandb
from param_decomp_lab.run_sink import RunSink

RUN_META_FILENAME = "run_meta.yaml"


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


class ExperimentConfig[T: BaseConfig, D: BaseConfig](BaseConfig):
    """Full YAML schema for an in-repo experiment.

    Subclass with concrete `target` / `data` types per experiment:

        class LMExperimentConfig(ExperimentConfig[LMTargetConfig, LMDataConfig]):
            pass

    Omit the `eval:` block to skip eval entirely; omit `wandb:` to skip wandb (the run
    still writes `run_meta.yaml` + checkpoints locally).
    """

    pd: PDConfig
    runtime: RuntimeConfig
    cadence: Cadence
    target: T
    data: D
    eval: EvalConfig | None = None
    wandb: WandbConfig | None = None


def init_pd_run[T: BaseConfig, D: BaseConfig](
    cfg: ExperimentConfig[T, D],
    *,
    group: str | None,
    tags: str | None,
    run_id: str | None = None,
) -> RunSink:
    """Allocate `run_id` + `out_dir`, write `run_meta.yaml`, return a sink.

    Local-only when `cfg.wandb is None`, else wandb-backed. Non-main DDP ranks get a
    silent no-op sink without touching disk or wandb. `group` is a "launched together"
    id; `tags` is a comma-separated string of orthogonal labels.
    """
    if not is_main_process():
        return RunSink.silent()
    run_id = run_id or generate_run_id("param_decomp")
    out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id
    meta_path = out_dir / RUN_META_FILENAME
    cfg.to_file(meta_path)
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
    try_wandb(wandb.save, str(meta_path), base_path=str(out_dir), policy="now")
    return sink
