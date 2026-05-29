"""Shared config schema for in-repo experiment YAMLs.

Each experiment subclasses `ExperimentConfig` to fix the concrete `target` / `data`
types.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import wandb
from pydantic import Field, PositiveInt

from param_decomp.base_config import BaseConfig, runtime_cast
from param_decomp.configs import Cadence, PDConfig, RuntimeConfig
from param_decomp.distributed import is_main_process
from param_decomp_lab.eval_metrics import AnyEvalMetricConfig
from param_decomp_lab.infra.run_files import generate_run_id
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.infra.wandb import try_wandb
from param_decomp_lab.resumption.provenance import ResumeProvenance
from param_decomp_lab.run_sink import OnePoolSink, ThreePoolSink

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
    resume_provenance: ResumeProvenance | None = None
    """Set on resumed runs (parent run dir + step); `None` for fresh runs. Lives on the
    config so it flows into `run_meta.yaml` and `wandb.config` via `init_pd_run`, making a
    resumed run's lineage visible in the wandb UI."""


class _PdRunInputs(Protocol):
    """The slice of an experiment config `init_pd_run` reads.

    Lets the single-pool `ExperimentConfig` and the standalone 3-pool
    `ThreePoolLMExperimentConfig` share one sink builder without a common base — it
    only touches `cadence`, `wandb`, and `to_file`, never `pd` / `runtime`.
    """

    @property
    def cadence(self) -> Cadence: ...
    @property
    def wandb(self) -> "WandbConfig | None": ...
    def to_file(self, path: Path | str) -> None: ...


def init_pd_run[S: OnePoolSink | ThreePoolSink](
    cfg: _PdRunInputs,
    *,
    sink_class: type[S],
    group: str | None,
    tags: str | None,
    run_id: str | None = None,
    on_save: Callable[[int], None] | None = None,
) -> S:
    """Allocate `run_id` + `out_dir`, write `run_meta.yaml`, return a sink.

    `sink_class` picks the pool-specific sink (`OnePoolSink` for 1-pool runs,
    `ThreePoolSink` for 3-pool). The choice is the caller's; this helper just
    forwards through to the class's `local` / `with_wandb` / `silent` constructors.

    Local-only when `cfg.wandb is None`, else wandb-backed. Non-main DDP ranks get a
    silent no-op sink without touching disk or wandb. `group` is a "launched together"
    id; `tags` is a comma-separated string of orthogonal labels. `on_save` is an
    optional rank-0 callback the sink invokes after each checkpoint write.
    """
    if not is_main_process():
        return sink_class.silent()
    run_id = run_id or generate_run_id("param_decomp")
    out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id
    meta_path = out_dir / RUN_META_FILENAME
    cfg.to_file(meta_path)
    keep_last_n = cfg.cadence.keep_last_n_checkpoints
    if cfg.wandb is None:
        return sink_class.local(out_dir, keep_last_n_checkpoints=keep_last_n, on_save=on_save)
    parsed_tags = [s.strip() for s in tags.split(",") if s.strip()] if tags else None
    sink = sink_class.with_wandb(
        out_dir,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        run_id=run_id,
        config=runtime_cast(BaseConfig, cfg),
        group=group,
        tags=parsed_tags,
        keep_last_n_checkpoints=keep_last_n,
        on_save=on_save,
    )
    try_wandb(wandb.save, str(meta_path), base_path=str(out_dir), policy="now")
    return sink
