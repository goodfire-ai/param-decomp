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
from param_decomp_lab.infra.run_files import generate_run_id
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.run_sink import RunSink

RUN_META_FILENAME = "run_meta.yaml"


class WandbConfig(BaseConfig):
    """Wandb logging settings. Presence on ExperimentConfig opts in; omit to skip wandb.

    Attributes:
        project: Wandb project name.
        entity: Wandb entity; falls back to ``WANDB_ENTITY`` env / authenticated user
            when None.
    """

    project: str
    entity: str | None = None


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

    Omit the `eval:` block to skip eval entirely. Omit the `wandb:` block to skip wandb
    (the run still writes ``run_meta.yaml`` + checkpoints locally).

    Attributes:
        pd: PD algorithm config.
        runtime: Compute-substrate config (autocast, device, DP).
        cadence: Train-log + checkpoint cadence.
        target: Per-experiment target-model config.
        data: Per-experiment data config.
        eval: Optional eval-pass config; `None` skips eval entirely.
        wandb: Optional wandb logging config; `None` skips wandb entirely.
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
) -> RunSink:
    """Allocate run_id + out_dir, write run_meta, return a sink.

    Returns a local-only sink when `cfg.wandb` is None, or a wandb-backed sink when it
    is set. `group` collects runs that were launched together (wandb's native collapsing
    + workspace filter `ws.Metric("Group")`); `tags` is a comma-separated string of
    orthogonal user-defined labels. Both are no-ops in local-only mode.
    """
    run_id = generate_run_id("param_decomp")
    out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id
    cfg.to_file(out_dir / RUN_META_FILENAME)
    if cfg.wandb is None:
        return RunSink.local(out_dir)
    parsed_tags = [s.strip() for s in tags.split(",") if s.strip()] if tags else None
    return RunSink.with_wandb(
        out_dir,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        run_id=run_id,
        config=cfg,
        group=group,
        tags=parsed_tags,
    )
