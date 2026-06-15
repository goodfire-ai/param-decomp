"""Run-init helpers for in-repo experiments (sink + run-dir + wandb wiring).

The `ExperimentConfig` YAML schema lives in `param_decomp_config.experiment`.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import wandb

from param_decomp.distributed import is_main_process
from param_decomp_config.base import BaseConfig, runtime_cast
from param_decomp_config.experiment import WandbConfig
from param_decomp_config.pd import Cadence
from param_decomp_lab.infra.run_files import generate_run_id
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.infra.wandb import try_wandb
from param_decomp_lab.run_sink import OnePoolSink, ThreePoolSink

EXPERIMENT_CONFIG_FILENAME = "experiment_config.yaml"


class _PdRunInputs(Protocol):
    """The slice of an experiment config `init_pd_run` reads.

    Lets the single-pool `ExperimentConfig` and the standalone 3-pool
    `ThreePoolLMExperimentConfig` share one sink builder without a common base — it
    only touches `cadence`, `wandb`, and `to_file`, never `pd` / `runtime`.
    """

    @property
    def cadence(self) -> Cadence: ...
    @property
    def wandb(self) -> WandbConfig | None: ...
    def to_file(self, path: Path | str) -> None: ...


def init_pd_run[S: OnePoolSink | ThreePoolSink](
    cfg: _PdRunInputs,
    *,
    sink_class: type[S],
    group: str | None,
    tags: str | None,
    resume_wandb: bool,
    run_id: str | None = None,
    on_save: Callable[[int], None] | None = None,
) -> S:
    """Allocate `run_id` + `out_dir`, write `experiment_config.yaml`, return a sink.

    `sink_class` picks the pool-specific sink (`OnePoolSink` for 1-pool runs,
    `ThreePoolSink` for 3-pool). The choice is the caller's; this helper just
    forwards through to the class's `local` / `with_wandb` / `silent` constructors.

    Local-only when `cfg.wandb is None`, else wandb-backed. Non-main DDP ranks get a
    silent no-op sink without touching disk or wandb. `group` is a "launched together"
    id; `tags` is a comma-separated string of orthogonal labels. `resume_wandb=True`
    continues the existing wandb run (in-place SLURM-requeue resume); `False` creates a
    new run. `on_save` is an optional rank-0 callback the sink invokes after each
    checkpoint write.
    """
    if not is_main_process():
        return sink_class.silent()
    run_id = run_id or generate_run_id("param_decomp")
    out_dir = PARAM_DECOMP_OUT_DIR / "runs" / run_id
    cfg_path = out_dir / EXPERIMENT_CONFIG_FILENAME
    cfg.to_file(cfg_path)
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
        resume=resume_wandb,
        group=group,
        tags=parsed_tags,
        keep_last_n_checkpoints=keep_last_n,
        on_save=on_save,
    )
    try_wandb(wandb.save, str(cfg_path), base_path=str(out_dir), policy="now")
    return sink
