"""Output sink for a PD run: local files + opportunistic wandb + checkpoints.

Encapsulates the side-effect plumbing the trainer would otherwise do itself.
`optimize` takes a `RunSink` and calls `.log(...)` / `.checkpoint(...)` /
`.finish()` — no more nested ``if out_dir is not None`` and
``if wandb.run is not None`` checks, no more reliance on global wandb state.

Three usage modes:

    # Driver-mediated (run_pd does this for you):
    sink = RunSink.for_run(run_cfg, wandb_project="my-proj", launch_id=...)

    # Notebook with local persistence:
    sink = RunSink.local(Path("/tmp/my_run"))

    # Notebook with local + wandb:
    sink = RunSink.with_wandb(Path("/tmp/my_run"), project="my-proj", name="my-run")

    # No persistence (tests, quick checks):
    sink = RunSink.silent()

All ranks construct a sink; non-main ranks transparently get a no-op sink
(out_dir is squashed to None, wandb is treated as inactive). The trainer
never has to check rank.
"""

import os
from pathlib import Path
from typing import Any

import wandb
from PIL import Image

from param_decomp.driver_path import load_driver
from param_decomp.log import logger
from param_decomp.run import RUN_CONFIG_FILENAME, RunConfig
from param_decomp.settings import PARAM_DECOMP_OUT_DIR
from param_decomp.utils.distributed_utils import is_main_process
from param_decomp.utils.logging_utils import local_log
from param_decomp.utils.run_utils import save_file
from param_decomp.utils.wandb_utils import init_wandb, try_wandb


class RunSink:
    """Side-effect sink for training: structured metrics + checkpoints.

    No-op on non-main ranks. Construct via one of the classmethods below
    rather than calling ``__init__`` directly.
    """

    def __init__(self, *, out_dir: Path | None, wandb_active: bool) -> None:
        # Non-main ranks get a no-op sink no matter what the caller passed.
        if not is_main_process():
            out_dir = None
            wandb_active = False
        self._out_dir = out_dir
        self._wandb_active = wandb_active

    @property
    def out_dir(self) -> Path | None:
        """Where local artifacts land; ``None`` if persistence is disabled."""
        return self._out_dir

    # ------------------------------ Constructors ------------------------------

    @classmethod
    def for_run(
        cls,
        run_cfg: RunConfig,
        *,
        wandb_project: str | None = None,
        launch_id: str | None = None,
    ) -> "RunSink":
        """Driver-mediated setup. Mirrors what ``run_pd`` does today:

        - Creates ``PARAM_DECOMP_OUT_DIR/decompositions/<run_id>/``.
        - Writes ``run_config.yaml`` next to (eventual) checkpoints.
        - Initializes W&B from ``run_cfg`` fields if ``wandb_project`` is set,
          and pushes the spec to W&B.

        Side effects only happen on the main rank; non-main ranks return a
        no-op sink.
        """
        out_dir: Path | None = None
        wandb_active = False
        if is_main_process():
            out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_cfg.run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            run_cfg.write(out_dir / RUN_CONFIG_FILENAME)
            logger.info(f"Run ID: {run_cfg.run_id}")
            logger.info(f"Output directory: {out_dir}")
            if wandb_project:
                init_wandb(
                    wandb_project,
                    run_cfg.run_id,
                    configs={
                        "pd": run_cfg.pd,
                        "logging": run_cfg.logging,
                        "runtime": run_cfg.runtime,
                    },
                    name=run_cfg.name,
                    tags=_wandb_tags(
                        driver_name=load_driver(run_cfg.driver_path).name,
                        launch_id=launch_id,
                    ),
                    view_meta=run_cfg.view_meta,
                )
                wandb.save(str(out_dir / RUN_CONFIG_FILENAME), base_path=out_dir, policy="now")
                wandb_active = True
            logger.info(run_cfg.pd)
        return cls(out_dir=out_dir, wandb_active=wandb_active)

    @classmethod
    def local(cls, out_dir: Path) -> "RunSink":
        """Local-files-only sink. No wandb."""
        if is_main_process():
            out_dir.mkdir(parents=True, exist_ok=True)
        return cls(out_dir=out_dir, wandb_active=False)

    @classmethod
    def with_wandb(
        cls,
        out_dir: Path,
        *,
        project: str,
        name: str | None = None,
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> "RunSink":
        """Local files + wandb. Notebook users who want wandb without going
        through ``run_pd`` use this.
        """
        if is_main_process():
            out_dir.mkdir(parents=True, exist_ok=True)
            wandb.init(project=project, name=name, tags=tags or [], config=config or {})
        return cls(out_dir=out_dir, wandb_active=True)

    @classmethod
    def silent(cls) -> "RunSink":
        """No-op sink. No local persistence, no wandb."""
        return cls(out_dir=None, wandb_active=False)

    # ------------------------------ Output API ------------------------------

    def log(
        self,
        metrics: dict[str, Any],
        *,
        step: int,
        section: str | None = None,
    ) -> None:
        """Log a flat metrics dict.

        Writes to ``{out_dir}/{step}.json`` (via ``local_log``) if local
        persistence is on, and to wandb if active. ``section`` is prefixed to
        every W&B key (e.g. ``"eval"`` → ``"eval/loss/total"``); local logs
        are written with the unsectioned keys.
        """
        if self._out_dir is not None:
            local_log(metrics, step, self._out_dir)
        if self._wandb_active:
            wandb_metrics = (
                {f"{section}/{k}": _wandb_value(v) for k, v in metrics.items()}
                if section is not None
                else {k: _wandb_value(v) for k, v in metrics.items()}
            )
            try_wandb(wandb.log, wandb_metrics, step=step)

    def checkpoint(self, state_dict: dict[str, Any], *, step: int) -> None:
        """Save model ``state_dict`` to ``{out_dir}/model_{step}.pth`` + push to wandb."""
        if self._out_dir is None:
            return
        path = self._out_dir / f"model_{step}.pth"
        save_file(state_dict, path)
        logger.info(f"Saved checkpoint to {path}")
        if self._wandb_active:
            try_wandb(wandb.save, str(path), base_path=str(self._out_dir), policy="now")

    def finish(self) -> None:
        """End-of-run cleanup."""
        if self._wandb_active and wandb.run is not None:
            wandb.finish()


def _wandb_value(v: Any) -> Any:
    """Wrap non-wandb-native types (e.g. ``PIL.Image``) for ``wandb.log``."""
    if isinstance(v, Image.Image):
        return wandb.Image(v)
    return v


def _wandb_tags(*, driver_name: str, launch_id: str | None) -> list[str]:
    """Tags attached to every wandb run from ``RunSink.for_run``.

    ``driver_name`` lets the W&B UI filter by experiment kind (lm / tms /
    resid_mlp). ``launch_id`` groups every run from one ``pd-run`` invocation
    (single launches and sweep arrays alike). The SLURM array job id is picked
    up from ``$SLURM_ARRAY_JOB_ID``.
    """
    tags = [driver_name]
    if launch_id is not None:
        tags.append(launch_id)
    slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
    if slurm_array_job_id is not None:
        tags.append(f"slurm-array-job-id_{slurm_array_job_id}")
    return tags
