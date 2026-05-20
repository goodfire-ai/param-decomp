"""`RunSink`: the **writer** end of a PD run.

Three-phase lifecycle: `RunConfig` (recipe) → `RunSink` (writer during
training) → `SavedRun` (reader after). A `RunSink` lives inside the training
process for its duration and dies with it; it owns the active `wandb.run`
session and the on-disk output directory for the run.

Three usage modes:

    # Recipe-mediated (run_pd does this for you):
    sink = RunSink.for_run(run_cfg, *, wandb_project=..., launch_id=...)

    # Notebook with local persistence:
    sink = RunSink.local(Path("/tmp/my_run"))

    # Notebook with local + wandb:
    sink = RunSink.with_wandb(Path("/tmp/my_run"), project="my-proj")

    # No persistence (tests, quick checks):
    sink = RunSink.silent()

All ranks construct a sink; non-main ranks transparently get a no-op sink
(`out_dir` is `None`, wandb is treated as inactive). The trainer never has
to check rank for logging.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb
from PIL import Image
from tqdm import tqdm

from param_decomp.log import logger
from param_decomp.recipes import RunRecipe
from param_decomp.run import RUN_CONFIG_FILENAME, RunConfig
from param_decomp.settings import PARAM_DECOMP_OUT_DIR
from param_decomp.utils.distributed_utils import is_main_process
from param_decomp.utils.logging_utils import local_log
from param_decomp.utils.run_utils import save_file
from param_decomp.utils.wandb_utils import init_wandb, try_wandb


@dataclass(frozen=True)
class RunSink:
    """Side-effect sink for a training run: structured metrics + checkpoints + console.

    Construct via one of the classmethods, not ``__init__`` directly. Non-main
    ranks always get a no-op sink (``out_dir=None``, ``_wandb_active=False``).
    """

    out_dir: Path | None
    _wandb_active: bool

    # =========================== Constructors ===========================

    @classmethod
    def for_run(
        cls,
        run_cfg: RunConfig,
        *,
        wandb_project: str | None = None,
        launch_id: str | None = None,
        recipe: RunRecipe[Any] | None = None,
    ) -> "RunSink":
        """Recipe-mediated setup. Used by ``run_pd``.

        On the main rank: creates ``PARAM_DECOMP_OUT_DIR/decompositions/<run_id>/``,
        writes ``run_config.yaml`` next to (eventual) checkpoints, and (if
        ``wandb_project`` is set) inits wandb with tags derived from the recipe
        name + ``launch_id`` + ``$SLURM_ARRAY_JOB_ID``.

        Pass ``recipe=...`` if you've already resolved the run's materializer.
        Otherwise resolved internally.

        On non-main ranks: skips all the I/O and returns a no-op handle.
        """
        if not is_main_process():
            return cls(out_dir=None, _wandb_active=False)

        out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_cfg.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        run_cfg.write(out_dir / RUN_CONFIG_FILENAME)
        logger.info(f"Run ID: {run_cfg.run_id}")
        logger.info(f"Output directory: {out_dir}")

        wandb_active = False
        if wandb_project:
            run_kind = _run_kind_name(run_cfg, recipe=recipe)
            init_wandb(
                wandb_project,
                run_cfg.run_id,
                configs={
                    "pd": run_cfg.pd,
                    "logging": run_cfg.logging,
                    "runtime": run_cfg.runtime,
                },
                name=run_cfg.name,
                tags=_wandb_tags(run_kind=run_kind, launch_id=launch_id),
                view_meta=run_cfg.view_meta,
            )
            wandb.save(str(out_dir / RUN_CONFIG_FILENAME), base_path=out_dir, policy="now")
            wandb_active = True
        logger.info(run_cfg.pd)
        return cls(out_dir=out_dir, _wandb_active=wandb_active)

    @classmethod
    def local(cls, out_dir: Path) -> "RunSink":
        """Notebook: local-files-only sink. No wandb."""
        if not is_main_process():
            return cls(out_dir=None, _wandb_active=False)
        out_dir.mkdir(parents=True, exist_ok=True)
        return cls(out_dir=out_dir, _wandb_active=False)

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
        """Notebook: local files + wandb. Inits wandb with the given project/name/tags."""
        if not is_main_process():
            return cls(out_dir=None, _wandb_active=False)
        out_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(project=project, name=name, tags=tags or [], config=config or {})
        return cls(out_dir=out_dir, _wandb_active=True)

    @classmethod
    def silent(cls) -> "RunSink":
        """No persistence, no wandb. Useful for tests / quick interactive runs."""
        return cls(out_dir=None, _wandb_active=False)

    # =========================== Output API ===========================

    def log(
        self,
        metrics: dict[str, Any],
        *,
        step: int,
        section: str | None = None,
    ) -> None:
        """Emit a flat metrics dict.

        Writes to ``{out_dir}/{step}.json`` (via ``local_log``) if local
        persistence is on, and to wandb if active. ``section`` is prefixed to
        every W&B key (e.g. ``"eval"`` → ``"eval/loss/total"``); local logs
        use the unsectioned keys.
        """
        if self.out_dir is not None:
            local_log(metrics, step, self.out_dir)
        if self._wandb_active:
            wandb_metrics = (
                {f"{section}/{k}": _wandb_value(v) for k, v in metrics.items()}
                if section is not None
                else {k: _wandb_value(v) for k, v in metrics.items()}
            )
            try_wandb(wandb.log, wandb_metrics, step=step)

    def console(self, *lines: str) -> None:
        """Print lines to stderr via ``tqdm.write``. No-op on non-main ranks."""
        if not is_main_process():
            return
        for line in lines:
            tqdm.write(line)

    def checkpoint(self, state_dict: dict[str, Any], *, step: int) -> None:
        """Save model ``state_dict`` to ``{out_dir}/model_{step}.pth`` + push to wandb."""
        if self.out_dir is None:
            return
        path = self.out_dir / f"model_{step}.pth"
        save_file(state_dict, path)
        logger.info(f"Saved checkpoint to {path}")
        if self._wandb_active:
            try_wandb(wandb.save, str(path), base_path=str(self.out_dir), policy="now")

    def finish(self) -> None:
        """End-of-run cleanup."""
        if self._wandb_active and wandb.run is not None:
            wandb.finish()


def _wandb_value(v: Any) -> Any:
    """Wrap non-wandb-native types (e.g. ``PIL.Image``) for ``wandb.log``."""
    if isinstance(v, Image.Image):
        return wandb.Image(v)
    return v


def _run_kind_name(
    run_cfg: RunConfig,
    *,
    recipe: RunRecipe[Any] | None,
) -> str:
    resolved_recipe = recipe if recipe is not None else run_cfg.recipe.load()
    return resolved_recipe.name


def _wandb_tags(*, run_kind: str, launch_id: str | None) -> list[str]:
    """Tags attached to every wandb run from ``RunSink.for_run``."""
    tags = [run_kind]
    if launch_id is not None:
        tags.append(launch_id)
    slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
    if slurm_array_job_id is not None:
        tags.append(f"slurm-array-job-id_{slurm_array_job_id}")
    return tags
