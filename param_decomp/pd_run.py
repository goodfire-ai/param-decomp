"""The `PDRun` domain object — both directions of a run's I/O.

A `PDRun` is a handle to a parameter-decomposition run. The same object models
both lifecycle directions:

**Write-side** (during training):
    PDRun.for_run(run_cfg, *, wandb_project, launch_id)   # driver-mediated
    PDRun.local(out_dir)                                  # notebook, files
    PDRun.with_wandb(out_dir, *, project, ...)            # notebook + wandb
    PDRun.silent()                                        # no persistence

Methods: ``.log()``, ``.console()``, ``.checkpoint()``, ``.finish()``.

**Read-side** (reload):
    PDRun.from_path(path)                                 # full reload handle
    PDRun.run_cfg_from_path(path)                         # spec only

Methods: ``.load_model()``, ``.load_target()``,
``.build_train_loader()``, ``.build_eval_loader()``.

The two halves are intentionally one type: a run has one identity (config,
driver, output location) that's relevant in both. Methods that require fields
the constructor didn't populate (e.g. ``load_model()`` on a fresh
``for_run`` handle, before training has saved a checkpoint) assert.

On non-main DDP ranks, write-side constructors automatically yield a no-op
handle (no ``out_dir``, no wandb session); the trainer never has to check
rank for logging.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from param_decomp.configs import PDConfig
from param_decomp.driver_path import load_driver
from param_decomp.experiments.driver import ExperimentDriver
from param_decomp.log import logger
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.models.component_model import ComponentModel
from param_decomp.run import RUN_CONFIG_FILENAME, RunConfig
from param_decomp.settings import PARAM_DECOMP_OUT_DIR
from param_decomp.types import ModelPath
from param_decomp.utils.distributed_utils import DistributedState, is_main_process
from param_decomp.utils.logging_utils import local_log
from param_decomp.utils.run_files import resolve_config_path, resolve_run_files
from param_decomp.utils.run_utils import save_file
from param_decomp.utils.wandb_utils import init_wandb, try_wandb


@dataclass(frozen=True)
class PDRun:
    """Domain object for a PD run.

    Construct via the classmethods, not ``__init__`` directly. Field invariants:

    - ``run_cfg`` + ``driver`` are both set together (driver-mediated handles) or
      both ``None`` (notebook handles built from ``local`` / ``with_wandb`` /
      ``silent``).
    - ``checkpoint_path`` is only set by ``from_path`` (you have a checkpoint
      to reload). For ``for_run``, it's ``None`` until training writes one;
      ``load_model()`` will fail.
    - ``out_dir`` / ``_wandb_active`` reflect the write-side state. ``None`` /
      ``False`` on non-main ranks regardless of constructor.
    """

    run_cfg: RunConfig | None
    driver: ExperimentDriver[Any] | None
    out_dir: Path | None
    checkpoint_path: Path | None
    _wandb_active: bool = False

    # =========================== Constructors ===========================

    @classmethod
    def for_run(
        cls,
        run_cfg: RunConfig,
        *,
        wandb_project: str | None = None,
        launch_id: str | None = None,
    ) -> "PDRun":
        """Start a new driver-mediated run.

        Resolves the driver, creates
        ``PARAM_DECOMP_OUT_DIR/decompositions/<run_id>/``, writes
        ``run_config.yaml``, and (if ``wandb_project`` is set) inits wandb with
        tags derived from the driver name + ``launch_id`` + ``$SLURM_ARRAY_JOB_ID``.

        On non-main ranks: skips all the I/O and returns a no-op handle
        (``out_dir=None``, ``_wandb_active=False``).
        """
        driver = load_driver(run_cfg.driver_path)
        assert isinstance(run_cfg, driver.config_type), (
            f"RunConfig has type {type(run_cfg).__name__}, "
            f"expected {driver.config_type.__name__} from driver {run_cfg.driver_path}"
        )
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
                    tags=_wandb_tags(driver_name=driver.name, launch_id=launch_id),
                    view_meta=run_cfg.view_meta,
                )
                wandb.save(str(out_dir / RUN_CONFIG_FILENAME), base_path=out_dir, policy="now")
                wandb_active = True
            logger.info(run_cfg.pd)
        return cls(
            run_cfg=run_cfg,
            driver=driver,
            out_dir=out_dir,
            checkpoint_path=None,
            _wandb_active=wandb_active,
        )

    @classmethod
    def from_path(cls, path: ModelPath) -> "PDRun":
        """Reload an existing run from disk or wandb.

        Resolves the spec + checkpoint, instantiates the driver from the spec's
        ``driver_path``, validates the subclass match. Write-side methods are
        no-ops on the returned handle (no ``out_dir`` set, no wandb session
        attached).
        """
        files = resolve_run_files(
            path, config_filename=RUN_CONFIG_FILENAME, checkpoint_prefix="model"
        )
        run_cfg = RunConfig.from_file(files.config_path)
        driver = load_driver(run_cfg.driver_path)
        assert isinstance(run_cfg, driver.config_type), (
            f"RunConfig has type {type(run_cfg).__name__}, expected {driver.config_type.__name__}"
        )
        return cls(
            run_cfg=run_cfg,
            driver=driver,
            out_dir=None,
            checkpoint_path=files.checkpoint_path,
            _wandb_active=False,
        )

    @classmethod
    def run_cfg_from_path(cls, path: ModelPath) -> RunConfig:
        """Load just the ``RunConfig`` without resolving the checkpoint."""
        return RunConfig.from_file(resolve_config_path(path, config_filename=RUN_CONFIG_FILENAME))

    @classmethod
    def local(cls, out_dir: Path) -> "PDRun":
        """Notebook: local-files-only sink. No wandb, no driver."""
        if is_main_process():
            out_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            run_cfg=None,
            driver=None,
            out_dir=out_dir if is_main_process() else None,
            checkpoint_path=None,
            _wandb_active=False,
        )

    @classmethod
    def with_wandb(
        cls,
        out_dir: Path,
        *,
        project: str,
        name: str | None = None,
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> "PDRun":
        """Notebook: local files + wandb. Inits wandb with the given project/name/tags."""
        out_dir_final: Path | None = None
        wandb_active = False
        if is_main_process():
            out_dir.mkdir(parents=True, exist_ok=True)
            wandb.init(project=project, name=name, tags=tags or [], config=config or {})
            out_dir_final = out_dir
            wandb_active = True
        return cls(
            run_cfg=None,
            driver=None,
            out_dir=out_dir_final,
            checkpoint_path=None,
            _wandb_active=wandb_active,
        )

    @classmethod
    def silent(cls) -> "PDRun":
        """No persistence, no wandb. Useful for tests / quick interactive runs."""
        return cls(
            run_cfg=None,
            driver=None,
            out_dir=None,
            checkpoint_path=None,
            _wandb_active=False,
        )

    # =========================== Read-side ===========================

    @property
    def pd_config(self) -> PDConfig:
        assert self.run_cfg is not None, "no run_cfg on this PDRun (notebook handle?)"
        return self.run_cfg.pd

    @property
    def name(self) -> str:
        return self.driver.name if self.driver is not None else "custom"

    def load_target(self) -> PDTarget:
        assert self.driver is not None and self.run_cfg is not None, (
            "load_target requires a driver-mediated PDRun (from_path or for_run)"
        )
        return self.driver.build_target(self.run_cfg)

    def build_train_loader(
        self,
        *,
        device: str,
        batch_size_override: int | None = None,
        dist_state: DistributedState | None = None,
    ) -> DataLoader[Any]:
        assert self.driver is not None and self.run_cfg is not None, (
            "build_train_loader requires a driver-mediated PDRun"
        )
        return self.driver.build_train_loader(
            self.run_cfg,
            device=device,
            batch_size_override=batch_size_override,
            dist_state=dist_state,
        )

    def build_eval_loader(
        self,
        *,
        device: str,
        batch_size_override: int | None = None,
        dist_state: DistributedState | None = None,
    ) -> DataLoader[Any]:
        assert self.driver is not None and self.run_cfg is not None, (
            "build_eval_loader requires a driver-mediated PDRun"
        )
        return self.driver.build_eval_loader(
            self.run_cfg,
            device=device,
            batch_size_override=batch_size_override,
            dist_state=dist_state,
        )

    def load_model(self) -> ComponentModel:
        assert self.checkpoint_path is not None, (
            "load_model requires a saved checkpoint; this PDRun came from "
            "for_run / local / with_wandb / silent and has no checkpoint yet"
        )
        target = self.load_target()
        return ComponentModel.from_checkpoint(
            config=self.pd_config,
            checkpoint_path=self.checkpoint_path,
            target_model=target.model,
            run_batch=target.run_batch,
            tied_weights=target.tied_weights,
        )

    # =========================== Write-side ===========================

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

        No-op on non-main ranks (``out_dir`` and ``_wandb_active`` were
        squashed by the constructor).
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


def load_component_model(path: ModelPath) -> ComponentModel:
    """Load a `ComponentModel` from a saved driver-mediated PD run.

    The run's driver reconstructs the target from the saved `RunConfig`.
    Notebook callers without a `RunConfig` on disk should use
    ``ComponentModel.from_checkpoint(...)`` directly.
    """
    return PDRun.from_path(path).load_model()


def _wandb_value(v: Any) -> Any:
    """Wrap non-wandb-native types (e.g. ``PIL.Image``) for ``wandb.log``."""
    if isinstance(v, Image.Image):
        return wandb.Image(v)
    return v


def _wandb_tags(*, driver_name: str, launch_id: str | None) -> list[str]:
    """Tags attached to every wandb run from ``PDRun.for_run``."""
    tags = [driver_name]
    if launch_id is not None:
        tags.append(launch_id)
    slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
    if slurm_array_job_id is not None:
        tags.append(f"slurm-array-job-id_{slurm_array_job_id}")
    return tags
