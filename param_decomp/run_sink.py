"""`RunSink`: the **writer** end of a PD run.

Three-phase lifecycle: `RunConfig` (recipe) → `RunSink` (writer during
training) → `SavedRun` (reader after). A `RunSink` lives inside the training
process for its duration and dies with it; it owns the active `wandb.run`
session and the on-disk output directory for the run.

For driver-mediated runs, ``run_pd`` constructs a sink for you via its own
internal helper — you don't touch ``RunSink`` directly. The three public
constructors below are the notebook / script entry points.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb
from PIL import Image
from tqdm import tqdm

from param_decomp.log import logger
from param_decomp.utils.distributed_utils import is_main_process
from param_decomp.utils.logging_utils import local_log
from param_decomp.utils.run_utils import save_file
from param_decomp.utils.wandb_utils import try_wandb


@dataclass(frozen=True)
class RunSink:
    """Side-effect sink for a training run: metrics + checkpoints + console output.

    Construct via one of the classmethods, not ``__init__`` directly. The
    three notebook constructors map to the three useful combinations of
    ``(out_dir, wandb)``:

        |              | no out_dir   | with out_dir                       |
        |--------------|--------------|------------------------------------|
        | no wandb     | ``silent()`` | ``local(out_dir)``                 |
        | with wandb   |  (n/a)       | ``with_wandb(out_dir, project=…)`` |

    The (no out_dir, with wandb) combination is omitted because a wandb-only
    run can't be reloaded later — if you want wandb you almost always want a
    local copy of the checkpoint too.

    Driver-mediated runs (``pd-run`` / ``_worker.py``) don't use these
    classmethods — they go through ``run_pd``, which constructs a sink via
    an internal helper that knows about ``RunConfig`` fields.

    Non-main DDP ranks transparently get a no-op sink (``out_dir=None``,
    ``_wandb_active=False``). The trainer never has to check rank for logging.
    """

    out_dir: Path | None
    _wandb_active: bool

    # =========================== Constructors ===========================

    @classmethod
    def silent(cls) -> "RunSink":
        """No persistence, no wandb. Useful for tests / quick interactive runs."""
        return cls(out_dir=None, _wandb_active=False)

    @classmethod
    def local(cls, out_dir: Path) -> "RunSink":
        """Local-files-only sink. No wandb."""
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
        """Local files + wandb. Calls ``wandb.init(...)`` with the given project/name/tags."""
        if not is_main_process():
            return cls(out_dir=None, _wandb_active=False)
        out_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(project=project, name=name, tags=tags or [], config=config or {})
        return cls(out_dir=out_dir, _wandb_active=True)

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
