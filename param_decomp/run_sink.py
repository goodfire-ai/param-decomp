"""`RunSink`: side-effect sink for a PD training run.

Owns three things the trainer used to read off `LoggingConfig`:

1. **Output channels**: local files (``out_dir``) + optional wandb.
2. **Cadence**: when to emit train logs, when to eval, when to checkpoint.
3. **Console output**.

Three constructors:

    sink = RunSink.local(out_dir, train_log_freq=..., eval_freq=..., ...)
    sink = RunSink.with_wandb(out_dir, project="...", train_log_freq=..., ...)
    sink = RunSink.silent(train_log_freq=..., ...)            # tests / quick checks

Non-main ranks transparently get a no-op sink (``out_dir=None``, wandb inactive)
regardless of which constructor is called. The trainer never has to check rank.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb
from PIL import Image
from tqdm import tqdm

from param_decomp.base_config import BaseConfig
from param_decomp.log import logger
from param_decomp.utils.distributed_utils import is_main_process
from param_decomp.utils.logging_utils import local_log
from param_decomp.utils.run_utils import save_file
from param_decomp.utils.wandb_utils import init_wandb, try_wandb


@dataclass(frozen=True)
class RunSink:
    """Side-effect sink for a training run.

    Construct via one of the classmethods rather than the dataclass directly.
    Non-main ranks always get a no-op handle (`out_dir=None`, `_wandb_active=False`).
    """

    out_dir: Path | None
    train_log_freq: int
    eval_freq: int
    slow_eval_freq: int
    n_eval_steps: int
    slow_eval_on_first_step: bool
    save_freq: int | None
    _wandb_active: bool

    def __post_init__(self) -> None:
        assert self.train_log_freq > 0, "train_log_freq must be positive"
        assert self.eval_freq > 0, "eval_freq must be positive"
        assert self.slow_eval_freq > 0, "slow_eval_freq must be positive"
        assert self.slow_eval_freq % self.eval_freq == 0, (
            f"slow_eval_freq ({self.slow_eval_freq}) must be a multiple of "
            f"eval_freq ({self.eval_freq})"
        )
        assert self.n_eval_steps > 0, "n_eval_steps must be positive"
        if self.save_freq is not None:
            assert self.save_freq > 0, "save_freq must be positive when set"

    # =========================== Constructors ===========================

    @classmethod
    def local(
        cls,
        out_dir: Path,
        *,
        train_log_freq: int,
        eval_freq: int,
        slow_eval_freq: int,
        n_eval_steps: int,
        save_freq: int | None = None,
        slow_eval_on_first_step: bool = True,
    ) -> "RunSink":
        """Notebook / script: local files only, no wandb."""
        if not is_main_process():
            return cls._silent_noop(
                train_log_freq=train_log_freq,
                eval_freq=eval_freq,
                slow_eval_freq=slow_eval_freq,
                n_eval_steps=n_eval_steps,
                save_freq=save_freq,
                slow_eval_on_first_step=slow_eval_on_first_step,
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            out_dir=out_dir,
            train_log_freq=train_log_freq,
            eval_freq=eval_freq,
            slow_eval_freq=slow_eval_freq,
            n_eval_steps=n_eval_steps,
            slow_eval_on_first_step=slow_eval_on_first_step,
            save_freq=save_freq,
            _wandb_active=False,
        )

    @classmethod
    def with_wandb(
        cls,
        out_dir: Path,
        *,
        project: str,
        run_id: str,
        train_log_freq: int,
        eval_freq: int,
        slow_eval_freq: int,
        n_eval_steps: int,
        save_freq: int | None = None,
        slow_eval_on_first_step: bool = True,
        name: str | None = None,
        tags: list[str] | None = None,
        configs: dict[str, BaseConfig] | None = None,
        view_meta: dict[str, Any] | None = None,
    ) -> "RunSink":
        """Notebook / script: local files + wandb."""
        if not is_main_process():
            return cls._silent_noop(
                train_log_freq=train_log_freq,
                eval_freq=eval_freq,
                slow_eval_freq=slow_eval_freq,
                n_eval_steps=n_eval_steps,
                save_freq=save_freq,
                slow_eval_on_first_step=slow_eval_on_first_step,
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        init_wandb(
            project,
            run_id,
            configs=configs or {},
            name=name,
            tags=tags,
            view_meta=view_meta,
        )
        return cls(
            out_dir=out_dir,
            train_log_freq=train_log_freq,
            eval_freq=eval_freq,
            slow_eval_freq=slow_eval_freq,
            n_eval_steps=n_eval_steps,
            slow_eval_on_first_step=slow_eval_on_first_step,
            save_freq=save_freq,
            _wandb_active=True,
        )

    @classmethod
    def silent(
        cls,
        *,
        train_log_freq: int = 50,
        eval_freq: int = 100,
        slow_eval_freq: int = 100,
        n_eval_steps: int = 1,
        save_freq: int | None = None,
        slow_eval_on_first_step: bool = True,
    ) -> "RunSink":
        """No persistence, no wandb. Useful for tests / quick interactive runs."""
        return cls._silent_noop(
            train_log_freq=train_log_freq,
            eval_freq=eval_freq,
            slow_eval_freq=slow_eval_freq,
            n_eval_steps=n_eval_steps,
            save_freq=save_freq,
            slow_eval_on_first_step=slow_eval_on_first_step,
        )

    @classmethod
    def _silent_noop(
        cls,
        *,
        train_log_freq: int,
        eval_freq: int,
        slow_eval_freq: int,
        n_eval_steps: int,
        save_freq: int | None,
        slow_eval_on_first_step: bool,
    ) -> "RunSink":
        return cls(
            out_dir=None,
            train_log_freq=train_log_freq,
            eval_freq=eval_freq,
            slow_eval_freq=slow_eval_freq,
            n_eval_steps=n_eval_steps,
            slow_eval_on_first_step=slow_eval_on_first_step,
            save_freq=save_freq,
            _wandb_active=False,
        )

    # =========================== Cadence gating ===========================

    def should_log_train(self, step: int) -> bool:
        return step % self.train_log_freq == 0

    def should_eval(self, step: int) -> bool:
        return step % self.eval_freq == 0

    def should_run_slow_eval(self, step: int) -> bool:
        if step == 0:
            return self.slow_eval_on_first_step
        return step % self.slow_eval_freq == 0

    def should_save(self, step: int, *, total_steps: int) -> bool:
        if step == total_steps:
            return True
        if self.save_freq is None or step == 0:
            return False
        return step % self.save_freq == 0

    # =========================== Output API ===========================

    def log(
        self,
        metrics: dict[str, Any],
        *,
        step: int,
        section: str | None = None,
    ) -> None:
        """Emit a flat metrics dict to disk (if `out_dir`) and to wandb (if active).

        `section` is prefixed to every W&B key (`"eval"` → `"eval/loss/total"`);
        local logs use the unsectioned keys.
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
        """Print lines to stderr via `tqdm.write`. No-op on non-main ranks."""
        if not is_main_process():
            return
        for line in lines:
            tqdm.write(line)

    def checkpoint(self, state_dict: dict[str, Any], *, step: int) -> None:
        """Save `state_dict` to `{out_dir}/model_{step}.pth` + push to wandb."""
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
    """Wrap non-wandb-native types (e.g. `PIL.Image`) for `wandb.log`."""
    if isinstance(v, Image.Image):
        return wandb.Image(v)
    return v
