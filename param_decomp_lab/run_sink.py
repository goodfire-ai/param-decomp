"""Concrete `RunSink` used by the in-repo experiments and lab tooling.

Owns two things: **where output goes** (local files + optional wandb) and
**console output**. Timing — when the trainer emits — lives elsewhere:
`param_decomp.configs.Cadence` (train-log + checkpoint periods) and
`param_decomp.optimize.EvalLoop` (eval period).

Three constructors:

    sink = RunSink.local(out_dir)
    sink = RunSink.with_wandb(out_dir, project=..., run_id=..., ...)
    sink = RunSink.silent()                                # tests / quick checks

Non-main ranks transparently get a no-op sink (``out_dir=None``, wandb inactive)
regardless of which constructor is called. The trainer never has to check rank.

The core trainer (`param_decomp.optimize`) accepts anything that satisfies the
`param_decomp.RunSink` Protocol; this class is the lab's implementation of that
contract.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb
from PIL import Image
from tqdm import tqdm

from param_decomp.base_config import BaseConfig
from param_decomp.distributed import is_main_process
from param_decomp.log import logger
from param_decomp_lab.infra.run_files import save_file
from param_decomp_lab.infra.wandb import init_wandb, try_wandb


def _local_log(data: dict[str, Any], step: int, out_dir: Path) -> None:
    """Write a step's metrics, figures, and custom charts to disk.

    PIL images go to ``{out_dir}/figures/<key>_<step>.png``; ``wandb.plot.CustomChart``
    payloads go to ``{out_dir}/figures/<key>_<step>.json``; everything else is appended
    as one JSON line to ``{out_dir}/metrics.jsonl``.
    """
    metrics_file = out_dir / "metrics.jsonl"
    metrics_file.touch(exist_ok=True)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    metrics_without_images: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, Image.Image):
            filename = f"{k.replace('/', '_')}_{step}.png"
            v.save(fig_dir / filename)
            logger.info(f"Saved figure {k} to {fig_dir / filename}")
        elif isinstance(v, wandb.plot.CustomChart):
            json_path = fig_dir / f"{k.replace('/', '_')}_{step}.json"
            payload = {"columns": list(v.table.columns), "data": list(v.table.data), "step": step}
            with open(json_path, "w") as f:
                json.dump(payload, f, default=str)
            logger.info(f"Saved custom chart data {k} to {json_path}")
        else:
            metrics_without_images[k] = v

    with open(metrics_file, "a") as f:
        f.write(json.dumps({"step": step, **metrics_without_images}) + "\n")


@dataclass(frozen=True)
class RunSink:
    """Side-effect sink for a training run.

    Construct via one of the classmethods (`local`, `with_wandb`, `silent`) rather than
    the dataclass directly. Non-main ranks always get a no-op handle regardless of which
    constructor is called.

    Attributes:
        out_dir: Local directory for metrics/figures/checkpoints; ``None`` disables disk
            output (silent sink or non-main rank).
        _wandb_active: Whether wandb logging is live for this process.
    """

    out_dir: Path | None
    _wandb_active: bool

    # =========================== Constructors ===========================

    @classmethod
    def local(cls, out_dir: Path) -> "RunSink":
        """Build a sink that writes to local files only (no wandb).

        Args:
            out_dir: Directory to create and write all run artifacts into.

        Returns:
            A sink writing to ``out_dir`` on the main rank, or a silent no-op on others.
        """
        if not is_main_process():
            return cls._silent_noop()
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Train+eval logs saved to directory: {out_dir}")
        return cls(out_dir=out_dir, _wandb_active=False)

    @classmethod
    def with_wandb(
        cls,
        out_dir: Path,
        *,
        project: str,
        run_id: str,
        name: str | None = None,
        tags: list[str] | None = None,
        configs: dict[str, BaseConfig] | None = None,
        view_meta: dict[str, Any] | None = None,
    ) -> "RunSink":
        """Build a sink that writes to local files and a wandb run.

        Initializes wandb on the main rank via `init_wandb` before returning. Non-main
        ranks skip both disk and wandb init and return a silent no-op.

        Args:
            out_dir: Directory to create and write all run artifacts into.
            project: wandb project name.
            run_id: Stable identifier for the wandb run.
            name: Display name for the wandb run.
            tags: Tags to attach to the wandb run.
            configs: Named pydantic configs to log under wandb config.
            view_meta: Extra metadata passed to `init_wandb` for the in-house viewer.

        Returns:
            A sink writing to ``out_dir`` and wandb on the main rank, a silent no-op
            on others.
        """
        if not is_main_process():
            return cls._silent_noop()
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Train+eval logs saved to directory: {out_dir}")
        init_wandb(
            project,
            run_id,
            configs=configs or {},
            name=name,
            tags=tags,
            view_meta=view_meta,
        )
        return cls(out_dir=out_dir, _wandb_active=True)

    @classmethod
    def silent(cls) -> "RunSink":
        """Build a sink that drops everything (no disk, no wandb).

        Returns:
            A no-op sink suitable for tests and quick interactive runs.
        """
        return cls._silent_noop()

    @classmethod
    def _silent_noop(cls) -> "RunSink":
        return cls(out_dir=None, _wandb_active=False)

    # =========================== Output API ===========================

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Emit a flat metrics dict to disk and/or wandb.

        Args:
            metrics: Pre-namespaced metric keys (``train/...``, ``eval/...``) to values.
                Values may be scalars, PIL images, or ``wandb.plot.CustomChart`` payloads.
            step: Training step the values were measured at.
        """
        if self.out_dir is not None:
            _local_log(metrics, step, self.out_dir)
        if self._wandb_active:
            try_wandb(wandb.log, {k: _wandb_value(v) for k, v in metrics.items()}, step=step)

    def console(self, *lines: str) -> None:
        """Print lines to stderr via `tqdm.write`, one per line.

        No-op on non-main ranks.

        Args:
            *lines: Lines to print.
        """
        if not is_main_process():
            return
        for line in lines:
            tqdm.write(line)

    def checkpoint(self, state_dict: dict[str, Any], step: int) -> None:
        """Save `state_dict` to ``{out_dir}/model_{step}.pth`` and push to wandb.

        No-op when ``out_dir`` is ``None``. Wandb upload happens only when wandb is
        active for this process.

        Args:
            state_dict: Tensor state dict to serialize via `save_file`.
            step: Training step used in the checkpoint filename.
        """
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
