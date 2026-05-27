"""Concrete `RunSink` for the in-repo experiments: local files + optional wandb.

Non-main ranks transparently get a no-op sink regardless of which constructor is used.
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
from param_decomp.training_state import TrainingState
from param_decomp_lab.infra.run_files import save_file
from param_decomp_lab.infra.wandb import init_wandb, try_wandb


def _local_log(data: dict[str, Any], step: int, out_dir: Path) -> None:
    """Write a step's metrics, figures, and custom charts to disk.

    PIL images go to `{out_dir}/figures/<key>_<step>.png`; `wandb.plot.CustomChart`
    payloads go to `{out_dir}/figures/<key>_<step>.json`; everything else is appended
    as one JSON line to `{out_dir}/metrics.jsonl`.
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
    """Construct via `local`, `with_wandb`, or `silent` (not the dataclass directly).

    Non-main ranks always get a no-op handle. `out_dir=None` disables disk output.
    """

    out_dir: Path | None
    _wandb_active: bool
    keep_last_n_checkpoints: int | None = None

    # =========================== Constructors ===========================

    @classmethod
    def local(cls, out_dir: Path, *, keep_last_n_checkpoints: int | None = None) -> "RunSink":
        """Sink that writes to local files only (no wandb)."""
        if not is_main_process():
            return cls(out_dir=None, _wandb_active=False)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Train+eval logs saved to directory: {out_dir}")
        return cls(
            out_dir=out_dir,
            _wandb_active=False,
            keep_last_n_checkpoints=keep_last_n_checkpoints,
        )

    @classmethod
    def with_wandb(
        cls,
        out_dir: Path,
        *,
        project: str,
        run_id: str,
        config: BaseConfig,
        entity: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
        view_meta: dict[str, Any] | None = None,
        keep_last_n_checkpoints: int | None = None,
    ) -> "RunSink":
        """Sink that writes to local files and a wandb run.

        Initializes wandb on the main rank via `init_wandb`; non-main ranks return a
        silent no-op.
        """
        if not is_main_process():
            return cls(out_dir=None, _wandb_active=False)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Train+eval logs saved to directory: {out_dir}")
        init_wandb(
            project,
            run_id,
            config,
            entity=entity,
            name=name,
            tags=tags,
            group=group,
            view_meta=view_meta,
        )
        return cls(
            out_dir=out_dir,
            _wandb_active=True,
            keep_last_n_checkpoints=keep_last_n_checkpoints,
        )

    @classmethod
    def silent(cls) -> "RunSink":
        """No-op sink for tests and quick interactive runs."""
        return cls(out_dir=None, _wandb_active=False)

    # =========================== Output API ===========================

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Emit a flat metrics dict to disk and/or wandb.

        Values may be scalars, PIL images, or `wandb.plot.CustomChart` payloads.
        """
        if self.out_dir is not None:
            _local_log(metrics, step, self.out_dir)
        if self._wandb_active:
            try_wandb(wandb.log, {k: _wandb_value(v) for k, v in metrics.items()}, step=step)

    def console(self, *lines: str) -> None:
        """Print lines to stderr via `tqdm.write`. No-op on non-main ranks."""
        if not is_main_process():
            return
        for line in lines:
            tqdm.write(line)

    def checkpoint(self, snapshot: TrainingState) -> None:
        """Save the snapshot as two files: `model_<step>.pth` + `training_<step>.pth`.

        `model_<step>.pth` is just the component-model state dict — the artifact
        downstream tools (`SavedRun.load_model`, postprocessing) consume.
        `training_<step>.pth` is the full `TrainingState` (configs, optimizer
        state, metric states, step) needed for resumption.

        No-op when `out_dir is None` (silent sink / non-main rank); wandb upload
        only when wandb is active. Prunes older (model, training) pairs after the
        write when ``keep_last_n_checkpoints`` is set.
        """
        if self.out_dir is None:
            return
        model_path = self.out_dir / f"model_{snapshot.step}.pth"
        save_file(snapshot.component_model, model_path)
        training_path = self.out_dir / f"training_{snapshot.step}.pth"
        save_file(snapshot, training_path)
        logger.info(f"Saved checkpoint to {model_path} (+ {training_path.name})")
        if self._wandb_active:
            try_wandb(wandb.save, str(model_path), base_path=str(self.out_dir), policy="now")
            try_wandb(wandb.save, str(training_path), base_path=str(self.out_dir), policy="now")
        if self.keep_last_n_checkpoints is not None:
            _prune_old_checkpoints(self.out_dir, keep_last_n=self.keep_last_n_checkpoints)

    def finish(self) -> None:
        """End-of-run cleanup."""
        if self._wandb_active and wandb.run is not None:
            wandb.finish()


def _wandb_value(v: Any) -> Any:
    """Wrap non-wandb-native types (e.g. `PIL.Image`) for `wandb.log`."""
    if isinstance(v, Image.Image):
        return wandb.Image(v)
    return v


def _prune_old_checkpoints(out_dir: Path, *, keep_last_n: int) -> None:
    """Delete (``model_<step>.pth``, ``training_<step>.pth``) pairs except for the
    top ``keep_last_n`` by step number.

    A "pair" is the two files written together by :meth:`RunSink.checkpoint`; we
    glob each independently and prune by step rather than assuming both must
    exist, since a future caller might write only one of them.
    """

    def steps(prefix: str) -> list[int]:
        out: list[int] = []
        for p in out_dir.glob(f"{prefix}_*.pth"):
            try:
                out.append(int(p.stem.removeprefix(f"{prefix}_")))
            except ValueError:
                continue
        return out

    all_steps = sorted(set(steps("model")) | set(steps("training")))
    if len(all_steps) <= keep_last_n:
        return
    to_delete = all_steps[: len(all_steps) - keep_last_n]
    for step in to_delete:
        for prefix in ("model", "training"):
            path = out_dir / f"{prefix}_{step}.pth"
            if path.is_file():
                path.unlink()
