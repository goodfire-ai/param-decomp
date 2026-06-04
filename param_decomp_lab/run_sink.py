"""Concrete `RunSink` classes used by the in-repo experiments and lab tooling.

Two pool-specific sinks (`OnePoolSink`, `ThreePoolSink`) share a private base
(`_LabSinkBase`) for the local-files / wandb / console / log plumbing. They
differ in how a checkpoint is persisted: `OnePoolSink.checkpoint` writes the
`TrainingState` synchronously via `_persist`; `ThreePoolSink.checkpoint_written`
persists nothing (the trainer already wrote per-rank partials) and only fires
the async consolidation+eval job via the `on_save` hook.

Three constructors on each:

    sink = OnePoolSink.local(out_dir)
    sink = OnePoolSink.with_wandb(out_dir, project=..., run_id=..., ...)
    sink = OnePoolSink.silent()                                # tests / quick checks

(same shape for `ThreePoolSink`). Non-main ranks transparently get a no-op
sink regardless of which constructor is called.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

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

    PIL images go to ``{out_dir}/figures/<key>_<step>.png``; ``wandb.plot.CustomChart``
    payloads go to ``{out_dir}/figures/<key>_<step>.json``; everything else is
    appended as one JSON line to ``{out_dir}/metrics.jsonl``.
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
class _LabSinkBase:
    """Shared local-files + wandb plumbing for `OnePoolSink` / `ThreePoolSink`.

    Pool-specific subclasses inherit constructors and the log / console /
    finish / `_persist` helpers, and add typed `checkpoint` methods.
    """

    out_dir: Path | None
    _wandb_active: bool
    keep_last_n_checkpoints: int | None = None
    on_save: Callable[[int], None] | None = None
    """Rank-0 hook called with `step` after each checkpoint write. Used to fire
    async slow-eval SLURM jobs per checkpoint; no-op on silent sinks."""

    @classmethod
    def local(
        cls,
        out_dir: Path,
        *,
        keep_last_n_checkpoints: int | None = None,
        on_save: Callable[[int], None] | None = None,
    ) -> Self:
        """Sink that writes to local files only (no wandb)."""
        if not is_main_process():
            return cls(out_dir=None, _wandb_active=False)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Train+eval logs saved to directory: {out_dir}")
        return cls(
            out_dir=out_dir,
            _wandb_active=False,
            keep_last_n_checkpoints=keep_last_n_checkpoints,
            on_save=on_save,
        )

    @classmethod
    def with_wandb(
        cls,
        out_dir: Path,
        *,
        project: str,
        run_id: str,
        config: BaseConfig,
        metric_short_names: dict[str, str],
        entity: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
        view_meta: dict[str, Any] | None = None,
        keep_last_n_checkpoints: int | None = None,
        on_save: Callable[[int], None] | None = None,
    ) -> Self:
        """Sink that writes to local files + a wandb run. Non-main ranks are silent."""
        if not is_main_process():
            return cls(out_dir=None, _wandb_active=False)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Train+eval logs saved to directory: {out_dir}")
        init_wandb(
            project,
            run_id,
            config,
            metric_short_names,
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
            on_save=on_save,
        )

    @classmethod
    def silent(cls) -> Self:
        """No-op sink for tests and quick interactive runs."""
        return cls(out_dir=None, _wandb_active=False)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Emit a flat metrics dict to disk and/or wandb."""
        if self.out_dir is not None:
            _local_log(metrics, step, self.out_dir)
        if self._wandb_active:
            try_wandb(wandb.log, {k: _wandb_value(v) for k, v in metrics.items()}, step=step)

    def console(self, *lines: str) -> None:
        """Print lines via `tqdm.write`. No-op on non-main ranks."""
        if not is_main_process():
            return
        for line in lines:
            tqdm.write(line)

    def finish(self) -> None:
        """End-of-run cleanup."""
        if self._wandb_active and wandb.run is not None:
            wandb.finish()

    def _persist(self, snapshot: TrainingState, *, final: bool) -> None:
        """Save the snapshot as `model_<step>.pth` + `training_<step>.pth`.

        `model_<step>.pth` is just the component-model state dict — the artifact
        downstream tools (`SavedRun.load_model`, postprocessing) consume.
        `training_<step>.pth` is the full canonical state (model + optimizer
        state + PPGD sources), used only for resumption. No-op on a silent /
        non-main-rank sink. Prunes older (model, training) pairs after the
        write when ``keep_last_n_checkpoints`` is set.

        Wandb cloud upload only happens on ``final=True`` and is limited to
        ``model_<step>.pth`` — `training_<step>.pth` is multi-GB at XL and
        only useful for in-cluster resumption (which reads from local FS).
        """
        if self.out_dir is None:
            return
        model_path = self.out_dir / f"model_{snapshot.step}.pth"
        save_file(snapshot.component_model, model_path)
        training_path = self.out_dir / f"training_{snapshot.step}.pth"
        save_file(snapshot, training_path)
        logger.info(f"Saved checkpoint to {model_path} (+ {training_path.name})")
        if final and self._wandb_active:
            try_wandb(wandb.save, str(model_path), base_path=str(self.out_dir), policy="now")
        if self.keep_last_n_checkpoints is not None:
            _prune_old_checkpoints(self.out_dir, keep_last_n=self.keep_last_n_checkpoints)
        if self.on_save is not None:
            self.on_save(snapshot.step)


@dataclass(frozen=True)
class OnePoolSink(_LabSinkBase):
    """Lab sink for 1-pool runs (satisfies `OnePoolRunSink`)."""

    def checkpoint(self, snapshot: TrainingState, *, final: bool) -> None:
        self._persist(snapshot, final=final)


@dataclass(frozen=True)
class ThreePoolSink(_LabSinkBase):
    """Lab sink for 3-pool runs (satisfies `ThreePoolRunSink`).

    Persists nothing itself: the trainer has already written self-contained
    per-rank partials to the scratch dir. This only fires the rank-0 `on_save`
    hook, which submits the async job that consolidates those partials into
    ``model_<step>.pth`` + ``training_<step>.pth`` and runs the slow eval.
    """

    def checkpoint_written(self, step: int, *, final: bool) -> None:
        del final  # the async consolidation path is identical for the final save
        if self.out_dir is None:
            return
        if self.on_save is not None:
            self.on_save(step)


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
