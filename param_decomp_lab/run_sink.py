"""Concrete `RunSink` classes used by the in-repo experiments and lab tooling.

Two pool-specific sinks (`OnePoolSink`, `ThreePoolSink`) share a private base
(`_LabSinkBase`) for the local-files / wandb / console / log plumbing. The
subclasses differ only in `checkpoint`'s typed parameter — 1-pool takes
`TrainingState`, 3-pool takes `ThreePoolTrainingState`. Each delegates to a
shared `_persist` for the actual save.

Three constructors on each:

    sink = OnePoolSink.local(out_dir)
    sink = OnePoolSink.with_wandb(out_dir, project=..., run_id=..., ...)
    sink = OnePoolSink.silent()                                # tests / quick checks

(same shape for `ThreePoolSink`). Non-main ranks transparently get a no-op
sink regardless of which constructor is called.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import wandb
from PIL import Image
from tqdm import tqdm

from param_decomp.base_config import BaseConfig
from param_decomp.distributed import is_main_process
from param_decomp.log import logger
from param_decomp.training_state import ThreePoolTrainingState, TrainingState
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

    @classmethod
    def local(cls, out_dir: Path) -> Self:
        """Sink that writes to local files only (no wandb)."""
        if not is_main_process():
            return cls(out_dir=None, _wandb_active=False)
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
        config: BaseConfig,
        entity: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
        view_meta: dict[str, Any] | None = None,
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
            entity=entity,
            name=name,
            tags=tags,
            group=group,
            view_meta=view_meta,
        )
        return cls(out_dir=out_dir, _wandb_active=True)

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

    def _persist(self, snapshot: TrainingState | ThreePoolTrainingState) -> None:
        """Save the snapshot as `model_<step>.pth` + `training_<step>.pth`.

        `model_<step>.pth` is just the component-model state dict — the
        artifact downstream tools (`SavedRun.load_model`, postprocessing)
        consume. `training_<step>.pth` is the full canonical state. No-op
        on a silent / non-main-rank sink.
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


@dataclass(frozen=True)
class OnePoolSink(_LabSinkBase):
    """Lab sink for 1-pool runs (satisfies `OnePoolRunSink`)."""

    def checkpoint(self, snapshot: TrainingState) -> None:
        self._persist(snapshot)


@dataclass(frozen=True)
class ThreePoolSink(_LabSinkBase):
    """Lab sink for 3-pool runs (satisfies `ThreePoolRunSink`)."""

    def checkpoint(self, snapshot: ThreePoolTrainingState) -> None:
        self._persist(snapshot)


def _wandb_value(v: Any) -> Any:
    """Wrap non-wandb-native types (e.g. `PIL.Image`) for `wandb.log`."""
    if isinstance(v, Image.Image):
        return wandb.Image(v)
    return v
