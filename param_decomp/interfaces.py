from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch.nn as nn
import wandb
from wandb.apis.public import Run

from param_decomp.log import logger
from param_decomp.param_decomp_types import ModelPath
from param_decomp.settings import PARAM_DECOMP_OUT_DIR
from param_decomp.utils.general_utils import fetch_latest_local_checkpoint
from param_decomp.utils.wandb_utils import (
    download_wandb_file,
    fetch_latest_wandb_checkpoint,
    fetch_wandb_run_dir,
    parse_wandb_run_path,
)


@dataclass
class RunFiles:
    """Resolved local paths for a saved run."""

    config_path: Path
    checkpoint_path: Path
    extras: dict[str, Path] = field(default_factory=dict)


def resolve_run_files(
    path: ModelPath,
    *,
    config_filename: str,
    checkpoint_filename: str | None = None,
    checkpoint_prefix: str | None = None,
    extras_from_config_path: Callable[[Path], list[str]] = lambda _: [],
) -> RunFiles:
    """Locate a run's files locally, downloading from wandb if needed.

    Exactly one of `checkpoint_filename` or `checkpoint_prefix` must be given.
    `extras_from_config_path` is called with the resolved config path to determine which
    additional files belong to the run (e.g. artifacts whose names live inside the manifest).
    """
    assert (checkpoint_filename is None) != (checkpoint_prefix is None), (
        "Exactly one of checkpoint_filename or checkpoint_prefix is required"
    )

    try:
        entity, project, run_id = parse_wandb_run_path(str(path))
    except ValueError:
        return _resolve_local(
            Path(path),
            config_filename=config_filename,
            checkpoint_filename=checkpoint_filename,
            checkpoint_prefix=checkpoint_prefix,
            extras_from_config_path=extras_from_config_path,
        )

    wandb_path = f"{entity}/{project}/{run_id}"
    run_dir = PARAM_DECOMP_OUT_DIR / "runs" / f"{project}-{run_id}"

    if run_dir.exists():
        logger.info(f"Loading run from {run_dir}")
        try:
            files = _resolve_local(
                run_dir,
                config_filename=config_filename,
                checkpoint_filename=checkpoint_filename,
                checkpoint_prefix=checkpoint_prefix,
                extras_from_config_path=extras_from_config_path,
            )
        except (FileNotFoundError, ValueError):
            logger.info(f"Cached run is incomplete, downloading from wandb: {wandb_path}")
        else:
            all_paths = [files.config_path, files.checkpoint_path, *files.extras.values()]
            if all(p.exists() for p in all_paths):
                return files
            logger.info(f"Cached run is missing files, downloading from wandb: {wandb_path}")
    else:
        logger.info(f"Downloading run from wandb: {wandb_path}")

    return _download_from_wandb(
        wandb_path,
        config_filename=config_filename,
        checkpoint_prefix=checkpoint_prefix,
        extras_from_config_path=extras_from_config_path,
    )


def resolve_config_path(path: ModelPath, *, config_filename: str) -> Path:
    """Locate just a run's config file, without resolving or downloading checkpoints."""
    try:
        entity, project, run_id = parse_wandb_run_path(str(path))
    except ValueError:
        path_obj = Path(path)
        return (path_obj if path_obj.is_dir() else path_obj.parent) / config_filename

    run_dir = PARAM_DECOMP_OUT_DIR / "runs" / f"{project}-{run_id}"
    config_path = run_dir / config_filename
    if config_path.exists():
        return config_path

    logger.info(f"Downloading config from wandb: {entity}/{project}/{run_id}")
    api = wandb.Api()
    run: Run = api.run(f"{entity}/{project}/{run_id}")
    run_dir = fetch_wandb_run_dir(run.id)
    return download_wandb_file(run, run_dir, config_filename)


def _resolve_local(
    path: Path,
    *,
    config_filename: str,
    checkpoint_filename: str | None,
    checkpoint_prefix: str | None,
    extras_from_config_path: Callable[[Path], list[str]],
) -> RunFiles:
    if path.is_dir():
        run_dir = path
        if checkpoint_filename is not None:
            checkpoint_path = run_dir / checkpoint_filename
        else:
            assert checkpoint_prefix is not None
            checkpoint_path = fetch_latest_local_checkpoint(run_dir, prefix=checkpoint_prefix)
    else:
        run_dir = path.parent
        checkpoint_path = path
    config_path = run_dir / config_filename
    extras = {name: run_dir / name for name in extras_from_config_path(config_path)}
    return RunFiles(config_path=config_path, checkpoint_path=checkpoint_path, extras=extras)


def _download_from_wandb(
    wandb_path: str,
    *,
    config_filename: str,
    checkpoint_prefix: str | None,
    extras_from_config_path: Callable[[Path], list[str]],
) -> RunFiles:
    api = wandb.Api()
    run: Run = api.run(wandb_path)
    run_dir = fetch_wandb_run_dir(run.id)

    config_path = download_wandb_file(run, run_dir, config_filename)
    checkpoint = fetch_latest_wandb_checkpoint(run, prefix=checkpoint_prefix)
    checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
    extras = {
        name: download_wandb_file(run, run_dir, name)
        for name in extras_from_config_path(config_path)
    }
    return RunFiles(config_path=config_path, checkpoint_path=checkpoint_path, extras=extras)


class LoadableModule(nn.Module, ABC):
    """Base class for nn.Modules that can be loaded from a local path or wandb run id."""

    @classmethod
    @abstractmethod
    def from_pretrained(cls, _path: ModelPath) -> "LoadableModule":
        """Load a pretrained model from a local path or wandb run id."""
        raise NotImplementedError("Subclasses must implement from_pretrained method.")
