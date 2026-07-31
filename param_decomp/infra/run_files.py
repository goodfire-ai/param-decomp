"""Run directories, IDs, snapshots, and on-disk file resolution (incl. W&B cache)."""

import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import wandb
from wandb.apis.public import Run as WandbRun

from param_decomp.core.log import logger
from param_decomp.infra.paths import ModelPath
from param_decomp.infra.wandb import (
    download_wandb_file,
    fetch_latest_checkpoint_name,
    fetch_latest_wandb_checkpoint,
    parse_wandb_run_path,
)

RunType = Literal[
    "param_decomp", "train", "clustering/runs", "clustering/ensembles", "clustering/harvests"
]

RUN_TYPE_ABBREVIATIONS: Final[dict[RunType, str]] = {
    "param_decomp": "p",
    "train": "t",
    "clustering/runs": "c",
    "clustering/ensembles": "e",
    "clustering/harvests": "ch",
}


def generate_run_id(run_type: RunType) -> str:
    """Generate a unique run identifier.

    Format: `{type_abbr}-{random_hex}`
    """
    type_abbr = RUN_TYPE_ABBREVIATIONS[run_type]
    return f"{type_abbr}-{secrets.token_hex(4)}"


def fetch_latest_local_checkpoint(run_dir: Path, prefix: str | None = None) -> Path:
    """Fetch the latest checkpoint from a local run directory."""
    filenames = [file.name for file in run_dir.iterdir() if file.name.endswith(".pth")]
    latest_checkpoint_name = fetch_latest_checkpoint_name(filenames, prefix)
    return run_dir / latest_checkpoint_name


@dataclass
class RunFiles:
    """Resolved local paths for a saved run."""

    config_path: Path
    checkpoint_path: Path
    extras: dict[str, Path] = field(default_factory=dict)


def _wandb_cache_dir(data_root: Path, run_id: str) -> Path:
    return data_root / "runs" / run_id


def resolve_run_files(
    path: ModelPath,
    *,
    data_root: Path,
    config_filename: str,
    checkpoint_filename: str | None = None,
    checkpoint_prefix: str | None = None,
    extras_from_config_path: Callable[[Path], list[str]] = lambda _: [],
) -> RunFiles:
    """Locate a run's files locally, downloading from W&B if needed.

    Exactly one of `checkpoint_filename` or `checkpoint_prefix` must be given.
    `data_root` locates the W&B download cache (`<data_root>/runs/<run_id>`).
    `extras_from_config_path` is called with the resolved config path to determine which
    additional files belong to the run (e.g. artifacts whose names live inside the config).
    """
    assert (checkpoint_filename is None) != (checkpoint_prefix is None), (
        "Exactly one of checkpoint_filename or checkpoint_prefix is required"
    )

    try:
        entity, project, run_id = parse_wandb_run_path(str(path))
    except ValueError:
        return _resolve_local_run_files(
            Path(path),
            config_filename=config_filename,
            checkpoint_filename=checkpoint_filename,
            checkpoint_prefix=checkpoint_prefix,
            extras_from_config_path=extras_from_config_path,
        )

    wandb_path = f"{entity}/{project}/{run_id}"
    run_dir = _wandb_cache_dir(data_root, run_id)

    if run_dir.exists():
        logger.info(f"Loading run from {run_dir}")
        try:
            files = _resolve_local_run_files(
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

    return _download_run_files_from_wandb(
        wandb_path,
        data_root=data_root,
        config_filename=config_filename,
        checkpoint_filename=checkpoint_filename,
        checkpoint_prefix=checkpoint_prefix,
        extras_from_config_path=extras_from_config_path,
    )


def resolve_config_path(path: ModelPath, *, data_root: Path, config_filename: str) -> Path:
    """Locate just a run's config file, without resolving or downloading checkpoints."""
    try:
        entity, project, run_id = parse_wandb_run_path(str(path))
    except ValueError:
        path_obj = Path(path)
        return (path_obj if path_obj.is_dir() else path_obj.parent) / config_filename

    run_dir = _wandb_cache_dir(data_root, run_id)
    config_path = run_dir / config_filename
    if config_path.exists():
        return config_path

    logger.info(f"Downloading config from wandb: {entity}/{project}/{run_id}")
    api = wandb.Api()
    run: WandbRun = api.run(f"{entity}/{project}/{run_id}")
    return download_wandb_file(run, run_dir, config_filename)


def _resolve_local_run_files(
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


def _download_run_files_from_wandb(
    wandb_path: str,
    *,
    data_root: Path,
    config_filename: str,
    checkpoint_filename: str | None,
    checkpoint_prefix: str | None,
    extras_from_config_path: Callable[[Path], list[str]],
) -> RunFiles:
    api = wandb.Api()
    run: WandbRun = api.run(wandb_path)
    _entity, _project, run_id = parse_wandb_run_path(wandb_path)
    run_dir = _wandb_cache_dir(data_root, run_id)

    config_path = download_wandb_file(run, run_dir, config_filename)
    if checkpoint_filename is not None:
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint_filename)
    else:
        checkpoint = fetch_latest_wandb_checkpoint(run, prefix=checkpoint_prefix)
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
    extras = {
        name: download_wandb_file(run, run_dir, name)
        for name in extras_from_config_path(config_path)
    }
    return RunFiles(config_path=config_path, checkpoint_path=checkpoint_path, extras=extras)
