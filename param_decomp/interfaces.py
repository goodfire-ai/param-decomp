from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import torch.nn as nn
import wandb
import yaml
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
class RunInfo[T]:
    """Base class for run information from a training run of a target model or PD.

    Subclasses should set the following class variables:
        - config_class: The pydantic config class to instantiate
        - config_filename: Name of the config file (e.g., "final_config.yaml")
        - checkpoint_filename: Fixed checkpoint filename (e.g., "tms.pth"), or None to use prefix
        - checkpoint_prefix: Prefix for fetch_latest when checkpoint_filename is None (e.g., "model")
        - extra_files: List of additional files to download (e.g., ["label_coeffs.json"])

    Subclasses with extra_files should override _process_extra_files to handle them.
    """

    checkpoint_path: Path
    config: T

    # Subclasses must set these
    config_class: ClassVar[type]
    config_filename: ClassVar[str]

    # Set one of these for checkpoint resolution
    checkpoint_filename: ClassVar[str | None] = None  # Fixed filename
    checkpoint_prefix: ClassVar[str | None] = None  # For fetching the latest checkpoint

    # Additional files to download
    extra_files: ClassVar[list[str]] = []

    @classmethod
    def _process_extra_files(
        cls, _file_paths: dict[str, Path], _init_kwargs: dict[str, Any]
    ) -> None:
        """Hook for subclasses that need to load extra files into init kwargs.

        Args:
            _file_paths: Dict mapping filename to local path for all extra_files
            _init_kwargs: Dict that will be passed to cls(**init_kwargs). Modify in place.
        """
        pass

    @classmethod
    def from_path(cls, path: ModelPath) -> Self:
        """Load run info from wandb or local path.

        Args:
            path: The path to local checkpoint or wandb project. If a wandb project, format must be
                `wandb:<entity>/<project>/<run_id>` or `wandb:<entity>/<project>/runs/<run_id>`.
                If `api.entity` is set (e.g. via setting WANDB_ENTITY in .env), <entity> can be
                omitted, and if `api.project` is set, <project> can be omitted. If local path,
                assumes config file is in the same directory as the checkpoint.
        """
        try:
            entity, project, run_id = parse_wandb_run_path(str(path))
        except ValueError:
            # Direct path to checkpoint file
            file_paths = cls._resolve_from_checkpoint_path(Path(path))
        else:
            # Wandb path - check cache first
            run_dir = PARAM_DECOMP_OUT_DIR / "runs" / f"{project}-{run_id}"
            if run_dir.exists():
                logger.info(f"Loading run from {run_dir}")
                file_paths = cls._resolve_from_run_dir(run_dir)
            else:
                logger.info(f"Downloading run from wandb: {entity}/{project}/{run_id}")
                file_paths = cls._download_from_wandb(f"{entity}/{project}/{run_id}")

        with open(file_paths["config"]) as f:
            config = cls.config_class(**yaml.safe_load(f))

        init_kwargs: dict[str, Any] = {
            "checkpoint_path": file_paths["checkpoint"],
            "config": config,
        }
        cls._process_extra_files(file_paths, init_kwargs)
        return cls(**init_kwargs)

    @classmethod
    def _resolve_from_checkpoint_path(cls, checkpoint_path: Path) -> dict[str, Path]:
        """Resolve file paths when user provides direct checkpoint path."""
        parent = checkpoint_path.parent
        return {
            "config": parent / cls.config_filename,
            "checkpoint": checkpoint_path,
            **{f: parent / f for f in cls.extra_files},
        }

    @classmethod
    def _resolve_from_run_dir(cls, run_dir: Path) -> dict[str, Path]:
        """Resolve file paths from a wandb run directory (cached or downloaded)."""
        if cls.checkpoint_filename:
            checkpoint = run_dir / cls.checkpoint_filename
        else:
            assert cls.checkpoint_prefix is not None, (
                "Must set either checkpoint_filename or checkpoint_prefix"
            )
            checkpoint = fetch_latest_local_checkpoint(run_dir, prefix=cls.checkpoint_prefix)
        return {
            "config": run_dir / cls.config_filename,
            "checkpoint": checkpoint,
            **{f: run_dir / f for f in cls.extra_files},
        }

    @classmethod
    def _download_from_wandb(cls, wandb_path: str) -> dict[str, Path]:
        """Download files from wandb and return local paths."""
        api = wandb.Api()
        run: Run = api.run(wandb_path)
        run_dir = fetch_wandb_run_dir(run.id)

        checkpoint = fetch_latest_wandb_checkpoint(
            run, prefix=cls.checkpoint_prefix if cls.checkpoint_prefix else None
        )

        return {
            "config": download_wandb_file(run, run_dir, cls.config_filename),
            "checkpoint": download_wandb_file(run, run_dir, checkpoint.name),
            **{f: download_wandb_file(run, run_dir, f) for f in cls.extra_files},
        }


class LoadableModule(nn.Module, ABC):
    """Base class for nn.Modules that can be loaded from a local path or wandb run id."""

    @classmethod
    @abstractmethod
    def from_pretrained(cls, _path: ModelPath) -> "LoadableModule":
        """Load a pretrained model from a local path or wandb run id."""
        raise NotImplementedError("Subclasses must implement from_pretrained method.")

    @classmethod
    @abstractmethod
    def from_run_info(cls, _run_info: RunInfo[Any]) -> "LoadableModule":
        """Load a pretrained model from a run info object."""
        raise NotImplementedError("Subclasses must implement from_run_info method.")
