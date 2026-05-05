from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb
import yaml
from tokenizers import Tokenizer as HFTokenizer
from transformers import AutoTokenizer

from param_decomp.settings import PARAM_DECOMP_OUT_DIR
from param_decomp.utils.wandb_utils import parse_wandb_run_path


@dataclass
class WandbDownloadedFiles:
    checkpoint: Path
    config: Path
    model_config: Path
    tokenizer: Path | None


def _cache_dir(project: str, run_id: str) -> Path:
    return PARAM_DECOMP_OUT_DIR / "pretrain_cache" / f"{project}-{run_id}"


def _download_wandb_files(entity: str, project: str, run_id: str) -> WandbDownloadedFiles:
    """Download core artifacts for the given W&B run."""
    cache_dir = _cache_dir(project, run_id)
    cache_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    files = list(run.files())

    # Locate config file
    config_file = None
    for f in files:
        if f.name.endswith("final_config.yaml"):
            config_file = f
            break
    if config_file is None:
        raise FileNotFoundError("Could not find 'final_config.yaml' in the W&B run files.")

    model_config_file = None
    for f in files:
        if f.name.endswith("model_config.yaml"):
            model_config_file = f
            break
    if model_config_file is None:
        raise FileNotFoundError("Could not find 'model_config.yaml' in the W&B run files.")

    tokenizer_file = None
    for f in files:
        if f.name.endswith("tokenizer.json"):
            tokenizer_file = f
            break

    # Locate latest checkpoint by step number
    step_re = re.compile(r"^model_step_(\d+)\.pt$")
    ckpt_candidates: list[tuple[int, Any]] = []
    for f in files:
        m = step_re.match(f.name)
        if m:
            ckpt_candidates.append((int(m.group(1)), f))
    if not ckpt_candidates:
        raise FileNotFoundError(
            "Could not find any 'model_step_*.pt' checkpoint in the W&B run files."
        )
    ckpt_candidates.sort(key=lambda t: t[0])
    _, latest_ckpt_file = ckpt_candidates[-1]

    # Skip download if file already exists to avoid race conditions in multi-process contexts
    config_file.download(root=str(cache_dir), exist_ok=True)
    model_config_file.download(root=str(cache_dir), exist_ok=True)
    latest_ckpt_file.download(root=str(cache_dir), exist_ok=True)
    if tokenizer_file is not None:
        tokenizer_file.download(root=str(cache_dir), exist_ok=True)

    return WandbDownloadedFiles(
        checkpoint=cache_dir / latest_ckpt_file.name,
        config=cache_dir / config_file.name,
        model_config=cache_dir / model_config_file.name,
        tokenizer=cache_dir / tokenizer_file.name if tokenizer_file is not None else None,
    )


def _extract_hf_tokenizer_path(config_dict: dict[str, Any]) -> str | None:
    """Extract HF tokenizer path from config dict, returning None if not found."""
    train_dataset_config = config_dict.get("train_dataset_config")
    if not isinstance(train_dataset_config, dict):
        return None
    hf_tok_path = train_dataset_config.get("hf_tokenizer_path")
    return hf_tok_path if isinstance(hf_tok_path, str) else None


@dataclass
class PretrainRunInfo:
    """Run info from training a model with param_decomp.pretrain.

    TODO: Perhaps inherit from RunInfo instead
    """

    checkpoint_path: Path
    config_dict: dict[str, Any]
    model_config_dict: dict[str, Any]
    tokenizer_path: Path | None
    hf_tokenizer_path: str | None

    @classmethod
    def from_path(cls, path: str | Path) -> PretrainRunInfo:
        """Load run info from a W&B run string or a local path.

        W&B formats:
        - "entity/project/runId" (compact form)
        - "entity/project/runs/runId" (with /runs/)
        - "wandb:entity/project/runId" (with wandb: prefix)
        - "wandb:entity/project/runs/runId" (full wandb: form)
        - "https://wandb.ai/entity/project/runs/runId..." (URL)

        Local: path to a checkpoint file
        """
        try:
            entity, project, run_id = parse_wandb_run_path(str(path))
        except ValueError:
            # Not a W&B path, treat as local
            pass
        else:
            # W&B path - download files
            downloaded = _download_wandb_files(entity, project, run_id)

            with open(downloaded.config) as f:
                config_dict = yaml.safe_load(f)

            with open(downloaded.model_config) as f:
                model_config_dict = yaml.safe_load(f)

            return cls(
                checkpoint_path=downloaded.checkpoint,
                config_dict=config_dict,
                model_config_dict=model_config_dict,
                tokenizer_path=downloaded.tokenizer,
                hf_tokenizer_path=_extract_hf_tokenizer_path(config_dict),
            )

        # Local path
        ckpt_path = Path(path)
        assert ckpt_path.is_file(), f"Expected a file, got {ckpt_path}"
        # Look for configs and tokenizer in parent.parent (output_dir)
        output_dir = ckpt_path.parent.parent
        config_path = output_dir / "final_config.yaml"
        model_config_path = output_dir / "model_config.yaml"
        assert config_path.exists(), (
            f"Expected config at {config_path} next to checkpoint {ckpt_path}"
        )
        assert model_config_path.exists(), (
            f"Expected model config at {model_config_path} next to checkpoint {ckpt_path}"
        )
        tokenizer_path = output_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            tokenizer_path = None

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        with open(model_config_path) as f:
            model_config_dict = yaml.safe_load(f)

        return cls(
            checkpoint_path=ckpt_path,
            config_dict=config_dict,
            model_config_dict=model_config_dict,
            tokenizer_path=tokenizer_path,
            hf_tokenizer_path=_extract_hf_tokenizer_path(config_dict),
        )

    def load_tokenizer(self) -> HFTokenizer:
        """Load tokenizer with simple HF/local logic like in dataloaders.py."""
        assert self.hf_tokenizer_path is not None or self.tokenizer_path is not None, (
            "Either hf_tokenizer_path or tokenizer_path must be specified"
        )
        # Prefer HF path if specified
        if self.hf_tokenizer_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                self.hf_tokenizer_path,
                add_bos_token=False,
                unk_token="[UNK]",
                eos_token="[EOS]",
                bos_token=None,
            )
            return tokenizer.backend_tokenizer  # pyright: ignore[reportAttributeAccessIssue]

        # Next, prefer tokenizer.json adjacent to outputs (downloaded from wandb or local)
        if self.tokenizer_path is not None and self.tokenizer_path.exists():
            return HFTokenizer.from_file(str(self.tokenizer_path))

        raise FileNotFoundError("Could not resolve a tokenizer for this PretrainRunInfo")
