"""The local store of pretrain checkpoints, keyed by W&B run reference.

The readers (`param_decomp.targets.llama_simple_mlp`) want a store entry
`<data_root>/pretrain_cache/<project>-<run_id>/` holding exactly one
`model_step_<N>.safetensors` plus a `model_config.yaml`. Two writers produce that layout:
a local `param_decomp.pretrain.train` run writes it directly as its output, and
`resolved_cache_dir` fetches it from the named W&B run on first use (short-circuiting on
a complete entry, so cold starts are idempotent across ranks and requeues). Keeping the
fetch here is what lets `targets/` stay network-free.
"""

from fnmatch import fnmatch
from pathlib import Path
from typing import NoReturn

import wandb
from wandb.apis.public import Run

from param_decomp.core.log import logger
from param_decomp.infra.wandb import download_wandb_file, parse_wandb_run_path

CHECKPOINT_GLOB = "model_step_*.safetensors"
TORCH_CHECKPOINT_GLOB = "model_step_*.pt"
MODEL_CONFIG_FILENAME = "model_config.yaml"

_CONVERTER = "param_decomp_jax/jax_single_pool/tools/convert_llama_simple_mlp_checkpoint.py"


def _run_reference(run_path: str) -> tuple[str, str, str]:
    """`(entity, project, run_id)` for a pretrain run, with no ambient lookup.

    The bare `p-…` shortcut is refused deliberately: it resolves the entity from the
    environment (against the library's no-ambient-paths rule) and names a decomposition
    run, never a pretrain run.
    """
    assert "/" in run_path, (
        f"pretrain run_path must spell out its entity and project, got {run_path!r} — "
        "the bare `p-…` shortcut resolves the entity from ambient env and names a "
        "decomposition run, not a pretrain run"
    )
    return parse_wandb_run_path(run_path)


def cache_dir_for_run(data_root: Path, run_path: str) -> Path:
    """`<data_root>/pretrain_cache/<project>-<run_id>` — the cache key for `run_path`,
    derived purely from the run reference. Touches neither disk nor network."""
    _entity, project, run_id = _run_reference(run_path)
    return data_root / "pretrain_cache" / f"{project}-{run_id}"


def is_complete(cache_dir: Path) -> bool:
    """Whether `cache_dir` holds a checkpoint the readers can open: exactly one
    converted safetensors file plus its `model_config.yaml`."""
    return (cache_dir / MODEL_CONFIG_FILENAME).is_file() and len(
        list(cache_dir.glob(CHECKPOINT_GLOB))
    ) == 1


def resolved_model_config_dir(data_root: Path, run_path: str) -> Path:
    """Resolve architecture metadata without fetching the model's weights."""
    cache_dir = cache_dir_for_run(data_root, run_path)
    if (cache_dir / MODEL_CONFIG_FILENAME).is_file():
        return cache_dir

    entity, project, run_id = _run_reference(run_path)
    run = wandb.Api().run(f"{entity}/{project}/{run_id}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    download_wandb_file(run, cache_dir, MODEL_CONFIG_FILENAME)
    return cache_dir


def resolved_cache_dir(data_root: Path, run_path: str) -> Path:
    """Resolve `run_path` to its store entry, downloading the W&B run's checkpoint and
    `model_config.yaml` into it first if it isn't already complete.

    A complete entry short-circuits without touching the network, so this is safe to
    call on every rank's cold start and on a requeue. A cold entry needs network, and
    W&B credentials when the run isn't public.
    """
    cache_dir = cache_dir_for_run(data_root, run_path)
    if is_complete(cache_dir):
        return cache_dir

    entity, project, run_id = _run_reference(run_path)
    logger.info(f"fetching pretrain checkpoint {entity}/{project}/{run_id} -> {cache_dir}")
    run = wandb.Api().run(f"{entity}/{project}/{run_id}")
    filenames = [f.name for f in run.files()]

    cache_dir.mkdir(parents=True, exist_ok=True)
    download_wandb_file(run, cache_dir, MODEL_CONFIG_FILENAME)

    match [n for n in filenames if fnmatch(n, CHECKPOINT_GLOB)]:
        case [checkpoint]:
            download_wandb_file(run, cache_dir, checkpoint)
            assert is_complete(cache_dir), cache_dir
            return cache_dir
        case []:
            _raise_for_missing_safetensors(run, cache_dir, run_path, filenames)
        case several:
            raise AssertionError(
                f"{run_path} carries {len(several)} checkpoints ({several}) — the cache "
                f"layout holds exactly one, so pick a step and stage it by hand"
            )


def _raise_for_missing_safetensors(
    run: Run, cache_dir: Path, run_path_for_err: str, filenames: list[str]
) -> NoReturn:
    """The run has no converted checkpoint. Torch-era runs carry `model_step_<N>.pt`;
    download it so the one remaining step is a local conversion, then say so."""
    torch_checkpoints = [n for n in filenames if fnmatch(n, TORCH_CHECKPOINT_GLOB)]
    assert torch_checkpoints, (
        f"{run_path_for_err} has no {CHECKPOINT_GLOB} and no {TORCH_CHECKPOINT_GLOB} — it is not "
        f"a pretrain run this loader can consume. Files: {sorted(filenames)[:20]}"
    )
    staged = [download_wandb_file(run, cache_dir, n) for n in torch_checkpoints]
    raise AssertionError(
        f"{run_path_for_err} is a torch-era pretrain run: it ships {torch_checkpoints} and this "
        f"loader reads {CHECKPOINT_GLOB}. The torch checkpoint(s) are now downloaded at "
        f"{[str(p) for p in staged]} — convert one in a torch venv with `{_CONVERTER}` "
        f"(git tag `torch-oracle`), writing `model_step_<N>.safetensors` beside them in "
        f"{cache_dir}. Conversion needs torch, which this library deliberately does not "
        f"depend on."
    )
