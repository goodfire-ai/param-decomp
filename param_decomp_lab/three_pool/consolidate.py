"""Offline consolidation of a 3-pool checkpoint from its scratch partials.

The 3-pool train loop writes per-rank partials to ``scratch_dir/step_<S>/`` (see
``ThreePoolTrainer.snapshot``) and then continues — it never assembles the
canonical checkpoint. This module does that assembly off the critical path,
driven by the async SLURM job that already runs the slow eval. It reads every
partial for a step, builds the full ``ComponentModel`` state_dict + the canonical
``ThreePoolTrainingState``, writes ``model_<S>.pth`` + ``training_<S>.pth``,
prunes old ``training_*.pth`` (keeping all ``model_*.pth``), and removes the
step's scratch dir.

Idempotent: if ``training_<S>.pth`` already exists, consolidation is a no-op (the
async job may be retried). If the scratch dir is already gone (consolidation ran
before, or the run died), it's also a no-op — the consolidated artifacts are the
source of truth.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.ci_fns import CiConfig
from param_decomp.ci_sigmoids import SigmoidType
from param_decomp.log import logger
from param_decomp.training_state import ThreePoolTrainingState
from param_decomp_lab.infra.run_files import save_file
from param_decomp_lab.three_pool.checkpoint import assemble_model_state_dict_from_partials

SNAPSHOT_SCRATCH_DIRNAME = ".snapshot_scratch"
"""Subdir under the run's out_dir that holds the per-step partials. The trainer
writes here; consolidation reads + cleans here."""

CONSOLIDATE_META_FILENAME = "meta.pth"

DEFAULT_KEEP_LAST_N_TRAINING = 3
"""How many ``training_<step>.pth`` files to keep after consolidation. All
``model_<step>.pth`` files are always kept (they're the downstream artifact)."""


def step_scratch_dir(scratch_dir: Path, step: int) -> Path:
    return scratch_dir / f"step_{step}"


def consolidate_step(
    *,
    scratch_dir: Path,
    out_dir: Path,
    step: int,
    target_model: nn.Module,
    run_batch: RunBatch,
    ci_config: CiConfig,
    sigmoid_type: SigmoidType,
    keep_last_n_training: int,
) -> None:
    """Assemble + persist the step-S checkpoint from its scratch partials.

    No-op (returns early) when ``training_<S>.pth`` already exists or the scratch
    dir for the step is missing — both mean the work is already done (or was done
    by a prior, possibly-retried, invocation).
    """
    training_path = out_dir / f"training_{step}.pth"
    if training_path.is_file():
        logger.info(f"consolidate: {training_path.name} already exists; skipping")
        return

    step_dir = step_scratch_dir(scratch_dir, step)
    if not step_dir.is_dir():
        logger.warning(
            f"consolidate: scratch dir {step_dir} missing and no {training_path.name}; "
            f"nothing to consolidate for step {step}"
        )
        return

    # PD_3POOL_SNAPSHOT_RANK0_SLEEP_S models the old slow rank-0 read. It now
    # lives here, off the train loop, so injecting it proves the train loop's PG
    # timeout is unaffected by a slow consolidation (the watchdog-safe test).
    sleep_s = os.environ.get("PD_3POOL_SNAPSHOT_RANK0_SLEEP_S", "").strip()
    if sleep_s:
        logger.info(f"consolidate: sleep {sleep_s}s (fault injection, off train loop)")
        time.sleep(float(sleep_s))

    meta = torch.load(step_dir / CONSOLIDATE_META_FILENAME, map_location="cpu", weights_only=False)
    world_size: int = meta["world_size"]
    all_sites: tuple[str, ...] = tuple(meta["all_sites"])
    c_per_site: dict[str, int] = meta["c_per_site"]

    partials: list[dict[str, Any]] = []
    for r in range(world_size):
        partials.append(
            torch.load(step_dir / f"rank_{r}.pth", map_location="cpu", weights_only=False)
        )
    logger.info(f"consolidate: read {world_size} partials for step {step}")

    model_state = assemble_model_state_dict_from_partials(
        partials=partials,
        target_model=target_model,
        run_batch=run_batch,
        ci_config=ci_config,
        sigmoid_type=sigmoid_type,
        c_per_site=c_per_site,
        all_sites=all_sites,
    )

    components_optimizer: dict[str, dict[str, Any]] = {}
    ci_fn_optimizer: dict[str, dict[str, Any]] = {}
    ppgd_by_rank: dict[int, dict[str, Any]] = {}
    for r, partial in enumerate(partials):
        pool: str = partial["pool"]
        match pool:
            case "layerwise":
                components_optimizer.update(partial["optimizer_by_name"])
            case "ci":
                ci_fn_optimizer.update(partial["optimizer_by_name"])
            case "ppgd":
                if "ppgd" in partial:
                    ppgd_by_rank[r] = partial["ppgd"]
            case _:
                raise AssertionError(f"unknown pool {pool!r} in rank-{r} partial")

    state = ThreePoolTrainingState(
        step=step,
        pd_config=meta["pd_config"],
        runtime_config=meta["runtime_config"],
        three_pool_config=meta["three_pool_config"],
        layout_fingerprint=meta["layout_fingerprint"],
        component_model=model_state,
        components_optimizer=components_optimizer,
        ci_fn_optimizer=ci_fn_optimizer,
        ppgd_state_by_rank=ppgd_by_rank,
    )

    model_path = out_dir / f"model_{step}.pth"
    save_file(model_state, model_path)
    save_file(state, training_path)
    logger.info(f"consolidate: wrote {model_path.name} + {training_path.name}")

    _prune_old_training(out_dir, keep_last_n=keep_last_n_training)

    shutil.rmtree(step_dir, ignore_errors=True)
    logger.info(f"consolidate: removed scratch {step_dir}")


def _prune_old_training(out_dir: Path, *, keep_last_n: int) -> None:
    """Delete all but the last ``keep_last_n`` ``training_<step>.pth`` files.

    ``model_<step>.pth`` files are never pruned — they're the downstream artifact
    and are cheap relative to the full training state.
    """
    steps: list[int] = []
    for p in out_dir.glob("training_*.pth"):
        try:
            steps.append(int(p.stem.removeprefix("training_")))
        except ValueError:
            continue
    steps.sort()
    if len(steps) <= keep_last_n:
        return
    for step in steps[: len(steps) - keep_last_n]:
        path = out_dir / f"training_{step}.pth"
        if path.is_file():
            path.unlink()
            logger.info(f"consolidate: pruned old {path.name}")
