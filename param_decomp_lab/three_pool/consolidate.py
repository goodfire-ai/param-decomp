"""Offline consolidation of a 3-pool checkpoint from its scratch partials.

The 3-pool train loop writes per-rank partials to ``scratch_dir/step_<S>/`` (see
``ThreePoolTrainer.snapshot``) and then continues — it never assembles the
canonical checkpoint. This module does that assembly off the critical path,
driven by the async SLURM job that already runs the slow eval. It reads every
(small, parameter-shaped) partial for a step, builds the full ``ComponentModel``
state_dict + the canonical ``ThreePoolTrainingState``, writes ``model_<S>.pth`` +
``training_<S>.pth``, prunes old ``training_*.pth`` (keeping all ``model_*.pth``)
+ old ``ppgd_*/`` shard dirs, and removes the step's scratch dir.

The data-shaped PPGD adversarial sources are NOT here: each adversary rank writes
its own shard to ``ppgd_<S>/rank_<r>.pth`` at snapshot time, in parallel
(``snapshot`` in the trainers), so consolidation streams only the small partials.
``load_ppgd_shard`` reads a rank's shard back on resume; ``_prune_old_ppgd`` keeps
only the newest few dirs (they're huge).

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
from torch import Tensor

from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.ci_sigmoids import SigmoidType
from param_decomp.log import logger
from param_decomp.training_state import ThreePoolTrainingState
from param_decomp_config.ci_fn import CiConfig
from param_decomp_lab.infra.run_files import save_file
from param_decomp_lab.three_pool.checkpoint import assemble_model_state_dict_from_partials

SNAPSHOT_SCRATCH_DIRNAME = ".snapshot_scratch"
"""Subdir under the run's out_dir that holds the per-step partials. The trainer
writes here; consolidation reads + cleans here."""

CONSOLIDATE_META_FILENAME = "meta.pth"

DEFAULT_KEEP_LAST_N_TRAINING = 3
"""How many ``training_<step>.pth`` files to keep after consolidation. All
``model_<step>.pth`` files are always kept (they're the downstream artifact)."""

DEFAULT_KEEP_LAST_N_PPGD = 1
"""How many ``ppgd_<step>/`` shard dirs to keep. These are huge (the data-shaped
adversarial sources — TBs at large batch), so we keep only the latest: a standard
resume uses the newest checkpoint, and an older checkpoint with its shards pruned
still resumes (the adversary re-warms via ``n_warmup``)."""


def step_scratch_dir(scratch_dir: Path, step: int) -> Path:
    return scratch_dir / f"step_{step}"


def ppgd_shard_dirname(step: int) -> str:
    """Dir name (under the run's out_dir) holding the step's per-rank PPGD source shards."""
    return f"ppgd_{step}"


def load_ppgd_shard(ppgd_shard_dir: Path | None, rank: int) -> dict[str, Any] | None:
    """Read this rank's PPGD source shard for resume.

    Returns ``None`` (⇒ the adversary re-warms from scratch) when no shard dir was
    given or this rank's shard is absent (e.g. an old checkpoint whose shards were
    pruned, or a non-adversary rank).
    """
    if ppgd_shard_dir is None:
        return None
    shard = ppgd_shard_dir / f"rank_{rank}.pth"
    if not shard.is_file():
        logger.warning(f"resume: no PPGD shard {shard}; adversary rank {rank} will re-warm")
        return None
    return torch.load(shard, map_location="cpu", weights_only=False)


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

    Streams only the small, parameter-shaped partials — the data-shaped PPGD
    shards were written at snapshot time (``ppgd_<S>/``) and are read on resume;
    here they're only pruned to the latest few.

    No-op (returns early) when ``training_<S>.pth`` already exists or the scratch
    dir for the step is missing — both mean the work is already done (or was done
    by a prior, possibly-retried, invocation).
    """
    training_path = out_dir / f"training_{step}.pth"
    step_dir = step_scratch_dir(scratch_dir, step)
    if training_path.is_file():
        # Already consolidated. Clean up any scratch left behind by a prior run
        # that wrote the checkpoint but died before removing it (e.g. crashed in
        # prune), so this step stops looking unconsolidated.
        logger.info(f"consolidate: {training_path.name} already exists; skipping")
        shutil.rmtree(step_dir, ignore_errors=True)
        return

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

    # The data-shaped PPGD sources are NOT in the partials — each adversary rank wrote its
    # own shard to ppgd_<step>/rank_<r>.pth at snapshot time, in parallel (read back per-rank
    # on resume). Consolidation streams only the small, parameter-shaped partials.
    #
    # Stream the partials one at a time: collect the model params + route the optimizer state,
    # then drop the partial. Peak RAM is ~one partial + the assembled CPU model, not the full
    # set at once.
    collected_model_params: dict[str, Tensor] = {}
    components_optimizer: dict[str, dict[str, Any]] = {}
    ci_fn_optimizer: dict[str, dict[str, Any]] = {}
    for r in range(world_size):
        partial = torch.load(step_dir / f"rank_{r}.pth", map_location="cpu", weights_only=False)
        for k, v in partial["model_params"].items():
            assert k not in collected_model_params, f"duplicate model param {k!r} across partials"
            collected_model_params[k] = v
        match partial["pool"]:
            case "chunkwise":
                components_optimizer.update(partial["optimizer_by_name"])
            case "ci" | "pool_a":
                ci_fn_optimizer.update(partial["optimizer_by_name"])
            case "ppgd":
                pass
            case other:
                raise AssertionError(f"unknown pool {other!r} in rank-{r} partial")
        del partial
    logger.info(f"consolidate: streamed {world_size} partials for step {step}")

    model_state = assemble_model_state_dict_from_partials(
        collected_model_params=collected_model_params,
        target_model=target_model,
        run_batch=run_batch,
        ci_config=ci_config,
        sigmoid_type=sigmoid_type,
        c_per_site=c_per_site,
        all_sites=all_sites,
    )

    state = ThreePoolTrainingState(
        step=step,
        pd_config=meta["pd_config"],
        runtime_config=meta["runtime_config"],
        three_pool_config=meta["three_pool_config"],
        layout_fingerprint=meta["layout_fingerprint"],
        component_model=model_state,
        components_optimizer=components_optimizer,
        ci_fn_optimizer=ci_fn_optimizer,
    )

    model_path = out_dir / f"model_{step}.pth"
    save_file(model_state, model_path)
    save_file(state, training_path)
    logger.info(f"consolidate: wrote {model_path.name} + {training_path.name}")

    _prune_old_training(out_dir, keep_last_n=keep_last_n_training)
    _prune_old_ppgd(out_dir, keep_last_n=DEFAULT_KEEP_LAST_N_PPGD)

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
        # missing_ok: a concurrently-running consolidation job for another step
        # may have already pruned this same old file. That's benign — the target
        # state (this file gone) is reached either way.
        path.unlink(missing_ok=True)
        logger.info(f"consolidate: pruned old {path.name}")


def _prune_old_ppgd(out_dir: Path, *, keep_last_n: int) -> None:
    """Delete all but the last ``keep_last_n`` CONSOLIDATED ``ppgd_<step>/`` shard dirs.

    Only prunes ppgd dirs whose step is consolidated (``training_<step>.pth`` exists).
    A ppgd dir for an unconsolidated step is in flight — its shards were written at
    snapshot time but its checkpoint hasn't been assembled yet, and it may become the
    resume target — so it must survive until its own consolidation prunes it. (Pruning
    purely by dir name would delete the step currently being consolidated when a newer,
    not-yet-consolidated ppgd dir already sits on disk.)
    """
    consolidated_steps: list[int] = []
    for d in out_dir.glob("ppgd_*"):
        if not d.is_dir():
            continue
        try:
            step = int(d.name.removeprefix("ppgd_"))
        except ValueError:
            continue
        if (out_dir / f"training_{step}.pth").is_file():
            consolidated_steps.append(step)
    consolidated_steps.sort()
    if len(consolidated_steps) <= keep_last_n:
        return
    for step in consolidated_steps[: len(consolidated_steps) - keep_last_n]:
        # ignore_errors: a concurrent consolidation for another step may have pruned it already.
        shutil.rmtree(out_dir / ppgd_shard_dirname(step), ignore_errors=True)
        logger.info(f"consolidate: pruned old {ppgd_shard_dirname(step)}/")


def unconsolidated_steps(out_dir: Path) -> list[int]:
    """Steps that have scratch partials on disk but no `training_<step>.pth` yet.

    These are saves the async job never finished consolidating (it crashed, was
    preempted, or never ran). They're safe + cheap to consolidate by re-running.
    """
    scratch = out_dir / SNAPSHOT_SCRATCH_DIRNAME
    if not scratch.is_dir():
        return []
    out: list[int] = []
    for d in scratch.glob("step_*"):
        if not d.is_dir():
            continue
        step = int(d.name.removeprefix("step_"))
        if not (out_dir / f"training_{step}.pth").is_file():
            out.append(step)
    return sorted(out)
