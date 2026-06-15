"""Consolidate an FSDP DCP save + run slow eval; log into the parent's wandb run.

The FSDP sibling of `experiments.lm.async_eval`. Kept as a SEPARATE file rather than a
`variant="fsdp"` branch of `async_eval` because the consolidation MECHANISM differs at the
root: the pooled `async_eval` reads per-rank partials written by the train loop
(`three_pool.consolidate.consolidate_step`), whereas FSDP reads sharded
`torch.distributed.checkpoint` (DCP) shards into a fresh `LMComponentModel`
(`fsdp.consolidate.consolidate`, driven by a `build_full_model` callable). Forcing both
through one entry would mean two disjoint consolidation paths under one `variant` switch
plus an `FsdpLMExperimentConfig` branch in the pooled file — more entanglement than a
focused sibling. This file reuses the pure eval helpers from `async_eval` so only the
consolidation step is duplicated.

Invoked by `fsdp_run.submit_slurm_async_consolidate_and_eval` after a sharded save, off the
train-loop critical path. The training side hands us:
  * ``--run <run_path>`` — the parent training run (SavedFsdpLMRun-compatible)
  * ``--step <N>``       — which DCP checkpoint to consolidate + eval
  * ``--eval-config <yaml>`` — an ``EvalConfig`` listing exactly which (slow) metrics to run

Usage:
    python -m param_decomp_lab.experiments.lm.fsdp_async_eval \\
        --run <run_path> --step <N> --eval-config <yaml.path>
"""

import gc
import time
from pathlib import Path

import fire
import torch

from param_decomp.decomposition_targets import resolve_decomposition_targets
from param_decomp.distributed import is_main_process
from param_decomp.log import logger
from param_decomp_config.experiment import EvalConfig
from param_decomp_lab.component_model_io import load_component_model
from param_decomp_lab.distributed import (
    get_device,
    init_distributed,
    with_distributed_cleanup,
)
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
from param_decomp_lab.experiments.lm.async_eval import (
    _log_eval_to_wandb,
    _resolve_eval_checkpoint_path,
    _run_eval_pass,
    _step_from_checkpoint_name,
)
from param_decomp_lab.experiments.lm.fsdp_run import FsdpLMExperimentConfig
from param_decomp_lab.experiments.lm.run import (
    _resolve_train_run_id,
    build_lm_loader,
    build_target,
    make_run_batch,
)
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel
from param_decomp_lab.experiments.utils import EXPERIMENT_CONFIG_FILENAME
from param_decomp_lab.fsdp.consolidate import consolidate
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.seed import set_seed
from param_decomp_lab.three_pool.consolidate import DEFAULT_KEEP_LAST_N_TRAINING


def _build_full_model(cfg: FsdpLMExperimentConfig) -> LMComponentModel:
    """A fresh, unsharded CPU `LMComponentModel` — the consolidation assembly buffer.

    The frozen target is rebuilt from the vendored weights; the trainable V/U + CI fn are
    freshly initialised (DCP overwrites them on load). Mirrors `load_vendored_component_model`
    minus the checkpoint load.
    """
    target_model = build_target(cfg.target)
    target_model.eval()
    target_model.requires_grad_(False)
    resolved_targets = resolve_decomposition_targets(
        target_model, list(cfg.pd.decomposition_targets)
    )
    return LMComponentModel.build(
        target_model=target_model,
        decomposition_targets=resolved_targets,
        ci_config=cfg.pd.ci_config,
        sigmoid_type=cfg.pd.sigmoid_type,
    )


def _consolidate_or_wait(
    *,
    cfg: FsdpLMExperimentConfig,
    out_dir: Path,
    step: int,
    wait_timeout_s: float = 1800.0,
) -> None:
    """Rank 0 assembles `model_<step>.pth` + `training_<step>.pth` from the DCP shards (a
    single-process op, no NCCL); the other ranks wait on the shared FS for the result.

    Fail-fast on both sides — never wedge holding GPUs: rank 0 writes a
    `.consolidate_failed_<step>` sentinel and re-raises on failure; the other ranks poll for
    the success file OR the sentinel, and time out.
    """
    training_path = out_dir / f"training_{step}.pth"
    fail_sentinel = out_dir / f".consolidate_failed_{step}"
    fail_sentinel.unlink(missing_ok=True)
    if is_main_process():
        try:
            with torch.device("cpu"):
                consolidate(
                    run_dir=out_dir,
                    step=step,
                    build_full_model=lambda: _build_full_model(cfg),
                    pd_config=cfg.pd,
                    runtime_config_dump=cfg.runtime.model_dump(),
                    keep_last_n_training=DEFAULT_KEEP_LAST_N_TRAINING,
                )
        except BaseException:
            fail_sentinel.touch()
            raise
        gc.collect()
        return
    deadline = time.perf_counter() + wait_timeout_s
    while not training_path.is_file():
        assert not fail_sentinel.is_file(), (
            f"rank-0 consolidation failed (sentinel {fail_sentinel.name}); aborting wait"
        )
        assert time.perf_counter() < deadline, (
            f"timed out after {wait_timeout_s}s waiting for {training_path} "
            f"(rank-0 consolidation did not finish)"
        )
        time.sleep(2.0)


@with_distributed_cleanup
def main(
    run: str | Path,
    *,
    step: int,
    eval_config: str | Path | None = None,
    group: str | None = None,
    tags: str | None = None,
) -> None:
    """Consolidate an FSDP DCP checkpoint and post slow-eval results to the parent wandb run.

    Args:
        run: SavedFsdpLMRun-compatible reference (wandb URL / entity/project/runId /
            bare ``p-xxxxxxxx`` / local out_dir).
        step: Which DCP checkpoint to consolidate + evaluate.
        eval_config: Path to an `EvalConfig` YAML that fully specifies the (slow) metrics to
            run. If omitted, falls back to the parent run's ``cfg.eval``.
        group / tags: optional wandb metadata for the resumed run.
    """
    # Read the config from experiment_config.yaml directly (not `SavedFsdpLMRun.from_path`):
    # no `model_<step>.pth` exists yet (this job assembles it below), and `from_path` eagerly
    # resolves one.
    train_run_id = _resolve_train_run_id(run)
    out_dir = PARAM_DECOMP_OUT_DIR / "runs" / train_run_id
    cfg = FsdpLMExperimentConfig.from_file(out_dir / EXPERIMENT_CONFIG_FILENAME)

    if eval_config is not None:
        eval_cfg = EvalConfig.from_file(Path(eval_config))
    else:
        assert cfg.eval is not None, (
            f"fsdp_async_eval requires either --eval-config or the parent config to declare "
            f"an `eval:` block ({run})"
        )
        eval_cfg = cfg.eval

    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
    set_seed(cfg.pd.seed)
    device = get_device()

    # Assemble model_<step>.pth + training_<step>.pth from the DCP shards. Rank 0 does the
    # single-process assembly on CPU; the other ranks WAIT on the file (not an NCCL barrier).
    _consolidate_or_wait(cfg=cfg, out_dir=out_dir, step=step)

    # Consolidation may be the only job — when there are no slow metrics the DCP save still
    # needs assembling, but there is no eval pass to run.
    if not eval_cfg.metrics:
        if is_main_process():
            logger.info("fsdp_async_eval: no metrics; consolidation-only, skipping eval pass")
        return

    checkpoint_path = _resolve_eval_checkpoint_path(run, step)
    resolved_step = _step_from_checkpoint_name(checkpoint_path.name)
    if is_main_process():
        logger.info(f"fsdp_async_eval: {run} @ step {resolved_step} (train run_id={train_run_id})")

    # Mirror the pooled `async_eval`: load the consolidated checkpoint into a core
    # `ComponentModel` so `_run_eval_pass` (which needs `forward(cache_type="input") ->
    # OutputWithCache` + `calc_weight_deltas`) works identically to the 3-pool eval path.
    component_model = load_component_model(
        pd_config=cfg.pd,
        checkpoint_path=checkpoint_path,
        target_model=build_target(cfg.target),
        run_batch=make_run_batch(cfg.target),
    )
    component_model.to(device)

    eval_loader = build_lm_loader(
        cfg.target,
        cfg.data,
        split="eval",
        device=device,
        batch_size=eval_cfg.batch_size,
        dist_state=dist_state,
        seed=cfg.pd.seed,
    )
    eval_metrics = [EVAL_METRIC_CLASSES[m.type](m) for m in eval_cfg.metrics]
    for m in eval_metrics:
        m.bind(model=component_model, device=device)

    results = _run_eval_pass(
        component_model=component_model,
        eval_loader=eval_loader,
        eval_metrics=eval_metrics,
        n_steps=eval_cfg.n_steps,
        device=device,
        step=resolved_step,
        pd_config=cfg,
    )

    if is_main_process():
        _log_eval_to_wandb(
            results,
            cfg=cfg,
            train_run_id=train_run_id,
            step=resolved_step,
            group=group,
            tags=tags,
        )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
