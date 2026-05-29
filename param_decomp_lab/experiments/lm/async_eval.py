"""Consolidate a 3-pool save + run eval metrics; log into the parent's wandb run.

This is an internal sbatch target, not a user-facing CLI. Invoked by
:func:`param_decomp_lab.experiments.lm.run.submit_slurm_async_consolidate_and_eval`
after a training save, off the critical path of training. For 3-pool runs it
first assembles ``model_<step>.pth`` + ``training_<step>.pth`` from the train
loop's per-rank partials (rank 0;
:func:`param_decomp_lab.three_pool.consolidate.consolidate_step`), then runs the
parent's slow metrics against the assembled checkpoint.

The training side hands us:
  * ``--run <run_path>`` — the parent training run (SavedLMRun-compatible)
  * ``--step <N>``       — which checkpoint to eval
  * ``--eval-config <yaml>`` — an ``EvalConfig`` listing exactly which metrics to run

We load the checkpoint, build the metrics from the override config, run one eval
pass, and post results to the parent's wandb run via ``wandb.init(id=<train_run_id>,
resume="must")``.

Usage:
    python -m param_decomp_lab.experiments.lm.async_eval \\
        --run <run_path> --step <N> --eval-config <yaml.path>
"""

import gc
from pathlib import Path
from typing import Any

import fire
import torch
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader

from param_decomp.component_model import ComponentModel
from param_decomp.distributed import is_main_process
from param_decomp.log import logger
from param_decomp.metrics.base import Metric
from param_decomp.metrics.output import collect_metric_outputs
from param_decomp.optimize import _build_metric_context
from param_decomp.torch_helpers import bf16_autocast, loop_dataloader
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.component_model_io import load_component_model
from param_decomp_lab.distributed import (
    get_device,
    init_distributed,
    with_distributed_cleanup,
)
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
from param_decomp_lab.experiments.lm.run import (
    LMExperimentConfig,
    _resolve_train_run_id,
    build_lm_loader,
    build_target,
    make_run_batch,
)
from param_decomp_lab.experiments.utils import RUN_META_FILENAME, EvalConfig
from param_decomp_lab.infra.run_files import resolve_run_files
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.infra.wandb import get_wandb_entity, try_wandb
from param_decomp_lab.run_sink import _wandb_value
from param_decomp_lab.seed import set_seed
from param_decomp_lab.three_pool.consolidate import (
    DEFAULT_KEEP_LAST_N_TRAINING,
    SNAPSHOT_SCRATCH_DIRNAME,
    consolidate_step,
)


def _resolve_eval_checkpoint_path(run_path: str | Path, step: int | None) -> Path:
    """Locate the `model_<step>.pth` on disk, downloading from W&B if needed."""
    if step is None:
        files = resolve_run_files(
            run_path, config_filename=RUN_META_FILENAME, checkpoint_prefix="model"
        )
        return files.checkpoint_path
    filename = f"model_{step}.pth"
    files = resolve_run_files(
        run_path, config_filename=RUN_META_FILENAME, checkpoint_filename=filename
    )
    return files.checkpoint_path


def _step_from_checkpoint_name(filename: str) -> int:
    """Parse the step number out of a `model_<step>.pth` filename."""
    assert filename.startswith("model_") and filename.endswith(".pth"), (
        f"expected `model_<step>.pth`, got {filename!r}"
    )
    return int(filename.removeprefix("model_").removesuffix(".pth"))


def _run_eval_pass(
    *,
    component_model: ComponentModel,
    eval_loader: DataLoader[Any],
    eval_metrics: list[Metric[Any]],
    n_steps: int,
    device: str,
    step: int,
    pd_config: Any,
) -> dict[str, Any]:
    """One full eval pass; returns the flattened metric output dict."""
    assert n_steps >= 1, f"n_steps must be at least 1, got {n_steps}"
    eval_iterator = loop_dataloader(eval_loader)
    with torch.no_grad(), bf16_autocast(enabled=pd_config.runtime.autocast_bf16):
        for m in eval_metrics:
            m.reset()
        for _ in range(n_steps):
            ctx = _build_metric_context(
                next(eval_iterator),
                step=step,
                is_eval=True,
                device=device,
                wrapped_model=component_model,
                component_model=component_model,
                config=pd_config.pd,
                reconstruction_loss=recon_loss_kl,
            )
            for m in eval_metrics:
                m.update(ctx)
        results = collect_metric_outputs(eval_metrics)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return results


def _log_eval_to_wandb(
    results: dict[str, Any],
    *,
    cfg: Any,
    train_run_id: str,
    step: int,
    group: str | None,
    tags: str | None,
) -> None:
    """Resume the parent's wandb run and log `slow_eval/<k>` for each result at `step`.

    Async slow-eval submissions write retroactively (the wandb run's current step
    has advanced past ``step`` while we were computing). Wandb's default step axis
    rejects non-monotonic writes, so we route slow-eval keys onto a dedicated
    ``slow_eval/step`` axis via ``wandb.define_metric``. In-train fast eval still
    uses the default step axis under the ``eval/`` prefix; the two namespaces are
    side-by-side in the wandb UI.
    """
    if cfg.wandb is None:
        logger.info("No wandb config on parent run; skipping wandb log of eval results.")
        return
    parsed_tags = [s.strip() for s in tags.split(",") if s.strip()] if tags else None
    wandb.init(
        id=train_run_id,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or get_wandb_entity(),
        resume="must",
        group=group,
        tags=parsed_tags,
    )
    wandb.define_metric("slow_eval/step")
    wandb.define_metric("slow_eval/*", step_metric="slow_eval/step")
    payload: dict[str, Any] = {f"slow_eval/{k}": _wandb_value(v) for k, v in results.items()}
    payload["slow_eval/step"] = step
    try_wandb(wandb.log, payload)
    wandb.finish()


@with_distributed_cleanup
def main(
    run: str | Path,
    *,
    step: int | None = None,
    eval_config: str | Path | None = None,
    group: str | None = None,
    tags: str | None = None,
) -> None:
    """Eval a checkpoint and post results to the parent wandb run.

    Args:
        run: SavedLMRun-compatible reference (wandb URL / entity/project/runId /
            bare ``p-xxxxxxxx`` / local out_dir).
        step: Which checkpoint to evaluate. Default: latest.
        eval_config: Path to an `EvalConfig` YAML that fully specifies the metrics
            to run. If omitted, falls back to the parent run's ``cfg.eval``.
        group / tags: optional wandb metadata for the resumed run.
    """
    # Read the config from run_meta.yaml directly rather than `SavedLMRun.from_path`:
    # for a 3-pool run no `model_<step>.pth` exists yet (this job assembles it
    # below), and `from_path` eagerly resolves one, which would crash here.
    train_run_id = _resolve_train_run_id(run)
    out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / train_run_id
    cfg = LMExperimentConfig.from_file(out_dir / RUN_META_FILENAME)

    if eval_config is not None:
        eval_cfg = EvalConfig.from_file(Path(eval_config))
    else:
        assert cfg.eval is not None, (
            f"async_eval requires either --eval-config or the parent config to "
            f"declare an `eval:` block ({run})"
        )
        eval_cfg = cfg.eval

    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
    set_seed(cfg.pd.seed)
    device = get_device()

    target_model = build_target(cfg.target)
    run_batch = make_run_batch(cfg.target)

    # 3-pool runs write only per-rank partials on the train loop; this async job
    # consolidates them into model_<step>.pth + training_<step>.pth (off the
    # train-loop critical path) before evaluating. Rank 0 does the assembly; all
    # ranks barrier so the model file exists before any rank resolves it.
    if cfg.three_pool is not None:
        assert step is not None, "3-pool async consolidation requires an explicit --step"
        if is_main_process():
            consolidate_step(
                scratch_dir=out_dir / SNAPSHOT_SCRATCH_DIRNAME,
                out_dir=out_dir,
                step=step,
                target_model=target_model,
                run_batch=run_batch,
                ci_config=cfg.pd.ci_config,
                sigmoid_type=cfg.pd.sigmoid_type,
                keep_last_n_training=DEFAULT_KEEP_LAST_N_TRAINING,
            )
        if dist.is_initialized():
            dist.barrier()

    # Consolidation may be the only job to do — when there are no slow metrics
    # the 3-pool save still needs assembling, but there is no eval pass to run.
    if not eval_cfg.metrics:
        if is_main_process():
            logger.info("async_eval: no metrics; consolidation-only, skipping eval pass")
        return

    checkpoint_path = _resolve_eval_checkpoint_path(run, step)
    resolved_step = _step_from_checkpoint_name(checkpoint_path.name)
    if is_main_process():
        logger.info(f"async_eval: {run} @ step {resolved_step} (train run_id={train_run_id})")

    component_model = load_component_model(
        pd_config=cfg.pd,
        checkpoint_path=checkpoint_path,
        target_model=target_model,
        run_batch=run_batch,
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
