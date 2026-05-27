"""Generic experiment orchestration: fresh-run and resume-from-snapshot drivers.

Each experiment (`lm`, `tms`, `resid_mlp`) packages its per-experiment callables
into an :class:`ExperimentBundle` and dispatches to :func:`run_fresh` /
:func:`run_resumed` from its own ``run.py::main``. The generic surface — config
load, distributed init, device + seed, runtime cfg refresh, sink construction,
trainer build, train loop, sink finish, plus resume-specific snapshot read and
provenance write — lives here.

The bundle defines a uniform signature for the per-experiment build callables
so the runner doesn't need to know which experiment it's serving. Non-DDP
experiments (TMS, ResidMLP) simply set ``uses_distributed=False`` and accept
``dist_state=None`` in their build callables.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.nn as nn
from torch.utils.data import DataLoader

from param_decomp.batch_and_loss_fns import ReconstructionLoss, RunBatch
from param_decomp.distributed import DistributedState, is_main_process
from param_decomp.log import logger
from param_decomp.optimize import EvalLoop, Trainer
from param_decomp_lab.distributed import get_device, init_distributed
from param_decomp_lab.experiments.utils import RUN_META_FILENAME, ExperimentConfig, init_pd_run
from param_decomp_lab.resumption import (
    ResumeConfig,
    ResumeProvenance,
    read_training_snapshot,
    resolve_step,
    write_provenance,
)
from param_decomp_lab.seed import set_seed


@dataclass(frozen=True)
class ExperimentBundle[CfgT: ExperimentConfig[Any, Any]]:
    """Per-experiment callables that the generic runner needs.

    Each experiment constructs one of these in its ``run.py`` and passes it to
    :func:`run_fresh` / :func:`run_resumed`. The build callables take the full
    config (so they can reach ``cfg.target`` / ``cfg.data`` themselves) and the
    runtime context (``device``, ``dist_state``) the runner has set up. Bundle
    is generic over the experiment's config type so callers stay type-checked.

    Attributes:
        config_cls: The pydantic ``ExperimentConfig`` subclass for this experiment.
            Used by the resume path to ``.from_file`` the parent's
            ``run_meta.yaml``.
        build_target: Construct the (frozen) target model.
        build_train_loader: Construct the training data loader.
        build_eval_loop: Construct the eval loop, or return ``None`` to skip.
        make_run_batch: Construct the per-experiment ``RunBatch`` adapter.
        reconstruction_loss: The recon loss closure for this experiment.
        uses_distributed: Whether to run ``init_distributed`` at the top of the
            runner. ``False`` for single-process experiments (TMS, ResidMLP);
            ``True`` for LM.
        refine_cfg: Optional hook called after ``build_target`` to return a
            refined cfg (e.g. TMS's ``tied_weights`` depend on the constructed
            target model). Default ``None`` means no refinement.
    """

    config_cls: type[CfgT]
    build_target: Callable[[CfgT], nn.Module]
    build_train_loader: Callable[[CfgT, str, DistributedState | None], DataLoader[Any]]
    build_eval_loop: Callable[[CfgT, str, DistributedState | None], EvalLoop | None]
    make_run_batch: Callable[[CfgT], RunBatch]
    reconstruction_loss: ReconstructionLoss
    uses_distributed: bool = False
    refine_cfg: Callable[[CfgT, nn.Module], CfgT] | None = None


def _setup_runtime[CfgT: ExperimentConfig[Any, Any]](
    cfg: CfgT, bundle: ExperimentBundle[CfgT]
) -> tuple[CfgT, str, DistributedState | None]:
    """Init distributed (if requested), seed RNG, derive device, refresh cfg.runtime.

    Returns the refreshed cfg, the device string, and the dist state (or ``None``).
    The returned cfg has ``runtime.device`` and ``runtime.dp`` updated for the
    current resume / submission environment so downstream consumers see truth.
    """
    dist_state = init_distributed() if bundle.uses_distributed else None
    if bundle.uses_distributed and is_main_process():
        logger.info(f"Distributed state: {dist_state}")
    set_seed(cfg.pd.seed)
    device = get_device()
    refreshed = cfg.model_copy(
        update={
            "runtime": cfg.runtime.model_copy(
                update={
                    "device": device,
                    "dp": dist_state.world_size if dist_state is not None else None,
                }
            )
        }
    )
    return refreshed, device, dist_state


def run_fresh[CfgT: ExperimentConfig[Any, Any]](
    bundle: ExperimentBundle[CfgT],
    config_path: Path,
    *,
    group: str | None = None,
    tags: str | None = None,
    run_id: str | None = None,
) -> None:
    """Fresh-run driver: parse YAML, build everything via the bundle, train from step 0."""
    cfg = bundle.config_cls.from_file(config_path)
    cfg, device, dist_state = _setup_runtime(cfg, bundle)

    target_model = bundle.build_target(cfg)
    if bundle.refine_cfg is not None:
        cfg = bundle.refine_cfg(cfg, target_model)
    train_loader = bundle.build_train_loader(cfg, device, dist_state)
    eval_loop = bundle.build_eval_loop(cfg, device, dist_state)
    sink = init_pd_run(cfg, group=group, tags=tags, run_id=run_id)

    try:
        trainer = Trainer(
            target_model=target_model,
            run_batch=bundle.make_run_batch(cfg),
            reconstruction_loss=bundle.reconstruction_loss,
            pd_config=cfg.pd,
            runtime_config=cfg.runtime,
        )
        trainer.run(train_loader, sink, cfg.cadence, eval_loop)
    finally:
        sink.finish()


def run_resumed[CfgT: ExperimentConfig[Any, Any]](
    bundle: ExperimentBundle[CfgT],
    resume_cfg_path: Path,
    *,
    group: str | None = None,
    tags: str | None = None,
    run_id: str | None = None,
) -> None:
    """Resume-run driver: read parent ``run_meta.yaml`` + ``training_<step>.pth``,
    rebuild experiment via the bundle, continue training.

    The parent's saved config is the source of cfg truth; only the runtime fields
    (``device``, ``dp``) are refreshed for the resume environment. For mid-trajectory
    edits to the saved ``pd_config`` (e.g. extending ``steps``), the caller should
    use ``read_training_snapshot`` directly and call ``Trainer.from_snapshot`` —
    this generic driver is the "continue with original config" path.
    """
    resume_cfg = ResumeConfig.from_file(resume_cfg_path)
    parent_cfg = bundle.config_cls.from_file(resume_cfg.from_run / RUN_META_FILENAME)

    if is_main_process():
        logger.info(f"Resuming from {resume_cfg.from_run} @ step {resume_cfg.step}")

    cfg, device, dist_state = _setup_runtime(parent_cfg, bundle)

    resolved_step = resolve_step(resume_cfg.from_run, resume_cfg.step)
    snapshot = read_training_snapshot(resume_cfg.from_run, resolved_step)
    # Override the saved device with the current resume environment. The dict
    # field is mutable even on a frozen dataclass.
    snapshot.runtime_config["device"] = device

    target_model = bundle.build_target(cfg)
    if bundle.refine_cfg is not None:
        cfg = bundle.refine_cfg(cfg, target_model)
    train_loader = bundle.build_train_loader(cfg, device, dist_state)
    eval_loop = bundle.build_eval_loop(cfg, device, dist_state)
    sink = init_pd_run(cfg, group=group, tags=tags, run_id=run_id)
    if sink.out_dir is not None:
        write_provenance(
            sink.out_dir,
            ResumeProvenance(parent_run_dir=resume_cfg.from_run, parent_step=resolved_step),
        )

    try:
        trainer = Trainer.from_snapshot(
            snapshot,
            target_model=target_model,
            run_batch=bundle.make_run_batch(cfg),
            reconstruction_loss=bundle.reconstruction_loss,
        )
        trainer.run(train_loader, sink, cfg.cadence, eval_loop)
    finally:
        sink.finish()
