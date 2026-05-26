"""Language-model PD experiment: YAML -> `optimize()` glue, plus the saved-run reload class.

The fresh-run path (`main`) and the reload path (`SavedLMRun`) both consume the
module-level `build_target` / `build_loader` / `make_run_batch` functions so there's
no duplication between them. Run via ``pd-lm path/to/config.yaml``; multi-process
(DDP) entry via ``torchrun`` of the same module.
"""

import atexit
import importlib
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import fire
import torch
import torch.distributed as dist
from pydantic import Discriminator
from torch.utils.data import DataLoader

from param_decomp._trace import trace
from param_decomp.base_config import BaseConfig
from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import DistributedState, is_main_process
from param_decomp.log import logger
from param_decomp.optimize import EvalLoop, Trainer
from param_decomp.three_pool import ThreePoolConfig, ThreePoolTrainer
from param_decomp.two_pool import TwoPoolConfig, TwoPoolTrainer
from param_decomp_lab.batch_and_loss_fns import make_run_batch as _make_run_batch
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.component_model_io import load_component_model
from param_decomp_lab.distributed import (
    ensure_cached_and_call,
    get_device,
    init_distributed,
    with_distributed_cleanup,
)
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
from param_decomp_lab.experiments.lm.data import (
    LMDataConfig,
    collate_fn_for,
    create_lm_data_loader,
    rank_batch_size,
)
from param_decomp_lab.experiments.utils import RUN_META_FILENAME, ExperimentConfig
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import (
    generate_run_id,
    resolve_config_path,
    resolve_run_files,
)
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.resumption import (
    ResumableRunSink,
    ResumeConfig,
    ResumeProvenance,
    read_resume_snapshot,
    resolve_step,
    write_provenance,
)
from param_decomp_lab.run_sink import RunSink
from param_decomp_lab.seed import set_seed


def _resolve_class(fqn: str) -> type:
    """Load a class from a fully-qualified name, e.g. 'transformers.LlamaForCausalLM'."""
    module_path, _, class_name = fqn.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class HFTarget(BaseConfig):
    """Load a HuggingFace model via ``<class>.from_pretrained(<hub_id>)``.

    Attributes:
        kind: Discriminator literal for `LMTargetSpec`.
        model_class: Fully-qualified class name, e.g. `transformers.GPT2LMHeadModel`.
        model_name: Hugging Face Hub identifier passed to `from_pretrained`.
    """

    kind: Literal["hf"] = "hf"
    model_class: str
    model_name: str


class PretrainedTarget(BaseConfig):
    """Load an in-repo lab-pretrained model from a wandb/local pretrain run.

    Attributes:
        kind: Discriminator literal for `LMTargetSpec`.
        model_class: Fully-qualified class name, e.g. `transformers.LlamaForCausalLM`.
        run_path: Any form `PretrainRunInfo.from_path` accepts — compact W&B
            (``entity/project/runId``), full W&B (``entity/project/runs/runId``), or a
            local checkpoint path.
    """

    kind: Literal["pretrained"] = "pretrained"
    model_class: str
    run_path: ModelPath


class HFWeightsInVendored(BaseConfig):
    """Load HF pretrained weights into a vendored `param_decomp_lab.experiments.lm.pretrain.models.*`
    architecture via `<class>.from_hf_pretrained(<hub_id>)`.

    Useful when the decomposition target needs structural changes vs HF — e.g.
    `GPT2Simple`'s separate q/k/v projections vs HF's fused `c_attn`.
    """

    kind: Literal["hf_weights_in_vendored"] = "hf_weights_in_vendored"
    model_class: str  # must expose `from_hf_pretrained`
    model_name: str  # HF hub id


LMTargetSpec = Annotated[
    HFTarget | PretrainedTarget | HFWeightsInVendored,
    Discriminator("kind"),
]
"""Discriminated union over LM target sources, keyed on ``kind`` (`"hf"` vs `"pretrained"`)."""


class LMTargetConfig(BaseConfig):
    """How to load the LM target plus how to extract its prediction tensor.

    Attributes:
        spec: Discriminated union selecting between a HuggingFace model and an in-repo
            lab-pretrained model.
        output_extract: Key/index passed to the `RunBatch` helper to pull the prediction
            tensor out of the model's forward output (defaults to `"logits"`).
    """

    spec: LMTargetSpec
    output_extract: int | str | None = "logits"
    activation_checkpointing: bool = False
    """If True and the target model exposes `enable_activation_checkpointing()`, turn on
    per-block gradient checkpointing on the frozen target forward. Trades ~33% extra
    compute for ~10-15x less stored activation memory under 2-pool — the main lever for
    raising `b_per_rank` on deep targets."""


class LMExperimentConfig(ExperimentConfig[LMTargetConfig, LMDataConfig]):
    """Full YAML schema for an LM PD run."""

    two_pool: TwoPoolConfig | None = None
    """When set, training runs under the 2-pool strategy
    (:func:`param_decomp.two_pool.optimize_two_pool`) instead of the single-process
    :func:`param_decomp.optimize.optimize`. The ``eval`` block is currently ignored
    on the 2-pool path."""

    three_pool: ThreePoolConfig | None = None
    """When set, training runs under the 3-pool strategy
    (:func:`param_decomp.three_pool.optimize_three_pool`). Mutually exclusive
    with ``two_pool``. The ``eval`` block is currently ignored on the 3-pool
    path. Requires the data loader to read the FULL global batch on every
    rank — the ``main()`` dispatch below builds the loader with
    ``dist_state=None`` when this is set."""


def build_target(target_cfg: LMTargetConfig) -> Any:
    """Load the LM target model in eval mode, dispatching on `target_cfg.spec.kind`."""
    spec = target_cfg.spec
    cls = _resolve_class(spec.model_class)
    match spec:
        case HFTarget():
            target_model = ensure_cached_and_call(cls.from_pretrained, spec.model_name)
        case PretrainedTarget():
            from param_decomp_lab.experiments.lm.pretrain.run_info import PretrainRunInfo

            run_info = ensure_cached_and_call(PretrainRunInfo.from_path, spec.run_path)
            if "model_type" not in run_info.model_config_dict:
                run_info.model_config_dict["model_type"] = spec.model_class.rsplit(".", 1)[-1]
            target_model = cls.from_run_info(run_info)
        case HFWeightsInVendored():
            assert hasattr(cls, "from_hf_pretrained"), (
                f"HFWeightsInVendored target requires {spec.model_class!r} to expose a "
                "`from_hf_pretrained` classmethod"
            )
            target_model = ensure_cached_and_call(cls.from_hf_pretrained, spec.model_name)
    _maybe_enable_activation_checkpointing(target_model, target_cfg)
    target_model.eval()
    return target_model


def _maybe_enable_activation_checkpointing(target_model: Any, target_cfg: LMTargetConfig) -> None:
    if not target_cfg.activation_checkpointing:
        return
    assert hasattr(target_model, "enable_activation_checkpointing"), (
        f"activation_checkpointing=True but {type(target_model).__name__} has no "
        "`enable_activation_checkpointing()` method"
    )
    target_model.enable_activation_checkpointing()


def build_loader(
    target_cfg: LMTargetConfig,
    data_cfg: LMDataConfig,
    *,
    split: Literal["train", "eval"],
    device: str,
    batch_size: int,
    dist_state: DistributedState | None = None,
    seed: int | None = None,
) -> DataLoader[Any]:
    """Build the LM `DataLoader` for the requested split.

    The eval seed is offset by 1 so eval shuffles differently from train when both are
    constructed from the same `pd_config.seed`.
    """
    del target_cfg, device
    effective_seed = (seed or 0) + (1 if split == "eval" else 0)
    split_name = data_cfg.eval_split if split == "eval" else data_cfg.train_split
    loader, _ = create_lm_data_loader(
        data_cfg,
        split=split_name,
        batch_size=rank_batch_size(batch_size, dist_state, label=f"{split}_batch_size"),
        seed=effective_seed,
        dist_state=dist_state,
        collate_fn=collate_fn_for(data_cfg),
    )
    return loader


def make_run_batch(target_cfg: LMTargetConfig) -> RunBatch:
    """Return the `RunBatch` callable bound to `target_cfg.output_extract`."""
    return _make_run_batch(target_cfg.output_extract)


@dataclass(frozen=True)
class SavedLMRun:
    """Handle to a completed LM PD run on disk or in W&B.

    Attributes:
        cfg: The resolved `LMExperimentConfig` from ``run_meta.yaml``.
        checkpoint_path: Resolved local path to the chosen ``model_<step>.pth`` file.
    """

    cfg: LMExperimentConfig
    checkpoint_path: Path

    @classmethod
    def from_path(cls, path: ModelPath) -> "SavedLMRun":
        """Resolve a run directory or W&B path into a fully-validated `SavedLMRun`."""
        files = resolve_run_files(
            path, config_filename=RUN_META_FILENAME, checkpoint_prefix="model"
        )
        return cls(
            cfg=LMExperimentConfig.from_file(files.config_path),
            checkpoint_path=files.checkpoint_path,
        )

    @classmethod
    def cfg_from_path(cls, path: ModelPath) -> LMExperimentConfig:
        """Load just ``run_meta.yaml`` without resolving the checkpoint.

        Useful for app endpoints that only need config introspection (e.g. the run
        picker showing architecture summaries) and want to avoid the W&B checkpoint
        download that ``from_path`` triggers.
        """
        return LMExperimentConfig.from_file(
            resolve_config_path(path, config_filename=RUN_META_FILENAME)
        )

    def load_model(self) -> ComponentModel:
        """Materialize the `ComponentModel` from the saved checkpoint."""
        return load_component_model(
            pd_config=self.cfg.pd,
            checkpoint_path=self.checkpoint_path,
            target_model=build_target(self.cfg.target),
            run_batch=make_run_batch(self.cfg.target),
        )

    def build_loader(
        self,
        *,
        split: Literal["train", "eval"],
        device: str,
        batch_size: int,
        dist_state: DistributedState | None = None,
        seed: int | None = None,
    ) -> DataLoader[Any]:
        """Rebuild a `DataLoader` for the requested split."""
        return build_loader(
            self.cfg.target,
            self.cfg.data,
            split=split,
            device=device,
            batch_size=batch_size,
            dist_state=dist_state,
            seed=seed,
        )


@with_distributed_cleanup
def main(config_path: str | Path | None = None, *, resume: str | Path | None = None) -> None:
    """Run an LM PD experiment end-to-end.

    Exactly one of ``config_path`` (fresh run) or ``--resume`` (continue a
    prior run) must be provided.

    Args:
        config_path: Path to the experiment YAML config. Required for a fresh
            run; ignored when ``--resume`` is set.
        resume: Path to a :class:`ResumeConfig` YAML pointing at a prior run
            to continue. When set, the parent run's ``run_meta.yaml`` is the
            source of cfg truth (modulo narrow ``overrides``).
    """
    if resume is not None:
        assert config_path is None, "pass either config_path or --resume, not both"
        _resume_main(Path(resume))
    else:
        assert config_path is not None, "must provide either config_path or --resume"
        _fresh_main(Path(config_path))


def _broadcast_out_dir(dist_state: DistributedState | None) -> Path:
    """Generate a run_id on rank 0 and broadcast to all ranks so the
    ResumableRunSink can write to a consistent path everywhere.
    """
    run_id = generate_run_id("param_decomp") if is_main_process() else None
    if dist_state is not None:
        run_id_box: list[str | None] = [run_id]
        dist.broadcast_object_list(run_id_box, src=0)
        run_id = run_id_box[0]
    assert isinstance(run_id, str)
    return PARAM_DECOMP_OUT_DIR / "decompositions" / run_id


def _build_resumable_sink(out_dir: Path, *, rank: int) -> ResumableRunSink:
    """Wrap a (rank-aware) :class:`RunSink` with per-rank resume-shard writes."""
    base = RunSink.local(out_dir)  # silent-noop off main rank
    return ResumableRunSink(base, run_dir=out_dir, rank=rank)


def _maybe_enable_memory_profile(rank: int) -> None:
    """Opt-in CUDA memory-history recorder for offline ``memory_viz`` analysis.

    Activated by env vars:
      * ``PD_MEMORY_PROFILE_RANKS=0,32,96`` — comma-separated ranks to profile.
      * ``PD_MEMORY_PROFILE_OUT=/abs/path/to/dir`` — dump directory.

    Dumps to ``<dir>/mem_rank<R>.pickle`` on normal exit and on uncaught
    exception. Load the pickle at https://pytorch.org/memory_viz.
    """
    prof_ranks_env = os.environ.get("PD_MEMORY_PROFILE_RANKS")
    if not prof_ranks_env:
        return
    if rank not in {int(r) for r in prof_ranks_env.split(",") if r.strip()}:
        return
    out_dir = Path(os.environ["PD_MEMORY_PROFILE_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mem_rank{rank}.pickle"
    logger.info(f"[mem-profile] recording rank={rank} → {out_path}")
    # ``stacks='python'`` captures Python tracebacks at each allocation event so
    # the offline analyzer can blame allocations on the actual nn.Module / step
    # phase that asked for them. Default is C++ stacks which are unreadable noise
    # (they all bottom out in the CUDACachingAllocator).
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=200_000,
    )

    def _dump() -> None:
        torch.cuda.memory._dump_snapshot(str(out_path))
        logger.info(f"[mem-profile] dumped rank={rank} → {out_path}")

    atexit.register(_dump)
    prev_excepthook = sys.excepthook

    def _excepthook(
        exctype: type[BaseException],
        value: BaseException,
        tb: Any,
    ) -> None:
        _dump()
        prev_excepthook(exctype, value, tb)

    sys.excepthook = _excepthook

    def _sigterm_dump(signum: int, _frame: Any) -> None:
        _dump()
        # Re-raise as default to ensure the process actually exits.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    signal.signal(signal.SIGTERM, _sigterm_dump)
    signal.signal(signal.SIGUSR1, _sigterm_dump)


def _fresh_main(config_path: Path) -> None:
    """Fresh-run path: parse YAML, build everything, train from step 0."""
    trace("_fresh_main: enter")
    cfg = LMExperimentConfig.from_file(config_path)
    trace("_fresh_main: cfg loaded")

    dist_state = init_distributed()
    trace("_fresh_main: init_distributed done")
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
    _maybe_enable_memory_profile(dist_state.rank if dist_state is not None else 0)
    set_seed(cfg.pd.seed)
    device = get_device()
    cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"device": device})})

    assert not (cfg.two_pool is not None and cfg.three_pool is not None), (
        "two_pool and three_pool are mutually exclusive; set only one."
    )

    trace("_fresh_main: broadcasting out_dir")
    out_dir = _broadcast_out_dir(dist_state)
    if is_main_process():
        cfg.to_file(out_dir / RUN_META_FILENAME)
    rank = dist_state.rank if dist_state is not None else 0
    sink = _build_resumable_sink(out_dir, rank=rank)

    trace("_fresh_main: build_target: enter")
    target_model = build_target(cfg.target)
    trace("_fresh_main: build_target: done")
    trace("_fresh_main: _build_train_loader: enter")
    train_loader = _build_train_loader(cfg, device, dist_state)
    trace("_fresh_main: _build_train_loader: done")

    try:
        trace("_fresh_main: calling _construct_and_run_trainer")
        _construct_and_run_trainer(
            cfg=cfg,
            target_model=target_model,
            train_loader=train_loader,
            sink=sink,
            device=device,
            dist_state=dist_state,
        )
    finally:
        sink.finish()


def _resume_main(resume_cfg_path: Path) -> None:
    """Resume-run path: load parent cfg + per-rank shard, continue training."""
    resume_cfg = ResumeConfig.from_file(resume_cfg_path)
    parent_cfg = LMExperimentConfig.from_file(resume_cfg.from_run / RUN_META_FILENAME)

    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
        logger.info(f"Resuming from {resume_cfg.from_run} @ step {resume_cfg.step}")
    set_seed(parent_cfg.pd.seed)
    device = get_device()
    rank = dist_state.rank if dist_state is not None else 0

    # Apply ResumeOverrides to the parent cfg for the rest of the lab's wiring
    # (eval, cadence, loader, etc.). Trainer.from_snapshot applies the same
    # patch internally to validate the saved pd_config.
    cfg_overrides = (
        resume_cfg.overrides.to_pd_config_patch() if resume_cfg.overrides is not None else None
    )
    effective_pd = (
        parent_cfg.pd.model_copy(update=cfg_overrides)
        if cfg_overrides is not None
        else parent_cfg.pd
    )
    effective_cfg = parent_cfg.model_copy(
        update={
            "pd": effective_pd,
            "runtime": parent_cfg.runtime.model_copy(update={"device": device}),
        }
    )

    assert not (effective_cfg.two_pool is not None and effective_cfg.three_pool is not None), (
        "parent's two_pool and three_pool are both set — corrupt run_meta.yaml"
    )

    out_dir = _broadcast_out_dir(dist_state)
    if is_main_process():
        effective_cfg.to_file(out_dir / RUN_META_FILENAME)
        resolved_step = resolve_step(resume_cfg.from_run, resume_cfg.step)
        write_provenance(
            out_dir,
            ResumeProvenance(parent_run_dir=resume_cfg.from_run, parent_step=resolved_step),
        )
    sink = _build_resumable_sink(out_dir, rank=rank)

    target_model = build_target(effective_cfg.target)
    train_loader = _build_train_loader(effective_cfg, device, dist_state)
    snapshot = read_resume_snapshot(resume_cfg, rank=rank, current_device=device)

    try:
        _construct_and_run_trainer(
            cfg=effective_cfg,
            target_model=target_model,
            train_loader=train_loader,
            sink=sink,
            device=device,
            dist_state=dist_state,
            resume_snapshot=snapshot,
            cfg_overrides=cfg_overrides,
        )
    finally:
        sink.finish()


def _build_train_loader(
    cfg: LMExperimentConfig, device: str, dist_state: DistributedState | None
) -> DataLoader[Any]:
    """Construct the train loader, accounting for the multi-pool full-batch rule.

    Both 2-pool and 3-pool require the FULL global batch on every rank: each
    pool slices it locally (``my_batch_slice_a`` / ``my_batch_slice_b`` in
    2-pool; pool-specific helpers in 3-pool). Only the single-pool path uses
    the standard `DistributedSampler`-sharded loader.
    """
    is_multi_pool = cfg.two_pool is not None or cfg.three_pool is not None
    loader_dist_state = None if is_multi_pool else dist_state
    return build_loader(
        cfg.target,
        cfg.data,
        split="train",
        device=device,
        batch_size=cfg.pd.batch_size,
        dist_state=loader_dist_state,
        seed=cfg.pd.seed,
    )


def _construct_and_run_trainer(
    *,
    cfg: LMExperimentConfig,
    target_model: Any,
    train_loader: DataLoader[Any],
    sink: ResumableRunSink,
    device: str,
    dist_state: DistributedState | None,
    resume_snapshot: Any | None = None,
    cfg_overrides: dict[str, Any] | None = None,
) -> None:
    """Build the right concrete trainer for the cfg's pool kind and run it.

    Fresh and resume share this code path; ``resume_snapshot`` switches
    between :meth:`Trainer` and :meth:`Trainer.from_snapshot` construction.
    """
    run_batch = make_run_batch(cfg.target)
    if cfg.three_pool is not None:
        # 3-pool path: eval not wired through yet — cfg.eval (if set) is ignored.
        if resume_snapshot is not None:
            trainer = ThreePoolTrainer.from_snapshot(
                resume_snapshot,
                target_model=target_model,
                run_batch=run_batch,
                reconstruction_loss=recon_loss_kl,
                cfg_overrides=cfg_overrides,
            )
        else:
            trainer = ThreePoolTrainer(
                target_model=target_model,
                run_batch=run_batch,
                reconstruction_loss=recon_loss_kl,
                pd_config=cfg.pd,
                runtime_config=cfg.runtime,
                three_pool_config=cfg.three_pool,
            )
        trainer.run(train_loader, sink, cfg.cadence)
    elif cfg.two_pool is not None:
        # 2-pool path: eval not wired through yet — cfg.eval (if set) is ignored.
        if resume_snapshot is not None:
            two_trainer = TwoPoolTrainer.from_snapshot(
                resume_snapshot,
                target_model=target_model,
                run_batch=run_batch,
                reconstruction_loss=recon_loss_kl,
                cadence=cfg.cadence,
                cfg_overrides=cfg_overrides,
            )
        else:
            two_trainer = TwoPoolTrainer(
                target_model=target_model,
                run_batch=run_batch,
                reconstruction_loss=recon_loss_kl,
                pd_config=cfg.pd,
                runtime_config=cfg.runtime,
                two_pool_config=cfg.two_pool,
                cadence=cfg.cadence,
            )
        two_trainer.run(train_loader, sink, cfg.cadence)
    else:
        eval_loop = _build_eval_loop(cfg, device, dist_state)
        if resume_snapshot is not None:
            one_trainer = Trainer.from_snapshot(
                resume_snapshot,
                target_model=target_model,
                run_batch=run_batch,
                reconstruction_loss=recon_loss_kl,
                cfg_overrides=cfg_overrides,
            )
        else:
            one_trainer = Trainer(
                target_model=target_model,
                run_batch=run_batch,
                reconstruction_loss=recon_loss_kl,
                pd_config=cfg.pd,
                runtime_config=cfg.runtime,
            )
        one_trainer.run(train_loader, sink, cfg.cadence, eval_loop)


def _build_eval_loop(
    cfg: LMExperimentConfig,
    device: str,
    dist_state: DistributedState | None,
) -> EvalLoop | None:
    """Build the optional `EvalLoop` from `cfg.eval`, returning None when eval is disabled."""
    if cfg.eval is None:
        return None
    eval_loader = build_loader(
        cfg.target,
        cfg.data,
        split="eval",
        device=device,
        batch_size=cfg.eval.batch_size,
        dist_state=dist_state,
        seed=cfg.pd.seed,
    )
    return EvalLoop(
        loader=eval_loader,
        metrics=[EVAL_METRIC_CLASSES[m.type](m) for m in cfg.eval.metrics],
        n_steps=cfg.eval.n_steps,
        every=cfg.eval.every,
        slow_every=cfg.eval.slow_every,
        slow_on_first_step=cfg.eval.slow_on_first_step,
    )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
