"""`FsdpLMTrainer` — the single-pool FSDP2 LM PD trainer.

Runs the full VPD algorithm (faith + importance-min + stochastic-recon +
persistent-PGD, authored as ordinary ``pd.loss_metrics``) through the SHARED step
helpers in :mod:`param_decomp.train_step`, but scales the **vendored**
``LMComponentModel`` via FSDP2 (memory) + torch.compile (speed) instead of DDP.

This is the integration hub: it composes the sibling ``fsdp/`` modules
(``component_adapter`` / ``config`` / ``checkpoint`` / ``sdpa_strict`` /
``grad_clip``) and mirrors the construction + loop-tail of
:class:`param_decomp.optimize.Trainer`, diverging only where the model wrap
(FSDP2, not DDP), the save path (sharded DCP, not a full ``TrainingState`` on the
loop), and residual-start require it.

Construction (per ``CONTRACTS.md`` build sequence, steps 1-8):
  1. ``insert_identity_operations_`` (if configured) + freeze + eval +
     ``resolve_decomposition_targets`` (mirror core ``Trainer.__init__``).
  2. ``LMComponentModel.build`` on CPU.
  3. If ``shard_frozen_target``: convert each ``ComponentLinear``'s frozen
     ``target_weight`` / ``bias`` BUFFER to a no-grad ``nn.Parameter`` so FSDP2
     shards the 8B target (buffers are replicated; params are sharded).
  4. If ``checkpoint_blocks``: enable per-block activation checkpointing on the
     vendored target.
  5. ``fully_shard`` each transformer block (target + ci-fn) and the roots,
     then ``.to(device)``.
  6. Per-rank inductor/triton cache dirs, then ``model.compile()`` /
     ``ci_fn.compile()`` per flags.
  7. Wrap in ``FsdpComponentAdapter``; build the two AdamW optimizers; instantiate
     the loss metrics bound to the adapter.
  8. ``verify_flash_attention_available`` once.

The reconstruction loss is the LM KL loss (``recon_loss_kl``) — single-sourced
here rather than threaded through the constructor, because this path is LM-only
and the recon loss is invariant across every LM trainer (single-pool, 2-pool,
3-pool all use ``recon_loss_kl``).
"""

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Self, cast

import torch
import torch._dynamo
import torch.nn as nn
from torch import optim
from torch.distributed.fsdp import fully_shard
from torch.utils.data import DataLoader

from param_decomp.batch_and_loss_fns import ReconstructionLoss, move_batch_to_device
from param_decomp.ci_fns import (
    GlobalCiFnWrapper,
    GlobalSharedTransformerCiFn,
    LayerwiseCiFnWrapper,
)
from param_decomp.component_model import ComponentModelProtocol
from param_decomp.decomposition_targets import (
    insert_identity_operations_,
    resolve_decomposition_targets,
)
from param_decomp.distributed import (
    avg_metrics_across_ranks,
    get_distributed_state,
    is_main_process,
    seed_all_ranks,
    seed_per_rank,
    sync_across_processes,
)
from param_decomp.faithfulness_warmup import run_faithfulness_warmup
from param_decomp.log import logger
from param_decomp.metrics.base import Metric
from param_decomp.metrics.persistent_pgd_recon import validate_pgd_scope
from param_decomp.optimize import tie_component_weights
from param_decomp.run_sink import ThreePoolRunSink
from param_decomp.torch_helpers import loop_dataloader
from param_decomp.train_step import (
    EvalLoop,
    _install_sigterm_flag,
    empty_cuda_cache_and_collect,
    run_eval_pass,
    run_loss_step,
    scheduled_lrs,
)
from param_decomp_config.pd import Cadence, PDConfig
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.experiments.lm.vendored.component_model import (
    ComponentTarget,
    LMComponentModel,
)
from param_decomp_lab.experiments.lm.vendored.gpt2 import ComponentGPT2
from param_decomp_lab.experiments.lm.vendored.llama_3_1.components import (
    ComponentLinear,
    ComponentLlama,
)
from param_decomp_lab.fsdp.checkpoint import load_dcp, save_dcp
from param_decomp_lab.fsdp.component_adapter import FsdpComponentAdapter
from param_decomp_lab.fsdp.config import FsdpRuntimeConfig
from param_decomp_lab.fsdp.consolidate import (
    CI_FN_OPTIMIZER_NAME,
    COMPONENTS_OPTIMIZER_NAME,
)
from param_decomp_lab.fsdp.grad_clip import clip_grad_norm_no_sync
from param_decomp_lab.fsdp.sdpa_strict import verify_flash_attention_available
from param_decomp_lab.metrics.dispatch import instantiate_lab_metrics

_FA_PROBE_SEQ_LEN = 512
"""Representative seq len for the startup flash-attention dispatch probe. FA
dispatchability is governed by head_dim / dtype / is_causal, not the exact seq
len, so a fixed representative length suffices (the production seq len lives in
the data config, not the trainer)."""


def _convert_frozen_buffers_to_params_(model: LMComponentModel) -> None:
    """Convert each ``ComponentLinear``'s frozen ``target_weight`` / ``bias`` BUFFER
    to a no-grad ``nn.Parameter`` so FSDP2 shards it.

    FSDP2 shards parameters (including ``requires_grad=False`` ones) but replicates
    buffers on every rank. The 8B target's decomposed-layer weights live in
    ``ComponentLinear.target_weight`` buffers; left as buffers they'd be replicated
    and defeat the point of FSDP. (The embedding / lm_head / norm weights are already
    no-grad ``nn.Parameter``s — frozen by ``componentize_*`` — so FSDP shards them
    without conversion.)
    """
    for module in model.modules():
        if not isinstance(module, ComponentLinear):
            continue
        target_weight = module.target_weight
        delattr(module, "target_weight")
        module.register_parameter("target_weight", nn.Parameter(target_weight, requires_grad=False))
        bias = module.bias
        if bias is not None:
            delattr(module, "bias")
            module.register_parameter("bias", nn.Parameter(bias, requires_grad=False))


def _target_transformer_blocks(model: ComponentTarget) -> list[nn.Module]:
    """The vendored target's transformer blocks, for per-block ``fully_shard``.

    ComponentGPT2 keeps them in ``_h``; ComponentLlama in ``_layers`` (each is the
    plain Python list backing the ``nn.ModuleList``)."""
    match model:
        case ComponentGPT2():
            return list(model._h)
        case ComponentLlama():
            return list(model._layers)


def _ci_fn_transformer_blocks(
    ci_fn: GlobalCiFnWrapper | LayerwiseCiFnWrapper,
) -> list[nn.Module]:
    """Every transformer block inside the CI fn, for per-block ``fully_shard``.

    Both wrappers ultimately hold ``GlobalSharedTransformerCiFn`` instances (the
    layerwise wrapper wraps one per site); the shardable units are those modules'
    ``_blocks`` ModuleLists.
    """
    blocks: list[nn.Module] = []
    for module in ci_fn.modules():
        if isinstance(module, GlobalSharedTransformerCiFn):
            blocks.extend(module._blocks)
    return blocks


def _set_per_rank_compile_cache_env() -> None:
    """Per-rank inductor / triton cache dirs, set BEFORE the first ``.compile()``.

    Defensive against shared-cache contention across the concurrent compilers when
    ``/tmp`` is shared across ranks on a node (mirrors the 3-pool recipe)."""
    dist_state = get_distributed_state()
    rank = dist_state.rank if dist_state is not None else 0
    user = os.environ.get("USER", "u")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_{user}_r{rank}"
    os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_{user}_r{rank}"


class FsdpLMTrainer:
    """Single-pool FSDP2 LM PD trainer.

    Construction shards the vendored ``LMComponentModel`` with FSDP2, optionally
    compiles it, wraps it in an ``FsdpComponentAdapter``, and builds the two
    optimizers + loss metrics. :meth:`run` advances the loop from ``self.step`` to
    ``pd_config.steps``, saving via sharded DCP and firing the sink's ``on_save``
    (the composition root wires the async consolidate + slow-eval there).
    """

    pd_config: PDConfig
    runtime_config: FsdpRuntimeConfig
    reconstruction_loss: ReconstructionLoss
    lm: LMComponentModel
    adapter: FsdpComponentAdapter
    components_optimizer: optim.Optimizer
    ci_fn_optimizer: optim.Optimizer
    loss_metrics: dict[str, Metric[Any]]
    step: int

    def __init__(
        self,
        *,
        target_model: nn.Module,
        pd_config: PDConfig,
        runtime_config: FsdpRuntimeConfig,
    ) -> None:
        self.pd_config = pd_config
        self.runtime_config = runtime_config
        self.reconstruction_loss = recon_loss_kl
        self.step = 0

        device = runtime_config.device
        dist_state = get_distributed_state()
        world_size = dist_state.world_size if dist_state is not None else 1

        validate_pgd_scope(
            pd_config.loss_metrics, batch_size=pd_config.batch_size, world_size=world_size
        )

        # --- 1. identity ops + freeze + resolve targets (mirror core Trainer) ---
        if pd_config.identity_decomposition_targets is not None:
            insert_identity_operations_(
                target_model,
                identity_decomposition_targets=pd_config.identity_decomposition_targets,
            )
        target_model.requires_grad_(False)
        target_model.eval()
        decomposition_targets = resolve_decomposition_targets(
            target_model, pd_config.all_decomposition_target_configs
        )

        # --- 2. build the vendored component model on CPU ---
        seed_all_ranks(pd_config.seed)
        self.lm = LMComponentModel.build(
            target_model=target_model,
            decomposition_targets=decomposition_targets,
            ci_config=pd_config.ci_config,
            sigmoid_type=pd_config.sigmoid_type,
        )
        ci_fn = self.lm.ci_fn
        assert ci_fn is not None, "single-pool FSDP trainer requires the CI fn intact"

        # Tie weights on the unsharded CPU components, before `fully_shard` replaces
        # `.data` with DTensor shards (tying mutates `.data` directly).
        if pd_config.tied_weights is not None:
            tie_component_weights(self.lm, pd_config.tied_weights)

        # --- 3. frozen target buffers -> no-grad params so FSDP2 shards them ---
        if runtime_config.shard_frozen_target:
            _convert_frozen_buffers_to_params_(self.lm)

        # --- 4. per-block activation checkpointing on the target ---
        if runtime_config.checkpoint_blocks:
            self.lm.model.enable_activation_checkpointing()

        # --- 5. fully_shard each transformer block + the ci-fn, then move to device ---
        # Only the transformer blocks (which hold every trainable component V/U and, under
        # `shard_frozen_target`, the in-block frozen target weights) and the ci-fn are sharded.
        # The frozen embedding / final-norm / lm_head are deliberately left REPLICATED: residual
        # start runs the embedding via `residual_at`, which bypasses the root module's forward
        # hook, so a root-sharded embedding weight (a DTensor) would meet the plain-tensor token
        # ids and raise `aten.embedding got mixed torch.Tensor and DTensor`. Giving the embedding
        # its own `fully_shard` instead makes it a root entered before the parent forward and
        # trips FSDP2's single-root lazy-init. Leaving these few frozen modules replicated (~1GB
        # for the 8B target) sidesteps both and is negligible against the 32 sharded blocks.
        for block in _target_transformer_blocks(self.lm.model):
            fully_shard(block)
        for block in _ci_fn_transformer_blocks(ci_fn):
            fully_shard(block)
        fully_shard(ci_fn)
        self.lm = self.lm.to(device)

        # --- 6. per-rank compile caches, then compile per flags ---
        if runtime_config.compile_model or runtime_config.compile_ci_fn:
            _set_per_rank_compile_cache_env()
            # The masked block forward specializes per block instance (dynamo guards on each
            # block's submodule identity), so a target with more than ~8 blocks in the
            # decomposed suffix exceeds dynamo's default recompile_limit (8) and SILENTLY falls
            # back to eager — losing the compile win. Raise the ceiling above the block count
            # (cost: front-loaded recompiles at startup; steady state stays compiled).
            n_blocks = len(_target_transformer_blocks(self.lm.model))
            torch._dynamo.config.recompile_limit = max(
                torch._dynamo.config.recompile_limit, 8 * n_blocks
            )
        if runtime_config.compile_model:
            self.lm.model.compile()
        if runtime_config.compile_ci_fn:
            ci_fn.compile()

        # Diverge stochastic RNG per rank so masks/sources differ across DP workers.
        seed_per_rank(pd_config.seed)

        # --- 7. adapter + optimizers + metrics ---
        self.adapter = FsdpComponentAdapter(self.lm)

        self._component_params: list[nn.Parameter] = []
        for module_path in self.lm.target_module_paths:
            self._component_params.extend(self.lm.components[module_path].parameters())
        assert self._component_params, "no component params to optimize"
        self._ci_fn_params = list(ci_fn.parameters())

        self.components_optimizer = optim.AdamW(
            self._component_params,
            lr=pd_config.components_optimizer.lr_schedule.start_val,
            betas=pd_config.components_optimizer.betas,
            weight_decay=pd_config.components_optimizer.weight_decay,
        )
        self.ci_fn_optimizer = optim.AdamW(
            self._ci_fn_params,
            lr=pd_config.ci_fn_optimizer.lr_schedule.start_val,
            betas=pd_config.ci_fn_optimizer.betas,
            weight_decay=pd_config.ci_fn_optimizer.weight_decay,
        )

        self.loss_metrics = instantiate_lab_metrics(pd_config, self.adapter, device)

        # --- 8. flash-attention dispatch probe ---
        cfg = self.lm.model.config
        verify_flash_attention_available(
            head_dim=cfg.n_embd // cfg.n_head,
            n_heads=cfg.n_head,
            seq_len=_FA_PROBE_SEQ_LEN,
            is_causal=True,
            device=torch.device(device),
            dtype=torch.bfloat16,
        )

    # ============================ Resume (DCP) ============================

    @property
    def _optimizers_by_name(self) -> dict[str, optim.Optimizer]:
        """The two optimizers keyed by their stable DCP name (matches ``checkpoint.py``
        / ``consolidate.py`` so a save round-trips into either trainer)."""
        return {
            COMPONENTS_OPTIMIZER_NAME: self.components_optimizer,
            CI_FN_OPTIMIZER_NAME: self.ci_fn_optimizer,
        }

    def load_from_dcp(self, run_dir: Path, step: int) -> None:
        """Resume-in-place from sharded DCP shards at ``run_dir/.dcp/step_<step>/``.

        Loads the trainable model + optimizer shards directly into the live
        FSDP2-sharded model (no full-gather), and the loss-metric states via the
        freshly-built metrics' ``state_dict()`` skeleton (DCP fills it in place).
        Advances ``self.step`` to ``step``.
        """
        skeleton = {name: m.state_dict() for name, m in self.loss_metrics.items()}
        loaded = load_dcp(
            self.lm,
            self._optimizers_by_name,
            step=step,
            in_dir=run_dir,
            loss_metric_states=skeleton,
        )
        for name, m in self.loss_metrics.items():
            m.load_state_dict(loaded[name])
        self.step = step

    @classmethod
    def from_dcp(
        cls,
        run_dir: Path,
        step: int,
        *,
        target_model: nn.Module,
        pd_config: PDConfig,
        runtime_config: FsdpRuntimeConfig,
    ) -> Self:
        """Build a fresh trainer at the production topology, then load DCP shards into it."""
        trainer = cls(target_model=target_model, pd_config=pd_config, runtime_config=runtime_config)
        trainer.load_from_dcp(run_dir, step)
        return trainer

    # ============================ Training loop ============================

    def run(
        self,
        train_loader: DataLoader[Any],
        sink: ThreePoolRunSink,
        cadence: Cadence,
        run_dir: Path,
        eval_loop: EvalLoop | None = None,
    ) -> None:
        """Advance training from ``self.step`` to ``pd_config.steps``.

        Mirrors :meth:`param_decomp.optimize.Trainer.run`, diverging where FSDP
        requires it:
          - ``run_loss_step`` is wrapped in ``adapter.use_cached_residual(batch)``
            for residual-start.
          - in-train eval is FAST-ONLY (slow eval is async, off-loop).
          - saves write sharded DCP shards to ``run_dir/.dcp/step_<S>/`` (the same
            ``run_dir`` that ``latest_dcp_step`` / ``consolidate`` read) then call
            ``sink.checkpoint_written(step, final=...)``, which fires the rank-0
            ``on_save`` hook (async consolidate + slow-eval). The DCP shards ARE the
            persistent on-loop checkpoint (resume-in-place loads them directly) — not
            throwaway scratch — so they live in the run dir. No full ``TrainingState``
            on the loop.
          - grad clip uses the DTensor-aware no-sync path.
          - SIGTERM saves a sharded DCP checkpoint then breaks (SLURM requeue resumes).
        """
        pd_config = self.pd_config
        runtime_config = self.runtime_config
        device = runtime_config.device

        train_iterator = loop_dataloader(train_loader)
        eval_iterator = loop_dataloader(eval_loop.loader) if eval_loop is not None else None

        for _ in range(self.step):
            next(train_iterator)

        component_model = self.adapter

        if self.step == 0 and pd_config.faithfulness_warmup_steps > 0:
            run_faithfulness_warmup(component_model, self._component_params, pd_config)

        all_instances = self._build_all_metric_instances(eval_loop, device, component_model)
        sigterm = _install_sigterm_flag()

        # Steady-state throughput: averaged over each train-log interval with a CUDA sync, so
        # `tok_per_s_per_gpu` is real GPU time (the apples-to-apples number vs the JAX path).
        # The first interval includes the one-time torch.compile cost; read later intervals.
        world_size = self.runtime_config.dp or 1
        last_log_t: float | None = None
        last_log_step = self.step

        for step in range(self.step, pd_config.steps + 1):
            self.step = step
            self.components_optimizer.zero_grad()
            self.ci_fn_optimizer.zero_grad()

            components_lr, ci_fn_lr = scheduled_lrs(
                step, total_steps=pd_config.steps, config=pd_config
            )
            for group in self.components_optimizer.param_groups:
                group["lr"] = components_lr
            for group in self.ci_fn_optimizer.param_groups:
                group["lr"] = ci_fn_lr

            # Move to device before `use_cached_residual`: residual-start runs the embedding
            # on the raw batch outside `run_loss_step` (which does its own device move), so the
            # token ids must already be on-device or the embedding lookup mixes cpu/cuda.
            batch = move_batch_to_device(next(train_iterator), device)
            with self.adapter.use_cached_residual(batch):
                _, batch_log_data = run_loss_step(
                    batch=batch,
                    step=step,
                    device=device,
                    wrapped_model=self.adapter,
                    component_model=component_model,
                    loss_metrics=self.loss_metrics,
                    config=pd_config,
                    reconstruction_loss=self.reconstruction_loss,
                    autocast_bf16=runtime_config.autocast_bf16,
                )

            if cadence.should_log_train(step):
                torch.cuda.synchronize(device)
                now = time.perf_counter()
                if last_log_t is not None and step > last_log_step:
                    n_interval = step - last_log_step
                    elapsed = now - last_log_t
                    seq_len = batch.shape[-1]
                    tok_per_s = pd_config.batch_size * seq_len * n_interval / elapsed
                    batch_log_data["perf/step_time_s"] = elapsed / n_interval
                    batch_log_data["perf/tok_per_s"] = tok_per_s
                    batch_log_data["perf/tok_per_s_per_gpu"] = tok_per_s / world_size
                last_log_t = now
                last_log_step = step
                batch_log_data["mem/peak_gb_per_rank"] = (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                )
                batch_log_data = cast(
                    defaultdict[str, float],
                    avg_metrics_across_ranks(batch_log_data, device=device),
                )
                batch_log_data["schedules/lr/components"] = components_lr
                batch_log_data["schedules/lr/ci_fn"] = ci_fn_lr
                sink.console(
                    f"--- Step {step} ---",
                    f"LR[components]: {components_lr:.6f}",
                    f"LR[ci_fn]: {ci_fn_lr:.6f}",
                    *(f"train/{name}: {value:.15f}" for name, value in batch_log_data.items()),
                )
                sink.log({f"train/{k}": v for k, v in batch_log_data.items()}, step=step)

            if eval_loop is not None and eval_loop.should_eval(step):
                assert eval_iterator is not None
                fast_metrics, _ = run_eval_pass(
                    eval_iterator=eval_iterator,
                    n_steps=eval_loop.n_steps,
                    slow_step=False,
                    all_instances=all_instances,
                    step=step,
                    device=device,
                    wrapped_model=self.adapter,
                    component_model=component_model,
                    config=pd_config,
                    reconstruction_loss=self.reconstruction_loss,
                    autocast_bf16=runtime_config.autocast_bf16,
                )
                sink.console(*(f"eval/{k}: {v}" for k, v in fast_metrics.items()))
                sink.log({f"eval/{k}": v for k, v in fast_metrics.items()}, step=step)
                empty_cuda_cache_and_collect()

            if step == pd_config.steps or cadence.should_save(step) or sigterm.received:
                is_final = step == pd_config.steps or sigterm.received
                save_dcp(
                    self.lm,
                    self._optimizers_by_name,
                    step=step,
                    loss_metric_states={n: m.state_dict() for n, m in self.loss_metrics.items()},
                    out_dir=run_dir,
                )
                sink.checkpoint_written(step, final=is_final)
            if sigterm.received:
                if is_main_process():
                    logger.info(f"SIGTERM received; saved DCP checkpoint at step {step}, exiting")
                break

            if step != pd_config.steps:
                sync_across_processes()
                if pd_config.components_optimizer.grad_clip_norm is not None:
                    clip_grad_norm_no_sync(
                        self._component_params, pd_config.components_optimizer.grad_clip_norm
                    )
                if pd_config.ci_fn_optimizer.grad_clip_norm is not None:
                    clip_grad_norm_no_sync(
                        self._ci_fn_params, pd_config.ci_fn_optimizer.grad_clip_norm
                    )
                self.components_optimizer.step()
                self.ci_fn_optimizer.step()

        if is_main_process():
            logger.info("Finished training loop.")

    def _build_all_metric_instances(
        self,
        eval_loop: EvalLoop | None,
        device: str,
        component_model: ComponentModelProtocol,
    ) -> dict[str, Metric[Any]]:
        """Merge loss + eval-only metric instances keyed by class name (mirror core
        ``Trainer._build_all_metric_instances``); rejects name collisions."""
        eval_only_instances: dict[str, Metric[Any]] = {}
        if eval_loop is not None:
            for m in eval_loop.metrics:
                m.bind(model=component_model, device=device)
                metric_name = type(m).__name__
                assert metric_name not in eval_only_instances, (
                    f"duplicate eval metric {metric_name!r}"
                )
                eval_only_instances[metric_name] = m
            overlap = sorted(set(self.loss_metrics) & set(eval_only_instances))
            assert not overlap, (
                f"eval_loop.metrics overlap with pd_config.loss_metrics: {overlap}. Loss "
                "metrics are automatically evaluated; remove the duplicates from eval_loop.metrics."
            )
        return {**self.loss_metrics, **eval_only_instances}
