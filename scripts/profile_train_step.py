"""Profile real param-decomp LM training steps without evals.

This intentionally mirrors the train-step path in ``run_param_decomp.optimize`` while adding
phase timers around the expensive regions. Run under torchrun for distributed profiling, e.g.

    torchrun --standalone --nproc_per_node=8 scripts/profile_train_step.py \
        --config-path param_decomp/experiments/lm/pile_llama_simple_mlp-4L.yaml \
        --batch-size 64 \
        --out-dir profiling_runs/train-step-n8
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, IterableDataset

from param_decomp.configs import (
    Config,
    LMTaskConfig,
    LossMetricConfigType,
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
    RepeatAcrossBatchScope,
)
from param_decomp.data import (
    DatasetConfig,
    create_data_loader,
    input_ids_collate_fn,
    loop_dataloader,
)
from param_decomp.identity_insertion import insert_identity_operations_
from param_decomp.log import logger
from param_decomp.losses import compute_losses
from param_decomp.metrics import faithfulness_loss
from param_decomp.models.batch_and_loss_fns import (
    ReconstructionLoss,
    RunBatch,
    make_run_batch,
    recon_loss_kl,
)
from param_decomp.models.component_model import (
    ComponentModel,
    OutputWithCache,
    move_batch_to_device,
)
from param_decomp.persistent_pgd import PersistentPGDState
from param_decomp.pretrain.run_info import PretrainRunInfo
from param_decomp.routing import AllLayersRouter
from param_decomp.utils.component_utils import calc_ci_l_zero
from param_decomp.utils.distributed_utils import (
    DistributedState,
    avg_metrics_across_ranks,
    cleanup_distributed,
    ensure_cached_and_call,
    get_device,
    get_distributed_state,
    init_distributed,
    is_main_process,
    seed_per_rank,
    sync_across_processes,
)
from param_decomp.utils.general_utils import (
    bf16_autocast,
    dict_safe_update_,
    get_scheduled_value,
    replace_pydantic_model,
    resolve_class,
    set_seed,
)
from param_decomp.utils.logging_utils import get_grad_norms_dict
from param_decomp.utils.module_utils import expand_module_patterns
from param_decomp.utils.run_utils import parse_config, save_file

Strategy = Literal["ddp", "zero1", "none"]


@dataclass
class PhaseStats:
    count: int = 0
    cpu_ms_sum: float = 0.0
    cuda_ms_sum: float = 0.0
    cpu_ms_max: float = 0.0
    cuda_ms_max: float = 0.0
    peak_allocated_gb_max: float = 0.0
    peak_reserved_gb_max: float = 0.0
    peak_delta_gb_max: float = 0.0
    allocated_after_gb_max: float = 0.0

    def add(
        self,
        *,
        cpu_ms: float,
        cuda_ms: float,
        peak_allocated_gb: float,
        peak_reserved_gb: float,
        peak_delta_gb: float,
        allocated_after_gb: float,
    ) -> None:
        self.count += 1
        self.cpu_ms_sum += cpu_ms
        self.cuda_ms_sum += cuda_ms
        self.cpu_ms_max = max(self.cpu_ms_max, cpu_ms)
        self.cuda_ms_max = max(self.cuda_ms_max, cuda_ms)
        self.peak_allocated_gb_max = max(self.peak_allocated_gb_max, peak_allocated_gb)
        self.peak_reserved_gb_max = max(self.peak_reserved_gb_max, peak_reserved_gb)
        self.peak_delta_gb_max = max(self.peak_delta_gb_max, peak_delta_gb)
        self.allocated_after_gb_max = max(self.allocated_after_gb_max, allocated_after_gb)

    def to_json(self) -> dict[str, float | int]:
        cpu_avg = self.cpu_ms_sum / self.count if self.count else 0.0
        cuda_avg = self.cuda_ms_sum / self.count if self.count else 0.0
        return {
            "count": self.count,
            "cpu_ms_avg": cpu_avg,
            "cuda_ms_avg": cuda_avg,
            "cpu_ms_max": self.cpu_ms_max,
            "cuda_ms_max": self.cuda_ms_max,
            "peak_allocated_gb_max": self.peak_allocated_gb_max,
            "peak_reserved_gb_max": self.peak_reserved_gb_max,
            "peak_delta_gb_max": self.peak_delta_gb_max,
            "allocated_after_gb_max": self.allocated_after_gb_max,
        }


@dataclass
class StepStats:
    count: int = 0
    cpu_ms_sum: float = 0.0
    cuda_ms_sum: float = 0.0
    cpu_ms_max: float = 0.0
    cuda_ms_max: float = 0.0
    losses: dict[str, float] = field(default_factory=dict)

    def add(self, *, cpu_ms: float, cuda_ms: float, losses: dict[str, float]) -> None:
        self.count += 1
        self.cpu_ms_sum += cpu_ms
        self.cuda_ms_sum += cuda_ms
        self.cpu_ms_max = max(self.cpu_ms_max, cpu_ms)
        self.cuda_ms_max = max(self.cuda_ms_max, cuda_ms)
        self.losses = losses

    def to_json(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "cpu_ms_avg": self.cpu_ms_sum / self.count if self.count else 0.0,
            "cuda_ms_avg": self.cuda_ms_sum / self.count if self.count else 0.0,
            "cpu_ms_max": self.cpu_ms_max,
            "cuda_ms_max": self.cuda_ms_max,
            "last_losses": self.losses,
        }


class PhaseTimer:
    def __init__(
        self,
        profiler: TrainStepProfiler,
        name: str,
    ) -> None:
        self._profiler = profiler
        self._name = name
        self._record_ctx: Any | None = None
        self._cpu_start = 0.0
        self._start_event: torch.cuda.Event | None = None
        self._end_event: torch.cuda.Event | None = None
        self._base_allocated = 0

    def __enter__(self) -> None:
        self._record_ctx = record_function(self._name)
        self._record_ctx.__enter__()

        if not self._profiler.enabled:
            return

        if self._profiler.cuda:
            torch.cuda.synchronize(self._profiler.device)
            self._base_allocated = torch.cuda.memory_allocated(self._profiler.device)
            torch.cuda.reset_peak_memory_stats(self._profiler.device)
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        self._cpu_start = time.perf_counter()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._profiler.enabled:
            cuda_ms = 0.0
            if self._profiler.cuda:
                assert self._start_event is not None
                assert self._end_event is not None
                self._end_event.record()
                torch.cuda.synchronize(self._profiler.device)
                cuda_ms = self._start_event.elapsed_time(self._end_event)
                peak_allocated = torch.cuda.max_memory_allocated(self._profiler.device)
                peak_reserved = torch.cuda.max_memory_reserved(self._profiler.device)
                allocated_after = torch.cuda.memory_allocated(self._profiler.device)
            else:
                peak_allocated = 0
                peak_reserved = 0
                allocated_after = 0

            cpu_ms = 1000 * (time.perf_counter() - self._cpu_start)
            self._profiler.add_phase(
                self._name,
                cpu_ms=cpu_ms,
                cuda_ms=cuda_ms,
                peak_allocated_gb=peak_allocated / 1e9,
                peak_reserved_gb=peak_reserved / 1e9,
                peak_delta_gb=max(0.0, (peak_allocated - self._base_allocated) / 1e9),
                allocated_after_gb=allocated_after / 1e9,
            )

        if self._record_ctx is not None:
            self._record_ctx.__exit__(exc_type, exc, tb)


class TrainStepProfiler:
    def __init__(self, device: torch.device | str, enabled: bool) -> None:
        self.device = torch.device(device)
        self.enabled = enabled
        self.cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.phases: defaultdict[str, PhaseStats] = defaultdict(PhaseStats)
        self.steps = StepStats()

    def phase(self, name: str) -> PhaseTimer:
        return PhaseTimer(self, name)

    def add_phase(
        self,
        name: str,
        *,
        cpu_ms: float,
        cuda_ms: float,
        peak_allocated_gb: float,
        peak_reserved_gb: float,
        peak_delta_gb: float,
        allocated_after_gb: float,
    ) -> None:
        self.phases[name].add(
            cpu_ms=cpu_ms,
            cuda_ms=cuda_ms,
            peak_allocated_gb=peak_allocated_gb,
            peak_reserved_gb=peak_reserved_gb,
            peak_delta_gb=peak_delta_gb,
            allocated_after_gb=allocated_after_gb,
        )

    def step_start(self) -> tuple[float, torch.cuda.Event | None]:
        if self.cuda:
            torch.cuda.synchronize(self.device)
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_event = None
        return time.perf_counter(), start_event

    def step_end(
        self,
        *,
        cpu_start: float,
        cuda_start: torch.cuda.Event | None,
        losses: dict[str, float],
    ) -> None:
        cuda_ms = 0.0
        if self.cuda:
            assert cuda_start is not None
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize(self.device)
            cuda_ms = cuda_start.elapsed_time(end_event)
        cpu_ms = 1000 * (time.perf_counter() - cpu_start)
        self.steps.add(cpu_ms=cpu_ms, cuda_ms=cuda_ms, losses=losses)

    def to_json(self) -> dict[str, Any]:
        return {
            "steps": self.steps.to_json(),
            "phases": {name: stats.to_json() for name, stats in sorted(self.phases.items())},
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--config-json", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--profile-label", type=str, default="")
    parser.add_argument("--strategy", choices=["ddp", "zero1", "none"], default="ddp")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--trace-steps", type=int, default=0)
    parser.add_argument("--trace-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trace-shapes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-train-logging", action="store_true")
    parser.add_argument("--disable-loss", action="append", default=[])
    parser.add_argument("--ppgd-warmup-steps", type=int, default=None)
    parser.add_argument("--use-delta-component", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--autocast-bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-steps-for-schedule", type=int, default=None)
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--synthetic-vocab-size", type=int, default=1024)
    parser.add_argument("--synthetic-data-batches", type=int, default=16)
    return parser.parse_args()


def json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


def config_with_overrides(args: argparse.Namespace) -> Config:
    config = parse_config(args.config_path, args.config_json)

    updates: dict[str, Any] = {"wandb_project": None}
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        updates["eval_batch_size"] = args.eval_batch_size
    if args.autocast_bf16 is not None:
        updates["autocast_bf16"] = args.autocast_bf16
    if args.use_delta_component is not None:
        updates["use_delta_component"] = args.use_delta_component

    config = replace_pydantic_model(config, updates)

    if args.disable_loss:
        disabled = set(args.disable_loss)
        loss_metric_configs = [
            cfg for cfg in config.loss_metric_configs if cfg.classname not in disabled
        ]
        config = replace_pydantic_model(
            config,
            {"loss_metric_configs": [cfg.model_dump(mode="json") for cfg in loss_metric_configs]},
        )

    if args.ppgd_warmup_steps is not None:
        updated_loss_configs: list[dict[str, Any]] = []
        for cfg in config.loss_metric_configs:
            cfg_dict = cfg.model_dump(mode="json")
            if isinstance(cfg, PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig):
                cfg_dict["n_warmup_steps"] = args.ppgd_warmup_steps
            updated_loss_configs.append(cfg_dict)
        config = replace_pydantic_model(config, {"loss_metric_configs": updated_loss_configs})

    return config


def load_target_model(config: Config) -> nn.Module:
    pretrained_model_class = resolve_class(config.pretrained_model_class)
    assert hasattr(pretrained_model_class, "from_pretrained"), (
        f"Model class {pretrained_model_class} should have a `from_pretrained` method"
    )
    assert config.pretrained_model_name is not None

    if config.pretrained_model_class.startswith("param_decomp.pretrain"):
        run_info = ensure_cached_and_call(PretrainRunInfo.from_path, config.pretrained_model_name)
        if "model_type" not in run_info.model_config_dict:
            run_info.model_config_dict["model_type"] = config.pretrained_model_class.split(".")[-1]
        assert hasattr(pretrained_model_class, "from_run_info")
        target_model = pretrained_model_class.from_run_info(run_info)  # pyright: ignore[reportAttributeAccessIssue]
    else:
        target_model = ensure_cached_and_call(
            pretrained_model_class.from_pretrained,  # pyright: ignore[reportAttributeAccessIssue]
            config.pretrained_model_name,
        )

    target_model.eval()
    target_model.requires_grad_(False)
    return cast(nn.Module, target_model)


class SyntheticInputIdsDataset(IterableDataset[torch.Tensor]):
    def __init__(
        self,
        *,
        batch_size: int,
        n_ctx: int,
        vocab_size: int,
        seed: int,
        n_batches: int,
    ) -> None:
        self._batches = [
            torch.randint(
                low=0,
                high=vocab_size,
                size=(batch_size, n_ctx),
                dtype=torch.long,
                generator=torch.Generator(device="cpu").manual_seed(seed + idx),
            )
            for idx in range(n_batches)
        ]

    def __iter__(self) -> Any:
        while True:
            yield from self._batches


def build_synthetic_train_loader(
    config: Config,
    dist_state: DistributedState | None,
    vocab_size: int,
    n_batches: int,
) -> DataLoader[Any]:
    assert isinstance(config.task_config, LMTaskConfig)
    if dist_state is not None:
        assert config.batch_size % dist_state.world_size == 0 and config.batch_size > 0, (
            f"Batch size {config.batch_size} is not divisible by world size "
            f"{dist_state.world_size}."
        )
        train_rank_batch_size = config.batch_size // dist_state.world_size
        rank = dist_state.rank
    else:
        train_rank_batch_size = config.batch_size
        rank = 0

    dataset = SyntheticInputIdsDataset(
        batch_size=train_rank_batch_size,
        n_ctx=config.task_config.max_seq_len,
        vocab_size=vocab_size,
        seed=config.seed + 1009 * rank,
        n_batches=n_batches,
    )
    return DataLoader(dataset, batch_size=None)


def build_train_loader(config: Config, dist_state: DistributedState | None) -> DataLoader[Any]:
    assert isinstance(config.task_config, LMTaskConfig)
    train_data_config = DatasetConfig(
        name=config.task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=config.task_config.train_data_split,
        n_ctx=config.task_config.max_seq_len,
        is_tokenized=config.task_config.is_tokenized,
        streaming=config.task_config.streaming,
        column_name=config.task_config.column_name,
        shuffle_each_epoch=config.task_config.shuffle_each_epoch,
        seed=config.task_config.dataset_seed,
    )

    if dist_state is not None:
        assert config.batch_size % dist_state.world_size == 0 and config.batch_size > 0, (
            f"Batch size {config.batch_size} is not divisible by world size "
            f"{dist_state.world_size}."
        )
        train_rank_batch_size = config.batch_size // dist_state.world_size
    else:
        train_rank_batch_size = config.batch_size

    for cfg in config.loss_metric_configs:
        if isinstance(
            cfg, PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig
        ) and isinstance(cfg.scope, RepeatAcrossBatchScope):
            n = cfg.scope.n_sources
            assert train_rank_batch_size % n == 0, (
                f"repeat_across_batch n_sources={n} must divide per-rank batch_size="
                f"{train_rank_batch_size}"
            )

    loader, _tokenizer = create_data_loader(
        dataset_config=train_data_config,
        batch_size=train_rank_batch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed,
        dist_state=dist_state,
        collate_fn=input_ids_collate_fn,
    )
    return loader


def run_faithfulness_warmup_profiled(
    component_model: ComponentModel,
    component_params: list[torch.nn.Parameter],
    config: Config,
    profiler: TrainStepProfiler,
) -> None:
    if config.faithfulness_warmup_steps == 0:
        return

    optimizer = optim.AdamW(
        component_params,
        lr=config.faithfulness_warmup_lr,
        weight_decay=config.faithfulness_warmup_weight_decay,
    )
    for _ in range(config.faithfulness_warmup_steps):
        with profiler.phase("init.faithfulness_warmup_step"):
            optimizer.zero_grad()
            weight_deltas = component_model.calc_weight_deltas()
            loss = faithfulness_loss(weight_deltas)
            loss.backward()
            optimizer.step()

    del optimizer
    torch.cuda.empty_cache()
    gc.collect()


def wrap_model(
    model: ComponentModel,
    strategy: Strategy,
    dist_state: DistributedState | None,
) -> tuple[nn.Module, ComponentModel]:
    if strategy == "none":
        assert dist_state is None, "strategy=none must run without torchrun"
        return model, model

    assert dist_state is not None, f"strategy={strategy} requires torchrun"
    if dist_state.backend == "nccl":
        wrapped_model = DistributedDataParallel(
            model,
            device_ids=[dist_state.local_rank],
            output_device=dist_state.local_rank,
        )
    else:
        wrapped_model = DistributedDataParallel(model)
    return wrapped_model, cast(ComponentModel, wrapped_model.module)


def make_optimizer(
    strategy: Strategy,
    params: list[torch.nn.Parameter],
    lr: float,
) -> torch.optim.Optimizer:
    if strategy == "zero1":
        from torch.distributed.optim import ZeroRedundancyOptimizer

        return ZeroRedundancyOptimizer(
            params,
            optimizer_class=torch.optim.AdamW,
            lr=lr,
            weight_decay=0,
        )
    return optim.AdamW(params, lr=lr, weight_decay=0)


@dataclass
class TrainState:
    wrapped_model: nn.Module
    component_model: ComponentModel
    component_params: list[torch.nn.Parameter]
    ci_fn_params: list[torch.nn.Parameter]
    optimizer: torch.optim.Optimizer
    ppgd_states: dict[
        PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig, PersistentPGDState
    ]


def init_train_state(
    *,
    target_model: nn.Module,
    config: Config,
    device: str,
    train_loader: DataLoader[Any],
    run_batch: RunBatch,
    reconstruction_loss: ReconstructionLoss,
    strategy: Strategy,
    profiler: TrainStepProfiler,
) -> TrainState:
    if config.identity_module_info is not None:
        insert_identity_operations_(
            target_model,
            identity_module_info=config.identity_module_info,
        )

    target_model.requires_grad_(False)
    module_path_info = expand_module_patterns(target_model, config.all_module_info)

    with profiler.phase("init.component_model"):
        model = ComponentModel(
            target_model=target_model,
            run_batch=run_batch,
            module_path_info=module_path_info,
            ci_config=config.ci_config,
            sigmoid_type=config.sigmoid_type,
        )
        model.to(device)

    seed_per_rank(config.seed)
    dist_state = get_distributed_state()
    with profiler.phase(f"init.wrap.{strategy}"):
        wrapped_model, component_model = wrap_model(model, strategy, dist_state)

    component_params: list[torch.nn.Parameter] = []
    for name in component_model.target_module_paths:
        component_params.extend(component_model.components[name].parameters())
    ci_fn_params = list(component_model.ci_fn.parameters())
    optimized_params = component_params + ci_fn_params

    with profiler.phase(f"init.optimizer.{strategy}"):
        optimizer = make_optimizer(strategy, optimized_params, config.lr_schedule.start_val)

    run_faithfulness_warmup_profiled(component_model, component_params, config, profiler)

    persistent_pgd_configs: list[
        PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig
    ] = [
        cfg
        for cfg in config.loss_metric_configs
        if isinstance(cfg, PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig)
    ]

    train_iterator = loop_dataloader(train_loader)
    with profiler.phase("init.ppgd_state_shape_forward"):
        sample_out = model(next(train_iterator))
    batch_dims = sample_out.shape[:-1]

    with profiler.phase("init.ppgd_states"):
        ppgd_states = {
            ppgd_cfg: PersistentPGDState(
                module_to_c=model.module_to_c,
                batch_dims=batch_dims,
                device=device,
                use_delta_component=config.use_delta_component,
                cfg=ppgd_cfg,
                reconstruction_loss=reconstruction_loss,
            )
            for ppgd_cfg in persistent_pgd_configs
        }

    return TrainState(
        wrapped_model=wrapped_model,
        component_model=component_model,
        component_params=component_params,
        ci_fn_params=ci_fn_params,
        optimizer=optimizer,
        ppgd_states=ppgd_states,
    )


def run_ppgd_warmups_profiled(
    *,
    active_ppgd_configs: list[PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig],
    train_state: TrainState,
    batch: Any,
    target_out: torch.Tensor,
    ci: dict[str, torch.Tensor],
    weight_deltas: dict[str, torch.Tensor] | None,
    profiler: TrainStepProfiler,
) -> None:
    all_layers = AllLayersRouter()
    for ppgd_cfg in active_ppgd_configs:
        state = train_state.ppgd_states[ppgd_cfg]
        for warmup_idx in range(state._n_warmup_steps):  # noqa: SLF001
            prefix = f"ppgd_warmup.{ppgd_cfg.classname}.{warmup_idx}"
            with profiler.phase(f"{prefix}.forward_loss"):
                loss = state.compute_recon_loss(
                    train_state.component_model,
                    batch,
                    target_out,
                    ci,
                    weight_deltas,
                    router=all_layers,
                )
            with profiler.phase(f"{prefix}.source_grads"):
                grads = state.get_grads(loss, retain_graph=False)
            with profiler.phase(f"{prefix}.source_step"):
                state.step(grads)


def train_step_profiled(
    *,
    step: int,
    max_steps_for_schedule: int,
    train_iterator: Any,
    train_state: TrainState,
    config: Config,
    device: str,
    reconstruction_loss: ReconstructionLoss,
    profiler: TrainStepProfiler,
    include_train_logging: bool,
) -> None:
    cpu_start, cuda_start = profiler.step_start()

    with profiler.phase("optimizer.zero_grad"):
        train_state.optimizer.zero_grad()

    with profiler.phase("lr_schedule"):
        step_lr = get_scheduled_value(
            step=step,
            total_steps=max_steps_for_schedule,
            config=config.lr_schedule,
        )
        for group in train_state.optimizer.param_groups:
            group["lr"] = step_lr

    frac = step / max_steps_for_schedule
    active_ppgd_configs = [
        cfg
        for cfg in train_state.ppgd_states
        if isinstance(cfg, PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig)
        and frac >= cfg.start_frac
    ]

    with profiler.phase("ppgd.update_lr"):
        for ppgd_cfg in active_ppgd_configs:
            train_state.ppgd_states[ppgd_cfg].update_lr(step, max_steps_for_schedule)

    with profiler.phase("weight_deltas"):
        weight_deltas = train_state.component_model.calc_weight_deltas()

    with profiler.phase("data.next"):
        raw_batch = next(train_iterator)

    with profiler.phase("data.to_device"):
        batch = move_batch_to_device(raw_batch, device)

    batch_log_data: defaultdict[str, float] = defaultdict(float)

    with bf16_autocast(enabled=config.autocast_bf16):
        with profiler.phase("target_forward_input_cache"):
            target_model_output: OutputWithCache = train_state.wrapped_model(
                batch, cache_type="input"
            )

        with profiler.phase("ci_forward"):
            ci = train_state.component_model.calc_causal_importances(
                pre_weight_acts=target_model_output.cache,
                detach_inputs=False,
                sampling=config.sampling,
            )

        run_ppgd_warmups_profiled(
            active_ppgd_configs=active_ppgd_configs,
            train_state=train_state,
            batch=batch,
            target_out=target_model_output.output,
            ci=ci.lower_leaky,
            weight_deltas=weight_deltas if config.use_delta_component else None,
            profiler=profiler,
        )

        losses: dict[LossMetricConfigType, torch.Tensor] = {}
        for loss_cfg in config.loss_metric_configs:
            phase_name = f"loss.{loss_cfg.classname}"
            with profiler.phase(phase_name):
                loss_result = compute_losses(
                    loss_metric_configs=[loss_cfg],
                    model=train_state.component_model,
                    batch=batch,
                    ci=ci,
                    target_out=target_model_output.output,
                    weight_deltas=weight_deltas,
                    current_frac_of_training=step / max_steps_for_schedule,
                    sampling=config.sampling,
                    use_delta_component=config.use_delta_component,
                    n_mask_samples=config.n_mask_samples,
                    ppgd_states=train_state.ppgd_states,
                    reconstruction_loss=reconstruction_loss,
                )
            losses.update(loss_result)

    with profiler.phase("total_loss"):
        total_loss = torch.tensor(0.0, device=device)
        for loss_cfg, loss_val in losses.items():
            assert loss_cfg.coeff is not None
            total_loss = total_loss + loss_cfg.coeff * loss_val
            batch_log_data[f"train/loss/{loss_cfg.classname}"] = loss_val.item()
        batch_log_data["train/loss/total"] = total_loss.item()

    ppgd_grads: dict[
        PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig,
        dict[str, torch.Tensor],
    ] = {}
    for ppgd_cfg in active_ppgd_configs:
        if ppgd_cfg not in losses:
            continue
        with profiler.phase(f"ppgd_final_grads.{ppgd_cfg.classname}"):
            ppgd_grads[ppgd_cfg] = train_state.ppgd_states[ppgd_cfg].get_grads(
                losses[ppgd_cfg], retain_graph=True
            )

    with profiler.phase("backward.total_loss"):
        total_loss.backward()

    for ppgd_cfg in active_ppgd_configs:
        if ppgd_cfg not in ppgd_grads:
            continue
        with profiler.phase(f"ppgd_final_step.{ppgd_cfg.classname}"):
            train_state.ppgd_states[ppgd_cfg].step(ppgd_grads[ppgd_cfg])

    with profiler.phase("ci_l_zero"):
        for layer_name, layer_ci in ci.lower_leaky.items():
            l0_val = calc_ci_l_zero(layer_ci, config.ci_alive_threshold)
            batch_log_data[f"train/l0/{layer_name}"] = l0_val

    if include_train_logging:
        with profiler.phase("train_logging.metric_allreduce"):
            avg_metrics = avg_metrics_across_ranks(batch_log_data, device=device)
            batch_log_data = cast(defaultdict[str, float], avg_metrics)
        with profiler.phase("train_logging.grad_norms"):
            grad_norms = get_grad_norms_dict(train_state.component_model, device)
            dict_safe_update_(
                batch_log_data, {f"train/grad_norms/{k}": v for k, v in grad_norms.items()}
            )

    with profiler.phase("pre_optimizer_barrier"):
        sync_across_processes()

    if config.grad_clip_norm_components is not None:
        with profiler.phase("grad_clip.components"):
            clip_grad_norm_(train_state.component_params, config.grad_clip_norm_components)

    if config.grad_clip_norm_ci_fns is not None:
        with profiler.phase("grad_clip.ci_fns"):
            clip_grad_norm_(train_state.ci_fn_params, config.grad_clip_norm_ci_fns)

    with profiler.phase("optimizer.step"):
        train_state.optimizer.step()

    profiler.step_end(
        cpu_start=cpu_start,
        cuda_start=cuda_start,
        losses=dict(batch_log_data),
    )


def gather_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    dist_state = get_distributed_state()
    if dist_state is None:
        return [payload]
    gathered: list[dict[str, Any] | None] = [None for _ in range(dist_state.world_size)]
    dist.all_gather_object(gathered, payload)
    return [p for p in gathered if p is not None]


def summarize_rank_payloads(rank_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    phase_names = sorted(
        {
            phase_name
            for payload in rank_payloads
            for phase_name in payload["profiler"]["phases"]
        }
    )
    phase_summary: dict[str, Any] = {}
    for phase_name in phase_names:
        per_rank = [
            payload["profiler"]["phases"].get(phase_name, {"count": 0})
            for payload in rank_payloads
        ]
        phase_summary[phase_name] = {
            "count_rank0": per_rank[0].get("count", 0),
            "cuda_ms_avg_rank0": per_rank[0].get("cuda_ms_avg", 0.0),
            "cpu_ms_avg_rank0": per_rank[0].get("cpu_ms_avg", 0.0),
            "max_cuda_ms_avg": max(p.get("cuda_ms_avg", 0.0) for p in per_rank),
            "max_cpu_ms_avg": max(p.get("cpu_ms_avg", 0.0) for p in per_rank),
            "max_peak_delta_gb": max(p.get("peak_delta_gb_max", 0.0) for p in per_rank),
            "max_peak_allocated_gb": max(p.get("peak_allocated_gb_max", 0.0) for p in per_rank),
            "per_rank": per_rank,
        }

    step_per_rank = [payload["profiler"]["steps"] for payload in rank_payloads]
    return {
        "steps": {
            "cuda_ms_avg_rank0": step_per_rank[0].get("cuda_ms_avg", 0.0),
            "cpu_ms_avg_rank0": step_per_rank[0].get("cpu_ms_avg", 0.0),
            "max_cuda_ms_avg": max(p.get("cuda_ms_avg", 0.0) for p in step_per_rank),
            "max_cpu_ms_avg": max(p.get("cpu_ms_avg", 0.0) for p in step_per_rank),
            "per_rank": step_per_rank,
        },
        "phases": phase_summary,
    }


def print_summary_table(summary: dict[str, Any], limit: int = 40) -> None:
    phases = summary["phases"]
    rows = sorted(
        phases.items(),
        key=lambda item: item[1]["max_cuda_ms_avg"],
        reverse=True,
    )
    print("Top phases by max rank avg CUDA time:", flush=True)
    print(
        f"{'phase':48} {'rank0_cuda_ms':>14} {'max_cuda_ms':>12} {'peak_delta_gb':>13}",
        flush=True,
    )
    for name, stats in rows[:limit]:
        print(
            f"{name[:48]:48} "
            f"{stats['cuda_ms_avg_rank0']:14.3f} "
            f"{stats['max_cuda_ms_avg']:12.3f} "
            f"{stats['max_peak_delta_gb']:13.3f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ.get("RANK", "0"))
    try:
        config = config_with_overrides(args)
        dist_state = init_distributed()
        device = get_device()
        set_seed(config.seed)

        if is_main_process():
            logger.info(f"Profiling train steps on device={device}, dist_state={dist_state}")
            save_file(config.model_dump(mode="json"), args.out_dir / "profile_config.yaml")

        init_profiler = TrainStepProfiler(device=device, enabled=is_main_process())

        with init_profiler.phase("init.target_model"):
            target_model = load_target_model(config)

        with init_profiler.phase("init.train_loader"):
            if args.synthetic_data:
                train_loader = build_synthetic_train_loader(
                    config,
                    dist_state,
                    vocab_size=args.synthetic_vocab_size,
                    n_batches=args.synthetic_data_batches,
                )
            else:
                train_loader = build_train_loader(config, dist_state)

        train_state = init_train_state(
            target_model=target_model,
            config=config,
            device=device,
            train_loader=train_loader,
            run_batch=make_run_batch(config.output_extract),
            reconstruction_loss=recon_loss_kl,
            strategy=cast(Strategy, args.strategy),
            profiler=init_profiler,
        )

        train_iterator = loop_dataloader(train_loader)
        max_steps_for_schedule = args.max_steps_for_schedule or config.steps

        warmup_profiler = TrainStepProfiler(device=device, enabled=False)
        for i in range(args.warmup_steps):
            train_step_profiled(
                step=i,
                max_steps_for_schedule=max_steps_for_schedule,
                train_iterator=train_iterator,
                train_state=train_state,
                config=config,
                device=device,
                reconstruction_loss=recon_loss_kl,
                profiler=warmup_profiler,
                include_train_logging=False,
            )

        sync_across_processes()
        torch.cuda.empty_cache()
        gc.collect()

        measured_profiler = TrainStepProfiler(device=device, enabled=True)
        first_measured_step = args.warmup_steps

        def run_measured_step(step_idx: int) -> None:
            train_step_profiled(
                step=first_measured_step + step_idx,
                max_steps_for_schedule=max_steps_for_schedule,
                train_iterator=train_iterator,
                train_state=train_state,
                config=config,
                device=device,
                reconstruction_loss=recon_loss_kl,
                profiler=measured_profiler,
                include_train_logging=args.include_train_logging,
            )

        trace_steps = min(args.trace_steps, args.measure_steps)
        if trace_steps > 0:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            with profile(
                activities=activities,
                record_shapes=args.trace_shapes,
                profile_memory=args.trace_memory,
                with_stack=False,
            ) as torch_profile:
                for step_idx in range(trace_steps):
                    run_measured_step(step_idx)
                    torch_profile.step()
            trace_path = args.out_dir / f"trace_rank{rank}.json"
            torch_profile.export_chrome_trace(str(trace_path))

        for step_idx in range(trace_steps, args.measure_steps):
            run_measured_step(step_idx)

        payload = {
            "rank": rank,
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "hostname": os.uname().nodename,
            "profiler": measured_profiler.to_json(),
            "init_profiler": init_profiler.to_json(),
        }
        rank_payloads = gather_payload(payload)

        if is_main_process():
            world_size = dist_state.world_size if dist_state is not None else 1
            rank_batch_size = config.batch_size // world_size
            summary = {
                "args": vars(args) | {"out_dir": str(args.out_dir)},
                "world_size": world_size,
                "rank_batch_size": rank_batch_size,
                "global_batch_size": config.batch_size,
                "autocast_bf16": config.autocast_bf16,
                "losses": [cfg.classname for cfg in config.loss_metric_configs],
                "summary": summarize_rank_payloads(rank_payloads),
                "ranks": rank_payloads,
            }
            out_path = args.out_dir / "result.json"
            out_path.write_text(json_dump(summary))
            print_summary_table(summary["summary"])
            print(f"wrote {out_path}", flush=True)

        sync_across_processes()

    except BaseException as exc:
        if rank == 0:
            err = {
                "args": vars(args) | {"out_dir": str(args.out_dir)},
                "rank": rank,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            (args.out_dir / "error.json").write_text(json_dump(err))
            traceback.print_exc()
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            cleanup_distributed()


if __name__ == "__main__":
    main()
