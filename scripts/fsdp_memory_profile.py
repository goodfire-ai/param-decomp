"""Memory profile harness for fused-site FSDP2.

This script intentionally mirrors the training path in `run_param_decomp.py` without
using the full data/config stack:

  1. target-only forward with input cache
  2. global CI forward on cached pre-weight activations
  3. one or more decomposed forwards with optional delta masks
  4. one backward and optimizer step

Run under torchrun for distributed strategies, e.g.

    torchrun --standalone --nproc_per_node=4 scripts/fsdp_memory_profile.py \
        --strategy fsdp --target-scale jose --batch 8 --seq 512 --out-dir /tmp/profile
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
from torch import nn

from param_decomp.configs import (
    AttnConfig,
    GlobalCiConfig,
    GlobalSharedTransformerCiConfig,
    ModulePatternInfoConfig,
)
from param_decomp.models.batch_and_loss_fns import make_run_batch
from param_decomp.models.component_model import ComponentModel, OutputWithCache
from param_decomp.models.mask_info import make_mask_infos
from param_decomp.pretrain.models.llama_simple_mlp import LlamaSimpleMLP, LlamaSimpleMLPConfig
from param_decomp.utils.fsdp import fsdp_wrap
from param_decomp.utils.module_utils import expand_module_patterns

Strategy = Literal["none", "ddp", "zero1", "fsdp"]


@dataclass(frozen=True)
class TargetSpec:
    n_layer: int
    n_embd: int
    n_head: int
    n_intermediate: int
    n_key_value_heads: int
    vocab_size: int = 50277
    n_ctx: int = 512


TARGET_SPECS: dict[str, TargetSpec] = {
    # Current Jose-scale target used by the rough repro.
    "jose": TargetSpec(
        n_layer=4,
        n_embd=768,
        n_head=6,
        n_intermediate=3072,
        n_key_value_heads=6,
    ),
    # Synthetic scale points. They are random-init targets; the point is memory shape.
    "1b": TargetSpec(
        n_layer=18,
        n_embd=2048,
        n_head=16,
        n_intermediate=8192,
        n_key_value_heads=16,
    ),
    "2b": TargetSpec(
        n_layer=36,
        n_embd=2048,
        n_head=16,
        n_intermediate=8192,
        n_key_value_heads=16,
    ),
    "4b": TargetSpec(
        n_layer=36,
        n_embd=3072,
        n_head=24,
        n_intermediate=12288,
        n_key_value_heads=24,
    ),
}


JOSE_MODULE_INFO = [
    ModulePatternInfoConfig(module_pattern="h.*.mlp.c_fc", C=3072),
    ModulePatternInfoConfig(module_pattern="h.*.mlp.down_proj", C=3584),
    ModulePatternInfoConfig(module_pattern="h.*.attn.q_proj", C=512),
    ModulePatternInfoConfig(module_pattern="h.*.attn.k_proj", C=512),
    ModulePatternInfoConfig(module_pattern="h.*.attn.v_proj", C=1024),
    ModulePatternInfoConfig(module_pattern="h.*.attn.o_proj", C=1024),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["none", "ddp", "zero1", "fsdp"], required=True)
    parser.add_argument("--target-scale", choices=sorted(TARGET_SPECS), required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ci-d-model", type=int, default=2048)
    parser.add_argument("--ci-blocks", type=int, default=8)
    parser.add_argument("--ci-mlp", type=int, default=8192)
    parser.add_argument("--ci-heads", type=int, default=16)
    parser.add_argument("--ci-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--target-checkpointing", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--autocast-bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--forward-autocast-bf16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override forward autocast independently of FSDP param dtype. Useful for testing "
            "FSDP fp32 shards with bf16 activations."
        ),
    )
    parser.add_argument("--delta-masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-faithfulness", action="store_true")
    parser.add_argument("--decomposed-forwards", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--record-snapshot", action="store_true")
    parser.add_argument("--snapshot-max-entries", type=int, default=200_000)
    parser.add_argument("--profile-label", type=str, default="")
    return parser.parse_args()


def distributed_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def init_dist_if_needed(strategy: Strategy) -> tuple[int, int, int, torch.device]:
    rank, local_rank, world_size = distributed_env()
    if strategy != "none":
        assert world_size > 1, f"{strategy} requires torchrun with WORLD_SIZE > 1"
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        assert world_size == 1, "strategy=none should be launched without torchrun"
        torch.cuda.set_device(0)
    device = torch.device(f"cuda:{local_rank}")
    return rank, local_rank, world_size, device


def barrier(strategy: Strategy) -> None:
    if strategy != "none" and dist.is_initialized():
        dist.barrier()


def rprint(rank: int, *args: object) -> None:
    if rank == 0:
        print(*args, flush=True)


def memory_sample(label: str, device: torch.device) -> dict[str, float | str]:
    torch.cuda.synchronize(device)
    return {
        "label": label,
        "allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "max_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
    }


def reset_for_phase(device: torch.device) -> float:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    base = torch.cuda.memory_allocated(device) / 1e9
    torch.cuda.reset_peak_memory_stats(device)
    return base


def build_target(spec: TargetSpec, target_checkpointing: bool) -> LlamaSimpleMLP:
    cfg = LlamaSimpleMLPConfig(
        model_type="LlamaSimpleMLP",
        n_layer=spec.n_layer,
        n_embd=spec.n_embd,
        n_head=spec.n_head,
        n_intermediate=spec.n_intermediate,
        vocab_size=spec.vocab_size,
        n_ctx=spec.n_ctx,
        block_size=spec.n_ctx,
        n_key_value_heads=spec.n_key_value_heads,
        rotary_dim=spec.n_embd // spec.n_head,
        use_grouped_query_attention=True,
        gradient_checkpointing=target_checkpointing,
    )
    return LlamaSimpleMLP(cfg)


def build_ci_config(args: argparse.Namespace) -> GlobalCiConfig:
    return GlobalCiConfig(
        mode="global",
        fn_type="global_shared_transformer",
        simple_transformer_ci_cfg=GlobalSharedTransformerCiConfig(
            d_model=args.ci_d_model,
            n_blocks=args.ci_blocks,
            mlp_hidden_dim=[args.ci_mlp],
            attn_config=AttnConfig(
                n_heads=args.ci_heads,
                max_len=args.seq,
                rope_base=10000.0,
            ),
            gradient_checkpointing=args.ci_checkpointing,
        ),
    )


def trainable_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def wrap_and_optimizer(
    strategy: Strategy,
    cm: ComponentModel,
    local_rank: int,
    autocast_bf16: bool,
    lr: float,
) -> tuple[nn.Module, ComponentModel, torch.optim.Optimizer]:
    wrapped: nn.Module
    component_model = cm

    if strategy == "none":
        wrapped = cm
    elif strategy in {"ddp", "zero1"}:
        wrapped = torch.nn.parallel.DistributedDataParallel(
            cm,
            device_ids=[local_rank],
            output_device=local_rank,
        )
        component_model = wrapped.module  # type: ignore[assignment]
    elif strategy == "fsdp":
        wrapped = fsdp_wrap(cm, device_id=local_rank, autocast_bf16=autocast_bf16)
    else:
        raise AssertionError(f"unknown strategy: {strategy}")

    params = [p for p in wrapped.parameters() if p.requires_grad]
    if strategy == "zero1":
        from torch.distributed.optim import ZeroRedundancyOptimizer

        optimizer = ZeroRedundancyOptimizer(
            params,
            optimizer_class=torch.optim.AdamW,
            lr=lr,
            weight_decay=0,
        )
    else:
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0)

    return wrapped, component_model, optimizer


def make_delta_masks(
    component_model: ComponentModel,
    batch: int,
    seq: int,
    device: torch.device,
    enabled: bool,
) -> dict[str, torch.Tensor] | None:
    if not enabled:
        return None
    return {name: torch.ones(batch, seq, device=device) for name in component_model.components}


def make_ones_masks(
    component_model: ComponentModel,
    batch: int,
    seq: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        site_name: torch.ones(batch, seq, site.C, device=device)
        for site_name, site in component_model.components.items()
    }


def call_model(module: nn.Module, *args: Any, **kwargs: Any) -> Any:
    # Keeps pyright/mypy out of runtime code paths. DDP/FSDP both support kwargs here.
    return module(*args, **kwargs)


def target_only_forward(
    wrapped: nn.Module,
    idx: torch.Tensor,
    autocast_enabled: bool,
) -> None:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        out = call_model(wrapped, idx, cache_type="input")
    assert isinstance(out, OutputWithCache)
    # Force logits to stay live until this point.
    _ = out.output.float().mean().item()


def target_plus_ci_forward(
    wrapped: nn.Module,
    component_model: ComponentModel,
    idx: torch.Tensor,
    sampling: str,
    autocast_enabled: bool,
) -> None:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        target_out = call_model(wrapped, idx, cache_type="input")
        assert isinstance(target_out, OutputWithCache)
        ci = component_model.calc_causal_importances(
            pre_weight_acts=target_out.cache,
            detach_inputs=False,
            sampling=sampling,  # type: ignore[arg-type]
        )
        # Keep outputs live through the end of the measured region.
        total = target_out.output.float().mean()
        total = total + sum(v.float().mean() for v in ci.lower_leaky.values())
    _ = total.item()


def single_decomposed_step(
    wrapped: nn.Module,
    component_model: ComponentModel,
    optimizer: torch.optim.Optimizer,
    idx: torch.Tensor,
    batch: int,
    seq: int,
    delta_masks_enabled: bool,
    autocast_enabled: bool,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        masks = make_ones_masks(component_model, batch, seq, idx.device)
        delta_masks = make_delta_masks(component_model, batch, seq, idx.device, delta_masks_enabled)
        out = call_model(wrapped, idx, mask_infos=make_mask_infos(masks, delta_masks=delta_masks))
        loss = out.float().pow(2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def jose_like_step(
    wrapped: nn.Module,
    component_model: ComponentModel,
    optimizer: torch.optim.Optimizer,
    idx: torch.Tensor,
    batch: int,
    seq: int,
    delta_masks_enabled: bool,
    include_faithfulness: bool,
    decomposed_forwards: int,
    sampling: str,
    autocast_enabled: bool,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        target_out = call_model(wrapped, idx, cache_type="input")
        assert isinstance(target_out, OutputWithCache)
        target_logits = target_out.output
        ci = component_model.calc_causal_importances(
            pre_weight_acts=target_out.cache,
            detach_inputs=False,
            sampling=sampling,  # type: ignore[arg-type]
        )
        delta_masks = make_delta_masks(component_model, batch, seq, idx.device, delta_masks_enabled)
        losses: list[torch.Tensor] = []
        for forward_idx in range(decomposed_forwards):
            if forward_idx == 0:
                masks = ci.lower_leaky
            else:
                masks = {name: torch.ones_like(value) for name, value in ci.lower_leaky.items()}
            out = call_model(
                wrapped,
                idx,
                mask_infos=make_mask_infos(masks, delta_masks=delta_masks),
            )
            losses.append((out.float() - target_logits.float()).pow(2).mean())

        if include_faithfulness:
            faith_sum, faith_numel = component_model.calc_faithfulness_terms()
            losses.append((faith_sum / faith_numel) * 1e-3)

        loss = sum(losses)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def measure_phase(
    label: str,
    fn,
    strategy: Strategy,
    device: torch.device,
    repeat: int = 1,
) -> dict[str, Any]:
    barrier(strategy)
    base = reset_for_phase(device)
    times: list[float] = []
    for _ in range(repeat):
        barrier(strategy)
        t0 = time.perf_counter()
        fn()
        barrier(strategy)
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    sample = memory_sample(label, device)
    sample["base_allocated_gb"] = base
    sample["phase_peak_delta_gb"] = float(sample["max_allocated_gb"]) - base
    sample["times_ms"] = [1000 * t for t in times]
    sample["avg_time_ms"] = 1000 * sum(times) / len(times)
    return sample


def gather_rank_payload(strategy: Strategy, payload: dict[str, Any]) -> list[dict[str, Any]]:
    if strategy == "none":
        return [payload]
    gathered: list[dict[str, Any] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, payload)
    return [p for p in gathered if p is not None]


def summarize_across_ranks(rank_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    phase_names = rank_payloads[0]["phases"].keys()
    phase_summary: dict[str, Any] = {}
    for phase_name in phase_names:
        per_rank = [p["phases"][phase_name] for p in rank_payloads]
        phase_summary[phase_name] = {
            "max_peak_allocated_gb": max(p["max_allocated_gb"] for p in per_rank),
            "max_phase_peak_delta_gb": max(p["phase_peak_delta_gb"] for p in per_rank),
            "max_allocated_after_gb": max(p["allocated_gb"] for p in per_rank),
            "avg_time_ms_rank0": per_rank[0]["avg_time_ms"],
            "per_rank": per_rank,
        }

    construction_names = rank_payloads[0]["construction"].keys()
    construction_summary: dict[str, Any] = {}
    for name in construction_names:
        per_rank = [p["construction"][name] for p in rank_payloads]
        construction_summary[name] = {
            "max_allocated_gb": max(p["allocated_gb"] for p in per_rank),
            "max_reserved_gb": max(p["reserved_gb"] for p in per_rank),
            "max_peak_allocated_gb": max(p["max_allocated_gb"] for p in per_rank),
            "per_rank": per_rank,
        }

    return {
        "construction": construction_summary,
        "phases": phase_summary,
        "ranks": rank_payloads,
    }


def main() -> None:
    args = parse_args()
    strategy: Strategy = args.strategy
    rank, local_rank, world_size, device = init_dist_if_needed(strategy)

    try:
        torch.manual_seed(1234 + rank)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if args.record_snapshot:
            torch.cuda.memory._record_memory_history(max_entries=args.snapshot_max_entries)

        spec = TARGET_SPECS[args.target_scale]
        rprint(
            rank,
            (
                f"strategy={strategy} world_size={world_size} target={args.target_scale} "
                f"batch={args.batch} seq={args.seq} ci_ckpt={args.ci_checkpointing} "
                f"target_ckpt={args.target_checkpointing} bf16={args.autocast_bf16}"
            ),
        )

        construction: dict[str, Any] = {}
        torch.cuda.reset_peak_memory_stats(device)
        construction["start"] = memory_sample("start", device)

        target = build_target(spec, args.target_checkpointing).to(device).eval()
        target.requires_grad_(False)
        construction["after_target"] = memory_sample("after_target", device)
        n_frozen_target = total_param_count(target)

        module_path_info = expand_module_patterns(target, JOSE_MODULE_INFO)
        cm = ComponentModel(
            target_model=target,
            run_batch=make_run_batch(output_extract=0),
            module_path_info=module_path_info,
            ci_config=build_ci_config(args),
            sigmoid_type="leaky_hard",
        ).to(device)
        construction["after_component_model"] = memory_sample("after_component_model", device)

        n_total_after_sites = total_param_count(target)
        n_trainable = trainable_param_count(cm)
        rprint(
            rank,
            (
                f"frozen_target_params={n_frozen_target / 1e9:.3f}B "
                f"target_plus_site_params={n_total_after_sites / 1e9:.3f}B "
                f"trainable_params={n_trainable / 1e9:.3f}B"
            ),
        )

        wrapped, component_model, optimizer = wrap_and_optimizer(
            strategy=strategy,
            cm=cm,
            local_rank=local_rank,
            autocast_bf16=args.autocast_bf16,
            lr=args.lr,
        )
        barrier(strategy)
        construction["after_wrap"] = memory_sample("after_wrap", device)
        construction["after_optimizer"] = memory_sample("after_optimizer", device)

        idx = torch.randint(
            0,
            spec.vocab_size,
            (args.batch, args.seq),
            device=device,
        )
        construction["after_batch"] = memory_sample("after_batch", device)

        autocast_enabled = args.autocast_bf16 and strategy != "fsdp"
        if args.forward_autocast_bf16 is not None:
            autocast_enabled = args.forward_autocast_bf16
        phases: dict[str, Any] = {}

        phases["target_only"] = measure_phase(
            "target_only",
            lambda: target_only_forward(wrapped, idx, autocast_enabled),
            strategy,
            device,
        )
        rprint(rank, f"target_only peak={phases['target_only']['max_allocated_gb']:.2f} GB")

        phases["target_plus_ci_forward"] = measure_phase(
            "target_plus_ci_forward",
            lambda: target_plus_ci_forward(
                wrapped,
                component_model,
                idx,
                sampling="continuous",
                autocast_enabled=autocast_enabled,
            ),
            strategy,
            device,
        )
        rprint(
            rank,
            f"target_plus_ci_forward peak={phases['target_plus_ci_forward']['max_allocated_gb']:.2f} GB",
        )

        if strategy in {"none", "fsdp"}:
            phases["single_decomposed_step"] = measure_phase(
                "single_decomposed_step",
                lambda: single_decomposed_step(
                    wrapped,
                    component_model,
                    optimizer,
                    idx,
                    args.batch,
                    args.seq,
                    args.delta_masks,
                    autocast_enabled,
                ),
                strategy,
                device,
            )
            rprint(
                rank,
                f"single_decomposed_step peak={phases['single_decomposed_step']['max_allocated_gb']:.2f} GB",
            )

        for _ in range(args.warmup_steps):
            jose_like_step(
                wrapped,
                component_model,
                optimizer,
                idx,
                args.batch,
                args.seq,
                args.delta_masks,
                args.include_faithfulness,
                args.decomposed_forwards,
                "continuous",
                autocast_enabled,
            )
        construction["after_jose_warmup"] = memory_sample("after_jose_warmup", device)

        phases["jose_like_step"] = measure_phase(
            "jose_like_step",
            lambda: jose_like_step(
                wrapped,
                component_model,
                optimizer,
                idx,
                args.batch,
                args.seq,
                args.delta_masks,
                args.include_faithfulness,
                args.decomposed_forwards,
                "continuous",
                autocast_enabled,
            ),
            strategy,
            device,
            repeat=args.measure_steps,
        )
        rprint(rank, f"jose_like_step peak={phases['jose_like_step']['max_allocated_gb']:.2f} GB")

        if args.record_snapshot:
            snap_path = args.out_dir / f"memory_snapshot_rank{rank}.pickle"
            torch.cuda.memory._dump_snapshot(str(snap_path))
            torch.cuda.memory._record_memory_history(enabled=None)

        payload = {
            "rank": rank,
            "local_rank": local_rank,
            "hostname": socket.gethostname(),
            "construction": construction,
            "phases": phases,
        }
        rank_payloads = gather_rank_payload(strategy, payload)

        if rank == 0:
            summary = {
                "args": vars(args) | {"out_dir": str(args.out_dir)},
                "target_spec": asdict(spec),
                "world_size": world_size,
                "frozen_target_params": n_frozen_target,
                "target_plus_site_params": n_total_after_sites,
                "trainable_params": n_trainable,
                "summary": summarize_across_ranks(rank_payloads),
            }
            out_path = args.out_dir / "result.json"
            out_path.write_text(json.dumps(summary, indent=2))
            print(f"wrote {out_path}", flush=True)

        barrier(strategy)

    except BaseException as exc:
        if rank == 0:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            err = {
                "args": vars(args) | {"out_dir": str(args.out_dir)},
                "rank": rank,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            (args.out_dir / "error.json").write_text(json.dumps(err, indent=2))
            traceback.print_exc()
        raise
    finally:
        if strategy != "none" and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is required for this profiling harness", file=sys.stderr)
        sys.exit(1)
    main()
