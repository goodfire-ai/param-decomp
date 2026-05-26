"""Standalone profile of `GlobalSharedTransformerCiFn` fwd+bwd at GPT-2 XL Q/K scale.

Reproduces the ``ci/8a_bwd_lower_leaky_only`` phase from the 3-pool trainer in a
single process so ``torch.profiler`` can actually run (the distributed setup
deadlocks the moment CUPTI activates).

Config matches ``param_decomp_lab/experiments/lm/_xl_production/gpt2_xl_qk_smoke.yaml``:
  d_model=4096, n_layers=8, n_heads=32, max_len=1024, mlp_hidden_dims=[16384]
  96 sites, each input_dim=1600 (GPT-2 XL d_model), C=1024
  per-rank batch B=2, S=1024 (the per-rank CI-pool batch size, not the trainer batch_size)
  fp32 inputs under autocast_bf16

The backward call mirrors ``_fused_backward_through_ci_fn`` (step_ci.py):
``torch.autograd.backward(lower_leaky_tensors, g_ci_total_seeds, retain_graph=True)``
with 96 separate seeds per site.
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, schedule

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from param_decomp.ci_fns import (  # noqa: E402
    GlobalSharedTransformerCiFn,
    TargetLayerConfig,
)
from param_decomp.ci_sigmoids import SIGMOID_TYPES  # noqa: E402


# --- Config matching gpt2_xl_qk_smoke.yaml -----------------------------------
N_SITES = 96
SITE_INPUT_DIM = 1600  # GPT-2 XL d_model
C_PER_SITE = 1024
D_MODEL = 4096
N_LAYERS = 8
N_HEADS = 32
MAX_LEN = 1024
MLP_HIDDEN_DIMS = [16384]
BATCH = 2
SEQ = 1024
DEVICE = "cuda"


def build_ci_fn() -> GlobalSharedTransformerCiFn:
    target_cfgs = {
        f"h.{i // 2}.attn.{'q' if i % 2 == 0 else 'k'}_proj": TargetLayerConfig(
            input_dim=SITE_INPUT_DIM, C=C_PER_SITE
        )
        for i in range(N_SITES)
    }
    return GlobalSharedTransformerCiFn(
        target_model_layer_configs=target_cfgs,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_len=MAX_LEN,
        mlp_hidden_dims=MLP_HIDDEN_DIMS,
    ).to(DEVICE)


def build_inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        f"h.{i // 2}.attn.{'q' if i % 2 == 0 else 'k'}_proj": torch.randn(
            BATCH, SEQ, SITE_INPUT_DIM, device=DEVICE, dtype=torch.float32, requires_grad=False
        )
        for i in range(N_SITES)
    }


def build_grad_seeds(layer_order: list[str]) -> list[torch.Tensor]:
    return [
        torch.randn(BATCH, SEQ, C_PER_SITE, device=DEVICE, dtype=torch.float32)
        for _ in layer_order
    ]


def fwd_and_split(
    ci_fn: GlobalSharedTransformerCiFn,
    inputs: dict[str, torch.Tensor],
    lower_leaky_fn: torch.nn.Module,
) -> list[torch.Tensor]:
    """Mirror ``_sigmoid_and_split_global`` (component_model.py).

    fwd under bf16 autocast → lower_leaky sigmoid → split into 96 site tensors.
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ci_out = ci_fn(inputs)
    lower = lower_leaky_fn(ci_out)
    splits = torch.split(lower, ci_fn.split_sizes, dim=-1)
    assert len(splits) == len(ci_fn.layer_order)
    return list(splits)


def bwd(lower_leaky_tensors: list[torch.Tensor], grad_seeds: list[torch.Tensor]) -> None:
    """Mirror ``_fused_backward_through_ci_fn`` 8a."""
    torch.autograd.backward(
        tensors=lower_leaky_tensors,
        grad_tensors=grad_seeds,
        retain_graph=True,
    )


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    torch.cuda.set_device(0)

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")

    ci_fn = build_ci_fn()
    # yaml sigmoid_type="leaky_hard" → ComponentModel sets lower_leaky_fn=lower_leaky_hard
    lower_leaky_fn = SIGMOID_TYPES["lower_leaky_hard"]
    inputs = build_inputs()
    grad_seeds = build_grad_seeds(ci_fn.layer_order)

    n_params = sum(p.numel() for p in ci_fn.parameters())
    print(f"ci_fn n_params: {n_params:,}")

    # --- Warmup ---------------------------------------------------------------
    print("\n=== Warmup (3 iters) ===")
    for i in range(3):
        for p in ci_fn.parameters():
            p.grad = None
        lower = fwd_and_split(ci_fn, inputs, lower_leaky_fn)
        bwd(lower, grad_seeds)
        torch.cuda.synchronize()
        print(f"  warmup {i} done")

    # --- Wall-time measurements (no profiler) --------------------------------
    print("\n=== Wall time (no profiler, 5 iters) ===")
    fwd_ms_list: list[float] = []
    bwd_ms_list: list[float] = []
    for _ in range(5):
        for p in ci_fn.parameters():
            p.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lower = fwd_and_split(ci_fn, inputs, lower_leaky_fn)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        bwd(lower, grad_seeds)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        fwd_ms_list.append((t1 - t0) * 1000)
        bwd_ms_list.append((t2 - t1) * 1000)
    print(f"  fwd ms: {[f'{x:.1f}' for x in fwd_ms_list]}  median={sorted(fwd_ms_list)[2]:.1f}")
    print(f"  bwd ms: {[f'{x:.1f}' for x in bwd_ms_list]}  median={sorted(bwd_ms_list)[2]:.1f}")

    # --- GPU-only compute (CUDA events around bwd) ---------------------------
    print("\n=== Pure GPU bwd time via CUDA events (5 iters) ===")
    gpu_bwd_ms: list[float] = []
    for _ in range(5):
        for p in ci_fn.parameters():
            p.grad = None
        lower = fwd_and_split(ci_fn, inputs, lower_leaky_fn)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        bwd(lower, grad_seeds)
        end.record()
        torch.cuda.synchronize()
        gpu_bwd_ms.append(start.elapsed_time(end))
    print(f"  GPU bwd (events): {[f'{x:.1f}' for x in gpu_bwd_ms]}  median={sorted(gpu_bwd_ms)[2]:.1f}")

    # --- torch.profiler -------------------------------------------------------
    print("\n=== torch.profiler ===")
    out_dir = Path(__file__).resolve().parent
    trace_path = out_dir / "profile_ci_fn_trace.json"
    text_path = out_dir / "profile_ci_fn_output.txt"

    n_wait, n_warmup, n_active = 1, 1, 3
    prof_schedule = schedule(wait=n_wait, warmup=n_warmup, active=n_active, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for _ in range(n_wait + n_warmup + n_active):
            for p in ci_fn.parameters():
                p.grad = None
            lower = fwd_and_split(ci_fn, inputs, lower_leaky_fn)
            bwd(lower, grad_seeds)
            torch.cuda.synchronize()
            prof.step()

    print(f"  trace: {trace_path}")
    prof.export_chrome_trace(str(trace_path))

    # --- key_averages tables --------------------------------------------------
    sections: list[str] = []
    sections.append(f"device: {torch.cuda.get_device_name(0)}")
    sections.append(f"torch: {torch.__version__}")
    sections.append(f"ci_fn n_params: {n_params:,}")
    sections.append(f"wall fwd ms (median): {sorted(fwd_ms_list)[2]:.1f}")
    sections.append(f"wall bwd ms (median): {sorted(bwd_ms_list)[2]:.1f}")
    sections.append(f"GPU-event bwd ms (median): {sorted(gpu_bwd_ms)[2]:.1f}")
    sections.append("")

    sections.append("=" * 80)
    sections.append("TOP 25 by self_cpu_time_total (sorted desc)")
    sections.append("=" * 80)
    sections.append(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))

    sections.append("\n" + "=" * 80)
    sections.append("TOP 25 by self_cuda_time_total (sorted desc)")
    sections.append("=" * 80)
    sections.append(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))

    sections.append("\n" + "=" * 80)
    sections.append("TOP 25 by cpu_time_total")
    sections.append("=" * 80)
    sections.append(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))

    out_text = "\n".join(sections)
    print(out_text)
    text_path.write_text(out_text)
    print(f"\nfull text saved to: {text_path}")


if __name__ == "__main__":
    main()
