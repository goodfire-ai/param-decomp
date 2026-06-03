"""LW-step MFU: achievable bf16 GEMM peak vs the LW step's sustained FLOP/s.

(1) Measure the GPU's *achievable* dense bf16 matmul throughput (a sweep of square GEMMs) —
    the realistic ceiling, not the spec sheet.
(2) Count the LW step's model FLOPs with FlopCounterMode (counts mm/bmm/addmm/sdpa incl. the
    backward + the activation-ckpt recompute — i.e. FLOPs actually executed).
(3) Time the LW step (no counter, to avoid overhead).
=> achieved TFLOP/s = executed_flops / step_time ; MFU = achieved / peak.

Also counts a single masked forward alone, so the recompute / target-fwd redundancy is visible.
Single GPU, no comm. Run: srun --gres=gpu:1 python scripts/bench_lw_mfu.py
"""

import sys
import time
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.flop_counter import FlopCounterMode

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from param_decomp.components import make_components  # noqa: E402
from param_decomp.fused_linear_kl import fused_linear_kl_div  # noqa: E402
from param_decomp.masks import ComponentsMaskInfo  # noqa: E402
from param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple import (  # noqa: E402
    GPT2Simple,
    GPT2SimpleConfig,
)
from param_decomp_lab.experiments.lm.vendored.gpt2 import (  # noqa: E402
    ComponentGPT2,
    componentize_gpt2,
)

N_LAYERS, D_MODEL, N_HEADS, VOCAB, SEQ, C, BL = 48, 1600, 25, 50257, 1024, 1024, 256
BL_FLOP = 8  # FlopCounterMode's dispatch mode defeats ckpt memory savings → count small, scale
DEV = "cuda"
SITE = "h.0.attn.q_proj"


def gemm_peak_tflops() -> float:
    """Max sustained dense bf16 GEMM TFLOP/s over a size sweep."""
    best = 0.0
    for n in (4096, 8192, 16384):
        a = torch.randn(n, n, device=DEV, dtype=torch.bfloat16)
        b = torch.randn(n, n, device=DEV, dtype=torch.bfloat16)
        for _ in range(3):
            (a @ b).sum()
        torch.cuda.synchronize()
        iters = 20
        t0 = time.perf_counter()
        for _ in range(iters):
            a @ b
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        tflops = 2 * n**3 / dt / 1e12
        print(f"  GEMM {n}^3: {tflops:7.0f} TFLOP/s")
        best = max(best, tflops)
        del a, b
        torch.cuda.empty_cache()
    return best


def build() -> ComponentGPT2:
    cfg = GPT2SimpleConfig(
        model_type="GPT2Simple",
        n_layer=N_LAYERS,
        n_head=N_HEADS,
        n_embd=D_MODEL,
        vocab_size=VOCAB,
        block_size=SEQ,
    )
    cg = componentize_gpt2(GPT2Simple(cfg), make_components(GPT2Simple(cfg), {SITE: C})).to(DEV)
    cg.enable_activation_checkpointing()
    return cg


def lw_step(cg: ComponentGPT2, opt: torch.optim.Optimizer, idx: Tensor, *, backward: bool) -> None:
    bl = idx.shape[0]
    opt.zero_grad(set_to_none=True)
    with cg.bypass_lm_head() as lm_w, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            target_h = cg(idx, None)
        ci = torch.rand(bl, SEQ, C, device=DEV)
        u = torch.rand(bl, SEQ, C, device=DEV)
        mi = {SITE: ComponentsMaskInfo(component_mask=ci + (1 - ci) * u, routing_mask="all")}
        pred_h = cg(idx, mask_infos=mi)
        loss, n = fused_linear_kl_div(
            pred_h.reshape(-1, D_MODEL), target_h.reshape(-1, D_MODEL), lm_w
        )
        if backward:
            (loss / n).backward()
    if backward:
        opt.step()


def masked_fwd_flops(cg: ComponentGPT2, idx: Tensor) -> float:
    with (
        FlopCounterMode(display=False) as fc,
        cg.bypass_lm_head(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        torch.no_grad(),
    ):
        cg(idx, None)
    return fc.get_total_flops()


def main() -> None:
    assert torch.cuda.is_available()
    name = torch.cuda.get_device_name(0)
    print(f"GPU {name}\n\n[1] achievable bf16 GEMM peak:")
    peak = gemm_peak_tflops()
    print(f"  => peak ~{peak:.0f} TFLOP/s\n")

    cg = build()
    opt = torch.optim.Adam([p for p in cg.parameters() if p.requires_grad], lr=1e-4)
    idx_flop = torch.randint(0, VOCAB, (BL_FLOP, SEQ), device=DEV)

    scale = BL / BL_FLOP  # model FLOPs are exactly linear in batch
    print(f"[2] LW step FLOPs via FlopCounterMode at bl={BL_FLOP}, scaled ×{scale:.0f} → bl={BL}")
    print("    (incl. backward + ckpt recompute):")
    with FlopCounterMode(display=False) as fc:
        lw_step(cg, opt, idx_flop, backward=True)
        torch.cuda.synchronize()
    step_flops = fc.get_total_flops() * scale
    one_fwd = masked_fwd_flops(cg, idx_flop) * scale
    print(f"  full step      : {step_flops / 1e12:8.1f} TFLOP")
    print(
        f"  one fwd (ref)  : {one_fwd / 1e12:8.1f} TFLOP  → step = {step_flops / one_fwd:.2f}× one forward"
    )

    print(f"\n[3] timed LW step at bl={BL} (no counter, ckpt on):")
    idx = torch.randint(0, VOCAB, (BL, SEQ), device=DEV)
    for _ in range(3):
        lw_step(cg, opt, idx, backward=True)
    torch.cuda.synchronize()
    nt = 5
    t0 = time.perf_counter()
    for _ in range(nt):
        lw_step(cg, opt, idx, backward=True)
    torch.cuda.synchronize()
    step_s = (time.perf_counter() - t0) / nt
    achieved = step_flops / step_s / 1e12
    print(f"  step time      : {step_s:.2f}s")
    print(f"  achieved       : {achieved:.0f} TFLOP/s")
    print(f"\n>>> MFU = {100 * achieved / peak:.1f}%  ({achieved:.0f} / {peak:.0f} TFLOP/s)")
    print(
        f">>> step does {step_flops / one_fwd:.1f}× one forward's FLOPs; the ckpt recompute "
        f"(~{one_fwd / 1e12:.0f} TFLOP) and the redundant target_fwd (~{one_fwd / 1e12:.0f} TFLOP) are avoidable"
    )


if __name__ == "__main__":
    main()
