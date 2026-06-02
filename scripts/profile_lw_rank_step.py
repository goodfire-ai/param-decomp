"""Single-GPU LW-rank step: pure-compute step time + kernel self-time breakdown.

Times what ONE layerwise rank actually computes per step — target_fwd (bypass, no_grad) +
per-owned-site streaming recon (bypass + fused linear-KL, bl_lw=256, block ckpt) + Adam — on
ONE GPU with NO cross-pool comm. So the measured step time is the LW COMPUTE FLOOR; the gap to
the live distributed step (~14s) is the cross-pool/comm dead time. Also dumps the torch.profiler
kernel self-time table (GEMM vs attention vs norm/elementwise vs cast vs ckpt-recompute) to show
where the compute goes.

Run: srun --gres=gpu:1 python scripts/profile_lw_rank_step.py
"""

import sys
import time
from pathlib import Path

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from param_decomp.components import make_components  # noqa: E402
from param_decomp.fused_linear_kl import fused_linear_kl_div  # noqa: E402
from param_decomp.masks import ComponentsMaskInfo  # noqa: E402
from param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple import (  # noqa: E402
    GPT2Simple,
    GPT2SimpleConfig,
)
from param_decomp_lab.experiments.lm.vendored.gpt2 import componentize_gpt2  # noqa: E402

N_LAYERS, D_MODEL, N_HEADS, VOCAB, SEQ, C = 48, 1600, 25, 50257, 1024, 1024
SITES_PER_BLOCK, BL = 2, 256
DEV = "cuda"
OWNED = [f"h.{i // 2}.attn.{'q' if i % 2 == 0 else 'k'}_proj" for i in range(SITES_PER_BLOCK)]


def build() -> object:
    cfg = GPT2SimpleConfig(
        model_type="GPT2Simple", n_layer=N_LAYERS, n_head=N_HEADS, n_embd=D_MODEL,
        vocab_size=VOCAB, block_size=SEQ,
    )
    m = GPT2Simple(cfg)
    cg = componentize_gpt2(m, make_components(m, {s: C for s in OWNED})).to(DEV)
    cg.enable_activation_checkpointing()
    return cg


def lw_step(cg: object, opt: torch.optim.Optimizer, idx: Tensor) -> None:
    """One LW-rank step: detached target_fwd + per-site bypass+fused-KL recon fwd/bwd + opt."""
    opt.zero_grad(set_to_none=True)
    with cg.bypass_lm_head() as lm_w, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            target_h = cg(idx, None)
        for site in OWNED:
            ci = torch.rand(BL, SEQ, C, device=DEV)
            src = torch.rand(BL, SEQ, C, device=DEV, requires_grad=True)
            mi = {site: ComponentsMaskInfo(component_mask=ci + (1 - ci) * src, routing_mask="all")}
            pred_h = cg(idx, mi)
            loss, n = fused_linear_kl_div(
                pred_h.reshape(-1, D_MODEL), target_h.reshape(-1, D_MODEL), lm_w
            )
            (loss / n).backward()
    opt.step()


def main() -> None:
    assert torch.cuda.is_available()
    cg = build()
    opt = torch.optim.Adam([p for p in cg.parameters() if p.requires_grad], lr=1e-4)
    idx = torch.randint(0, VOCAB, (BL, SEQ), device=DEV)
    print(f"single LW rank: {SITES_PER_BLOCK} sites, bl_lw={BL}, seq={SEQ}, ckpt ON, no comm")

    for _ in range(3):  # warmup (compile/alloc)
        lw_step(cg, opt, idx)
    torch.cuda.synchronize()

    n_timed = 5
    t0 = time.perf_counter()
    for _ in range(n_timed):
        lw_step(cg, opt, idx)
    torch.cuda.synchronize()
    step_s = (time.perf_counter() - t0) / n_timed
    print(f"\n>>> pure-compute LW step time: {step_s:.2f}s  (live distributed step ~14.1s)")
    print(f">>> implied cross-pool/comm dead time: ~{14.1 - step_s:.1f}s "
          f"({100 * (14.1 - step_s) / 14.1:.0f}% of the live step)\n")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        lw_step(cg, opt, idx)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))


if __name__ == "__main__":
    main()
