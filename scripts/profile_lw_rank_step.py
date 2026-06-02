"""Single-GPU LW-rank step: fp32-mask vs bf16-mask A/B (step time + kernel self-time).

Times what ONE layerwise rank computes per step — target_fwd (bypass, no_grad) + per-owned-site
streaming recon (bypass + fused linear-KL, bl_lw=256, block ckpt) + Adam — on ONE GPU with NO
cross-pool comm, under two mask dtypes:

  fp32 mask (current prod): the CI leaf is fp32, so `component_acts(bf16) * mask(fp32)`
    type-promotes the [bl,seq,C] activation to fp32 and the next einsum casts it back to bf16
    — a bf16->fp32->bf16 round-trip per site, plus the mask itself is built in fp32.
  bf16 mask (the fix):      build the mask in bf16 so the masked forward is bf16 end-to-end.

The gap between the two is the cast-elimination win. `ci` is the grad leaf (matches the real LW
path, where `u = rand_like` carries no grad and the stoch grad lands on the CI values).

NOTE: omits the weight-delta path (delta + delta_mask), so absolute step time is a LOWER bound
on the real LW step; the fp32-vs-bf16 DELTA is the quantity of interest.

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
from param_decomp_lab.experiments.lm.vendored.gpt2 import (  # noqa: E402
    ComponentGPT2,
    componentize_gpt2,
)

N_LAYERS, D_MODEL, N_HEADS, VOCAB, SEQ, C = 48, 1600, 25, 50257, 1024, 1024
SITES_PER_BLOCK, BL = 1, 256  # real topology: 1 site/block
DEV = "cuda"
OWNED = [f"h.{i // 2}.attn.{'q' if i % 2 == 0 else 'k'}_proj" for i in range(SITES_PER_BLOCK)]


def build() -> ComponentGPT2:
    cfg = GPT2SimpleConfig(
        model_type="GPT2Simple",
        n_layer=N_LAYERS,
        n_head=N_HEADS,
        n_embd=D_MODEL,
        vocab_size=VOCAB,
        block_size=SEQ,
    )
    m = GPT2Simple(cfg)
    cg = componentize_gpt2(m, make_components(m, {s: C for s in OWNED})).to(DEV)
    cg.enable_activation_checkpointing()
    return cg


def lw_step(
    cg: ComponentGPT2, opt: torch.optim.Optimizer, idx: Tensor, mask_dtype: torch.dtype
) -> None:
    """One LW-rank step: detached target_fwd + per-site bypass+fused-KL recon fwd/bwd + opt.

    ``ci`` is the grad leaf at ``mask_dtype``; ``u`` is a no-grad source. fp32 ``mask_dtype``
    reproduces the prod cast round-trip; bf16 keeps the masked forward bf16 end-to-end.
    """
    opt.zero_grad(set_to_none=True)
    with cg.bypass_lm_head() as lm_w, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            target_h = cg(idx, None)
        for site in OWNED:
            ci = torch.rand(BL, SEQ, C, device=DEV, dtype=mask_dtype, requires_grad=True)
            u = torch.rand(BL, SEQ, C, device=DEV, dtype=mask_dtype)
            mask = ci + (1 - ci) * u
            mi = {site: ComponentsMaskInfo(component_mask=mask, routing_mask="all")}
            pred_h = cg(idx, mi)
            loss, n = fused_linear_kl_div(
                pred_h.reshape(-1, D_MODEL), target_h.reshape(-1, D_MODEL), lm_w
            )
            (loss / n).backward()
    opt.step()


def measure(cg: ComponentGPT2, opt: torch.optim.Optimizer, idx: Tensor, mask_dtype: torch.dtype) -> float:
    for _ in range(3):  # warmup (alloc/caching allocator)
        lw_step(cg, opt, idx, mask_dtype)
    torch.cuda.synchronize()
    n_timed = 5
    t0 = time.perf_counter()
    for _ in range(n_timed):
        lw_step(cg, opt, idx, mask_dtype)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_timed


def main() -> None:
    assert torch.cuda.is_available()
    cg = build()
    opt = torch.optim.Adam([p for p in cg.parameters() if p.requires_grad], lr=1e-4)
    idx = torch.randint(0, VOCAB, (BL, SEQ), device=DEV)
    print(f"single LW rank: {SITES_PER_BLOCK} site, bl_lw={BL}, seq={SEQ}, ckpt ON, no comm\n")

    times: dict[str, float] = {}
    for label, dt in (("fp32 mask (prod)", torch.float32), ("bf16 mask (fix)", torch.bfloat16)):
        step_s = measure(cg, opt, idx, dt)
        times[label] = step_s
        print(f">>> {label}: {step_s:.2f}s / step")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as prof:
            lw_step(cg, opt, idx, dt)
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=18))
        print()

    fp32_s, bf16_s = times["fp32 mask (prod)"], times["bf16 mask (fix)"]
    print(
        f">>> WIN: {fp32_s:.2f}s -> {bf16_s:.2f}s "
        f"({100 * (fp32_s - bf16_s) / fp32_s:.0f}% faster, {fp32_s / bf16_s:.2f}x) per LW step"
    )


if __name__ == "__main__":
    main()
