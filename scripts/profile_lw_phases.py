"""Single-GPU LW-rank step: per-phase CUDA-event attribution.

The mask-dtype A/B showed 0% — the mask ([256,1024,1024]=0.27B) is noise next to the recon
loss's full-vocab KL ([256,1024,50257]=13.2B). This times each phase to find where the ~14s
actually goes: target_fwd / masked recon fwd (ckpt) / fused_linear_kl over the vocab / backward
(which includes the ckpt recompute). Also runs a 'cheap loss' variant (mean of pred_h instead of
the vocab KL) to isolate how much of the step is specifically the vocab projection + KL.

Run: srun --gres=gpu:1 python scripts/profile_lw_phases.py
"""

import sys
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
from param_decomp_lab.experiments.lm.vendored.gpt2 import ComponentGPT2, componentize_gpt2  # noqa: E402

N_LAYERS, D_MODEL, N_HEADS, VOCAB, SEQ, C, BL = 48, 1600, 25, 50257, 1024, 1024, 256
DEV = "cuda"
SITE = "h.0.attn.q_proj"


def build() -> ComponentGPT2:
    cfg = GPT2SimpleConfig(
        model_type="GPT2Simple", n_layer=N_LAYERS, n_head=N_HEADS, n_embd=D_MODEL,
        vocab_size=VOCAB, block_size=SEQ,
    )
    cg = componentize_gpt2(GPT2Simple(cfg), make_components(GPT2Simple(cfg), {SITE: C})).to(DEV)
    cg.enable_activation_checkpointing()
    return cg


class Timer:
    """Accumulating CUDA-event phase timer."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}

    def time(self, name: str, fn):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn()
        e.record()
        torch.cuda.synchronize()
        self.totals[name] = self.totals.get(name, 0.0) + s.elapsed_time(e) / 1000.0
        return out


def run(cg: ComponentGPT2, opt: torch.optim.Optimizer, idx: Tensor, t: Timer, *, cheap: bool) -> None:
    opt.zero_grad(set_to_none=True)
    with cg.bypass_lm_head() as lm_w, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            target_h = t.time("1_target_fwd", lambda: cg(idx, None))
        ci = torch.rand(BL, SEQ, C, device=DEV)
        u = torch.rand(BL, SEQ, C, device=DEV)
        mask = ci + (1 - ci) * u
        mi = {SITE: ComponentsMaskInfo(component_mask=mask, routing_mask="all")}
        pred_h = t.time("2_masked_fwd", lambda: cg(idx, mask_infos=mi))
        if cheap:
            loss = t.time("3_loss_cheap", lambda: pred_h.float().pow(2).mean())
        else:
            loss_n = t.time(
                "3_fused_kl",
                lambda: fused_linear_kl_div(
                    pred_h.reshape(-1, D_MODEL), target_h.reshape(-1, D_MODEL), lm_w
                ),
            )
            loss = loss_n[0] / loss_n[1]
        t.time("4_backward", lambda: loss.backward())
    opt.step()


def report(label: str, t: Timer, n: int) -> None:
    print(f"\n=== {label} (avg of {n}) ===")
    total = sum(t.totals.values()) / n
    for k in sorted(t.totals):
        v = t.totals[k] / n
        print(f"  {k:16s} {v:6.2f}s  ({100 * v / total:4.1f}%)")
    print(f"  {'TOTAL':16s} {total:6.2f}s")


def main() -> None:
    assert torch.cuda.is_available()
    cg = build()
    opt = torch.optim.Adam([p for p in cg.parameters() if p.requires_grad], lr=1e-4)
    idx = torch.randint(0, VOCAB, (BL, SEQ), device=DEV)

    for cheap, label in ((False, "fused-KL recon (prod)"), (True, "cheap loss (isolates KL+vocab)")):
        for _ in range(2):
            run(cg, opt, idx, Timer(), cheap=cheap)  # warmup
        t = Timer()
        n = 4
        for _ in range(n):
            run(cg, opt, idx, t, cheap=cheap)
        report(label, t, n)


if __name__ == "__main__":
    main()
