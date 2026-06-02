"""Single-GPU bl-ceiling + block-ckpt gain for the GPT2-XL PPGD-pool per-rank work.

The PPGD pool holds a FULL V/U replica (all 96 q/k sites) + adversarial `sources`
(per-batch-per-position, one [bl, seq, C] tensor per site) and runs a masked full-model
recon forward whose mask is `ci + (1-ci)*source`; PGD backprops to `sources` and the
V/U train too. This probe measures the per-rank `bl_pp` ceiling and the block-ckpt gain
at GPT2-XL scale, including the source tensors + an Adam step over {V, U, sources} (the
PPGD memory profile), with block checkpointing off vs on.

Random weights (memory profile == real). bf16 autocast. All q/k sites masked. This is the
pool that actually blocked b256 (bl=16 OOM'd pre-ckpt).

Run (memory sweep): srun --gres=gpu:1 python scripts/probe_ppgd_bl_ceiling.py
Run (per-kernel profile): srun --gres=gpu:1 python scripts/probe_ppgd_bl_ceiling.py --profile
"""

import gc
import sys
from pathlib import Path

import fire
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from param_decomp.components import make_components  # noqa: E402
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
DEV = "cuda"
SITES = [f"h.{layer}.attn.{p}_proj" for layer in range(N_LAYERS) for p in ("q", "k")]
BLS = [4, 8, 16, 24, 32, 48]
PROFILE_BL = 8  # the b256-verified per-rank PPGD fit


def build(ckpt: bool) -> ComponentGPT2:
    cfg = GPT2SimpleConfig(
        model_type="GPT2Simple",
        n_layer=N_LAYERS,
        n_head=N_HEADS,
        n_embd=D_MODEL,
        vocab_size=VOCAB,
        block_size=SEQ,
    )
    model = GPT2Simple(cfg)
    cg = componentize_gpt2(model, make_components(model, {s: C for s in SITES})).to(DEV)
    if ckpt:
        cg.enable_activation_checkpointing()
    return cg


def run(bl: int, ckpt: bool) -> tuple[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    try:
        cg = build(ckpt)
        idx = torch.randint(0, VOCAB, (bl, SEQ), device=DEV)
        sources = [torch.rand(bl, SEQ, C, device=DEV, requires_grad=True) for _ in SITES]
        vu_params = [p for p in cg.parameters() if p.requires_grad]
        opt = torch.optim.Adam(vu_params + sources, lr=1e-3)
        opt.zero_grad(set_to_none=True)
        mask_infos = {}
        for s, src in zip(SITES, sources, strict=True):
            ci = torch.rand(bl, SEQ, C, device=DEV)
            mask_infos[s] = ComponentsMaskInfo(
                component_mask=ci + (1 - ci) * src, routing_mask="all"
            )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = cg(idx, mask_infos)
            loss = logits.float().pow(2).mean()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        del cg, idx, sources, vu_params, opt, mask_infos, logits, loss
        return "OK", peak
    except torch.cuda.OutOfMemoryError:
        return "OOM", torch.cuda.max_memory_allocated() / 1e9
    finally:
        torch.cuda.empty_cache()
        gc.collect()


def ppgd_step(
    cg: ComponentGPT2,
    opt: torch.optim.Optimizer,
    idx: torch.Tensor,
    sources: list[torch.Tensor],
) -> None:
    """One PPGD per-rank step: masked recon fwd + bwd to {V, U, sources} + Adam."""
    opt.zero_grad(set_to_none=True)
    mask_infos = {}
    for s, src in zip(SITES, sources, strict=True):
        ci = torch.rand(src.shape[0], SEQ, C, device=DEV)
        mask_infos[s] = ComponentsMaskInfo(component_mask=ci + (1 - ci) * src, routing_mask="all")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = cg(idx, mask_infos)
        loss = logits.float().pow(2).mean()
    loss.backward()
    opt.step()


def profile_step() -> None:
    """Per-kernel torch.profiler breakdown of the PPGD recon fwd+bwd at a fixed bl (single-GPU)."""
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    bl = PROFILE_BL
    print(
        f"GPU {torch.cuda.get_device_name(0)} | profiling GPT2-XL PPGD per-rank phase "
        f"(masked recon fwd+bwd to V/U + sources, {len(SITES)} sites C={C}) at bl={bl}, "
        f"seq={SEQ}, block-ckpt ON, single-GPU no-comm\n"
    )
    cg = build(ckpt=True)
    idx = torch.randint(0, VOCAB, (bl, SEQ), device=DEV)
    sources = [torch.rand(bl, SEQ, C, device=DEV, requires_grad=True) for _ in SITES]
    opt = torch.optim.Adam([p for p in cg.parameters() if p.requires_grad] + sources, lr=1e-3)

    for _ in range(3):  # warmup (alloc/autotune)
        ppgd_step(cg, opt, idx, sources)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        ppgd_step(cg, opt, idx, sources)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))


def main(profile: bool = False) -> None:
    if profile:
        profile_step()
        return
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    cap = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(
        f"GPU {torch.cuda.get_device_name(0)} ~{cap:.0f}GB | GPT2-XL PPGD per-rank "
        f"(full V/U replica {len(SITES)} sites C={C} + sources + Adam), seq {SEQ}"
    )
    print(f"\n{'bl':>4} | {'plain peak':>14} | {'ckpt peak':>14}")
    print("-" * 40)

    def fmt(r: tuple[str, float]) -> str:
        return "OOM" if r[0] == "OOM" else "skip" if r[0] == "skip" else f"{r[1]:.1f}GB"

    plain_oom = ckpt_oom = False
    for bl in BLS:
        rp = ("skip", 0.0) if plain_oom else run(bl, False)
        rc = ("skip", 0.0) if ckpt_oom else run(bl, True)
        plain_oom = plain_oom or rp[0] == "OOM"
        ckpt_oom = ckpt_oom or rc[0] == "OOM"
        print(f"{bl:>4} | {fmt(rp):>14} | {fmt(rc):>14}", flush=True)
    print("\n=> plain bl_pp ceiling = last OK below first OOM | block ckpt extends it")


if __name__ == "__main__":
    fire.Fire(main)
