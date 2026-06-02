"""RNG-pinned grad-equivalence check for the bf16-mask cast-elimination changes.

Validates that the two cast-elim edits preserve grad SCALING (not just loss curves — per the
team lesson that grad-scaling bugs hide from loss curves and need a real one-backward check):

  (1) components.py forward: `component_acts * mask.to(component_acts.dtype)`
  (2) drop the fp32 upcast in `_releaf_ci_fp32_for_grads` (bf16 CI leaf)

A pure dtype change cannot alter scaling — only precision — so the pass criterion is:
the shipped g_CI (always bf16 on the wire), V.grad, and U.grad keep ratio ≈ 1.0 and cosine
≈ 1.0 vs the original fp32 path, with only bf16-level (~1e-2) elementwise error.

Two-phase so it brackets the edits:
  PHASE=baseline  → run on the ORIGINAL code, save grads to --out.
  PHASE=compare   → run after the edits, load baseline, compare fp32-leaf AND bf16-leaf paths.

Tiny model so it runs in seconds on one GPU. The masked recon mirrors `_layerwise_one_site`
(bypass_lm_head + fused_linear_kl_div) exactly, using the real make_mask_infos / forward.

Run: srun --gres=gpu:1 python scripts/check_bf16_mask_grad_equiv.py baseline --out /tmp/g0.pt
     srun --gres=gpu:1 python scripts/check_bf16_mask_grad_equiv.py compare  --out /tmp/g0.pt
"""

import sys
from pathlib import Path

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from param_decomp.components import make_components  # noqa: E402
from param_decomp.fused_linear_kl import fused_linear_kl_div  # noqa: E402
from param_decomp.masks import make_mask_infos  # noqa: E402
from param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple import (  # noqa: E402
    GPT2Simple,
    GPT2SimpleConfig,
)
from param_decomp_lab.experiments.lm.vendored.gpt2 import componentize_gpt2  # noqa: E402

N_LAYER, D_MODEL, N_HEAD, VOCAB, SEQ, C, BL = 2, 128, 4, 512, 32, 64, 8
SITE = "h.0.attn.q_proj"
DEV = "cuda"


def build():
    cfg = GPT2SimpleConfig(
        model_type="GPT2Simple",
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=D_MODEL,
        vocab_size=VOCAB,
        block_size=SEQ,
    )
    torch.manual_seed(0)
    m = GPT2Simple(cfg)
    return componentize_gpt2(m, make_components(m, {SITE: C})).to(DEV)


def fixed_inputs() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    g = torch.Generator(device=DEV).manual_seed(1234)
    idx = torch.randint(0, VOCAB, (BL, SEQ), device=DEV, generator=g)
    ci_recv = torch.rand(BL, SEQ, C, device=DEV, dtype=torch.bfloat16, generator=g)  # bf16 wire
    u = torch.rand(BL, SEQ, C, device=DEV, dtype=torch.float32, generator=g)  # source base (fp32)
    delta_mask = torch.rand(BL, SEQ, device=DEV, dtype=torch.float32, generator=g)
    return idx, ci_recv, u, delta_mask


def lw_grads(cg, idx, ci_recv, u, delta_mask, leaf_dtype: torch.dtype) -> dict[str, Tensor]:
    """One masked-recon fwd+bwd mirroring _layerwise_one_site; returns shipped grads (bf16 g_CI)."""
    for p in cg.parameters():
        p.grad = None
    leaf = ci_recv.detach().to(leaf_dtype).clone().requires_grad_(True)
    u_l = u.to(leaf_dtype)
    dm = delta_mask.to(leaf_dtype)
    comp = cg.get_submodule(SITE)
    with cg.bypass_lm_head() as lm_w, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            target_h = cg(idx, None)
        mask = leaf + (1 - leaf) * u_l
        delta = comp.target_weight - comp.components.weight
        mi = make_mask_infos(
            {SITE: mask}, weight_deltas_and_masks={SITE: (delta, dm)}, routing_masks="all"
        )
        pred_h = cg(idx, mask_infos=mi)
        loss, n = fused_linear_kl_div(
            pred_h.reshape(-1, D_MODEL), target_h.reshape(-1, D_MODEL), lm_w
        )
        (loss / n).backward()
    return {
        "g_ci_wire": leaf.grad.detach().to(torch.bfloat16).float(),  # what crosses the wire
        "g_V": comp.components.V.grad.detach().float(),
        "g_U": comp.components.U.grad.detach().float(),
    }


def report(label: str, ref: dict[str, Tensor], got: dict[str, Tensor]) -> None:
    print(f"\n[{label}]")
    for k in ref:
        a, b = ref[k].flatten(), got[k].flatten()
        ratio = (b.abs().mean() / a.abs().mean()).item()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        rel = ((b - a).abs() / (a.abs() + 1e-8)).mean().item()
        flag = "OK" if (0.98 < ratio < 1.02 and cos > 0.999) else "  <-- CHECK"
        print(f"  {k:10s} ratio={ratio:.4f} cos={cos:.5f} mean_rel_err={rel:.4f} {flag}")


def main() -> None:
    assert torch.cuda.is_available()
    phase = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path("/tmp/g0.pt")
    cg = build()
    idx, ci_recv, u, delta_mask = fixed_inputs()

    if phase == "baseline":
        g = lw_grads(cg, idx, ci_recv, u, delta_mask, torch.float32)
        torch.save(g, out)
        print(f"baseline (fp32 leaf, ORIGINAL code) saved -> {out}")
        for k, v in g.items():
            print(f"  {k:10s} |mean|={v.abs().mean().item():.3e}")
    elif phase == "compare":
        ref = torch.load(out)
        report("new fp32 leaf vs baseline (isolates components.py edit)",
               ref, lw_grads(cg, idx, ci_recv, u, delta_mask, torch.float32))
        report("new bf16 leaf vs baseline (full fix)",
               ref, lw_grads(cg, idx, ci_recv, u, delta_mask, torch.bfloat16))
    else:
        raise SystemExit(f"unknown phase {phase!r}")


if __name__ == "__main__":
    main()
