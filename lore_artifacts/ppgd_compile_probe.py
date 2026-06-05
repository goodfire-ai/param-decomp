"""Probe: is torch.compile + autograd.grad (PPGD's path) correct & faster on the
vendored-Llama masked forward (ckpt + flash-SDPA in the graph)?

PPGD is the one heavy pool left uncompiled, skipped only because autograd.grad-under-compile
was never validated (three_pool/optimize.py comment). This isolates that exact combo on a
small-but-real VendoredLlama: masked forward via make_mask_infos, then ONE torch.autograd.grad
w.r.t. [V, U, ci_leaves] (the multi-target grad PPGD does), eager vs compiled.
"""

import time

import torch

from param_decomp.ci_fns import LayerwiseCiConfig
from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp.masks import make_mask_infos
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel
from param_decomp_lab.experiments.lm.vendored.llama_3_1.config import VendoredLlamaConfig
from param_decomp_lab.experiments.lm.vendored.llama_3_1.model import VendoredLlama

DEV = torch.device("cuda")
B, T, C = 2, 128, 32
SITES = ["layers.1.mlp.gate_proj", "layers.1.mlp.up_proj", "layers.1.mlp.down_proj"]


def build_model() -> LMComponentModel:
    cfg = VendoredLlamaConfig(
        model_type="VendoredLlama",
        max_position_embeddings=T,
        vocab_size=64,
        n_layer=4,
        n_head=4,
        n_key_value_heads=2,
        n_embd=128,
        n_intermediate=256,
    )
    torch.manual_seed(0)
    base = VendoredLlama(cfg)
    for p in base.parameters():
        p.requires_grad_(False)
    targets = [DecompositionTarget(module_path=s, C=C) for s in SITES]
    ci_config = LayerwiseCiConfig(fn_type="mlp", hidden_dims=[32])
    torch.manual_seed(1)
    lm = LMComponentModel.build(base, targets, ci_config, sigmoid_type="leaky_hard").to(DEV)
    lm.model.enable_activation_checkpointing()
    return lm


def make_inputs(lm: LMComponentModel):
    torch.manual_seed(2)
    idx = torch.randint(0, 64, (B, T), device=DEV)
    with torch.no_grad():
        target_out = lm(idx).detach()
    # CI mask leaves [B, T, C] per site, fp32 requires_grad — exactly PPGD's _releaf_ci_fp32.
    ci_leaves = {
        s: torch.rand(B, T, C, device=DEV, dtype=torch.float32, requires_grad=True) for s in SITES
    }
    return idx, target_out, ci_leaves


def grads_via_autograd_grad(lm, idx, target_out, ci_leaves, *, autocast: bool) -> dict[str, torch.Tensor]:
    """One masked forward + ONE torch.autograd.grad w.r.t. [V, U, ci_leaves] — PPGD's D5."""
    mask_infos = make_mask_infos({s: ci_leaves[s] for s in SITES}, routing_masks="all")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        out = lm(idx, mask_infos)
        loss = ((out - target_out) ** 2).sum()
    v_params = [lm.components[s].V for s in SITES]
    u_params = [lm.components[s].U for s in SITES]
    leaves = [ci_leaves[s] for s in SITES]
    flat = torch.autograd.grad(loss, v_params + u_params + leaves, retain_graph=False)
    n = len(SITES)
    out_grads = {"loss": loss.detach()}
    for i, s in enumerate(SITES):
        out_grads[f"V/{s}"] = flat[i]
        out_grads[f"U/{s}"] = flat[n + i]
        out_grads[f"ci/{s}"] = flat[2 * n + i]
    return out_grads


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


def time_it(fn, iters=10) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


def compare(autocast: bool) -> tuple[float, float, float, float]:
    """Fresh model: eager then compiled autograd.grad. Returns (worst_grad_relerr,
    loss_relerr, t_eager_ms, t_compiled_ms)."""
    lm = build_model()
    idx, target_out, ci_leaves = make_inputs(lm)
    eager = grads_via_autograd_grad(lm, idx, target_out, ci_leaves, autocast=autocast)
    t_eager = time_it(lambda: grads_via_autograd_grad(lm, idx, target_out, ci_leaves, autocast=autocast))

    lm.model.compile()
    compiled = grads_via_autograd_grad(lm, idx, target_out, ci_leaves, autocast=autocast)
    t_compiled = time_it(lambda: grads_via_autograd_grad(lm, idx, target_out, ci_leaves, autocast=autocast))

    worst = 0.0
    for k in eager:
        if k == "loss":
            continue
        e = rel_err(compiled[k], eager[k])
        worst = max(worst, e)
        print(f"    {k:28s} rel_err = {e:.2e}")
    loss_err = abs(compiled["loss"].item() - eager["loss"].item()) / (abs(eager["loss"].item()) + 1e-12)
    print(f"    {'loss':28s} rel_err = {loss_err:.2e}")
    return worst, loss_err, t_eager, t_compiled


def main() -> None:
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name()}")

    print("\n=== fp32 (no autocast) — the CORRECTNESS check ===")
    w32, l32, te32, tc32 = compare(autocast=False)
    print(f"  WORST grad rel_err (fp32) = {w32:.2e}   eager {te32:.2f}ms -> compiled {tc32:.2f}ms ({te32/tc32:.2f}x)")

    print("\n=== bf16 autocast — the PRODUCTION numerics + speed ===")
    wbf, lbf, tebf, tcbf = compare(autocast=True)
    print(f"  WORST grad rel_err (bf16) = {wbf:.2e}   eager {tebf:.2f}ms -> compiled {tcbf:.2f}ms ({tebf/tcbf:.2f}x)")

    correct = w32 < 1e-3
    print(
        f"\nVERDICT: compile+autograd.grad CORRECT = {'PASS' if correct else 'FAIL'} "
        f"(fp32 worst {w32:.2e}); bf16 delta {wbf:.2e} is {'benign autocast reordering' if correct else 'SUSPECT'}; "
        f"speedup ~{tebf/tcbf:.1f}x"
    )


if __name__ == "__main__":
    main()
