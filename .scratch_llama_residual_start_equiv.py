"""Validate residual-start == full forward on real Llama-3.1-8B @ L18: logits bit-identity
and V/U grad equivalence (the house rule: real one-backward grad check, RNG pinned)."""

import torch
import torch.nn.functional as F
from param_decomp_lab.experiments.lm.vendored.llama import VendoredLlama

from param_decomp.ci_fns import AttnConfig, GlobalCiConfig, GlobalSharedTransformerCiConfig
from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp.masks import make_mask_infos
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel

dev = "cuda"
torch.manual_seed(0)
target = VendoredLlama.from_hf_pretrained("meta-llama/Llama-3.1-8B").eval()
targets = [
    DecompositionTarget(module_path=f"layers.18.mlp.{p}", C=2048)
    for p in ("gate_proj", "up_proj", "down_proj")
]
ci_config = GlobalCiConfig(
    fn_type="global_shared_transformer",
    mode="global",
    hidden_dims=None,
    simple_transformer_ci_cfg=GlobalSharedTransformerCiConfig(
        d_model=2048,
        n_blocks=2,
        mlp_hidden_dim=[8192],
        attn_config=AttnConfig(n_heads=16, max_len=1024, rope_base=10000.0),
    ),
)
cm = LMComponentModel.build(target, targets, ci_config, "leaky_hard").to(dev)
model = cm.model
start = model.decomposition_start_layer
print("decomposition_start_layer:", start)

idx = torch.randint(0, 128256, (2, 256), device=dev)

# fixed mask_infos (constant across both paths)
torch.manual_seed(1)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    _, pwa = model.forward_with_pre_weight_acts(idx)
    ci = cm.calc_causal_importances(pwa, sampling="continuous")
cmask, wdm = {}, {}
for site in model.target_module_paths:
    ci_s = ci.lower_leaky[site].detach()
    u = torch.rand_like(ci_s)
    cmask[site] = (ci_s + (1 - ci_s) * u).detach()
    wdm[site] = (
        model.target_weight(site) - cm.components[site].weight.detach(),
        torch.rand(ci_s.shape[:-1], device=dev, dtype=ci_s.dtype),
    )
mask_infos = make_mask_infos(cmask, routing_masks="all", weight_deltas_and_masks=wdm)

# --- forward bit-identity (no grad, bypass head) ---
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), cm.bypass_lm_head():
    full = model(idx, mask_infos=mask_infos).float()
    r = model.residual_at(idx, start)
    rs = model.forward_from_residual(r, start, mask_infos=mask_infos).float()
print(f"forward  max|Δ| full-vs-residual-start: {(full - rs).abs().max().item():.3e}")


def grads(use_residual_start: bool):
    for s in model.target_module_paths:
        cm.components[s].V.grad = None
        cm.components[s].U.grad = None
    tgt = torch.zeros(idx.shape[0], idx.shape[1], model.config.n_embd, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with cm.bypass_lm_head():
            if use_residual_start:
                r = model.residual_at(idx, start)
                out = model.forward_from_residual(r, start, mask_infos=mask_infos)
            else:
                out = model(idx, mask_infos=mask_infos)
        F.mse_loss(out.float(), tgt).backward()
    return {s: cm.components[s].V.grad.detach().clone() for s in model.target_module_paths}


gf, gr = grads(False), grads(True)
for s in model.target_module_paths:
    d = (gf[s] - gr[s]).abs().max().item()
    rel = d / gf[s].abs().max().item()
    print(f"  V.grad {s:28s} max|Δ|={d:.3e} rel={rel:.3e}")
print("RESIDUAL-START EQUIVALENCE OK")
