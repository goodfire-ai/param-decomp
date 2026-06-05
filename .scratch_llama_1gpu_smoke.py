"""Single-GPU integration smoke: componentize real Llama-3.1-8B @ L18 (3 MLP matrices) +
a 1.7B global-shared-transformer CI fn, run the layerwise masked-recon forward/backward, and
verify grads flow to V/U and the CI fn. Also reports peak memory (single-rank probe)."""

import torch
import torch.nn.functional as F
from param_decomp_lab.experiments.lm.vendored.llama import VendoredLlama

from param_decomp.ci_fns import AttnConfig, GlobalCiConfig, GlobalSharedTransformerCiConfig
from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp.masks import make_mask_infos
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel

dev = "cuda"
GB = 1024**3
torch.manual_seed(0)

target = VendoredLlama.from_hf_pretrained("meta-llama/Llama-3.1-8B")
target.enable_activation_checkpointing()
target.eval()

targets = [
    DecompositionTarget(module_path=f"layers.18.mlp.{p}", C=4096)
    for p in ("gate_proj", "up_proj", "down_proj")
]
ci_config = GlobalCiConfig(
    fn_type="global_shared_transformer",
    mode="global",
    hidden_dims=None,
    simple_transformer_ci_cfg=GlobalSharedTransformerCiConfig(
        d_model=4096,
        n_blocks=8,
        mlp_hidden_dim=[16384],
        attn_config=AttnConfig(n_heads=32, max_len=1024, rope_base=10000.0),
    ),
)
cm = LMComponentModel.build(target, targets, ci_config, "leaky_hard").to(dev)
n_ci = sum(p.numel() for p in cm.ci_fn.parameters())
n_vu = sum(
    p.numel() for s in cm.target_module_paths for p in (cm.components[s].V, cm.components[s].U)
)
print(
    f"CI fn params: {n_ci / 1e9:.3f}B | V/U params: {n_vu / 1e6:.1f}M | sites: {cm.target_module_paths}"
)
print(f"resident after .to(cuda): {torch.cuda.memory_allocated() / GB:.1f} GB")

idx = torch.randint(0, 128256, (4, 1024), device=dev)

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), cm.bypass_lm_head():
    clean_hidden = cm(idx).float()  # [4,1024,4096], target path

with torch.autocast("cuda", dtype=torch.bfloat16):
    _, pwa = cm.forward_with_pre_weight_acts(idx)
    print("pre_weight_acts:", {k: tuple(v.shape) for k, v in pwa.items()})
    ci = cm.calc_causal_importances(pwa, sampling="continuous")
    cmask, wdm = {}, {}
    for site in cm.target_module_paths:
        ci_s = ci.lower_leaky[site]
        u = torch.rand_like(ci_s)
        cmask[site] = ci_s + (1 - ci_s) * u
        delta = cm.target_weight(site) - cm.components[site].weight
        dmask = torch.rand(ci_s.shape[:-1], device=dev, dtype=ci_s.dtype)
        wdm[site] = (delta, dmask)
    mask_infos = make_mask_infos(cmask, routing_masks="all", weight_deltas_and_masks=wdm)
    with cm.bypass_lm_head():
        pred_hidden = cm(idx, mask_infos=mask_infos).float()

mean_l0 = sum((ci.lower_leaky[s] > 0).float().sum(-1).mean().item() for s in cm.target_module_paths)
loss = F.mse_loss(pred_hidden, clean_hidden)
print(f"recon mse (random init, pre-train): {loss.item():.4f} | summed mean_l0/site: {mean_l0:.0f}")
loss.backward()

for site in cm.target_module_paths:
    gV, gU = cm.components[site].V.grad, cm.components[site].U.grad
    assert gV is not None and torch.isfinite(gV).all(), f"bad V grad {site}"
    assert gU is not None and torch.isfinite(gU).all(), f"bad U grad {site}"
ci_grads = [p.grad for p in cm.ci_fn.parameters() if p.grad is not None]
ci_grad_norm = torch.sqrt(sum((g.float() ** 2).sum() for g in ci_grads)).item()
print(
    f"V/U grads finite ✓ | CI-fn params with grad: {len(ci_grads)} | CI grad norm: {ci_grad_norm:.3e}"
)
print(
    f"PEAK mem (target+CIfn+VU+acts, fp32 target, bl=4): {torch.cuda.max_memory_allocated() / GB:.1f} GB"
)
print("SMOKE PASSED")
