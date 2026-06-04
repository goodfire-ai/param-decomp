"""Vendored Llama-3.1 component model: componentize fidelity + residual-start equivalence.

CPU, tiny random config (no HF weights). The HF-weight equivalence (vs LlamaForCausalLM) is a
GPU/network check kept out of the unit suite.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from param_decomp.components import make_components
from param_decomp.masks import make_mask_infos
from param_decomp_lab.experiments.lm.vendored.llama_3_1.components import (
    ComponentLinear,
    ComponentLlama,
    componentize_llama,
)
from param_decomp_lab.experiments.lm.vendored.llama_3_1.config import (
    Llama3RopeScaling,
    VendoredLlamaConfig,
)
from param_decomp_lab.experiments.lm.vendored.llama_3_1.model import VendoredLlama

C = 8


def _tiny() -> VendoredLlama:
    cfg = VendoredLlamaConfig(
        model_type="VendoredLlama",
        max_position_embeddings=128,
        vocab_size=64,
        n_layer=4,
        n_head=4,
        n_key_value_heads=2,
        n_embd=32,
        n_intermediate=64,
        rope_theta=500000.0,
        rope_scaling=Llama3RopeScaling(),
    )
    return VendoredLlama(cfg).eval()


def _componentized(model: VendoredLlama, layer: int = 2) -> ComponentLlama:
    targets = {f"layers.{layer}.mlp.{p}": C for p in ("gate_proj", "up_proj", "down_proj")}
    return componentize_llama(model, make_components(model, targets))


def _mask_infos(cm: ComponentLlama, idx: Tensor, seed: int):
    """Rebuilt per call (as the real step does each forward) so the V/U-dependent weight-delta
    graph is fresh; `seed` pins the random masks so two paths get identical masks."""
    torch.manual_seed(seed)
    with torch.no_grad():
        _, pwa = cm.forward_with_pre_weight_acts(idx)
    cmask, wdm = {}, {}
    for site in cm.target_module_paths:
        x = pwa[site]
        ci = torch.rand(*x.shape[:-1], C)
        u = torch.rand_like(ci)
        cmask[site] = ci + (1 - ci) * u
        delta = cm.target_weight(site) - cm.components[site].weight
        wdm[site] = (delta, torch.rand(*x.shape[:-1]))
    return make_mask_infos(cmask, routing_masks="all", weight_deltas_and_masks=wdm)


def test_componentize_clean_forward_is_bit_identical():
    model = _tiny()
    idx = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        base = model(idx)
    cm = _componentized(model)
    with torch.no_grad():
        clean = cm(idx)  # mask_infos=None routes through the frozen target
    assert torch.equal(base, clean)
    assert set(cm.target_module_paths) == {
        f"layers.2.mlp.{p}" for p in ("gate_proj", "up_proj", "down_proj")
    }
    assert all(isinstance(m, ComponentLinear) for m in cm.component_modules.values())


def test_decomposition_start_layer():
    cm = _componentized(_tiny(), layer=3)
    assert cm.decomposition_start_layer == 3


def test_pre_weight_acts_shapes():
    cm = _componentized(_tiny())
    idx = torch.randint(0, 64, (2, 16))
    _, pwa = cm.forward_with_pre_weight_acts(idx)
    assert pwa["layers.2.mlp.gate_proj"].shape == (2, 16, 32)  # d_model
    assert pwa["layers.2.mlp.up_proj"].shape == (2, 16, 32)
    assert pwa["layers.2.mlp.down_proj"].shape == (2, 16, 64)  # n_intermediate


def test_residual_start_matches_full_forward_logits_and_grads():
    torch.manual_seed(0)
    cm = _componentized(_tiny())
    start = cm.decomposition_start_layer
    idx = torch.randint(0, 64, (2, 16))

    with torch.no_grad():
        mi = _mask_infos(cm, idx, seed=7)
        full = cm(idx, mask_infos=mi)
        rs = cm.forward_from_residual(cm.residual_at(idx, start), start, mask_infos=mi)
    assert torch.allclose(full, rs, atol=1e-6), (full - rs).abs().max()

    def vu_grads(use_residual_start: bool) -> dict[str, Tensor]:
        for s in cm.target_module_paths:
            cm.components[s].V.grad = cm.components[s].U.grad = None
        mask_infos = _mask_infos(cm, idx, seed=7)  # identical masks, fresh delta graph
        out = (
            cm.forward_from_residual(cm.residual_at(idx, start), start, mask_infos=mask_infos)
            if use_residual_start
            else cm(idx, mask_infos=mask_infos)
        )
        F.mse_loss(out, torch.zeros_like(out)).backward()
        result: dict[str, Tensor] = {}
        for s in cm.target_module_paths:
            g = cm.components[s].V.grad
            assert g is not None
            result[s] = g.clone()
        return result

    gf, gr = vu_grads(False), vu_grads(True)
    for s in cm.target_module_paths:
        assert torch.allclose(gf[s], gr[s], atol=1e-6), (s, (gf[s] - gr[s]).abs().max())


def test_use_cached_residual_context_matches_full_forward():
    # The context the 3-pool steps actually use: forward() inside it runs the cached suffix.
    cm = _componentized(_tiny())
    idx = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        mi = _mask_infos(cm, idx, seed=3)
        full = cm(idx, mask_infos=mi)
        with cm.use_cached_residual(idx):
            cached = cm(idx, mask_infos=mi)
    assert torch.allclose(full, cached, atol=1e-6), (full - cached).abs().max()
