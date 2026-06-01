"""Parity for the vendored ComponentGPT2.

- clean forward (`mask_infos=None`) is bit-identical to the original `GPT2Simple`;
- masked forward is bit-identical to the hook-based `ComponentModel` path (same components);
- masked forward under activation checkpointing yields grads (to V/U AND to the mask `source`)
  bit-identical to the non-checkpointed run — with grad entering ONLY through the masks
  (embeddings are frozen, so the block input does not require grad: the real 3-pool case).
"""

from typing import Any, cast

import torch
from torch import Tensor

from param_decomp.component_model import ComponentModel
from param_decomp.components import make_components
from param_decomp.masks import ComponentsMaskInfo
from param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple import (
    GPT2Simple,
    GPT2SimpleConfig,
)
from param_decomp_lab.experiments.lm.vendored.gpt2 import componentize_gpt2

SITES = ["h.0.attn.q_proj", "h.0.attn.k_proj", "h.1.attn.q_proj", "h.1.attn.k_proj"]
C, B, T, VOCAB = 6, 2, 5, 32


def _small_model() -> GPT2Simple:
    cfg = GPT2SimpleConfig(
        model_type="GPT2Simple", n_layer=2, n_head=2, n_embd=16, vocab_size=VOCAB, block_size=8
    )
    torch.manual_seed(0)
    return GPT2Simple(cfg)


def _module_to_c() -> dict[str, int]:
    return {s: C for s in SITES}


def _oracle_logits(
    m: GPT2Simple, idx: Tensor, comps: dict[str, Any], mask_infos: dict[str, ComponentsMaskInfo]
) -> Tensor:
    """Hook-based ComponentModel forward, replicated by attaching the real hook to the leaves."""
    handles = []
    for path, comp in comps.items():
        mi = mask_infos[path]

        def hook(
            mod: Any, args: Any, kwargs: Any, output: Any, comp: Any = comp, mi: Any = mi, path: str = path
        ) -> Any:
            return cast(Any, ComponentModel._components_and_cache_hook)(
                None, mod, args, kwargs, output,
                module_name=path, components=comp, mask_info=mi, cache_type="none", cache={},
            )

        handles.append(m.get_submodule(path).register_forward_hook(hook, with_kwargs=True))
    out = m(idx)[0]
    assert out is not None
    for h in handles:
        h.remove()
    return out


def _mask_infos(comps: dict[str, Any], *, with_delta: bool) -> dict[str, ComponentsMaskInfo]:
    torch.manual_seed(1)
    out: dict[str, ComponentsMaskInfo] = {}
    for path, comp in comps.items():
        wd = (torch.randn_like(comp.weight), torch.rand(B, T)) if with_delta else None
        out[path] = ComponentsMaskInfo(
            component_mask=torch.rand(B, T, C), routing_mask="all", weight_delta_and_mask=wd
        )
    return out


def test_clean_forward_matches_original() -> None:
    m = _small_model()
    idx = torch.randint(0, VOCAB, (B, T))
    ref = m(idx)[0]
    assert ref is not None
    ref = ref.detach().clone()
    cg = componentize_gpt2(m, make_components(m, _module_to_c()))
    assert torch.equal(cg(idx, None), ref)


def test_masked_forward_matches_hook_oracle() -> None:
    m = _small_model()
    idx = torch.randint(0, VOCAB, (B, T))
    comps = make_components(m, _module_to_c())
    mis = _mask_infos(comps, with_delta=True)
    oracle = _oracle_logits(m, idx, comps, mis).detach().clone()
    cg = componentize_gpt2(m, comps)
    assert torch.equal(cg(idx, mis), oracle)


def test_masked_ckpt_grad_equivalence() -> None:
    m = _small_model()
    idx = torch.randint(0, VOCAB, (B, T))
    comps = make_components(m, _module_to_c())
    cg = componentize_gpt2(m, comps)

    def run(use_ckpt: bool) -> dict[str, Tensor]:
        cg._use_activation_checkpointing = use_ckpt
        for p in cg.parameters():
            p.grad = None
        torch.manual_seed(2)
        sources: dict[str, Tensor] = {}
        mis: dict[str, ComponentsMaskInfo] = {}
        for path, comp in comps.items():
            ci = torch.rand(B, T, C)
            src = torch.rand(B, T, C, requires_grad=True)
            sources[path] = src
            mis[path] = ComponentsMaskInfo(component_mask=ci + (1 - ci) * src, routing_mask="all")
        loss = cg(idx, mis).float().pow(2).mean()
        loss.backward()
        grads: dict[str, Tensor] = {}
        for path, comp in comps.items():
            v_grad, u_grad, src_grad = comp.V.grad, comp.U.grad, sources[path].grad
            assert v_grad is not None and u_grad is not None
            assert src_grad is not None, f"no grad to source {path} (ckpt={use_ckpt})"
            grads[f"{path}.V"] = v_grad.clone()
            grads[f"{path}.U"] = u_grad.clone()
            grads[f"{path}.src"] = src_grad.clone()
        return grads

    g_plain = run(False)
    g_ckpt = run(True)
    for k in g_plain:
        assert torch.equal(g_plain[k], g_ckpt[k]), f"grad mismatch at {k}"
