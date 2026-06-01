"""Parity: vendored ComponentLinear/ComponentEmbedding == ComponentModel's hook.

`ComponentModel._components_and_cache_hook` uses no `self` state, so we call it directly as
the oracle and assert the in-tree modules reproduce it bit-for-bit across every routing /
delta / cache combination.
"""

from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from param_decomp.component_model import ComponentModel
from param_decomp.components import EmbeddingComponents, LinearComponents
from param_decomp.masks import ComponentsMaskInfo
from param_decomp_lab.experiments.lm.vendored.component_modules import (
    ComponentEmbedding,
    ComponentLinear,
)

B, T, D_IN, D_OUT, C, VOCAB, DIM = 2, 3, 16, 24, 8, 32, 12


def _hook(
    output: Tensor,
    components: object,
    mask_info: ComponentsMaskInfo,
    x: Tensor,
    cache_type: str,
    cache: dict[str, Tensor],
) -> Tensor:
    # `_components_and_cache_hook` uses no `self` state and is awkwardly typed; cast to Any
    # so we can call it as a plain oracle with self=None.
    out = cast(Any, ComponentModel._components_and_cache_hook)(
        None,
        None,
        [x],
        {},
        output,
        module_name="m",
        components=components,
        mask_info=mask_info,
        cache_type=cache_type,
        cache=cache,
    )
    assert isinstance(out, Tensor)
    return out


def _linear_setup() -> tuple[LinearComponents, Tensor, ComponentLinear, Tensor, Tensor]:
    torch.manual_seed(0)
    comps = LinearComponents(C=C, d_in=D_IN, d_out=D_OUT, bias=torch.randn(D_OUT))
    w = torch.randn(D_OUT, D_IN)
    cl = ComponentLinear(comps, w)
    x = torch.randn(B, T, D_IN)
    target_out = F.linear(x, w, comps.bias)
    return comps, w, cl, x, target_out


def test_linear_routing_all() -> None:
    comps, _, cl, x, target_out = _linear_setup()
    mi = ComponentsMaskInfo(component_mask=torch.rand(B, T, C), routing_mask="all")
    assert torch.equal(cl(x, mi), _hook(target_out, comps, mi, x, "none", {}))


def test_linear_routing_partial() -> None:
    comps, _, cl, x, target_out = _linear_setup()
    routing = torch.rand(B, T) < 0.5
    mi = ComponentsMaskInfo(component_mask=torch.rand(B, T, C), routing_mask=routing)
    assert torch.equal(cl(x, mi), _hook(target_out, comps, mi, x, "none", {}))


def test_linear_weight_delta() -> None:
    comps, _, cl, x, target_out = _linear_setup()
    wd = (torch.randn(D_OUT, D_IN), torch.rand(B, T))
    mi = ComponentsMaskInfo(
        component_mask=torch.rand(B, T, C), routing_mask="all", weight_delta_and_mask=wd
    )
    assert torch.equal(cl(x, mi), _hook(target_out, comps, mi, x, "none", {}))


def test_linear_component_acts_cache() -> None:
    comps, _, cl, x, target_out = _linear_setup()
    mi = ComponentsMaskInfo(component_mask=torch.rand(B, T, C), routing_mask="all")
    cache_hook: dict[str, Tensor] = {}
    hook_out = _hook(target_out, comps, mi, x, "component_acts", cache_hook)
    cache_cl: dict[str, Tensor] = {}
    cl_out = cl(x, mi, component_acts_cache=cache_cl)
    assert torch.equal(cl_out, hook_out)
    assert torch.equal(cache_cl["pre_detach"], cache_hook["m_pre_detach"])
    assert torch.equal(cache_cl["post_detach"], cache_hook["m_post_detach"])


def test_linear_no_mask_is_target() -> None:
    _, _, cl, x, target_out = _linear_setup()
    assert torch.equal(cl(x, None), target_out)


def _embedding_setup() -> tuple[EmbeddingComponents, ComponentEmbedding, Tensor, Tensor]:
    torch.manual_seed(1)
    comps = EmbeddingComponents(C=C, vocab_size=VOCAB, embedding_dim=DIM)
    w = torch.randn(VOCAB, DIM)
    ce = ComponentEmbedding(comps, w)
    x = torch.randint(0, VOCAB, (B, T))
    target_out = w[x]
    return comps, ce, x, target_out


def test_embedding_routing_all() -> None:
    comps, ce, x, target_out = _embedding_setup()
    mi = ComponentsMaskInfo(component_mask=torch.rand(B, T, C), routing_mask="all")
    assert torch.equal(ce(x, mi), _hook(target_out, comps, mi, x, "none", {}))


def test_embedding_routing_partial_with_delta() -> None:
    comps, ce, x, target_out = _embedding_setup()
    routing = torch.rand(B, T) < 0.5
    wd = (torch.randn(VOCAB, DIM), torch.rand(B, T))
    mi = ComponentsMaskInfo(
        component_mask=torch.rand(B, T, C), routing_mask=routing, weight_delta_and_mask=wd
    )
    assert torch.equal(ce(x, mi), _hook(target_out, comps, mi, x, "none", {}))
