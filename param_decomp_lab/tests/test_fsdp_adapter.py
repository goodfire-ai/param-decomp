"""`FsdpComponentAdapter` presents the core `ComponentModel` surface over a vendored
`LMComponentModel`. We build a tiny vendored GPT2 component model (same fixtures as
`test_vendored_component_model.py`) and assert the adapter's
`forward(batch, cache_type="input")` matches a direct `forward_with_pre_weight_acts`.
"""

import torch
from torch import nn

from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp_config.ci_fn import LayerwiseCiConfig
from param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple import (
    GPT2Simple,
    GPT2SimpleConfig,
)
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel
from param_decomp_lab.fsdp.component_adapter import FsdpComponentAdapter

SITES = ["h.0.attn.q_proj", "h.0.attn.k_proj", "h.1.attn.q_proj", "h.1.attn.k_proj"]
C, B, T, VOCAB, D = 6, 2, 5, 32, 16
SEED = 7


def _frozen_model() -> GPT2Simple:
    cfg = GPT2SimpleConfig(
        model_type="GPT2Simple", n_layer=2, n_head=2, n_embd=D, vocab_size=VOCAB, block_size=8
    )
    m = GPT2Simple(cfg)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _build_lm() -> LMComponentModel:
    targets = [DecompositionTarget(module_path=s, C=C) for s in SITES]
    ci_config = LayerwiseCiConfig(fn_type="mlp", hidden_dims=[32])
    torch.manual_seed(SEED)
    return LMComponentModel.build(_frozen_model(), targets, ci_config, sigmoid_type="leaky_hard")


def test_adapter_input_cache_matches_direct() -> None:
    lm = _build_lm()
    adapter = FsdpComponentAdapter(lm)
    idx = torch.randint(0, VOCAB, (B, T))

    out = adapter(idx, cache_type="input")
    direct_logits, direct_acts = lm.forward_with_pre_weight_acts(idx)

    assert torch.allclose(out.output, direct_logits)
    assert set(out.cache) == set(direct_acts) == set(SITES)
    for site in SITES:
        assert torch.allclose(out.cache[site], direct_acts[site]), site


def test_adapter_none_cache_matches_clean_forward() -> None:
    lm = _build_lm()
    adapter = FsdpComponentAdapter(lm)
    idx = torch.randint(0, VOCAB, (B, T))
    assert torch.allclose(adapter(idx), lm.forward(idx))


def test_adapter_is_module_with_no_own_params() -> None:
    lm = _build_lm()
    adapter = FsdpComponentAdapter(lm)
    assert isinstance(adapter, nn.Module)
    own = [n for n, _ in adapter.named_parameters() if not n.startswith("lm.")]
    assert own == []


def test_adapter_delegates_pure_queries() -> None:
    lm = _build_lm()
    adapter = FsdpComponentAdapter(lm)
    assert adapter.module_to_c == lm.module_to_c
    assert adapter.target_module_paths == lm.target_module_paths
    assert set(adapter.components) == set(lm.components)
    assert adapter.ci_fn is lm.ci_fn
    assert adapter.model is lm.model
