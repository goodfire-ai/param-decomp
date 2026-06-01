"""Equivalence: vendored LMComponentModel == core ComponentModel on the shared surface.

Both are built on identical GPT2Simple weights with identical RNG so their components and CI fn
initialise the same; we then assert `calc_causal_importances`, `calc_weight_deltas`,
`target_weight`, the clean forward, and `pre_weight_acts` (vs `cache_type="input"`) all match
bit-for-bit.
"""

from typing import Any

import torch
from torch import Tensor, nn

from param_decomp.ci_fns import LayerwiseCiConfig
from param_decomp.component_model import ComponentModel
from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple import (
    GPT2Simple,
    GPT2SimpleConfig,
)
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel

SITES = ["h.0.attn.q_proj", "h.0.attn.k_proj", "h.1.attn.q_proj", "h.1.attn.k_proj"]
C, B, T, VOCAB, D = 6, 2, 5, 32, 16
SEED = 7


def _run_batch(model: nn.Module, batch: Any) -> Tensor:
    out = model(batch)
    return out[0] if isinstance(out, tuple) else out


def _frozen_model() -> GPT2Simple:
    cfg = GPT2SimpleConfig(
        model_type="GPT2Simple", n_layer=2, n_head=2, n_embd=D, vocab_size=VOCAB, block_size=8
    )
    m = GPT2Simple(cfg)  # weights init from the model's own seed-42 generator, independent of global RNG
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _build_pair() -> tuple[ComponentModel, LMComponentModel]:
    targets = [DecompositionTarget(module_path=s, C=C) for s in SITES]
    ci_config = LayerwiseCiConfig(fn_type="mlp", hidden_dims=[32])

    torch.manual_seed(SEED)
    cm = ComponentModel(
        _frozen_model(), _run_batch, targets, ci_config, sigmoid_type="leaky_hard"
    )
    torch.manual_seed(SEED)
    lm = LMComponentModel.build(_frozen_model(), targets, ci_config, sigmoid_type="leaky_hard")
    return cm, lm


def test_target_weight_and_weight_deltas_match() -> None:
    cm, lm = _build_pair()
    for site in SITES:
        assert torch.equal(cm.target_weight(site), lm.target_weight(site))
    cm_d, lm_d = cm.calc_weight_deltas(), lm.calc_weight_deltas()
    assert set(cm_d) == set(lm_d)
    for site in cm_d:
        assert torch.equal(cm_d[site], lm_d[site])


def test_calc_causal_importances_match() -> None:
    cm, lm = _build_pair()
    torch.manual_seed(0)
    acts = {site: torch.randn(B, T, D) for site in SITES}
    cm_ci = cm.calc_causal_importances(acts, sampling="continuous", detach_inputs=False)
    lm_ci = lm.calc_causal_importances(acts, sampling="continuous", detach_inputs=False)
    for field in ("lower_leaky", "upper_leaky", "pre_sigmoid"):
        cm_d, lm_d = getattr(cm_ci, field), getattr(lm_ci, field)
        assert set(cm_d) == set(lm_d) == set(SITES)
        for site in SITES:
            assert torch.equal(cm_d[site], lm_d[site]), f"{field}/{site}"


def test_clean_forward_and_pre_weight_acts_match() -> None:
    cm, lm = _build_pair()
    idx = torch.randint(0, VOCAB, (B, T))

    cm_clean = cm(idx)
    lm_clean = lm(idx)
    assert torch.equal(cm_clean, lm_clean)

    cm_out = cm(idx, cache_type="input")
    lm_logits, lm_acts = lm.forward_with_pre_weight_acts(idx)
    assert torch.equal(cm_out.output, lm_logits)
    assert set(cm_out.cache) == set(lm_acts) == set(SITES)
    for site in SITES:
        assert torch.equal(cm_out.cache[site], lm_acts[site]), site
