import random

import torch

from param_decomp_lab.autointerp.config import CompactSkepticalConfig
from param_decomp_lab.autointerp.providers import OpenRouterLLMConfig
from param_decomp_lab.eval_metrics.autointerp_labels import (
    AutointerpLabels,
    AutointerpLabelsConfig,
)


def _make_metric(k: int, seed: int) -> AutointerpLabels:
    cfg = AutointerpLabelsConfig(
        k=k,
        seed=seed,
        activation_threshold=0.1,
        max_examples=20,
        context_tokens_per_side=10,
        llm=OpenRouterLLMConfig(),
        template_strategy=CompactSkepticalConfig(),
    )
    return AutointerpLabels(cfg)


def _ci_shapes(c_per_site: dict[str, int]) -> dict[str, torch.Tensor]:
    return {site: torch.zeros(1, 1, c) for site, c in c_per_site.items()}


def test_selection_count_and_bounds():
    c_per_site = {"layer.0.mlp": 10, "layer.1.mlp": 5, "layer.2.attn": 7}
    selection = _make_metric(k=12, seed=3)._build_selection(_ci_shapes(c_per_site))
    total_selected = sum(len(v) for v in selection.values())
    assert total_selected == 12
    for site, locals_ in selection.items():
        assert locals_ == sorted(locals_)
        assert all(0 <= i < c_per_site[site] for i in locals_)


def test_selection_is_deterministic_in_seed():
    c_per_site = {"a": 8, "b": 8, "c": 8}
    s1 = _make_metric(k=6, seed=42)._build_selection(_ci_shapes(c_per_site))
    s2 = _make_metric(k=6, seed=42)._build_selection(_ci_shapes(c_per_site))
    s3 = _make_metric(k=6, seed=43)._build_selection(_ci_shapes(c_per_site))
    assert s1 == s2
    assert s1 != s3


def test_selection_matches_uniform_concatenated_sampling():
    """The selection must be the flat uniform sample mapped back to (site, local)."""
    c_per_site = {"a": 4, "b": 6, "c": 3}  # sites sorted: a,b,c -> offsets 0,4,10
    k, seed = 7, 11
    selection = _make_metric(k=k, seed=seed)._build_selection(_ci_shapes(c_per_site))

    flat = sorted(random.Random(seed).sample(range(sum(c_per_site.values())), k))
    offsets = {"a": 0, "b": 4, "c": 10}
    expected: dict[str, list[int]] = {}
    for idx in flat:
        site = "a" if idx < 4 else ("b" if idx < 10 else "c")
        expected.setdefault(site, []).append(idx - offsets[site])
    assert selection == {s: sorted(v) for s, v in expected.items()}
