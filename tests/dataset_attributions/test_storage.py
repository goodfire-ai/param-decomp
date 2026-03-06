"""Tests for DatasetAttributionStorage."""

from pathlib import Path

import torch
from torch import Tensor

from spd.dataset_attributions.storage import DatasetAttributionStorage

VOCAB_SIZE = 4
D_MODEL = 4
LAYER_0 = "0.glu.up"
LAYER_1 = "1.glu.up"
C0 = 3  # components in layer 0
C1 = 2  # components in layer 1


def _make_storage(seed: int = 0, n_tokens: int = 640) -> DatasetAttributionStorage:
    """Build storage for test topology.

    Sources by target:
        "0.glu.up": ["embed"]             -> embed edge (C0, VOCAB_SIZE)
        "1.glu.up": ["embed", "0.glu.up"] -> embed edge (C1, VOCAB_SIZE) + regular (C1, C0)
        "output":   ["0.glu.up", "1.glu.up"] -> unembed (D_MODEL, C0), (D_MODEL, C1)
        "output":   ["embed"]             -> embed_unembed (D_MODEL, VOCAB_SIZE)
    """
    g = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=g)

    return DatasetAttributionStorage(
        regular_attr={LAYER_1: {LAYER_0: rand(C1, C0)}},
        regular_attr_abs={LAYER_1: {LAYER_0: rand(C1, C0)}},
        embed_attr={LAYER_0: rand(C0, VOCAB_SIZE), LAYER_1: rand(C1, VOCAB_SIZE)},
        embed_attr_abs={LAYER_0: rand(C0, VOCAB_SIZE), LAYER_1: rand(C1, VOCAB_SIZE)},
        unembed_attr={LAYER_0: rand(D_MODEL, C0), LAYER_1: rand(D_MODEL, C1)},
        embed_unembed_attr=rand(D_MODEL, VOCAB_SIZE),
        w_unembed=rand(D_MODEL, VOCAB_SIZE),
        ci_sum={LAYER_0: rand(C0).abs() + 1.0, LAYER_1: rand(C1).abs() + 1.0},
        component_act_sq_sum={LAYER_0: rand(C0).abs() + 1.0, LAYER_1: rand(C1).abs() + 1.0},
        logit_sq_sum=rand(VOCAB_SIZE).abs() + 1.0,
        embed_token_count=torch.randint(100, 1000, (VOCAB_SIZE,), generator=g),
        ci_threshold=1e-6,
        n_tokens_processed=n_tokens,
    )


class TestNComponents:
    def test_counts_all_target_layers(self):
        storage = _make_storage()
        assert storage.n_components == C0 + C1


class TestGetTopSources:
    def test_component_target_returns_entries(self):
        storage = _make_storage()
        results = storage.get_top_sources(f"{LAYER_1}:0", k=5, sign="positive", metric="attr")
        assert all(r.value > 0 for r in results)
        assert len(results) <= 5

    def test_component_target_includes_embed(self):
        storage = _make_storage()
        results = storage.get_top_sources(f"{LAYER_1}:0", k=20, sign="positive", metric="attr")
        layers = {r.layer for r in results}
        assert "embed" in layers or LAYER_0 in layers

    def test_output_target(self):
        storage = _make_storage()
        results = storage.get_top_sources("output:0", k=5, sign="positive", metric="attr")
        assert len(results) <= 5

    def test_output_target_attr_abs_returns_empty(self):
        storage = _make_storage()
        results = storage.get_top_sources("output:0", k=5, sign="positive", metric="attr_abs")
        assert results == []

    def test_target_only_in_embed_attr(self):
        storage = _make_storage()
        results = storage.get_top_sources(f"{LAYER_0}:0", k=5, sign="positive", metric="attr")
        assert len(results) <= 5
        assert all(r.layer == "embed" for r in results)

    def test_attr_abs_metric(self):
        storage = _make_storage()
        results = storage.get_top_sources(f"{LAYER_1}:0", k=5, sign="positive", metric="attr_abs")
        assert len(results) <= 5

    def test_no_nan_in_results(self):
        storage = _make_storage()
        results = storage.get_top_sources(f"{LAYER_1}:0", k=20, sign="positive", metric="attr")
        assert all(not torch.isnan(torch.tensor(r.value)) for r in results)


class TestGetTopTargets:
    def test_component_source(self):
        storage = _make_storage()
        results = storage.get_top_targets(
            f"{LAYER_0}:0", k=5, sign="positive", metric="attr", include_outputs=False
        )
        assert len(results) <= 5
        assert all(r.value > 0 for r in results)

    def test_embed_source(self):
        storage = _make_storage()
        results = storage.get_top_targets(
            "embed:0", k=5, sign="positive", metric="attr", include_outputs=False
        )
        assert len(results) <= 5

    def test_include_outputs(self):
        storage = _make_storage()
        results = storage.get_top_targets(f"{LAYER_0}:0", k=20, sign="positive", metric="attr")
        assert len(results) > 0

    def test_embed_source_with_outputs(self):
        storage = _make_storage()
        results = storage.get_top_targets("embed:0", k=20, sign="positive", metric="attr")
        assert len(results) > 0

    def test_attr_abs_skips_output_targets(self):
        storage = _make_storage()
        results = storage.get_top_targets(f"{LAYER_0}:0", k=20, sign="positive", metric="attr_abs")
        assert all(r.layer != "output" for r in results)


class TestSaveLoad:
    def test_roundtrip(self, tmp_path: Path):
        original = _make_storage()
        path = tmp_path / "attrs.pt"
        original.save(path)

        loaded = DatasetAttributionStorage.load(path)

        assert loaded.ci_threshold == original.ci_threshold
        assert loaded.n_tokens_processed == original.n_tokens_processed
        assert loaded.n_components == original.n_components

    def test_roundtrip_query_consistency(self, tmp_path: Path):
        original = _make_storage()
        path = tmp_path / "attrs.pt"
        original.save(path)
        loaded = DatasetAttributionStorage.load(path)

        orig_results = original.get_top_sources(f"{LAYER_1}:0", k=5, sign="positive", metric="attr")
        load_results = loaded.get_top_sources(f"{LAYER_1}:0", k=5, sign="positive", metric="attr")

        assert len(orig_results) == len(load_results)
        for orig, loaded in zip(orig_results, load_results, strict=True):
            assert orig.component_key == loaded.component_key
            assert abs(orig.value - loaded.value) < 1e-5


class TestMerge:
    def test_two_workers_additive(self, tmp_path: Path):
        s1 = _make_storage(seed=0, n_tokens=320)
        s2 = _make_storage(seed=42, n_tokens=320)

        p1 = tmp_path / "rank_0.pt"
        p2 = tmp_path / "rank_1.pt"
        s1.save(p1)
        s2.save(p2)

        merged = DatasetAttributionStorage.merge([p1, p2])

        assert merged.n_tokens_processed == 640

    def test_single_file(self, tmp_path: Path):
        original = _make_storage(seed=7, n_tokens=640)
        path = tmp_path / "rank_0.pt"
        original.save(path)

        merged = DatasetAttributionStorage.merge([path])

        assert merged.n_tokens_processed == original.n_tokens_processed

        orig_results = original.get_top_sources(f"{LAYER_1}:0", k=5, sign="positive", metric="attr")
        merge_results = merged.get_top_sources(f"{LAYER_1}:0", k=5, sign="positive", metric="attr")
        for o, m in zip(orig_results, merge_results, strict=True):
            assert o.component_key == m.component_key
            assert abs(o.value - m.value) < 1e-5
