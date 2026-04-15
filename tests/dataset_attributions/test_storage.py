"""Tests for DatasetAttributionStorage."""

import math
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


# ---------------------------------------------------------------------------
# Deterministic normalization and merge tests with hand-computed values
# ---------------------------------------------------------------------------

# Minimal topology: 1 layer with 2 components, vocab=2, d_model=2
_L = "0.up"
_NC = 2
_V = 2
_D = 2
_N_TOKENS = 100


def _deterministic_storage(
    _regular_val: float = 10.0,
    embed_val: float = 6.0,
    ci_sum_val: float = 50.0,
    act_sq_sum_val: float = 400.0,
    embed_count_val: int = 200,
    n_tokens: int = _N_TOKENS,
) -> DatasetAttributionStorage:
    """Storage with uniform known values for hand-computation."""
    return DatasetAttributionStorage(
        regular_attr={},
        regular_attr_abs={},
        embed_attr={_L: torch.full((_NC, _V), embed_val)},
        embed_attr_abs={_L: torch.full((_NC, _V), embed_val * 2)},
        unembed_attr={_L: torch.full((_D, _NC), 3.0)},
        embed_unembed_attr=torch.full((_D, _V), 1.0),
        w_unembed=torch.eye(_D, _V),
        ci_sum={_L: torch.full((_NC,), ci_sum_val)},
        component_act_sq_sum={_L: torch.full((_NC,), act_sq_sum_val)},
        logit_sq_sum=torch.full((_V,), 900.0),
        embed_token_count=torch.full((_V,), embed_count_val, dtype=torch.long),
        ci_threshold=0.0,
        n_tokens_processed=n_tokens,
    )


class TestNormalizationCorrectness:
    """Verify normalization produces exact expected values from known inputs.

    Formula: normalized = raw / source_denom / target_rms
    - source_denom: ci_sum[source] for components, embed_token_count[tok] for embed
    - target_rms: sqrt(act_sq_sum[target] / n_tokens) for components,
                  sqrt(logit_sq_sum[tok] / n_tokens) for output
    """

    def test_embed_to_component_normalization(self):
        s = _deterministic_storage()
        results = s.get_top_sources(f"{_L}:0", k=_V, sign="positive", metric="attr")

        # raw = embed_attr[_L][0, :] = 6.0 for each vocab entry
        # source_denom = embed_count = 200.0
        # target_rms = sqrt(400 / 100) = 2.0
        # normalized = 6.0 / 200.0 / 2.0 = 0.015
        assert len(results) == _V
        for r in results:
            assert r.layer == "embed"
            assert abs(r.value - 0.015) < 1e-6

    def test_embed_to_component_abs_metric(self):
        s = _deterministic_storage()
        results = s.get_top_sources(f"{_L}:0", k=_V, sign="positive", metric="attr_abs")

        # raw = embed_attr_abs[_L][0, :] = 12.0
        # same denoms: 200.0, 2.0
        # normalized = 12.0 / 200.0 / 2.0 = 0.03
        assert len(results) == _V
        for r in results:
            assert abs(r.value - 0.03) < 1e-6

    def test_component_to_output_normalization(self):
        s = _deterministic_storage()
        results = s.get_top_sources("output:0", k=5, sign="positive", metric="attr")

        # unembed_attr[_L] = 3.0 * ones(2, 2), w_unembed = eye(2, 2)
        # For output:0, w = w_unembed[:, 0] = [1, 0]
        # raw per source component = w @ unembed_attr[_L] = [1,0] @ [[3,3],[3,3]] = [3, 3]
        # but actually w @ attr_matrix where attr_matrix is (d_model, n_components):
        # raw = w @ unembed_attr[_L] = [1,0] @ [[3,3],[3,3]] = [3, 3]  shape (n_c,)
        # source_denom = ci_sum = 50.0
        # target_rms = sqrt(900 / 100) = 3.0
        # normalized = 3.0 / 50.0 / 3.0 = 0.02
        component_results = [r for r in results if r.layer == _L]
        assert len(component_results) == _NC
        for r in component_results:
            assert abs(r.value - 0.02) < 1e-6

    def test_embed_to_output_normalization(self):
        s = _deterministic_storage()
        results = s.get_top_sources("output:0", k=10, sign="positive", metric="attr")

        # embed_unembed_attr = 1.0 * ones(2, 2), w = [1, 0]
        # raw per embed token = w @ embed_unembed_attr = [1,0] @ [[1,1],[1,1]] = [1, 1]
        # source_denom = embed_count = 200.0
        # target_rms = 3.0
        # normalized = 1.0 / 200.0 / 3.0 ≈ 0.001667
        embed_results = [r for r in results if r.layer == "embed"]
        assert len(embed_results) == _V
        for r in embed_results:
            assert abs(r.value - 1.0 / 200.0 / 3.0) < 1e-6

    def test_sign_filtering(self):
        """Positive sign excludes negative values, negative sign excludes positive."""
        s = DatasetAttributionStorage(
            regular_attr={},
            regular_attr_abs={},
            embed_attr={_L: torch.tensor([[5.0, -3.0]])},
            embed_attr_abs={_L: torch.tensor([[5.0, -3.0]])},
            unembed_attr={},
            embed_unembed_attr=torch.zeros(_D, _V),
            w_unembed=torch.eye(_D, _V),
            ci_sum={_L: torch.tensor([1.0])},
            component_act_sq_sum={_L: torch.tensor([100.0])},
            logit_sq_sum=torch.ones(_V),
            embed_token_count=torch.ones(_V, dtype=torch.long),
            ci_threshold=0.0,
            n_tokens_processed=100,
        )

        pos = s.get_top_sources(f"{_L}:0", k=10, sign="positive", metric="attr")
        neg = s.get_top_sources(f"{_L}:0", k=10, sign="negative", metric="attr")

        assert all(r.value > 0 for r in pos)
        assert all(r.value < 0 for r in neg)
        assert len(pos) == 1  # only embed:0 is positive
        assert len(neg) == 1  # only embed:1 is negative


class TestMergeNumericCorrectness:
    """Verify merge produces correct normalized values."""

    def test_merge_equals_sum_of_parts(self, tmp_path: Path):
        """Two workers with known values; merged queries should equal manual computation."""
        s1 = _deterministic_storage(
            embed_val=4.0, ci_sum_val=20.0, act_sq_sum_val=100.0, embed_count_val=80, n_tokens=40
        )
        s2 = _deterministic_storage(
            embed_val=8.0, ci_sum_val=30.0, act_sq_sum_val=500.0, embed_count_val=120, n_tokens=60
        )

        p1, p2 = tmp_path / "r0.pt", tmp_path / "r1.pt"
        s1.save(p1)
        s2.save(p2)
        merged = DatasetAttributionStorage.merge([p1, p2])

        assert merged.n_tokens_processed == 100

        # Merged raw: embed_attr[_L][0, 0] = 4.0 + 8.0 = 12.0
        # Merged embed_count[0] = 80 + 120 = 200
        # Merged act_sq_sum[_L][0] = 100 + 500 = 600
        # target_rms = sqrt(600 / 100) = sqrt(6)
        # normalized = 12.0 / 200.0 / sqrt(6)
        expected = 12.0 / 200.0 / math.sqrt(6)

        results = merged.get_top_sources(f"{_L}:0", k=_V, sign="positive", metric="attr")
        assert len(results) == _V
        for r in results:
            assert abs(r.value - expected) < 1e-6

    def test_merge_identity(self, tmp_path: Path):
        """Merging a single file produces identical query results."""
        s = _deterministic_storage()
        path = tmp_path / "single.pt"
        s.save(path)
        merged = DatasetAttributionStorage.merge([path])

        for key in [f"{_L}:0", f"{_L}:1"]:
            orig = s.get_top_sources(key, k=10, sign="positive", metric="attr")
            mrgd = merged.get_top_sources(key, k=10, sign="positive", metric="attr")
            assert len(orig) == len(mrgd)
            for o, m in zip(orig, mrgd, strict=True):
                assert o.component_key == m.component_key
                assert abs(o.value - m.value) < 1e-6
