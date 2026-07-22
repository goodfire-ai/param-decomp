"""Tests for the Harvester class and extract_padding_firing_windows."""

from pathlib import Path

import numpy as np
import pytest

from param_decomp_lab.harvest.accumulator import Harvester, extract_padding_firing_windows
from param_decomp_lab.harvest.reservoir import WINDOW_PAD_SENTINEL, ActivationWindows

LAYERS = [("layer_0", 4), ("layer_1", 4)]
N_TOTAL = 8
VOCAB_SIZE = 10
MAX_EXAMPLES = 5
CONTEXT_TOKENS_PER_SIDE = 1
WINDOW = 2 * CONTEXT_TOKENS_PER_SIDE + 1  # 3

ACT_TYPES = ["ci", "inner"]


def _make_harvester(collect_component_cooccurrence: bool = True) -> Harvester:
    return Harvester(
        layers=LAYERS,
        vocab_size=VOCAB_SIZE,
        max_examples_per_component=MAX_EXAMPLES,
        context_tokens_per_side=CONTEXT_TOKENS_PER_SIDE,
        max_examples_per_batch_per_component=5,
        collect_component_cooccurrence=collect_component_cooccurrence,
    )


def _cooc(h: Harvester) -> np.ndarray:
    assert h.cooccurrence_counts is not None
    return h.cooccurrence_counts


def _make_activation_windows(
    comp_idx: list[int],
    token_windows: np.ndarray,
    firings: np.ndarray | None = None,
) -> ActivationWindows:
    n = len(comp_idx)
    w = token_windows.shape[1]
    if firings is None:
        firings = np.ones((n, w), dtype=np.bool_)
    return ActivationWindows(
        component_idx=np.array(comp_idx),
        token_windows=token_windows,
        firing_windows=firings,
        activation_windows={at: np.ones((n, w)) for at in ACT_TYPES},
    )


class TestInit:
    def test_array_shapes(self):
        h = _make_harvester()
        assert h.firing_counts.shape == (N_TOTAL,)
        assert _cooc(h).shape == (N_TOTAL, N_TOTAL)
        assert h.input_cooccurrence.shape == (N_TOTAL, VOCAB_SIZE)
        assert h.input_marginals.shape == (VOCAB_SIZE,)
        assert h.output_cooccurrence.shape == (N_TOTAL, VOCAB_SIZE)
        assert h.output_marginals.shape == (VOCAB_SIZE,)
        assert h.reservoir.tokens.shape == (N_TOTAL, MAX_EXAMPLES, WINDOW)
        assert h.reservoir.n_items.shape == (N_TOTAL,)
        assert h.reservoir.n_seen.shape == (N_TOTAL,)

    def test_layer_offsets(self):
        h = _make_harvester()
        assert h.layer_offsets == {"layer_0": 0, "layer_1": 4}

    def test_component_keys(self):
        h = _make_harvester()
        expected = [f"layer_0:{i}" for i in range(4)] + [f"layer_1:{i}" for i in range(4)]
        assert h.component_keys == expected

    def test_arrays_initialized_to_zero(self):
        h = _make_harvester()
        assert h.firing_counts.sum() == 0
        assert _cooc(h).sum() == 0
        assert h.reservoir.n_items.sum() == 0
        assert h.reservoir.n_seen.sum() == 0
        assert h.total_tokens_processed == 0

    def test_reservoir_tokens_initialized_to_sentinel(self):
        h = _make_harvester()
        assert (h.reservoir.tokens == WINDOW_PAD_SENTINEL).all()


class TestReservoirAdd:
    def test_fills_up_to_k(self):
        h = _make_harvester()
        k = h.reservoir.k
        comp = 2

        for i in range(k):
            aw = _make_activation_windows([comp], np.full((1, WINDOW), i, dtype=np.int64))
            h.reservoir.add(aw, h.rng)

        assert h.reservoir.n_items[comp] == k
        assert h.reservoir.n_seen[comp] == k
        for i in range(k):
            assert int(h.reservoir.tokens[comp, i, 0]) == i

    def test_replacement_after_k(self):
        h = _make_harvester()
        k = h.reservoir.k
        comp = 0

        n_extra = 100
        for i in range(k + n_extra):
            aw = _make_activation_windows([comp], np.full((1, WINDOW), i, dtype=np.int64))
            h.reservoir.add(aw, h.rng)

        assert h.reservoir.n_items[comp] == k
        assert h.reservoir.n_seen[comp] == k + n_extra

    def test_n_items_never_exceeds_k(self):
        h = _make_harvester()
        k = h.reservoir.k
        comp = 1

        for i in range(k * 10):
            aw = _make_activation_windows(
                [comp], np.full((1, WINDOW), i % VOCAB_SIZE, dtype=np.int64)
            )
            h.reservoir.add(aw, h.rng)

        assert h.reservoir.n_items[comp] == k
        assert h.reservoir.n_seen[comp] == k * 10

    def test_multiple_components_in_one_call(self):
        h = _make_harvester()
        aw = _make_activation_windows([0, 0, 3, 3, 3], np.arange(5 * WINDOW).reshape(5, WINDOW))
        h.reservoir.add(aw, h.rng)

        assert h.reservoir.n_items[0] == 2
        assert h.reservoir.n_seen[0] == 2
        assert h.reservoir.n_items[3] == 3
        assert h.reservoir.n_seen[3] == 3
        assert h.reservoir.n_items[1] == 0
        assert h.reservoir.n_items[2] == 0

    def test_independent_component_tracking(self):
        h = _make_harvester()
        k = h.reservoir.k

        for i in range(k):
            aw = _make_activation_windows([0], np.full((1, WINDOW), i, dtype=np.int64))
            h.reservoir.add(aw, h.rng)

        aw = _make_activation_windows([1], np.full((1, WINDOW), 99, dtype=np.int64))
        h.reservoir.add(aw, h.rng)

        assert h.reservoir.n_items[0] == k
        assert h.reservoir.n_seen[0] == k
        assert h.reservoir.n_items[1] == 1
        assert h.reservoir.n_seen[1] == 1


class TestSaveLoadRoundtrip:
    def test_roundtrip_preserves_all_fields(self, tmp_path: Path):
        h = _make_harvester()

        h.firing_counts[0] = 10
        h.firing_counts[3] = 5
        h.activation_sums["ci"][0] = 2.5
        _cooc(h)[0, 3] = 7
        h.input_cooccurrence[0, 2] = 15
        h.input_marginals[2] = 100
        h.output_cooccurrence[0, 5] = 0.3
        h.output_marginals[5] = 1.0
        h.total_tokens_processed = 500

        aw = _make_activation_windows([0], np.array([[1, 2, 3]]))
        h.reservoir.add(aw, h.rng)

        path = tmp_path / "harvester.npz"
        h.save(path)
        loaded = Harvester.load(path)

        assert loaded.layer_names == h.layer_names
        assert loaded.c_per_layer == h.c_per_layer
        assert loaded.vocab_size == h.vocab_size
        assert loaded.max_examples_per_component == h.max_examples_per_component
        assert loaded.context_tokens_per_side == h.context_tokens_per_side
        assert loaded.total_tokens_processed == h.total_tokens_processed
        assert loaded.layer_offsets == h.layer_offsets

        for field in [
            "firing_counts",
            "cooccurrence_counts",
            "input_cooccurrence",
            "input_marginals",
            "output_cooccurrence",
            "output_marginals",
        ]:
            np.testing.assert_array_equal(getattr(loaded, field), getattr(h, field))

        for act_type in h.activation_sums:
            np.testing.assert_array_equal(
                loaded.activation_sums[act_type], h.activation_sums[act_type]
            )

        np.testing.assert_array_equal(loaded.reservoir.tokens, h.reservoir.tokens)
        np.testing.assert_array_equal(loaded.reservoir.n_items, h.reservoir.n_items)
        np.testing.assert_array_equal(loaded.reservoir.n_seen, h.reservoir.n_seen)


class TestMerge:
    def test_accumulators_sum(self):
        h1 = _make_harvester()
        h2 = _make_harvester()

        h1.firing_counts[0] = 10
        h2.firing_counts[0] = 20
        h1.activation_sums["ci"][1] = 3.0
        h2.activation_sums["ci"][1] = 7.0
        _cooc(h1)[0, 1] = 5
        _cooc(h2)[0, 1] = 3
        h1.input_cooccurrence[0, 2] = 10
        h2.input_cooccurrence[0, 2] = 5
        h1.input_marginals[2] = 100
        h2.input_marginals[2] = 200
        h1.output_cooccurrence[0, 0] = 0.5
        h2.output_cooccurrence[0, 0] = 0.3
        h1.output_marginals[0] = 1.0
        h2.output_marginals[0] = 2.0
        h1.total_tokens_processed = 100
        h2.total_tokens_processed = 200

        h1.merge(h2)

        assert h1.firing_counts[0] == 30
        assert h1.activation_sums["ci"][1] == 10.0
        assert _cooc(h1)[0, 1] == 8
        assert h1.input_cooccurrence[0, 2] == 15
        assert h1.input_marginals[2] == 300
        assert h1.output_cooccurrence[0, 0] == pytest.approx(0.8)
        assert h1.output_marginals[0] == 3.0
        assert h1.total_tokens_processed == 300

    def test_merge_asserts_matching_structure(self):
        h1 = _make_harvester()
        h_different = Harvester(
            layers=[("other", 4)],
            vocab_size=VOCAB_SIZE,
            max_examples_per_component=MAX_EXAMPLES,
            context_tokens_per_side=CONTEXT_TOKENS_PER_SIDE,
            max_examples_per_batch_per_component=5,
            collect_component_cooccurrence=True,
        )
        with pytest.raises(AssertionError):
            h1.merge(h_different)

    def test_merge_reservoir_both_underfilled(self):
        h1 = _make_harvester()
        h2 = _make_harvester()

        for i in range(2):
            aw = _make_activation_windows([0], np.full((1, WINDOW), i, dtype=np.int64))
            h1.reservoir.add(aw, h1.rng)
        for i in range(2):
            aw = _make_activation_windows([0], np.full((1, WINDOW), 10 + i, dtype=np.int64))
            h2.reservoir.add(aw, h2.rng)

        h1.merge(h2)
        assert h1.reservoir.n_items[0] == 4
        assert h1.reservoir.n_seen[0] == 4

    def test_merge_reservoir_n_seen_sums(self):
        h1 = _make_harvester()
        h2 = _make_harvester()
        k = MAX_EXAMPLES

        for i in range(k + 10):
            aw = _make_activation_windows([0], np.full((1, WINDOW), i % VOCAB_SIZE, dtype=np.int64))
            h1.reservoir.add(aw, h1.rng)
        for i in range(k + 5):
            aw = _make_activation_windows([0], np.full((1, WINDOW), i % VOCAB_SIZE, dtype=np.int64))
            h2.reservoir.add(aw, h2.rng)

        seen_before = int(h1.reservoir.n_seen[0]) + int(h2.reservoir.n_seen[0])
        h1.merge(h2)

        assert h1.reservoir.n_items[0] == k
        assert h1.reservoir.n_seen[0] == seen_before

    def test_merge_preserves_other_components(self):
        h1 = _make_harvester()
        h2 = _make_harvester()

        aw1 = _make_activation_windows([0], np.full((1, WINDOW), 1, dtype=np.int64))
        h1.reservoir.add(aw1, h1.rng)
        aw2 = _make_activation_windows([3], np.full((1, WINDOW), 2, dtype=np.int64))
        h2.reservoir.add(aw2, h2.rng)

        h1.merge(h2)
        assert h1.reservoir.n_items[0] == 1
        assert h1.reservoir.n_items[3] == 1


class TestBuildResults:
    def _make_harvester_with_firings(self) -> Harvester:
        h = _make_harvester()

        h.total_tokens_processed = 100
        h.firing_counts[0] = 10
        h.firing_counts[1] = 5
        h.activation_sums["ci"][0] = 2.0
        h.activation_sums["ci"][1] = 1.0

        h.input_cooccurrence[0, 0] = 8
        h.input_cooccurrence[1, 1] = 3
        h.input_marginals[0] = 50
        h.input_marginals[1] = 30
        h.output_cooccurrence[0, 0] = 5.0
        h.output_cooccurrence[1, 1] = 2.0
        h.output_marginals[0] = 20.0
        h.output_marginals[1] = 15.0

        for i in range(3):
            aw = _make_activation_windows([0], np.array([[i, i + 1, i + 2]]))
            h.reservoir.add(aw, h.rng)

        aw = _make_activation_windows([1], np.array([[5, 6, 7]]))
        h.reservoir.add(aw, h.rng)

        return h

    def test_yields_only_firing_components(self):
        h = self._make_harvester_with_firings()
        results = list(h.build_results(pmi_top_k_tokens=3))

        keys = {r.component_key for r in results}
        assert keys == {"layer_0:0", "layer_0:1"}

    def test_skips_zero_firing_components(self):
        h = self._make_harvester_with_firings()
        results = list(h.build_results(pmi_top_k_tokens=3))

        keys = {r.component_key for r in results}
        for cidx in range(2, 4):
            assert f"layer_0:{cidx}" not in keys
        for cidx in range(4):
            assert f"layer_1:{cidx}" not in keys

    def test_component_data_structure(self):
        h = self._make_harvester_with_firings()
        results = list(h.build_results(pmi_top_k_tokens=3))

        comp0 = next(r for r in results if r.component_key == "layer_0:0")
        assert comp0.layer == "layer_0"
        assert comp0.component_idx == 0
        assert comp0.firing_density == pytest.approx(10.0 / 100)
        assert comp0.mean_activations["ci"] == pytest.approx(2.0 / 100)
        assert len(comp0.activation_examples) == 3
        assert comp0.input_token_pmi is not None
        assert comp0.output_token_pmi is not None

    def test_activation_examples_have_correct_data(self):
        h = self._make_harvester_with_firings()
        results = list(h.build_results(pmi_top_k_tokens=3))

        comp0 = next(r for r in results if r.component_key == "layer_0:0")
        ex = comp0.activation_examples[0]
        assert len(ex.token_ids) > 0
        assert len(ex.firings) == len(ex.token_ids)
        for act_type in ex.activations:
            assert len(ex.activations[act_type]) == len(ex.token_ids)

    def test_second_layer_component_keys(self):
        h = _make_harvester()
        h.total_tokens_processed = 100
        h.firing_counts[5] = 8
        h.activation_sums["ci"][5] = 1.6
        h.input_marginals[0] = 50
        h.input_cooccurrence[5, 0] = 4
        h.output_marginals[0] = 10.0
        h.output_cooccurrence[5, 0] = 2.0

        aw = _make_activation_windows([5], np.array([[1, 2, 3]]))
        h.reservoir.add(aw, h.rng)

        results = list(h.build_results(pmi_top_k_tokens=3))
        assert len(results) == 1
        assert results[0].component_key == "layer_1:1"
        assert results[0].layer == "layer_1"
        assert results[0].component_idx == 1

    def test_no_results_when_nothing_fires(self):
        h = _make_harvester()
        h.total_tokens_processed = 100
        results = list(h.build_results(pmi_top_k_tokens=3))
        assert results == []

    def test_sentinel_tokens_stripped_from_examples(self):
        h = _make_harvester()
        h.total_tokens_processed = 100
        h.firing_counts[0] = 5
        h.activation_sums["ci"][0] = 1.0
        h.input_marginals[0] = 50
        h.input_cooccurrence[0, 0] = 3
        h.output_marginals[0] = 10.0
        h.output_cooccurrence[0, 0] = 1.0

        h.reservoir.tokens[0, 0] = np.array([WINDOW_PAD_SENTINEL, 5, 6])
        h.reservoir.firings[0, 0] = np.array([False, True, True])
        for at in h.reservoir.acts:
            h.reservoir.acts[at][0, 0] = np.array([0.0, 0.8, 0.9])
        h.reservoir.n_items[0] = 1
        h.reservoir.n_seen[0] = 1

        results = list(h.build_results(pmi_top_k_tokens=3))
        assert len(results) == 1
        ex = results[0].activation_examples[0]
        assert WINDOW_PAD_SENTINEL not in ex.token_ids
        assert len(ex.token_ids) == 2


class TestProcessBatch:
    def _make_batch_inputs(
        self, B: int = 2, S: int = 4
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], np.ndarray]:
        rng = np.random.default_rng(0)
        batch = rng.integers(0, VOCAB_SIZE, (B, S))
        firings = {layer: np.zeros((B, S, c), dtype=np.bool_) for layer, c in LAYERS}
        activations = {layer: {at: np.zeros((B, S, c)) for at in ACT_TYPES} for layer, c in LAYERS}
        output_probs = np.zeros((B, S, VOCAB_SIZE))
        return batch, firings, activations, output_probs

    def test_updates_total_tokens(self):
        h = _make_harvester()
        B, S = 2, 4
        batch, firings, activations, output_probs = self._make_batch_inputs(B, S)

        h.process_batch(batch, firings, activations, output_probs)
        assert h.total_tokens_processed == B * S

    def test_firing_counts_accumulate(self):
        h = _make_harvester()
        B, S = 1, 2
        batch, firings, activations, output_probs = self._make_batch_inputs(B, S)
        firings["layer_0"][0, 0, 0] = True
        firings["layer_0"][0, 1, 0] = True

        h.process_batch(batch, firings, activations, output_probs)
        assert h.firing_counts[0] == 2
        assert h.firing_counts[1] == 0

    def test_activation_sums_accumulate(self):
        h = _make_harvester()
        B, S = 1, 1
        batch, firings, activations, output_probs = self._make_batch_inputs(B, S)
        activations["layer_0"]["ci"][0, 0, 2] = 0.75

        h.process_batch(batch, firings, activations, output_probs)
        assert float(h.activation_sums["ci"][2]) == pytest.approx(0.75)

    def test_cooccurrence_counts(self):
        h = _make_harvester()
        B, S = 1, 1
        batch, firings, activations, output_probs = self._make_batch_inputs(B, S)
        firings["layer_0"][0, 0, 0] = True
        firings["layer_0"][0, 0, 2] = True

        h.process_batch(batch, firings, activations, output_probs)
        assert _cooc(h)[0, 2] == 1
        assert _cooc(h)[2, 0] == 1
        assert _cooc(h)[0, 0] == 1
        assert _cooc(h)[2, 2] == 1

    def test_cooccurrence_disabled(self, tmp_path: Path):
        h = _make_harvester(collect_component_cooccurrence=False)
        assert h.cooccurrence_counts is None

        B, S = 1, 1
        batch, firings, activations, output_probs = self._make_batch_inputs(B, S)
        firings["layer_0"][0, 0, 0] = True
        firings["layer_0"][0, 0, 2] = True
        h.process_batch(batch, firings, activations, output_probs)
        assert h.cooccurrence_counts is None

        path = tmp_path / "harvester.npz"
        h.save(path)
        loaded = Harvester.load(path)
        assert loaded.cooccurrence_counts is None
        assert loaded.collect_component_cooccurrence is False


class TestExtractPaddingFiringWindows:
    def test_center_window(self):
        rng = np.random.default_rng(0)
        batch = np.array([[10, 11, 12, 13, 14]])
        firings = np.zeros((1, 5, 2), dtype=np.bool_)
        firings[0, 2, 0] = True
        activations = {"ci": np.zeros((1, 5, 2))}
        activations["ci"][0, 2, 0] = 0.9

        result = extract_padding_firing_windows(batch, firings, activations, 10, 1, rng)
        assert result is not None
        assert result.token_windows.shape == (1, 3)
        assert result.token_windows[0].tolist() == [11, 12, 13]
        assert float(result.activation_windows["ci"][0, 1]) == pytest.approx(0.9)

    def test_left_boundary_padding(self):
        rng = np.random.default_rng(0)
        batch = np.array([[10, 11, 12]])
        firings = np.zeros((1, 3, 1), dtype=np.bool_)
        firings[0, 0, 0] = True
        activations = {"ci": np.zeros((1, 3, 1))}

        result = extract_padding_firing_windows(batch, firings, activations, 10, 2, rng)
        assert result is not None
        tok_w = result.token_windows
        assert tok_w.shape == (1, 5)
        assert tok_w[0, 0] == WINDOW_PAD_SENTINEL
        assert tok_w[0, 1] == WINDOW_PAD_SENTINEL
        assert tok_w[0, 2] == 10
        assert tok_w[0, 3] == 11
        assert tok_w[0, 4] == 12

    def test_right_boundary_padding(self):
        rng = np.random.default_rng(0)
        batch = np.array([[10, 11, 12]])
        firings = np.zeros((1, 3, 1), dtype=np.bool_)
        firings[0, 2, 0] = True
        activations = {"ci": np.zeros((1, 3, 1))}

        result = extract_padding_firing_windows(batch, firings, activations, 10, 2, rng)
        assert result is not None
        tok_w = result.token_windows
        assert tok_w[0, 0] == 10
        assert tok_w[0, 1] == 11
        assert tok_w[0, 2] == 12
        assert tok_w[0, 3] == WINDOW_PAD_SENTINEL
        assert tok_w[0, 4] == WINDOW_PAD_SENTINEL

    def test_no_firings_returns_none(self):
        rng = np.random.default_rng(0)
        batch = np.array([[0, 1, 2]])
        firings = np.zeros((1, 3, 2), dtype=np.bool_)
        activations = {"ci": np.zeros((1, 3, 2))}

        result = extract_padding_firing_windows(batch, firings, activations, 10, 1, rng)
        assert result is None

    def test_multiple_firings(self):
        rng = np.random.default_rng(0)
        batch = np.array([[0, 1, 2, 3, 4]])
        firings = np.zeros((1, 5, 3), dtype=np.bool_)
        firings[0, 1, 0] = True
        firings[0, 3, 2] = True
        activations = {"ci": np.zeros((1, 5, 3))}

        result = extract_padding_firing_windows(batch, firings, activations, 10, 1, rng)
        assert result is not None
        assert result.token_windows.shape == (2, 3)
        windows = sorted(result.token_windows.tolist())
        assert windows == [[0, 1, 2], [2, 3, 4]]
