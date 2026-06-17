"""Tests for ActivationExamplesReservoir."""

import numpy as np
import pytest

from param_decomp_lab.harvest.reservoir import (
    WINDOW_PAD_SENTINEL,
    ActivationExamplesReservoir,
    ActivationWindows,
)

N_COMPONENTS = 4
K = 3
WINDOW = 3

ACT_TYPES = ["ci", "inner"]


def _make_reservoir() -> ActivationExamplesReservoir:
    return ActivationExamplesReservoir.create(N_COMPONENTS, K, WINDOW)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_activation_window(
    comp: list[int],
    tokens: np.ndarray,
    firings: np.ndarray | None = None,
) -> ActivationWindows:
    n = len(comp)
    w = tokens.shape[1]
    if firings is None:
        firings = np.ones((n, w), dtype=np.bool_)
    return ActivationWindows(
        component_idx=np.array(comp),
        token_windows=tokens,
        firing_windows=firings,
        activation_windows={at: np.ones((n, w)) * 0.5 for at in ACT_TYPES},
    )


class TestAdd:
    def test_fills_up_to_k(self):
        r = _make_reservoir()
        rng = _rng()
        comp = 1

        for i in range(K):
            r.add(_make_activation_window([comp], np.full((1, WINDOW), i, dtype=np.int64)), rng)

        assert r.n_items[comp] == K
        assert r.n_seen[comp] == K
        for i in range(K):
            assert int(r.tokens[comp, i, 0]) == i

    def test_replacement_after_k(self):
        r = _make_reservoir()
        rng = _rng(42)
        comp = 0

        n_total = K + 50
        for i in range(n_total):
            r.add(_make_activation_window([comp], np.full((1, WINDOW), i, dtype=np.int64)), rng)

        assert r.n_items[comp] == K
        assert r.n_seen[comp] == n_total

    def test_written_data_matches_input(self):
        r = _make_reservoir()
        rng = _rng()
        tokens = np.array([[7, 8, 9]])
        firings = np.array([[True, False, True]])
        aw = ActivationWindows(
            component_idx=np.array([2]),
            token_windows=tokens,
            firing_windows=firings,
            activation_windows={"ci": np.array([[0.1, 0.2, 0.3]])},
        )
        r.add(aw, rng)

        np.testing.assert_array_equal(r.tokens[2, 0], tokens[0])
        np.testing.assert_array_equal(r.firings[2, 0], firings[0])
        np.testing.assert_allclose(r.acts["ci"][2, 0], np.array([0.1, 0.2, 0.3]))


class TestMerge:
    def test_merge_combines_underfilled(self):
        r1 = _make_reservoir()
        r2 = _make_reservoir()
        rng = _rng()

        r1.add(_make_activation_window([0], np.full((1, WINDOW), 1, dtype=np.int64)), rng)
        r2.add(_make_activation_window([0], np.full((1, WINDOW), 2, dtype=np.int64)), rng)

        r1.merge(r2, rng)
        assert r1.n_items[0] == 2
        assert r1.n_seen[0] == 2

    def test_merge_weighted_by_n_seen(self):
        rng = _rng(0)

        n_trials = 200
        heavy_wins = 0
        for _ in range(n_trials):
            r_heavy = _make_reservoir()
            r_light = _make_reservoir()

            for _ in range(K):
                r_heavy.add(
                    _make_activation_window([0], np.full((1, WINDOW), 1, dtype=np.int64)), rng
                )
            r_heavy.n_seen[0] = 1000

            for _ in range(K):
                r_light.add(
                    _make_activation_window([0], np.full((1, WINDOW), 2, dtype=np.int64)), rng
                )
            r_light.n_seen[0] = 1

            r_heavy.merge(r_light, rng)
            from_heavy = int((r_heavy.tokens[0, :, 0] == 1).sum())
            if from_heavy == K:
                heavy_wins += 1

        assert heavy_wins > n_trials * 0.8

    def test_merge_n_seen_sums(self):
        r1 = _make_reservoir()
        r2 = _make_reservoir()
        rng = _rng()

        for i in range(K + 5):
            r1.add(_make_activation_window([0], np.full((1, WINDOW), i % 10, dtype=np.int64)), rng)
        for i in range(K + 3):
            r2.add(_make_activation_window([0], np.full((1, WINDOW), i % 10, dtype=np.int64)), rng)

        total = int(r1.n_seen[0]) + int(r2.n_seen[0])
        r1.merge(r2, rng)
        assert r1.n_seen[0] == total
        assert r1.n_items[0] == K


class TestExamples:
    def test_yields_correct_items(self):
        r = _make_reservoir()
        rng = _rng()
        for i in range(2):
            aw = ActivationWindows(
                component_idx=np.array([0]),
                token_windows=np.full((1, WINDOW), i + 10, dtype=np.int64),
                firing_windows=np.ones((1, WINDOW), dtype=np.bool_),
                activation_windows={"ci": np.ones((1, WINDOW)) * (i + 1) * 0.1},
            )
            r.add(aw, rng)

        examples = list(r.examples(0))
        assert len(examples) == 2
        ex0 = examples[0]
        assert ex0.token_ids == [10, 10, 10]
        assert all(ex0.firings)
        assert ex0.activations["ci"] == [pytest.approx(0.1)] * 3

    def test_filters_sentinels(self):
        r = _make_reservoir()
        r.tokens[0, 0] = np.array([WINDOW_PAD_SENTINEL, 5, 6])
        r.firings[0, 0] = np.array([False, True, True])
        r.acts["ci"] = np.zeros((N_COMPONENTS, K, WINDOW))
        r.acts["ci"][0, 0] = np.array([0.0, 0.8, 0.9])
        r.n_items[0] = 1
        r.n_seen[0] = 1

        examples = list(r.examples(0))
        assert len(examples) == 1
        ex = examples[0]
        assert ex.token_ids == [5, 6]
        assert ex.firings == [True, True]
        assert ex.activations["ci"] == [pytest.approx(0.8), pytest.approx(0.9)]

    def test_empty_component_yields_nothing(self):
        r = _make_reservoir()
        assert list(r.examples(0)) == []


class TestStateDictRoundtrip:
    def test_roundtrip_preserves_data(self):
        r = _make_reservoir()
        rng = _rng()
        for i in range(2):
            aw = ActivationWindows(
                component_idx=np.array([1]),
                token_windows=np.full((1, WINDOW), i + 5, dtype=np.int64),
                firing_windows=np.ones((1, WINDOW), dtype=np.bool_),
                activation_windows={"ci": np.ones((1, WINDOW)) * 0.5},
            )
            r.add(aw, rng)

        sd = r.state_dict()
        restored = ActivationExamplesReservoir.from_state_dict(sd)

        assert restored.k == r.k
        assert restored.window == r.window
        np.testing.assert_array_equal(restored.tokens, r.tokens)
        np.testing.assert_array_equal(restored.firings, r.firings)
        for at in r.acts:
            np.testing.assert_array_equal(restored.acts[at], r.acts[at])
        np.testing.assert_array_equal(restored.n_items, r.n_items)
        np.testing.assert_array_equal(restored.n_seen, r.n_seen)
