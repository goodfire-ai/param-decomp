"""Tests for ActivationExamplesReservoir."""

import random

import pytest
import torch

from spd.harvest.reservoir import (
    WINDOW_PAD_SENTINEL,
    ActivationExamplesReservoir,
    ActivationWindows,
)

DEVICE = torch.device("cpu")
N_COMPONENTS = 4
K = 3
WINDOW = 3

ACT_TYPES = ["ci", "inner"]


def _make_reservoir() -> ActivationExamplesReservoir:
    return ActivationExamplesReservoir.create(N_COMPONENTS, K, WINDOW, DEVICE)


def _make_activation_window(
    comp: list[int],
    tokens: torch.Tensor,
    firings: torch.Tensor | None = None,
) -> ActivationWindows:
    n = len(comp)
    w = tokens.shape[1]
    if firings is None:
        firings = torch.ones(n, w, dtype=torch.bool)
    return ActivationWindows(
        component_idx=torch.tensor(comp),
        token_windows=tokens,
        firing_windows=firings,
        activation_windows={at: torch.ones(n, w) * 0.5 for at in ACT_TYPES},
    )


class TestAdd:
    def test_fills_up_to_k(self):
        r = _make_reservoir()
        comp = 1

        for i in range(K):
            r.add(_make_activation_window([comp], torch.full((1, WINDOW), i, dtype=torch.long)))

        assert r.n_items[comp] == K
        assert r.n_seen[comp] == K
        for i in range(K):
            assert r.tokens[comp, i, 0].item() == i

    def test_replacement_after_k(self):
        r = _make_reservoir()
        comp = 0
        random.seed(42)

        n_total = K + 50
        for i in range(n_total):
            r.add(_make_activation_window([comp], torch.full((1, WINDOW), i, dtype=torch.long)))

        assert r.n_items[comp] == K
        assert r.n_seen[comp] == n_total

    def test_written_data_matches_input(self):
        r = _make_reservoir()
        tokens = torch.tensor([[7, 8, 9]])
        firings = torch.tensor([[True, False, True]])
        aw = ActivationWindows(
            component_idx=torch.tensor([2]),
            token_windows=tokens,
            firing_windows=firings,
            activation_windows={"ci": torch.tensor([[0.1, 0.2, 0.3]])},
        )
        r.add(aw)

        assert torch.equal(r.tokens[2, 0], tokens[0])
        assert torch.equal(r.firings[2, 0], firings[0])
        assert torch.allclose(r.acts["ci"][2, 0], torch.tensor([0.1, 0.2, 0.3]))


class TestMerge:
    def test_merge_combines_underfilled(self):
        r1 = _make_reservoir()
        r2 = _make_reservoir()

        r1.add(_make_activation_window([0], torch.full((1, WINDOW), 1, dtype=torch.long)))
        r2.add(_make_activation_window([0], torch.full((1, WINDOW), 2, dtype=torch.long)))

        r1.merge(r2)
        assert r1.n_items[0] == 2
        assert r1.n_seen[0] == 2

    def test_merge_weighted_by_n_seen(self):
        torch.manual_seed(0)

        n_trials = 200
        heavy_wins = 0
        for _ in range(n_trials):
            r_heavy = _make_reservoir()
            r_light = _make_reservoir()

            for _ in range(K):
                r_heavy.add(
                    _make_activation_window([0], torch.full((1, WINDOW), 1, dtype=torch.long))
                )
            r_heavy.n_seen[0] = 1000

            for _ in range(K):
                r_light.add(
                    _make_activation_window([0], torch.full((1, WINDOW), 2, dtype=torch.long))
                )
            r_light.n_seen[0] = 1

            r_heavy.merge(r_light)
            from_heavy = (r_heavy.tokens[0, :, 0] == 1).sum().item()
            if from_heavy == K:
                heavy_wins += 1

        assert heavy_wins > n_trials * 0.8

    def test_merge_n_seen_sums(self):
        r1 = _make_reservoir()
        r2 = _make_reservoir()

        for i in range(K + 5):
            r1.add(_make_activation_window([0], torch.full((1, WINDOW), i % 10, dtype=torch.long)))
        for i in range(K + 3):
            r2.add(_make_activation_window([0], torch.full((1, WINDOW), i % 10, dtype=torch.long)))

        total = r1.n_seen[0].item() + r2.n_seen[0].item()
        r1.merge(r2)
        assert r1.n_seen[0] == total
        assert r1.n_items[0] == K


class TestExamples:
    def test_yields_correct_items(self):
        r = _make_reservoir()
        for i in range(2):
            aw = ActivationWindows(
                component_idx=torch.tensor([0]),
                token_windows=torch.full((1, WINDOW), i + 10, dtype=torch.long),
                firing_windows=torch.ones(1, WINDOW, dtype=torch.bool),
                activation_windows={"ci": torch.ones(1, WINDOW) * (i + 1) * 0.1},
            )
            r.add(aw)

        examples = list(r.examples(0))
        assert len(examples) == 2
        ex0 = examples[0]
        assert ex0.token_ids == [10, 10, 10]
        assert all(ex0.firings)
        assert ex0.activations["ci"] == [pytest.approx(0.1)] * 3

    def test_filters_sentinels(self):
        r = _make_reservoir()
        r.tokens[0, 0] = torch.tensor([WINDOW_PAD_SENTINEL, 5, 6])
        r.firings[0, 0] = torch.tensor([False, True, True])
        r.acts["ci"] = torch.zeros(N_COMPONENTS, K, WINDOW)
        r.acts["ci"][0, 0] = torch.tensor([0.0, 0.8, 0.9])
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
        for i in range(2):
            aw = ActivationWindows(
                component_idx=torch.tensor([1]),
                token_windows=torch.full((1, WINDOW), i + 5, dtype=torch.long),
                firing_windows=torch.ones(1, WINDOW, dtype=torch.bool),
                activation_windows={"ci": torch.ones(1, WINDOW) * 0.5},
            )
            r.add(aw)

        sd = r.state_dict()
        restored = ActivationExamplesReservoir.from_state_dict(sd, device=DEVICE)

        assert restored.k == r.k
        assert restored.window == r.window
        assert torch.equal(restored.tokens, r.tokens)
        assert torch.equal(restored.firings, r.firings)
        for at in r.acts:
            assert torch.equal(restored.acts[at], r.acts[at])
        assert torch.equal(restored.n_items, r.n_items)
        assert torch.equal(restored.n_seen, r.n_seen)

    def test_state_dict_on_cpu(self):
        r = _make_reservoir()
        r.add(_make_activation_window([0], torch.full((1, WINDOW), 1, dtype=torch.long)))

        sd = r.state_dict()
        assert isinstance(sd["tokens"], torch.Tensor) and sd["tokens"].device == torch.device("cpu")
        assert isinstance(sd["n_items"], torch.Tensor) and sd["n_items"].device == torch.device(
            "cpu"
        )
