"""Legacy harvest.db (JSON-blob) -> v2 scope shards, read back through reader + ScopeStore."""

import json
import sqlite3
from pathlib import Path

import pytest

from param_decomp_lab.scope.artifacts import SiteShardReader
from param_decomp_lab.scope.backend.store import ScopeStore
from param_decomp_lab.scope.migrate_legacy_harvest import migrate_subrun

RUN_ID = "p-legacy-test"
SITE = "model.layers.0.mlp.down_proj"
SITE_B = "model.layers.0.mlp.up_proj"
SUBRUN = "h-20260101_000000"
N_TOKENS = 2 * 1 * 512  # n_batches * batch_size * LEGACY_SEQ_LEN
WINDOW = 5


@pytest.fixture
def out_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("param_decomp_lab.scope.artifacts.PARAM_DECOMP_OUT_DIR", tmp_path)
    monkeypatch.setattr("param_decomp_lab.scope.backend.store.PARAM_DECOMP_OUT_DIR", tmp_path)
    monkeypatch.setattr(
        "param_decomp_lab.scope.migrate_legacy_harvest.PARAM_DECOMP_OUT_DIR", tmp_path
    )
    return tmp_path


def _example(token_ids: list[int], cis: list[float]) -> dict[str, object]:
    return {
        "token_ids": token_ids,
        "firings": [1] * len(token_ids),
        "activations": {
            "causal_importance": cis,
            "component_activation": [2 * c for c in cis],
        },
    }


def _pmi(pairs: list[tuple[int, float]]) -> str:
    return json.dumps({"top": pairs, "bottom": pairs[::-1]})


def _write_legacy_db(out_dir: Path) -> None:
    """Components 0 and 2 fired; component 1 is absent (a legacy dead component)."""
    subrun_dir = out_dir / "runs" / RUN_ID / "harvest" / SUBRUN
    subrun_dir.mkdir(parents=True)
    db = sqlite3.connect(subrun_dir / "harvest.db")
    db.execute(
        """CREATE TABLE components (
            component_key TEXT, layer TEXT, component_idx INT, firing_density REAL,
            n_activation_examples INT, mean_activations TEXT, activation_examples TEXT,
            input_token_pmi TEXT, output_token_pmi TEXT)"""
    )
    db.execute("CREATE TABLE config (key TEXT, value TEXT)")
    db.executemany(
        "INSERT INTO config VALUES (?, ?)",
        [
            ("n_batches", "2"),
            ("batch_size", "1"),
            ("activation_examples_per_component", "2"),
            ("activation_context_tokens_per_side", "2"),
            ("pmi_token_top_k", "2"),
        ],
    )
    comp0_examples = [
        _example([1, 2, 3, 4, 5], [0.1, 0.2, 0.9, 0.2, 0.1]),
        _example([6, 7, 8], [0.05, 0.5, 0.05]),
    ]
    comp2_examples = [_example([9, 10, 11, 12], [0.0, 0.3, 0.0, 0.0])]
    rows = [
        (
            f"{SITE}:0",
            SITE,
            0,
            3 / N_TOKENS,
            2,
            json.dumps({"causal_importance": 0.01, "component_activation": 0.02}),
            json.dumps(comp0_examples),
            _pmi([(262, 8.0), (11, 4.0)]),
            _pmi([(262, 7.0), (12, 3.0)]),
        ),
        (
            f"{SITE}:2",
            SITE,
            2,
            1 / N_TOKENS,
            1,
            json.dumps({"causal_importance": 0.001, "component_activation": 0.002}),
            json.dumps(comp2_examples),
            _pmi([(30, 5.0), (31, 2.0)]),
            _pmi([(32, 5.0), (33, 2.0)]),
        ),
        (
            f"{SITE_B}:0",
            SITE_B,
            0,
            1 / N_TOKENS,
            1,
            json.dumps({"causal_importance": 0.005, "component_activation": 0.006}),
            json.dumps([_example([20, 21], [0.4, 0.1])]),
            _pmi([(40, 5.0), (41, 2.0)]),
            _pmi([(42, 5.0), (43, 2.0)]),
        ),
    ]
    db.executemany("INSERT INTO components VALUES (?,?,?,?,?,?,?,?,?)", rows)
    db.commit()
    db.close()


def test_migrate_round_trip(out_dir: Path) -> None:
    _write_legacy_db(out_dir)
    published_by_site = {p.parent.name: p for p in migrate_subrun(RUN_ID, SUBRUN, "gpt2")}
    assert set(published_by_site) == {SITE, SITE_B}

    reader = SiteShardReader(published_by_site[SITE])
    assert reader.meta.n_components == 3
    assert reader.meta.k_examples == 2
    assert reader.meta.window == WINDOW
    assert reader.meta.n_tokens_seen == N_TOKENS

    ex0 = reader.examples(0)
    assert list(ex0.lengths) == [5, 3]
    assert list(ex0.token_ids[0]) == [1, 2, 3, 4, 5]
    assert list(ex0.token_ids[1]) == [6, 7, 8, 0, 0]  # zero-padded past the real tokens
    assert ex0.ci[0].max() == pytest.approx(0.9, abs=1e-3)
    assert ex0.act[1, 1] == pytest.approx(1.0, abs=1e-3)  # 2 * ci

    assert reader.examples(1).token_ids.shape[0] == 0  # gap becomes a dead slot

    row = reader.db.execute(
        "SELECT firing_count, firing_density, max_act, mean_ci, n_examples "
        "FROM components WHERE idx = 0"
    ).fetchone()
    firing_count, density, max_act, mean_ci, n_examples = row
    assert firing_count == 3
    assert density == pytest.approx(3 / N_TOKENS)
    assert max_act == pytest.approx(0.9)
    assert mean_ci == pytest.approx(0.01)
    assert n_examples == 2

    detail = ScopeStore().component_detail(RUN_ID, SITE, 0, 0, 10)
    assert detail.n_examples == 2
    assert detail.examples[0].max_act == pytest.approx(0.9, abs=1e-3)  # CI-ranked first
    assert detail.input_pmi[0][1] == 8.0
    assert len(detail.examples[1].tokens) == 3

    reader_b = SiteShardReader(published_by_site[SITE_B])
    assert reader_b.meta.n_components == 1
    assert list(reader_b.examples(0).lengths) == [2]
