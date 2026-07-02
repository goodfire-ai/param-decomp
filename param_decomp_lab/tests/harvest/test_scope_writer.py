"""Round-trip: a populated Harvester -> native scope shards -> SiteShardReader + HarvestRepo.

Exercises the store-unification write path (`harvest.scope_writer.write_scope_shards`, wired
through `HarvestRepo.save_results`) against the v2 shard reader and the harvest-consumer
facade. The reservoir is filled through the real `process_batch`, so the left-packing,
per-example `lengths`, and CI/act values are the genuine accumulator output.
"""

from pathlib import Path

import numpy as np
import pytest

from param_decomp_lab.harvest.accumulator import Harvester
from param_decomp_lab.harvest.config import HarvestConfig, ParamDecompHarvestConfig
from param_decomp_lab.harvest.repo import HarvestRepo
from param_decomp_lab.harvest.scope_writer import CI_ACT_TYPE, COMPONENT_ACT_TYPE
from param_decomp_lab.scope.artifacts import SiteShardReader

LAYERS = [("layer_0", 4), ("layer_1", 3)]
VOCAB_SIZE = 12
MAX_EXAMPLES = 8
CONTEXT = 2
WINDOW = 2 * CONTEXT + 1
RUN_ID = "p-761bc061"
SUBRUN_ID = "h-20260101_000000"


@pytest.fixture
def out_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("param_decomp_lab.scope.artifacts.PARAM_DECOMP_OUT_DIR", tmp_path)
    monkeypatch.setattr("param_decomp_lab.harvest.schemas.PARAM_DECOMP_OUT_DIR", tmp_path)
    return tmp_path


def _fire(firings: dict[str, np.ndarray], layer: str, s: int, c: int) -> None:
    firings[layer][0, s, c] = True


def _populated_harvester() -> Harvester:
    h = Harvester(
        layers=LAYERS,
        vocab_size=VOCAB_SIZE,
        max_examples_per_component=MAX_EXAMPLES,
        context_tokens_per_side=CONTEXT,
        max_examples_per_batch_per_component=MAX_EXAMPLES,
        collect_component_cooccurrence=True,
    )
    B, S = 1, 6
    batch = np.arange(B * S).reshape(B, S) % VOCAB_SIZE
    firings = {layer: np.zeros((B, S, c), dtype=np.bool_) for layer, c in LAYERS}
    acts = {
        layer: {at: np.zeros((B, S, c)) for at in (CI_ACT_TYPE, COMPONENT_ACT_TYPE)}
        for layer, c in LAYERS
    }
    output_probs = np.zeros((B, S, VOCAB_SIZE))

    # layer_0:0 fires twice, layer_0:1 once, layer_1:0 once; all others stay dead.
    for s in (1, 3):
        _fire(firings, "layer_0", s, 0)
        acts["layer_0"][CI_ACT_TYPE][0, s, 0] = 0.9
        acts["layer_0"][COMPONENT_ACT_TYPE][0, s, 0] = 4.0
    _fire(firings, "layer_0", 4, 1)
    acts["layer_0"][CI_ACT_TYPE][0, 4, 1] = 0.5
    _fire(firings, "layer_1", 2, 0)
    acts["layer_1"][CI_ACT_TYPE][0, 2, 0] = 0.7

    h.process_batch(batch, firings, acts, output_probs)
    return h


def _config() -> HarvestConfig:
    return HarvestConfig(
        method_config=ParamDecompHarvestConfig(wandb_path=RUN_ID),
        n_batches=1,
        batch_size=1,
        pmi_token_top_k=5,
    )


def test_shard_stores_all_components_and_reads_back(out_dir: Path) -> None:
    h = _populated_harvester()
    HarvestRepo.save_results(h, _config(), RUN_ID, SUBRUN_ID, "gpt2")

    reader = SiteShardReader(out_dir / "runs" / RUN_ID / "scope" / "layer_0" / SUBRUN_ID)
    # store-all: dead components keep a slot so shard idx == component idx
    assert reader.meta.n_components == 4

    ex = reader.examples(0)
    assert ex.token_ids.shape[0] == 2  # layer_0:0 fired twice
    assert ex.lengths.shape == (2,)
    for j in range(2):
        length = int(ex.lengths[j])
        assert 0 < length <= WINDOW
        assert (ex.token_ids[j, length:] == 0).all()  # zero-padded past the real tokens
        assert ex.ci[j, :length].max() == pytest.approx(0.9, abs=1e-3)

    assert reader.examples(2).token_ids.shape[0] == 0  # dead component: empty pool


@pytest.mark.usefixtures("out_dir")
def test_facade_reconstructs_fired_components() -> None:
    h = _populated_harvester()
    HarvestRepo.save_results(h, _config(), RUN_ID, SUBRUN_ID, "gpt2")

    repo = HarvestRepo(RUN_ID, SUBRUN_ID, readonly=True)
    summary = repo.get_summary()
    assert set(summary) == {"layer_0:0", "layer_0:1", "layer_1:0"}

    comp = repo.get_component("layer_0:0")
    assert comp is not None
    assert comp.layer == "layer_0"
    assert comp.component_idx == 0
    assert comp.firing_density == pytest.approx(2 / h.total_tokens_processed)
    assert comp.mean_activations[CI_ACT_TYPE] == pytest.approx(1.8 / h.total_tokens_processed)
    assert len(comp.activation_examples) == 2
    for ex in comp.activation_examples:
        assert len(ex.firings) == len(ex.token_ids)
        assert len(ex.activations[CI_ACT_TYPE]) == len(ex.token_ids)
        assert len(ex.activations[COMPONENT_ACT_TYPE]) == len(ex.token_ids)

    assert repo.get_component("layer_0:2") is None  # dead component filtered out


@pytest.mark.usefixtures("out_dir")
def test_densities_and_counts() -> None:
    h = _populated_harvester()
    HarvestRepo.save_results(h, _config(), RUN_ID, SUBRUN_ID, "gpt2")

    repo = HarvestRepo(RUN_ID, SUBRUN_ID, readonly=True)
    assert repo.get_component_count() == 3
    densities = dict(repo.get_component_densities(min_examples=1))
    assert set(densities) == {"layer_0:0", "layer_0:1", "layer_1:0"}
    assert densities["layer_0:0"] == pytest.approx(2 / h.total_tokens_processed)
