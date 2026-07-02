"""Viewer read path (ScopeStore) over hand-built shards: catalog states, listing
sort/page/search, detail example ranking + label join, newest-subrun-wins, 404s,
and cross-thread access (the server runs sync endpoints in a FastAPI threadpool).
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from param_decomp_lab.scope.artifacts import (
    FORMAT_VERSION,
    ComponentExamples,
    SiteMeta,
    SiteShardWriter,
    open_labels_db,
    scope_dir,
)
from param_decomp_lab.scope.backend.contract import ScopeNotFoundError
from param_decomp_lab.scope.backend.store import ScopeStore

RUN_ID = "p-store-test"
SITE = "layer_0"
SUBRUN = "h-20260101_000000"
K = 3
W = 5
TOKEN_ID = 262  # gpt2 " the"


@pytest.fixture
def out_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("param_decomp_lab.scope.artifacts.PARAM_DECOMP_OUT_DIR", tmp_path)
    monkeypatch.setattr("param_decomp_lab.scope.backend.store.PARAM_DECOMP_OUT_DIR", tmp_path)
    return tmp_path


def _examples(peak_cis: list[float]) -> ComponentExamples:
    n = len(peak_cis)
    ci = np.zeros((n, W), dtype=np.float16)
    act = np.zeros((n, W), dtype=np.float16)
    for j, peak in enumerate(peak_cis):
        ci[j, :] = peak / 2
        ci[j, W // 2] = peak
        act[j, :] = peak
    return ComponentExamples(
        token_ids=np.full((n, W), TOKEN_ID, dtype=np.uint32),
        firings=np.ones((n, W), dtype=np.uint8),
        ci=ci,
        act=act,
        lengths=np.full((n,), W, dtype=np.uint16),
    )


def _write_site(run_id: str, site: str, subrun_id: str, mean_cis: list[float]) -> None:
    """One component per entry; component i's three examples peak at
    (0.2, 1.0, 0.5) x mean_cis[i], and its firing_density grows with i."""
    meta = SiteMeta(
        format_version=FORMAT_VERSION,
        run_id=run_id,
        site=site,
        subrun_id=subrun_id,
        n_components=len(mean_cis),
        k_examples=K,
        window=W,
        tokenizer_name="gpt2",
        n_tokens_seen=1000,
        pmi_top_k=2,
        provenance="test",
        created_at="2026-01-01T00:00:00Z",
    )
    writer = SiteShardWriter(meta)
    for idx, mean_ci in enumerate(mean_cis):
        writer.write_component(
            idx,
            _examples([0.2 * mean_ci, mean_ci, 0.5 * mean_ci]),
            firing_count=10 * (idx + 1),
            firing_density=0.01 * (idx + 1),
            max_act=mean_ci,
            mean_ci=mean_ci,
            mean_act=mean_ci / 2,
            input_pmi=[(TOKEN_ID, 8.0), (11, 4.0)],
            output_pmi=[(TOKEN_ID, 7.0)],
        )
    writer.publish()


def _write_label(run_id: str, site: str, idx: int, text: str) -> None:
    conn = open_labels_db(run_id, readonly=False)
    conn.execute(
        "INSERT INTO labels VALUES (?,?,?,?,?,?,?)",
        (site, idx, text, "test-model", 0.01, "2026-01-01T00:00:00Z", "test"),
    )
    conn.commit()
    conn.close()


@pytest.mark.usefixtures("out_dir")
def test_catalog_states() -> None:
    _write_site(RUN_ID, SITE, SUBRUN, [0.9, 0.5, 0.7])
    (scope_dir(RUN_ID) / "layer_1" / ".tmp-h-x-1").mkdir(parents=True)
    _write_label(RUN_ID, SITE, 0, "alpha")
    _write_label(RUN_ID, SITE, 2, "beta")

    catalog = ScopeStore().catalog()
    (run,) = catalog.runs
    by_site = {s.site: s for s in run.sites}
    assert by_site[SITE].n_components == 3
    assert by_site[SITE].n_labeled == 2
    assert [s.status for s in by_site[SITE].subruns] == ["present"]
    assert [s.status for s in by_site["layer_1"].subruns] == ["in_flight"]
    assert by_site["layer_1"].n_components == 0


@pytest.mark.usefixtures("out_dir")
def test_list_sorting_paging_search() -> None:
    _write_site(RUN_ID, SITE, SUBRUN, [0.4, 0.9, 0.1, 0.6])
    _write_label(RUN_ID, SITE, 1, "alpha cat")
    _write_label(RUN_ID, SITE, 2, "beta cat")
    store = ScopeStore()

    listing = store.list_components(RUN_ID, SITE, "mean_ci", 0, 10, "")
    assert listing.total == 4
    assert [r.idx for r in listing.items] == [1, 3, 0, 2]
    assert listing.items[0].label == "alpha cat"

    page = store.list_components(RUN_ID, SITE, "mean_ci", 1, 2, "")
    assert [r.idx for r in page.items] == [0, 2]

    by_density = store.list_components(RUN_ID, SITE, "density", 0, 10, "")
    assert [r.idx for r in by_density.items] == [3, 2, 1, 0]

    hits = store.list_components(RUN_ID, SITE, "mean_ci", 0, 10, "alpha")
    assert hits.total == 1
    assert hits.items[0].idx == 1

    unlabeled_first = store.list_components(RUN_ID, SITE, "unlabeled_first", 0, 10, "")
    assert [r.idx for r in unlabeled_first.items] == [3, 0, 1, 2]


@pytest.mark.usefixtures("out_dir")
def test_detail_ranked_examples_and_label() -> None:
    _write_site(RUN_ID, SITE, SUBRUN, [0.4, 0.9, 0.1])
    _write_label(RUN_ID, SITE, 1, "quote detector")
    store = ScopeStore()

    detail = store.component_detail(RUN_ID, SITE, 1, 0, 2)
    assert detail.rank == 0 and detail.prev_idx is None and detail.next_idx == 0
    assert detail.n_examples == 3 and len(detail.examples) == 2
    peaks = [e.max_act for e in detail.examples]
    assert peaks == sorted(peaks, reverse=True)
    assert peaks[0] == pytest.approx(0.9, abs=1e-2)
    assert len(detail.examples[0].tokens) == W
    assert len(detail.examples[0].cis) == W and len(detail.examples[0].acts) == W
    assert detail.label is not None
    assert detail.label.text == "quote detector" and detail.label.model == "test-model"
    assert len(detail.input_pmi) == 2 and detail.input_pmi[0][1] == 8.0

    last_page = store.component_detail(RUN_ID, SITE, 1, 1, 2)
    assert len(last_page.examples) == 1
    assert last_page.examples[0].max_act == pytest.approx(0.2 * 0.9, abs=1e-2)

    unlabeled = store.component_detail(RUN_ID, SITE, 2, 0, 10)
    assert unlabeled.label is None
    assert unlabeled.rank == 2 and unlabeled.prev_idx == 0 and unlabeled.next_idx is None


@pytest.mark.usefixtures("out_dir")
def test_newest_subrun_wins() -> None:
    _write_site(RUN_ID, SITE, "h-20260101_000000", [0.4, 0.9])
    _write_site(RUN_ID, SITE, "h-20260202_000000", [0.9, 0.1])
    listing = ScopeStore().list_components(RUN_ID, SITE, "mean_ci", 0, 10, "")
    assert [r.idx for r in listing.items] == [0, 1]
    assert listing.items[0].mean_ci == pytest.approx(0.9)


@pytest.mark.usefixtures("out_dir")
def test_unknown_site_and_component_404() -> None:
    _write_site(RUN_ID, SITE, SUBRUN, [0.4])
    store = ScopeStore()
    with pytest.raises(ScopeNotFoundError):
        store.list_components(RUN_ID, "layer_404", "mean_ci", 0, 10, "")
    with pytest.raises(ScopeNotFoundError):
        store.component_detail(RUN_ID, SITE, 99, 0, 10)


@pytest.mark.usefixtures("out_dir")
def test_reads_work_across_threads() -> None:
    """Cached sqlite connections must be usable from any threadpool thread."""
    _write_site(RUN_ID, SITE, SUBRUN, [0.4, 0.9, 0.1])
    store = ScopeStore()
    store.component_detail(RUN_ID, SITE, 0, 0, 10)  # populate caches on this thread

    def read(idx: int) -> int:
        store.component_detail(RUN_ID, SITE, idx, 0, 10)
        return store.list_components(RUN_ID, SITE, "mean_ci", 0, 10, "").total

    with ThreadPoolExecutor(max_workers=4) as pool:
        totals = list(pool.map(read, [0, 1, 2, 0, 1, 2]))
    assert totals == [3] * 6
