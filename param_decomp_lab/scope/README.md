# scope

A large-scale visualization surface for param-decomp runs, designed for up to ~19M
components per run. Everything is paged and incremental: no view, payload, or in-memory
structure is ever O(total components), every response is ≤50KB gzipped (asserted
server-side), and each navigation is a single SSR round trip — built for high-RTT,
low-bandwidth links. Separate from `param_decomp_lab/app/`.

## Run it

```bash
source .venv/bin/activate
python param_decomp_lab/scope/run_scope.py
# first run installs frontend deps; then open the printed http://localhost:<port>
```

Routes: `/` catalog (availability grid, polls every 5s), `/r/<run>/s/<site>` paged
component browser, `/r/<run>/s/<site>/c/<idx>` component detail.

## Frozen API contract

The team is building the real artifact store against these routes — do not change them.

- `GET /api/catalog` →
  `{"runs": [{"run_id", "sites": [{"site", "n_components", "n_labeled", "subruns": [{"subrun_id", "status": "present"|"in_flight", "n_batches", "progress"}]}]}]}`
- `GET /api/runs/{run_id}/sites/{site}/components?sort=density|max_act|unlabeled_first&page=int&page_size=50&q=str` →
  `{"total", "page", "items": [{"idx", "density", "max_act", "label": str|null}]}` —
  `q` searches label text; sorting/filtering happens in the data source; `page` is 0-indexed.
- `GET /api/runs/{run_id}/sites/{site}/components/{idx}` →
  `{"idx", "density", "max_act", "label": {"text", "model", "cost_usd", "created_at"}|null, "input_pmi": [[str, float], ...], "output_pmi": [...], "examples": [{"tokens": [str], "acts": [float], "cis": [float], "max_act"}, ...]}` —
  ≤30 examples, 41-token windows, tokens are display strings (server-side detokenization).
- `POST /api/runs/{run_id}/sites/{site}/components/{idx}/label` → generates + stores a
  label, returns the label object.

Sites with no `present` subrun (and out-of-range component indices) 404.

## Where the real data plugs in

`backend/data_source.py` defines the `ScopeDataSource` Protocol plus the response
models. `backend/fixture_data_source.py` is the only implementation today: deterministic
seeded synthetic data (2 runs; one 3×30k-component, one 20-site with mixed
present/in-flight/absent availability that progresses in real time). The real
mmap-backed implementation will be a second class satisfying the same Protocol, swapped
in where `server.py` constructs `data_source`. Listing sort/filter must operate on
compact per-site columns (the fixture's `_SiteColumns` is the shape of that store);
component detail is materialized one idx at a time.
