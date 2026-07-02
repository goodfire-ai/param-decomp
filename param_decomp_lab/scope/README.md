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

`run_scope.py` seeds a throwaway fixture store (synthetic shards, see `fixture.py`) into
a temp `PARAM_DECOMP_OUT_DIR` and points the backend at it, so dev needs no real harvest.

Routes: `/` catalog (availability grid), `/r/<run>/s/<site>` paged component browser,
`/r/<run>/s/<site>/c/<idx>` component detail.

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

Sites with no `present` subrun (and out-of-range component indices) 404. The viewer is
read-only: data enters pages only through SSR `load()`, there is no client fetch.

## The read path

`backend/contract.py` holds the response models (mirrored by `frontend/src/lib/types.ts`).
`backend/store.py` is the one implementation: `ScopeStore` reads the artifact store
(`../artifacts.py`) — newest complete subrun per site, sqlite indexes for listing
sort/filter, one mmap seek + gpt2 detokenization for detail. `server.py` constructs it
directly; there is no data-source abstraction. Dev data is real shards written by
`fixture.py` into a temp store — same format, same read path.
