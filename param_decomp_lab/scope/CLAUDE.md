# scope

Large-scale component viewer: harvest orchestration · activation viewer · autointerp.
Built for the 19M-component regime; the old app (`param_decomp_lab/app/`) remains the
rich small-model tool. Design doc lineage: the 2026-06-12 spike (Fermi sizing, feature
whitelist F0–F3, replace-don't-combine subruns).

## Architecture

```
browser ── SvelteKit SSR (frontend/, adapter-node) ── FastAPI (backend/) ── artifact store
                 one load() per route                    mmap + sqlite        (run, site, subrun)
```

- **Artifact store** (`artifacts.py`): per-(run, site, subrun) shards under
  `PARAM_DECOMP_OUT_DIR/runs/<run>/scope/`. `examples.bin` = fixed-shape mmap arrays
  ([C, k, W] tokens/firings/ci/act); `site.db` = indexed per-component scalars,
  top-k PMI; `labels.db` per run. Shards are immutable and
  published by atomic rename; **subruns are attempts — newest complete wins, never
  combined** (re-harvest instead). `convert.py` migrates legacy harvest subruns.
- **Backend** (`backend/`): the API contract lives in `data_source.py` (pydantic
  models + `ScopeDataSource` Protocol). Two sources: `FixtureDataSource` (synthetic,
  for frontend dev) and `ArtifactDataSource` (real). `--data-source` picks at launch.
- **Frontend** (`frontend/`): SvelteKit 2 / Svelte 5 / TS. Dark-only, read-only,
  two-pane master-detail. Routes: catalogue (`/`), then a persistent site shell
  (`r/[run]/s/[site]/+layout`) holding the component sidebar (search · site switch ·
  sort · paged list) with the routed detail as its right pane — the site index
  (`+page`) is the empty-detail placeholder, `c/[idx]/+page` is the component detail.
  The sidebar survives component navigation (SvelteKit layout persistence); each
  component link carries the sidebar's `sort/page/q` query so the layout `load` stays
  consistent and one round trip repaints only the detail.
- **Deploy** (`deploy/serve.sbatch`): always-on dev-pool CPU supervisor job, stable
  tunnel port 10010. Ship code with `deploy/release.sh frontend|backend|both` —
  in-place process bounce (~1s frontend / ~15s backend), tunnel untouched. Full job
  requeue is only for node failures or sbatch-script changes (scancel + sbatch — the
  spooled script does not update on requeue).

## Invariants — hold these when extending

1. **O(k) per component, never O(C)/O(vocab)/O(dataset)** — in storage rows, API
   payloads, and browser memory. Every response ≤50KB gzipped (asserted in
   `server.py::budgeted_json`; keep new routes behind it).
2. **Counts, not ratios, in stored artifacts** (`firing_count` next to density):
   combining/extending stays a code change, not a re-harvest.
3. **Data enters pages only through `load()`** (`+page.server.ts` / `+layout.server.ts`)
   — one round trip per navigation, no client fetch (the link is 111ms RTT; round trips
   are the budget). The viewer is **read-only**: there is NO client-side fetch at all —
   no label POST, no catalogue polling. This matters in production: the `/api` proxy
   exists only in `vite.config.ts` (dev), so under adapter-node (`node build`) any
   client fetch to `/api` 404s. SSR `load()` hits the backend directly and works in both.
   The old app's `Loadable<T>` state machine is deliberately absent: SSR makes "loading"
   a non-state.
4. **Partial data is the normal case**: a site can be absent / in-flight / present;
   pages must render sensibly with any subset of sites present.
5. **All colors/shadows/radii come from tokens in `frontend/src/app.css`** — a single
   dark theme lives there; component styles reference `var(--token)` only. Enforced by
   `frontend/scripts/check-style-tokens.mjs` (part of `npm run check`); it will fail
   the build on any literal color outside app.css (`rgba(var(--hl), α)` alpha
   composition over a token triplet is allowed). A reskin must stay a one-file edit.
6. **Frozen API contract**: backend pydantic models and `frontend/src/lib/types.ts`
   mirror each other by hand — change them in the same commit. (If drift bites,
   generate types from OpenAPI; don't half-fix.)
7. When a UI pattern appears a third time, extract it to `frontend/src/lib/ui/` —
   not before.

## Commands

```bash
python param_decomp_lab/scope/run_scope.py            # dev: backend(fixture) + vite
python -m param_decomp_lab.scope.backend.server --port N --data-source artifacts
cd param_decomp_lab/scope/frontend && npm run check   # token lint + svelte-check
npm run build                                         # adapter-node → build/
param_decomp_lab/scope/deploy/release.sh frontend     # ship a build, ~1s, no tunnel churn
```
