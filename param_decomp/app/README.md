# PD Visualization App

A web app for exploring PD decompositions: attribution graphs, component activations,
correlations, dataset search, autointerp labels, interventions, and clustering. Built
with Svelte 5 (frontend) and FastAPI (backend).

## Quick Start

```bash
make install      # Backend Python deps
make install-app  # Frontend npm deps (one-time)
make app          # Launch backend + frontend dev servers (recommended)
```

To run the servers separately:

```bash
# Backend
python -m param_decomp.app.run_app  # also starts frontend; pass --no-frontend for backend-only

# Frontend
cd param_decomp/app/frontend && npm run dev
```

## Project Structure

```
param_decomp/app/
├── run_app.py                # All-in-one launcher (backend + frontend)
├── backend/
│   ├── server.py             # FastAPI app, CORS, exception handlers, router registration
│   ├── state.py              # StateManager singleton + HarvestRepo (lazy-loaded)
│   ├── compute.py            # Attribution + intervention computation
│   ├── optim_cis.py          # Sparse-CI optimisation, PGD
│   ├── app_tokenizer.py      # AppTokenizer wrapper for HF tokenizers
│   ├── database.py           # SQLite schema + access (NFS-safe)
│   ├── schemas.py            # Pydantic API models
│   ├── dependencies.py       # FastAPI dependency injection helpers
│   ├── utils.py              # Logging/timing utilities
│   └── routers/              # One router per feature area (see CLAUDE.md)
└── frontend/
    ├── package.json
    ├── vite.config.ts
    ├── svelte.config.js
    ├── index.html
    └── src/
        ├── main.ts
        ├── App.svelte
        ├── lib/
        │   ├── api/          # Modular API client (one file per backend router)
        │   ├── *.svelte.ts   # Reactive run/display/cluster state (Svelte 5 runes)
        │   └── *.ts          # Shared types and utilities
        └── components/       # Svelte components (see CLAUDE.md for breakdown)
```

See `CLAUDE.md` for the full router list, frontend file map, data structures, core
computations, and database schema.

## For ML Researchers: Web Dev Cheatsheet

### npm

`package.json` is the JS equivalent of `pyproject.toml`; `npm` is the package manager.

```bash
npm install      # Install deps
npm run dev      # Dev server
npm run check    # Svelte type check
npm run lint     # ESLint
npm run format   # Prettier
```

### Svelte 5 idioms used here

- `$state(value)` — reactive state (replaces `let`)
- `$derived(expr)` — computed value (replaces `$:`)
- `$effect(() => {})` — side effect (replaces `onMount`)
- `bind:value={x}` — two-way binding
- `onclick={handler}` — event handler (replaces `on:click`)
- Use `SvelteSet` / `SvelteMap` from `svelte/reactivity` for reactive collections

## Data Flow Example: Loading a W&B Run

1. **User input** (`App.svelte`): user enters a wandb path and submits.
2. **Frontend API call** (`lib/api/runs.ts`): `POST /api/runs/load`.
3. **Backend route** (`backend/routers/runs.py::load_run`): downloads the
   `ComponentModel`, builds `sources_by_target`, and stores it in the singleton
   `StateManager` (`backend/state.py`).
4. **Lazy harvest data** (`HarvestRepo` on `StateManager`): pre-harvested correlations,
   token stats, activation contexts, and interpretations load on first access.
5. **UI**: dependent tabs (Activation Contexts, Prompt Attributions, etc.) become
   available once the run is loaded.

## API Type Safety

Backend (`backend/schemas.py`) and frontend (`frontend/src/lib/api/*.ts`) types must be
kept in sync manually. When adding or changing an endpoint, update both.
