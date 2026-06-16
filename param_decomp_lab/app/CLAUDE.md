# PD App

Web-based **read-only viewer** for exploring saved decompositions.

- **Backend**: Python FastAPI (`backend/`) — **reads JAX runs natively, imports zero
  torch** (#841). It opens the run's orbax checkpoint via
  `jax_single_pool.load_run.open_jax_run` (the model forward) and reads the run's
  torch-free `param_decomp_config.lm.LMExperimentConfig` for target/data/algorithm
  metadata, plus the pre-computed harvest / autointerp / cluster repos.
- **Frontend**: Svelte 5 + TypeScript (`frontend/`)
- **Database**: SQLite at `PARAM_DECOMP_OUT_DIR/app/prompt_attr.db` (shared across team via
  NFS) — only `runs` + `prompts` now (attribution graphs / interventions are gone).
- **TODOs**: See `TODO.md` for open work items

## What the backend does NOT do (CUT in #841, dropped-feature sweep per TRANSITION §1)

On-the-fly **attribution graphs, interventions, and circuit optimization** were removed
when the app became a read-only JAX viewer (#841) — they were the only torch-requiring
code. Deleted there: `backend/compute.py`, `backend/optim_cis.py`,
`backend/routers/{graphs,intervention,agents}.py`, the `/probe` endpoint, and the MCP
tools that ran them (`optimize_graph`, `run_ablation`, `probe_component`,
`save_graph_artifact`).

The **dataset-attribution and graph-interp surface** then went too, per TRANSITION §1:
`backend/routers/{dataset_attributions,graph_interp}.py`, the frontend
`api/{datasetAttributions,graphInterp}.ts` clients, `ModelGraph.svelte`,
`DatasetAttributionsSection.svelte`, `GraphInterpBadge.svelte`, and the backend
`dataset_attributions/` + `graph_interp/` modules. `param_decomp_lab/editing/` (a
downstream consumer of the dropped torch compute) was deleted with them. The torch
oracle for that compute lives at git tag `torch-oracle`; these are backlog items to
re-home onto a JAX path.

Live LM next-token probabilities (prompt previews, dataset-search rows, MCP
`create_prompt`) come from `backend/inference.py::next_token_probs`, which reads
`open_jax_run(...).forward(...).output_probs` (softmax of the clean frozen-target
logits) — the same quantity the old torch path computed.

`open_jax_run` is SimpleMLP-only today; loading a llama8b run raises (phase-4 follow-up).
Pre-#727 runs that declare `weights_dtype: float32` are refused by
`build_experiment_config`'s dtype assert — re-stamp the run's `experiment_config.yaml`
with a supported dtype to load it.

## Project Context

This is a **rapidly iterated research tool**. Key implications:

- **Database is persistent shared state**: Lives at `PARAM_DECOMP_OUT_DIR/app/prompt_attr.db` on NFS, shared across the team. Do not delete. Uses DELETE journal mode (NFS-safe) with `fcntl.flock` write locking for concurrent access.
  - **Schema changes require manual migration**: Update the `CREATE TABLE IF NOT EXISTS` statements to match the desired schema, then manually `ALTER TABLE` the real DB (back it up first). No automatic migration framework — just SQL.
  - Keep the CREATE TABLE statements as the source of truth for the schema.
- **Prefer simplicity**: Avoid over-engineering for hypothetical future needs
- **Fail loud and fast**: The users are a small team of highly technical people. Errors are good. We want to know immediately if something is wrong. No soft failing, assert, assert, assert
- **Token display**: Always ship token strings rendered server-side via `AppTokenizer`, never raw token IDs. For embed/output layers, `component_idx` is a token ID — resolve it to a display string in the backend response.

## Running the App

```bash
python -m param_decomp_lab.app.run_app
```

This launches both backend (FastAPI/uvicorn) and frontend (Vite) dev servers.

---

## Architecture Overview

### Backend Structure

```
backend/
├── server.py              # FastAPI app, CORS, routers (no torch, no CUDA check)
├── state.py               # Singleton StateManager + RunState (LoadedJaxRun + repos)
├── topology.py            # AppTopology: torch-free canonical ↔ concrete path mapping
├── inference.py           # next_token_probs from open_jax_run(...).forward().output_probs
├── app_tokenizer.py       # AppTokenizer: wraps HF tokenizers for display/encoding
├── schemas.py             # Pydantic API models
├── dependencies.py        # FastAPI dependency injection
├── utils.py               # Logging/timing utilities
├── database.py            # SQLite interface (runs + prompts only)
└── routers/
    ├── runs.py            # Load a JAX run (open_jax_run + LMExperimentConfig) + GET /api/status
    ├── run_registry.py    # Architecture + data-availability lookups for the frontend run list
    ├── prompts.py         # Prompt management (next-token probs via JAX forward)
    ├── activation_contexts.py  # Serves pre-harvested activation contexts
    ├── correlations.py    # Component correlations + token stats + interpretations
    ├── autointerp_compare.py   # List autointerp subruns + serve interpretations from each
    ├── data_sources.py    # Provenance: subrun IDs, configs, counts (harvest/autointerp)
    ├── pretrain_info.py   # Target-model architecture lookups without loading checkpoints
    ├── investigations.py  # List and serve investigation outputs
    ├── clusters.py        # Component clustering
    ├── dataset_search.py  # Dataset search (reads dataset from run config)
    └── mcp.py             # MCP endpoint for Claude Code (read-only tools only)
```

Note: Activation contexts, correlations, and token stats are loaded from pre-harvested
data (see `param_decomp_lab/harvest/`). The backend never runs the decomposition forward
except to read clean-logit next-token probabilities (`inference.py`).

`AppTopology` (canonical ↔ concrete, e.g. `0.mlp.up` ↔ `h.0.mlp.c_fc`) is built from the
JAX target's model-type name + decomposed site names via the shared `_PathSchema`
definitions in `param_decomp_lab/topology/path_schemas.py` (torch-free). The torch-coupled
`TransformerTopology` (built from an `nn.Module`) is NOT used by the app.

### Frontend Structure

```
frontend/src/
├── App.svelte
├── lib/
│   ├── api/                      # Modular API client (one file per router)
│   │   ├── index.ts              # Re-exports all API modules
│   │   ├── runs.ts               # Run loading
│   │   ├── runRegistry.ts        # Run-list metadata + data availability
│   │   ├── prompts.ts            # Prompt management
│   │   ├── activationContexts.ts # Activation contexts
│   │   ├── correlations.ts       # Correlations + interpretations
│   │   ├── autointerpCompare.ts  # Autointerp subrun comparison
│   │   ├── dataSources.ts        # Data provenance
│   │   ├── pretrainInfo.ts       # Target-model architecture
│   │   ├── investigations.ts     # Investigation outputs
│   │   ├── dataset.ts            # Dataset search
│   │   └── clusters.ts           # Component clustering
│   ├── index.ts                  # Shared utilities (Loadable<T> pattern)
│   ├── graphLayout.ts               # Shared graph layout (parseLayer, row sorting)
│   ├── promptAttributionsTypes.ts # TypeScript types
│   ├── colors.ts                 # Color utilities
│   ├── registry.ts               # Component registry
│   ├── runState.svelte.ts        # Global run-scoped state (Svelte 5 runes)
│   ├── displaySettings.svelte.ts # Display settings state (Svelte 5 runes)
│   └── clusterMapping.svelte.ts  # Cluster mapping state
└── components/
    ├── RunSelector.svelte            # Run selection screen
    ├── RunView.svelte                # Tab shell (read-only viewer)
    ├── ActivationContextsTab.svelte  # Component firing patterns tab (default tab)
    ├── ActivationContextsViewer.svelte
    ├── ActivationContextsPagedTable.svelte
    ├── DatasetSearchTab.svelte       # Dataset search UI
    ├── DatasetSearchResults.svelte
    ├── ClusterPathInput.svelte       # Cluster path selector (dropdown populated from registry.ts)
    ├── TokenHighlights.svelte        # Token highlighting
    ├── prompt-attr/                  # Attribution-graph layout helpers reused by the investigations ArtifactGraph
    │   ├── NodeTooltip.svelte            # Hover card
    │   ├── ComponentNodeCard.svelte      # Component details
    │   ├── ComponentCorrelationPills.svelte
    │   ├── OutputNodeCard.svelte         # Output node details
    │   └── graphUtils.ts                 # Layout helpers
    └── ui/                               # Reusable UI components
        ├── ComponentCorrelationMetrics.svelte
        ├── ComponentPillList.svelte
        ├── DisplaySettingsDropdown.svelte
        ├── EdgeAttributionList.svelte
        ├── InterpretationBadge.svelte    # LLM interpretation labels
        ├── SectionHeader.svelte
        ├── SetOverlapVis.svelte
        ├── StatusText.svelte
        ├── TokenPillList.svelte
        └── TokenStatsSection.svelte
```

---

## Key Data Structures

### Node Keys

Node keys follow the format `"layer:seq:cIdx"` where:

- `layer`: Model layer name (e.g., `h.0.attn.q_proj`, `h.2.mlp.c_fc`)
- `seq`: Sequence position (0-indexed)
- `cIdx`: Component index within the layer

### Pseudo-layers

`wte` and `output` are **pseudo-layers** for display only:

- `wte` (word token embedding): Input embeddings, single pseudo-component (idx 0)
- `output`: Output logits, component_idx = token_id

The backend no longer computes attribution graphs or interventions (CUT — see top of
file).

---

## Data Flow

### Run Loading

```
POST /api/runs/load(wandb_path, context_length)
  → run_dir = PARAM_DECOMP_OUT_DIR/runs/<run_id>
  → open_jax_run(run_dir)                       # orbax checkpoint → forward()
  → LMExperimentConfig.from_file(experiment_config.yaml)   # torch-free target/data/pd
  → AppTopology.from_model_type(...)            # canonical path mapping
  → AppTokenizer.from_pretrained(data.tokenizer_name)
  → open harvest/interp repos
  → store in StateManager singleton
  ← {status, run_id, wandb_path}
```

### Component Correlations & Interpretations

```
GET /api/correlations/components/{layer}/{component_idx}
  → Load from HarvestRepo (pre-harvested data)
  ← ComponentCorrelationsResponse (precision, recall, jaccard, pmi)

GET /api/correlations/token_stats/{layer}/{component_idx}
  → Load from HarvestRepo
  ← TokenStatsResponse (input/output token associations)

GET /api/correlations/interpretation/{layer}/{component_idx}
  → Load from HarvestRepo (autointerp results)
  ← InterpretationResponse (label, confidence, reasoning)
```

### Dataset Search

```
POST /api/dataset/search?query=...
  → Search the loaded run's training dataset (reads dataset_name from config)
  ← DatasetSearchMetadata (includes dataset_name)

GET /api/dataset/results?page=1&page_size=20
  ← Paginated search results (text + generic metadata dict)
```

---

## Database Schema

Located at `PARAM_DECOMP_OUT_DIR/app/prompt_attr.db` (shared via NFS). Uses DELETE journal mode with `fcntl.flock` write locking for safe concurrent access from multiple backends.

| Table     | Key                        | Purpose             |
| --------- | -------------------------- | ------------------- |
| `runs`    | `wandb_path`               | W&B run references  |
| `prompts` | `(run_id, context_length)` | Token sequences     |

The `graphs` / `intervention_runs` tables are gone (attribution + intervention CUT). The
real shared DB may still carry those tables — they're just unused.

Note: Activation contexts, correlations, token stats, and interpretations are loaded from pre-harvested data at `PARAM_DECOMP_OUT_DIR/{harvest,autointerp}/` (see `param_decomp_lab/harvest/` and `param_decomp_lab/autointerp/`).

---

## State Management

### Backend (`state.py`)

```python
StateManager.get() → AppState:
  - db: PromptAttrDB (always available)
  - run_state: RunState | None
      - jax_run: LoadedJaxRun       # open_jax_run output: forward() + site_names + vocab
      - topology: AppTopology       # torch-free canonical ↔ concrete path mapping
      - tokenizer: AppTokenizer     # Token display, encoding, span construction
      - config: PDConfig            # the run's algorithm config (for /status config_yaml)
      - lm_target, lm_data          # torch-free LMExperimentConfig sub-configs
      - context_length
      - harvest / interp repos  # pre-computed data
  - dataset_search_state: DatasetSearchState | None  # Cached search results

HarvestRepo:  # Lazy-loads from PARAM_DECOMP_OUT_DIR/runs/<run_id>/harvest/
  - correlations: CorrelationStorage | None
  - token_stats: TokenStatsStorage | None
  - activation_contexts: dict[str, ComponentData] | None
  - interpretations: dict[str, InterpretationResult] | None
```

### Frontend (`PromptAttributionsTab.svelte`)

- `promptCards` - All open prompt analysis cards
- `activeCard` / `activeGraph` - Current selection
- `pinnedNodes` - Highlighted nodes for tracing
- `componentDetailsCache` - Lazy-loaded component info

---

## Svelte 5 Conventions

- Use `SvelteSet`/`SvelteMap` from `svelte/reactivity` instead of `Set`/`Map` - they're reactive without `$state()` wrapping
- **Isolate nullability at higher levels**: Handle loading/error/null states in wrapper components so inner components can assume data is present. Pass loaded data as props rather than having children read from context and check status. This avoids optional chaining and null checks scattered throughout the codebase.
  - `RunView` guards with `{#if runState.prompts.status === "loaded" && ...}` and passes `.data` as props to `PromptAttributionsTab` - the status check both guards rendering and narrows the type
  - `ActivationContextsTab` loads data and shows loading state, then renders `ActivationContextsViewer` only when data is ready

---

## Performance Notes

- **Edge limit**: `GLOBAL_EDGE_LIMIT = 50000` in graph visualization
- **SSE streaming**: Long computations stream progress updates
- **Lazy loading**: Component details fetched on hover/pin
