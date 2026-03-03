# SPD App

Web-based visualization and analysis tool for exploring neural network component decompositions.

- **Backend**: Python FastAPI (`backend/`)
- **Frontend**: Svelte 5 + TypeScript (`frontend/`)
- **Database**: SQLite at `.data/app/prompt_attr.db` (relative to repo root)
- **TODOs**: See `TODO.md` for open work items

## Project Context

This is a **rapidly iterated research tool**. Key implications:

- **Please do not code for backwards compatibility**: Schema changes don't need migrations, expect state can be deleted, etc.
- **Database is disposable**: Delete `.data/app/prompt_attr.db` if schema changes break things
- **Prefer simplicity**: Avoid over-engineering for hypothetical future needs
- **Fail loud and fast**: The users are a small team of highly technical people. Errors are good. We want to know immediately if something is wrong. No soft failing, assert, assert, assert
- **Token display**: Always ship token strings rendered server-side via `AppTokenizer`, never raw token IDs. For embed/output layers, `component_idx` is a token ID — resolve it to a display string in the backend response.

## Running the App

```bash
python -m spd.app.run_app
```

This launches both backend (FastAPI/uvicorn) and frontend (Vite) dev servers.

---

## Architecture Overview

### Backend Structure

```
backend/
├── server.py              # FastAPI app, CORS, routers
├── state.py               # Singleton StateManager + HarvestRepo (lazy-loaded harvest data)
├── compute.py             # Core attribution computation
├── app_tokenizer.py       # AppTokenizer: wraps HF tokenizers for display/encoding
├── (topology lives at spd/topology.py — TransformerTopology)
├── schemas.py             # Pydantic API models
├── dependencies.py        # FastAPI dependency injection
├── utils.py               # Logging/timing utilities
├── database.py            # SQLite interface
├── optim_cis.py           # Sparse CI optimization
└── routers/
    ├── runs.py            # Load W&B runs + GET /api/model_info
    ├── graphs.py          # Compute attribution graphs
    ├── prompts.py         # Prompt management
    ├── activation_contexts.py  # Serves pre-harvested activation contexts
    ├── intervention.py    # Selective component activation
    ├── correlations.py    # Component correlations + token stats + interpretations
    ├── clusters.py        # Component clustering
    ├── dataset_search.py  # SimpleStories dataset search
    ├── agents.py          # Various useful endpoints that AI agents should look at when helping
    ├── mcp.py             # MCP (Model Context Protocol) endpoint for Claude Code 
    ├── dataset_search.py  # Dataset search (reads dataset from run config)
    └── agents.py          # Various useful endpoints that AI agents should look at when helping
```

Note: Activation contexts, correlations, and token stats are now loaded from pre-harvested data (see `spd/harvest/`). The app no longer computes these on-the-fly.

### Frontend Structure

```
frontend/src/
├── App.svelte
├── lib/
│   ├── api/                      # Modular API client (one file per router)
│   │   ├── index.ts              # Re-exports all API modules
│   │   ├── runs.ts               # Run loading
│   │   ├── graphs.ts             # Attribution graph computation
│   │   ├── prompts.ts            # Prompt management
│   │   ├── activationContexts.ts # Activation contexts
│   │   ├── correlations.ts       # Correlations + interpretations
│   │   ├── intervention.ts       # Selective activation
│   │   ├── dataset.ts            # Dataset search
│   │   └── clusters.ts           # Component clustering
│   ├── index.ts                  # Shared utilities (Loadable<T> pattern)
│   ├── graphLayout.ts               # Shared graph layout (parseLayer, row sorting)
│   ├── promptAttributionsTypes.ts # TypeScript types
│   ├── interventionTypes.ts
│   ├── colors.ts                 # Color utilities
│   ├── registry.ts               # Component registry
│   ├── runState.svelte.ts        # Global run-scoped state (Svelte 5 runes)
│   ├── displaySettings.svelte.ts # Display settings state (Svelte 5 runes)
│   └── clusterMapping.svelte.ts  # Cluster mapping state
└── components/
    ├── RunSelector.svelte            # Run selection screen
    ├── PromptAttributionsTab.svelte   # Main analysis container
    ├── PromptAttributionsGraph.svelte # SVG graph visualization
    ├── ActivationContextsTab.svelte  # Component firing patterns tab
    ├── ActivationContextsViewer.svelte
    ├── ActivationContextsPagedTable.svelte
    ├── DatasetSearchTab.svelte       # Dataset search UI
    ├── DatasetSearchResults.svelte
    ├── ClusterPathInput.svelte       # Cluster path selector
    ├── ComponentProbeInput.svelte    # Component probe UI
    ├── TokenHighlights.svelte        # Token highlighting
    ├── prompt-attr/
    │   ├── InterventionsView.svelte      # Selective activation UI
    │   ├── StagedNodesPanel.svelte       # Pinned nodes list
    │   ├── NodeTooltip.svelte            # Hover card
    │   ├── ComponentNodeCard.svelte      # Component details
    │   ├── ComponentCorrelationPills.svelte
    │   ├── OutputNodeCard.svelte         # Output node details
    │   ├── PromptPicker.svelte
    │   ├── PromptCardHeader.svelte
    │   ├── PromptCardTabs.svelte
    │   ├── ViewControls.svelte
    │   ├── ComputeProgressOverlay.svelte # Progress during computation
    │   ├── TokenDropdown.svelte
    │   ├── graphUtils.ts                 # Layout helpers
    │   └── types.ts                      # UI state types
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

### Non-Interventable Nodes

`wte` and `output` are **pseudo-layers** for visualization only:

- `wte` (word token embedding): Input embeddings, single pseudo-component (idx 0)
- `output`: Output logits, component_idx = token_id

These appear in attribution graphs but **cannot be intervened on**.
Only internal layers (attn/mlp projections) support selective activation.

Helper: `isInterventableNode()` in `promptAttributionsTypes.ts`

### Backend Types (`compute.py`)

```python
Node(layer: str, seq_pos: int, component_idx: int)

Edge(source: Node, target: Node, strength: float, is_cross_seq: bool)
# strength = gradient * activation
# is_cross_seq = True for k/v → o_proj (attention pattern)

PromptAttributionResult(edges: list[Edge], output_probs: Tensor[seq, vocab], node_ci_vals: dict[str, float])
# node_ci_vals maps "layer:seq:c_idx" → CI value
```

### Frontend Types (`promptAttributionsTypes.ts`)

```typescript
GraphData = {
  id: number,
  tokens: string[],
  edges: Edge[],                              // {src, tgt, val}
  outputProbs: Record<string, OutputProbEntry>, // "seq:cIdx" → {prob, token}
  nodeCiVals: Record<string, number>,         // node_key → CI value
  maxAbsAttr: number,
  l0_total: number,                           // total active components
  optimization?: OptimizationResult
}
```

---

## Core Computations

### Attribution Graph (`compute.py`)

**Entry points**:

- `compute_prompt_attributions()` - Uses model's natural CI values
- `compute_prompt_attributions_optimized()` - Sparse CI optimization

**Algorithm** (`compute_edges_from_ci`):

1. Forward pass with CI masks → component activations cached
2. For each target layer, for each alive (seq_pos, component):
   - Compute gradient of target w.r.t. all source layers
   - `strength = grad * source_activation`
   - Create Edge for each alive source component

**Cross-sequence edges**: `topology.is_cross_seq_pair()` detects k/v → o_proj in same attention block.
These have gradients across sequence positions (causal attention pattern).

### Causal Importance (CI)

CI determines which components are "alive":

- Computed via `model.calc_causal_importances()`
- Thresholded: `ci >= ci_threshold` → active
- For output layer: `prob >= output_prob_threshold`

### CI Optimization (`optim_cis.py`)

Finds sparse CI mask that:

- Preserves prediction of target `label_token`
- Minimizes L0 (active component count)
- Uses importance minimality + CE loss (or KL loss)

### Intervention Forward & Masked Predictions

`compute_intervention_forward()`: Forward pass with binary masks (selected=1, rest=0) → top-k predictions.

`compute_masked_predictions()`: Three forward passes for a node selection:
- **CI**: mask = selection (binary on/off)
- **Stochastic**: mask = selection + (1-selection) × rand (binomial leakage)
- **Adversarial** (optimized graphs only): mask = selection + (1-selection) × PGD source

**Training PGD vs Eval PGD**: The PGD settings in the graph optimization config (`adv_pgd_n_steps`, `adv_pgd_step_size`) are a *training* regularizer — they make CI optimization robust. The PGD in masked predictions is an *eval* metric — it measures worst-case leakage for a given node selection. Eval PGD defaults are in `compute.py` (`EVAL_PGD_N_STEPS`, `EVAL_PGD_STEP_SIZE`). Adversarial eval requires a loss target (CE or KL at a position), which only exists on optimized graphs.

**Base intervention run**: Created automatically during graph computation. Uses all nodes with CI > 0 (the graph's rounded CI mask). Persisted as an `intervention_run` so predictions are available synchronously — no lazy fetching.

---

## Data Flow

### Run Loading

```
POST /api/runs/load(wandb_path)
  → Load ComponentModel + tokenizer from W&B
  → Build sources_by_target (valid gradient paths)
  → Store in StateManager singleton
  ← LoadedRun
```

### Graph Computation (SSE streaming)

```
POST /api/graphs
  → compute_prompt_attributions()
  → Stream progress: {type: "progress", current, total, stage}
  ← {type: "complete", data: GraphData}
```

### Intervention

```
POST /api/intervention {text, nodes: ["h.0.attn.q_proj:3:5", ...]}
  → compute_intervention_forward()
  ← InterventionResponse with top-k predictions
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

Located at `.data/app/prompt_attr.db`. Delete this file if schema changes cause issues.

| Table              | Key                                | Purpose                                           |
| ------------------ | ---------------------------------- | ------------------------------------------------- |
| `runs`             | `wandb_path`                       | W&B run references                                |
| `prompts`          | `(run_id, context_length)`         | Token sequences                                   |
| `graphs`           | `(prompt_id, optimization_params)` | Attribution edges + output probs + node CI values |
| `intervention_runs`| `graph_id`                         | Saved intervention results                        |

Note: Activation contexts, correlations, token stats, and interpretations are loaded from pre-harvested data at `SPD_OUT_DIR/{harvest,autointerp}/` (see `spd/harvest/` and `spd/autointerp/`).

---

## State Management

### Backend (`state.py`)

```python
StateManager.get() → AppState:
  - db: PromptAttrDB (always available)
  - run_state: RunState | None
      - model: ComponentModel
      - topology: TransformerTopology  # Model topology (embedding, unembed, cross-seq roles)
      - tokenizer: AppTokenizer     # Token display, encoding, span construction
      - sources_by_target: dict[target_layer → source_layers]
      - config, context_length
      - harvest: HarvestRepo       # Lazy-loaded pre-harvested data
  - dataset_search_state: DatasetSearchState | None  # Cached search results

HarvestRepo:  # Lazy-loads from SPD_OUT_DIR/harvest/<run_id>/
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
