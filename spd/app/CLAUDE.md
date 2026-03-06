# SPD App

Web-based visualization and analysis tool for exploring neural network component decompositions.

- **Backend**: Python FastAPI (`backend/`)
- **Frontend**: Svelte 5 + TypeScript (`frontend/`)
- **Database**: SQLite at `SPD_OUT_DIR/app/prompt_attr.db` (shared across team via NFS)
- **TODOs**: See `TODO.md` for open work items

## Project Context

This is a **rapidly iterated research tool**. Key implications:

- **Database is persistent shared state**: Lives at `SPD_OUT_DIR/app/prompt_attr.db` on NFS, shared across the team. Do not delete. Uses DELETE journal mode (NFS-safe) with `fcntl.flock` write locking for concurrent access.
  - **Schema changes require manual migration**: Update the `CREATE TABLE IF NOT EXISTS` statements to match the desired schema, then manually `ALTER TABLE` the real DB (back it up first). No automatic migration framework — just SQL.
  - Keep the CREATE TABLE statements as the source of truth for the schema.
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
├── compute.py             # Core attribution computation + intervention evaluation
├── app_tokenizer.py       # AppTokenizer: wraps HF tokenizers for display/encoding
├── (topology lives at spd/topology.py — TransformerTopology)
├── schemas.py             # Pydantic API models
├── dependencies.py        # FastAPI dependency injection
├── utils.py               # Logging/timing utilities
├── database.py            # SQLite interface
├── optim_cis.py           # Sparse CI optimization, loss configs, PGD
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
    ├── ClusterPathInput.svelte       # Cluster path selector (dropdown populated from registry.ts)
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

PromptAttributionResult(edges, ci_masked_out_logits, target_out_logits, node_ci_vals, node_subcomp_acts)

TokenPrediction(token, token_id, prob, logit, target_prob, target_logit)

InterventionResult(input_tokens, ci, stochastic, adversarial, ci_loss, stochastic_loss, adversarial_loss)
# ci/stochastic/adversarial are list[list[TokenPrediction]] (per-position top-k)
# losses are evaluated using the graph's implied loss context
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

### Interventions (`compute.py → compute_intervention`)

A single unified function evaluates a node selection under three masking regimes:

- **CI**: mask = selection (binary on/off)
- **Stochastic**: mask = selection + (1-selection) × Uniform(0,1)
- **Adversarial**: PGD optimizes alive-but-unselected components to maximize loss; non-alive get Uniform(0,1)

Returns `InterventionResult` with top-k `TokenPrediction`s per position for each regime, plus per-regime loss values.

**Loss context**: Every graph has an implied loss that interventions evaluate against:

- **Standard/manual graphs** → `MeanKLLossConfig` (mean KL divergence from target across all positions)
- **Optimized graphs** → the graph's optimization loss (CE for a specific token at a position, or KL at a position)

This loss is used for two things: (1) what PGD maximizes during adversarial evaluation, and (2) the `ci_loss`/`stochastic_loss`/`adversarial_loss` metrics reported in `InterventionResult`.

**Alive masks**: `compute_intervention` recomputes the model's natural CI (one forward pass + `calc_causal_importances`) and binarizes at 0 to get alive masks. This ensures the alive set is always the full model's CI — not the graph's potentially sparse optimized CI. PGD can only manipulate alive-but-unselected components.

**Training PGD vs Eval PGD**: The PGD settings in the graph optimization config (`adv_pgd_n_steps`,
`adv_pgd_step_size`) are a _training_ regularizer — they make CI optimization robust. The PGD in
`compute_intervention` is an _eval_ metric — it measures worst-case performance for a given node
selection. Eval PGD defaults are in `compute.py` (`DEFAULT_EVAL_PGD_CONFIG`).

**Base intervention run**: Created automatically during graph computation. Uses all interventable nodes with CI > 0. Persisted as an `intervention_run` so predictions are available synchronously.

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
POST /api/intervention/run {graph_id, selected_nodes, top_k, adv_pgd}
  → compute_intervention(active_nodes, graph_alive_masks, loss_config)
  ← InterventionRunSummary {id, selected_nodes, result: InterventionResult}

InterventionResult = {
  input_tokens, ci, stochastic, adversarial,  // TokenPrediction[][] per regime
  ci_loss, stochastic_loss, adversarial_loss   // loss under each regime
}
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

Located at `SPD_OUT_DIR/app/prompt_attr.db` (shared via NFS). Uses DELETE journal mode with `fcntl.flock` write locking for safe concurrent access from multiple backends.

| Table               | Key                                | Purpose                                                  |
| ------------------- | ---------------------------------- | -------------------------------------------------------- |
| `runs`              | `wandb_path`                       | W&B run references                                       |
| `prompts`           | `(run_id, context_length)`         | Token sequences                                          |
| `graphs`            | `(prompt_id, optimization_params)` | Attribution edges + CI/target logits + node CI values    |
| `intervention_runs` | `graph_id`                         | Saved `InterventionResult` JSON (single `result` column) |

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
