# App Backend Review & Action Items

Review date: 2026-03-04. Scope: `spd/app/backend/` — all Python files.

Context: the app is a **researcher-first local tool** (frontend + backend launched together, opened in browser). Errors should be loud, silent failures absent, the prompt DB is deletable short-term state, no backwards compatibility needed.

## Overview

The backend is ~6,500 lines across 18 Python files. The core architecture (FastAPI + SQLite + singleton state + SSE streaming) is sound. The main concerns are: a few real bugs, accumulated dead code, some silent failures that violate the "loud errors" principle, and a few design seams where complexity hides.

### File size inventory

| File | Lines | Risk |
|---|---|---|
| `routers/mcp.py` | 1637 | High — mixed concerns, largest file |
| `routers/graphs.py` | 1036 | Medium — streaming complexity |
| `compute.py` | 920 | Low — core algorithm, well-structured |
| `database.py` | 827 | Medium — manual serialization |
| `optim_cis.py` | 504 | Low |
| `routers/dataset_search.py` | 473 | Medium — hardcoded dataset names |
| `routers/correlations.py` | 386 | Low |
| `routers/graph_interp.py` | 373 | Low |
| `routers/investigations.py` | 317 | Low |
| `routers/pretrain_info.py` | 246 | Low |
| `server.py` | 212 | Low — clean |
| `routers/activation_contexts.py` | 207 | Low |
| `routers/runs.py` | 191 | Low |
| `routers/dataset_attributions.py` | 170 | Low |
| `routers/intervention.py` | 169 | Low — clean |
| `state.py` | 132 | Low — clean |
| `app_tokenizer.py` | 119 | Low |
| `routers/prompts.py` | 115 | Low |

---

## Bugs

### 1. `dataset_search.py:262` — KeyError on tokenized results

`get_tokenized_results` accesses `result["story"]` but `search_dataset` stores results with key `"text"` (line 137: `results.append({"text": text, ...})`). This will crash with `KeyError: 'story'` whenever tokenized results are requested.

**Fix:** Change line 262 from `result["story"]` to `result["text"]`. Also line 287: the metadata exclusion list references `"story"` — should be `"text"`.

### 2. `dataset_search.py` — random endpoints hardcode SimpleStories

`get_random_samples` (line 336) and `get_random_samples_with_loss` (line 415) both hardcode `load_dataset("lennart-finke/SimpleStories", ...)` and access `item_dict["story"]`. Since primary models are now Pile-trained, these endpoints are broken for current research. They also don't use `DepLoadedRun` to get the dataset name from the run config like `search_dataset` does.

**Fix:** Make them take `DepLoadedRun`, read `task_config.dataset_name` and `task_config.column_name`, and use those instead of hardcoded values. Or, if the random endpoints aren't used with Pile models, consider deleting them.

---

## Dead code to delete

### 3. `ForkedInterventionRunRecord` + `forked_intervention_runs` table

`database.py:117-125` defines `ForkedInterventionRunRecord`. Lines 256-265 create the `forked_intervention_runs` table. Lines 744-827 implement 3 CRUD methods (`save_forked_intervention_run`, `get_forked_intervention_runs`, `delete_forked_intervention_run`). No router references any of these — the fork endpoints were removed. Delete all of it.

Files: `database.py`

### 4. `optim_cis.py:500-504` — `get_out_dir()` never called

Dead utility function that creates a local `out/` directory. Nothing references it.

Files: `optim_cis.py`

### 5. Unused schemas in `graphs.py:188-209`

`ComponentStats`, `PromptSearchQuery`, and `PromptSearchResponse` are defined but no endpoint uses them. They appear to be leftovers from a removed prompt search feature. The `PromptPreview` in `graphs.py:114` also duplicates the one in `prompts.py:25`.

Files: `routers/graphs.py`

### 6. `spd/app/TODO.md` was empty

(This file — now repurposed for this review.)

---

## Design issues

### 7. `OptimizationParams` mixes config inputs with computed outputs

`database.py:69-82` — Fields like `imp_min_coeff`, `steps`, `pnorm`, `beta` are optimization *inputs*. Fields like `ci_masked_label_prob`, `stoch_masked_label_prob`, `adv_pgd_label_prob` are computed *outputs*. These metrics are mutated in-place after construction in `graphs.py:759-761`.

This makes the object's contract unclear — is it immutable config or mutable state?

**Suggestion:** Either nest the metrics in a sub-model (`metrics: OptimMetrics | None`), or at minimum stop mutating after construction (compute the metrics before constructing `OptimizationParams`).

### 8. `StoredGraph.id = -1` sentinel value

`database.py:90` uses `-1` as "unsaved graph". If a graph is accidentally used before being saved, that `-1` leaks into API responses or DB queries. `id: int | None = None` is more honest and lets the type system catch misuse.

### 9. GPU lock accessed two different ways

- `graphs.py:603,844` — `stream_computation(work, manager._gpu_lock)` reaches into the private lock directly
- `intervention.py:86` — `with manager.gpu_lock():` uses the context manager

The stream pattern is inherently different (hold lock across SSE generator lifetime), but accessing `_gpu_lock` directly breaks encapsulation.

**Suggestion:** Add a `stream_with_gpu_lock(work)` method on `StateManager` that encapsulates the lock acquisition + SSE streaming pattern. Then `graphs.py` calls `manager.stream_with_gpu_lock(work)` instead of reaching into privates.

### 10. `load_run` returns untyped dicts

`runs.py:96,139` returns `{"status": "loaded", "run_id": ...}` and `{"status": "already_loaded", ...}`. No response model, so the frontend has no type-safe contract for this endpoint.

**Fix:** Define a `LoadRunResponse(BaseModel)` with `status`, `run_id`, `wandb_path`.

### 11. Edge truncation is invisible to the user

`graphs.py:903` logs a warning when edges exceed `GLOBAL_EDGE_LIMIT = 50_000` and are truncated, but this info only goes to server logs. The researcher never sees it.

**Fix:** Add `edges_truncated: bool` (or `total_edge_count: int`) to `GraphData` so the frontend can show a notice.

### 12. Module-level `DEVICE = get_device()` in multiple files

`graphs.py:266`, `intervention.py:48`, `dataset_search.py`, `prompts.py:18` all call `get_device()` at import time. Fine in practice but makes testing and non-GPU imports impossible.

**Suggestion:** Move to a function call or lazily-evaluated property when/if this becomes a testing bottleneck. Low priority.

### 13. `_GRAPH_INTERP_MOCK_MODE` cross-router import

`runs.py:13` imports `MOCK_MODE` from `routers/graph_interp.py` and uses it in the status endpoint (line 174). The TODO comment says to remove it. This cross-router dependency for a mock flag should be cleaned up — the mock mode should either be a config flag on `StateManager` or deleted entirely.

---

## Silent failure patterns (violate "loud errors" principle)

### 14. `compute.py:79-86` — output node capping is silent

`compute_layer_alive_info` caps output nodes to `MAX_OUTPUT_NODES_PER_POS = 15` per position without any logging or indication. If a researcher has >15 high-probability output tokens at a position, they silently lose some.

At minimum, log when capping occurs.

### 15. `correlations.py:291,302` — token stats returns `None` silently

`get_component_token_stats` returns `None` when token stats haven't been harvested. This means the endpoint returns a `200 null` response, which the frontend has to special-case. An explicit 404 with a message is more honest.

### 16. `correlations.py:112,260` — interpretations/intruder scores return `{}` silently

`get_all_interpretations` and `get_intruder_scores` return empty dicts when data isn't available. This is defensible for bulk endpoints (the frontend can check emptiness), but it means the researcher has no way to distinguish "no interpretations exist" from "interpretations not yet generated." Consider logging or adding a `has_interpretations` flag to `LoadedRun`.

Note: `LoadedRun.autointerp_available` already partially addresses this. But the endpoints themselves don't use it — they independently check `loaded.interp is None`.

---

## Lower priority / nice-to-haves

### 17. `extract_node_ci_vals` Python double loop

`compute.py:640-648` iterates every `(seq_pos, component_idx)` pair in Python. For large models (39K components × 512 seq), this is a lot of Python overhead. Could be vectorized to only extract non-zero entries.

### 18. `database.py` manual graph get-or-create race

Lines 528-539: catches `IntegrityError` on manual graph save, then re-queries. There's a small race window between the failed insert and the re-query. Acceptable for a single-user local app but worth noting.

### 19. `mcp.py` is 1637 lines

The MCP router is the largest file, mixing tool definitions, implementation logic, and JSON-RPC handling. It has module-level global state (`_investigation_config`). This file would benefit from being split, but it's also likely to be rewritten when MCP tooling matures, so the ROI of refactoring now is debatable.

---

## Suggested priority order for implementation

1. Fix `result["story"]` KeyError (bug #1) — 2 min
2. Delete dead code (items #3-5) — 10 min
3. Fix random dataset endpoints or delete if unused (#2) — 15 min
4. Add `edges_truncated` to GraphData (#11) — 10 min
5. Type the `load_run` response (#10) — 5 min
6. Clean up `_GRAPH_INTERP_MOCK_MODE` (#13) — 5 min
7. Deduplicate `MAX_OUTPUT_NODES_PER_POS` (#5 partial) — 2 min
8. `StoredGraph.id` sentinel → `None` (#8) — 10 min
9. Split `OptimizationParams` (#7) — 20 min
10. GPU lock encapsulation (#9) — 15 min
