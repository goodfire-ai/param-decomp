# Graph + Interventions Issue Log (2026-03-03)

Scope: `spd/app` local attribution graphs and intervention flows.

## Status Summary

- Subgraph-related issues are noted but currently deferred by request.
- Highest-priority active issue is intervention text reconstruction from display tokens.

## Issues

1. Deferred: manual subgraph creation can crash on missing base intervention run
- Severity: Critical
- Status: Deferred (ignore for now)
- Symptom: manual graph creation path initializes `interventionRuns: []`, but intervention state builder requires at least one baked/base run.
- References:
  - `spd/app/frontend/src/components/PromptAttributionsTab.svelte:519`
  - `spd/app/frontend/src/components/PromptAttributionsTab.svelte:526`
  - `spd/app/frontend/src/lib/interventionTypes.ts:105`

2. Intervention input text reconstructed from display tokens
- Severity: High
- Status: Open
- Symptom: draft forward path uses `activeCard.tokens.join("")`. These tokens are display spans with escaped control characters, so intervention input can differ from original prompt text/token IDs.
- Impact: user-triggered forward interventions can run on altered text.
- References:
  - `spd/app/frontend/src/components/PromptAttributionsTab.svelte:371`
  - `spd/app/backend/app_tokenizer.py:102`

3. Invalid node keys often fail as 500 via `assert` instead of 4xx validation
- Severity: Medium
- Status: Open
- Symptom: malformed/out-of-range node keys can surface as internal errors.
- References:
  - `spd/app/backend/compute.py:421`
  - `spd/app/backend/compute.py:428`
  - `spd/app/backend/compute.py:439`
  - `spd/app/backend/routers/intervention.py:119`

4. Clone action shown when active run is draft, but handler rejects cloning draft
- Severity: Medium
- Status: Open
- Symptom: UX exposes clone button in cases that throw runtime error.
- References:
  - `spd/app/frontend/src/components/prompt-attr/InterventionsView.svelte:1018`
  - `spd/app/frontend/src/components/PromptAttributionsTab.svelte:427`

5. Manual graph get-or-create is not idempotent for intervention runs
- Severity: Medium
- Status: Open
- Symptom: manual graph save may return existing graph ID, but base intervention run save still executes, creating duplicates over repeated requests.
- References:
  - `spd/app/backend/database.py:533`
  - `spd/app/backend/routers/graphs.py:616`

6. Subgraph generation always uses standard compute path
- Severity: Medium
- Status: Open (subgraph-related)
- Symptom: generated subgraph from selection does not preserve optimized-graph semantics/metadata.
- References:
  - `spd/app/frontend/src/components/PromptAttributionsTab.svelte:505`
  - `spd/app/backend/routers/graphs.py:585`
  - `spd/app/backend/routers/intervention.py:250`

7. Frontend `EdgeData` type omits backend `is_cross_seq`
- Severity: Low
- Status: Open
- Symptom: type drift between frontend and backend API contracts.
- References:
  - `spd/app/frontend/src/lib/promptAttributionsTypes.ts:13`
  - `spd/app/backend/routers/graphs.py:154`

8. Inconsistent CI-threshold validation across graph endpoints
- Severity: Low
- Status: Open
- Symptom: `POST /api/graphs` accepts broader `ci_threshold` values than `GET /api/graphs/{prompt_id}`.
- References:
  - `spd/app/backend/routers/graphs.py:533`
  - `spd/app/backend/routers/graphs.py:1074`

## Testing Gaps

- Current app API tests cover graph streaming lightly but do not cover intervention endpoints or manual-subgraph edge cases.
- Reference: `tests/app/test_server_api.py`
