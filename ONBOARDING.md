# Onboarding for the VPD attention / clustering-drift work

Handoff for the next Claude. This is a snapshot of session-specific state — what we
discovered while getting the app running for Jose and starting Experiment 1 (drift analysis
of the merge loop). Read this first; then read `CLAUDE.md`, `spd/clustering/CLAUDE.md`,
`spd/app/CLAUDE.md` for the underlying repo conventions.

## TL;DR

- **Branch:** `feature/attn-win` (off `dev`, 2 commits ahead, pushed to origin).
- **Subject SPD run:** `goodfire/spd/s-55ea3f9b` aka "Jose" — 4-layer 67M Pile transformer,
  used for all the VPD-paper attention analyses.
- **App is up** at `http://100.109.239.81:10004/` (tailnet; refresh tunnel with `tunnel-url 5173 -t 8`).
- **Drift analysis** for the merge loop (Exp 1) is running in the background as
  `c-drift-jose-3k`; report lands in `~/.spd_jose_override/clustering/runs/c-drift-jose-3k/drift/`.
- **Avoid touching shared NFS** under `mechanisms/param-decomp/` — everything we needed to
  fix, we fixed in `~/.spd_jose_override/` (a local symlink farm used as `SPD_OUT_DIR`).

## Two big renames you'll trip over

The team did two cosmetic renames that the codebase hasn't fully caught up with:

1. **GitHub repo:** `goodfire-ai/spd` → `goodfire-ai/param-decomp`. The old URL still
   redirects, so `git push origin` works; you'll see a `remote: This repository moved` notice.
   No action needed unless you want to update the URL: `git remote set-url origin git@github.com:goodfire-ai/param-decomp.git`.

2. **Artifact directory on NFS:** `/mnt/polished-lake/artifacts/mechanisms/spd/` →
   `/mnt/polished-lake/artifacts/mechanisms/param-decomp/`. But:
   - `spd/settings.py:10` hardcodes `CLUSTER_BASE_PATH = .../mechanisms/spd` (broken).
   - `spd/app/frontend/src/lib/registry.ts` had four cluster mapping paths under
     `mechanisms/spd/clustering/runs/...` (we fixed these on `feature/attn-win`).
   - `scripts/blog/constants.py:15` still points at the old path (we didn't fix this).
   - A broken NFS symlink at `mechanisms/param-decomp/runs/spd-s-55ea3f9b` →
     `mechanisms/spd/spd/s-55ea3f9b` (orphaned). Owned by `oli`; we left it alone.

The package on disk is still `spd/` and the Python package name is still `spd` in
`pyproject.toml`. Only the GitHub remote and the artifact dir were renamed.

## Canonical run / artifact IDs for Jose

All paths are under `/mnt/polished-lake/artifacts/mechanisms/param-decomp/` (not the
stale `mechanisms/spd/` one).

| Asset | ID | Path |
|---|---|---|
| SPD (VPD) decomposition | `s-55ea3f9b` | `spd/s-55ea3f9b/` (final_config.yaml + model_400000.pth) |
| Pretrained target model | `t-9d2b8f02` | wandb only; 4-layer 67M Pile transformer |
| Clustering harvest snapshot | `ch-2b9414a4` | `clustering/harvests/ch-2b9414a4/` (10M samples × 10007 alive components, alpha=10) |
| Clustering run (**paper canonical**) | `c-651d85c4` | `clustering/runs/c-651d85c4/` — cut at **iteration 4423**, merge_config alpha=10 sampling=exp_rank decay=0.8 |
| Harvest (**big, 38k components**) | `h-20260227_010249` | `harvest/s-55ea3f9b/h-20260227_010249/` (51 GB) |
| Harvest (smaller, 10k) | `h-20260318_223737`, `h-20260319_121635` | ~14 GB each |
| Autointerp (**big, 38k labels**) | `a-20260227_134542` | `autointerp/s-55ea3f9b/a-20260227_134542/` |
| Autointerp (**canon paper-era, 10k**) | `a-20260323_123740_605511-canon-v6b-medium-10k` | name signals intent |
| Graph interp | `ti-20260323_154443-canon-flash-low` | name signals intent |
| App SQLite DB | n/a | `app/prompt_attr.db` (3.3 GB, NFS-shared) |

Slack: `#parameter-decomposition` (C08N7E5KNG7) for tech discussion, `#content-vpd` for
paper-release coordination.

## The stub-May-4 trap (important)

Three loaders pick the **lexicographically latest** subrun under their respective directories:

- `HarvestRepo.open_most_recent` (`spd/harvest/repo.py:54`) — no `.done` filter
- `InterpRepo.open` (`spd/autointerp/repo.py:48`) — `.done`-filtered
- `GraphInterpRepo.open` (`spd/graph_interp/repo.py:30`) — `.done`-filtered

In all three, there are **near-empty May 4 subruns that lex-sort last** (likely test/smoke
runs that completed with `.done` markers but produced 0–13 rows). If you point the loaders
at the real shared-NFS dir, you'll get the stubs. Concrete row counts I verified:

| Subrun | Real rows | What it is |
|---|---|---|
| `autointerp/.../a-20260504_191444_447537` | 3 interpretations | stub, lex-max |
| `autointerp/.../a-20260504_171306_190268` | 0 | stub |
| `autointerp/.../a-20260504_164829_522054` | 0 | stub |
| `harvest/.../h-20260504_165735` | 13k components, **0 scores** | stub |
| `harvest/.../h-20260227_010249` | 38,681 components | **canonical** |
| `autointerp/.../a-20260227_134542` | 38,678 | **canonical big** |
| `autointerp/.../a-20260323_..._canon-v6b-medium-10k` | 10,000 | **canonical paper-era** |

This is worth pinging Oli about — the stubs should probably be deleted or the loaders should
filter by row count. We didn't touch shared state.

## Local override: `~/.spd_jose_override/`

Used as `SPD_OUT_DIR` to mask the shared-NFS quirks. Layout:

```
~/.spd_jose_override/
├── app                              -> .../param-decomp/app                  (whole-tree symlink, no stubs)
├── clustering                       -> .../param-decomp/clustering            (whole-tree symlink)
├── dataset_attributions             -> .../param-decomp/dataset_attributions  (whole-tree symlink)
├── graph_interp/s-55ea3f9b/
│   └── ti-20260323_canon            -> .../param-decomp/graph_interp/s-55ea3f9b/ti-20260323_154443-canon-flash-low
├── harvest/s-55ea3f9b/
│   └── h-20260227_canon             -> .../param-decomp/harvest/s-55ea3f9b/h-20260227_010249
├── autointerp/s-55ea3f9b/
│   └── a-20260227_canon             -> .../param-decomp/autointerp/s-55ea3f9b/a-20260227_134542
└── runs/spd-s-55ea3f9b              -> .../param-decomp/spd/s-55ea3f9b        (fixes the broken oli symlink)
```

Each curated single-subrun directory forces the lex-max loaders to pick the canonical
subrun. The whole-tree symlinks (clustering, app, dataset_attributions) don't have the
stub problem so we passed them through.

The `runs/spd-s-55ea3f9b` link is essential: without it, `SPDRunInfo.from_path` falls back
to downloading from wandb, and the wandb-saved config has fields (`reader_hidden_dims`,
`d_resid_ci_fn`, `block_groups`, `transition_attn_config`, `transition_hidden_dim`) that
the current `dev` schema rejects (`extra="forbid"`). Commit `f869a6d5 Delete
GlobalReverseResidualCiFn CI function` (2026-04-23) is what removed those fields. The
on-disk `final_config.yaml` is slim and validates fine.

## Running the app

Backend and frontend both bound to `0.0.0.0` (frontend MUST be `--host 0.0.0.0` or the
tunnel-broker liveness probe tears the tunnel down after ~3 minutes — Opus debugged this
earlier; he found the broker runs a 60s sweep that does `bash -c "echo
>/dev/tcp/$compute_node/$compute_port"` from the login node).

```bash
# Kill anything stale, then launch backend + frontend separately so each survives shell exit.
cd ~/spd && source .venv/bin/activate
nohup env SPD_OUT_DIR=$HOME/.spd_jose_override uv run python spd/app/backend/server.py > /tmp/spd_backend.log 2>&1 &
disown
cd ~/spd/spd/app/frontend
nohup env BACKEND_URL=http://localhost:8000 npm run dev -- --port 5173 --strictPort --host 0.0.0.0 > /tmp/spd_frontend.log 2>&1 &
disown

# Once both are responding locally:
tunnel-url 5173 -t 8     # -> prints http://100.109.239.81:<port>/
```

Note: my Claude session moved compute nodes once (`h200-reserved-145-037` → `h200-dev-145-045`)
— if the tunnel goes dead and the URL is unreachable, that's the most likely cause. Just
re-request via `tunnel-url --stop 5173 && tunnel-url 5173 -t 8`.

In the app UI: select **Jose** (the wandb path is preset), then in the cluster mapping
dropdown the first option is now **"VPD paper canonical"** (= `c-651d85c4`) which we added
in commit `9344b8f1` on `feature/attn-win`.

## Branches you might care about

| Branch | Where | What |
|---|---|---|
| `main` | 2026-01-15 | Stale snapshot from open-source SPD release. Don't base off this. |
| `dev` | 2026-05-04 | Active branch. 743 commits ahead of `main`. All recent attention work is here. |
| `feature/attn-win` | local + origin | What we're on. `dev` + registry edit + drift analysis instrumentation. |
| `origin/spd-paper` | frozen | Original SPD paper code (the 2025 arxiv 2506.20790 release). Pre-VPD. |
| `feature/attn-analysis-guide` | local | Has `spd/docs/attention_behavior_analysis_guide.md` + per-layer writeups + extensive SIS/PKV/OV-overlap scripts. **Not merged to `dev`.** Lee said not to worry about it for now. |
| `origin/clustering-app` | unmerged | Has dendrogram + force graph + correlation matrix UI for clusters; relevant if you implement the "pick a cut from the dendrogram" follow-up. |

## What this session built (on `feature/attn-win`)

Commit `9344b8f1` "Add c-651d85c4 (VPD paper canonical) to Jose registry entry":
- `spd/app/frontend/src/lib/registry.ts`: added `c-651d85c4` as Jose's first cluster
  mapping option, and corrected the four existing paths from `mechanisms/spd/...` to
  `mechanisms/param-decomp/...`.

Commit `01b6ea96` "Drift analysis instrumentation for the merge loop (Experiment 1)":
- `spd/clustering/drift_analysis.py` — `DriftAnalysisCallback` (LogCallback implementation)
  + `apply_analytical_iter_update` (closed-form per-iter delta for the MDL cost's
  log-`k` terms; rank term T3 is invariant for unrelated pairs).
- `spd/clustering/scripts/run_drift_analysis.py` — CLI runner. Loads a snapshot, attaches
  the callback, calls `merge_iteration_memberships`.
- `spd/clustering/scripts/analyze_drift.py` — offline summarizer; reads the dumps, makes
  three plots, writes `drift_report.md`.
- `tests/clustering/test_drift_analytical_update.py` — 13 tests: 12 parametrized cases on
  synthetic data validating the analytical formula vs `compute_merge_costs`; 1 end-to-end
  callback test. All pass; pre-commit + basedpyright clean.

## Experiment 1 (drift analysis) — what it tests

The merge loop in `spd/clustering/merge.py:132–137` calls `compute_merge_costs` on the
full `k×k` cost matrix every iteration. The MDL cost decomposes:

```
F(i,j) = (s_Σ − s_i − s_j) · log₂((k−1)/k)              # T1
       + s_{ij}·log₂(k−1) − (s_i + s_j)·log₂(k)         # T2
       + α·(s_{ij}·r(P_{ij}) − s_i·r(P_i) − s_j·r(P_j)) # T3
```

After a merge of `(a,b)`, for unrelated `(i,j)`: T3 is **exactly invariant**; T1 and T2
drift `O(1/k)` per merge but admit a closed-form per-iter delta. So if drift stays bounded
and predictable, the full recompute is wasteful — we can do an analytical update (exact)
or a lazy priority queue (also exact).

The drift run dumps:
- **trackers**: 5000 randomly-sampled component-pair costs over time
- **top-tail**: cheapest 200 group-pairs (with min-orig-index identity for cross-iter matching)
- **scalars**: `k`, `s_Σ`, selected merge pair, per-iter MDL loss
- **analytical residuals**: every 50 iters, 500 random unrelated pairs, |predicted −
  actual| stats. Synthetic data confirmed residuals at **machine epsilon (~2.85e-7)**.

The user asked to skip the full-matrix snapshots and validate inline. The output dir is
~85 MB total (well under our 3 GB plan budget).

Recommendation flow in `analyze_drift.py`:
- If analytical residuals stay below 1e-3 → recommend implementing the analytical update.
- Else, if top-K stability mean > 0.7 → recommend a lazy priority queue.
- Else → revisit assumptions.

## Running drift analysis from scratch

```bash
cd ~/spd && source .venv/bin/activate

# Unit tests first
python -m pytest tests/clustering/test_drift_analytical_update.py -v

# Real run on Jose
SPD_OUT_DIR=$HOME/.spd_jose_override python -m spd.clustering.scripts.run_drift_analysis \
    --snapshot_path /mnt/polished-lake/artifacts/mechanisms/param-decomp/clustering/harvests/ch-2b9414a4 \
    --max_iters 3000 \
    --alpha 10.0 \
    --sampling exp_rank \
    --sampling_decay 0.8 \
    --n_track 5000 \
    --n_top 200 \
    --validate_every 50 \
    --n_validate_sample_pairs 500 \
    --run_id c-drift-jose-3k

# Then
python -m spd.clustering.scripts.analyze_drift \
    --dump_dir $HOME/.spd_jose_override/clustering/runs/c-drift-jose-3k/drift
```

Wall-clock for the 3k-iter run: **~1.5–2 hours on CPU**. Initial coactivation
(10007² × 4 bytes from a 10 M × 10007 CSR^T @ CSR) takes ~5 min; merge iters then go from
~7 s/iter at k=10007 down to under 1 s/iter as k shrinks. RAM peak ~45 GB (the node has
2 TB so this is fine).

A run is in flight as I write this (`c-drift-jose-3k`); should be done within ~1.5h of
2026-05-26 17:25 BST. Check progress with `tail /tmp/drift_run.log` or `ps -p 2133554`.

## Open threads (not started, just noted)

1. **App cluster cut-picker UI** — the plan was at `~/.claude/plans/okay-so-the-cluster-refactored-frog.md`
   v1 (before Lee pivoted to drift analysis). Idea: extend `ClusterPathInput.svelte` so you
   can pick a clustering-run dir + iteration via a slider, and the backend extracts the cut
   live from `history.zip` using `spd/clustering/scripts/get_cluster_mapping.py:get_cluster_mapping`.
   The full picture is in the plan file.

2. **Exp 2/3 — implement the speedup** — once the drift report confirms residuals are at
   machine epsilon, the next step is to actually replace the per-iter
   `compute_merge_costs(...)` in `merge.py:132–137` with `apply_analytical_iter_update(...)`
   plus a fresh recompute of the merged-group row/col. This should cut merge wall-clock
   roughly proportionally to how cheap the analytical update is vs the full recompute.

3. **`spd/app/CLAUDE.md` doesn't mention the new clustering endpoints** that the cut-picker
   plan would add. Update that doc when the picker lands.

4. **Layer 1 attention analysis writeup** is on `feature/attn-analysis-guide`; Lee said
   not to worry about it for this session. If you continue VPD attention work later, check
   memory note `reference-vpd-attn-pipeline` for the five-stage pipeline (Static SIS →
   component characterization → dynamic multiprompt → KV coactivation + OV overlap → write-up).

## Files to read in order

1. `spd/CLAUDE.md` — repo-wide conventions (research code, fail-fast, narrow types).
2. `spd/clustering/CLAUDE.md` — clustering pipeline overview.
3. `spd/app/CLAUDE.md` — app architecture (backend + frontend).
4. `spd/clustering/merge.py` (~210 lines) — the merge loop we're instrumenting.
5. `spd/clustering/compute_costs.py` — the MDL cost (T1/T2/T3 derivation lives here).
6. `spd/clustering/drift_analysis.py` — our new instrumentation.
7. `~/.claude/plans/okay-so-the-cluster-refactored-frog.md` — Experiment 1 plan (also
   contains the deferred cut-picker plan).
