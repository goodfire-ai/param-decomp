# Track-2 ledger — lives in lore (`pd-lore`)

The experiment ledger and cross-agent memory (specs, results, killed dead-ends) live in
the team's PD knowledge base **`pd-lore`**, not in this repo. That gives parallel agents on
separate worktrees/nodes one coherent, append-only, git-backed ledger — the thing flat
in-repo files lose under worktree-per-experiment. The **ruler** stays in-repo:
[`baselines.md`](baselines.md) (locked baselines) + `../../param_decomp_lab/speedup/quality_bundle.py`.

## Access (hosted MCP — no clone needed)

`pd-lore` is served over the tailnet. It's registered in the repo's
[`.mcp.json`](../../.mcp.json) as `lore-pd`. **One value must be filled in:** replace
`REPLACE_WITH_TAILNET` in that file with the real tailnet name (the full URL is
`http://pd-lore-mcp.<tailnet>.ts.net:8765/mcp`). Then the `mcp__lore-pd__kb_*` tools are
available to every agent session in this repo.

## Workflow (per experiment)

1. `kb_search` / `kb_list` first — has this idea (or a near neighbor) already been tried or
   killed? Don't re-run a recorded dead end.
2. `kb_append` a spec doc, body = [`TEMPLATE.md`](TEMPLATE.md): hypothesis, claim type,
   diff vs baseline, success/kill thresholds (quoted from `../README.md`), artifact links.
   Tag it so Track-2 docs are filterable (e.g. a `track2` / `spd-speedup` marker in the doc).
3. `kb_link` it to related docs with a typed edge: `builds-on`, `supersedes`,
   `killed-by`, `contradicts`, `references`.
4. As the stage changes (T0 → T1 → merged/killed/parked), update the doc; a clean
   "tried X, here's the run, it regressed Y" with artifact links is a first-class result.

The human reads the ledger through lore (`kb_list` / a curated index doc) — that, plus
`baselines.md`, is the whole report surface.
