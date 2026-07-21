# Placement rules — the ab-initio sharding design

Status: DRAFT (Oli + Claude, 2026-07-15; externally reviewed 2026-07-16 — fresh-Claude
verdict adopt-with-changes, fixes applied; Codex review pending). First increment: `placement.py` (the rules
engine + presets + tests). Not yet wired into the trainer — see "Migration" below.
History check (2026-07-15 archaeology pass over git + lore): DONE — see "Lessons from
history" below.

## Problem

Layout policy today is smeared across the codebase (spec pins in `site_out`, the
reconstruct fns, per-class `.shardings` methods), written in *mesh* vocabulary, with no
notion of *phase* — the same tensor legitimately wants different layouts at rest, in the
forward, and in the optimizer. Retrofit costs of that structure, observed: the muon NS
pathology (2.1×), the transient-stacking workaround, the owner-persistence refactor
(PR #989), and a per-optimizer `impl:` knob that is secretly a placement property.

## Design

Three vocabularies; the config owns only the mapping between them:

1. **Semantic axes** — dimension names declared once by the owning code:
   V/U stacks are `("stack", "d_in", "C")` / `("stack", "C", "d_out")`; the activation
   waist is `("batch", *positions, "d"|"C")`. Optimizer state inherits its param's names
   by tree structure (momentum of a V stack is `(stack, d_in, C)`) — optimizer sharding
   stops being a separate problem.
2. **Mesh axes** — the physical grid the run config declares.
3. **Rules table** — one `semantic axis → mesh axes` rule per placement ROW. The rows
   are TYPED FIELDS of `placement.PlacementRules` (`params.persist`, `params.zero1`,
   `params.forward`, `activations`; a future `optim/muon.ns` phase is a future field),
   never string keys — consumers reference fields, labels like `params/persist` are
   print-only. A tensor's PartitionSpec anywhere is *derived*: look up its dim names in
   the active row's rule; unlisted names replicate.

Phase boundaries (persist→forward at ENTRY; persist→optimizer at the update) are the
only reshard points, derived mechanically as the diff between two table rows and priced
by `transition_bytes` — the startup lint prints the whole policy + per-tensor audit
(`describe`), which is also the documentation.

Every historical layout is one table: **zero1** (intra-matrix ÷N), **owner** (stack
÷replicate, d ÷fsdp — PR #989 / SPEC D4 amendment), **ddp** (all replicated), and
muon's NS *layout* can be derived from an `optim/muon.ns` row (owner-resident: reshard
derives to zero; spread: two priced reshards). CORRECTED after external review: the
`impl:` knob does NOT fully dissolve — `stacked` is also an OP RESTRUCTURING
(concat same-canonical-shape leaves + pad + one batched NS), which a placement table
cannot express. The table owns where; the impl owns how.

### Constraints, not choices, from code

Models/optimizers declare *requirements*, never layouts: muon declares "each (a, b)
matrix unit device-local during `ns`"; adam "elementwise, unconstrained"; the scan
"consumes stack-leading trees"; the recon backward "batch stays sharded" (today's
OOM-avoidance pin, stated as a constraint). The framework validates a proposed table
against all declared constraints at startup (loud), and prices it (lint). Choosing the
table is then a human/agent decision against a checkable, costable artifact —
`sharding: auto` becomes a legitimate search problem; `zero1|owner|ddp` are preset
tables for everyone else.

### Deliberate weaknesses (the guardrails)

- Rule language: name → mesh axes, first-match, **no conditionals or expressions**.
  Conditionals become *row choices resolved at rules construction* (the `owner+zero1`
  preset's opt-in `params.zero1` row for stacks that don't tile `replicate` — strict
  `owner` errors on those instead; see "Decision at build time" below); true one-offs
  get a literal-spec override on a named row.
- Pin only load-bearing surfaces (persist trees, phase entries, the waist); GSPMD
  propagates between pins, as today.
- Cost model is an honest upper bound PER MATERIALIZATION (spec differs ⇒ ≤ full tensor
  bytes). Known-unmodeled: materialization multiplicity (recon-grid forwards, remat
  replays), residency-vs-regather (the real 8B memory trade), op-count, axis locality.
  Sufficient to rank resting layouts; NOT sufficient to drive `sharding: auto` — fixing
  the unit precedes any auto-search.
- Out-of-table placement policy that stays out (named, not smuggled): `ascend_replicate`
  and `gather_fp8` (RuntimeConfig knobs — an adversary-ascent phase and a layout×dtype
  transform), and program-position choices (shard_map, init out_shardings).


## Lessons from history (git + lore archaeology, 2026-07-15)

Nothing like this rules table was tried before. What WAS tried, and what it teaches:

1. **Torch-era declarative topology** (`World`/`TwoPoolLayout`, the 3-pool typed DAG —
   ~9k LOC): fine abstraction, deleted wholesale because it hand-rolled what GSPMD gives
   free (lore `2026-06-09--jax-vs-torch-codebase-portability-decision`: "the gnarliest,
   most bug-prone code in the repo"). ⇒ this layer's value must sit strictly ABOVE
   GSPMD (policy), never parallel to it (transport).
2. **The one prior generic rule** — inferred `shardable` + "shard the last axis" + silent
   replicate fallback (PR #883) — lived TWO DAYS before model-owned explicit placement
   replaced it (PR #891): shape inference couldn't express real layouts (Megatron
   col/row per-field) and the silent fallback was a memory landmine. ⇒ declarations
   beat inference; fail loud; and our "unlisted axis ⇒ replicated" default is only
   acceptable because `describe()` audits it at startup — the lint MUST flag
   large-tensor replication, never let it be quiet at scale.
3. **Phase is real and REMAT RE-DECIDES it** (the 2026-06-26 tp OOM: the backward
   replays `site_out` under remat and re-chooses layouts). ⇒ pins must live where remat
   replays them; forward-correct is not backward-correct; the table's phase dimension is
   the load-bearing novelty.
4. **The cost model must price op-COUNT and axis LOCALITY, not just bytes** (tp2 moved
   fewer bytes and lost 63% on op-count — lore `2026-07-01--tp-loses-...`), and
   **nested-axis ORDER inside one assignment is semantics** (fsdp-major vs
   replicate-major linearization cost ~13 GiB/rank/step of collective-permutes,
   PR #927 — discovered 2026-07-16 to have NEVER MERGED; its fsdp-major flip is now
   integrated here as `placement._ZERO1_DATA` + the CI-fn `.shardings`, and
   `tools/xplane_attr.py` came with it). ⇒ `transition_bytes` is a v0; rule tuples like `[node, device]` are
   ordered and the order must round-trip exactly.
5. **Some placement problems are program-POSITION problems** (shard_map for a post-scan
   psum; jit `out_shardings` at init) that no static table expresses. ⇒ named
   out-of-scope escape hatches, not rule-language growth — the no-conditionals guardrail
   stays.
6. Also: keep tp re-testable as a table edit (every "TP loses" verdict failed on a
   different axis than comms; do not structurally close the door), and pin the PRODUCER
   once (per-consumer resharding was a recurring pathology).

## Validation (2026-07-16)

- Compile matrix (probe script since deleted; results recorded here): {owner, zero1, ddp} ×
  {adamw, muon-stacked} all compile at a real 2×2 (replicate, fsdp) sim mesh. Finding:
  `impl: stacked` under ddp persist does a redundant NS spread (210 collective-permutes)
  — the stage-4 derive-NS-layout-from-table fix, evidenced.
- José scale (4L, dp=8, real GPUs): `sharding: owner` 0.204 s/step vs `sharding: ddp`
  0.201 s/step (~1.5%) — DDP works via one config word; the 4L sharding tax was already
  negligible. Startup audit + large-replication flag verified in production logs.
- D4 invariance harness: rules-driven derivation byte-equivalent (rel 9.6e-6).
- 8B (full32L dp=32): BLOCKED by a pre-existing tip regression, NOT this layer — the
  savesmoke OOMs (~48 GiB single alloc, empty op name) identically on owner+stacked,
  zero1+stacked, and the pre-stack per-site baseline (board:
  full32l-savesmoke-oom-at-tip). The flagship probe re-runs when that's fixed.

## Decision at build time, data downward (2026-07-21, Oli-approved)

The per-shape-group persist-vs-zero1 choice is made ONCE, at `PlacementRules`
construction — `from_config(spec, mesh, sites)` resolves a TOTAL assignment
(`ParamsPlacement.groups: {VUShape: GroupPlacement}`) from the run's site set — and
flows downward as data. Construction happens where the resolved site set and the run's
topology first coexist: config build (`experiments.lm.config._assert_placement_claims`,
against the `sharding.hsdp_abstract_mesh` the config's `runtime.{dp,tp}` implies — so a
`dp: N` misconfiguration refuses at `pd-lm` submit, on the login node, before sbatch)
and the composition roots (the same `from_config` at the concrete mesh). Shape groups
derive from config + target dims alone — no weights, no devices.

**The receiving end validates; it never re-decides.** `placement._component_stacks_rows`
(behind `component_stacks_shardings` / `component_stacks_audit`) asserts the assignment's
shape groups and stack lengths exactly match the arrays actually held — validation of
received data at a trust boundary. There is deliberately NO second implementation of the
tiles-or-fallback branch anywhere below construction (`_assign_groups` is the only one).
A shared-function-called-twice "preview" (decide at build from config, re-decide at
runtime from concrete arrays through one shared function) was explicitly rejected: two
call sites feeding one function from different worlds makes the preview only as
trustworthy as their input-equality, and a future runtime-dependent placement input
would silently break it. Deciding once and validating the data at the boundary fails
loudly on any divergence instead.

**Disjoint claim semantics — a sharding spec is a bidirectional CLAIM, checked at
construction.** A table WITHOUT a `params.zero1` row (`owner`, or an explicit table
omitting it) claims every shape group tiles the persist stack sharding — a non-tiling
group is an error. A table WITH one (`owner+zero1`) claims at least one group does NOT
tile — if every group tiles at the declared topology, that is ALSO an error: the config
claims a run shape that isn't happening, and a declared-but-unreachable arm is a
misconfiguration (common trigger: lowering `dp` for a smoke — an inline single-device
smoke cannot exercise the owner+zero1 layout). `zero1` / `ddp` shard no stack axis, so
every group takes persist trivially and there is no claim. One exemption, enumerated by
name: `from_config_for_consumer` skips only the reachability claim, for consumers
re-placing a finished run's arrays on their own (typically single-device) mesh
(`open_jax_run`) — the fail-closed direction still holds there.

## Prior art

t5x/flax "logical axis rules" (params carry semantic names; a rules list maps
logical→mesh) — proven at scale. flax's rules are a context manager, so coarse
phase-dependence IS expressible there (rarely used); what it lacks is the priced
transitions + audit. We reimplement ~150 lines locally rather than dragging flax into
an equinox codebase. This design's additions: explicit sites, transition pricing, the
startup audit, optimizer constraint declarations (planned). PyTorch DTensor/DeviceMesh
converges on the same vocabulary.

## Migration (staged, each lands green)

1. `placement.py` engine + presets + tests (THIS increment).
2. `RuntimeConfig.sharding: preset-name | PlacementTableConfig` (the explicit table is
   a pydantic model mirroring the typed rows: `params: {persist, zero1?, forward}` +
   `activations`, closed vocabulary — unknown rows die at parse); thread
   `PlacementRules` into `run.py` → `init_train_state`; `ComponentStacks.shardings` /
   CI-fn `.shardings` consume `rules.params.persist.sharding_for(axes)` instead of
   hardcoding specs (their old bodies became the `owner`/`zero1` preset rows). Startup
   prints `describe(...)` with every persistent tensor.
3. Reroute the activation pins (`site_out`, train step) through `activations` rules.
4. Muon: drop `impl:`, derive resident-vs-spread NS from the table; declare muon's
   locality constraint; wire the constraint checker.
5. `sharding: auto` (search over tables against the lint + a short probe) — optional,
   later, possibly agent-driven.
