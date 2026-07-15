# Placement rules — the ab-initio sharding design

Status: DRAFT (Oli + Claude, 2026-07-15). First increment: `placement.py` (the rules
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
3. **Rules table** — per placement SITE (`role/phase`: `params/persist`,
   `params/forward`, `optim/muon.ns`, `activations`), a mapping
   `semantic axis → mesh axes`. A tensor's PartitionSpec anywhere is *derived*:
   look up its dim names in the active site's rule; unlisted names replicate.

Phase boundaries (persist→forward at ENTRY; persist→optimizer at the update) are the
only reshard points, derived mechanically as the diff between two table rows and priced
by `transition_bytes` — the startup lint prints the whole policy + per-tensor audit
(`describe`), which is also the documentation.

Every historical layout is one table: **zero1** (intra-matrix ÷N), **owner** (stack
÷replicate, d ÷fsdp — PR #989 / SPEC D4 amendment), **ddp** (all replicated), and
muon's `impl: optax|stacked` distinction dissolves into whether `optim/muon.ns` matches
`params/persist` (owner-resident: reshard derives to zero) or spreads the stack axis
(transient: two reshards, priced).

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
  Conditionals become *site choices in consumer code* (e.g. the owner preset's
  `params/persist.subset` fallback for stacks that don't tile `replicate`); true
  one-offs get a literal-spec override on a named site.
- Pin only load-bearing surfaces (persist trees, phase entries, the waist); GSPMD
  propagates between pins, as today.
- Cost model is an honest upper bound (spec differs ⇒ ≤ full tensor bytes), refined
  only if a profile says the bound misleads.


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
   PR #927). ⇒ `transition_bytes` is a v0; rule tuples like `[node, device]` are
   ordered and the order must round-trip exactly.
5. **Some placement problems are program-POSITION problems** (shard_map for a post-scan
   psum; jit `out_shardings` at init) that no static table expresses. ⇒ named
   out-of-scope escape hatches, not rule-language growth — the no-conditionals guardrail
   stays.
6. Also: keep tp re-testable as a table edit (every "TP loses" verdict failed on a
   different axis than comms; do not structurally close the door), and pin the PRODUCER
   once (per-consumer resharding was a recurring pathology).

## Prior art

t5x/flax "logical axis rules" (params carry semantic names; first-match rules map
logical→mesh) — proven at scale; this design adds the two things t5x never needed:
phase-dependence and optimizer constraint declarations. PyTorch DTensor/DeviceMesh is
converging on the same vocabulary.

## Migration (staged, each lands green)

1. `placement.py` engine + presets + tests (THIS increment).
2. `RuntimeConfig.sharding: preset-name | rules-table`; thread `PlacementRules` into
   `run.py` → `init_train_state`; `DecompVU.shardings` / CI-fn `.shardings` consume
   `rules.sharding_for(site, axes)` instead of hardcoding specs (their current bodies
   become the `owner`/`zero1` preset rows). Startup prints `describe(...)` with every
   persistent tensor.
3. Reroute the activation pins (`site_out`, train step) through `activations` rules.
4. Muon: drop `impl:`, derive resident-vs-spread NS from the table; declare muon's
   locality constraint; wire the constraint checker.
5. `sharding: auto` (search over tables against the lint + a short probe) — optional,
   later, possibly agent-driven.
