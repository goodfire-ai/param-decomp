# NITS — tidy-up backlog (do NOT let these block the full-model decomp launch)

Full-model decomp is the priority. These are real-but-non-blocking tidiness items found in
the `feature/jax` unification review (2026-06-26). Knock them out after / alongside the
launch. Each is `file:line — what — fix`. Check off as done.

## TOP PERF ITEM — the batch-size ceiling (root cause of the b128 OOM)
- [ ] **Scan-over-layers retains the per-layer MLP-intermediate stack** (`param_decomp/targets/llama8b.py`
      masked-forward scan + `param_decomp/train.py` recon path). GROUND TRUTH from the b128 dump
      (`runs/p-efbc1750/hlo/*jit_step*buffer-assignment*`): the dominant buffers are
      `bf16[32, 8, 512, 14336]` = `[n_layers, per_DP_batch, seq, d_ff]` @ 3.5 GiB each, ~20 live ≈ 72 GiB
      (XLA `while.2249` phi + `loop_dynamic_update_slice` = each layer's MLP activation written into a
      preallocated [32, …] layer-stack). NOT the vocab logits (forensics guess) and NOT the CI gather
      (pre-mortem guess) — it's d_ff=14336, stacked over all 32 layers, scaling with **per-DP batch**.
      So `remat_recon_forwards` isn't actually discarding the per-layer MLP intermediates — the scan
      stacks them (its `ys`) instead of carrying only the residual + recomputing in the backward.
      FIX (unlocks large batch): make the masked-forward scan NOT stack/return per-layer d_ff activations
      (carry residual only; remat recomputes). HIGH IMPACT — caps batch size; b32/dp128/tp4 only *dodges*
      it (per-DP 1 → 8× smaller stack). Open: dig into why per-layer remat doesn't elide the stack.

## Understand / learn
- [ ] **Get to the bottom of exactly how Megatron row×col sharding works** — the precise
      mechanics: up-proj column-parallel → `d_ff/tp`-sharded hidden; down-proj row-parallel →
      all-reduce → replicated `d_model` residual. Which activations are sharded vs replicated,
      where the collectives land, and how it maps onto our CI fn (Megatron'd) vs target MLP
      (FSDP-only, hidden replicated). Directly informs lever 2 (Megatron the target MLP).

## Open design questions (Oli not fully satisfied)
- [ ] **Seq-width `(n, n+1)` acceptance** (`param_decomp/data.py`). Today we accept rows of
      width `seq_len` OR `seq_len+1` and truncate to `seq_len` (fineweb=512, pile=513=512+label).
      Works, but Oli's not fully satisfied — consider declaring an exact expected width per
      dataset (and the label convention explicitly) rather than accept-both-and-truncate.
- [ ] **Loss structure: flat polymorphic list vs record** (core + the two seams). Core should
      NOT have a first-class "groups of loss types" concept; we're in the flexible-loss regime
      (DI'd polymorphic losses, not config tailored per new loss type). But the current
      flat-tuple-then-`isinstance`-scan (faith/imp pulled out specially in `train.py`) is a
      middle ground. Decide between (a) truly flat polymorphic `LossTerm` list (uniform
      `.compute`, each term owns its schedule/phase) vs (b) a Record/product
      (`{faith, imp, recon: [...]}`, named roles, no scan). Discuss the USER-FACING CONFIG seam
      separately from the LAB→CORE seam — they can differ. (See main's torch history for a
      prior list↔record transition.)

## Config shape / placement
- [ ] **`remat_ci_fn` / `remat_recon_forwards` placement** (`param_decomp/configs.py` `RuntimeConfig`).
      Defensible (runtime = compute substrate), but `runtime:` is a grab-bag now (`dp`, `tp`,
      `autocast_bf16`, stale `device`, two `remat_*` bools). The two remat flags are one
      "rematerialization policy" — consider a `remat:` sub-config (or a `perf`/`memory`
      section), and prune dead fields. Decide the canonical home and migrate configs.
- [ ] **`remat_ci_fn` default is `False`** (`configs.py` ~L648). Now that it's threaded
      (fixed in c204b6afc), the default-false is still a footgun: a new full-model config that
      omits it silently gets the ~80-min-compile + OOM path. Options: default `True`, require
      it explicitly (no default), or a model-validator that forces it for large decompositions
      (site-count / total-C threshold).
- [ ] **Stale no-op fields in run yamls** (`configs/llama8b_full32L_seq512_b128_dp128.yaml`):
      `n_mask_samples`, `sampling`, `autocast_bf16`, `device` are stripped by back-compat
      validators but mislead readers. Remove from the configs (and confirm whether
      `autocast_bf16`/`device` are truly dead on the JAX path before deleting the schema fields).

## Stale comments / drift
- [x] **`launch.py` module docstring drift** — fixed (aecf861cb): "one srun task per node
      claiming all 8 GPUs."
- [~] **`lm.py` `recon_loss_fn` docstring arg order** — NON-ISSUE: signature and class doc both
      already say `(masked_output, clean_output)`. (CLAUDE.md's phrasing is the stale one.)

## Temporary hacks riding the branch
- [x] **Revert the `seq_len` TEMP HACK** (`param_decomp/data.py`) — done (aecf861cb): restored
      strict `in (seq_len, seq_len+1)`. The +1 is the real label-token convention (fineweb=512,
      pile=513=512+label; both truncate to seq_len) — NOT slop.

## Fragility (safe today, brittle)
- [x] **`train.py` loss-term extraction** — KEEP the terse `(faith_term,) = (...)` tuple-unpack
      (it already fails-fast on zero/multiple). The list+`[0]` "fix" was grosser; reverted.
      Deeper option if we ever want it: `build_loss_terms` returns a structured
      `(faith, imp, recon_terms)` instead of a flat tuple, so train.py never isinstance-scans.

## Provenance / housekeeping
- [ ] Local `feature/jax` was 1 behind origin's `d61db7282` (NCCL log-flood fix) at unification
      time — merged in (98fbb5b51). Keep an eye that future origin commits get merged, not
      cherry-picked (cherry-pick → duplicate-SHA divergence → force-push territory).
