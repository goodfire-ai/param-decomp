# Investigation results: empirical validation of SPD scaling estimates

Companion to `scaling_investigation_plan.md` and `fsdp_scaling_report.html`.

**Cluster context:** runs are on 1×NVIDIA H200 (140 GB HBM), not the 1×H100 (80 GB)
the report assumes. Per-rank memory and activation/B comparisons to the report are
unaffected; max-batch comparisons skew larger on H200.

---

## Phase 1: baseline Jose profile

**Setup:** 1× H200, `pile_llama_simple_mlp-4L`, target `t-9d2b8f02`, batch=8 per rank
(matches what each rank sees in normal 8×DDP Jose), 50 training steps, snapshot at
step 30, faithfulness warmup off, eval at step 0 only with eval_batch_size=8.

Run ID `p-0ffcbe39`, SLURM 8031.

### 1.1 Param counts (measured)

| Bucket | Measured (params) | fp32 size | Report's estimate | Δ |
|---|---:|---:|---:|---|
| target (frozen) | 66.9 M | 0.27 GB | 242 M | **−72%** |
| components | 121.1 M | 0.48 GB | 323 M | **−63%** |
| CI fn | 539.1 M | 2.16 GB | 400 M | +35% |
| **trainable subtotal** | **660.2 M** | **2.64 GB** | 723 M | −9% |

**Why the per-category numbers were off:** the report assumed `d_model=2048`,
`d_mlp=8192`, `vocab=20000` for the Jose target. The actual `model_config.yaml`
of `t-9d2b8f02` has `n_embd=768`, `n_intermediate=3072`, `vocab_size=50277`
(and `n_layer=4`, `n_head=6` with GQA n_kv=n_head). Recomputing from the real
config: target = 4 × 7.08M + 38.6M embed = 66.9M, components = 121.3M — exact
match against the measurement. So the report had the right *structure* (layer
counts, module list) but wrong *dimensions*; the trainable total only worked
out by accident because the underestimated target+components were balanced by
a (relative) overestimate of the CI fn.

### 1.2 Memory at peak

| Slot | GB | Notes |
|---|---:|---|
| target params (fp32) | 0.27 | frozen, no grad |
| trainable params (fp32) | 2.64 | components + CI fn |
| trainable grads (fp32) | 2.64 | matches snapshot's `optim/adam.py` agg |
| AdamW state (m+v, fp32) | 5.28 | 2× trainable; matches `optim/adam.py: 5.31 GB` |
| **fixed state** | **10.83 GB** | (target + 4× trainable) |
| activations + transient | 23.79 GB | peak − fixed |
| **peak (max_memory_allocated)** | **34.62 GB** | |

The snapshot at step 30 (post-`optimizer.step`, between training steps) shows
17.75 GB live, 35.80 GB reserved. The 6.92 GB above fixed state at the step
boundary is largely persistent PPGD state (~1.91 GB), retained CI/sigmoid
tensors (~1.28 GB), retained linear-layer activations (~1.27 GB) and component
weight materialization (~0.64 GB). Peak (34.62) is reached *during* a forward
pass when the full activation graph is alive.

**Activation/B = (peak − fixed) / B = 23.79 / 8 ≈ 3.0 GB per per-rank batch
element**, in bf16 (autocast on). Report predicted ~1 GB/B. **3× higher than
predicted — biggest single delta in the investigation.**

A note on what's in activations: the report counted activations against per-step
forwards but used a single calibration anchor. The actual training loop runs
5–7 component forwards + 2–3 backwards per step, plus the PPGD warmup
(`n_warmup_steps=2`) which retains autograd graphs across multiple forwards. The
3 GB/B figure reflects the steady-state peak across all of these, in bf16.

### 1.3 Top live allocations at step 30

```
824 MB  llama_simple_mlp.py:366 (forward)        target activation
637 MB  components.py:46 (forward)               component intermediate (V@U or einsum)
411 MB  llama_simple_mlp.py:366 (forward)        target activation
14× 67 MB  optimize() bucket                     AdamW slot tensors (paired for m/v)
```

By source file (>50 MB):
```
optim/adam.py            5.31 GB   AdamW state
persistent_pgd.py        1.91 GB   PPGD sources/grads (per-batch-per-position scope)
models/sigmoids.py       1.28 GB   mask/CI tensors retained for backward
modules/linear.py        1.27 GB   linear-layer forward activations
models/components.py     0.64 GB   component forward intermediates
models/llama_simple_mlp.py  0.33 GB   target forward activations (frozen)
```

### 1.4 Step time

Steady-state training step time: **~305 ms** post-snapshot (after step 30 when
`_record_memory_history` is disabled), **~326 ms** while the snapshot is being
recorded — snapshot recording adds ~7% overhead. Step 0 (3.95 s) is dominated by
the step-0 eval; step 50 (7.55 s) is the final-step logging path. Both are
excluded from the steady-state figure.

The plan's "warmup_steps_skipped = 5" in `profile_summary.json` correctly drops
the early steps but does not exclude the final step, so the reported
`avg_step_time_ms_post_warmup = 475 ms` is biased upward by step 50's 7.5 s.
Real steady-state is ~305–326 ms. (Will fix the average in a follow-up; the
per-step list is the source of truth.)

### 1.5 Predictions vs measurements — summary

| Quantity | Predicted | Measured | Δ |
|---|---:|---:|---|
| Trainable param total | 723 M | 660 M | −9% |
| Fixed state per rank (DDP) | 13 GB | 10.83 GB | −17% |
| Activation per per-rank batch element | ~1 GB | **~3 GB** | **+200%** |
| Peak memory at b=8 (single rank) | ~21 GB | **34.6 GB** | **+65%** |

The fixed-state calc was tight; activation/B was significantly low. The report's
§3 max-batch table is therefore optimistic — at any given target scale, max
batch under each strategy is likely ~3× lower than tabulated. **The ZeRO-1 →
1B-target threshold from §3 (b≈13) is more like b≈4 in practice**, which makes
gradient checkpointing more critical than the report implied.

### 1.6 Recommended report updates after Phase 1

1. Replace the assumed Jose target dimensions in §2 (`d_model=2048`,
   `vocab=20000`, etc.) with the actual `n_embd=768`, `vocab=50277`,
   `n_intermediate=3072`. The breakdown table should then come out to target
   ~67M, components ~121M, CI fn ~539M.
2. Update §3's calibration anchor: at Jose scale, single-rank b=64 OOMs; with
   per-rank b=8 we see ~3 GB/B activation cost in bf16, not 1 GB/B.
3. §3's max-batch table should be regenerated with the new activation-per-B and
   fixed-state numbers.
4. The "1.7×" CI-fn-vs-target ratio in §2's "your 4× target intuition" callout
   should become ~8× (CI fn 539M vs target 67M), which strengthens the case
   for prioritizing CI-fn checkpointing in §7b.

---

## Phase 2: ZeRO-1 measurement

**Setup:** same per-rank conditions as Phase 1 (per-rank batch=8, all other knobs
unchanged) at world size N ∈ {2, 4, 8}, comparing `optimizer_strategy=adamw` vs
`zero_adamw`. Six runs, 8 H200s shared across them, profile each at step 30.

| N | global batch | DDP peak | ZeRO-1 peak | Δ measured | Δ predicted = (N−1)/N × 8T | match |
|---:|---:|---:|---:|---:|---:|:---:|
| 2 | 16 | 37.24 GB | 34.61 GB | 2.63 GB | 2.64 GB | ✅ |
| 4 | 32 | 37.25 GB | 33.27 GB | 3.98 GB | 3.96 GB | ✅ |
| 8 | 64 | _pending_ | _pending_ | _pending_ | 4.62 GB | _pending_ |

(`8T = 8 × trainable_params_GB = 8 × 0.66 = 5.28 GB`, the optimizer state Adam
keeps in fp32. ZeRO-1 shards only that state, not params or grads.)

The **ZeRO-1 saving formula `8·T·(N−1)/N` per rank from §3 of the report holds
to <1% at N∈{2,4}**. At Jose this is a modest win (~5 GB freed at N=8); at the
report's hypothetical 1B target the trainable bumps to ~3 B params and the
saving per rank scales to ~21 GB at N=8 — meaningful headroom.

The DDP peak is essentially flat across N=2/4 because per-rank conditions are
identical (only the global batch grows). ZeRO-1 reduces peak monotonically as
predicted.

The manual LR-schedule loop at `run_param_decomp.py:254–255` and `clip_grad_norm_`
both worked unchanged under ZeRO-1; no save path is exercised in the profiling
runs (`save_freq=null`), so `consolidate_state_dict` wasn't tested but doesn't
affect runtime memory.

---

## Phase 4: weight-delta rewrite

**Tests** (`tests/test_components.py`): 6 unit tests verify `forward_with_target_weight`
matches `forward(weight_delta_and_mask=...)` on output and on V/U gradients,
across with/without mask and with/without bias. All pass.

**Microbenchmark** (`scripts/bench_weight_delta_rewrite.py`): runs both paths
through every Jose-shaped decomposed module (24 modules: 6 per layer × 4 layers,
at the *measured* dims n_embd=768 etc.) on 1×H200, fp32, with backward,
measuring `torch.cuda.max_memory_allocated`.

| batch (S=512) | materialize peak | rewrite peak | Δ memory | Δ time |
|---:|---:|---:|---:|---:|
| 8 | 1.518 GB | 1.609 GB | **+91 MB (+6.0%)** | +15.6% |
| 16 | 1.872 GB | 2.064 GB | +192 MB (+10.2%) | +22.5% |
| 32 | 2.575 GB | 2.969 GB | +393 MB (+15.3%) | +26.6% |
| 64 | 3.988 GB | 4.784 GB | +796 MB (+20.0%) | +29.1% |

**The rewrite is worse on memory and time at every batch size, and the gap
*scales with batch*.** This contradicts the report's §7d "strict win" claim.

The reason is the autograd retention pattern, not the matmul count:

- The *materialize* path saves one `[batch, seq, d_out]` activation
  (`unmasked_delta_out`) plus the materialized `[d_out, d_in]` weight delta
  (a constant ~9 MB per Jose module).
- The *rewrite* path saves two `[batch, seq, d_out]` activations
  (`target_out` and `unmasked_recon_out`), needed to backprop through
  `target_out − unmasked_recon_out`.

So per module, the rewrite adds `[batch, seq, d_out] − [d_out, d_in]` extra
retained memory. At Jose dims (worst module: d_out=3072, d_in=768) this is
about `3.1 MB × batch − 9.4 MB`, crossing zero around batch=3 and growing
linearly above it.

**Implication for the report:** §7d should be deleted or reversed. The
materialization is small and amortizable; the proposed rewrite makes things
worse. A genuine fix for the FSDP friction in §5c is `summon_full_params`
around `calc_weight_deltas`, accepting the per-step gather cost — *not* this
rewrite.

The hook integration is intentionally not landed: the math is verified, the
benchmark says don't ship it, and §5c's actual problem (FSDP-sharded V/U
during `calc_weight_deltas`) needs a different solution.

---

## Pending phases

- Phase 3 — gradient checkpointing on the CI fn (code committed, run pending in queue)
- Phase 5 (stretch) — 1B-target stress test (will use random-init target per Oli's note)
