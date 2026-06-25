# Full-32L decomposition: the memory model + how to size a run

How per-device memory and the global batch relate on the 2-D `(dp, tp)` mesh, for the
full Llama-3.1-8B decomposition (32 layers × 7 matrices = 224 sites). Written after the
2026-06-25 scaling session; numbers are measured where stated, modelled otherwise.

**Status:** per-layer remat is validated (numerically + memory win). The tp/per-DP
tradeoff for max batch is **partly resolved and partly open** — multiple OOMs came from
over-confident predictions (incl. one in the first draft of this doc). The only full-model
config *proven to train* is the small A/B (tp8/dp32/per-DP1, 154 GiB). Treat the
sizing conclusions below as a model to be confirmed by a buffer-composition probe, not as
settled fact.

## The one equation

```
mesh         = (dp_axis, tp),  dp_axis = N / tp        # N = total devices = runtime.dp
per-DP batch = B / dp_axis     = B · tp / N            # B = global batch
global batch = per-DP × dp_axis = per-DP × N / tp
```

`runtime.dp` is the **total device count**, NOT the dp-axis size. `runtime.tp` is the
intra-node tensor-parallel width.

**The batch shards on the `dp` axis only** (`shard_batch` → `P("dp")`); the `tp` ranks
hold a **replica** of the same batch slice (`local_dp = local_device_count // tp`; at
`tp = 8` that's `1`, so a whole node holds one replica). So the natural unit is **per-DP**
(= per-dp-axis-position = "per node" at tp=8), and the `tp` GPUs in a dp-position each
carry that per-DP batch.

## What lives where (per device)

| Term | Sharding | Per-device size |
|---|---|---|
| V/U params + Adam state | `dp` × `tp` → `/N` | `/N` — **tp-independent** |
| CI-fn params + Adam state | `dp` × `tp` → `/N` | `/N` — **tp-independent** |
| Frozen target params | FSDP on `dp_axis` only (heads/intermediate replicated) | `/dp_axis = ·tp/N` (grows with tp; small, target is 8B) |
| **Target activations** | **NOT TP'd** (full width) | **`∝ per-DP`, tp-INDEPENDENT** ← the binding term at seq512 |
| CI activations | Megatron-TP'd | `∝ per-DP / tp` |
| CI-fn FSDP gather transient | un-shards `dp` → keeps `tp` shard | `∝ param / tp` (grows as tp drops) |

Two consequences fall straight out of the table:

1. **Resident is tiny at high dp.** At dp256, V/U+CI+optimizer ≈ 2.7 GiB/device (`/256`);
   at dp128 ≈ 5.4 GiB; at dp32 ≈ 22 GiB. So small-mesh probes (e.g. the dp32 A/B)
   *over-count* resident relative to a large-mesh production run.
2. **The seq512 binding term — target activations — is tp-independent.** It scales with
   per-DP and full width regardless of tp, because we do NOT tensor-parallel the target
   (the cuDNN flash-attn partitioner needs q/k/v identically sharded → heads replicated;
   the MLP intermediate is left replicated too).

## The sizing rule (low tp *probably* helps batch — but the ceiling erodes, hard)

Naive version: if the binding term (target activations) were the *only* tp-relevant term,
the per-DP ceiling would be tp-flat and `B_max(tp) = perDP_ceiling × N/tp ∝ 1/tp` → lowest
tp wins outright.

**Reality (measured 2026-06-25): the ceiling is NOT tp-flat — it drops meaningfully as tp
falls.** tp8 per-DP ceiling ≈ 4–6; **tp2 ceiling < 4** (per-DP 4 OOM'd at the train step).
The eroding term is the **CI-fn FSDP gather transient `∝ param/tp`** — 4× larger at tp2
than tp8 — and it is **first-order, not the "second-order" hand-wave I used earlier.**

So the real comparison is `B_max(tp) = ceiling(tp) × N/tp` with a ceiling that *shrinks* as
tp drops. Rough numbers at dp128: tp8 `~5 × 16 = 80` vs tp2 `~2–3 × 64 = 128–192`. Low tp
still **probably** wins (the 4× `dp_axis` beats the ceiling erosion) — but by far less than
`1/tp`, and the achievable tp2 batch is ~B128–192, **NOT B256**. Whether the CI gather
eventually dominates and flips the conclusion at very low tp (tp1) is **unresolved** —
needs the buffer-composition measurement below, not more extrapolation.

Caveat I got wrong twice this session and corrected:
- A **fixed-global-batch tp sweep confounds tp with per-DP** (lower tp ⇒ lower per-DP at
  fixed B). Its "flat memory across tp" is two effects cancelling (per-DP activation
  saving ≈ CI-gather growth), NOT "memory is tp-independent." To isolate tp you must hold
  **per-DP** fixed.
- At **fixed per-DP**, lower tp is mildly *worse* (CI terms grow) — but that's not the
  metric for maximising batch; `B_max(tp) ∝ 1/tp` is.

## Per-layer remat (the enabling fix)

The recon forward is `lax.scan` over the 32 layers. The masked forward is gradient-
checkpointed at the **scan-body** granularity (`scan(checkpoint(block))`), so the backward
recomputes **one layer at a time** and stores only the per-layer carry (the residual,
4096-wide), instead of stacking every layer's activations (`[32, ·, 14336]`) — the term
that dominated peak before. `remat` is a keyword-only arg threaded through the
`DecomposedModel.masked_output` protocol (scan models remat per-layer; toys whole-forward).

Validated 2026-06-25:
- **Numerically transparent**: faith @ step 150 = 3.746e-4 (per-layer) vs 3.754e-4
  (whole-forward baseline 128760) — 0.2%, i.e. fp/placement noise across two distributed
  runs. Plus 87 unit tests pass.
- **Memory win grows with per-DP** (little to stack at per-DP 1, lots at high per-DP):
  - seq64 tp re-sweep (B16/dp32): tp8 **166.7 → 111.9** (−33%); **tp4/tp2 went from OOM
    → 101.3 / 109 GiB** (fit). The whole-forward sweep had tp4/tp2 hard-OOM.
  - seq512 A/B (tp8/per-DP1/dp32): **173.5 → 154.4** (−11%, small because per-DP 1 has
    little activation to stack).

## Measured calibration (seq512, per-layer remat)

| config | per-DP | result |
|---|---|---|
| tp8 / dp32 / B4 | 1 | **154.4 GiB ✅ COMPLETED 150 steps** (job 128800) — the only proven-fitting full-model run so far |
| tp2 / dp128 / B256 | 4 | ❌ **OOM at the train step** (87 GiB alloc on `jit_step`). Compiled with NO remat-floor warning + wrote ckpt-0, then OOM'd *executing* — run `p-f39007db` / job 128809 |
| tp8 / dp256 / B256 | 8 | ❌ OOM (compile wanted 227.5 GiB; train step) |
| tp8 / dp256 / B512 | 16 | ❌ OOM (227.5 GiB) |

→ **per-DP ceiling is tp-dependent, NOT flat:** tp8 ≈ 4–6 (per-DP1 fits, per-DP8 OOMs);
**tp2 < 4** (per-DP4 OOMs). Lower tp ⇒ lower ceiling (CI gather `∝param/tp`).

⚠️ **"Compiles ≠ fits."** The tp2/per-DP4 run compiled *without* the
`hlo_rematerialization` floor warning and still OOM'd at runtime (fragmentation /
runtime-only transients push past the static plan). The remat warning is a *sufficient*
no-fit signal, not a *necessary* one — only an actual step completing (a logged
`peak_gb_per_rank`) proves a fit.

The only full-model decomposition proven to train so far is the small **A/B: tp8 / dp32 /
B4 / per-DP1**, 154.4 GiB, 150 steps. The B256/tp2 attempt OOM'd.

## How global-batch-matched tp1 / tp4 would go (B256 / dp128, predicted)

At fixed B256/dp128, per-DP = `256·tp/128 = 2·tp`:

| tp | dp_axis | per-DP | status |
|---|---|---|---|
| 1 | 128 | **2** | untested; CI gather is biggest here (`param/1`) + no `tp` axis (GSPMD shifts) |
| 2 | 64 | **4** | ❌ **OOM (measured)** — per-DP 4 is above the tp2 ceiling |
| 4 | 32 | **8** | ❌ OOM (per-DP 8 ≥ the tp8/dp256 OOM, and tp4 gather > tp8) |
| 8 | 16 | 16 | ❌ OOM (per-DP 16) |

So at fixed B256/dp128, **everything ≥ tp2 OOMs** — none of tp2/4/8 fit at B256. To fit
B256 you'd need either per-DP < 4 (→ more dp_axis → tp1, or more GPUs) or to cut the
per-device cost (smaller eval, Megatron the target, etc).

**tp1 is now the open question, and genuinely uncertain.** Pro: `dp_axis = 128`, the most
batch-per-GPU. Con: its CI gather is the largest (`param/1`), so its ceiling erodes most —
and we've now seen the erosion is real (tp2 already fell below per-DP 4). Whether tp1's 2×
`dp_axis` beats its further-eroded ceiling is exactly what the global-batch sweep CANNOT
tell us (it confounds tp and per-DP). It needs a **fixed-tp per-DP ceiling probe at tp1**,
and/or the buffer-composition measurement, before any prediction.

## Open / unmeasured

- **Exact peak-buffer composition under per-layer remat** (carries vs faith-deltas vs CI
  gather vs logits). Needs a **GPU** `--xla_dump_to` compile probe — the CPU-forced probe
  can't build the 50B-param init in host RAM.
- **perDP ceiling at tp1 and tp2** (push B until OOM).
- **Megatron the target MLP intermediate**: would shard the 14336 hidden `/tp`, but
  per-layer remat already made it a transient; the dominant stored term is now the 4096
  residual carries, which it doesn't touch. Re-evaluate only after the composition probe.
