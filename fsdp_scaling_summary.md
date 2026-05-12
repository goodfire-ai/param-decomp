# FSDP Scaling Investigation — Results Summary

*Branch `feature/fsdp-wrap`, May 2026. H200 cluster (140 GB HBM per GPU).*

---

## The problem

The full SPD training loop makes 5–7 `model(batch, mask_infos=...)` calls per step
(target forward, PPGD warmups, each loss term). FSDP1's bookkeeping fails under
this multi-forward pattern (`setStorage out of bounds`). ZeRO-1 is the current
workaround but only shards optimizer state — activations still dominate, and the
4B target OOMs even at b=1.

---

## Phase 1: Jose baseline profile

**Setup:** 1×H200, `pile_llama_simple_mlp-4L`, target `t-9d2b8f02` (67M params),
b=8 per rank, bf16 autocast. Run `p-0ffcbe39`.

| Bucket | Measured |
|---|---:|
| Target params (frozen) | 67 M / 0.27 GB |
| Trainable (components + CI fn) | 660 M / 2.64 GB |
| AdamW state (m+v) | 5.28 GB |
| **Fixed state total** | **10.83 GB** |
| Activations + transient | 23.79 GB |
| **Peak (max_memory_allocated)** | **34.62 GB** |

**Activation cost: ~3 GB per per-rank batch element (bf16), not the ~1 GB/B
the report predicted. 3× higher than expected — activations dominate, not params.**

The training loop runs multiple forwards + PPGD warmups per step; the 3 GB/B figure
captures the full steady-state peak across all of them.

---

## Phase 2: ZeRO-1 validation

ZeRO-1 saves exactly `8·T·(N−1)/N` per rank as predicted, confirmed to <1% at N=2,4:

| N | ZeRO-1 saving (measured) | ZeRO-1 saving (predicted) |
|---:|---:|---:|
| 2 | 2.63 GB | 2.64 GB ✓ |
| 4 | 3.98 GB | 3.96 GB ✓ |

At Jose scale this is modest (< 5 GB at N=8). At 1B-target scale, trainable params
~2.9B → ~21 GB saved per rank at N=8 — meaningful but activations still dominate.

---

## Phase 4: weight-delta rewrite (§7d refuted)

Report §7d claimed rewriting `calc_weight_deltas` into the forward would be a "strict
win" under FSDP. Microbenchmark (`scripts/bench_weight_delta_rewrite.py`) showed the
opposite: **the rewrite adds 6–20% more memory and 15–30% more compute, scaling worse
with batch**. The rewrite retains two `[B,S,d_out]` tensors vs the materialize path's
one. §7d should be discarded.

---

## Phase 5: FSDP2 implementation

### Architecture change

Replaced the old `LinearComponents` + `EmbeddingComponents` (stored separately in a
`ModuleDict`) with **fused `DecomposedLinear` / `DecomposedEmbedding` sites** that own
their V/U params and wrap the frozen target submodule directly. `ComponentModel` installs
these in-place via `install_decomposed_sites`. Each site is its own FSDP2 unit.

`fsdp_wrap` applies `fully_shard` bottom-up on:
- every `DecomposedLinear` / `DecomposedEmbedding`
- every CI-fn `TransformerBlock`
- the `GlobalSharedTransformerCiFn` root

The `ComponentModel` root is intentionally **not wrapped** — wrapping it makes
`wte`/`ln_f`/`lm_head` DTensor-owned, leaking DTensors into the frozen target forward
where they mix with regular Tensor outputs from FSDP units and crash the `aten.bmm`
dispatcher.

Key fixes required:
- `output_dtype=float32` in `MixedPrecisionPolicy` — materializes DTensor outputs as regular
  Tensors at each unit boundary; without this, `bmm` in `CausalSelfAttention` gets mixed inputs.
- `_untie_target_weights_()` — breaks `wte.weight = lm_head.weight` sharing before wrapping;
  tied params cause `setStorage out of bounds` under FSDP.
- `calc_weight_deltas_full()` — gathers sharded V/U DTensors outside any forward (used in both
  eval and training) before passing deltas to downstream `bmm` against regular Tensors.

### Smoke test results (N=4, b=1, fp32, no target block ckpt)

Random 1B/2B/4B targets at Jose CI-fn shape (489M trainable CI fn):

| Target | Trainable | Post-wrap | Decomposed fwd+bwd peak |
|---|---:|---:|---:|
| 1B (18L) | 2.89 B | 3.94 GB | 26.73 GB |
| 2B | 4.44 B | 6.49 GB | 43.44 GB |
| 4B | 6.97 B | 11.11 GB | **72.17 GB** |

Compare to baseline:
- **1B b=2 ZeRO-1+ckpt: 72.75 GB → FSDP b=1: 26.73 GB (2.7× less)**
- **4B b=1 ZeRO-1+ckpt: OOM (>140 GB) → FSDP b=1: 72.17 GB (fits H200)**

### With CI fn gradient checkpointing

Enabling `ci_config.simple_transformer_ci_cfg.gradient_checkpointing=true`
drops activation memory sharply (all N=4, b=1):

| Target | No ckpt | CI fn ckpt | Reduction |
|---|---:|---:|---:|
| 1B | 26.73 GB | 10.47 GB | **61%** |
| 2B | 43.44 GB | 17.11 GB | **61%** |
| 4B | 72.17 GB | 28.42 GB | **61%** |

The CI fn has 8 transformer blocks at d_model=2048 — its activations dominate peak
at these scales. Checkpointing it recomputes each block during backward rather than
storing its activation graph.

### End-to-end training on Jose (N=4)

Job 13409, runs `p-429cbcc5` (b=8/rank) and `p-4c54b2fd` (b=32/rank):

| Config | Per-rank b | Peak per GPU |
|---|---:|---:|
| FSDP N=4 | 8 | **30 GB** |
| FSDP N=4 | 32 | **110 GB** |

Both runs completed cleanly through the full training loop including evaluation.
The b=32 run shows the activation cost (~2.5 GB/B at b=32) — activations dominate
even under FSDP, since params are sharded but activations are per-rank.

### Activation scaling law under FSDP

Params scale as `1/N` (sharded). Activations stay per-rank and scale as `~B`:

```
peak ≈ params/N + activations_per_B × B
     ≈ params/N + 3 × B   (GB, Jose-scale CI fn, fp32 decomposed fwd)
```

At 1B-target scale with N=4 and CI fn ckpt:
- Fixed (sharded params): ~4 GB
- Activation/B: ~6.5 GB (from slurm-13376 slope)
- At b=1: ~10.5 GB ✓ (matches measurement)
- At b=4: ~30 GB — comfortable headroom on H200

This means **4B target at b=2 under FSDP N=4 with CI fn ckpt** should land near
~28 GB + 2 × 6.5 = ~41 GB — fits on a single H200 with plenty of room.

---

## New this session: bf16 + target block checkpointing

### bf16 FSDP support

`fsdp_wrap` now takes `autocast_bf16=True` to use:
```python
MixedPrecisionPolicy(param_dtype=bfloat16, reduce_dtype=bfloat16, output_dtype=float32)
```

Params are stored as bf16 shards (halving sharded-param memory) and allreduce runs
in bf16 (halving bandwidth). Outputs stay fp32 — this is the key constraint that
prevents DTensor/dtype mixing in the frozen target's `bmm` calls. **Expected saving
at 1B-target scale N=4: ~1.5 GB per rank from the bf16 shards alone.**

Note: explicit `autocast` context is still disabled under FSDP (`autocast_active =
autocast_bf16 and parallel_strategy != "fsdp"`). The bf16 savings come from
`param_dtype`, not the autocast kernel path.

### Target block gradient checkpointing

New `target_gradient_checkpointing: bool = False` in `Config`. When enabled, each
transformer block in `LlamaSimpleMLP.forward` uses `torch.utils.checkpoint.checkpoint`
with `use_reentrant=False`. This checkpoints the entire block (target attention +
decomposed MLP sites), recomputing its internals during backward.

For a 1B model (18 layers, d_model=2048, n_intermediate=8192, seq=512):
- Estimated activation saving: ~576 MB per batch element (attn Q/K/V + MLP intermediates)
- At b=4: ~2.3 GB saved

Usable via `--override target_gradient_checkpointing=true` in `pd-local`.

---

## What activations dominate: an aside

Even at Jose scale (67M target, ~3 GB/B activations measured), the 3 GB/B figure is
high because SPD's training loop runs **5–7 forward passes per step** through the
ComponentModel (target fwd, PPGD warmup × 2, each loss term with its own decomposed
fwd). Each contributes its own activation graph while earlier graphs are still live.
Gradient checkpointing on the CI fn (the largest module) removes the bulk of that.

At 1B-target scale, the target itself becomes another major contributor (~600 MB/B
per block × 18 layers without checkpointing). That's why `target_gradient_checkpointing`
matters at 1B scale even though it doesn't at 67M Jose scale.

---

## Summary table: what works, what's next

| Feature | Status |
|---|---|
| FSDP2 basic wrapping | ✅ Working |
| DTensor leak fix (eval + weight-delta path) | ✅ Fixed |
| Tied-weight untying pre-wrap | ✅ Fixed |
| CI fn gradient checkpointing under FSDP | ✅ Working |
| End-to-end training on Jose (N=4) | ✅ Verified (b=8, b=32) |
| bf16 params under FSDP (`param_dtype=bfloat16`) | ✅ Implemented (untested on cluster) |
| Target block gradient checkpointing | ✅ Implemented (untested on cluster) |
| Checkpoint save/load under FSDP | ❌ Not yet (`save_freq=null` required) |
| Loss-trajectory equivalence test (DDP vs FSDP) | ❌ Pending |
| Migration of Jose/Thomas checkpoints | ❌ Pending (state-dict key rename) |

---

## Headline numbers

| Scenario | Peak per GPU | Notes |
|---|---:|---|
| Jose (67M), DDP b=8, bf16 | 34.6 GB | Phase 1 baseline |
| Jose (67M), FSDP N=4, b=8 | 30 GB | end-to-end verified |
| Jose (67M), FSDP N=4, b=32 | 110 GB | near H200 ceiling |
| 1B, FSDP N=4, b=1, no ckpt | 26.7 GB | smoke test |
| 1B, FSDP N=4, b=1, CI fn ckpt | 10.5 GB | smoke test |
| 2B, FSDP N=4, b=1, CI fn ckpt | 17.1 GB | smoke test |
| 4B, FSDP N=4, b=1, CI fn ckpt | 28.4 GB | smoke test |
| 1B, ZeRO-1+ckpt, b=2 | 72.75 GB | old ceiling |
| 4B, ZeRO-1+ckpt, b=1 | OOM | new FSDP unlocks this |
