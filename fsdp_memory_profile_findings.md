# FSDP Memory Profile Findings

Workspace: `/mnt/polished-lake/home/braun/param-decomp-fsdp-profile`

Branch: `codex/fsdp-memory-profile`

Harness:

- `scripts/fsdp_memory_profile.py`
- `scripts/launch_fsdp_memory_profiles.py`

Raw result roots:

- `profiling_runs/smoke-n2-20260513-092348`
- `profiling_runs/jose-n4-20260513-092617`
- `profiling_runs/jose-n8-20260513-093522`
- `profiling_runs/scale-n4-20260513-093517`
- `profiling_runs/autocast-n4-20260513-094312`
- `profiling_runs/faith-n4-20260513-094851`

All main runs used H200s, sequence length 512, two decomposed forwards per Jose-like step, delta masks enabled, and no faithfulness term unless explicitly noted. FSDP numbers below use fp32 parameter shards because the current bf16 FSDP path fails.

## Short Answer

FSDP is buying a real fixed-state reduction, and that becomes decisive at large target scale. It is not buying much at Jose scale once batch size is moderately large, because activations dominate and the current FSDP path disables bf16 autocast.

At Jose scale, ZeRO-1 is usually the better tradeoff: close memory at moderate/high batch and much faster. At 1B+ target scale, FSDP becomes the only tested path with enough fixed-state headroom to make larger models comfortable, and it fits a 4.23B-frozen-target synthetic run at batch 2 on 4 H200s.

## Jose Scale, N=4

Peak per-rank memory for the Jose-like step:

| per-rank batch | DDP | ZeRO-1 | FSDP fp32 |
|---:|---:|---:|---:|
| 1 | 16.67 GB | 10.73 GB | 5.66 GB |
| 4 | 18.12 GB | 14.13 GB | 11.43 GB |
| 8 | 24.42 GB | 20.46 GB | 19.32 GB |
| 16 | 37.28 GB | 33.32 GB | 35.09 GB |
| 32 | 62.99 GB | 59.03 GB | 66.65 GB |

Fixed state after warmup:

| strategy | after wrap | after first optimizer step |
|---|---:|---:|
| DDP | 5.56 GB | 10.90 GB |
| ZeRO-1 | 5.56 GB | 6.94 GB |
| FSDP fp32 | 1.00 GB | 2.39 GB |

Interpretation: FSDP saves about 4.5 GB vs ZeRO-1 after optimizer state exists, but the per-batch activation term dominates by batch 16-32. Current FSDP is also much slower: at batch 32, FSDP took 2714 ms/step vs DDP 819 ms and ZeRO-1 1002 ms.

## Jose Scale, N=8

The world-size curve behaves as expected: FSDP fixed memory falls, dynamic memory does not.

| per-rank batch | DDP | ZeRO-1 | FSDP fp32 |
|---:|---:|---:|---:|
| 1 | 16.67 GB | 9.74 GB | 4.51 GB |
| 4 | 18.12 GB | 13.47 GB | 10.42 GB |
| 8 | 24.42 GB | 19.80 GB | 18.31 GB |
| 16 | 37.28 GB | 32.66 GB | 34.09 GB |
| 32 | 62.99 GB | 58.37 GB | 65.64 GB |

FSDP after-warmup fixed state dropped from 2.39 GB at N=4 to 1.39 GB at N=8. The batch-dependent term was effectively unchanged.

## Synthetic Large Targets, N=4

The scale labels are based on frozen target size. Trainable params include components plus the CI transformer.

| target | trainable | strategy | batch | peak | after warmup |
|---:|---:|---|---:|---:|---:|
| 1.01B | 2.89B | FSDP fp32 | 1 | 33.37 GB | 10.51 GB |
| 1.01B | 2.89B | FSDP fp32 | 2 | 42.63 GB | 10.51 GB |
| 1.01B | 2.89B | ZeRO-1 | 1 | 49.88 GB | 33.08 GB |
| 1.92B | 5.39B | FSDP fp32 | 1 | 63.79 GB | 18.92 GB |
| 1.92B | 5.39B | FSDP fp32 | 2 | 81.44 GB | 18.92 GB |
| 1.92B | 5.39B | ZeRO-1 | 1 | 93.63 GB | 61.74 GB |
| 4.23B | 7.52B | FSDP fp32 | 1 | 95.89 GB | 27.99 GB |
| 4.23B | 7.52B | FSDP fp32 | 2 | 118.98 GB | 27.99 GB |

This is where FSDP is clearly buying something. At 2B target scale, FSDP saves about 30 GB peak vs ZeRO-1 at batch 1. At 4B target scale, FSDP still fits batch 2 on a 143 GB H200.

## Checkpointing

CI checkpointing helps Jose-scale FSDP a lot:

| batch | FSDP no CI ckpt | FSDP CI ckpt |
|---:|---:|---:|
| 8 | 23.35 GB | 19.32 GB |
| 16 | 43.15 GB | 35.09 GB |
| 32 | 82.77 GB | 66.65 GB |

At large target scale, CI checkpointing alone barely changed peak memory. The target/decomposed forward dominates there.

Target block checkpointing is currently broken in this fused-site path. Runs with `target_checkpointing=True` failed for FSDP and ZeRO-1 with:

```text
torch.utils.checkpoint: A different number of tensors was saved during the original forward and recomputation.
```

## Correctness/Usability Blockers

1. **FSDP bf16 parameter shards fail.**

   `fsdp_wrap(..., autocast_bf16=True)` failed before the step with:

   ```text
   Expected query, key, and value to have the same dtype,
   but got query.dtype: float key.dtype: float and value.dtype: c10::BFloat16
   ```

   This happened with and without CI checkpointing. The successful FSDP runs use `autocast_bf16=False`.

2. **Target checkpointing is broken with fused decomposition sites.**

   This is not only an FSDP issue; the ZeRO-1 large-target run with target checkpointing failed the same way.

3. **Faithfulness terms are not FSDP-safe as currently registered.**

   A Jose b=8 FSDP run with `--include-faithfulness` failed inside FSDP2 registered method dispatch:

   ```text
   IndexError: tuple index out of range
   ```

   The failing path is `component_model.calc_faithfulness_terms()` calling `site.faithfulness_terms()` after `register_fsdp_forward_method(module, "faithfulness_terms")`. The no-arg registered method path appears incompatible with this FSDP2 version.

4. **Forward autocast with fp32 FSDP shards is mixed.**

   For Jose, forcing bf16 forward autocast while keeping fp32 FSDP shards improved memory and speed: batch 32 dropped from 66.65 GB / 2714 ms to 56.53 GB / 936 ms. For large targets it was worse, likely because autocast creates cached/copy weight pressure for large frozen target weights: 4B b=1 rose from 95.89 GB to 123.39 GB.

## Recommendation

Use FSDP for large-target fit work, not for Jose-scale throughput work yet.

For the current branch, the viable FSDP operating point is:

- `autocast_bf16=False`
- `target_gradient_checkpointing=False`
- no faithfulness loss under FSDP, or fix the registered method path first
- CI checkpointing on for Jose-scale/high-batch runs

Before calling FSDP production-ready, fix or explicitly gate:

1. bf16 FSDP mixed precision for CI attention
2. target checkpointing with fused sites
3. FSDP faithfulness-term dispatch
4. an activation bf16 story that does not create large-target weight-copy pressure
