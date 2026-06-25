# How JAX executes the full-32L step

Reverse-engineered from the compiled HLO of the working run (job 128760), dump at
`/mnt/data/artifacts/mechanisms/param-decomp/reshard_hlo6/`. The decomposition is the
full Llama-3.1-8B (VPD) on **32 GPUs**, mesh **`(dp=4, tp=8)`** (FSDP+data-parallel
across the dp axis, Megatron tensor-parallel within the node on the tp axis).

Source of the step: `param_decomp/train.py` (`make_train_step`), `param_decomp/targets/llama8b.py`
(clean/masked forward, scan over layers), `param_decomp/ci_fn.py` (chunkwise transformer CI fn).

## Method / reproducibility

- Canonical module: `module_67910.jit_step.sm_10.0a_gpu_after_optimizations.txt` (~88 MB,
  495 205 lines). There are 7 per-rank `jit_step` modules; their **collective counts are
  bit-identical** (verified below), so one rank's HLO is a faithful picture of the single
  SPMD program.
- Companion files: `*-memory-usage-report.txt` (peak + cumulative buffers),
  `*-buffer-assignment.txt` (per-buffer sizes, 7 MB).
- Analysis scripts written for this report live in the scratchpad
  (`analyze3.py` = while/computation structure, `coll_shapes.py` = collectives by result
  shape, `agaxis.py` = collectives by mesh axis). The one-liners are inline below.

**Mesh proof** (HLO header + replica groups):
```bash
grep -oE "num_partitions=[0-9]+" $F | sort -u                 # -> num_partitions=32
grep -oE "replica_groups=\{[^}]*\}+" $F | sort | uniq -c      # -> 346x {{0..7}...}=tp(8 contiguous),
                                                              #     23x {{0,8,16,24}...}=dp(4 stride-8)
grep -E "all-gather-start\(" $F | grep -oE "replica_groups=.*?dimensions=\{[0-9]+\}" \
  | sed -E 's/operand.*//' | sort | uniq -c                  # axis names: 'axis_0'=4 (dp), 'axis_2'=8 (tp)
```
The named-axis replica groups (`mesh['axis_0'=4,'axis_1'=1,'axis_2'=8] {'axis_2'}`) make the
dp(4)/tp(8) attribution unambiguous.

**Cross-rank identity** (the program is one SPMD module):
```
module 67910: all-gather=1124 all-reduce=667 reduce-scatter=113 all-to-all=348 collective-permute=2362
module 84444: (identical)   module 298064: (identical)   module 686911: (identical)
```

---

## 1. Collectives inventory

Static op counts over the whole module (every computation, counted once at the textual
site — collectives inside a `while` body or a `cond` branch are emitted once but **execute
per scan iteration / per taken branch**, so wall-clock multiplies these by the trip count):

| collective | count | proven role |
|---|---:|---|
| `all-gather-start` / `-done` | **1124** | FSDP/TP weight reconstruction (dp + tp gathers) |
| `all-reduce-start` / `-done` | **667** | TP matmul partials (`dot_general`) + loss/grad `reduce_sum` |
| `reduce-scatter` | **113** | FSDP gradient reduce-scatter (over dp) + a few vmap reductions |
| `collective-permute-start` | **2362** | shard-rotation steps of ring all-gather/reduce-scatter and small reshards |
| `all-to-all` | **348** | C↔batch / head reshards at the V·U×CI and attention seams |

### all-gather: dp vs tp split (which weights)
`agaxis.py` groups each all-gather by the mesh axis it reduces over:
```
all-gather-start: total 1124
  by group SIZE: size=8 -> 623   (tp gathers)   size=4 -> 287  (dp gathers)
  (+ ~214 in iota replica-group forms, mostly the same two axes transposed)
```
- **size-8 (tp) gathers (623)** reconstruct the **Megatron-sharded dim**: the CI-fn weights
  (`qkv` head-on-tp, `w1/w2` mlp-hidden-on-tp, `out_w` C-on-tp) and V/U (`V` C-on-tp,
  `U` C-on-tp). Example result shapes from `coll_shapes.py`:
  `f32[1024,3584] -> f32[1024,14336]` (×4 = tp), `f32[512,1025] -> f32[512,8200]` (×8 = tp).
- **size-4 (dp) gathers (287)** are the **FSDP gathers** reconstructing the dp-sharded dim:
  the frozen target layer weights (`Wg/Wu/Wd/Wq/Wk/Wv/Wo`, all `/dp` on `d`), the CI-fn
  FSDP dim, and V/U's FSDP dim. Example: `bf16[32,512,1024] -> bf16[32,512,4096]` (×24,
  the stacked target reconstructing `d=1024 -> 4096`), `bf16[512,1024]->bf16[4,512,1024]`.

**Are the FSDP gathers bounded by `/tp` as expected?** For the **CI fn and V/U** — yes: each
weight is sharded on *both* axes (`P("dp","tp")` etc.), so the FSDP all-gather operates on a
tensor already cut by tp and reconstructs only the dp factor (size-4 gather of a `/tp` slab).
For the **frozen target** — the target is **dp-only** (see §2 / the actual
`LlamaLayer.shardings` / `FrozenAttn.shardings`: `P(None,None,"dp")`), so its FSDP gathers
are size-4 over the full (non-tp-cut) `d` dim. That is the intended design (target is
dp-only FSDP; the `LlamaDecomposedModel.shardings` docstring's "`/(dp·tp)`" claim is stale —
the per-layer shardings it delegates to only place `dp`).

### all-reduce attribution
`grep -E "all-reduce-start" $F | grep -oE 'op_name="[^"]*"'` (collapsed):
```
314 jvp()/transpose                  # backward of TP matmuls
124 dot_general                      # forward TP matmul partial-sum closes (Megatron all-reduce)
122 jvp()/reduce_sum                 # loss / grad scalar reductions
 84 .../checkpoint/rematted_computation/while/body/.../cond/branch_1/dot_general  # remat'd recon TP matmuls
 84 jvp()/closed_call/while/body/.../cond/branch_1/dot_general                    # recon-forward TP matmuls
```
So the all-reduces are TP-matmul partial closes (`dot_general`, inside the per-layer scan
`cond` branches that run the decomposed `x@V/(·)@U` and MLP) plus loss/grad `reduce_sum`.
The 346 contiguous-8 (`{{0..7}}`) replica groups are exactly these tp reductions.

### reduce-scatter (FSDP grad)
`reduce-scatter: 113`, **all 113** in the `replica_groups=[8,4]<=[4,8]T(1,0)` form — the
[4,8] mesh transposed so the inner factor 4 = **dp** (8 groups of 4 devices), op_names
`reduce_sum` / `transpose(jvp(vmap()))/reduce_sum`. The result shapes are f32 trainable
gradients (`f32[1024,3584]`, `f32[3584,1280]`, `f32[1024,14336]`, …) = the V/U and CI-fn
grads. This is the **FSDP gradient reduce-scatter** over dp, plus a few CI-fn-vmap reductions. Combined with the size-4 all-gathers, this is the standard FSDP
all-gather-forward / reduce-scatter-backward pattern.

### all-to-all (the reshards)
`all-to-all: 348`, op_names dominated by small element-wise ops *inside* the masked-forward
`cond` branches (`jvp()/sub` ×289, `add` ×220, `select_n`, `mul`) and the CI-fn vmap
(`transpose(jvp(vmap()))/add_any`). These are the C↔batch and replicated↔head reshards at
the V·U×CI-mask seam and the attention q/k/v seam — many small ones, **no single dominant
C→batch arena** (evidence that `batch_sharded_ci` is doing its job; see §5).

---

## 2. Memory story

**Peak (`Total bytes`)**: `145 093 140 110` = **135.13 GiB** per GPU
(`module_67910.…-memory-usage-report.txt`, line 2).

### Biggest single buffers (`buffer-assignment.txt`, `allocation N: size …`)
| size | what | sharded? |
|---:|---|---|
| **85 790 420 776 B (≈79.9 GiB)** | `allocation 14490: preallocated-temp` | the working arena (activations, gathered weights, collective scratch, remat buffers) |
| 1 050 673 152 B (1.0 GiB) ×2 | `bf16[128256,4096]` = embed + lm_head | **replicated** (P()) |
| 939 524 096 B (0.94 GiB) ×3 | `bf16[32,14336,1024]` (Wg, Wu), `bf16[32,1024,14336]` (Wd) | **dp-sharded** (`d=4096 -> 1024`) |
| 637 534 208 B (0.64 GiB) ×6 | `f32[32,1024,4864]` | CI-fn weights + Adam moments (stacked over 32 chunks) |

The **largest preallocated-temp arena is the ~80 GiB `allocation 14490`** — i.e. ~59% of
peak is transient working memory, the rest is parameters + optimizer state.

### Confirming the target weights are SHARDED (not replicated)
Entry layout (`head -1 $F`, the jit arg shapes) for the frozen target:
```
bf16[32,4096,1024]   Wq   (full d_in=4096 -> 1024 = /dp)        # 32 = stacked layer axis
bf16[32,1024,1024]   Wk, Wv (full [32,1024,4096] -> 1024 = /dp)
bf16[32,1024,4096]   Wo   (full [32,4096,4096] -> 1024 on d)
bf16[32,14336,1024]  Wg, Wu (full [32,14336,4096] -> 1024 = /dp); intermediate 14336 REPLICATED
bf16[32,1024,14336]  Wd   (full [32,4096,14336] -> 1024 = /dp)
bf16[128256,4096]    embed, lm_head  -> REPLICATED (full vocab×d)
```
Every per-layer matrix carries a sharded `d`-axis of **1024 = 4096/dp(4)**; the intermediate
(14336) and head dims stay full → the target is **dp-only FSDP, `/4`**, exactly as
`LlamaLayer.shardings` / `FrozenAttn.shardings` specify (`P(None,None,"dp")` /
`P(None,"dp",None)`). embed + lm_head are replicated. This sharding of the frozen target —
new in this branch — is what brings the full 32-layer model + its backward/remat copies
under the per-GPU budget.

The trainable V/U and the CI fn carry **both** axes (the C-dim on tp, the matmul dim on dp):
e.g. CI-fn `f32[3584,1280]` / `f32[1280,1024]` (in/out proj, tp+dp sliced), V/U entries
`f32[1024,256]` etc. The persistent adversarial sources appear as `f32[1,512,8193]`,
`f32[1,512,4097]`, `f32[1,512,2049]`, `f32[1,512,10241]` — scope **`sc`** = `(1, T=512, C+1)`,
broadcasting over batch (not the full `(B,T,C+1)`).

---

## 3. Attention

### Target attention = cuDNN flash (no materialized score) — PROVEN
```bash
grep -oE 'custom_call_target="[^"]*"' $F | sort | uniq -c
#   43  "__cudnn$fmhaSoftmax"          (forward flash)
#    6  "__cudnn$fmhaSoftmaxBackward"  (backward flash; few because most forwards are detached PGD/clean)
```
The fmha operands are `bf16[1,32,512,128]` (q: batch=1 per-dp-rank, **32 heads**, T=512,
head_dim=128) and `bf16[1,512,8,128]` (k/v: **8 kv-heads** — native GQA, k/v are *not*
repeated to 32). The leading `1` = batch sharded on dp (global B=4 → 1/rank). There is **no
`[B,H,T,T]` score** for the target: filtering `[1,32,512,512]` outside fmha returns only
`reshape` ops, never a softmax/score buffer.

q/k/v are **batch-parallel, heads replicated**: head=32 is *not* divided by tp=8 in the fmha
operands, and the q/k/v share one spec — which is what cuDNN's custom partitioner demands
(`op_name="…/custom_partitioning"` on every fmha call). This matches the *actually-compiled*
`FrozenAttn.shardings` docstring (batch on dp, heads replicated), **not** the head-on-tp
branch in `FrozenAttn.core` (that path is a no-op here because the projection outputs land
heads-replicated, so the `with_sharding_constraint` to `P("dp",None,None,None)` keeps heads
whole).

### CI-fn attention = xla, tiny materialized score — PROVEN
The CI fn uses `jax.nn.dot_product_attention(..., implementation="xla")`. Its score **is**
materialized but small: `f32[32,8,512,512]` (×96) and `bf16[32,8,512,512]` (×24) — leading
**32 = n_chunks** (one chunk per layer, run under `eqx.filter_vmap`), 8 heads, 512×512. These
are the "tens of MB" scores the source docstring promises (and they shard on tp, so per-rank
is smaller). No cuDNN custom-call appears in the CI-fn vmap region; the only custom-calls in
the module are the target's fmha.

---

## 4. Structure: scans, recon chunking, backward + remat

**19 `while` loops**, all trip-count **32** (the layer scan) except the nested PGD scan
(trip 4). Trip-count proof:
```bash
awk '/^%fused_compare.5 \(/{p=1} p{print} p&&/^}/{exit}' $F   # -> constant(32), direction=LT  (layer scan)
awk '/^%fused_compare.20 \(/{p=1} p{print} p&&/^}/{exit}' $F  # -> constant(4),  direction=LT  (PGD-step scan)
```

Each `while` maps to a phase via its `op_name` (`grep -E "%while\.[0-9]+ = " $F | grep -oE 'op_name=…'`):
| op_name root | count | phase |
|---|---:|---|
| `while` | 2 | plain forward scans — `clean_output` + `read_activations` (CI taps) |
| `jvp()/closed_call/while` | 5 | forward (jvp trace) scans of the differentiated recon/warmup forwards |
| `transpose(jvp(jvp()))/checkpoint/while` | 5 | **backward** (transpose-of-jvp) scans over the recon forwards |
| `transpose(jvp(jvp()))/checkpoint/rematted_computation/while` | 5 | the **rematerialized recompute** of those forwards inside the backward |
| `while/body/cc/jvp()/while` | 1 | **nested scan-in-scan**: PGD-ascent outer loop (trip 4), layer-scan body |
| `while/body/cc/transpose(jvp())/while` | 1 | backward of the nested PGD ascent |

So the de-chunked recon path compiles to **5 differentiated forward scans** (the recon loss
plan's entries/draws + the faith-warmup forward), each one block-body compiled once and run
32× — never unrolled. The clean target and the CI-input taps are 2 more plain forward scans.

**Remat is ON** (`remat_recon_forwards=true`): the backward scans appear in both a
`checkpoint/while` (the saved boundary) and a `checkpoint/rematted_computation/while` (the
recompute) form. Remat markers are pervasive: `grep -c rematted_computation $F` = **8108**,
`grep -oE "/checkpoint/" $F | wc -l` = **18806**. The backward is the XLA
`transpose(jvp(...))` of the forward scan (reverse-mode = transpose of forward-mode jvp),
nested two deep (`jvp(jvp())`) over the remat'd region.

**The nested PGD scan** (`while.2977` outer, body contains the 32-layer `while`): the outer
loop is the fresh-PGD sign-ascend / persistent warmup `lax.scan` (trip 4 = PGD steps); each
step runs a full layer-scan forward + its backward to get the source gradient. This is the
`for state_key …: lax.scan(warmup_body, …, length=n_warmup_steps)` /
`lax.scan(sign_ascend_body, …, length=n_steps)` in `train.py`.

**The CI-fn vmap over chunks** shows up as `op_name` roots `vmap(abc,dc->abd)`,
`jvp(vmap(abc,dc->abd))`, `transpose(jvp(vmap(...)))` on the all-gathers and all-to-alls —
i.e. the per-chunk `ChunkTransformer` stacked on the leading 32-chunk axis and run under one
`eqx.filter_vmap` (the leading 32 on the `f32[32,8,512,512]` scores and `f32[32,1024,4864]`
weights).

---

## 5. Mask / CI flow: born batch-on-dp, no C→batch reshard storm

`train.py::batch_sharded_ci` reshards the CI-fn output to batch-sharded **once**, at the
producer (the CI head's `out_w` is C-on-tp, so its output is born C-sharded; without one
producer-side pin GSPMD would insert a separate C→batch reshard for *every* consumer — each
recon forward, the adversaries, imp-min, forward and backward).

**HLO evidence that the pin works**: the all-to-alls (the reshards) total only **348**, and
their op_names are small element-wise ops scattered across the masked-forward `cond` branches
and the CI vmap (`jvp()/sub`, `add`, `select_n`, `mul`) — **not** a single huge repeated
C→batch all-to-all on the CI output buffer. The CI masks reach the masked forwards already
batch-on-dp: the masked-forward activations are `bf16[32,1,512,C]` (leading 32 = layer scan,
`1` = batch-on-dp per-rank slice). The masks ride the same `(1, T, C)` per-rank leading
shape as the sources, and the `with_sharding_constraint` in `batch_shard_leading`
(`P("dp", None, …)`) pins every batch-leading activation, so GSPMD does **not** insert a
per-consumer reshard — the all-to-all budget stays small and is dominated by the
intrinsic C/head seams, not by an un-pinned CI broadcast.

---

## What is proven vs inferred

**Proven directly from HLO:**
- Mesh `(dp=4, tp=8)`, 32 partitions; identical SPMD program across all 7 ranks.
- Collective counts and their per-axis (dp/tp) split and op_name attribution.
- Target weights dp-sharded `/4` (entry layout + buffer-assignment); embed/lm_head replicated.
- Peak 135.13 GiB; the ~80 GiB preallocated-temp arena is the single largest allocation.
- Target attention = `__cudnn$fmha…` flash with GQA (32 q-heads / 8 kv-heads), no `[B,H,T,T]` score; batch on dp, heads replicated.
- CI-fn attention = xla, materialized `f32[32,8,512,512]` score under a 32-chunk vmap.
- 19 scans, trip 32 (layers) / 4 (PGD); 5 differentiated recon forward scans + their remat backward; nested PGD scan-in-scan; remat on.
- `batch_sharded_ci` keeps the reshard (all-to-all) budget small.

**Inferred (consistent with HLO but not a single decisive line):**
- The exact attribution of *each* size-4 all-gather to a specific weight family (target vs
  CI-FSDP-dim vs V/U-FSDP-dim) — separated by shape and op_name, but the three families
  overlap in some shapes.
- The split of the 80 GiB temp arena between activations / gathered weights / collective
  scratch (the arena aliases hundreds of values; not decomposed per-category here).
- That all 5 `jvp/while` forward scans are distinct recon plan entries vs warmup — the
  op_names confirm the *kinds* (recon vs PGD-warmup vs faith) but the run's exact loss-term
  plan would need the run config to map 1:1.
