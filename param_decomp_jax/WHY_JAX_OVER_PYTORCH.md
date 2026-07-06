# Why we use JAX over PyTorch for PD training

This doc is the rationale for running production parameter-decomposition (PD) training on the
JAX backend (`jax_single_pool`) rather than PyTorch. It collects the throughput measurements,
explains the *mechanism* behind the gap, records every PyTorch lever we tried to close it, and
states the decision.

## TL;DR

- **JAX is ~2.3× faster per GPU on the PD training step, and the gap holds at production
  scale.** Measured 2.45× at the 4L model (8-GPU) and **2.27× end-to-end at Llama-3.1-8B**
  (32-GPU). It is not an idle-GPU or CPU-affinity artifact.
- **The gap is structural.** JAX is functional by construction and JITs the *entire* step —
  forward, multi-forward recon loss, nested-grad PGD adversary, backward, *and* the optimizer —
  into one XLA program that fuses thousands of small kernels. Eager PyTorch cannot, and its
  *emulation* of the functional approach is slower than the eager baseline it was meant to beat.
- **We tried the PyTorch levers.** The best drop-in is the AOTAutograd "intermediate" at
  +9–28% (scale-dependent). The full functional rewrite is **~2× slower**. None come close.
- **Bonus capability:** JAX/GSPMD shards parameters and optimizer state across the mesh, so the
  8B decomposition fits at **69 GB/rank** (would fit an 80 GB H100). The shipped PyTorch
  single-pool trainer is DDP — it *replicates* optimizer state, so the same run does not fit a
  80 GB card no matter how many GPUs you add.

**Decision: JAX is the training backend for PD at scale.** Keep the AOTAutograd intermediate as
an optional PyTorch flag for small/local runs; do not pursue the full functional rewrite.

---

## Results

All runs use the PD step with a PGD adversary. Two recipes appear below — fresh-PGD
(`n1 ss0.1`) at 4L and PersistentPGD (`broadcast_across_batch`) at 8B — and the throughput
conclusion is robust across both.

### 4L model (`pile_llama_simple_mlp-4L`), 8-GPU matched tier — fresh-PGD `n1 ss0.1`

Per-GPU throughput (samples/s/GPU):

| Backend | it/s | per-GPU vs JAX |
|---|---|---|
| JAX (`@jax.jit` whole step) | 9.78 | 1.00× |
| PyTorch DDP (eager) | 3.99 | 0.41× |
| PyTorch FSDP (speed-tuned) | 4.16 | 0.43× |

JAX is **~2.3–2.45× faster per GPU**, robust across the 8- and 16-GPU tiers. With the
CPU-affinity bug fixed, torch runs ~97–100% GPU util but at lower power (~540 W vs ~670 W) and
higher memory-bandwidth util (~64% vs ~46%) — the classic signature of many small, HBM-bound
eager kernels vs whole-program fusion.

### Llama-3.1-8B, block-18 MLP decomposed — end-to-end, matched 32×B200 (NEW)

The real test: a full distributed run at production scale, not a micro-benchmark. Both legs use
the **same config** (`b128_cmp32`: global batch 128, seq 2048, C=24576 across
gate/up/down_proj, PersistentPGD `broadcast_across_batch`, no eval), on **32 B200 GPUs / 4
nodes each**. Global batch is identical, so a step = 128 sequences on both sides → fair. The
PyTorch leg is the **stock HuggingFace `transformers.LlamaForCausalLM`** (exactly how the
reference run `p-19645bf7` loaded its target), trained eager via `pd-lm` (DDP); the JAX leg is
`jsp-train`.

| Backend | s/step | it/s (global) | tok/s | tok/s/GPU | peak mem/rank | 200k-step projection |
|---|---|---|---|---|---|---|
| PyTorch (eager DDP, stock HF) | 2.18 | 0.459 | ~120k | ~3.76k | (B200-only, see below) | **5.05 days** |
| JAX (`jsp-train`) | 0.960 | 1.042 | ~273k | ~8.53k | 69 GB | **2.22 days** |
| **ratio** | **2.27× JAX** | | **2.27×** | **2.27×** | | **2.3× shorter** |

Steady-state, post-compile (torch confirmed step 26→180 at 2.18–2.19 s/it; JAX confirmed step
40/60/80 at 0.960/0.958/0.961 s/step). Numeric parity checked: faithfulness-warmup final loss
~1.1e-5 on both; main-loop recon losses same order of magnitude.

Runs: torch `p-3566fc83`, jax `p-26887a36`, wandb project `param-decomp-llama`.

**This measurement settles the open question from the 4L study.** That study could only
*speculate* that JAX's edge would shrink at 8B (the reasoning: big MLP matmuls dominate, leaving
fewer small kernels to fuse — and indeed the torch self-compile win collapses from +28% to +9%
at 8B, see the lever ladder below). The end-to-end 8B number tests it directly: the JAX edge
shrinks only **marginally** (2.45× → 2.27×), nowhere near the collapse the compile-win trend
implied. The reason is that the PD step is far more than its MLP matmuls — the **4-block,
d_model=4096 CI transformer**, the **PGD adversary loop**, the **multi-forward recon**, and the
**optimizer** all contribute kernels that JAX fuses and eager PyTorch launches individually,
plus DDP all-reduce traffic that GSPMD avoids. The gap is intrinsic to the *step's structure*,
not just its matmul count, so it does not wash out at scale.

> One-time compile cost (excluded from steady-state above): JAX paid ~12.5 min of XLA
> compilation (≈11.5 min warmup-step graph + ≈1 min main-step graph); PyTorch paid ~2 min of
> first-step CUDA init / cuDNN autotuning. Negligible amortized over a 200k-step run; relevant
> for very short jobs.

### Single-GPU lever ladders — what PyTorch can buy with effort

These are the PyTorch step-variant micro-benchmarks (single GPU). They quantify how far each
torch compilation strategy gets *on its own*, and show why none of them reach JAX.

**4L, batch 8:**

| Approach | it/s | vs eager |
|---|---|---|
| Eager | 2.73 | — |
| `compile(model + ci_fn)` only | 3.42 | +25% |
| AOTAutograd intermediate (compile loss region; eager adversary + optimizer) | 3.49 | +28% |
| Full functional rewrite (`torch.func`, one graph) | 1.35 | **−50%** |

**Llama-3.1-8B, block-18 MLP, single H100, batch 1, seq 512:**

| Approach | it/s | vs eager |
|---|---|---|
| AOTAutograd intermediate, eager | 1.47 | — |
| AOTAutograd intermediate, compiled | 1.61 | +9% |
| Full functional rewrite, eager | 0.78 | −47% |
| Full functional rewrite, compiled | 0.82 | −44% |

Two conclusions:

1. **The full functional rewrite is slower than the simple intermediate at *both* scales**
   (0.39× at 4L, 0.51× at 8B). The hope that functorch overhead would amortize at 8B is false:
   PyTorch's *emulation* of functional execution costs more than the fusion it buys.
2. **PyTorch's self-compile win shrinks with scale (+28% → +9%)** as the big matmuls come to
   dominate the step. Note this is torch-compile-vs-torch-eager — a different quantity from the
   JAX-vs-torch-eager gap, which (per the 8B end-to-end result above) stays near 2.3×.

---

## Why JAX is faster: the mechanism

JAX compiles the **entire training step** as one program:

```python
@jax.jit
def step(state, batch, key):          # state = params + opt state + PGD sources
    adv   = jax.grad(adv_recon)(...)  # PGD adversary: grad inside the step
    g, m  = jax.value_and_grad(loss)(state.params)
    state = optax_adamw_update(state, g)
    return state, m                    # XLA sees fwd + recon fwds + adversary
                                       # + backward + AdamW as ONE fused graph
```

XLA then fuses what the profiler counts as **~16.9k kernel launches/step** (~30% matmul, 50%+
unfused elementwise/copy — `mul`, `copy_` ~2356×/step, `add`, `where`) down to a handful of
fused kernels. PyTorch cannot reproduce this, for four concrete reasons.

### 1. Single-program capture *including the optimizer and cross-call state*

`torch.compile` reliably captures the model **forward** (and, via AOTAutograd, the backward of
a compiled region). It does **not** fold in `AdamW.step()` or the Python glue between the
multiple forward calls the loss needs (clean pass + stochastic-subset pass + PGD passes). Those
stay in eager Python, launching their own kernels every step.

CUDA graphs (`compile(mode="reduce-overhead")`) — torch's closest analog to whole-program
capture — **fail outright here**: the PD step calls the compiled model several times per step
and holds the outputs across calls, which trips
`"accessing tensor output of CUDAGraphs overwritten by subsequent run"` (static-buffer
aliasing). JAX's functional jit has no aliasing concept, so this failure mode does not exist.

### 2. Differentiating through the PGD adversarial inner loop, in-graph

The PGD adversary is a sign-ascent loop that itself requires gradients — nested/higher-order
autodiff *inside* the step. In JAX, `jax.grad` for the adversary composes freely under the
outer `jit`. In PyTorch:

- The eager implementation uses `tensor.requires_grad_()` + `torch.autograd.grad` in a loop.
  Dynamo **cannot trace `requires_grad_()` in fullgraph mode**
  (`"Unsupported Tensor.requires_grad_()"`), so this region cannot be compiled.
- The functional alternative, `torch.func.grad`, *can* express it — but see gap 4.

(Partial workaround, used in the intermediate: because the PGD update uses `sign(grad)`, the
outer gradient through the adversary search is zero, so the search can be detached and run eager
while only the **final** masked forward is compiled. That recovers part of the cost but leaves
the search and optimizer eager.)

### 3. Whole-step fusion across the multi-forward loss

Even setting aside the adversary and optimizer, the PD loss is several forward passes whose
intermediate activations JAX keeps resident and fuses end-to-end. PyTorch materializes and
re-reads them across eager call boundaries — the HBM-bound elementwise/copy traffic visible as
the higher memory-bandwidth utilization (~64% vs ~46%) and the thousands of `mul`/`copy_`
kernels. This is the bulk of the gap that survives at 8B, where these passes run through the big
model but the per-pass glue is still eager.

### 4. Functional-by-construction vs functional-by-emulation

JAX programs are pure: parameters are explicit arguments, there is no module state, so the
whole-step trace and nested `grad` are free. PyTorch emulates this with
`torch.func.functional_call`, which **re-binds the parameter dict into a stateful `nn.Module` on
every call** — overhead JAX never pays. Two things compound it:

- **Custom autograd Functions with a grad-direction-dependent backward.** The CI leaky-hard
  sigmoid (`LowerLeakyHardSigmoidFunction`) has a backward that branches on the *sign of the
  incoming gradient* (`where(grad_output < 0, alpha*grad, 0)`). `functorch` can't traverse a
  custom `autograd.Function` unless it's ported to the `setup_context` / `generate_vmap_rule`
  protocol (we did port it — necessary but not sufficient).
- **functorch under `torch.compile` fuses worse than plain AOTAutograd** (+5–18% vs +28%).

Net: when we *do* build the pure single-graph step the JAX way (`functional_call` +
`torch.func.grad_and_value` main grad + nested `torch.func.grad` adversary + hand-rolled
functional AdamW, all under one `torch.compile`), it is **~2× slower** than the intermediate.
The emulation cost of being functional in PyTorch exceeds the fusion benefit.

---

## Bonus: memory and sharding

Beyond throughput, JAX fits the 8B decomposition where the shipped PyTorch path does not.

- **JAX/GSPMD** shards parameters, optimizer state, and activations across the device mesh from
  a single annotated program. The 8B run peaks at **69 GB/rank** — it would fit an 80 GB H100.
- **The single-pool PyTorch `Trainer` we ship uses DDP**, which *replicates* the full parameter
  and optimizer state on every rank. The 8B, 3-target decomposition's Adam state does not fit a
  single 80 GB GPU, and adding ranks does not help (replicated, not sharded). Our 8B torch leg
  therefore requires 183 GB B200s. PyTorch *can* shard via FSDP/ZeRO in principle, but that is a
  separate trainer subsystem, not the core path — whereas JAX gets sharding essentially for free
  from GSPMD without a second code path.

> Practical note: `origin/main` cannot run the 8B config at all today — no vendored Llama, no
> pre-tokenized-parquet loader, no `weights_dtype`. The full 8B torch training path lives only
> on `feature/jax`. The 4L study established that torch-on-main ≈ torch-on-feature/jax in
> throughput, so the feature/jax torch leg is a faithful stand-in for "PyTorch performance."

---

## Caveats and fairness

- **Matched compute.** 8B legs ran on identical hardware (32×B200 / 4 nodes) and identical
  global batch (128), so a step is the same 128 sequences either side. Steady-state numbers
  exclude one-time compile/init.
- **PyTorch leg is eager.** `pd-lm` does not apply `torch.compile`. Even granting torch its best
  measured lever (+9% at 8B via the AOTAutograd intermediate), a compiled torch 8B step would be
  ~2.0 s/it — still ~2.1× slower than JAX. The conclusion is not an artifact of leaving torch
  uncompiled.
- **Different PGD recipes across scales** (fresh-PGD `n1 ss0.1` at 4L, PersistentPGD
  `broadcast_across_batch` at 8B). Both exercise the adversary-in-the-step structure that drives
  the gap; the ~2.3× holds across both.
- **Eval excluded.** The `b128_cmp32` config has no eval block, so both 8B legs are pure-train —
  no eval-overhead asymmetry. (In other configs JAX defers heavy eval metrics offline while
  torch runs them inline; that would *widen* the wall-clock gap, not narrow it.)

## Reproduction

```bash
# from feature/jax, venv active
# torch leg (stock HF Llama, eager DDP, 32 GPU / 4 nodes)
pd-lm param_decomp_jax/jax_single_pool/configs/torch/llama8b_l18_b128_cmp32_hf.yaml \
  --dp 32 --time 00:35:00 --job_name ai-rt-8b-torch-hf --group ai-rt-8b-cmp32-0616

# jax leg (same config via the shared-config wrapper, 4 nodes = 32 GPU)
pd-jax-lm param_decomp_jax/jax_single_pool/configs/llama8b_l18_b128_cmp32_hf_from_torch.yaml \
  --nodes 4 --time 00:35:00
```

Read steady-state `s/step` from each run's logs (torch: tqdm `s/it`; jax:
`train/perf/step_time_s`), dropping the first ~1–2 logged steps to exclude compile/init.

Gotchas worth knowing before you re-run:
- The shared `HF_DATASETS_CACHE` may hold a parquet builder lock owned by another user →
  `PermissionError` on every torch rank. Use a per-user `HF_DATASETS_CACHE` (set in
  `ddp_launch.py`'s `DDP_ENV`) plus `streaming: true` — never re-convert the 488 GB parquet
  artifact into a per-user arrow cache.
- Ensure `--cpu-bind=none` is present in the srun launchers (`ddp_launch.py`, `slurm.py`,
  `jax_launch.py`); without it, jobs sbatch'd from a narrow interactive allocation pin all torch
  ranks to a few cores and run ~3× slow (JAX is unaffected). This invalidated earlier torch rt
  timings.

## Bottom line

The gap is **structural**: JAX is fast because it is functional by construction and JITs the
*whole step* — forward, multi-forward loss, nested-grad adversary, backward, and optimizer —
into one XLA program that fuses thousands of small kernels, and it shards across devices from
the same program. PyTorch can fuse the feed-forward loss region (+9–28% via the AOTAutograd
intermediate) but cannot drop-in fuse the PGD adversarial autograd-in-a-loop or the optimizer,
its functional *emulation* of the JAX approach is slower than the eager baseline, and its
shipped single-pool trainer cannot even fit the 8B run on commodity 80 GB cards. The ~2.3×
per-GPU advantage holds from 4L to production 8B.

**Recommendation:** run production PD training on JAX. Keep the AOTAutograd intermediate
(compile the loss region; eager adversary + optimizer) as an optional PyTorch runtime flag for
small/local work — a low-risk +9–28%. Do **not** pursue the full functional rewrite: the
remaining gap to JAX is inherent to eager PyTorch's design.
