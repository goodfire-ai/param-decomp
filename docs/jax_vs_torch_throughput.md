# JAX vs PyTorch throughput: why JAX is faster, and what it can do that PyTorch can't

Investigation of the per-GPU throughput gap on the PD training step (fresh-PGD `n1 ss0.1`
recipe). Measured on `pile_llama_simple_mlp-4L` and re-checked at `Llama-3.1-8B` (single
block decomposed).

## Key results

**4L model, 8-GPU matched tier (samples/s/GPU):**

| Backend | it/s | per-GPU vs JAX |
|---|---|---|
| JAX (`@jax.jit` whole step) | 9.78 | 1.00× |
| PyTorch DDP (eager) | 3.99 | 0.41× |
| PyTorch FSDP (speed-tuned) | 4.16 | 0.43× |

JAX is **~2.3× faster per GPU**, robust across the 8- and 16-GPU tiers. Not an idle-GPU
artifact: with the CPU-affinity bug fixed, torch runs ~97–100% GPU util but at lower power
(~540 W vs ~670 W) and higher memory-bandwidth util (~64% vs ~46%) — the classic signature
of many small, HBM-bound eager kernels vs whole-program fusion.

**Single-GPU lever ladder (4L, batch 8):**

| Approach | it/s | vs eager |
|---|---|---|
| Eager | 2.73 | — |
| `compile(model + ci_fn)` only | 3.42 | +25% |
| AOTAutograd intermediate (compile loss region; eager adversary + optimizer) | 3.49 | +28% |
| Full functional rewrite (`torch.func`, one graph) | 1.35 | **−50%** |

**Llama-3.1-8B, block-18 MLP decomposed (single H100, batch 1, seq 512):**

| Approach | it/s | vs eager |
|---|---|---|
| AOTAutograd intermediate, eager | 1.47 | — |
| AOTAutograd intermediate, compiled | 1.61 | +9% |
| Full functional rewrite, eager | 0.78 | −47% |
| Full functional rewrite, compiled | 0.82 | −44% |

Two conclusions:

1. **The full functional rewrite is slower than the simple intermediate at *both* scales**
   (0.39× at 4L, 0.51× at 8B). The hope that functorch overhead would amortize at 8B is
   false. PyTorch's *emulation* of functional execution costs more than the fusion it buys.
2. **The compile win shrinks with scale (+28% → +9%).** At 8B the big MLP matmuls dominate
   the step, so there are far fewer small elementwise kernels left for any compiler to fuse.
   By the same logic, JAX's per-GPU edge is expected to shrink at 8B too.

## What JAX does that PyTorch can't (the actual mechanism)

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

XLA then fuses what the profiler counts as **~16.9k kernel launches/step** (~30% matmul,
50%+ unfused elementwise/copy — `mul`, `copy_` ~2356×/step, `add`, `where`) down to a
handful of fused kernels. PyTorch can't reproduce this because of three concrete capability
gaps:

### 1. Single-program capture *including the optimizer and cross-call state*

`torch.compile` reliably captures the model **forward** (and, via AOTAutograd, the backward
of a compiled region). It does **not** fold in the `AdamW.step()` or the Python glue between
the multiple forward calls the loss needs (clean pass + stochastic-subset pass + PGD passes).
Those stay in eager Python, launching their own kernels every step.

CUDA graphs (`compile(mode="reduce-overhead")`) — torch's closest analog to whole-program
capture — **fail outright here**: the PD step calls the compiled model several times per step
and holds the outputs across calls, which trips
`"accessing tensor output of CUDAGraphs overwritten by subsequent run"` (static-buffer
aliasing). JAX's functional jit has no aliasing concept, so this failure mode doesn't exist.

### 2. Differentiating through the PGD adversarial inner loop, in-graph

The PGD adversary is a sign-ascent loop that itself requires gradients — i.e.
higher-order/nested autodiff *inside* the step. In JAX, `jax.grad` for the adversary composes
freely under the outer `jit`. In PyTorch:

- The eager implementation uses `tensor.requires_grad_()` + `torch.autograd.grad` in a loop.
  Dynamo **cannot trace `requires_grad_()` in fullgraph mode**
  (`"Unsupported Tensor.requires_grad_()"`), so this region can't be compiled.
- The functional alternative, `torch.func.grad`, *can* express it — but see gap 3.

(Partial workaround, used in the intermediate: because the PGD update uses `sign(grad)`, the
outer gradient through the adversary search is zero, so the search can be detached and run
eager while only the **final** masked forward is compiled. That recovers part of the cost but
leaves the search and optimizer eager.)

### 3. Functional-by-construction vs functional-by-emulation

JAX programs are pure: parameters are explicit arguments, there is no module state, so the
whole-step trace and nested `grad` are free. PyTorch emulates this with
`torch.func.functional_call`, which **re-binds the parameter dict into a stateful `nn.Module`
on every call** — an overhead JAX never pays. Compounding it, two things resist the functional
path:

- **Custom autograd Functions with a grad-direction-dependent backward.** The CI leaky-hard
  sigmoid (`LowerLeakyHardSigmoidFunction`) has a backward that branches on the *sign of the
  incoming gradient* (`where(grad_output < 0, alpha*grad, 0)`). `functorch` can't traverse a
  custom `autograd.Function` unless it's ported to the `setup_context` / `generate_vmap_rule`
  protocol (we did port it — necessary but not sufficient).
- **functorch under `torch.compile` fuses worse than plain AOTAutograd.** Compiling the
  functional step gave only +5–18%, vs +28% for the AOTAutograd intermediate.

Net: when we *do* build the pure single-graph step the JAX way (`functional_call` +
`torch.func.grad_and_value` main grad + nested `torch.func.grad` adversary + hand-rolled
functional AdamW, all under one `torch.compile`), it is **~2× slower** than the intermediate.
The emulation cost of being functional in PyTorch exceeds the fusion benefit.

## Bottom line

The gap is **structural**: JAX is fast because it is functional by construction and JITs the
*whole step* — forward, multi-forward loss, nested-grad adversary, backward, and optimizer —
into one XLA program that fuses thousands of small kernels. PyTorch can fuse the feed-forward
loss region (+28% at 4L via the AOTAutograd intermediate) but cannot drop-in fuse the PGD
adversarial autograd-in-a-loop or the optimizer, and its functional *emulation* of the JAX
approach is slower than the eager baseline it was meant to beat.

**Recommendation:** ship the AOTAutograd intermediate (compile the loss region; keep the
adversary search and optimizer eager) as a torch runtime flag — a low-risk +9–28% depending
on scale. Do **not** pursue the full functional rewrite. The remaining gap to JAX is inherent
to eager PyTorch's design and shrinks at larger model scale where matmuls dominate.
