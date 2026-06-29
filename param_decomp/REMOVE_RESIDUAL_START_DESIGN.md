# Remove residual-start / prefix-suffix — full-model-only engine

Status: in progress (branch `feature/jax-full32L-port`). SPEC-amending — see §SPEC below.

## Why

The "residual-start" design splits the target at `first_layer` into a **prefix** (embedding +
layers `0..first_layer-1`, run once per batch to harvest the residual) and a **suffix** (the
decomposed tail). It buys one thing: for a *partial* decomposition you avoid re-running the
frozen lead layers every step.

For the runs we actually care about — **full-model** decompositions — `first_layer = 0`, so the
prefix is just the embedding and the machinery buys nothing. What it costs:

- A first-class `Prefix` object + `prefix_residual` harvest + `sample_batch -> residual` engine
  seam + `first_layer` offset arithmetic smeared through every forward method.
- A hard ceiling on activation tapping: nothing inside the prefix is reachable (the earliest tap
  is `resid.{first_layer}`), because the prefix runs outside the step graph and its intermediates
  are discarded.

Per the repo's YAGNI / "collapse variant behaviour" rules, this is optionality that the common
case doesn't use. Remove it; express the rare case ad-hoc (see §Partial decomposition).

## The corrected abstraction

The key realization: the engine never actually needs the **residual tensor** or the architecture's
internal width `d`. It only needs:

1. **`inputs`** — a per-step batch (tokens for an LM), to hand to the model's forwards. Must be a
   pure function of `step` for O(1) SLURM-requeue resume.
2. **`leading`** — the `(batch, *position)` shape that masks / sources / routes live in
   (`[*leading, C]`). This is data-shape, **not** the residual width `d`.

`leading` is available from the **taps** the engine already computes — taps are `[*leading, d_tap]`,
so `leading = next(iter(taps.values())).shape[:-1]` is target-agnostic (taps always have a trailing
feature dim). The residual stream + `d` stay **entirely inside the target module**.

So the engine becomes **input-and-`leading`-driven, blind to `d`**:

```
inputs = token_batch(step)                       # step-indexed (resume); tokens, not a residual
taps   = model.read_activations(inputs, names)   # leading = a_tap.shape[:-1]
ci     = ci_fn(taps)
masks  = build from (leading, ci)
clean  = model.clean_output(inputs)              # model embeds internally
masked = model.masked_output(vu, inputs, masks, …)
```

The model's forward methods take `inputs` (Any) and embed internally (LM) or treat `inputs` as the
waist directly (TMS, where the input already is `[B, d]`). Re-embedding across `clean_output` /
`read_activations` / `masked_output` is one gather on identical indices — XLA CSE dedups it within
the step; negligible.

## Surgery by file

1. `targets/llama8b.py` — fold `embed` (the token→residual lookup) onto `LlamaDecomposedModel`;
   forward methods take `inputs` and embed internally. Delete `first_layer` field + every
   `± self.first_layer`; delete `Prefix`, `load_prefix_from_hf`, `prefix_residual`,
   `make_real_target_residual`, `first_decomposed_layer`, and `_load_suffix_layers`'s slicing (load
   all layers). Site names use absolute layer index = the model's own index.
2. `targets/llama_simple_mlp.py` — same surgery (it carries its own `first_layer` + embedding).
3. `run.py` (engine) — loop holds `inputs = token_batch(step)`; `leading` from taps; pass `inputs`
   (not a residual) to the forwards. Swap the `sample_batch: (int)->residual` arg for the
   `(int)->tokens` source. Engine no longer names "residual".
4. `experiments/lm/run.py` (composition root) — delete prefix build / harvest / `prefix_residual_fn`;
   `build_target` returns `(lm, vocab)`; the data source is the step-indexed `ShardServer.local_batch`.
5. `lm.py` — protocol methods take `inputs`; drop `prefix`/`AnyPrefix`; doc the input-driven contract.
6. `targets/target_aliases.py` — drop `AnyPrefix`.
7. Bench scaffolding (`experiments/llama8b_real.py`, `experiments/mem_probe.py`) — `first_layer`
   references; fix or delete (low priority, not the production path).

## SPEC

Amend the `residual-start` glossary entry and §1.1 pseudocode: the model is the full stack,
embedding-owned; the engine consumes tokens and operates in `leading`-space; there is no prefix /
`first_layer`. Deliberate amendment per the one rule (cite in the commit).

## Partial decomposition (the dropped capability)

Decomposing only a subset of layers still **works** unchanged: layers with no decomposed sites run
their frozen `x @ W` path in the masked forward (the `live_kinds`-empty branch). It just runs the
lead layers frozen every step instead of caching them — the perf we deprioritized. If that perf is
ever wanted, express it ad-hoc with the existing seam: a custom model that *is* the suffix + a data
source that maps the real prefix (embed + frozen layers) over tokens. No library support needed; the
original layer index survives only as interpretability metadata (a label), not a forward-threaded field.

## Out of scope / still open

- **The 192 GiB main-step OOM is orthogonal** — it lives in the masked-forward scan, untouched by
  this refactor. Still needs a memory-dump diagnosis before any 128-GPU relaunch.
- **Test goldens** (`stacked_parity`, `equivalence`) reference the suffix / `first_layer`. For
  `first_layer=0` fixtures the numerics are identical (full model ≡ embedding-prefix + suffix); a
  `first_layer>0` fixture changes the computed quantity and the test is adapted. Assessed before edits.

## Sequencing

This refactor first → then the clean scan (`clean_output` + single scan-only masked forward) on the
simplified model → then OOM diagnosis + GPU re-validation.
