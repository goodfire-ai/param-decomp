# Investigation plan: empirical validation of SPD scaling estimates

**For:** an agent (or engineer) with cluster access, picking up this work after the FSDP feasibility report.
**Companion document:** `fsdp_scaling_report.html` in the repo root. Read it first.

---

## What this plan is for

The report makes specific numerical claims that were derived from algebra plus a single calibration anchor (Jose runs at `batch_size: 64` on H100). Your job is to **replace those estimates with measured numbers**, and to **stress-test the architectural claims about FSDP friction** before anyone commits to a refactor.

The deliverable is a results document (`investigation_results.md` in the repo root) saying which predictions held, which didn't, and what should change in the report.

You are **not** trying to make FSDP work end-to-end yet. That's a much bigger lift that depends on the architectural refactor in §8 Phase 2 of the report. We're validating premises, not building.

---

## Prerequisites

- Cluster access with SLURM
- `.venv` in the repo root or a worktree (`uv sync` if missing)
- WandB credentials in `.env` (see `.env.example`)
- 1× H100 80GB allocation for Phase 1 (memory profiling)
- 8× H100 80GB allocation for Phase 2+ (need >1 rank to test ZeRO-1)
- Read access to the existing Jose run (`goodfire/spd/runs/s-55ea3f9b`) and its target (`goodfire/spd/runs/t-9d2b8f02`) for cross-checks
- ≤8 GPUs at any one time (project policy from `CLAUDE.md`)

Activate the venv before running anything:

```bash
source .venv/bin/activate
```

---

## Working style

- One feature branch per phase: `feature/profile-memory`, `feature/zero-1`, etc. Don't merge to main without review.
- Each phase ends with: (a) a short writeup appended to `investigation_results.md`, and (b) a PR that's at least open, even if not merged.
- Each phase's runs go to a dedicated WandB project (`spd-scaling-investigation`) so they don't pollute the main project's view.
- All overrides via the existing config system. Don't hardcode anything; flag everything.
- Profiling runs should be **20–50 steps**, not 400k. We're measuring memory and step time, not training to convergence.

---

## Phase 1 — baseline memory profile of Jose

**Goal:** get an actual per-rank memory breakdown at Jose scale. Verify or refute the report's ~13 GB fixed-state and ~1 GB/batch-element activation estimates.

### 1.1 Code changes

Add two config flags to `Config` in `param_decomp/configs.py`:

```python
profile_memory: bool = False
profile_memory_step: int = 30
```

In `param_decomp/run_param_decomp.py`, wrap the training loop:

```python
# After optimizer setup, before `for step in tqdm(...)`
if config.profile_memory and is_main_process():
    torch.cuda.memory._record_memory_history(max_entries=200_000)
    log_param_breakdown(component_model)  # see below

# Inside the loop, after the optimizer.step()
if (
    config.profile_memory
    and step == config.profile_memory_step
    and is_main_process()
):
    snap_path = out_dir / "memory_snapshot.pickle"
    torch.cuda.memory._dump_snapshot(str(snap_path))
    torch.cuda.memory._record_memory_history(enabled=None)
    logger.info(f"Memory snapshot dumped to {snap_path}")
    logger.info(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    logger.info(f"\n{torch.cuda.memory_summary(abbreviated=True)}")
```

Helper (anywhere sensible — `param_decomp/utils/general_utils.py` is fine):

```python
def log_param_breakdown(component_model: ComponentModel) -> None:
    target_p = sum(p.numel() for p in component_model.target_model.parameters())
    component_p = sum(
        p.numel()
        for n in component_model.target_module_paths
        for p in component_model.components[n].parameters()
    )
    ci_p = sum(p.numel() for p in component_model.ci_fn.parameters())
    logger.info(f"target_params:    {target_p:>14,}  ({target_p * 4 / 1e9:.2f} GB fp32)")
    logger.info(f"component_params: {component_p:>14,}  ({component_p * 4 / 1e9:.2f} GB fp32)")
    logger.info(f"ci_fn_params:     {ci_p:>14,}  ({ci_p * 4 / 1e9:.2f} GB fp32)")
    logger.info(f"trainable total:  {component_p + ci_p:>14,}")
```

### 1.2 Run

```bash
pd-local pile_llama_simple_mlp-4L \
  --override 'steps=50,eval_freq=1000000,save_freq=null,profile_memory=true,profile_memory_step=30,wandb_project=spd-scaling-investigation,wandb_run_name_prefix=phase1-baseline-'
```

If `pd-local` doesn't accept `--override` syntax, check `param_decomp/scripts/run_local.py` — it might be `--config-overrides` or you might need a separate YAML override file. **Don't guess; read the script.**

If running on the cluster but not via SLURM (i.e. you `srun --pty bash` into a node), `pd-local` should work directly. Otherwise use `pd-run` and check the SLURM logs.

### 1.3 Analysis

Pull the snapshot file locally (or open it on the cluster if you have a remote browser) and load it at <https://pytorch.org/memory_viz>. You want:

- **Top-line:** peak memory over the run (call it `M_peak`)
- **Param breakdown** from `log_param_breakdown` output
- **Memory category breakdown** from the visualizer:
  - parameters
  - gradients
  - optimizer state (AdamW m + v)
  - activations (the big saved-for-backward tensors)
  - workspace/temporary
- **Step time** averaged over steps 30–50

### 1.4 What to write up

Append to `investigation_results.md`:

```markdown
## Phase 1: baseline Jose profile

**Setup:** 1× H100 80GB, batch=64, steps=50, sample at step 30.

**Param counts (measured):**
- target: ___ M
- components: ___ M
- ci_fn: ___ M
- trainable total: ___ M

**Memory at peak:** ___ GB
- params: ___ GB
- grads: ___ GB
- optimizer state: ___ GB
- activations: ___ GB
- workspace: ___ GB

**Predicted vs measured:**
- Fixed state: predicted 13 GB, measured ___ GB. (Δ ___ )
- Activation/B: predicted 1.0 GB, measured ___ GB. (Δ ___ )

**Step time:** ___ ms

**Recommended report updates:**
- (list any predictions that were materially wrong)
```

### 1.5 Success criterion

You can write the sentence: *"On 1×H100, Jose at batch 64 uses peak X GB; fixed state is Y GB; activation/B is Z GB."* And `(M_peak − fixed_state) / 64` is within ±50% of the predicted 1 GB/B.

If it's outside ±50%, something's wrong with the report's calibration. **Flag it, don't paper over it.**

---

## Phase 2 — ZeRO-1

**Goal:** confirm the predicted `8·T·(N−1)/N` per-rank saving (≈5.25 GB at Jose scale, N=8). Verify save/load still works.

### 2.1 Code changes

Add to `Config`:

```python
optimizer_strategy: Literal["adamw", "zero_adamw"] = "adamw"
```

At the optimizer construction site in `param_decomp/run_param_decomp.py` (around line 207):

```python
if config.optimizer_strategy == "adamw":
    optimizer = optim.AdamW(optimized_params, lr=config.lr_schedule.start_val, weight_decay=0)
elif config.optimizer_strategy == "zero_adamw":
    from torch.distributed.optim import ZeroRedundancyOptimizer
    assert dist_state is not None and dist_state.world_size > 1, (
        "ZeRO-1 requires a distributed run with world_size > 1"
    )
    optimizer = ZeroRedundancyOptimizer(
        optimized_params,
        optimizer_class=optim.AdamW,
        lr=config.lr_schedule.start_val,
        weight_decay=0,
    )
```

**Things to verify:**
1. The manual LR-schedule loop at lines 254–255 (`group["lr"] = step_lr`) still works. ZeRO exposes `param_groups` so it should — but test it.
2. `clip_grad_norm_(component_params, ...)` at lines 418–421: still works under ZeRO-1 because params and grads are still replicated. Confirm with a quick print of the grad norm before and after a couple of steps.
3. State-dict round-trip: if any code path saves optimizer state (currently looks like only model state is saved at line 405 — verify), it needs `optimizer.consolidate_state_dict(to=0)` first.

### 2.2 Run

```bash
pd-local pile_llama_simple_mlp-4L --dp 8 \
  --override 'steps=50,eval_freq=1000000,save_freq=null,optimizer_strategy=zero_adamw,profile_memory=true,profile_memory_step=30,wandb_run_name_prefix=phase2-zero1-'
```

### 2.3 Analysis

- Per-rank peak memory under ZeRO-1 (call it `M_zero`)
- Saving: `M_peak (Phase 1) − M_zero`
- Predicted: 5.25 GB at Jose scale, N=8
- Step time delta vs Phase 1 — ZeRO-1 has a small extra all-gather of params at optimizer step time, expect 5–15% slowdown

### 2.4 Sanity check across N

Run the same with `--dp 4` and `--dp 2` if budget allows. Predicted savings:
- N=2: 4·T = 3.0 GB
- N=4: 6·T = 4.5 GB
- N=8: 7·T = 5.25 GB

If the saving doesn't scale as `(N−1)/N`, something's wrong with the implementation.

### 2.5 Failure modes to watch
- ZeRO-1 + per-param-group LRs has historically had bugs. Print the LR after `optimizer.zero_grad()` to confirm it sticks.
- If clip_grad_norm reports a norm that's identical to the AdamW baseline, good. If it reports something different, investigate before trusting any of the rest.

---

## Phase 3 — gradient checkpointing on the CI transformer

**Goal:** quantify the activation-memory reduction from checkpointing the CI transformer's blocks. The CI transformer is the largest single contributor to per-step activation memory (it's a ~400M-param transformer over batch×seq).

### 3.1 Code changes

Find `GlobalSharedTransformerCiFn`:

```bash
grep -rn "class GlobalSharedTransformerCiFn" param_decomp/
```

Locate its block list (probably `nn.ModuleList` of transformer blocks). In its forward, replace the block iteration:

```python
# Before
for block in self.blocks:
    x = block(x)

# After
from torch.utils.checkpoint import checkpoint
if self.gradient_checkpointing:
    for block in self.blocks:
        x = checkpoint(block, x, use_reentrant=False)
else:
    for block in self.blocks:
        x = block(x)
```

Add `gradient_checkpointing: bool = False` to the appropriate config (probably `simple_transformer_ci_cfg`). Wire it through.

**Same exercise for the target model's transformer blocks** if you want to test that lever too — find `LlamaSimpleMLP` (probably in `param_decomp/pretrain/models/`) and apply the same pattern. Add a separate flag `target_gradient_checkpointing`.

### 3.2 Caveat: frozen target activations

The target is frozen, so its activations don't have grads. `torch.utils.checkpoint` with `use_reentrant=False` should handle this fine — but if you see `RuntimeError: element 0 of tensors does not require grad`, fall back to `use_reentrant=True` or wrap the checkpoint call to manually re-enable grad on the input.

### 3.3 Run

Two configurations stacked on top of Phase 2's ZeRO-1:

```bash
# CI fn checkpointing
pd-local pile_llama_simple_mlp-4L --dp 8 \
  --override '...optimizer_strategy=zero_adamw,ci_config.simple_transformer_ci_cfg.gradient_checkpointing=true,profile_memory=true,...'

# Both
pd-local pile_llama_simple_mlp-4L --dp 8 \
  --override '...optimizer_strategy=zero_adamw,ci_config.simple_transformer_ci_cfg.gradient_checkpointing=true,target_gradient_checkpointing=true,profile_memory=true,...'
```

### 3.4 Analysis

For each:
- Peak memory delta vs Phase 2 baseline
- Step time delta (expect ~25–50% slowdown on the checkpointed forward, less on overall step)
- Loss over the first 30 steps — must match the no-checkpointing run within numerical tolerance. If it diverges, something's wrong with the rng or a re-entrancy bug.

### 3.5 Maximum-batch test

After confirming functional correctness, find the max batch size that fits on a single H100 with both checkpointing flags on, by binary search:

```bash
pd-local pile_llama_simple_mlp-4L --dp 8 \
  --override 'batch_size=128,...'  # try 128, 96, 80, etc.
```

This is the empirical answer to the "max batch with grad ckpt" column in §3 of the report.

---

## Phase 4 — weight-delta rewrite

**Goal:** validate the algebraic identity `x @ (W − V@U) = x @ W − (x @ V) @ U` rewrite, both for correctness and for memory.

### 4.1 The math, carefully

Current scheme in `LinearComponents.forward` (`param_decomp/models/components.py:438–480`):

```
out = (masked_acts) @ U                            # masked component path
    + delta_mask * (x @ (W_target − V@U))          # delta path, with materialized delta
    + bias
```

where `masked_acts = (x @ V) * mask`.

Rewrite (using `unmasked_acts = x @ V`):

```
out = (masked_acts) @ U
    + delta_mask * (x @ W_target)
    − delta_mask * (unmasked_acts @ U)
    + bias
```

Both are algebraically identical. The new form never materializes the `(d_out, d_in)` delta tensor — it computes it implicitly via two matmuls.

This means:
- `calc_weight_deltas()` at `run_param_decomp.py:263` no longer needs to compute the delta. We just need to pass the target weight (or a reference to it) into the component forward.
- The `WeightDeltaAndMask` namedtuple becomes `(target_weight_ref, delta_mask)` or similar.

### 4.2 Implementation

Probably the cleanest change:
1. Add a method `LinearComponents.forward_with_target_weight(x, mask, delta_mask, W_target)` that does the new math.
2. Keep the old `forward` path for backward compat behind a feature flag (e.g. `config.materialize_weight_delta: bool = True`).
3. When `materialize_weight_delta=False`, skip `calc_weight_deltas()` entirely and have the hook pass `W_target` reference into the components.
4. The `target_weight()` method already exists — use it.

### 4.3 Test before running

Before any cluster run, write a unit test in `tests/test_components.py`:

```python
def test_weight_delta_rewrite_equivalence():
    torch.manual_seed(0)
    d_in, d_out, C = 32, 48, 16
    batch = (4, 8)
    x = torch.randn(*batch, d_in)
    W_target = torch.randn(d_out, d_in)
    delta_mask = torch.rand(*batch)
    mask = torch.rand(*batch, C)
    bias = torch.randn(d_out)

    comp = LinearComponents(C=C, d_in=d_in, d_out=d_out, bias=bias)
    weight_delta = W_target - comp.weight

    out_old = comp.forward(x, mask=mask, weight_delta_and_mask=(weight_delta, delta_mask))
    out_new = comp.forward_with_target_weight(x, mask=mask, delta_mask=delta_mask, W_target=W_target)

    torch.testing.assert_close(out_old, out_new, atol=1e-5, rtol=1e-5)
```

Run it: `python -m pytest tests/test_components.py::test_weight_delta_rewrite_equivalence -v`

**Don't proceed to a cluster run until this test passes.**

### 4.4 Run

```bash
pd-local pile_llama_simple_mlp-4L --dp 8 \
  --override '...optimizer_strategy=zero_adamw,materialize_weight_delta=false,profile_memory=true,...'
```

### 4.5 Analysis

- Memory saved by removing `calc_weight_deltas()` materialization (~10 modules × ~16M floats × 2 bytes ≈ 320 MB at Jose scale, per the report's estimate)
- Step time delta — adding two matmuls per decomposed module per forward, but saving an O(d_in · d_out) materialization. Could go either way.
- Loss trajectory must match the old scheme within numerical tolerance — diverging means the test wasn't tight enough.

---

## Phase 5 (stretch) — 1B-target stress test

**Goal:** the smallest run that exercises the predicted DDP failure mode at 1B target scale.

### 5.1 Construct a 1B target

Two routes:

- **Route A (slow but principled):** train a deeper LlamaSimpleMLP via the existing pretrain pipeline. ~16 layers at d_model=2048, d_mlp=8192 lands around 1B params. See `param_decomp/pretrain/CLAUDE.md` for setup.
- **Route B (faster, riskier):** find an existing 1–1.5B Llama on HuggingFace whose architecture matches what `param_decomp.experiments.lm` supports (target modules: `nn.Linear`, `nn.Embedding`, `Conv1D`). Pick e.g. `meta-llama/Llama-3.2-1B` if licensing allows, or `EleutherAI/pythia-1.4b`.

Route B is faster but introduces unrelated risks (tokenizer mismatch, attention pattern differences). Probably worth doing first as a quick failure-mode probe.

### 5.2 Build a config

Copy `pile_llama_simple_mlp-4L.yaml` and modify:
- `pretrained_model_name`: point to the new target
- `module_info`: list the modules to decompose (probably just MLPs at first; attn is optional)
- Reduce `C` if needed to keep components manageable
- Set `steps=20`, `eval_freq=10000`, `save_freq=null`

Register it in `param_decomp/registry.py` as `lm_1b_probe` or similar.

### 5.3 Run, three configurations

```bash
# DDP only — expected: OOM
pd-local lm_1b_probe --dp 8 --override 'profile_memory=true,...'

# ZeRO-1 — expected: barely fits, batch ~13
pd-local lm_1b_probe --dp 8 --override 'optimizer_strategy=zero_adamw,profile_memory=true,...'

# ZeRO-1 + checkpointing — expected: comfortable, batch ~25
pd-local lm_1b_probe --dp 8 --override 'optimizer_strategy=zero_adamw,ci_config.simple_transformer_ci_cfg.gradient_checkpointing=true,target_gradient_checkpointing=true,profile_memory=true,...'
```

### 5.4 Analysis

Compare measured per-rank memory and max batch against the report's §3 predictions for the 1B-target row. Deltas of ±20% on memory and ±50% on batch are within tolerance. Outside that range, the report needs updating.

If even ZeRO-1 + checkpointing OOMs at 1B, that's a stronger signal than the report predicts and probably means activation/B grew faster than √(target). Document this — it's the most important finding in the whole investigation.

---

## Reporting

After each phase, append to `investigation_results.md` and (separately) post a 3–5 sentence Slack-style summary noting:
- Predicted vs measured (one number, one delta)
- Any surprises
- Whether the report needs updating

After all phases, do a single PR that updates `fsdp_scaling_report.html` with the measured numbers replacing the estimates. Specifically the §3 tables and the §7a saving claim.

---

## Things to absolutely not do

- **Don't try to make FSDP work end-to-end yet.** That's §8 Phase 2 of the report's plan. We're validating premises, not building.
- **Don't run full Jose (400k steps) for any of this.** 20–50 steps is enough for memory and step-time measurement.
- **Don't merge to main without review.** One PR per phase, against feature branches.
- **Don't burn the cluster.** Each phase should be ≤30 minutes of cumulative compute. Track usage; the project policy is ≤8 GPUs at any time.
- **Don't paper over surprising results.** If a measured number disagrees materially with the report's prediction, write it up clearly. The report is wrong, not your measurement (unless your measurement is obviously wrong, in which case dig in).
- **Don't commit secrets.** `.env` is gitignored; don't add WandB tokens to any tracked file.

---

## Useful references

- `fsdp_scaling_report.html` — the document this plan validates
- `CLAUDE.md` — repo conventions, environment setup, SLURM guidelines
- `param_decomp/utils/distributed_utils.py` — distributed helpers
- `param_decomp/run_param_decomp.py` — main loop, where most edits happen
- `param_decomp/models/component_model.py` — `ComponentModel`, hooks, `calc_weight_deltas`
- `param_decomp/models/components.py` — `LinearComponents`, the `weight` property (Phase 4)
- `param_decomp/experiments/lm/pile_llama_simple_mlp-4L.yaml` — the Jose config
- PyTorch memory profiling: <https://pytorch.org/blog/understanding-gpu-memory-1/> and <https://pytorch.org/memory_viz>
- ZeroRedundancyOptimizer docs: <https://pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer>

