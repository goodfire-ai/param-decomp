# FSDP Scaling Branch Review

Branch: `feature/fsdp-wrap`  
Date: 2026-05-12  
Scope: review of recent work against `main`, focused on the goal of scaling SPD to larger target models.

This file consolidates the branch-level context from:

- `scaling_investigation_plan.md`
- `investigation_results.md`
- `fsdp_implementation_plan.md`
- `fsdp_scaling_summary.md`
- `fsdp_scaling_report.html`

It also records the three highest-priority correctness issues found in review and the proposed fixes.

## Executive Summary

The branch has made real progress toward larger-model scaling:

- ZeRO-1 was validated empirically and gives the expected optimizer-state savings.
- The old hook-based decomposition architecture was replaced with fused `DecomposedLinear` / `DecomposedEmbedding` sites.
- FSDP2 wrapping now fits much larger synthetic targets in smoke tests, including a 4B target on H200 where ZeRO-1 OOMed.
- CI-function gradient checkpointing produces a large measured activation-memory reduction.

However, the current branch is not ready to merge for general training. Three P1 issues need to be fixed first:

1. `target_gradient_checkpointing` is incompatible with stateful fused decomposition-site binding.
2. The current FSDP weight-delta path materializes detached full deltas on every rank, losing gradients and reintroducing memory pressure.
3. Normal DDP training regresses because loss code receives a `DistributedDataParallel` wrapper where it expects a `ComponentModel`.

The recommended direction is to keep the fused-site architecture, but make the decomposition context explicit enough for checkpoint recomputation, move delta-component math inside each site, and preserve the underlying `ComponentModel` handle for DDP loss code.

## Consolidated Background

### Original Scaling Problem

SPD training makes multiple `ComponentModel` forwards per optimizer step:

- target/cache forward
- PPGD warmup forwards
- reconstruction or stochastic loss forwards
- evaluation forwards

This creates a high activation peak. The empirical Jose baseline showed roughly 3 GB of activation/transient memory per per-rank batch element, materially above the earlier estimate of about 1 GB/B.

At larger target sizes, ZeRO-1 helps but only shards optimizer state. It does not shard params, grads, or activations. The investigation found that full parameter sharding is needed for 4B-scale targets.

### Empirical Results To Carry Forward

Jose-scale baseline, 1xH200, per-rank batch 8:

| Bucket | Measured |
|---|---:|
| Target params | 67M / 0.27 GB fp32 |
| Trainable params | 660M / 2.64 GB fp32 |
| AdamW state | 5.28 GB |
| Fixed state total | 10.83 GB |
| Peak memory | 34.62 GB |
| Activation/transient cost | about 3 GB per per-rank batch element |

ZeRO-1 validation:

| World size | Measured saving | Predicted saving |
|---:|---:|---:|
| 2 | 2.63 GB | 2.64 GB |
| 4 | 3.98 GB | 3.96 GB |

FSDP2 smoke tests, N=4, per-rank batch 1, no target block checkpointing:

| Target | Trainable | Decomposed fwd+bwd peak |
|---|---:|---:|
| 1B | 2.89B | 26.73 GB |
| 2B | 4.44B | 43.44 GB |
| 4B | 6.97B | 72.17 GB |

CI-function gradient checkpointing reduced those FSDP smoke-test peaks by about 61%.

### Architecture Change

The old architecture stored components in a sibling `ModuleDict` and used target-module forward hooks to replace outputs. That is awkward for FSDP because the hook can cross FSDP-unit boundaries: the target module may be gathered while the sibling components are still sharded.

The branch replaces that with fused decomposition sites:

- `target_model.<path>` is replaced in place by a `DecomposedLinear` or `DecomposedEmbedding`.
- Each site owns both the frozen target submodule and its trainable `V` / `U` parameters.
- `ComponentModel.forward` temporarily binds per-site `mask_info` and cache state, then calls the target model normally.
- FSDP2 wraps the decomposition sites and the CI-function transformer blocks.

This is the right broad direction. The issues below are integration problems around checkpointing, delta handling, and DDP compatibility.

## Finding 1: Target Gradient Checkpointing Is Not Safe With Stateful Site Binding

Severity: P1

### Problem

`ComponentModel.forward` binds `mask_info`, cache dict, and cache type onto each decomposed site using a context manager:

```python
with ExitStack() as stack:
    for site_name, site in self.components.items():
        mask_info = mask_infos.get(site_name) if mask_infos is not None else None
        stack.enter_context(site.bind(mask_info, bound_cache, bound_cache_type))
    out = self._run_batch(self.target_model, batch)
```

`LlamaSimpleMLP.forward` now supports target block checkpointing:

```python
if self.config.gradient_checkpointing:
    x = checkpoint(block, x, use_reentrant=False)
else:
    x = block(x)
```

During backward, checkpoint recomputes `block(x)` after the original forward context has exited. The recomputation sees each decomposed site's `_mask_info` reset to `None`, so it runs the frozen target path instead of the decomposed path.

In a minimal repro, masked checkpointed forward failed with:

```text
CheckpointError: torch.utils.checkpoint: A different number of tensors was saved during the original forward and recomputation.
```

Even if a specific case did not error, the recomputation would be using the wrong forward path.

### Recommended Fix

Make the decomposition forward context explicit and checkpoint-aware.

Create a small context helper, likely in a new module such as `param_decomp/models/decomp_forward_context.py`, that can bind the same site state for both original forward and checkpoint recomputation.

Sketch:

```python
class DecompForwardContext:
    def __init__(
        self,
        sites: Mapping[str, DecomposedSite],
        mask_infos: Mapping[str, ComponentsMaskInfo] | None,
        cache_type: CacheType | None,
        cache: dict[str, Tensor] | None,
    ) -> None:
        ...

    @contextmanager
    def bind(self, *, cache: dict[str, Tensor] | None) -> Iterator[None]:
        with ExitStack() as stack:
            for site_name, site in self.sites.items():
                mask_info = self.mask_infos.get(site_name) if self.mask_infos is not None else None
                stack.enter_context(site.bind(mask_info, cache, self.cache_type))
            yield
```

Then route the context through the target model during the forward. For checkpointed target blocks, use PyTorch's `checkpoint(..., context_fn=...)`:

```python
def context_fn():
    return nullcontext(), decomp_ctx.bind(cache={})

x = checkpoint(block, x, use_reentrant=False, context_fn=context_fn)
```

The recompute cache should usually be a dummy dict, not the original cache, to avoid mutating training caches during backward. But the recompute must preserve the same `cache_type`, because `cache_type="component_acts"` changes graph structure by detaching and re-enabling gradients on component activations.

### Simpler Fallback

If the context plumbing is too invasive for the current merge, explicitly reject the unsafe combination:

```python
assert not (config.target_gradient_checkpointing and any_masked_decomposed_forward), (
    "target_gradient_checkpointing is not compatible with fused decomposition sites yet"
)
```

That is safe but weakens the larger-model scaling story. The checkpoint-aware context is the better fix.

### Tests To Add

- A target model with a checkpointed block containing a decomposed site.
- Compare masked forward/backward with checkpointing on vs off.
- Assert `V.grad` and `U.grad` are non-`None` and numerically close.
- Repeat for `cache_type="input"`.
- Add coverage for `cache_type="component_acts"` if that path is used in training/eval.

## Finding 2: FSDP Weight Deltas Are Detached, Replicated, And Too Expensive

Severity: P1

### Problem

The current FSDP path computes deltas with:

```python
weight_deltas = calc_weight_deltas_full(component_model)
```

`calc_weight_deltas_full`:

- unshards every module with `unshard`, not only decomposition sites
- computes `component_model.calc_weight_deltas()`
- calls `detach()`
- materializes `.full_tensor()` on every rank
- reshares afterward

This has three problems:

1. Faithfulness-loss gradients through weight deltas are lost under FSDP.
2. Full deltas are replicated on every rank, reintroducing large per-rank memory.
3. The helper unshards more than necessary, including CI-function FSDP units.

The branch comments acknowledge part of this:

```python
# known correctness regression under FSDP
```

That should not remain as the training behavior.

### Recommended Fix

Stop passing full `weight_deltas` through the training loss path. Instead, move delta-component math inside each decomposition site, where FSDP has already gathered the site's target weight and `V` / `U`.

Change `ComponentsMaskInfo` from carrying:

```python
weight_delta_and_mask: tuple[Tensor, Tensor] | None
```

to carrying something like:

```python
delta_mask: Tensor | None
```

Then in `DecomposedLinear._decomposed_forward`:

```python
component_acts = x @ self.V
masked_component_out = (component_acts * component_mask) @ self.U

if info.delta_mask is not None:
    target_out = self.linear(x)
    unmasked_component_out = component_acts @ self.U
    masked_component_out = masked_component_out + (
        info.delta_mask[..., None] * (target_out - unmasked_component_out)
    )
```

For `DecomposedEmbedding`:

```python
component_acts = self.V[idx]
masked_component_out = (component_acts * component_mask) @ self.U

if info.delta_mask is not None:
    target_out = self.embedding(idx)
    unmasked_component_out = component_acts @ self.U
    masked_component_out = masked_component_out + (
        info.delta_mask[..., None] * (target_out - unmasked_component_out)
    )
```

This avoids materializing `W_target - V @ U` globally. It also keeps the computation inside the FSDP unit that owns the relevant parameters.

### Faithfulness Loss Under FSDP

Do not keep the current detached full-delta path for training.

Two acceptable options:

1. Short-term: explicitly disallow `FaithfulnessLoss` and `faithfulness_warmup_steps > 0` under `parallel_strategy="fsdp"`, and require `use_delta_component=True`.
2. Longer-term: implement a per-site differentiable faithfulness scalar that is computed inside each site's gathered forward context and reduced across sites.

The short-term assert is preferable to silent incorrect gradients.

### Keep `calc_weight_deltas_full` Only For Eval/Debug

If full deltas remain needed for eval or diagnostics:

- rename the helper to make its cost obvious, for example `calc_weight_deltas_full_for_eval`
- restrict unsharding to `DecomposedLinear | DecomposedEmbedding`
- keep it out of training loss code

### Tests To Add

- Unit tests showing `delta_mask` site-local path matches old `weight_delta_and_mask` outputs and V/U gradients on small linear and embedding modules.
- A no-FSDP training-step test with `use_delta_component=True` that does not call `calc_weight_deltas()`.
- An FSDP smoke test that verifies no detached full-delta path is used during training.
- An assertion test that FSDP plus `FaithfulnessLoss` fails loudly until a differentiable FSDP-safe faithfulness path exists.

## Finding 3: DDP Loss Code Receives The Wrong Object

Severity: P1

### Problem

In the training loop, the branch now does:

```python
forward_model = cast(ComponentModel, wrapped_model)
```

Under FSDP2 this works because `fully_shard` mutates the `ComponentModel` in place, so `wrapped_model` is still effectively the same object.

Under DDP, `wrapped_model` is a `DistributedDataParallel` wrapper. Loss code and PPGD code expect a `ComponentModel` and dereference attributes such as:

- `target_module_paths`
- `module_to_c`
- `components`
- `target_model`

For example, persistent PPGD calls:

```python
routing_masks = router.get_masks(
    module_names=model.target_module_paths,
    mask_shape=batch_dims,
)
```

A DDP wrapper does not expose those attributes, so normal DDP training with PPGD-style losses will regress.

### Recommended Fix

Keep two handles:

- `wrapped_model`: only for the forward that must pass through DDP/FSDP machinery
- `component_model`: the real `ComponentModel`, used for attributes and loss-code calls

The smallest fix is:

```python
target_model_output = wrapped_model(batch, cache_type="input")

if config.parallel_strategy == "fsdp":
    forward_model = component_model
else:
    forward_model = component_model
```

That looks redundant, but it makes the important point explicit: loss code receives the `ComponentModel`, not the DDP wrapper.

The existing target/cache forward through `wrapped_model` should still be enough to set up DDP reducer state, matching the old behavior.

### Optional Proxy If Needed

If tests show all loss forwards must go through the DDP wrapper for gradient synchronization, introduce a narrow proxy:

```python
class ForwardingComponentModelProxy:
    def __init__(self, wrapped: nn.Module, component_model: ComponentModel) -> None:
        self._wrapped = wrapped
        self._component_model = component_model

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._wrapped(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._component_model, name)
```

Avoid this unless it is necessary. It is more magical than simply passing `component_model` to loss code.

### Tests To Add

- A 2-process DDP CPU or CUDA smoke test with a loss that reads `target_module_paths`.
- A PPGD or persistent-PPGD DDP smoke test if feasible.
- A regression test that `compute_losses(..., model=...)` receives an object exposing `ComponentModel` attributes in DDP mode.

## Follow-Up Findings Not Covered By The First Three

These are secondary but should be tracked.

### FSDP Checkpoint Save Path Is Misleading

At final step, `should_save` is true even with `save_freq=None`. Under FSDP, no file is written, but rank 0 still logs that the model was saved and may call `wandb.save` on a non-existent path.

Short-term fix: in the FSDP branch, skip the save/log/upload block entirely or log a clear "FSDP checkpoint save skipped" message.

Long-term fix: implement FSDP2 full state dict save with `torch.distributed.checkpoint.state_dict.get_model_state_dict` and `StateDictOptions(full_state_dict=True, cpu_offload=True)`.

### Checkpoint Migration Is Incomplete

State dict keys now live under:

```text
target_model.<site>.V
target_model.<site>.U
target_model.<site>.linear.weight
```

The deprecated-key mapper still targets `_components.*`, which no longer exists. Old `main` checkpoints likely cannot load.

Fix by updating `handle_deprecated_state_dict_keys_` to map old component keys to the new fused-site keys.

## Recommended Implementation Order

1. Fix DDP handle regression first. It is small, easy to test, and restores existing behavior.
2. Replace the FSDP training delta path with site-local `delta_mask` handling. Add asserts for FSDP plus faithfulness loss until a proper differentiable path exists.
3. Implement checkpoint-aware decomposition context for target block checkpointing.
4. Clean up FSDP checkpoint-save behavior.
5. Add checkpoint migration for old runs.

## Verification Plan

Minimum local tests:

```bash
uv run pytest tests/test_component_model.py tests/test_components.py tests/metrics/test_recon_losses.py tests/metrics/test_faithfulness_loss.py
```

Additional targeted tests to add before merge:

- checkpointed fused-site backward equivalence
- site-local delta-component equivalence
- DDP loss-code smoke test
- FSDP smoke test without full replicated deltas in training

Cluster validation after local tests:

1. Jose-scale FSDP N=4 with CI-function checkpointing.
2. Jose-scale DDP N=2 or N=4 with PPGD loss to verify DDP behavior did not regress.
3. 1B target FSDP N=4, per-rank batch 1 or 2.
4. 4B target FSDP N=4, per-rank batch 1.

## Consolidation And Cleanup Map

Recommended canonical docs after this review:

| File | Status | Recommendation |
|---|---|---|
| `fsdp_scaling_branch_review.md` | New canonical branch review | Keep |
| `fsdp_scaling_summary.md` | Useful but now partially superseded | Fold into this file, then archive/delete |
| `investigation_results.md` | Detailed empirical record | Move under `docs/scaling/` or keep as historical appendix |
| `fsdp_implementation_plan.md` | Partially stale after review findings | Replace with a shorter implementation checklist based on this file |
| `scaling_investigation_plan.md` | Historical execution plan | Archive/delete once results are preserved |
| `fsdp_scaling_report.html` | Original report artifact, currently untracked | Archive under `docs/scaling/` if still needed, otherwise delete |
| `scripts/bench_activation_breakdown.py` | Useful investigation tool | Keep if documented, otherwise move under `scripts/scaling/` |
| `scripts/bench_weight_delta_rewrite.py` | Useful negative-result benchmark | Keep as regression/benchmark evidence |
| `scripts/fsdp_memory_smoke.py` | Useful FSDP smoke test | Keep and document |
| `scripts/make_random_target.py` | Useful scaling utility | Keep and document |
| `scripts/analyze_memory_snapshot.py` | Useful profiling utility | Keep and document |

Do not delete the older files until the team decides whether they want historical detail preserved in git. The immediate cleanup should be to make this file the branch review entry point and link or move the older docs under a `docs/scaling/` archive.

