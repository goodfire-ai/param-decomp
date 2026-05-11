# Implementation plan: FSDP-with-fused-decomposition-sites

**For:** an engineer picking up the next chunk of work after the scaling investigation
(`scaling_investigation_plan.md`, `investigation_results.md`,
`fsdp_scaling_report.html`).

**Goal:** make SPD trainable at 1B+ targets on 80 GB H100s and 4B+ targets on
~150 GB H200s. The empirical case is in `investigation_results.md` — ZeRO-1 +
gradient checkpointing on the CI fn is sufficient through ~1B target on H100s
but OOMs at 4B even on H200. Full parameter sharding (FSDP) is required.

**Vibe:** one larger jump in code complexity rather than several incremental
steps. Throw the current hook-based ComponentModel architecture out and replace
with one where each decomposition site is a single first-class `nn.Module`
that FSDP can wrap cleanly. The CI fn stays a separate top-level module — it's
global by construction (reads concatenated activations from *all* sites and
emits CI scores for all sites at once), so there's no per-site CI piece to
fuse. Its constituent transformer blocks become independent FSDP units via
the auto-wrap policy.

---

## Background: why a refactor (not just FSDP-wrap-the-existing-model)

§5b of `fsdp_scaling_report.html` is the load-bearing observation: components
live in a `nn.ModuleDict` that's a *sibling* of the target model. Forward
replacement is a post-forward hook registered on the target submodule
(`param_decomp/models/component_model.py:489–501`) which dispatches into
`self._components[module_name]`. Under FSDP that hook crosses FSDP-unit
boundaries — the components' parameters live in a different FSDP unit than
the target module the hook is attached to, so when the hook fires the
components' sharded parameters are not gathered.

Three plausible fixes (§5b of the report, ranked by invasiveness):
1. **Fuse target submodule + its V/U components into a single `nn.Module`
   per decomposed site, FSDP-wrap each.** Clean. Eliminates the
   hook-crossing-units problem entirely. **The plan.** (Note: the report's
   §5b also mentions fusing in "its CI piece" — that's incorrect; the CI fn
   is global, not per-site.)
2. `summon_full_params` inside the hook. Works but defeats most of FSDP's
   benefit (per-call all-gather of components on the hot path).
3. Don't shard components. Doesn't work at 1B+ target where components are
   too big to leave replicated.

The other big architectural friction is `calc_weight_deltas()` at
`run_param_decomp.py:263`, which materializes
`weight_delta = W_target - V@U` once per step for every decomposed module.
Under FSDP, `V`, `U`, and `W_target` are all sharded; reading them as
properties returns local shards and the math is silently wrong or shape-errors
out. The fused-site design lets us do the delta math *inside* the site's
forward, where FSDP has already gathered both the target's weight and the
components' V/U.

(NB: §7d of the report originally proposed an algebraic rewrite to avoid
materializing the delta entirely. The investigation showed that rewrite is
strictly *worse* on memory and wall-time at SPD-realistic batch sizes
(`investigation_results.md` §4). Keep the materialized form, just do it
inside the fused module.)

---

## The new module: `DecomposedLinear`

Replaces the current `(nn.Linear in target, LinearComponents in sibling
ModuleDict, hook gluing them)` triple. One Python class per decomposed module
in the model tree.

```python
# param_decomp/models/decomposed_module.py  (new file)

class DecomposedLinear(nn.Module):
    """A linear module that can dispatch through a parameter decomposition.

    Owns:
      - The original `nn.Linear` from the target model (frozen).
      - V: (d_in, C), U: (C, d_out)            — the components.
      - bias: shared with the wrapped Linear (or None).

    Drop-in replacement for `nn.Linear` in the target tree: when no
    `mask_info` is set on the module, forward is exactly `self.linear(x)`.
    When a mask_info is set (via `set_mask_info()` context-manager pattern,
    see below), forward dispatches to the decomposed path.
    """

    def __init__(self, base: nn.Linear, C: int) -> None:
        super().__init__()
        self.linear = base                          # frozen target nn.Linear (its .weight stays the target weight)
        self.d_in, self.d_out = base.in_features, base.out_features
        self.C = C
        self.V = nn.Parameter(torch.empty(self.d_in, C))
        self.U = nn.Parameter(torch.empty(C, self.d_out))
        init_param_(self.V, fan_val=self.d_in, nonlinearity="linear")
        init_param_(self.U, fan_val=C, nonlinearity="linear")
        # bias is read from `self.linear.bias` directly — no duplicate parameter.
        self._mask_info: ComponentsMaskInfo | None = None
        self._cache: dict[str, Tensor] | None = None
        self._cache_type: CacheType | None = None

    def set_mask_info(self, info: ComponentsMaskInfo | None) -> None:
        self._mask_info = info

    def set_cache(self, cache: dict[str, Tensor] | None, kind: CacheType | None) -> None:
        self._cache, self._cache_type = cache, kind

    def forward(self, x: Tensor) -> Tensor:
        if self._cache is not None and self._cache_type == "input":
            self._cache[self._site_name] = x

        if self._mask_info is None:
            out = self.linear(x)
        else:
            out = self._decomposed_forward(x, self._mask_info)

        if self._cache is not None and self._cache_type == "output":
            self._cache[self._site_name] = out
        return out

    def _decomposed_forward(self, x: Tensor, info: ComponentsMaskInfo) -> Tensor:
        component_acts = einops.einsum(x, self.V, "... d_in, d_in C -> ... C")
        # (component_acts_cache handling stays the same — write into self._cache here.)
        masked_acts = component_acts * info.component_mask if info.component_mask is not None else component_acts
        components_out = einops.einsum(masked_acts, self.U, "... C, C d_out -> ... d_out")

        if info.weight_delta_and_mask is not None:
            weight_delta, delta_mask = info.weight_delta_and_mask
            delta_out = einops.einsum(x, weight_delta, "... d_in, d_out d_in -> ... d_out")
            components_out = components_out + einops.einsum(
                delta_mask, delta_out, "..., ... d_out -> ... d_out"
            )

        if self.linear.bias is not None:
            components_out = components_out + self.linear.bias

        if info.routing_mask == "all":
            return components_out
        else:
            return torch.where(info.routing_mask[..., None], components_out, self.linear(x))

    def calc_weight_delta(self) -> Tensor:
        """Materialize W_target - V@U for this site. Cheap, called once per step.

        Under FSDP this runs inside the site's FSDP unit so V/U/target_weight are
        all gathered. Stays an O(d_out · d_in) materialization per site (~9 MB
        at Jose dims, ~constant cost) — the alternative rewrite was empirically
        worse, see investigation_results.md §4.
        """
        return self.linear.weight - einops.einsum(self.V, self.U, "d_in C, C d_out -> d_out d_in")
```

(`DecomposedEmbedding` will be needed too for runs that decompose embeddings —
similar pattern. Skip for the first pass since Jose doesn't decompose embeddings.)

---

## What `ComponentModel` becomes

`ComponentModel` is currently a fairly heavyweight class: it owns the target
model + the components ModuleDict + the CI fn, registers hooks on demand, and
exposes `calc_weight_deltas`, `calc_causal_importances`, `target_weight`,
several `__call__` overloads, and the from_pretrained / from_run_info loading.

After the refactor it shrinks substantially:

- **`self.target_model`** — still here, but its decomposable submodules have
  been *replaced in place* by `DecomposedLinear` instances. So
  `target_model.h.0.mlp.c_fc` *is* a `DecomposedLinear`, not an `nn.Linear`
  with a hook attached.
- **`self.ci_fn`** — unchanged.
- **No more `self._components` ModuleDict.** The components are owned by
  their respective `DecomposedLinear` sites in the target tree.
- **No more `_attach_forward_hooks`.** `forward(...)` walks the tree, sets
  `mask_info` and `cache` slots on each `DecomposedLinear` via a context
  manager, calls `self.target_model(x)`, then clears the slots. Concretely:

  ```python
  @contextmanager
  def _bind_per_site(self, mask_infos, cache, cache_type):
      sites = list(self._iter_sites())
      try:
          for site in sites:
              site.set_mask_info(mask_infos.get(site._site_name) if mask_infos else None)
              site.set_cache(cache, cache_type)
          yield
      finally:
          for site in sites:
              site.set_mask_info(None)
              site.set_cache(None, None)
  ```

- **`calc_weight_deltas()`** becomes a one-liner that walks the sites and
  calls `site.calc_weight_delta()` on each.
- **`target_weight(name)`** becomes `self.get_site(name).linear.weight`.
- **`calc_causal_importances()`** is essentially unchanged — it consumes the
  cached input activations and runs the CI fn over them.
- **State-dict** is now naturally tree-shaped (V/U live under
  `target_model.h.0.mlp.c_fc.V` etc.) rather than spread across two trees with
  name mangling (`_components["h-0-mlp-c_fc"].V`).

The `__call__` overloads stay, the user-facing API is unchanged.

---

## Site discovery & construction

Today `ComponentModel.__init__` calls `_create_components` which iterates
`module_path_info` and builds the components ModuleDict. The refactored
version does the same iteration but *replaces* each named submodule in
`target_model` with a `DecomposedLinear` wrapping the original:

```python
# pseudocode
for module_path, C in module_path_info:
    base = target_model.get_submodule(module_path)
    assert isinstance(base, nn.Linear)
    parent_path, _, child = module_path.rpartition(".")
    parent = target_model.get_submodule(parent_path)
    new = DecomposedLinear(base, C=C)
    new._site_name = module_path        # used for mask_info lookup and caching
    setattr(parent, child, new)
```

This is the equivalent of the existing `insert_identity_operations_` style of
in-place tree rewrite (`param_decomp/identity_insertion.py`) — Oli's used this
pattern before, and pyright/torch are both fine with it.

(`Conv1D` and `nn.Embedding` would need their own `DecomposedConv1D` /
`DecomposedEmbedding` siblings if those are decomposed. Punt on the second
one; flag a TODO. Jose only decomposes `nn.Linear`s.)

---

## FSDP wrap policy

```python
# param_decomp/utils/fsdp.py  (new file)

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

def fsdp_wrap(model: ComponentModel, dist_state: DistributedState) -> ComponentModel:
    """Wrap with one FSDP unit per (DecomposedLinear, CI-fn-block, target-transformer-block).

    Leaves frozen target submodules that aren't decomposed (embedding layer, ln_f,
    rms norms, etc.) unsharded — they're small enough that the per-step all-gather
    cost would exceed the saving. The frozen target's transformer blocks are
    NO_SHARD-policy FSDP units: parameters stay on each rank but the wrap gives us
    a uniform module-tree shape for the auto-wrap policy.
    """
    from param_decomp.models.decomposed_module import DecomposedLinear
    from param_decomp.models.components import TransformerBlock

    def policy(module, recurse, nonwrapped_numel):
        if recurse:
            return True
        return isinstance(module, (DecomposedLinear, TransformerBlock))

    return FSDP(
        model,
        auto_wrap_policy=policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ) if config.autocast_bf16 else None,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        use_orig_params=True,                       # critical: see notes
    )
```

Knob notes:
- **`use_orig_params=True`** is critical so the optimizer construction in
  `run_param_decomp.py:198–207` (which builds `optimized_params` from
  `component_model.components[name].parameters() + ci_fn.parameters()`) keeps
  working without rewriting how params are collected.
- **`MixedPrecision`** is the proper FSDP replacement for the current
  `bf16_autocast` context manager and the §7c "bf16 master weights" lever
  rolled together. Activations end up in bf16, gradients reduced in bf16,
  parameters stored in fp32 (the FSDP default unless we set
  `MixedPrecision(param_dtype=bf16)`, which we leave for later if needed).
- **Don't outer-wrap with FSDP again.** The whole `ComponentModel` doesn't
  need a top-level FSDP unit — the auto-wrap policy will pick up the
  meaningful units inside.

---

## Replacing DDP and ZeRO-1 in the training loop

Today's `run_param_decomp.py:165–183` wraps with DDP. Today's also-still-there
ZeRO-1 path (added during the investigation) constructs
`ZeroRedundancyOptimizer` instead of plain AdamW when `optimizer_strategy="zero_adamw"`.

After the refactor:

```python
match config.parallel_strategy:
    case "ddp":
        wrapped_model = torch.nn.parallel.DistributedDataParallel(model, ...)
        optimizer = optim.AdamW(optimized_params, ...)
    case "zero_adamw":
        wrapped_model = DDP(...)
        optimizer = ZeroRedundancyOptimizer(optimized_params, optimizer_class=optim.AdamW, ...)
    case "fsdp":
        wrapped_model = fsdp_wrap(model, dist_state)
        optimizer = optim.AdamW(wrapped_model.parameters(), ...)   # use_orig_params=True keeps this working
```

(So the existing `optimizer_strategy` config field gets renamed to
`parallel_strategy` with a third option. ZeRO-1 stays as a stop-gap for
small targets / quick local runs where the FSDP comm cost isn't worth it.)

The downstream bits to also change in `run_param_decomp.py`:
- **`clip_grad_norm_`** at lines 418–421: under FSDP this needs to be
  `wrapped_model.clip_grad_norm_(max_norm)` — FSDP's `clip_grad_norm_` does
  the cross-rank reduction itself.
- **`component_model.state_dict()`** at line 405 (save path): wrap in
  `FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
  FullStateDictConfig(offload_to_cpu=True, rank0_only=True))` context.
- **The "must call wrapped_model once" line at run_param_decomp.py:268–272**
  is a DDP-ism (DDP requires forward to touch every parameter). FSDP doesn't
  have this constraint. Delete it under FSDP. Keep under DDP/ZeRO-1.

---

## Caller-side changes (the long tail)

Files that touch the components dict directly:

```
param_decomp/run_param_decomp.py
param_decomp/eval.py
param_decomp/persistent_pgd.py
param_decomp/plotting.py
param_decomp/adapters/param_decomp.py
param_decomp/app/backend/optim_cis.py
param_decomp/app/backend/compute.py
param_decomp/clustering/activations.py
param_decomp/dataset_attributions/harvester.py
param_decomp/editing/_editing.py
param_decomp/harvest/harvest_fn/param_decomp.py
param_decomp/metrics/*.py            (many)
```

Most of these access patterns are of three shapes:
1. `model.components[name].V` / `.U` — read component params.
2. `model.target_module_paths` — iterate site names.
3. `model.calc_weight_deltas()` — get all deltas.

Each becomes:
1. `model.get_site(name).V` / `.U` — read params via the new accessor.
2. `model.target_module_paths` — unchanged (the iteration order is the same;
   the helper just walks `DecomposedLinear` instances now).
3. `model.calc_weight_deltas()` — unchanged signature; implementation walks
   the sites.

Plan: add `ComponentModel.get_site(name) -> DecomposedLinear` and
`ComponentModel.components` as a `@property` returning a `dict[str, DecomposedLinear]`
view, then mechanically convert callers. Most don't need to know the underlying
architecture changed — they just want V, U, the weight, or the components dict.

**Audit step:** before doing this, grep for *direct mutation* of
`model.components` or `model._components` (not just reads). I think the only
mutator outside `ComponentModel.__init__` is `run_param_decomp.py:189–196`
(tied_weights). Confirm with `grep`.

---

## Save/load compatibility with existing checkpoints

The state-dict layout *will* change:

```
# old
target_model.h.0.mlp.c_fc.weight
_components.h-0-mlp-c_fc.V
_components.h-0-mlp-c_fc.U

# new
target_model.h.0.mlp.c_fc.linear.weight
target_model.h.0.mlp.c_fc.V
target_model.h.0.mlp.c_fc.U
```

Jose (`s-55ea3f9b`) and Thomas (`s-82ffb969`) are real runs whose checkpoints
people will want to reuse — both for analysis (app, harvest, autointerp) and
for re-derivation. We need a one-shot migration.

Plan: `param_decomp/scripts/migrate_checkpoint_to_decomposed.py` that loads an
old `final_config.yaml` + state-dict, constructs the new `ComponentModel`
layout, and remaps state-dict keys:
```
target_model.<path>.weight  →  target_model.<path>.linear.weight
target_model.<path>.bias    →  target_model.<path>.linear.bias
_components.<mangled>.V     →  target_model.<path>.V
_components.<mangled>.U     →  target_model.<path>.U
```

Run once per existing run, write new checkpoints alongside the old (or to a
sibling `migrated/` dir; don't overwrite). Update `ParamDecompRunInfo.from_path`
to prefer the migrated checkpoint when present.

Don't add a fallback that loads old checkpoints with new code — fail fast if a
non-migrated old checkpoint is passed.

---

## Rollout sequence

Three branches off `main`, each landing as its own PR. Each one passes the
existing test suite before the next starts.

### PR 1 — no-op refactor (no FSDP yet)

Lands the `DecomposedLinear` module + `ComponentModel` rewrite + caller updates,
but keeps DDP/ZeRO-1 as the only parallel strategies. The training loop and the
behaviour of every existing metric, loss, harvest job, and app endpoint should
be *functionally identical* after this PR.

Acceptance:
- All existing tests pass (`make test`).
- A 50-step `pile_llama_simple_mlp-4L` run produces the same loss trajectory
  as `main` (within numerical noise) on the same seed.
- App still loads Jose and shows the same components/CIs.
- Migration script works end-to-end on Jose's checkpoint.

This is the chunky PR — probably 1500-2500 LOC changed across the caller-list
above. Most of it mechanical (rename `model._components[name]` →
`model.get_site(name)`).

### PR 2 — FSDP path

Adds the `fsdp_wrap` helper, the `parallel_strategy="fsdp"` config option,
and the FSDP-specific `clip_grad_norm_` and state-dict save paths in
`run_param_decomp.py`. Keeps DDP/ZeRO-1 working unchanged.

Acceptance:
- Phase 5 1B-target run completes under FSDP at meaningful batch
  (target: ≥b=4 per rank on 4×H200; ≥b=2 per rank on 4×H100).
- 4B-target run *completes at all* on 4×H200 (this is the empirical "is FSDP
  enough" test — the investigation showed ZeRO-1+ckpt can't).
- Loss trajectory under FSDP matches the DDP no-op-refactor baseline within
  numerical tolerance for the first ~50 steps on Jose.
- Save and reload a checkpoint under FSDP and verify forward equivalence.

This PR is smaller — probably 200-500 LOC.

### PR 3 — cleanup

After PR 2 has been used in anger for a real run:
- Remove the `forward_with_target_weight` method on `LinearComponents` (it
  was a Phase 4 dead end, kept around with tests as a regression guard).
- Decide whether to keep ZeRO-1 as an option. If FSDP works fine even at
  Jose scale, drop ZeRO-1 — fewer code paths.
- Update `fsdp_scaling_report.html` with measured FSDP numbers (the final
  thing in `scaling_investigation_plan.md`'s deliverables).

---

## Risks and things to verify before writing PR 1

1. **PPGD autograd interactions.** `persistent_pgd.py:175` runs
   `torch.autograd.grad(loss, self.sources)` where `loss` flows through
   `ComponentModel` forwards. Under FSDP the graph crosses FSDP boundaries.
   Should work because sources aren't params — but worth a small standalone
   test before committing to the refactor.

2. **Tied weights** at `run_param_decomp.py:186–196`. The current scheme ties
   `U` of one component to `V.T` of another. Under FSDP these need to be in
   the same FSDP unit, which they currently aren't (different sites). Either
   (a) handle in `DecomposedLinear` by tying inside the site (probably not
   what Jose uses — its config doesn't show tied_weights), or (b) require
   `parallel_strategy != "fsdp"` when tied_weights is set. Verify Jose
   doesn't use tied_weights; deal with the general case later.

3. **`use_orig_params=True` cost.** FSDP's `use_orig_params=False` is the
   memory-efficient default but breaks the existing optimizer-param-list
   pattern. `use_orig_params=True` keeps the param identity stable but
   each FlatParameter is split per-rank in optimizer state. Small extra
   bookkeeping cost. Try this first; only switch off if it's measurably
   slow.

4. **`set_mask_info` + tree iteration on each step.** Currently the
   "attach hooks" context manager registers callbacks once per step.
   The new "set attribute, run forward, clear attribute" pattern does the
   same number of operations but via Python attribute assignment which is
   marginally faster. Should not measurably affect step time.

5. **Harvest hooks** (`param_decomp/harvest/harvest_fn/param_decomp.py`).
   The harvest pipeline registers its own hooks for collecting statistics.
   These currently land on the original target submodules. After the
   refactor they should land on `site.linear` (the inner wrapped target
   Linear), not on the `DecomposedLinear` itself, because we want the
   *target* output (not the decomposed output) for harvest. Audit.

---

## Out of scope (and why)

- **bf16 master weights** (§7c of the report). Subsumed by FSDP's
  `MixedPrecision(param_dtype=bf16)`. Don't add separately.
- **Tensor parallelism / pipeline parallelism / activation offload**
  (report §8 Phase 3). Only relevant if FSDP itself proves insufficient at
  scale. Wait for the Phase 5 FSDP measurement before opening this can.
- **Weight-delta algebraic rewrite** (report §7d). Empirically worse, see
  `investigation_results.md` §4. Don't ship.
- **Sharding the CI fn separately from the components.** Tempting because
  the CI fn is the biggest single per-batch activation contributor, but the
  auto-wrap policy above already wraps each CI-transformer block as its own
  FSDP unit, which is the equivalent.

---

## Estimated effort

- PR 1 (no-op refactor + migration script): 4–7 days, mostly mechanical
  caller-rewriting once the core pieces are in.
- PR 2 (FSDP path): 1–2 days.
- PR 3 (cleanup + report update): 1 day.

Total: ~1–2 calendar weeks of focused work for one engineer.

---

## References

- `investigation_results.md` — measurements that justify the refactor.
- `fsdp_scaling_report.html` — §5 (the SPD-specific frictions), §6 (full
  friction-point checklist), §8 Phase 2 (the original sketch of this
  refactor).
- `param_decomp/models/component_model.py` — the file that shrinks most.
- `param_decomp/models/components.py` — where `LinearComponents` lives today
  (it doesn't go away; its responsibilities just get folded into
  `DecomposedLinear`).
- `param_decomp/identity_insertion.py` — example of in-place tree rewrite
  (the pattern `DecomposedLinear` swap uses).
