# Session log — full32L OOM, fixes, envelope/tp-sweep, surprise audit (2026-06-26)

Living scratch for the current session. Companion docs: `SCALING_BOOK_NOTES.md` (the
parallelism framework), `SURPRISE_AUDIT.md` (the 38-finding code audit), `NITS.md` (backlog),
`IMPROVEMENTS_CATALOGUE.md` (branch unification).

## What we established (full32L OOM)
The full-model b128 OOM was **scan-stacking of per-iteration activations in two places**, same
disease, same fix shape (whole-forward remat → per-iteration remat):
1. **target MLP intermediate** `[32, per-DP, 512, 14336]` stacked over layers — fixed by
   per-layer remat of the recon forward AND the **adversary ascents** (they were `remat=False`
   on a wrong premise; the source grad needs the per-layer acts). Commits up to `62fd7a89`.
2. **CI-fn attention scores** `[32_chunks, …, 512, 512]` f32 stacked over chunks — `remat_ci_fn`
   was checkpointing the WHOLE CI fn (never bounds the chunk scan). Fixed: per-CHUNK remat
   (`e21a1424a`).
Earlier masking bug: `remat_ci_fn` was hardcoded `False` at the call site (`c204b6afc`).

Result: 72 GiB → 53.6 GiB, but **b128 at per-DP=8 still OOMs** (130600). A remaining ~53 GiB
contributor — not yet attributed (dump at `runs/p-871d5752`). The b32/tp4 dodge (per-DP=1) fits.

## Runs in flight
| job | config | per-DP | GPUs | status |
|---|---|---|---|---|
| 130600 | b128/dp128/tp8 (complete fix) | 8 | 128 | FAILED — OOM 53.6 GiB |
| 130604 | b64/dp64/tp1 | 1 | 64 | compiling |
| 130605 | b64/dp64/tp2 | 2 | 64 | compiling |
| 130607 | b64/dp64/tp4 | 4 | 64 | compiling |
| 130609 | b64/dp64/tp8 | 8 | 64 | compiling |

The tp sweep (global batch 64, per-DP = tp, resident constant) brackets the post-fix per-DP
ceiling + gives the step-time-vs-tp curve (the deferred topology decision).

## TODOs
- [ ] **Separate compile from the faith-warmup timing.** `run.py:356` sets `t0` before the
      warmup loop, and the first `faith_warmup_step` call compiles inside the timed region — so
      "faith warmup: N steps in Ts" conflates compile + 400 steps. It inverts with tp (tp1=338s
      slowest because biggest fusions → slowest compile, not compute). Time steps 2..N (or a
      post-compile window) for a clean step-time proxy.
- [ ] Attribute the remaining ~53 GiB contributor in `runs/p-871d5752` (b128 still OOMs post-fix).
- [ ] tp-sweep verdicts + step-time curve → pick the topology.
- [x] Surprise-audit cleanup batch (Oli's decisions) — DONE, four commits:
      - `5b67c916e` wired faith-warmup `weight_decay` (silent parity bug vs torch oracle) +
        removed dead `use_fused_kl` (torch-only memory lever per lore).
      - `acba6c210` dropped torch-era `output_extract` / `activation_checkpointing` from
        `LMTargetConfig` (code-read test: never read by JAX) + strip-shim for old configs.
      - `59981db03` kept `ci_alive_threshold` PER-METRIC (matches torch oracle; reverts the
        eval-global `aa22c7f44`). The real bug: the JAX port hoisted ONE cutoff from CI_L0 and
        fed it to BOTH the L0 pass and the density slow-eval step, so the density metric's own
        `ci_alive_threshold` was dead and density silently used CI_L0's. Split the built
        EvalConfig into `l0_ci_alive_threshold` / `density_ci_alive_threshold`, each from its own
        metric. No schema removed → no shim. (Subagent confirmed main never hoisted; the
        hoist was a migration artifact.)
      - eval `batch_size`: left as-is (Oli: "probably fine"); `n_samples`: no global, nothing to do.
- [ ] Remaining audit Tier-1/2 items (`SURPRISE_AUDIT.md`): lying logs (run.py ~462/503/505),
      stale attention comment (llama8b.py ~191-211), stale file refs.

## tp>1 OOM — ROOT CAUSED + FIXED (the headline)
The full32L tp>1 OOM (`RESOURCE_EXHAUSTED: 72.41 GiB`, tp4/tp8 at b64) is **GSPMD
batch-replicating the masked-forward weight-grad backward**. The `dp` axis is overloaded:
V/U masters are FSDP-sharded on `dp` (d_in/d_out) for persistence + C on `tp`; the intent is
to gather them for compute, but `site_out`'s matmuls were unconstrained so GSPMD re-decided
per-computation — forward gathers V (batch-sharded, healthy), the rematerialized weight-grad
backward shards the activation's feature dim on `dp` and REPLICATES the global batch → a
TP↔DP reshard it can't legalize → involuntary full remat → 72 GiB. Down_proj, NOT attention
(the `rotate_half` stack frame was a lie — classify by shapes: `[448,5120]=V_down`).

**Fix (`bf072ef01`):** pin `site_out` activations data-parallel (`x->P(dp,*None)`,
`x@V->P(dp,*None,tp)`), mesh-guarded. Forces symmetric FSDP gather/reduce-scatter. remat is
the trigger so the constraint lives inside the rematted region. CPU 2×2 repro confirms
(NOFIX=global-batch dot, FIX=per-DP only); `scratchpad/reshard_repro.py`.

**Fixed sweep in flight** (b64, 64-GPU each): tp1=130616, tp2=130617, tp4=130614, tp8=130618.
Watching: involuntary_remat count (was 16 → expect 0) + OOM + step_time-vs-tp curve.
Recorded: memory `project_full32l_weight_grad_reshard_fix` + `feedback_hlo_stack_frame_unreliable`; Lore.
