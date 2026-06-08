# spd-ppgd-nwarmup0 — drop PPGD inner warmup steps (2 → 0)

- **Claim type:** speedup
- **Stage:** proposed → T0
- **Branch / worktree:** `feature/spd-ppgd-nwarmup0` (off `feature/track2-setup`)
- **Owner agent:** setup session (first experiment)

## Hypothesis
`PersistentPGDReconLoss.n_warmup_steps` runs extra inner PGD source-optimization steps on each
train batch before the loss. The baseline uses 2. Each costs a masked component forward — a real
per-step cost. If the persistent sources carry enough state across steps, the warmup steps may be
redundant: **n_warmup_steps 2 → 0 should cut per-step time with little PPGD-recon/L0 cost.** A pure
config change (no core edit) — also the first end-to-end shake-out of the harness + judging.

## Diff vs baseline
Config only: `ss_llama_simple_mlp-2L-ppgd-nwarmup0.yaml` = `…-2L-baseline.yaml` with
`n_warmup_steps: 0` (one line). Eval block unchanged (ruler).

## Success / kill thresholds (from track2/README.md)
- **≥10% speedup** (step time, `pd-speedup-bench`) — else not worth it.
- Faithfulness **gate** held (CI-masked CE/KL within the 3-seed T0 band).
- Primary (PPGD recon + L0) within band or better, judged at ~50k vs the 3-seed baseline.

## Results (artifacts required)

### T0 (`ss_llama_simple_mlp-2L`, 50k/b32)
- benchmark (variant vs baseline step time): _TBD_
- `pd-speedup-compare p-db5adc3b,p-89a42376,p-993ae14a <run> --at_step 50000`: _TBD_
- run: _TBD_ (run_id + wandb URL)

### T1 — only if T0 promising.

## Verdict
_TBD_
