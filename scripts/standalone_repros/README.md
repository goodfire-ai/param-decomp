# Standalone reproducers for the 3-pool training hot paths

Each file in this directory builds the relevant `nn.Module`s at exact production
scale (matching `param_decomp_lab/experiments/lm/_xl_production/gpt2_xl_qk_smoke.yaml`)
and times the phase in a single process. Useful for distinguishing real GPU work
from misattributed waits in the distributed trainer (where `torch.profiler` can
deadlock against NCCL).

| Script                  | Phase reproduced            | Production observed |
|-------------------------|-----------------------------|---------------------|
| `ci_fn_bwd.py`          | `ci/8a_bwd_lower_leaky_only`  | ~626 ms             |
| `lw_d3_layerwise.py`    | `lw/D3_layerwise` (8 sites)   | ~748 ms             |
| `pgd_d3_warmup.py`      | `pgd/D3_warmup` (2 warmup)    | ~400 ms             |

Each script does 3 warmup iters, 10 timed iters, then a short `torch.profiler`
window. Outputs a `*_trace.json` (chrome trace) and `*_output.txt` (key_averages
tables) next to itself.

Run from the repo root via:

```bash
source .venv/bin/activate
srun --gres=gpu:1 --time=20:00 python scripts/standalone_repros/<name>.py
```

The LW and PPGD scripts build a randomly-initialized `GPT2Simple` at gpt2-xl
scale (no HF download). The CI fn script doesn't need a target model at all.
