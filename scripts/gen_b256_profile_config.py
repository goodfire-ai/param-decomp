"""Generate the 64-GPU profiling variant of the b256 GPT-2 XL Q/K 3-pool config.

Why a separate variant: the b256 production topology is 160 GPUs (96 LW @ 1 site/block
+ 32 CI + 32 PPGD). The profiling budget is <=100 GPUs (and the launcher forces multiples
of 8, so effectively <=96). We cannot reproduce b256's exact memory signature
(`bl_lw=256`, all 96 sites, `bl_ci=bl_pp=8`) under that budget, because the CI fn and PPGD
per-rank cost scale with the *number of sites* — so trimming sites would make those two
pools unrepresentative. Instead we KEEP all 96 sites and DROP the batch 256 -> 64, which
lets CI and PPGD fit at the b256-verified `bl=8` with only 8 ranks each, freeing GPUs for
LW. LW runs 48 blocks x 1 rank x 2 sites/block (`bl_lw=64`, still exercises LW activation
checkpointing). Total = 48 + 8 + 8 = 64 GPUs.

This run measures the cross-pool wait/overlap structure (prong 2). The per-pool *compute
floors* at true production scale (`bl_lw=256` LW, 96-site CI/PPGD) come from the single-GPU
probes (`scripts/probe_{lw_rank,ci_fn,ppgd}_bl_ceiling.py --profile`), not from this run.

Base = an existing generated b256 run config (it carries the full CI fn, all 96 sites, the
LW-only activation checkpointing, and seq_len 1024). LRs are irrelevant to a compute
profile, so we reuse the base's.
"""

import copy
from pathlib import Path
from typing import Any

import yaml

BASE = Path("param_decomp_lab/experiments/lm/_b256_run/b256_lrA_comp5e-4.yaml")
OUT = Path("param_decomp_lab/experiments/lm/_b256_run/b256_profile_64.yaml")

SITES = [f"h.{layer}.attn.{p}_proj" for layer in range(48) for p in ("q", "k")]
assert len(SITES) == 96

N_LW = 48  # 48 blocks, 1 rank/block, 2 consecutive sites/block -> all 96 sites
N_CI = 8  # bl_ci = 64 / 8 = 8 (CI pool fit fine at bl=8 in the 64-GPU run)
# PPGD's masked full-model recon over all 96 sites is the memory wall: bl_pp=8 OOM'd an
# 80GB H100 in the live 3-pool (run 603077), even though the standalone probe fit at bl=8
# (the probe doesn't hold the received full-model CI leaves + the fused V/U+CI+sources graph).
# So PPGD alone runs at bl_pp=4 (n_ppgd=16); LW/CI keep their higher bl for representativeness.
N_PPGD = 16  # bl_pp = 64 / 16 = 4
BATCH = 64  # bl_lw = 64 / 1 = 64 (forces LW activation checkpointing)
STEPS = 80  # skip the first ~30 (warmup/compile), average the steady-state tail


def build_profile_topology() -> dict[str, Any]:
    ci_ranks = list(range(N_LW, N_LW + N_CI))
    ppgd_ranks = list(range(N_LW + N_CI, N_LW + N_CI + N_PPGD))
    return {
        "use_fused_kl": True,
        "ci_ranks": ci_ranks,
        "ppgd_ranks": ppgd_ranks,
        "layerwise_block_groups": [
            {"ranks": [i], "owned_sites": SITES[2 * i : 2 * i + 2]} for i in range(N_LW)
        ],
    }


def main() -> None:
    cfg = copy.deepcopy(yaml.safe_load(BASE.read_text()))
    cfg["pd"]["batch_size"] = BATCH
    cfg["pd"]["steps"] = STEPS
    cfg["pd"]["faithfulness_warmup_steps"] = 2  # short; an init scheme, not steady state
    cfg["runtime"]["topology"] = build_profile_topology()
    cfg["target"]["activation_checkpointing"] = True  # LW-only (gated in optimize.py)
    cfg["cadence"]["save_every"] = None  # no checkpoint/consolidate during the profile
    cfg["cadence"]["train_log_every"] = 10  # always-on metrics; avoids profiler active window
    cfg["eval"] = None  # no eval pass perturbing steady-state timing
    cfg.pop("wandb", None)  # local sink only; metrics also print to the console log

    OUT.write_text(yaml.safe_dump(cfg, sort_keys=False))
    total = N_LW + N_CI + N_PPGD
    print(f"wrote {OUT}")
    print(
        f"topology: LW {N_LW} (1 rank/block, 2 sites/block -> 96 sites) + CI {N_CI} + "
        f"PPGD {N_PPGD} = {total} GPUs"
    )
    print(
        f"batch={BATCH}  bl_lw={BATCH // 1}  bl_ci={BATCH // N_CI}  "
        f"bl_pp={BATCH // N_PPGD}  steps={STEPS}"
    )


if __name__ == "__main__":
    main()
