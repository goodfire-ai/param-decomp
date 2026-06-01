"""Generate the b256 overnight 3-pool configs from the finished 200k run config.

Base = param_decomp_lab/experiments/lm/_200k/cfg.yaml (verified identical to the finished
run p-ecfda851's run_meta.yaml). Changes:
  - batch_size 16 -> 256; steps 200000 -> 50000.
  - target.activation_checkpointing -> true (the whole point).
  - topology -> 96 LW blocks (1 site/block, 1 GPU/block, bl_lw=256) + CI 16 + PPGD 16 = 128 GPU.
  - synchronized LR warmup: warmup_pct = 0.01 (500/50000 steps) on comp, ci_fn AND ppgd.
  - 2-run LR sweep (keeping the 5:1 comp:ci ratio), sqrt-projected from the b48 investigation:
      A: comp 5e-3 / ci 1e-3   (conservative)
      B: comp 1e-2 / ci 2e-3   (sqrt-projection of the b48 ~4.5e-3 center to b256)
  - PPGD LR unchanged (0.01) — per-position adversary, not batch-averaged.
  - faithfulness_warmup_steps left at 400 (treated as an init scheme, not part of LR warmup).
Also emits a smoke variant (steps 30, save_every 20, faith warmup 2) at the SAME 128-GPU
topology to validate save -> async-consolidate -> resume at production scale before the run.
"""

import copy
from pathlib import Path

import yaml

BASE = Path("param_decomp_lab/experiments/lm/_200k/cfg.yaml")
OUT_DIR = Path("param_decomp_lab/experiments/lm/_b256_run")
SITES = [f"h.{layer}.attn.{p}_proj" for layer in range(48) for p in ("q", "k")]
assert len(SITES) == 96

N_LW = 96  # one block per site, one GPU per block
CI_RANKS = list(range(N_LW, N_LW + 16))
PPGD_RANKS = list(range(N_LW + 16, N_LW + 32))
WARMUP_PCT = 0.01  # 500 / 50000, synchronized across comp / ci_fn / ppgd


def build_topology() -> dict:
    return {
        "use_fused_kl": True,
        "ci_ranks": CI_RANKS,
        "ppgd_ranks": PPGD_RANKS,
        "layerwise_block_groups": [
            {"ranks": [i], "owned_sites": [SITES[i]]} for i in range(N_LW)
        ],
    }


def make_cfg(base: dict, comp_lr: float, ci_lr: float) -> dict:
    cfg = copy.deepcopy(base)
    pd = cfg["pd"]
    pd["batch_size"] = 256
    pd["steps"] = 50000
    pd["components_optimizer"]["lr_schedule"]["start_val"] = comp_lr
    pd["components_optimizer"]["lr_schedule"]["warmup_pct"] = WARMUP_PCT
    pd["ci_fn_optimizer"]["lr_schedule"]["start_val"] = ci_lr
    pd["ci_fn_optimizer"]["lr_schedule"]["warmup_pct"] = WARMUP_PCT
    pd["losses"]["ppgd"]["optimizer"]["lr_schedule"]["warmup_pct"] = WARMUP_PCT
    cfg["target"]["activation_checkpointing"] = True
    cfg["runtime"]["topology"] = build_topology()
    cfg["cadence"]["save_every"] = 2500
    cfg["cadence"]["train_log_every"] = 50
    # Eval slices the batch per-rank by the TRAINING batch_local (= batch_size / n_pool), so the
    # eval batch must be >= the training batch or the last ranks of each pool get empty slices
    # ("batch size must be positive"). Match it to the training batch.
    cfg["eval"]["batch_size"] = 256
    return cfg


def main() -> None:
    base = yaml.safe_load(BASE.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = {"b256_lrA_comp5e-3": (5e-3, 1e-3), "b256_lrB_comp1e-2": (1e-2, 2e-3)}
    for name, (comp_lr, ci_lr) in runs.items():
        cfg = make_cfg(base, comp_lr, ci_lr)
        out = OUT_DIR / f"{name}.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"{name}: comp={comp_lr} ci={ci_lr} -> {out}")

    smoke = make_cfg(base, 5e-3, 1e-3)
    smoke["pd"]["steps"] = 30
    smoke["pd"]["faithfulness_warmup_steps"] = 2
    smoke["cadence"]["save_every"] = 20
    smoke["eval"]["every"] = 10
    smoke["eval"]["slow_every"] = 10
    smoke.pop("wandb", None)
    out = OUT_DIR / "b256_smoke.yaml"
    out.write_text(yaml.safe_dump(smoke, sort_keys=False))
    print(f"smoke: steps=30 save_every=20 -> {out}")

    total = N_LW + len(CI_RANKS) + len(PPGD_RANKS)
    print(f"\ntopology: LW {N_LW} (1 site/block, 1 gpu/block) + CI {len(CI_RANKS)} + "
          f"PPGD {len(PPGD_RANKS)} = {total} GPUs; bl_lw=256, warmup_pct={WARMUP_PCT}")


if __name__ == "__main__":
    main()
