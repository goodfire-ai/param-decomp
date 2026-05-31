"""Generate 3-pool topology-sweep smoke configs from a base cfg.

Each candidate varies only runtime.topology (ci/ppgd/LW rank allocation). Smoke
tweaks: steps high (cancel early — avoids the final-step save+consolidate),
eval/save effectively disabled, frequent train-log, no wandb (local metrics.jsonl).

Rank layout convention (matches the base cfg): LW ranks 0..n_lw-1 (block i owns
the npb consecutive ranks [i*npb, (i+1)*npb)), then CI, then PPGD.
Sites ordered h.{l}.attn.{q,k} for l in 0..47, chunked into n_blocks groups.
"""

import copy
from pathlib import Path

import yaml

BASE = Path("param_decomp_lab/experiments/lm/_200k/cfg.yaml")
OUT_DIR = Path("param_decomp_lab/experiments/lm/_200k/sweep")
N_LAYERS = 48
SITES = [f"h.{l}.attn.{p}_proj" for l in range(N_LAYERS) for p in ("q", "k")]
assert len(SITES) == 96

# (name, n_ci, n_ppgd, n_blocks, n_per_block)
CANDIDATES = [
    ("s0-ci4-pp4-b24x4", 4, 4, 24, 4),    # baseline control (tot 104)
    ("s1-ci8-pp16-b24x4", 8, 16, 24, 4),  # fast pools, LW@96 (tot 120)
    ("s2-ci8-pp16-b24x8", 8, 16, 24, 8),  # n_lw192, 4 sites/blk, bl_lw2 (tot 216)
    ("s3-ci8-pp16-b48x4", 8, 16, 48, 4),  # n_lw192, 2 sites/blk, bl_lw4 (tot 216)
    ("s4-ci8-pp16-b96x2", 8, 16, 96, 2),  # n_lw192, 1 site/blk,  bl_lw8 (tot 216)
]


def build_topology(n_ci: int, n_ppgd: int, n_blocks: int, n_per_block: int) -> dict:
    assert len(SITES) % n_blocks == 0
    sites_per_block = len(SITES) // n_blocks
    blocks = []
    for i in range(n_blocks):
        ranks = list(range(i * n_per_block, (i + 1) * n_per_block))
        owned = SITES[i * sites_per_block : (i + 1) * sites_per_block]
        blocks.append({"ranks": ranks, "owned_sites": owned})
    n_lw = n_blocks * n_per_block
    return {
        "use_fused_kl": True,
        "ci_ranks": list(range(n_lw, n_lw + n_ci)),
        "ppgd_ranks": list(range(n_lw + n_ci, n_lw + n_ci + n_ppgd)),
        "layerwise_block_groups": blocks,
    }


def main() -> None:
    base = yaml.safe_load(BASE.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, n_ci, n_ppgd, n_blocks, n_per_block in CANDIDATES:
        cfg = copy.deepcopy(base)
        cfg["runtime"]["topology"] = build_topology(n_ci, n_ppgd, n_blocks, n_per_block)
        cfg["pd"]["steps"] = 100000  # cancel early; never reaches final save
        cfg["cadence"]["train_log_every"] = 5
        cfg["cadence"]["save_every"] = 100_000_000
        cfg["eval"]["every"] = 100_000_000
        cfg["eval"]["slow_every"] = 100_000_000
        cfg["eval"]["slow_on_first_step"] = False
        cfg.pop("wandb", None)
        total = n_blocks * n_per_block + n_ci + n_ppgd
        out = OUT_DIR / f"{name}.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"{name}: tot={total} (nodes={total // 8})  ci={n_ci} ppgd={n_ppgd} "
              f"lw={n_blocks}x{n_per_block}={n_blocks * n_per_block}  -> {out}")


if __name__ == "__main__":
    main()
