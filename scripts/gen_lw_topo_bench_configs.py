"""Generate LW-block-topology benchmark configs from the 200k 3-pool base.

Varies ONLY layerwise_block_groups (the LW factoring). Holds n_ci=24, n_ppgd=24,
n_lw=96, batch_size=48 fixed. For each n_per_block in {24,8,4,2,1}, partition the 96
LW sites (h.{0..47}.attn.{q,k}_proj) contiguously into n_blocks=96/n_per_block blocks,
each owning 96/n_blocks sites, with contiguous rank ids [0..95]. CI ranks 96-119,
PPGD ranks 120-143 unchanged.

Each generated config also: steps=60, save_every=null (no mid-run checkpoint), eval
removed (no eval during the run), train_log_every=5 (so steps 5..60 are logged),
and drops wandb (pure local metrics.jsonl + console). Confirms each parses via the
ThreePoolLMExperimentConfig validator.
"""

from pathlib import Path

import yaml

from param_decomp_lab.experiments.lm.three_pool_run import ThreePoolLMExperimentConfig

BASE = Path(__file__).resolve().parents[1] / (
    "param_decomp_lab/experiments/lm/_xl_production/gpt2_xl_qk_200k.yaml"
)
OUT_DIR = Path(__file__).resolve().parents[1] / "param_decomp_lab/experiments/lm/_lw_topo_bench"

N_LAYERS = 48
SITES = [f"h.{l}.attn.{p}_proj" for l in range(N_LAYERS) for p in ("q", "k")]
assert len(SITES) == 96

N_PER_BLOCK_VARIANTS = [24, 8, 4, 2, 1]


def build_block_groups(n_per_block: int) -> list[dict]:
    assert 96 % n_per_block == 0
    n_blocks = 96 // n_per_block
    assert len(SITES) % n_blocks == 0
    sites_per_block = len(SITES) // n_blocks
    groups = []
    for b in range(n_blocks):
        ranks = list(range(b * n_per_block, (b + 1) * n_per_block))
        owned = SITES[b * sites_per_block : (b + 1) * sites_per_block]
        groups.append({"ranks": ranks, "owned_sites": owned})
    return groups


def main() -> None:
    base = yaml.safe_load(BASE.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for n_per_block in N_PER_BLOCK_VARIANTS:
        cfg = yaml.safe_load(BASE.read_text())  # fresh deep copy
        cfg["runtime"]["topology"]["layerwise_block_groups"] = build_block_groups(n_per_block)
        cfg["pd"]["steps"] = 60
        cfg["cadence"]["train_log_every"] = 5
        cfg["cadence"]["save_every"] = None
        cfg.pop("eval", None)
        cfg.pop("wandb", None)

        # Validate it parses through the real config tree (topology + cross-field checks).
        ThreePoolLMExperimentConfig.model_validate(cfg)

        bl_lw = cfg["pd"]["batch_size"] // n_per_block
        out = OUT_DIR / f"bench_npb{n_per_block:02d}.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
        n_blocks = 96 // n_per_block
        print(
            f"wrote {out.name}: n_per_block={n_per_block} n_blocks={n_blocks} "
            f"sites/block={96 // n_blocks} bl_lw={bl_lw} -> PARSES"
        )

    del base


if __name__ == "__main__":
    main()
