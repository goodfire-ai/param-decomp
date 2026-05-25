"""Multi-node equivalence sweep on the 5L canon-aligned config.

Question being tested: does the equivalence verdict (1-pool ≡ 2-pool ≡ 3-pool)
hold under multi-node distribution, or does NCCL across InfiniBand introduce
behavior we haven't seen in single-node testing?

Four cohorts × 5 seeds each:
  * 1pool DDP=2, single-node          — reference (and we already have N=10
                                       baseline at batch=24; this is N=5 at
                                       batch=64 to share the multi-node config).
  * 1pool DDP=16, 2-node              — isolated multi-node infra check.
  * 2pool 6×2 LW + 4 PPGD = 16, 2-node — multi-node 2-pool.
  * 3pool 5×2 LW + 2 CI + 4 PPGD = 16, 2-node — multi-node 3-pool.

All use the same canon-aligned hyperparameters (warmup=400, full coeffs,
grad_clip=0.01) from ``equiv_5L_*.yaml``. ``batch_size`` is bumped to 64 so
it divides all the parallelism factors (16, 2, 4).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from param_decomp_lab.infra.run_files import ExecutionStamp
from param_decomp_lab.infra.settings import REPO_ROOT
from param_decomp_lab.infra.slurm import (
    CUDA_FLAGS,
    GPUS_PER_NODE,
    SlurmConfig,
    generate_script,
    submit_slurm_job,
    torchrun_command,
)

BATCH_SIZE = 64
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class Cohort:
    name: str
    base_yaml: Path
    n_gpus: int
    overrides: dict[str, Any]  # nested overrides merged into the base yaml


def _two_pool_block_groups_6x2() -> list[dict[str, Any]]:
    """6 LW blocks × 2 ranks each, 5 sites per block (30 sites total)."""
    sites_per_layer = (
        "attn.q_proj",
        "attn.k_proj",
        "attn.v_proj",
        "attn.o_proj",
        "mlp.c_fc",
        "mlp.down_proj",
    )
    all_sites = [f"h.{layer}.{site}" for layer in range(5) for site in sites_per_layer]
    blocks = []
    for blk_idx in range(6):
        start = blk_idx * 5
        ranks = [blk_idx * 2, blk_idx * 2 + 1]
        blocks.append({"ranks": ranks, "owned_sites": all_sites[start : start + 5]})
    return blocks


def _three_pool_block_groups_5x2() -> list[dict[str, Any]]:
    """5 LW blocks × 2 ranks each, 6 sites per block (= one layer per block)."""
    sites_per_layer = (
        "attn.q_proj",
        "attn.k_proj",
        "attn.v_proj",
        "attn.o_proj",
        "mlp.c_fc",
        "mlp.down_proj",
    )
    blocks = []
    for layer in range(5):
        ranks = [layer * 2, layer * 2 + 1]
        owned_sites = [f"h.{layer}.{site}" for site in sites_per_layer]
        blocks.append({"ranks": ranks, "owned_sites": owned_sites})
    return blocks


COHORTS: tuple[Cohort, ...] = (
    Cohort(
        name="1pool-snode",
        base_yaml=REPO_ROOT / "param_decomp_lab/experiments/lm/equiv_5L_1pool.yaml",
        n_gpus=2,
        overrides={"pd": {"batch_size": BATCH_SIZE}},
    ),
    Cohort(
        name="1pool-mnode",
        base_yaml=REPO_ROOT / "param_decomp_lab/experiments/lm/equiv_5L_1pool.yaml",
        n_gpus=16,
        overrides={"pd": {"batch_size": BATCH_SIZE}},
    ),
    Cohort(
        name="2pool-mnode",
        base_yaml=REPO_ROOT / "param_decomp_lab/experiments/lm/equiv_5L_2pool.yaml",
        n_gpus=16,
        overrides={
            "pd": {"batch_size": BATCH_SIZE},
            "two_pool": {
                "pool_b_ranks": [12, 13, 14, 15],
                "block_groups": _two_pool_block_groups_6x2(),
            },
        },
    ),
    Cohort(
        name="3pool-mnode",
        base_yaml=REPO_ROOT / "param_decomp_lab/experiments/lm/equiv_5L_3pool.yaml",
        n_gpus=16,
        overrides={
            "pd": {"batch_size": BATCH_SIZE},
            "three_pool": {
                "ci_ranks": [10, 11],
                "ppgd_ranks": [12, 13, 14, 15],
                "layerwise_block_groups": _three_pool_block_groups_5x2(),
            },
        },
    ),
)

OUT_DIR = REPO_ROOT / "param_decomp_lab/experiments/lm/_mnode_equiv"


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _materialize(cohort: Cohort, seed: int) -> Path:
    with open(cohort.base_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg["pd"]["seed"] = seed
    _deep_merge(cfg, cohort.overrides)
    out = OUT_DIR / f"{cohort.name}_seed{seed}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def main() -> None:
    if OUT_DIR.is_dir():
        for p in OUT_DIR.iterdir():
            if p.is_file():
                p.unlink()

    stamp = ExecutionStamp.create(run_type="param_decomp", create_snapshot=True)
    print(f"Snapshot: {stamp.snapshot_ref}")

    for cohort in COHORTS:
        assert cohort.base_yaml.is_file(), cohort.base_yaml
        n_nodes_required = (cohort.n_gpus + GPUS_PER_NODE - 1) // GPUS_PER_NODE
        if cohort.n_gpus > GPUS_PER_NODE:
            assert cohort.n_gpus % GPUS_PER_NODE == 0, (
                f"{cohort.name}: n_gpus={cohort.n_gpus} not multiple of {GPUS_PER_NODE}"
            )

        for seed in SEEDS:
            yaml_path = _materialize(cohort, seed)
            job_name = f"mn-{cohort.name}-s{seed}"
            cmd = torchrun_command(
                job_name=job_name,
                snapshot_ref=stamp.snapshot_ref,
                python_module="param_decomp_lab.experiments.lm.run",
                script_args=str(yaml_path),
                n_gpus=cohort.n_gpus,
            )
            cfg = SlurmConfig(
                job_name=job_name,
                partition=None,
                n_gpus=GPUS_PER_NODE if n_nodes_required > 1 else cohort.n_gpus,
                n_nodes=n_nodes_required,
                time="01:00:00",
                snapshot_ref=stamp.snapshot_ref,
                comment=f"multi-node equiv — {job_name}",
                qos="scavenge",
            )
            r = submit_slurm_job(generate_script(cfg, cmd, env=CUDA_FLAGS), job_name)
            print(f"{job_name}: nodes={n_nodes_required} gpus={cohort.n_gpus} job_id={r.job_id}")


if __name__ == "__main__":
    main()
