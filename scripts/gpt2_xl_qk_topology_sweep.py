"""Topology sweep: find the fastest wall-clock-per-iter config for GPT-2 XL
Q/K-only 3-pool training, aiming for batch_size 128 at ~1s/iter.

For each topology variant we materialize a yaml and submit a short SLURM
job (warmup=0, steps=50, save_every=null) so we can measure ``perf/step_ms``
without paying the warmup cost. The runs all share the same git snapshot.

The base config (everything except topology + batch_size + steps + warmup)
mirrors gpt2_xl_full.yaml — see ``equiv_5L_experiments.md`` for the audit.

Topology choices honor the layout constraints:
  * n_ci must divide n_per_block (LW DDP arity).
  * n_ci must divide n_ppgd.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from param_decomp_lab.infra.run_files import ExecutionStamp
from param_decomp_lab.infra.settings import REPO_ROOT
from param_decomp_lab.infra.slurm import (
    SlurmConfig,
    generate_script,
    multi_node_torchrun_command,
    submit_slurm_job,
)


GPUS_PER_NODE = 8


N_LAYERS = 48
SITES_PER_LAYER = ("attn.q_proj", "attn.k_proj")
ALL_SITES = [f"h.{layer}.{site}" for layer in range(N_LAYERS) for site in SITES_PER_LAYER]


@dataclass(frozen=True)
class Topology:
    name: str
    sites_per_block: int
    ddp_per_block: int  # n_per_block_lw
    n_ci: int
    n_ppgd: int

    @property
    def n_blocks_lw(self) -> int:
        assert len(ALL_SITES) % self.sites_per_block == 0
        return len(ALL_SITES) // self.sites_per_block

    @property
    def n_lw_ranks(self) -> int:
        return self.n_blocks_lw * self.ddp_per_block

    @property
    def total_ranks(self) -> int:
        return self.n_lw_ranks + self.n_ci + self.n_ppgd

    def block_groups(self) -> list[dict[str, Any]]:
        groups = []
        for blk_idx in range(self.n_blocks_lw):
            start = blk_idx * self.sites_per_block
            end = start + self.sites_per_block
            ranks = [blk_idx * self.ddp_per_block + r for r in range(self.ddp_per_block)]
            groups.append({"ranks": ranks, "owned_sites": ALL_SITES[start:end]})
        return groups

    def ci_ranks(self) -> list[int]:
        base = self.n_lw_ranks
        return list(range(base, base + self.n_ci))

    def ppgd_ranks(self) -> list[int]:
        base = self.n_lw_ranks + self.n_ci
        return list(range(base, base + self.n_ppgd))

    def __post_init__(self) -> None:
        # Layout constraints (assertions match those in three_pool.layout).
        assert self.ddp_per_block % self.n_ci == 0, (
            f"n_per_block_lw ({self.ddp_per_block}) must be divisible by n_ci ({self.n_ci})"
        )
        assert self.n_ppgd % self.n_ci == 0, (
            f"n_ppgd ({self.n_ppgd}) must be divisible by n_ci ({self.n_ci})"
        )


TOPOLOGIES: tuple[Topology, ...] = (
    # All three vary `sites_per_block` while holding DDP=4, CI=4, PPGD=4 fixed,
    # so we can isolate "how does block partitioning affect throughput?". Each
    # total is a multiple of 8 (one full B200 node = 8 GPUs).
    #
    # User's preferred: 4 sites/block × DDP=4 → 24 blocks × 4 ranks = 96 LW.
    Topology(name="t104-s4-d4-ci4-pp4", sites_per_block=4, ddp_per_block=4, n_ci=4, n_ppgd=4),
    # Half the LW blocks, twice the sites per block.
    Topology(name="t56-s8-d4-ci4-pp4", sites_per_block=8, ddp_per_block=4, n_ci=4, n_ppgd=4),
    # Quarter the LW blocks: chunky blocks of 16 sites each.
    Topology(name="t32-s16-d4-ci4-pp4", sites_per_block=16, ddp_per_block=4, n_ci=4, n_ppgd=4),
)

BATCH_SIZE = 128
N_BENCHMARK_STEPS = 50


def _base_pd_config() -> dict[str, Any]:
    """Canon-aligned pd block (mirrors gpt2_xl_full.yaml; see equiv-5L audit)."""
    return {
        "seed": 0,
        "n_mask_samples": 1,
        "ci_config": {
            "mode": "global",
            "fn_type": "global_shared_transformer",
            "simple_transformer_ci_cfg": {
                "d_model": 4096,
                "n_blocks": 8,
                "mlp_hidden_dim": [16384],
                "attn_config": {"n_heads": 32, "max_len": 1024, "rope_base": 10000.0},
            },
        },
        "sampling": "continuous",
        "sigmoid_type": "leaky_hard",
        "decomposition_targets": [
            {"module_pattern": "h.*.attn.q_proj", "C": 1024},
            {"module_pattern": "h.*.attn.k_proj", "C": 1024},
        ],
        "identity_decomposition_targets": None,
        "use_delta_component": True,
        "batch_size": BATCH_SIZE,
        "steps": N_BENCHMARK_STEPS,
        "components_optimizer": {
            "lr_schedule": {
                "start_val": 5.0e-04,
                "warmup_pct": 0.0,
                "final_val_frac": 0.1,
                "fn_type": "cosine",
            },
            "grad_clip_norm": 0.01,
        },
        "ci_fn_optimizer": {
            "lr_schedule": {
                "start_val": 1.0e-04,
                "warmup_pct": 0.0,
                "final_val_frac": 0.1,
                "fn_type": "cosine",
            },
        },
        # Skip warmup for the benchmark — we only care about steady-state iter speed.
        "faithfulness_warmup_steps": 0,
        "faithfulness_warmup_lr": 0.001,
        "faithfulness_warmup_weight_decay": 0.0,
        "loss_metrics": [
            {"type": "FaithfulnessLoss", "coeff": 1.0e+08},
            {"type": "StochasticReconLayerwiseLoss", "coeff": 50.0},
            {
                "type": "ImportanceMinimalityLoss",
                "coeff": 4.0e-05,
                "pnorm": 2.0,
                "beta": 0.5,
                "p_anneal_start_frac": 0.0,
                "p_anneal_final_p": 0.3,
                "p_anneal_end_frac": 1.0,
                "eps": 1.0e-12,
            },
            {
                "type": "PersistentPGDReconLoss",
                "coeff": 1.0,
                "optimizer": {
                    "type": "adam",
                    "beta1": 0.5,
                    "beta2": 0.99,
                    "eps": 1.0e-08,
                    "lr_schedule": {
                        "start_val": 0.01,
                        "warmup_pct": 0.025,
                        "final_val_frac": 1.0,
                        "fn_type": "constant",
                    },
                },
                "scope": {"type": "per_batch_per_position"},
                "use_sigmoid_parameterization": False,
                "n_warmup_steps": 2,
                "n_samples": 1,
            },
        ],
    }


def _materialize_yaml(topo: Topology, out_dir: Path) -> Path:
    cfg = {
        "pd": _base_pd_config(),
        "cadence": {"train_log_every": 10, "save_every": None},
        "eval": None,
        "runtime": {"autocast_bf16": True, "device": "cuda", "dp": None},
        "target": {
            "spec": {
                "kind": "hf_weights_in_vendored",
                "model_class": "param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple.GPT2Simple",
                "model_name": "openai-community/gpt2-xl",
            },
            "output_extract": 0,
            "activation_checkpointing": True,
        },
        "data": {
            "tokenizer_name": "openai-community/gpt2",
            "max_seq_len": 1024,
            "dataset_name": "apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
            "column_name": "input_ids",
            "train_split": "train",
            "eval_split": "train",
            "is_tokenized": True,
            "streaming": True,
            "buffer_size": 1000,
            "shuffle_each_epoch": True,
        },
        "three_pool": {
            "use_fused_kl": True,
            "defer_vu_opt": True,
            "ci_ranks": topo.ci_ranks(),
            "ppgd_ranks": topo.ppgd_ranks(),
            "layerwise_block_groups": topo.block_groups(),
        },
    }
    out = out_dir / f"gpt2_xl_qk_{topo.name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, width=200)
    return out


def main() -> None:
    out_dir = REPO_ROOT / "param_decomp_lab/experiments/lm/_xl_sweep"
    # Wipe stale yamls so the snapshot doesn't carry old configs.
    if out_dir.is_dir():
        for p in out_dir.iterdir():
            if p.is_file():
                p.unlink()

    # Print topology summary up front.
    print(f"{'name':<28} {'sites/blk':>10} {'ddp/blk':>8} {'n_lw':>5} {'n_ci':>5} {'n_pp':>5} {'total':>6}")
    print("-" * 80)
    for t in TOPOLOGIES:
        print(f"{t.name:<28} {t.sites_per_block:>10} {t.ddp_per_block:>8} {t.n_lw_ranks:>5} {t.n_ci:>5} {t.n_ppgd:>5} {t.total_ranks:>6}")
    print()

    paths = [(t, _materialize_yaml(t, out_dir)) for t in TOPOLOGIES]

    stamp = ExecutionStamp.create(run_type="param_decomp", create_snapshot=True)
    print(f"Snapshot: {stamp.snapshot_ref}")

    for topo, yaml_path in paths:
        assert topo.total_ranks % GPUS_PER_NODE == 0, (
            f"topology {topo.name} has {topo.total_ranks} ranks, not a multiple of "
            f"{GPUS_PER_NODE} GPUs/node — torchrun expects uniform nproc-per-node"
        )
        n_nodes = topo.total_ranks // GPUS_PER_NODE
        job_name = f"xl-qk-{topo.name}"
        cmd = multi_node_torchrun_command(
            job_name=job_name,
            snapshot_ref=stamp.snapshot_ref,
            yaml_path=str(yaml_path),
            nproc_per_node=GPUS_PER_NODE,
        )
        cfg = SlurmConfig(
            job_name=job_name,
            partition=None,
            n_gpus=GPUS_PER_NODE,
            n_nodes=n_nodes,
            time="01:00:00",
            snapshot_ref=stamp.snapshot_ref,
            comment=f"xl qk topology sweep — {topo.name}",
        )
        r = submit_slurm_job(generate_script(cfg, cmd), job_name)
        print(f"{topo.name}: nodes={n_nodes} job_id={r.job_id}")


if __name__ == "__main__":
    main()
