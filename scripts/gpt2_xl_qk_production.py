"""Launch the GPT-2 XL Q/K-only 3-pool production run (or smoke test).

Default produces a config at:
  * d_model=4096, n_blocks=8, mlp_hidden=16384 CI fn (~2.64B params, ~10× target)
  * Q/K only, 96 sites total (48 layers × 2 sites)
  * Topology: 24 LW blocks × 4 ranks + 4 CI + 4 PPGD = 104 GPUs = 13 nodes
  * batch_size=128
  * 400-step faithfulness warmup
  * Cosine LR schedule (5e-4 components, 1e-4 ci_fn) to 0.1× final
  * grad_clip_norm=0.01 on components
  * activation_checkpointing=False (see ``equiv_5L_experiments.md`` for the
    reasoning: hook-based component injection isn't checkpoint-compatible,
    and 3-pool's layerwise streaming pattern doesn't need it anyway).

Usage:
  python scripts/gpt2_xl_qk_production.py            # 200k-step production run
  python scripts/gpt2_xl_qk_production.py --smoke    # 50-step smoke test, no save
"""

from typing import Any

import fire
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

N_LAYERS = 48
SITES_PER_LAYER = ("attn.q_proj", "attn.k_proj")
ALL_SITES = [f"h.{layer}.{site}" for layer in range(N_LAYERS) for site in SITES_PER_LAYER]
N_SITES = len(ALL_SITES)  # 96
N_LW_RANKS = N_SITES  # by construction (sites_per_block == ddp_per_block keeps this fixed)

DEFAULT_DDP_PER_BLOCK = 4
DEFAULT_N_CI = 4
DEFAULT_N_PPGD = 4


def _block_groups(ddp_per_block: int) -> list[dict[str, Any]]:
    """LW block layout: each block owns ``sites_per_block`` consecutive sites
    and has ``ddp_per_block`` DDP replicas. Total LW ranks =
    ``N_SITES / sites_per_block * ddp_per_block = N_SITES`` (always 96).
    Caller picks ``ddp_per_block``; ``sites_per_block`` is derived to keep
    LW size constant.
    """
    assert N_SITES % ddp_per_block == 0, (
        f"N_SITES={N_SITES} must be divisible by ddp_per_block={ddp_per_block}"
    )
    sites_per_block = ddp_per_block  # keeps total LW ranks = N_SITES
    n_blocks = N_SITES // sites_per_block
    return [
        {
            "ranks": [b * ddp_per_block + r for r in range(ddp_per_block)],
            "owned_sites": ALL_SITES[b * sites_per_block : (b + 1) * sites_per_block],
        }
        for b in range(n_blocks)
    ]


def _make_yaml_dict(
    *,
    steps: int,
    save_every: int | None,
    warmup_steps: int,
    n_ci: int,
    n_ppgd: int,
    ddp_per_block: int,
) -> dict[str, Any]:
    base_ci = N_LW_RANKS
    return {
        "pd": {
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
            "batch_size": 128,
            "steps": steps,
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
            "faithfulness_warmup_steps": warmup_steps,
            "faithfulness_warmup_lr": 0.001,
            "faithfulness_warmup_weight_decay": 0.0,
            "loss_metrics": [
                {"type": "FaithfulnessLoss", "coeff": 1.0e08},
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
        },
        "cadence": {
            "train_log_every": 100 if steps > 200 else 10,
            "save_every": save_every,
        },
        "eval": None,
        "runtime": {"autocast_bf16": True, "device": "cuda", "dp": None},
        "target": {
            "spec": {
                "kind": "hf_weights_in_vendored",
                "model_class": "param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple.GPT2Simple",
                "model_name": "openai-community/gpt2-xl",
            },
            "output_extract": 0,
            # See equiv_5L_experiments.md: PD's forward-hook-based component
            # injection is incompatible with torch.utils.checkpoint
            # (recomputation runs without hooks → "different number of tensors
            # saved" error). 3-pool's layerwise streaming runs forward+backward
            # per site, so per-rank peak activation memory is bounded by ONE
            # forward — checkpointing isn't needed at our batch sizes.
            "activation_checkpointing": False,
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
            "ci_ranks": list(range(base_ci, base_ci + n_ci)),
            "ppgd_ranks": list(range(base_ci + n_ci, base_ci + n_ci + n_ppgd)),
            "layerwise_block_groups": _block_groups(ddp_per_block),
        },
    }


def main(
    smoke: bool = False,
    batch_size: int = 16,
    profile: bool = True,
    torch_profile: bool = False,
    torch_profile_ranks: str | None = None,
    ci_bwd_profile: bool = False,
    n_ci: int = DEFAULT_N_CI,
    n_ppgd: int = DEFAULT_N_PPGD,
    ddp_per_block: int = DEFAULT_DDP_PER_BLOCK,
    compile_ci_fn: bool = False,
) -> None:
    """Submit the production XL Q/K run, or a smoke test.

    Args:
        smoke: If True, submits a 50-step smoke (no save, 1h limit) instead of
            the full 200k-step production run.
        batch_size: Smoke batch size override. The yaml's default (128) is the
            production target; the smoke shrinks it to fit while we figure out
            the memory profile. Ignored when ``smoke=False``.
        profile: When ``smoke=True``, enable CUDA memory-history recording on
            one rank per pool (LW block 0, CI rank 0, PPGD rank 0). Dumps
            ``mem_rank<R>.pickle`` files loadable at pytorch.org/memory_viz.
        torch_profile: When ``smoke=True``, enable torch.profiler. Dumps
            Chrome trace JSON + key-averages text summary per rank to
            ``_xl_production/torch_profile/<job>/``. Heavy: ~10-50% slowdown
            on profiled ranks. Drop the json into speedscope.app.
        torch_profile_ranks: Comma-separated rank list overriding the default
            "one per pool" (e.g. "96" for CI-only). Reducing scope mitigates
            multi-rank NCCL desync risk.
        n_ci: Number of CI pool ranks. Default 4. Bump to halve per-CI-rank batch.
        n_ppgd: Number of PPGD pool ranks. Default 4. Bump to halve per-PPGD-rank
            batch (which halves the PPGD-bound batch ceiling because PPGD's
            ``D3_warmup`` / ``D4_recon`` transient peak is per-rank).
    """
    total_ranks = N_LW_RANKS + n_ci + n_ppgd
    out_dir = REPO_ROOT / "param_decomp_lab/experiments/lm/_xl_production"
    out_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        cfg_dict = _make_yaml_dict(
            steps=50,
            save_every=None,
            warmup_steps=0,
            n_ci=n_ci,
            n_ppgd=n_ppgd,
            ddp_per_block=ddp_per_block,
        )
        cfg_dict["pd"]["batch_size"] = batch_size
        yaml_path = out_dir / "gpt2_xl_qk_smoke.yaml"
        job_name = "xl-qk-smoke"
        time_limit = "01:00:00"
    else:
        cfg_dict = _make_yaml_dict(
            steps=200000,
            save_every=5000,
            warmup_steps=400,
            n_ci=n_ci,
            n_ppgd=n_ppgd,
            ddp_per_block=ddp_per_block,
        )
        yaml_path = out_dir / "gpt2_xl_qk_production.yaml"
        job_name = "xl-qk-prod"
        # 7-day time limit (cluster max); production runs for as long as it fits.
        time_limit = "7-00:00:00"

    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)
    print(f"wrote {yaml_path}")

    assert total_ranks % GPUS_PER_NODE == 0, (
        f"total_ranks={total_ranks} not a multiple of {GPUS_PER_NODE}"
    )
    n_nodes = total_ranks // GPUS_PER_NODE

    stamp = ExecutionStamp.create(run_type="param_decomp", create_snapshot=True)
    print(f"Snapshot: {stamp.snapshot_ref}")

    env = dict(CUDA_FLAGS)
    # One rank per pool: LW block-0 rank-0 = 0, CI rank-0 = N_LW_RANKS,
    # PPGD rank-0 = N_LW_RANKS + n_ci.
    prof_ranks = [0, N_LW_RANKS, N_LW_RANKS + n_ci]
    if smoke and profile:
        prof_dir = out_dir / "mem_profile" / job_name
        env["PD_MEMORY_PROFILE_RANKS"] = ",".join(str(r) for r in prof_ranks)
        env["PD_MEMORY_PROFILE_OUT"] = str(prof_dir)
        # Restrict the very-chatty per-phase trace to one rank per pool;
        # the macro-boundary trace() calls in run/optimize stay rank-prefixed
        # but unrestricted so we see if any rank diverges.
        env["PD_TRACE_RANKS"] = ",".join(str(r) for r in prof_ranks)
        env["PD_PHASE_TRACE"] = "1"
        print(f"mem-profile: ranks={prof_ranks} → {prof_dir}")
        print(f"trace: ranks={prof_ranks}, phase_trace=on")
    if smoke and torch_profile:
        tp_dir = out_dir / "torch_profile" / job_name
        tp_ranks = torch_profile_ranks if torch_profile_ranks else ",".join(str(r) for r in prof_ranks)
        env["PD_TORCH_PROFILE_RANKS"] = tp_ranks
        env["PD_TORCH_PROFILE_OUT"] = str(tp_dir)
        print(f"torch-profile: ranks={tp_ranks} → {tp_dir} (chrome trace + key_avgs)")
        print("  view: drop the json into https://www.speedscope.app/")
    if compile_ci_fn:
        env["PD_COMPILE_CI_FN"] = "1"
        print("torch.compile on CI fn: ON")
    if ci_bwd_profile:
        env["PD_CI_FN_BWD_PROFILE"] = "1"
        print("CI fn bwd-stage profile: ON (per-block bwd ms via CUDA events)")
    print(
        f"topology: lw={N_LW_RANKS} ({N_SITES // ddp_per_block} blocks × DDP={ddp_per_block}, "
        f"{ddp_per_block} sites/block), ci={n_ci}, ppgd={n_ppgd}, total={total_ranks}"
    )

    cmd = torchrun_command(
        job_name=job_name,
        snapshot_ref=stamp.snapshot_ref,
        python_module="param_decomp_lab.experiments.lm.run",
        # Relative path — resolved against the snapshot checkout, not live FS.
        # Concurrent launches each get their own snapshotted yaml this way.
        script_args=str(yaml_path.relative_to(REPO_ROOT)),
        n_gpus=total_ranks,
    )
    slurm_cfg = SlurmConfig(
        job_name=job_name,
        partition=None,
        n_gpus=GPUS_PER_NODE,
        n_nodes=n_nodes,
        time=time_limit,
        snapshot_ref=stamp.snapshot_ref,
        comment=("GPT-2 XL Q/K 3-pool smoke test" if smoke else "GPT-2 XL Q/K 3-pool production"),
    )
    r = submit_slurm_job(generate_script(slurm_cfg, cmd, env=env), job_name)
    print(f"{job_name}: nodes={n_nodes} gpus={total_ranks} job_id={r.job_id}")


if __name__ == "__main__":
    fire.Fire(main)
