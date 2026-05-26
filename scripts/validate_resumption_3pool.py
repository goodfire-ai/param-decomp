"""End-to-end multi-node 3-pool resumption smoke.

Validates that a 3-pool training run can:
  1. Save resumable checkpoints mid-training.
  2. Be restarted from one of those checkpoints in a fresh slurm job.
  3. Produce reasonable (near-equivalent) losses on the resumed segment.

How:
  * **Baseline run**: 100 steps, ``save_every=50``. Saves at steps 50, 100.
    Records ``train/loss/total`` at the 10-step cadence.
  * **Resume run**: writes a ``ResumeConfig`` pointing at the baseline's
    ``resume/step_50/`` snapshot; submits ``lm/run.py --resume <yaml>``;
    trains from step 50 to step 100.
  * **Compare**: pull final-step loss from both runs' wandb logs (or
    stdout). They should be very close (streaming dataset + per-rank RNG
    re-seeding means not bit-exact at distributed scale, but qualitatively
    matched).

Topology: same as ``equiv_5L_multinode.py`` 3pool cohort — 5L GPT-2, batch=64,
5 LW blocks × 2 ranks + 2 CI + 4 PPGD = 16 GPUs / 2 nodes. Each leg ~2 min.

Usage:
    python scripts/validate_resumption_3pool.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from param_decomp_lab.infra.run_files import ExecutionStamp
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR, REPO_ROOT
from param_decomp_lab.infra.slurm import (
    CUDA_FLAGS,
    GPUS_PER_NODE,
    SlurmConfig,
    generate_script,
    submit_slurm_job,
    torchrun_command,
)

OUT_DIR = REPO_ROOT / "param_decomp_lab/experiments/lm/_resumption_validation"
BASELINE_STEPS = 100
SAVE_EVERY = 50
RESUME_FROM_STEP = 50


def _topology_3pool_5L_2node() -> tuple[int, dict[str, Any]]:
    """Topology matching the 5L mn-equiv 3-pool cohort.

    Returns ``(n_gpus, three_pool_dict)`` where three_pool_dict is suitable
    for embedding under ``three_pool:`` in the experiment YAML.
    """
    sites_per_layer = (
        "attn.q_proj",
        "attn.k_proj",
        "attn.v_proj",
        "attn.o_proj",
        "mlp.c_fc",
        "mlp.down_proj",
    )
    n_layers = 5
    block_layout: list[dict[str, Any]] = []
    for layer in range(n_layers):
        ranks = [layer * 2, layer * 2 + 1]
        owned = [f"h.{layer}.{site}" for site in sites_per_layer]
        block_layout.append({"ranks": ranks, "owned_sites": owned})
    n_lw = 10
    n_ci = 2
    n_ppgd = 4
    return n_lw + n_ci + n_ppgd, {
        "use_fused_kl": True,
        "defer_vu_opt": False,
        "ci_ranks": list(range(n_lw, n_lw + n_ci)),
        "ppgd_ranks": list(range(n_lw + n_ci, n_lw + n_ci + n_ppgd)),
        "layerwise_block_groups": block_layout,
    }


def _baseline_yaml_dict(steps: int, save_every: int, three_pool: dict[str, Any]) -> dict[str, Any]:
    """5L GPT-2 small + canonical 3-pool config from the mn-equiv cohort."""
    return {
        "pd": {
            "seed": 0,
            "n_mask_samples": 1,
            "ci_config": {
                "mode": "layerwise",
                "fn_type": "transformer",
                "transformer_cfg": {
                    "d_model": 128,
                    "n_blocks": 2,
                    "mlp_hidden_dim": [512],
                    "attn_config": {"n_heads": 4, "max_len": 1024, "rope_base": 10000.0},
                },
            },
            "sampling": "continuous",
            "sigmoid_type": "leaky_hard",
            "decomposition_targets": [
                {"module_pattern": "h.[01234].attn.q_proj", "C": 64},
                {"module_pattern": "h.[01234].attn.k_proj", "C": 64},
                {"module_pattern": "h.[01234].attn.v_proj", "C": 64},
                {"module_pattern": "h.[01234].attn.o_proj", "C": 64},
                {"module_pattern": "h.[01234].mlp.c_fc", "C": 64},
                {"module_pattern": "h.[01234].mlp.down_proj", "C": 64},
            ],
            "identity_decomposition_targets": None,
            "use_delta_component": True,
            "batch_size": 64,
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
                }
            },
            # No warmup — keeps the test fast. Real production uses 400.
            "faithfulness_warmup_steps": 0,
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
        "cadence": {"train_log_every": 10, "save_every": save_every},
        "eval": None,
        "runtime": {"autocast_bf16": True, "device": "cuda", "dp": None},
        "target": {
            "spec": {
                "kind": "hf_weights_in_vendored",
                "model_class": "param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple.GPT2Simple",
                "model_name": "openai-community/gpt2",
            },
            "output_extract": 0,
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
        "three_pool": three_pool,
    }


def _resume_yaml_dict(parent_run_dir: Path, step: int) -> dict[str, Any]:
    """Minimal ``ResumeConfig`` YAML — points at parent's checkpoint dir."""
    return {
        "from_run": str(parent_run_dir),
        "step": step,
        "overrides": None,
    }


def _wait_for_job(job_id: str, poll_s: int = 30) -> None:
    """Block until ``job_id`` is no longer queued. Just polls squeue."""
    import subprocess

    while True:
        r = subprocess.run(
            ["squeue", "-j", job_id, "-h", "--format=%T"],
            capture_output=True,
            text=True,
            check=False,
        )
        if not r.stdout.strip():
            return
        time.sleep(poll_s)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_ranks, three_pool = _topology_3pool_5L_2node()
    n_nodes = (total_ranks + GPUS_PER_NODE - 1) // GPUS_PER_NODE

    # === Baseline ===
    baseline_yaml_path = OUT_DIR / "baseline.yaml"
    baseline_cfg = _baseline_yaml_dict(
        steps=BASELINE_STEPS, save_every=SAVE_EVERY, three_pool=three_pool
    )
    baseline_yaml_path.write_text(yaml.safe_dump(baseline_cfg, sort_keys=False))
    print(f"baseline yaml → {baseline_yaml_path}")

    stamp = ExecutionStamp.create(run_type="param_decomp", create_snapshot=True)
    print(f"snapshot: {stamp.snapshot_ref}")

    baseline_job_name = "resumeval-baseline"
    cmd = torchrun_command(
        job_name=baseline_job_name,
        snapshot_ref=stamp.snapshot_ref,
        python_module="param_decomp_lab.experiments.lm.run",
        script_args=str(baseline_yaml_path.relative_to(REPO_ROOT)),
        n_gpus=total_ranks,
    )
    slurm_cfg = SlurmConfig(
        job_name=baseline_job_name,
        partition=None,
        n_gpus=GPUS_PER_NODE if n_nodes > 1 else total_ranks,
        n_nodes=n_nodes,
        time="00:30:00",
        snapshot_ref=stamp.snapshot_ref,
        comment="3-pool resumption validation: baseline leg",
    )
    r = submit_slurm_job(generate_script(slurm_cfg, cmd, env=CUDA_FLAGS), baseline_job_name)
    print(f"{baseline_job_name}: gpus={total_ranks} job_id={r.job_id}")
    print("waiting for baseline to finish…")
    _wait_for_job(r.job_id)

    # === Find the parent run dir the baseline wrote ===
    decomps_dir = PARAM_DECOMP_OUT_DIR / "decompositions"
    candidates = sorted(decomps_dir.glob("p-*"), key=lambda p: p.stat().st_mtime)
    parent_dir = candidates[-1] if candidates else None
    if parent_dir is None:
        raise RuntimeError(f"no decompositions/ output dir found under {decomps_dir}")
    print(f"parent run dir: {parent_dir}")
    # Spot-check the resume snapshots are there.
    expected = parent_dir / f"resume/step_{RESUME_FROM_STEP}"
    if not expected.exists():
        raise RuntimeError(f"baseline didn't write expected snapshot {expected}")
    print(f"found {expected}")

    # === Resume ===
    resume_yaml_path = OUT_DIR / "resume.yaml"
    resume_yaml_path.write_text(
        yaml.safe_dump(_resume_yaml_dict(parent_dir, RESUME_FROM_STEP), sort_keys=False)
    )
    print(f"resume yaml → {resume_yaml_path}")

    resume_stamp = ExecutionStamp.create(run_type="param_decomp", create_snapshot=True)
    resume_job_name = "resumeval-resume"
    resume_cmd = torchrun_command(
        job_name=resume_job_name,
        snapshot_ref=resume_stamp.snapshot_ref,
        python_module="param_decomp_lab.experiments.lm.run",
        # The lm/run.py CLI takes `--resume <path>`.
        script_args=f"--resume {resume_yaml_path.relative_to(REPO_ROOT)}",
        n_gpus=total_ranks,
    )
    resume_slurm = SlurmConfig(
        job_name=resume_job_name,
        partition=None,
        n_gpus=GPUS_PER_NODE if n_nodes > 1 else total_ranks,
        n_nodes=n_nodes,
        time="00:30:00",
        snapshot_ref=resume_stamp.snapshot_ref,
        comment="3-pool resumption validation: resume leg",
    )
    r2 = submit_slurm_job(
        generate_script(resume_slurm, resume_cmd, env=CUDA_FLAGS), resume_job_name
    )
    print(f"{resume_job_name}: gpus={total_ranks} job_id={r2.job_id}")
    print("waiting for resume to finish…")
    _wait_for_job(r2.job_id)
    print("done — inspect the two runs' logs to compare loss curves")
    print(f"  baseline log: ~/param_decomp_out/slurm_logs/slurm-{r.job_id}.out")
    print(f"  resume log:   ~/param_decomp_out/slurm_logs/slurm-{r2.job_id}.out")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
