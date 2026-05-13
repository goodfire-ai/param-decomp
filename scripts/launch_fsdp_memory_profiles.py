"""Submit Slurm arrays for FSDP memory profiling.

The launcher assumes it is run from the profiling worktree. It does not use a git
snapshot: jobs cd into this worktree and source its `.venv`, so uncommitted harness
changes are visible to the jobs.
"""

from __future__ import annotations

import argparse
import json
import shlex
import time
from dataclasses import dataclass
from pathlib import Path

from param_decomp.settings import REPO_ROOT
from param_decomp.utils.slurm import SlurmArrayConfig, generate_array_script, submit_slurm_job


@dataclass(frozen=True)
class ProfileJob:
    name: str
    strategy: str
    target_scale: str
    batch: int
    seq: int = 512
    ci_checkpointing: bool = False
    target_checkpointing: bool = False
    autocast_bf16: bool = True
    forward_autocast_bf16: bool | None = None
    delta_masks: bool = True
    include_faithfulness: bool = False
    decomposed_forwards: int = 2
    warmup_steps: int = 1
    measure_steps: int = 2
    record_snapshot: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["smoke", "jose", "scale", "autocast"], required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--partition", default="h200-reserved-default")
    parser.add_argument("--time", default="02:00:00")
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--out-root", type=Path, default=None)
    return parser.parse_args()


def bool_flag(name: str, value: bool) -> list[str]:
    return [f"--{name}" if value else f"--no-{name}"]


def suite_jobs(suite: str) -> list[ProfileJob]:
    if suite == "smoke":
        return [
            ProfileJob("jose_b1_fsdp_fp32_ci_ckpt", "fsdp", "jose", 1, ci_checkpointing=True, autocast_bf16=False),
            ProfileJob("jose_b1_fsdp_bf16_no_ci_ckpt", "fsdp", "jose", 1, ci_checkpointing=False, autocast_bf16=True),
            ProfileJob("jose_b1_fsdp_bf16_ci_ckpt", "fsdp", "jose", 1, ci_checkpointing=True, autocast_bf16=True),
            ProfileJob("jose_b1_zero1", "zero1", "jose", 1, ci_checkpointing=True),
            ProfileJob("jose_b1_ddp", "ddp", "jose", 1, ci_checkpointing=True),
        ]

    if suite == "jose":
        jobs: list[ProfileJob] = []
        for strategy in ["ddp", "zero1", "fsdp"]:
            for batch in [1, 4, 8, 16, 32]:
                fsdp = strategy == "fsdp"
                jobs.append(
                    ProfileJob(
                        f"jose_b{batch}_{strategy}_ci_ckpt",
                        strategy,
                        "jose",
                        batch,
                        ci_checkpointing=True,
                        autocast_bf16=not fsdp,
                    )
                )
        for batch in [8, 16, 32]:
            jobs.append(
                ProfileJob(
                    f"jose_b{batch}_fsdp_no_ci_ckpt",
                    "fsdp",
                    "jose",
                    batch,
                    ci_checkpointing=False,
                    autocast_bf16=False,
                )
            )
        return jobs

    if suite == "scale":
        jobs = []
        for target in ["1b", "2b", "4b"]:
            for batch in [1, 2]:
                jobs.append(
                    ProfileJob(
                        f"{target}_b{batch}_fsdp_ci_ckpt",
                        "fsdp",
                        target,
                        batch,
                        ci_checkpointing=True,
                        target_checkpointing=False,
                        autocast_bf16=False,
                    )
                )
            jobs.append(
                ProfileJob(
                    f"{target}_b1_fsdp_no_ckpt",
                    "fsdp",
                    target,
                    1,
                    ci_checkpointing=False,
                    target_checkpointing=False,
                    autocast_bf16=False,
                )
            )
        for target in ["1b", "2b"]:
            jobs.append(
                ProfileJob(
                    f"{target}_b1_zero1_ci_ckpt",
                    "zero1",
                    target,
                    1,
                    ci_checkpointing=True,
                    target_checkpointing=False,
                )
            )
        return jobs

    if suite == "autocast":
        return [
            ProfileJob(
                "jose_b8_fsdp_fp32_forward_bf16",
                "fsdp",
                "jose",
                8,
                ci_checkpointing=True,
                autocast_bf16=False,
                forward_autocast_bf16=True,
            ),
            ProfileJob(
                "jose_b32_fsdp_fp32_forward_bf16",
                "fsdp",
                "jose",
                32,
                ci_checkpointing=True,
                autocast_bf16=False,
                forward_autocast_bf16=True,
            ),
            ProfileJob(
                "1b_b1_fsdp_fp32_forward_bf16",
                "fsdp",
                "1b",
                1,
                ci_checkpointing=True,
                autocast_bf16=False,
                forward_autocast_bf16=True,
            ),
            ProfileJob(
                "4b_b1_fsdp_fp32_forward_bf16",
                "fsdp",
                "4b",
                1,
                ci_checkpointing=True,
                autocast_bf16=False,
                forward_autocast_bf16=True,
            ),
        ]

    raise AssertionError(f"unknown suite: {suite}")


def command_for_job(job: ProfileJob, gpus: int, out_root: Path, idx: int) -> str:
    out_dir = out_root / job.name
    args = [
        "scripts/fsdp_memory_profile.py",
        "--strategy",
        job.strategy,
        "--target-scale",
        job.target_scale,
        "--batch",
        str(job.batch),
        "--seq",
        str(job.seq),
        "--out-dir",
        str(out_dir),
        "--decomposed-forwards",
        str(job.decomposed_forwards),
        "--warmup-steps",
        str(job.warmup_steps),
        "--measure-steps",
        str(job.measure_steps),
        "--profile-label",
        job.name,
        *bool_flag("ci-checkpointing", job.ci_checkpointing),
        *bool_flag("target-checkpointing", job.target_checkpointing),
        *bool_flag("autocast-bf16", job.autocast_bf16),
        *bool_flag("delta-masks", job.delta_masks),
    ]
    if job.forward_autocast_bf16 is not None:
        args.extend(bool_flag("forward-autocast-bf16", job.forward_autocast_bf16))
    if job.include_faithfulness:
        args.append("--include-faithfulness")
    if job.record_snapshot:
        args.append("--record-snapshot")

    quoted_args = " ".join(shlex.quote(a) for a in args)
    port = 23000 + idx
    return (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        f"source .venv/bin/activate && "
        f"mkdir -p {shlex.quote(str(out_dir))} && "
        f"torchrun --standalone --nproc_per_node={gpus} --master_port={port} {quoted_args}"
    )


def main() -> None:
    args = parse_args()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_root = args.out_root or (REPO_ROOT / "profiling_runs" / f"{args.suite}-n{args.gpus}-{run_id}")
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = suite_jobs(args.suite)
    commands = [command_for_job(job, args.gpus, out_root, idx) for idx, job in enumerate(jobs)]
    comments = [job.name for job in jobs]

    manifest = {
        "suite": args.suite,
        "gpus": args.gpus,
        "partition": args.partition,
        "out_root": str(out_root),
        "jobs": [job.__dict__ for job in jobs],
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    script = generate_array_script(
        SlurmArrayConfig(
            job_name=f"fsdp-prof-{args.suite}-n{args.gpus}",
            partition=args.partition,
            n_gpus=args.gpus,
            n_nodes=1,
            time=args.time,
            max_concurrent_tasks=args.max_concurrent,
        ),
        commands=commands,
        env={
            "NCCL_DEBUG": "WARN",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        per_task_comments=comments,
    )
    result = submit_slurm_job(
        script,
        script_name_prefix=f"fsdp_profile_{args.suite}_n{args.gpus}",
        is_array=True,
        n_array_tasks=len(commands),
    )
    print(f"submitted job_id={result.job_id}")
    print(f"script={result.script_path}")
    print(f"logs={result.log_pattern}")
    print(f"out_root={out_root}")


if __name__ == "__main__":
    main()
