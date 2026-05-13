"""Submit real train-step profiling jobs.

The launcher uses the current worktree directly so uncommitted profiling changes are visible to
the jobs. Submit the main scaling matrix with:

    python scripts/launch_train_step_profiles.py --suite baseline --gpus 8 16 32
"""

from __future__ import annotations

import argparse
import json
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path

from param_decomp.settings import DEFAULT_PARTITION_NAME, REPO_ROOT
from param_decomp.utils.slurm import (
    SlurmArrayConfig,
    generate_array_script,
    submit_slurm_job,
)

GPUS_PER_NODE = 8


@dataclass(frozen=True)
class TrainStepProfileJob:
    name: str
    strategy: str = "ddp"
    disable_losses: list[str] = field(default_factory=list)
    ppgd_warmup_steps: int | None = None
    use_delta_component: bool | None = None
    include_train_logging: bool = False
    trace_steps: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["baseline", "trace", "ablation"], default="baseline")
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=REPO_ROOT / "param_decomp/experiments/lm/pile_llama_simple_mlp-4L.yaml",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--partition", default=DEFAULT_PARTITION_NAME)
    parser.add_argument("--time", default="06:00:00")
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def bool_flag(name: str, value: bool) -> list[str]:
    return [f"--{name}" if value else f"--no-{name}"]


def suite_jobs(suite: str) -> list[TrainStepProfileJob]:
    if suite == "baseline":
        return [TrainStepProfileJob(name="baseline_ddp")]

    if suite == "trace":
        return [TrainStepProfileJob(name="trace_ddp", trace_steps=1)]

    if suite == "ablation":
        return [
            TrainStepProfileJob(name="baseline_ddp"),
            TrainStepProfileJob(name="no_faithfulness", disable_losses=["FaithfulnessLoss"]),
            TrainStepProfileJob(name="ppgd_warmup0", ppgd_warmup_steps=0),
            TrainStepProfileJob(name="zero1", strategy="zero1"),
            TrainStepProfileJob(name="train_logging", include_train_logging=True),
        ]

    raise AssertionError(f"unknown suite: {suite}")


def validate_gpus(n_gpus: int) -> tuple[int, int]:
    if n_gpus < 2:
        raise ValueError("--gpus values must be at least 2 for distributed profiling")
    if n_gpus <= GPUS_PER_NODE:
        return 1, n_gpus
    if n_gpus % GPUS_PER_NODE != 0:
        raise ValueError(f"multi-node gpu counts must be divisible by {GPUS_PER_NODE}: {n_gpus}")
    return n_gpus // GPUS_PER_NODE, GPUS_PER_NODE


def profiler_args(
    *,
    job: TrainStepProfileJob,
    config_path: Path,
    out_dir: Path,
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
) -> list[str]:
    args = [
        "scripts/profile_train_step.py",
        "--config-path",
        str(config_path),
        "--out-dir",
        str(out_dir),
        "--profile-label",
        job.name,
        "--strategy",
        job.strategy,
        "--batch-size",
        str(batch_size),
        "--warmup-steps",
        str(warmup_steps),
        "--measure-steps",
        str(measure_steps),
        "--trace-steps",
        str(job.trace_steps),
    ]
    for loss_name in job.disable_losses:
        args.extend(["--disable-loss", loss_name])
    if job.ppgd_warmup_steps is not None:
        args.extend(["--ppgd-warmup-steps", str(job.ppgd_warmup_steps)])
    if job.use_delta_component is not None:
        args.extend(bool_flag("use-delta-component", job.use_delta_component))
    if job.include_train_logging:
        args.append("--include-train-logging")
    return args


def command_for_job(
    *,
    job: TrainStepProfileJob,
    n_gpus: int,
    config_path: Path,
    out_root: Path,
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
    idx: int,
) -> str:
    n_nodes, gpus_per_node = validate_gpus(n_gpus)
    out_dir = out_root / f"n{n_gpus}" / job.name
    script_args = profiler_args(
        job=job,
        config_path=config_path,
        out_dir=out_dir,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
    )
    quoted_args = " ".join(shlex.quote(a) for a in script_args)
    port = 24000 + idx

    common_setup = (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        "source .venv/bin/activate && "
        f"mkdir -p {shlex.quote(str(out_dir))}"
    )

    if n_nodes == 1:
        return (
            f"{common_setup} && "
            f"torchrun --standalone --nproc_per_node={gpus_per_node} "
            f"--master_port={port} {quoted_args}"
        )

    torchrun_cmd = (
        f"torchrun "
        f"--nnodes={n_nodes} "
        f"--node_rank=$SLURM_PROCID "
        f"--nproc_per_node={GPUS_PER_NODE} "
        f'--master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) '
        f"--master_port={port} "
        f"{quoted_args}"
    )
    per_node_command = f"{common_setup} && {torchrun_cmd}"
    srun_flags = f"--nodes={n_nodes} --ntasks={n_nodes} --ntasks-per-node=1"
    return f"srun {srun_flags} bash -c {shlex.quote(per_node_command)}"


def submit_for_gpu_count(args: argparse.Namespace, n_gpus: int, out_root: Path) -> None:
    n_nodes, gpus_per_node = validate_gpus(n_gpus)
    jobs = suite_jobs(args.suite)
    commands = [
        command_for_job(
            job=job,
            n_gpus=n_gpus,
            config_path=args.config_path,
            out_root=out_root,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            idx=idx,
        )
        for idx, job in enumerate(jobs)
    ]

    manifest = {
        "suite": args.suite,
        "gpus": n_gpus,
        "n_nodes": n_nodes,
        "gpus_per_node": gpus_per_node,
        "partition": args.partition,
        "config_path": str(args.config_path),
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "out_root": str(out_root),
        "jobs": [job.__dict__ for job in jobs],
        "commands": commands,
    }
    manifest_path = out_root / f"manifest_n{n_gpus}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    script = generate_array_script(
        SlurmArrayConfig(
            job_name=f"train-step-prof-{args.suite}-n{n_gpus}",
            partition=args.partition,
            n_gpus=gpus_per_node,
            n_nodes=n_nodes,
            time=args.time,
            max_concurrent_tasks=args.max_concurrent,
        ),
        commands=commands,
        env={
            "NCCL_DEBUG": "WARN",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "WANDB_DISABLED": "true",
            "WANDB_MODE": "offline",
            "WANDB_SILENT": "true",
        },
        per_task_comments=[job.name for job in jobs],
    )

    if args.dry_run:
        script_path = out_root / f"dry_run_n{n_gpus}.sh"
        script_path.write_text(script)
        print(f"wrote dry-run script={script_path}")
        print(f"manifest={manifest_path}")
        return

    result = submit_slurm_job(
        script,
        script_name_prefix=f"train_step_profile_{args.suite}_n{n_gpus}",
        is_array=True,
        n_array_tasks=len(commands),
    )
    print(f"submitted n{n_gpus} job_id={result.job_id}")
    print(f"script={result.script_path}")
    print(f"logs={result.log_pattern}")
    print(f"manifest={manifest_path}")


def main() -> None:
    args = parse_args()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_root = args.out_root or (REPO_ROOT / "profiling_runs" / f"train-step-{args.suite}-{run_id}")
    out_root.mkdir(parents=True, exist_ok=True)

    for n_gpus in args.gpus:
        submit_for_gpu_count(args, n_gpus, out_root)

    print(f"out_root={out_root}")


if __name__ == "__main__":
    main()
