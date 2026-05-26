"""SLURM submission for pretraining jobs."""

import subprocess
import sys
from pathlib import Path

from param_decomp.log import logger
from param_decomp_lab.infra.run_files import ExecutionStamp
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME, SLURM_LOGS_DIR
from param_decomp_lab.infra.slurm import SlurmConfig, generate_script, submit_slurm_job


def main(
    config_path: str,
    n_gpus: int = 1,
    partition: str | None = DEFAULT_PARTITION_NAME,
    time: str = "72:00:00",
    job_name: str = "pd-pretrain",
    local: bool = False,
) -> None:
    """Submit a pretraining job to SLURM, or run it locally with `--local`."""
    config_path_resolved = Path(config_path)
    assert config_path_resolved.exists(), f"Config not found: {config_path}"

    if local:
        _run_local(config_path_resolved, n_gpus)
    else:
        _submit_slurm(config_path_resolved, n_gpus, partition, time, job_name)


def _run_local(config_path: Path, n_gpus: int) -> None:
    """Run training in the current shell via `torchrun` (multi-GPU) or `python -m` (single)."""
    if n_gpus > 1:
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={n_gpus}",
            "-m",
            "param_decomp_lab.experiments.lm.pretrain.train",
            str(config_path),
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "param_decomp_lab.experiments.lm.pretrain.train",
            str(config_path),
        ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _submit_slurm(
    config_path: Path,
    n_gpus: int,
    partition: str | None,
    time: str,
    job_name: str,
) -> None:
    """Submit a `torchrun` invocation of `param_decomp_lab.experiments.lm.pretrain.train` to SLURM.

    Wraps it in a batch script and submits via `sbatch`. Creates an `ExecutionStamp`
    with a git snapshot for reproducibility.
    """
    SLURM_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Create git snapshot for reproducibility
    execution_stamp = ExecutionStamp.create(run_type="train", create_snapshot=True)
    logger.info(f"Run ID: {execution_stamp.run_id}")
    logger.info(f"Snapshot ref: {execution_stamp.snapshot_ref}")

    # Build the training command
    train_cmd = f"torchrun --standalone --nproc_per_node={n_gpus} -m param_decomp_lab.experiments.lm.pretrain.train {config_path}"

    config = SlurmConfig(
        job_name=job_name,
        partition=partition,
        n_gpus=n_gpus,
        time=time,
        snapshot_ref=execution_stamp.snapshot_ref,
    )

    script = generate_script(config, train_cmd)
    result = submit_slurm_job(script, job_name)

    print(f"Submitted job {result.job_id}")
    print(f"Log file: {result.log_pattern}")


def cli() -> None:
    """CLI entry point for pd-pretrain command."""
    import fire

    fire.Fire(main)


if __name__ == "__main__":
    cli()
