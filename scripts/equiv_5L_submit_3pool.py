"""Submit just the 3-pool equivalence-test run (used to iterate on the 3-pool
configuration without re-running 1-pool / 2-pool).
"""

from param_decomp_lab.infra.run_files import ExecutionStamp
from param_decomp_lab.infra.settings import REPO_ROOT
from param_decomp_lab.infra.slurm import SlurmConfig, generate_script, submit_slurm_job


def main() -> None:
    execution_stamp = ExecutionStamp.create(run_type="param_decomp", create_snapshot=True)
    print(f"Snapshot ref: {execution_stamp.snapshot_ref}")

    yaml = REPO_ROOT / "param_decomp_lab/experiments/lm/equiv_5L_3pool.yaml"
    n_gpus = 8
    name = "equiv-3pool"
    torchrun_cmd = (
        f"torchrun --standalone --nproc_per_node={n_gpus} "
        f"-m param_decomp_lab.experiments.lm.run {yaml}"
    )
    cfg = SlurmConfig(
        job_name=name,
        partition=None,
        n_gpus=n_gpus,
        time="01:00:00",
        snapshot_ref=execution_stamp.snapshot_ref,
        comment=f"5-layer GPT-2 equivalence test — {name}",
    )
    script = generate_script(cfg, torchrun_cmd)
    result = submit_slurm_job(script, name)
    print(f"{name}: job_id={result.job_id}  log={result.log_pattern}")


if __name__ == "__main__":
    main()
