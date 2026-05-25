"""Submit the three 5-layer GPT-2 equivalence-test runs to SLURM.

Three configs, identical pd/runtime/target/data, differing only in pool
topology:

    equiv_5L_1pool.yaml  — 1-pool, 2 GPUs (DDP arity 2)
    equiv_5L_2pool.yaml  — 2-pool, 6 GPUs (5 LW × 1 + 1 pool-B)
    equiv_5L_3pool.yaml  — 3-pool, 7 GPUs (5 LW × 1 + 1 CI + 1 PPGD)

Each job runs ``torchrun --standalone --nproc_per_node=N -m
param_decomp_lab.experiments.lm.run <yaml>``.

After completion, each run writes ``metrics.jsonl`` under
``$PARAM_DECOMP_OUT_DIR/decompositions/<run_id>/`` — the loss curves to
compare across strategies.
"""

from dataclasses import dataclass
from pathlib import Path

from param_decomp_lab.infra.run_files import ExecutionStamp
from param_decomp_lab.infra.settings import REPO_ROOT
from param_decomp_lab.infra.slurm import SlurmConfig, generate_script, submit_slurm_job


@dataclass(frozen=True)
class _RunSpec:
    name: str
    yaml: Path
    n_gpus: int


SPECS: tuple[_RunSpec, ...] = (
    _RunSpec(
        name="equiv-1pool",
        yaml=REPO_ROOT / "param_decomp_lab/experiments/lm/equiv_5L_1pool.yaml",
        n_gpus=2,
    ),
    _RunSpec(
        name="equiv-2pool",
        yaml=REPO_ROOT / "param_decomp_lab/experiments/lm/equiv_5L_2pool.yaml",
        n_gpus=6,
    ),
    _RunSpec(
        name="equiv-3pool",
        yaml=REPO_ROOT / "param_decomp_lab/experiments/lm/equiv_5L_3pool.yaml",
        n_gpus=7,
    ),
)


def main() -> None:
    """Submit all three equivalence-test runs and print their job IDs."""
    # One git snapshot shared across the three jobs so they're guaranteed to run
    # against the same code revision.
    execution_stamp = ExecutionStamp.create(run_type="param_decomp", create_snapshot=True)
    print(f"Snapshot ref: {execution_stamp.snapshot_ref}")

    for spec in SPECS:
        torchrun_cmd = (
            f"torchrun --standalone --nproc_per_node={spec.n_gpus} "
            f"-m param_decomp_lab.experiments.lm.run {spec.yaml}"
        )
        cfg = SlurmConfig(
            job_name=spec.name,
            partition=None,  # rely on cluster default (b200)
            n_gpus=spec.n_gpus,
            time="01:00:00",
            snapshot_ref=execution_stamp.snapshot_ref,
            comment=f"5-layer GPT-2 equivalence test — {spec.name}",
        )
        script = generate_script(cfg, torchrun_cmd)
        result = submit_slurm_job(script, spec.name)
        print(f"{spec.name}: job_id={result.job_id}  log={result.log_pattern}")


if __name__ == "__main__":
    main()
