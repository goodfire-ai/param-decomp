"""Submit a wide batch of equivalence-test variants in parallel.

Each spec names a yaml + the rank count to launch with. All spec rows share
the same git snapshot so the codebase is identical across runs.

Job IDs land in the experiment registry (scripts/equiv_5L_experiments.md);
the launcher prints them so they can be copy-pasted.
"""

from dataclasses import dataclass
from pathlib import Path

from param_decomp_lab.infra.run_files import ExecutionStamp
from param_decomp_lab.infra.settings import REPO_ROOT
from param_decomp_lab.infra.slurm import SlurmConfig, generate_script, submit_slurm_job


@dataclass(frozen=True)
class _Spec:
    name: str
    yaml: Path
    n_gpus: int


SPECS: tuple[_Spec, ...] = (
    # Baselines (all 3 strategies, simplest topologies).
    _Spec(
        name="equiv-1pool",
        yaml=REPO_ROOT / "param_decomp_lab/experiments/lm/equiv_5L_1pool.yaml",
        n_gpus=2,
    ),
    _Spec(
        name="equiv-2pool",
        yaml=REPO_ROOT / "param_decomp_lab/experiments/lm/equiv_5L_2pool.yaml",
        n_gpus=6,
    ),
    _Spec(
        name="equiv-3pool",
        yaml=REPO_ROOT / "param_decomp_lab/experiments/lm/equiv_5L_3pool.yaml",
        n_gpus=8,
    ),
    # 2-pool topology variants — different DDP arities.
    _Spec(
        name="equiv-2pool-nperblock2",
        yaml=REPO_ROOT / "param_decomp_lab/experiments/lm/equiv_5L_2pool_nperblock2.yaml",
        n_gpus=8,
    ),
    _Spec(
        name="equiv-2pool-poolb2",
        yaml=REPO_ROOT / "param_decomp_lab/experiments/lm/equiv_5L_2pool_poolb2.yaml",
        n_gpus=7,
    ),
    _Spec(
        name="equiv-2pool-1block4r",
        yaml=REPO_ROOT / "param_decomp_lab/experiments/lm/equiv_5L_2pool_1block4r.yaml",
        n_gpus=5,
    ),
    # 3-pool topology variant.
    _Spec(
        name="equiv-3pool-2blocks",
        yaml=REPO_ROOT / "param_decomp_lab/experiments/lm/equiv_5L_3pool_2blocks.yaml",
        n_gpus=8,
    ),
)


def main() -> None:
    execution_stamp = ExecutionStamp.create(run_type="param_decomp", create_snapshot=True)
    print(f"Snapshot ref: {execution_stamp.snapshot_ref}")
    for spec in SPECS:
        assert spec.yaml.is_file(), f"yaml not found: {spec.yaml}"
        torchrun_cmd = (
            f"torchrun --standalone --nproc_per_node={spec.n_gpus} "
            f"-m param_decomp_lab.experiments.lm.run {spec.yaml}"
        )
        cfg = SlurmConfig(
            job_name=spec.name,
            partition=None,
            n_gpus=spec.n_gpus,
            time="01:00:00",
            snapshot_ref=execution_stamp.snapshot_ref,
            comment=f"equivalence test — {spec.name}",
        )
        script = generate_script(cfg, torchrun_cmd)
        result = submit_slurm_job(script, spec.name)
        print(f"{spec.name}: job_id={result.job_id}  log={result.log_pattern}")


if __name__ == "__main__":
    main()
