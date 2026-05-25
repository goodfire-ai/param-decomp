"""Multi-seed sweep: 1-pool DDP=2 vs 2-pool 5×1 across 10 seeds each.

Tests whether the small "stoch ~2-4% higher in multi-pool" effect we saw
with N=1 is real or single-seed noise. Each pair shares a seed and a git
snapshot, so 1-pool[seed=k] and 2-pool[seed=k] differ only in topology.

Per-yaml-per-seed: write a tiny override yaml that re-references the base
yaml plus a ``pd.seed`` override. (Pydantic loads the full tree, so we
generate a complete yaml each time.)
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from param_decomp_lab.infra.run_files import ExecutionStamp
from param_decomp_lab.infra.settings import REPO_ROOT
from param_decomp_lab.infra.slurm import SlurmConfig, generate_script, submit_slurm_job


@dataclass(frozen=True)
class _Spec:
    name: str
    base_yaml: Path
    n_gpus: int


SPECS: tuple[_Spec, ...] = (
    _Spec(
        name="ms-1pool",
        base_yaml=REPO_ROOT / "param_decomp_lab/experiments/lm/equiv_5L_1pool.yaml",
        n_gpus=2,
    ),
    _Spec(
        name="ms-2pool",
        base_yaml=REPO_ROOT / "param_decomp_lab/experiments/lm/equiv_5L_2pool.yaml",
        n_gpus=6,
    ),
)

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

# Materialized yamls live under repo (so they're picked up by the snapshot).
# Untracked — each run wipes & re-materializes so stale yamls from a prior
# sweep don't leak into the snapshot.
OUT_DIR = REPO_ROOT / "param_decomp_lab/experiments/lm/_multiseed"


def _materialize(spec: _Spec, seed: int) -> Path:
    with open(spec.base_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg["pd"]["seed"] = seed
    out = OUT_DIR / f"{spec.name}_seed{seed}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def _wipe_out_dir() -> None:
    """Remove any stale materialized yamls before re-generating."""
    if OUT_DIR.is_dir():
        for p in OUT_DIR.iterdir():
            if p.is_file():
                p.unlink()


def main() -> None:
    _wipe_out_dir()
    paths: list[tuple[_Spec, int, Path]] = []
    for spec in SPECS:
        assert spec.base_yaml.is_file(), f"base yaml not found: {spec.base_yaml}"
        for seed in SEEDS:
            p = _materialize(spec, seed)
            paths.append((spec, seed, p))

    execution_stamp = ExecutionStamp.create(run_type="param_decomp", create_snapshot=True)
    print(f"Snapshot ref: {execution_stamp.snapshot_ref}")

    for spec, seed, yaml_path in paths:
        job_name = f"{spec.name}-s{seed}"
        torchrun_cmd = (
            f"torchrun --standalone --nproc_per_node={spec.n_gpus} "
            f"-m param_decomp_lab.experiments.lm.run {yaml_path}"
        )
        cfg = SlurmConfig(
            job_name=job_name,
            partition=None,
            n_gpus=spec.n_gpus,
            time="02:00:00",  # warmup=400 + steps=200 → ~3× longer than old runs
            snapshot_ref=execution_stamp.snapshot_ref,
            comment=f"multi-seed equivalence — {job_name}",
        )
        script = generate_script(cfg, torchrun_cmd)
        result = submit_slurm_job(script, job_name)
        print(f"{job_name}: job_id={result.job_id}  log={result.log_pattern}")


if __name__ == "__main__":
    main()
