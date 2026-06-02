"""Submit the 64-GPU b256 3-pool profiling run with profiler + memory env injected.

`pd-lm-3pool --dp 64` would submit the same multi-node job but only forwards the DDP env
(NCCL flags) — it does NOT propagate the `PD_TORCH_PROFILE_*` / `PD_MEMORY_PROFILE_*`
knobs. This launcher reuses the same infra (`create_git_snapshot` + `build_ddp_launch` +
`generate_script` + `submit_slurm_job`) and adds those knobs to the exported env so the
srun-propagated environment reaches every rank's torchrun → python process.

Profiles one rank per pool (LW leader 0, CI leader 48, PPGD leader 56). The profiler
active window (steps 44-47) is placed OFF the metric-log steps (every 10) so rank 0 never
does a `.item()`-after-`recv` while CUPTI is collecting — the historical ≥64-rank deadlock
mode. Memory history (CUPTI-free) is captured on the same three ranks regardless.
"""

from param_decomp_lab.infra.ddp_launch import build_ddp_launch
from param_decomp_lab.infra.git import create_git_snapshot
from param_decomp_lab.infra.run_files import generate_run_id
from param_decomp_lab.infra.slurm import SlurmConfig, generate_script, submit_slurm_job

CONFIG = "param_decomp_lab/experiments/lm/_b256_run/b256_profile_64.yaml"
OUT = "/mnt/home/dan.braun/param-decomp/profiling_out/b256_profile_64"
DP = 72  # 48 LW + 8 CI + 16 PPGD (must match the topology in b256_profile_64.yaml)
PROFILE_RANKS = "0,48,56"  # LW block-0 leader, CI leader, PPGD leader


def main() -> None:
    run_id = generate_run_id("param_decomp")
    snapshot_ref, commit = create_git_snapshot(snapshot_id=run_id)
    print(f"run_id={run_id}  snapshot={snapshot_ref} ({commit[:8]})")

    base_command = f"-m param_decomp_lab.experiments.lm.three_pool_run {CONFIG} --run_id {run_id}"
    launch = build_ddp_launch(
        base_command, dp=DP, job_name="b256-profile", snapshot_ref=snapshot_ref, port_seed=run_id
    )
    env = {
        **launch.env,
        "PYTHONUNBUFFERED": "1",
        # PPGD's recon is memory-tight; reduce allocator fragmentation (the OOM'd run had
        # ~1.4GB reserved-but-unallocated). Standard for these activation-heavy workloads.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PD_TORCH_PROFILE_RANKS": PROFILE_RANKS,
        "PD_TORCH_PROFILE_OUT": f"{OUT}/traces",
        "PD_TORCH_PROFILE_SKIP_FIRST": "43",
        "PD_TORCH_PROFILE_ACTIVE": "4",
        "PD_TORCH_PROFILE_MEMORY": "0",
        "PD_TORCH_PROFILE_SHAPES": "1",
        "PD_MEMORY_PROFILE_RANKS": PROFILE_RANKS,
        "PD_MEMORY_PROFILE_OUT": f"{OUT}/mem",
    }
    slurm_cfg = SlurmConfig(
        job_name="b256-profile",
        partition=None,
        n_gpus=launch.gpus_per_node,
        n_nodes=launch.n_nodes,
        time="01:00:00",
        snapshot_ref=snapshot_ref,
        comment=run_id,
    )
    result = submit_slurm_job(generate_script(slurm_cfg, launch.command, env=env), "b256-profile")
    print(f"job_id={result.job_id}  nodes={launch.n_nodes}  gpus={DP}")
    print(f"log={result.log_pattern}")
    print(f"traces -> {OUT}/traces   mem -> {OUT}/mem")


if __name__ == "__main__":
    main()
