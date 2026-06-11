"""Wrap a `-m <module> [args...]` invocation in a torchrun (single-node) or
srun + torchrun (multi-node) launcher, and report the SLURM topology to feed
into `SlurmConfig`.
"""

import shlex
from dataclasses import dataclass
from hashlib import sha256

from param_decomp_lab.infra.slurm import (
    SINGLETON_JOB_ID_BASH,
    generate_git_snapshot_setup,
)

GPUS_PER_NODE = 8

# Surface NCCL collective failures as Python exceptions instead of hanging the job —
# matters for multi-node where a single stalled rank otherwise hangs everyone silently.
# HF_HUB_*_TIMEOUT raise HF's 10s default: every rank hits the Hub at startup to resolve
# the dataset, and that burst can blow the 10s wire (hf_http.py retries catch the rest).
DDP_ENV = {
    "NCCL_DEBUG": "WARN",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "HF_HUB_ETAG_TIMEOUT": "30",
    "HF_HUB_DOWNLOAD_TIMEOUT": "30",
}


@dataclass(frozen=True)
class DDPLaunch:
    """A torchrun/srun-wrapped command and the SLURM topology it expects."""

    command: str
    n_nodes: int
    gpus_per_node: int
    env: dict[str, str]


def build_ddp_launch(
    base_command: str,
    *,
    dp: int,
    job_name: str,
    snapshot_ref: str | None,
    port_seed: str,
) -> DDPLaunch:
    """Wrap `base_command` (everything after `python`, e.g. `-m foo cfg.yaml`).

    `dp <= GPUS_PER_NODE` → single-node `torchrun --standalone`.
    `dp > GPUS_PER_NODE` → `srun bash -c '<setup>; torchrun --node_rank=$SLURM_PROCID ...'`.
    Multi-node requires `dp % GPUS_PER_NODE == 0` and a `snapshot_ref`: `/tmp` is
    node-local so each node clones its own workspace from the snapshot.

    `port_seed` keys the master port deterministically — pass the run id so parallel
    launches don't collide.
    """
    assert dp >= 2, f"dp must be at least 2 for DDP (got {dp})"
    port = _choose_master_port(port_seed)

    if dp <= GPUS_PER_NODE:
        command = " ".join(
            [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={dp}",
                f"--master_port={port}",
                base_command,
            ]
        )
        return DDPLaunch(command=command, n_nodes=1, gpus_per_node=dp, env=DDP_ENV)

    assert dp % GPUS_PER_NODE == 0, (
        f"Multi-node DDP requires dp ({dp}) to be a multiple of {GPUS_PER_NODE}"
    )
    assert snapshot_ref is not None, "Multi-node DDP requires a snapshot ref"

    n_nodes = dp // GPUS_PER_NODE
    # /tmp is node-local, so each node clones the snapshot into its own workspace.
    work_dir = (
        f"/tmp/$USER/param-decomp/workspace-{job_name}-{SINGLETON_JOB_ID_BASH}-node$SLURM_PROCID"
    )
    setup = generate_git_snapshot_setup(work_dir, snapshot_ref)
    torchrun_cmd = " ".join(
        [
            "torchrun",
            f"--nnodes={n_nodes}",
            "--node_rank=$SLURM_PROCID",
            f"--nproc_per_node={GPUS_PER_NODE}",
            '--master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)',
            f"--master_port={port}",
            base_command,
        ]
    )
    srun_prefix = (
        f"srun --nodes={n_nodes} --ntasks={n_nodes} --ntasks-per-node=1 --kill-on-bad-exit=1"
    )
    command = f"{srun_prefix} bash -c {shlex.quote(f'{setup}\n{torchrun_cmd}')}"
    return DDPLaunch(command=command, n_nodes=n_nodes, gpus_per_node=GPUS_PER_NODE, env=DDP_ENV)


def _choose_master_port(seed: str) -> int:
    """Stable, unprivileged port in [20000, 40000)."""
    return 20000 + (int(sha256(seed.encode()).hexdigest(), 16) % 20000)
