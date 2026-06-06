#!/bin/bash
# Minimal multi-node NCCL / InfiniBand verbs failure reproduction.
#
# Reproduces:
#   torch.distributed.DistBackendError: NCCL error ... ncclSystemError ...
#     Call to ibv_create_qp failed with error Cannot allocate memory
#     Call to ibv_reg_mr_iova2 failed
#
# Observed when running PyTorch DDP across 2 nodes (16 H100s total). A single
# 8-GPU node works fine -- the failure shows up only when NCCL has to set up
# IB queue pairs / memory regions for cross-node traffic.
#
# Submit with: sbatch scripts/nccl_ib_repro.sbatch
#
#SBATCH --job-name=nccl-ib-repro
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=nccl_ib_repro-%j.out

set -uo pipefail

# Hardcoded so the path is identical on every compute node. SLURM copies the
# submit script to /var/spool/slurmd on the node, so $0/dirname won't work.
REPO_DIR="/mnt/delicate-frog/home/braun/param-decomp"

echo "================================================================"
echo "SLURM job:      ${SLURM_JOB_ID:-N/A}"
echo "Allocated:      ${SLURM_JOB_NODELIST:-N/A}"
echo "Date:           $(date -Is)"
echo "Repo:           $REPO_DIR"
echo "================================================================"

# --- per-node diagnostics ---
echo
echo "--- per-node diagnostics ---"
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -c '
    echo "==== $(hostname) ===="
    echo "ulimit -l (locked memory, KB): $(ulimit -l)"
    echo "ulimit -n (open files):        $(ulimit -n)"
    echo
    echo "-- pre-existing NCCL_* env on this node --"
    env | grep -E "^NCCL_|^UCX_|^LD_PRELOAD" | sort || true
    echo
    echo "-- nvidia-smi --"
    nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
    echo
    echo "-- IB devices (ibstat) --"
    if command -v ibstat >/dev/null; then ibstat | head -60; else echo "ibstat not installed"; fi
    echo
    echo "-- ibv_devinfo (truncated) --"
    if command -v ibv_devinfo >/dev/null; then ibv_devinfo 2>&1 | head -40; else echo "ibv_devinfo not installed"; fi
    echo
'

# --- torch / NCCL version on rank-0 node ---
echo
echo "--- torch / NCCL versions ---"
cd "$REPO_DIR"
source .venv/bin/activate
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'nccl', torch.cuda.nccl.version())"

# Verbose NCCL diagnostics so providers see which HCA / port / transport NCCL picks.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH

# Multi-node torchrun config
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# Write the workload to a shared location both nodes can see.
WORKLOAD="$REPO_DIR/scripts/.nccl_ib_repro_workload.py"
cat > "$WORKLOAD" <<'PY'
import os
import socket
import time
import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(
        f"[rank {rank}/{world}] host={socket.gethostname()} local_rank={local_rank}",
        flush=True,
    )

    # First collective -- this is where NCCL builds the cross-node IB QPs and
    # registers memory regions. The failure in the real workload also appears
    # at the first collective (inside loss.backward()).
    x = torch.ones(1024, 1024, device="cuda")
    t0 = time.time()
    dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = time.time() - t0

    expected = float(world)
    got = x[0, 0].item()
    assert got == expected, f"all_reduce wrong: got {got}, expected {expected}"
    if rank == 0:
        print(
            f"all_reduce OK across {world} ranks in {dt:.3f}s "
            f"(value {got} == expected {expected})",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY
trap 'rm -f "$WORKLOAD"' EXIT

run_test() {
    local label="$1"
    shift
    echo
    echo "================================================================"
    echo "RUN: $label"
    echo "Env overrides: $*"
    echo "================================================================"
    # One srun task per node; each task launches torchrun with 8 procs.
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -c "
        cd '$REPO_DIR'
        source .venv/bin/activate
        export NCCL_DEBUG=$NCCL_DEBUG
        export NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS
        $* torchrun \
            --nnodes=2 \
            --node_rank=\$SLURM_PROCID \
            --nproc_per_node=8 \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            '$WORKLOAD'
    "
    local rc=$?
    echo "exit code: $rc"
    return $rc
}

# 1. Default path: lets NCCL pick its transport (will use IB on this cluster).
#    This is the case that reproduces the failure.
run_test "default (NCCL chooses IB)"
rc_default=$?

# 2. Force NCCL onto TCP only. NCCL_IB_DISABLE=1 alone is not enough on this
#    cluster because the NCCL RDMA Plugin is loaded by env (see NCCL_NET_PLUGIN
#    in the per-node diagnostics). NCCL_NET_PLUGIN=none disables the plugin so
#    the bundled NET/Socket path is used. Expected to succeed; isolates the
#    failure to the IB stack rather than NCCL or PyTorch.
run_test "NCCL_IB_DISABLE=1 + NCCL_NET_PLUGIN=none (TCP fallback)" \
    NCCL_IB_DISABLE=1 NCCL_NET_PLUGIN=none
rc_no_ib=$?

echo
echo "================================================================"
echo "Summary"
echo "  default (IB):              exit $rc_default"
echo "  NCCL_IB_DISABLE=1 (TCP):   exit $rc_no_ib"
echo "================================================================"

exit $rc_default
