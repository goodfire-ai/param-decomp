#!/bin/bash
# Simple "is InfiniBand actually working?" check across 2 nodes.
#
# A passing NCCL all_reduce alone does NOT prove IB works -- NCCL silently falls
# back to TCP if IB init fails. So this script checks three things:
#   1. The all_reduce completes and gives the correct result.
#   2. NCCL's own logs show it selected the IB transport (NET/IB), not Socket.
#   3. Cross-node bus bandwidth is in the IB regime (10s of GB/s), not TCP (~1 GB/s).
#
# Submit with: sbatch scripts/ib_check.sbatch
#
#SBATCH --job-name=ib-check
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=ib_check-%j.out

set -uo pipefail

# Hardcoded: SLURM copies this script to /var/spool/slurmd on each node, so
# $0/dirname is useless. Identical path on every node.
REPO_DIR="/mnt/delicate-frog/home/braun/param-decomp"

echo "================================================================"
echo "SLURM job:  ${SLURM_JOB_ID:-N/A}"
echo "Nodes:      ${SLURM_JOB_NODELIST:-N/A}"
echo "Date:       $(date -Is)"
echo "================================================================"

# --- per-node IB device presence ---
echo
echo "--- IB devices per node (ibstat -l) ---"
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -c '
    echo "==== $(hostname) ===="
    if command -v ibstat >/dev/null; then
        ibstat -l
        # Show link state / rate for each device, one line each.
        for d in $(ibstat -l); do
            state=$(ibstat "$d" 2>/dev/null | grep -m1 "State:" | tr -s " ")
            rate=$(ibstat "$d" 2>/dev/null | grep -m1 "Rate:" | tr -s " ")
            echo "  $d:$state$rate"
        done
    else
        echo "ibstat not installed"
    fi
'

# --- torch / NCCL versions ---
echo
echo "--- torch / NCCL versions ---"
cd "$REPO_DIR"
source .venv/bin/activate
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'nccl', torch.cuda.nccl.version())"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# Per-rank NCCL logs land here so we can inspect which transport was chosen.
NCCL_LOG_DIR="$REPO_DIR/scripts/.ib_check_nccl_logs"
rm -rf "$NCCL_LOG_DIR"
mkdir -p "$NCCL_LOG_DIR"

WORKLOAD="$REPO_DIR/scripts/.ib_check_workload.py"
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
    print(f"[rank {rank}/{world}] host={socket.gethostname()} local_rank={local_rank}", flush=True)

    # 1. Correctness: first collective builds the cross-node IB QPs / mem regions.
    x = torch.ones(1024, 1024, device="cuda")
    dist.all_reduce(x)
    torch.cuda.synchronize()
    got, expected = x[0, 0].item(), float(world)
    assert got == expected, f"all_reduce wrong: got {got}, expected {expected}"

    # 2. Bandwidth: large tensor, timed over several iters after a warmup.
    #    bus_bw uses the standard ring all_reduce factor 2*(N-1)/N.
    numel = 256 * 1024 * 1024  # 256M floats = 1 GiB
    buf = torch.ones(numel, device="cuda")
    for _ in range(3):  # warmup
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    dist.barrier()

    iters = 10
    t0 = time.time()
    for _ in range(iters):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters

    nbytes = numel * 4
    algbw = nbytes / dt / 1e9
    busbw = algbw * 2 * (world - 1) / world
    if rank == 0:
        print(
            f"all_reduce OK across {world} ranks | "
            f"{nbytes / 1e9:.2f} GB in {dt * 1e3:.1f} ms/iter | "
            f"algbw {algbw:.1f} GB/s | busbw {busbw:.1f} GB/s",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY
trap 'rm -f "$WORKLOAD"' EXIT

echo
echo "================================================================"
echo "Running 2-node x 8-GPU all_reduce over IB"
echo "================================================================"
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -c "
    cd '$REPO_DIR'
    source .venv/bin/activate
    # Be explicit that IB must be used; log NET decisions per rank to a file.
    export NCCL_IB_DISABLE=0
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,NET
    export NCCL_DEBUG_FILE='$NCCL_LOG_DIR/nccl.\$SLURM_PROCID.%h.%p.log'
    torchrun \
        --nnodes=2 \
        --node_rank=\$SLURM_PROCID \
        --nproc_per_node=8 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        '$WORKLOAD'
"
rc=$?
echo "all_reduce exit code: $rc"

# --- verdict: which transport did NCCL actually use? ---
echo
echo "================================================================"
echo "Transport selected by NCCL (from logs)"
echo "================================================================"
ib_hits=$(grep -rhoE "NET/IB|NET/IBext|GPU Direct RDMA" "$NCCL_LOG_DIR" 2>/dev/null | sort | uniq -c)
sock_hits=$(grep -rhoE "NET/Socket" "$NCCL_LOG_DIR" 2>/dev/null | sort | uniq -c)
echo "IB transport lines:"
echo "${ib_hits:-  (none)}"
echo "Socket transport lines:"
echo "${sock_hits:-  (none)}"
echo
echo "Any ibv errors in logs:"
grep -rhoE "ibv_[a-z_]+ failed|Cannot allocate memory|ncclSystemError" "$NCCL_LOG_DIR" 2>/dev/null | sort | uniq -c || echo "  (none)"

echo
echo "Per-rank 'NET/IB : Using' lines (one per rank if IB is live):"
grep -rh "Using.*NET/IB\|NET/IB.*Using\|via NET/IB" "$NCCL_LOG_DIR" 2>/dev/null | head -16

echo
echo "================================================================"
if [ "$rc" -eq 0 ] && [ -n "$ib_hits" ]; then
    echo "VERDICT: InfiniBand is WORKING (all_reduce passed + NCCL used NET/IB)."
elif [ "$rc" -eq 0 ]; then
    echo "VERDICT: all_reduce passed but NCCL did NOT report IB -- likely TCP fallback."
else
    echo "VERDICT: all_reduce FAILED (exit $rc) -- see ibv errors above."
fi
echo "NCCL logs kept in: $NCCL_LOG_DIR"
echo "================================================================"

exit $rc
