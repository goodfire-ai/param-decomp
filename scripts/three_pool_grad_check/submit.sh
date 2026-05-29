#!/bin/bash
# Submit one grad-check run (single-pool or 3-pool) on the current source tree.
# Usage: submit.sh <mode> <n_ci> <n_per_block> <n_ppgd> <out_subdir> <tag> [isolate]
#   mode: singlepool | threepool
#   isolate: pass "isolate" to zero imp+ppgd coeffs (stoch-only CI grad)
set -euo pipefail
MODE="$1"; N_CI="$2"; N_PER_BLOCK="$3"; N_PPGD="$4"; OUT_SUBDIR="$5"; TAG="$6"
ISOLATE_FLAG=""
if [ "${7:-}" == "isolate" ]; then ISOLATE_FLAG="--isolate-stoch"; fi
WORKTREE="/mnt/home/oli/param-decomp/.claude/worktrees/agent-a4ddc047c3fb0828e"
OUTDIR="${WORKTREE}/scripts/three_pool_grad_check/out/${OUT_SUBDIR}"
LOGDIR="${WORKTREE}/scripts/three_pool_grad_check/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

if [ "$MODE" == "singlepool" ]; then
  NGPU=1
else
  NGPU=$((N_CI + N_PER_BLOCK + N_PPGD))
fi
PORT=$((25000 + RANDOM % 2000))

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=pd-gc-${TAG}
#SBATCH --qos=opportunistic
#SBATCH --gpus=${NGPU}
#SBATCH --nodes=1
#SBATCH --time=0:30:00
#SBATCH --comment=3pool-real-grad-check-${TAG}
#SBATCH --output=${LOGDIR}/${TAG}-%j.out
set -euo pipefail
source ${WORKTREE}/.venv/bin/activate
cd ${WORKTREE}
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HF_HUB_OFFLINE=0
torchrun --standalone --nproc_per_node=${NGPU} --master_port=${PORT} \
  scripts/three_pool_grad_check/grad_check.py run \
  --mode ${MODE} --n_ci ${N_CI} --n_per_block ${N_PER_BLOCK} --n_ppgd ${N_PPGD} \
  ${ISOLATE_FLAG} --out ${OUTDIR}
EOF
