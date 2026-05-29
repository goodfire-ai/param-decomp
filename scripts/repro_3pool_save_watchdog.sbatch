#!/bin/bash
#SBATCH --job-name=3pool-save-repro
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --qos=opportunistic
#SBATCH --time=0:40:00
#SBATCH --output=/mnt/data/artifacts/mechanisms/param-decomp/slurm_logs/slurm-%j.out
#SBATCH --comment="3-pool checkpoint-save NCCL-watchdog repro/fix validation (oli)"

# Usage: sbatch repro_3pool_save_watchdog.sbatch <MODE> <RUN_ID>
#   MODE=before  -> rejoin barrier DISABLED + 30s PG timeout + 60s rank0 sleep
#                   => must HANG / NCCL-timeout at first save (demonstrates bug)
#   MODE=after   -> rejoin barrier ENABLED  + 30s PG timeout + 60s rank0 sleep
#                   => must SAVE + CONTINUE past multiple saves cleanly
#   MODE=resume  -> resume from a prior run's training_<step>.pth and continue

set -uo pipefail
umask 002

MODE="$1"
RUN_ID="$2"

export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Force the timing bug at small scale with a tiny PG timeout. The injected
# rank-0 read sleep models the XL ~10-min serial read. MODE picks the sleep:
#   before: sleep (60s) > timeout (30s) AND barrier disabled => ranks race ahead
#           into the next step's collective and trip the 30s watchdog (the bug).
#   after:  sleep (10s) < timeout (30s) AND barrier enabled  => ranks wait at the
#           rejoin barrier; rank 0 finishes its read well inside the timeout; all
#           rejoin and continue. This mirrors prod (30-min timeout > ~10-min read).
export PD_3POOL_PG_TIMEOUT_S=30
export PD_TRACE=1

WORKTREE=/mnt/home/oli/param-decomp/.claude/worktrees/agent-a5ef60a094ea6fd03
cd "$WORKTREE"
source .venv/bin/activate

CFG=param_decomp_lab/experiments/lm/_resumption_validation/gpt2_small_3pool_save_repro.yaml

case "$MODE" in
  before)
    export PD_3POOL_DISABLE_REJOIN_BARRIER=1
    export PD_3POOL_SNAPSHOT_RANK0_SLEEP_S=60
    echo "[repro] MODE=before: rejoin barrier DISABLED, rank0 sleep 60s > 30s timeout — expecting NCCL watchdog timeout at step 10 save"
    torchrun --standalone --nproc_per_node=8 \
      -m param_decomp_lab.experiments.lm.run "$CFG" --run_id "$RUN_ID"
    ;;
  after)
    export PD_3POOL_SNAPSHOT_RANK0_SLEEP_S=10
    echo "[repro] MODE=after: rejoin barrier ENABLED, rank0 sleep 10s < 30s timeout — expecting clean save + continue"
    torchrun --standalone --nproc_per_node=8 \
      -m param_decomp_lab.experiments.lm.run "$CFG" --run_id "$RUN_ID"
    ;;
  resume)
    RESUME_CFG="$3"
    export PD_3POOL_SNAPSHOT_RANK0_SLEEP_S=10
    echo "[repro] MODE=resume: resuming from $RESUME_CFG"
    torchrun --standalone --nproc_per_node=8 \
      -m param_decomp_lab.experiments.lm.run --resume "$RESUME_CFG" --run_id "$RUN_ID"
    ;;
  *)
    echo "unknown MODE: $MODE" >&2
    exit 2
    ;;
esac
