#!/bin/bash
# Submit one 8-GPU single-node 3-pool run for a given topology yaml.
# Usage: submit_threepool_topo.sh <yaml_basename_without_ext> <run_id> <master_port>
set -euo pipefail
YAML="$1"
RUN_ID="$2"
PORT="$3"
WORKTREE="/mnt/home/oli/param-decomp/.claude/worktrees/agent-ab6947e4c08bc68ae"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=pd-${RUN_ID}
#SBATCH --qos=opportunistic
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --time=1:30:00
#SBATCH --comment=multipool-bidir-divisibility-equiv-${RUN_ID}
#SBATCH --output=${WORKTREE}/scripts/hetero_topology_equiv/slurm-${RUN_ID}-%j.out
set -euo pipefail
source ${WORKTREE}/.venv/bin/activate
cd ${WORKTREE}
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
torchrun --standalone --nproc_per_node=8 --master_port=${PORT} \
  -m param_decomp_lab.experiments.lm.run \
  scripts/hetero_topology_equiv/${YAML}.yaml \
  --run_id ${RUN_ID}
EOF
