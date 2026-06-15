#!/bin/bash
#SBATCH --job-name=affinity-diagnostic
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8          # full node, so the cgroup cpuset is the whole node (0-127)
#SBATCH --time=00:02:00
#SBATCH --output=affinity-diagnostic-%j.out
#
# Diagnoses srun step CPU-affinity narrowing on a fully-allocated node, AND distinguishes
# the two causes that look identical in the output:
#   (A) a genuine cluster-side bug   -> srun narrows EVEN with SLURM_CPU_BIND unset
#   (B) SLURM_CPU_BIND propagation   -> srun narrows ONLY because a narrowed cpu-bind mask
#                                        was inherited from the submitting shell's environment
#
# IMPORTANT: submit this from a CLEAN login shell (NOT from inside an existing salloc/srun/
# job allocation, and NOT a shell where SLURM_CPU_BIND is exported). If SLURM_CPU_BIND is
# already set at submit time, sbatch propagates it into this job and every srun below will
# obey it -- which masks (A) and fakes (B). The first line of output tells you which you have.
#
# Run as a NORMAL (non-root) user: root bypasses CPU-affinity restrictions and always sees all CPUs.

aff() { grep Cpus_allowed_list /proc/self/status | awk '{print $2}'; }
cgroup() { cat /sys/fs/cgroup"$(awk -F: '{print $3}' /proc/self/cgroup)"/cpuset.cpus.effective; }

echo "node                       : $(hostname)"
echo "partition                  : $SLURM_JOB_PARTITION"
echo "inherited SLURM_CPU_BIND    : '${SLURM_CPU_BIND:-<unset>}'   <- if not <unset>, results below are contaminated"
echo "allocation cgroup cpuset    : $(cgroup)   <- the CPUs this job actually owns"
echo "batch step affinity         : $(aff)"
echo
echo "srun [1] inherited env       : $(srun --ntasks=1 bash -c 'grep Cpus_allowed_list /proc/self/status' | awk '{print $2}')"
echo "srun [2] SLURM_CPU_BIND unset: $(env -u SLURM_CPU_BIND -u SLURM_CPU_BIND_TYPE -u SLURM_CPU_BIND_LIST -u SLURM_CPU_BIND_VERBOSE srun --ntasks=1 bash -c 'grep Cpus_allowed_list /proc/self/status' | awk '{print $2}')"
echo "srun [3] --cpu-bind=none     : $(srun --ntasks=1 --cpu-bind=none bash -c 'grep Cpus_allowed_list /proc/self/status' | awk '{print $2}')"
echo
echo "INTERPRETATION:"
echo "  [2] == cgroup           -> no cluster bug; any narrowing in [1] is SLURM_CPU_BIND propagation (cause B)."
echo "  [2] narrowed != cgroup  -> genuine cluster-side narrowing (cause A); report this jobid."
