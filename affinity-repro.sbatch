#!/bin/bash
#SBATCH --job-name=affinity-repro
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:05:00
#SBATCH --output=affinity-repro-%j.out

# Reproduces the post-reconfigure step CPU-affinity bug on reno.
# The job cgroup gets the full allocation, but srun task launches inherit
# slurmd's own 4-core pod mask instead of being bound to the allocation.
# Bug present iff the srun default-bind mask != the cgroup cpuset.

echo "## node: $(hostname)"
echo "## AllocTRES: $(scontrol show job $SLURM_JOB_ID | grep -o 'AllocTRES=[^ ]*')"
echo

echo "## cgroup cpuset (what the allocation actually grants):"
cat /sys/fs/cgroup$(awk -F: '{print $3}' /proc/self/cgroup)/cpuset.cpus.effective
echo

echo "## batch step affinity:"
grep Cpus_allowed_list /proc/self/status
echo

echo "## srun step, DEFAULT bind (the bug -- squeezed onto slurmd's pod mask):"
srun --ntasks=1 grep Cpus_allowed_list /proc/self/status
echo

echo "## srun step, --cpu-bind=none (workaround -- restores full allocation):"
srun --ntasks=1 --cpu-bind=none grep Cpus_allowed_list /proc/self/status
