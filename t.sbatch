#!/bin/bash

#SBATCH --partition=dev
#SBATCH --gpus=8
#SBATCH --nodes=1

srun -N1 -p dev --gpus=8 bash -c '
    echo "task affinity: $(grep -o "[0-9,-]*" /proc/self/status | head -1)"
    grep Cpus_allowed_list /proc/self/status
    step=$(dirname $(dirname /sys/fs/cgroup$(awk -F: "{print \$3}" /proc/self/cgroup)))
    echo "step cpuset.cpus.effective: $(cat $step/cpuset.cpus.effective)"'
