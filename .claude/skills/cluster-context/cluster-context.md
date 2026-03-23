---
name: cluster-context
description: SLURM cluster memory/CPU defaults and how GPU job resource allocation works. Reference this before submitting SLURM jobs that need significant CPU RAM.
user-invocable: false
---

# Cluster Context

## SLURM Resource Defaults

The cluster has three default resource paths, configured globally via `scontrol show config`:

| Setting | Value | When it applies |
|---------|-------|-----------------|
| `DefCpuPerGPU` | 24 | Job requests GPUs and SLURM uses GPU-based allocation |
| `DefMemPerGPU` | ~252G | Job requests GPUs and SLURM uses GPU-based allocation |
| `DefMemPerCPU` | 10G | Job specifies tasks/CPUs without triggering GPU-based defaults |

### The `--ntasks-per-node` trap

When a job uses `--gpus-per-node=N` **without** `--ntasks-per-node`, SLURM applies the GPU-based defaults: 24 CPUs and ~252G RAM per GPU. This is what `pod` does and why dev pods get plenty of RAM.

When a job adds `--ntasks-per-node=1`, SLURM switches to task-based allocation: 1 task → 1 CPU → `DefMemPerCPU` (10G). The GPU-based memory default is bypassed entirely, even though a GPU is allocated.

**`spd/utils/slurm.py`** generates `--ntasks-per-node=1` in all sbatch scripts (needed for multi-node DDP). This means GPU jobs submitted through `generate_script` / `generate_array_script` get only 10G RAM by default.

### Workaround

For jobs that need significant CPU RAM (e.g. clustering with large activation tensors), explicitly pass `mem` to `SlurmConfig`:

```python
SlurmArrayConfig(
    job_name="my_job",
    partition="h200-reserved",
    n_gpus=1,
    mem="300G",  # Override the 10G default
)
```

### Partition details

Both `h200-dev` and `h200-reserved` have `DefMemPerNode=UNLIMITED` and `MaxMemPerNode=UNLIMITED` — the 10G limit comes from the cluster-level `DefMemPerCPU`, not from partition config.

### Quick reference

```bash
# Check cluster defaults
scontrol show config | grep -i "defmem\|DefCpu"

# Check what a job actually got
sacct -j <JOBID> --format=JobID,ReqMem,AllocCPUS,MaxRSS,State -n

# Check partition limits
scontrol show partition <name> | grep -i mem
```
