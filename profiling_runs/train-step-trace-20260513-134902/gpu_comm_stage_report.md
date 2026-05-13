# Train-Step GPU/Communication Stage Trace

Trace run: `profiling_runs/train-step-trace-20260513-134902`

Definitions:
- `wall_ms_max`: max rank stage wall time from the PyTorch `record_function` interval.
- `gpu_active_ms_max`: max rank union of CUDA kernel/memcpy/memset intervals inside that stage.
- `gpu_sum_ms_max`: max rank sum of CUDA kernel/memcpy/memset durations; this can exceed wall time when streams overlap.
- `comm_ms_max`: max rank union of NCCL kernel intervals inside that stage.
- `comm_count_max`: max rank NCCL kernel count inside that stage.

Caveat: this was a one-step PyTorch profiler trace with synthetic token batches to avoid Hugging Face streaming-dataset metadata rate limits. Use it for GPU-vs-NCCL attribution. The 100-step real-data baseline is still the stable throughput source; PyTorch tracing adds overhead and can exaggerate one-step rank skew.

## 8 GPUs

| phase | wall ms | GPU active ms | GPU sum ms | NCCL comm ms | NCCL calls | comm/wall |
|---|---:|---:|---:|---:|---:|---:|
| `ppgd_warmup.PersistentPGDReconLoss.0.forward_loss` | 806.201 | 16.856 | 16.856 | 0.000 | 0 | 0.0% |
| `backward.total_loss` | 273.428 | 247.658 | 287.048 | 193.998 | 46 | 71.0% |
| `loss.PersistentPGDReconLoss` | 253.651 | 16.569 | 16.569 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.1.forward_loss` | 200.822 | 16.576 | 16.576 | 0.000 | 0 | 0.0% |
| `target_forward_input_cache` | 159.125 | 157.806 | 157.806 | 150.871 | 1 | 94.8% |
| `ci_forward` | 136.803 | 19.839 | 19.839 | 0.000 | 0 | 0.0% |
| `loss.StochasticReconSubsetLoss` | 34.178 | 17.205 | 17.205 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.0.source_grads` | 24.316 | 16.589 | 16.589 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.1.source_grads` | 23.902 | 16.574 | 16.574 | 0.000 | 0 | 0.0% |
| `ppgd_final_grads.PersistentPGDReconLoss` | 23.642 | 16.576 | 16.576 | 0.000 | 0 | 0.0% |
| `optimizer.step` | 14.116 | 12.841 | 12.841 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.0.source_step` | 5.724 | 4.479 | 4.479 | 0.000 | 0 | 0.0% |
| `ppgd_final_step.PersistentPGDReconLoss` | 5.306 | 4.483 | 4.483 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.1.source_step` | 5.292 | 4.470 | 4.470 | 0.000 | 0 | 0.0% |
| `weight_deltas` | 4.766 | 3.734 | 3.734 | 0.000 | 0 | 0.0% |
| `loss.ImportanceMinimalityLoss` | 4.342 | 1.561 | 1.561 | 0.000 | 0 | 0.0% |
| `ci_l_zero` | 2.290 | 1.070 | 1.070 | 0.000 | 0 | 0.0% |
| `loss.FaithfulnessLoss` | 2.021 | 0.287 | 0.287 | 0.000 | 0 | 0.0% |
| `pre_optimizer_barrier` | 1.603 | 1.203 | 1.203 | 1.202 | 1 | 75.0% |
| `grad_clip.components` | 1.133 | 0.431 | 0.431 | 0.000 | 0 | 0.0% |
| `optimizer.zero_grad` | 0.782 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |
| `total_loss` | 0.535 | 0.024 | 0.024 | 0.000 | 0 | 0.0% |
| `data.to_device` | 0.332 | 0.004 | 0.004 | 0.000 | 0 | 0.0% |
| `lr_schedule` | 0.297 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |
| `data.next` | 0.278 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |
| `ppgd.update_lr` | 0.275 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |

## 16 GPUs

| phase | wall ms | GPU active ms | GPU sum ms | NCCL comm ms | NCCL calls | comm/wall |
|---|---:|---:|---:|---:|---:|---:|
| `backward.total_loss` | 309.784 | 262.790 | 289.536 | 232.576 | 46 | 75.1% |
| `target_forward_input_cache` | 298.770 | 297.332 | 297.332 | 293.607 | 1 | 98.3% |
| `ppgd_warmup.PersistentPGDReconLoss.0.forward_loss` | 158.084 | 9.226 | 9.226 | 0.000 | 0 | 0.0% |
| `loss.PersistentPGDReconLoss` | 84.700 | 8.996 | 8.996 | 0.000 | 0 | 0.0% |
| `ci_forward` | 35.959 | 11.170 | 11.170 | 0.000 | 0 | 0.0% |
| `loss.StochasticReconSubsetLoss` | 33.350 | 9.328 | 9.328 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.1.forward_loss` | 29.045 | 9.005 | 9.005 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.1.source_grads` | 28.816 | 8.938 | 8.938 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.0.source_grads` | 26.718 | 8.972 | 8.972 | 0.000 | 0 | 0.0% |
| `ppgd_final_grads.PersistentPGDReconLoss` | 23.925 | 8.922 | 8.922 | 0.000 | 0 | 0.0% |
| `optimizer.step` | 14.675 | 12.877 | 12.877 | 0.000 | 0 | 0.0% |
| `loss.ImportanceMinimalityLoss` | 5.725 | 1.024 | 1.024 | 0.000 | 0 | 0.0% |
| `weight_deltas` | 4.972 | 3.746 | 3.746 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.1.source_step` | 4.971 | 2.257 | 2.257 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.0.source_step` | 4.368 | 2.261 | 2.261 | 0.000 | 0 | 0.0% |
| `ppgd.update_lr` | 3.601 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |
| `ppgd_final_step.PersistentPGDReconLoss` | 3.223 | 2.258 | 2.258 | 0.000 | 0 | 0.0% |
| `loss.FaithfulnessLoss` | 2.691 | 0.287 | 0.287 | 0.000 | 0 | 0.0% |
| `ci_l_zero` | 2.126 | 0.679 | 0.679 | 0.000 | 0 | 0.0% |
| `pre_optimizer_barrier` | 2.062 | 1.663 | 1.663 | 1.662 | 1 | 80.6% |
| `optimizer.zero_grad` | 1.778 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |
| `grad_clip.components` | 1.494 | 0.436 | 0.436 | 0.000 | 0 | 0.0% |
| `total_loss` | 0.555 | 0.025 | 0.025 | 0.000 | 0 | 0.0% |
| `data.to_device` | 0.386 | 0.002 | 0.002 | 0.000 | 0 | 0.0% |
| `lr_schedule` | 0.316 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |
| `data.next` | 0.286 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |

## 32 GPUs

| phase | wall ms | GPU active ms | GPU sum ms | NCCL comm ms | NCCL calls | comm/wall |
|---|---:|---:|---:|---:|---:|---:|
| `backward.total_loss` | 223.283 | 166.328 | 187.052 | 147.250 | 46 | 65.9% |
| `target_forward_input_cache` | 198.822 | 197.589 | 197.589 | 195.387 | 1 | 98.3% |
| `ppgd_warmup.PersistentPGDReconLoss.0.forward_loss` | 78.072 | 5.498 | 5.498 | 0.000 | 0 | 0.0% |
| `loss.PersistentPGDReconLoss` | 56.465 | 5.243 | 5.243 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.1.forward_loss` | 29.276 | 5.246 | 5.246 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.1.source_grads` | 26.697 | 5.426 | 5.426 | 0.000 | 0 | 0.0% |
| `loss.StochasticReconSubsetLoss` | 26.545 | 5.488 | 5.488 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.0.source_grads` | 26.463 | 5.431 | 5.431 | 0.000 | 0 | 0.0% |
| `ppgd_final_grads.PersistentPGDReconLoss` | 23.820 | 5.391 | 5.391 | 0.000 | 0 | 0.0% |
| `ci_forward` | 20.562 | 6.920 | 6.920 | 0.000 | 0 | 0.0% |
| `optimizer.step` | 14.303 | 12.847 | 12.847 | 0.000 | 0 | 0.0% |
| `weight_deltas` | 4.852 | 3.753 | 3.753 | 0.000 | 0 | 0.0% |
| `loss.ImportanceMinimalityLoss` | 4.385 | 0.806 | 0.806 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.0.source_step` | 3.578 | 1.292 | 1.292 | 0.000 | 0 | 0.0% |
| `ppgd_final_step.PersistentPGDReconLoss` | 3.179 | 1.298 | 1.298 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.1.source_step` | 3.092 | 1.296 | 1.296 | 0.000 | 0 | 0.0% |
| `ci_l_zero` | 2.021 | 0.500 | 0.500 | 0.000 | 0 | 0.0% |
| `loss.FaithfulnessLoss` | 1.983 | 0.287 | 0.287 | 0.000 | 0 | 0.0% |
| `pre_optimizer_barrier` | 1.519 | 1.152 | 1.152 | 1.151 | 1 | 75.8% |
| `grad_clip.components` | 1.203 | 0.437 | 0.437 | 0.000 | 0 | 0.0% |
| `optimizer.zero_grad` | 0.900 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |
| `total_loss` | 0.558 | 0.024 | 0.024 | 0.000 | 0 | 0.0% |
| `data.to_device` | 0.347 | 0.003 | 0.003 | 0.000 | 0 | 0.0% |
| `lr_schedule` | 0.316 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |
| `data.next` | 0.285 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |
| `ppgd.update_lr` | 0.281 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |
