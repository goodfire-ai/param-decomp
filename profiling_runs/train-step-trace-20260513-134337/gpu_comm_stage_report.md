# Train-Step GPU/Communication Stage Trace

Trace run: `profiling_runs/train-step-trace-20260513-134337`

Definitions:
- `wall_ms_max`: max rank stage wall time from the PyTorch `record_function` interval.
- `gpu_active_ms_max`: max rank union of CUDA kernel/memcpy/memset intervals inside that stage.
- `gpu_sum_ms_max`: max rank sum of CUDA kernel/memcpy/memset durations; this can exceed wall time when streams overlap.
- `comm_ms_max`: max rank union of NCCL kernel intervals inside that stage.
- `comm_count_max`: max rank NCCL kernel count inside that stage.

Caveat: this was a one-step PyTorch profiler trace with synthetic token batches to avoid Hugging Face metadata rate limits. Use it for attribution of GPU-vs-NCCL work, not as the stable throughput timing; the 100-step real-data baseline remains the stable timing source.

## 8 GPUs

| phase | wall ms | GPU active ms | GPU sum ms | NCCL comm ms | NCCL calls | comm/wall |
|---|---:|---:|---:|---:|---:|---:|
| `ppgd_warmup.PersistentPGDReconLoss.0.forward_loss` | 313.473 | 16.961 | 16.961 | 0.000 | 0 | 0.0% |
| `backward.total_loss` | 279.915 | 250.561 | 289.791 | 196.616 | 46 | 70.2% |
| `target_forward_input_cache` | 192.473 | 191.026 | 191.026 | 184.066 | 1 | 95.6% |
| `loss.PersistentPGDReconLoss` | 144.233 | 16.652 | 16.652 | 0.000 | 0 | 0.0% |
| `ci_forward` | 137.780 | 19.943 | 19.943 | 0.000 | 0 | 0.0% |
| `loss.StochasticReconSubsetLoss` | 28.226 | 17.237 | 17.237 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.0.source_grads` | 25.996 | 16.671 | 16.671 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.1.source_grads` | 25.680 | 16.631 | 16.631 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.1.forward_loss` | 25.499 | 16.635 | 16.635 | 0.000 | 0 | 0.0% |
| `ppgd_final_grads.PersistentPGDReconLoss` | 23.609 | 16.617 | 16.617 | 0.000 | 0 | 0.0% |
| `optimizer.step` | 14.273 | 12.855 | 12.855 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.0.source_step` | 5.875 | 4.487 | 4.487 | 0.000 | 0 | 0.0% |
| `ppgd_final_step.PersistentPGDReconLoss` | 5.438 | 4.482 | 4.482 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.1.source_step` | 5.408 | 4.478 | 4.478 | 0.000 | 0 | 0.0% |
| `weight_deltas` | 4.897 | 3.753 | 3.753 | 0.000 | 0 | 0.0% |
| `loss.ImportanceMinimalityLoss` | 4.727 | 1.551 | 1.551 | 0.000 | 0 | 0.0% |
| `ci_l_zero` | 2.508 | 1.081 | 1.081 | 0.000 | 0 | 0.0% |
| `loss.FaithfulnessLoss` | 2.068 | 0.293 | 0.293 | 0.000 | 0 | 0.0% |
| `pre_optimizer_barrier` | 1.918 | 1.429 | 1.429 | 1.428 | 1 | 74.5% |
| `grad_clip.components` | 1.342 | 0.434 | 0.434 | 0.000 | 0 | 0.0% |
| `optimizer.zero_grad` | 1.104 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |
| `total_loss` | 0.650 | 0.024 | 0.024 | 0.000 | 0 | 0.0% |
| `data.next` | 0.441 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |
| `data.to_device` | 0.429 | 0.004 | 0.004 | 0.000 | 0 | 0.0% |
| `lr_schedule` | 0.421 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |
| `ppgd.update_lr` | 0.368 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |

## 16 GPUs

| phase | wall ms | GPU active ms | GPU sum ms | NCCL comm ms | NCCL calls | comm/wall |
|---|---:|---:|---:|---:|---:|---:|
| `backward.total_loss` | 227.471 | 180.730 | 207.411 | 150.686 | 46 | 66.2% |
| `ppgd_warmup.PersistentPGDReconLoss.0.forward_loss` | 156.903 | 9.256 | 9.256 | 0.000 | 0 | 0.0% |
| `loss.PersistentPGDReconLoss` | 74.568 | 9.022 | 9.022 | 0.000 | 0 | 0.0% |
| `target_forward_input_cache` | 50.406 | 49.038 | 49.038 | 45.299 | 1 | 89.9% |
| `ppgd_warmup.PersistentPGDReconLoss.1.source_grads` | 38.534 | 8.980 | 8.980 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.0.source_grads` | 37.643 | 8.983 | 8.983 | 0.000 | 0 | 0.0% |
| `ci_forward` | 35.776 | 11.207 | 11.207 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.1.forward_loss` | 30.311 | 9.039 | 9.039 | 0.000 | 0 | 0.0% |
| `loss.StochasticReconSubsetLoss` | 27.509 | 9.364 | 9.364 | 0.000 | 0 | 0.0% |
| `ppgd_final_grads.PersistentPGDReconLoss` | 23.165 | 8.947 | 8.947 | 0.000 | 0 | 0.0% |
| `optimizer.step` | 14.277 | 12.867 | 12.867 | 0.000 | 0 | 0.0% |
| `weight_deltas` | 5.070 | 3.749 | 3.749 | 0.000 | 0 | 0.0% |
| `loss.ImportanceMinimalityLoss` | 4.500 | 1.033 | 1.033 | 0.000 | 0 | 0.0% |
| `ppgd_final_step.PersistentPGDReconLoss` | 3.785 | 2.273 | 2.273 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.0.source_step` | 3.736 | 2.266 | 2.266 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.1.source_step` | 3.189 | 2.261 | 2.261 | 0.000 | 0 | 0.0% |
| `ci_l_zero` | 2.487 | 0.677 | 0.677 | 0.000 | 0 | 0.0% |
| `loss.FaithfulnessLoss` | 2.024 | 0.287 | 0.287 | 0.000 | 0 | 0.0% |
| `optimizer.zero_grad` | 1.297 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |
| `grad_clip.components` | 1.296 | 0.438 | 0.438 | 0.000 | 0 | 0.0% |
| `pre_optimizer_barrier` | 1.252 | 0.795 | 0.795 | 0.793 | 1 | 63.4% |
| `total_loss` | 0.626 | 0.024 | 0.024 | 0.000 | 0 | 0.0% |
| `data.to_device` | 0.603 | 0.002 | 0.002 | 0.000 | 0 | 0.0% |
| `lr_schedule` | 0.592 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |
| `data.next` | 0.573 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |
| `ppgd.update_lr` | 0.543 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |

## 32 GPUs

| phase | wall ms | GPU active ms | GPU sum ms | NCCL comm ms | NCCL calls | comm/wall |
|---|---:|---:|---:|---:|---:|---:|
| `backward.total_loss` | 215.897 | 163.414 | 183.889 | 144.483 | 46 | 66.9% |
| `ppgd_warmup.PersistentPGDReconLoss.0.forward_loss` | 78.160 | 5.469 | 5.469 | 0.000 | 0 | 0.0% |
| `target_forward_input_cache` | 78.010 | 76.736 | 76.736 | 74.527 | 1 | 95.5% |
| `loss.PersistentPGDReconLoss` | 52.548 | 5.212 | 5.212 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.1.source_grads` | 30.683 | 5.380 | 5.380 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.1.forward_loss` | 27.121 | 5.212 | 5.212 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.0.source_grads` | 26.181 | 5.389 | 5.389 | 0.000 | 0 | 0.0% |
| `ppgd_final_grads.PersistentPGDReconLoss` | 23.790 | 5.363 | 5.363 | 0.000 | 0 | 0.0% |
| `loss.StochasticReconSubsetLoss` | 23.644 | 5.466 | 5.466 | 0.000 | 0 | 0.0% |
| `ci_forward` | 20.505 | 6.910 | 6.910 | 0.000 | 0 | 0.0% |
| `optimizer.step` | 14.549 | 12.855 | 12.855 | 0.000 | 0 | 0.0% |
| `ci_l_zero` | 6.224 | 0.495 | 0.495 | 0.000 | 0 | 0.0% |
| `weight_deltas` | 6.213 | 3.740 | 3.740 | 0.000 | 0 | 0.0% |
| `pre_optimizer_barrier` | 6.128 | 5.652 | 5.652 | 5.651 | 1 | 92.2% |
| `loss.ImportanceMinimalityLoss` | 5.109 | 0.795 | 0.795 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.1.source_step` | 4.721 | 1.289 | 1.289 | 0.000 | 0 | 0.0% |
| `ppgd_warmup.PersistentPGDReconLoss.0.source_step` | 3.712 | 1.292 | 1.292 | 0.000 | 0 | 0.0% |
| `ppgd_final_step.PersistentPGDReconLoss` | 3.331 | 1.295 | 1.295 | 0.000 | 0 | 0.0% |
| `loss.FaithfulnessLoss` | 2.173 | 0.298 | 0.298 | 0.000 | 0 | 0.0% |
| `grad_clip.components` | 1.657 | 0.436 | 0.436 | 0.000 | 0 | 0.0% |
| `optimizer.zero_grad` | 1.598 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |
| `lr_schedule` | 0.993 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |
| `ppgd.update_lr` | 0.959 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |
| `total_loss` | 0.689 | 0.024 | 0.024 | 0.000 | 0 | 0.0% |
| `data.to_device` | 0.430 | 0.002 | 0.002 | 0.000 | 0 | 0.0% |
| `data.next` | 0.405 | 0.000 | 0.000 | 0.000 | 0 | 0.0% |
